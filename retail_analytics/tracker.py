"""
tracker.py — YOLO ByteTrack / BoT-SORT integration for persistent person tracking.

Wraps Ultralytics' built-in tracker (ByteTrack is the default) so that every
detected person receives a stable, persistent Track ID across video frames.
"""

import numpy as np
import torch
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Optional

from utils import setup_logger

logger = setup_logger()

# COCO class index for 'person'
PERSON_CLASS_ID = 0


@dataclass
class TrackedPerson:
    """A single tracked person in one frame with a persistent ID."""
    track_id: int           # Persistent tracking ID (from ByteTrack / BoT-SORT)
    bbox: tuple             # (x1, y1, x2, y2) in pixel coordinates
    confidence: float       # Detection confidence
    frame_index: int        # Video frame number this observation belongs to


class PersonTracker:
    """
    Persistent multi-person tracker using Ultralytics YOLO + ByteTrack.

    Instead of separating detection and tracking into two separate API calls,
    Ultralytics exposes model.track() which runs detection + ByteTrack in a
    single pass and assigns stable track IDs.

    Args:
        model_path:   YOLO weight file path or model name.
        conf_thresh:  Minimum detection confidence threshold.
        tracker_cfg:  Ultralytics tracker config name.
                      Options: 'bytetrack.yaml', 'botsort.yaml'.
        device:       Compute device ('cuda', 'cpu', or '' for auto).
        iou_thresh:   IoU threshold for matching detections to tracks.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_thresh: float = 0.35,
        tracker_cfg: str = "bytetrack.yaml",
        device: str = "",
        iou_thresh: float = 0.5,
    ):
        # Auto-detect device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        logger.info(
            f"[Tracker] Loading YOLO tracker '{model_path}' with '{tracker_cfg}' "
            f"on device='{self.device}'"
        )

        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as exc:
            logger.error(f"[Tracker] Failed to load model: {exc}")
            raise

        self.conf_thresh  = conf_thresh
        self.iou_thresh   = iou_thresh
        self.tracker_cfg  = tracker_cfg
        self._frame_count = 0

        logger.info(
            f"[Tracker] Ready — tracker={tracker_cfg}, conf={conf_thresh}, iou={iou_thresh}"
        )

    def update(self, frame: np.ndarray) -> List[TrackedPerson]:
        """
        Process one video frame and return a list of actively tracked persons.

        Each TrackedPerson has a persistent track_id that remains stable as
        long as the person stays in the scene.

        Args:
            frame: BGR image (H × W × 3).

        Returns:
            List of TrackedPerson objects for this frame.
        """
        if frame is None or frame.size == 0:
            return []

        self._frame_count += 1

        try:
            # model.track() runs detection + ByteTrack in one shot.
            # persist=True keeps track state across calls (essential!).
            results = self.model.track(
                source=frame,
                classes=[PERSON_CLASS_ID],
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                tracker=self.tracker_cfg,
                persist=True,          # CRITICAL: maintains track state between frames
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            logger.warning(f"[Tracker] Tracking failed on frame {self._frame_count}: {exc}")
            return []

        tracked_persons: List[TrackedPerson] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes

            # Ultralytics stores track IDs in boxes.id (None when no tracks yet)
            if boxes.id is None:
                continue

            for box, track_id in zip(boxes, boxes.id):
                cls = int(box.cls[0].cpu().numpy())
                if cls != PERSON_CLASS_ID:
                    continue

                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                tid  = int(track_id.cpu().numpy())

                tracked_persons.append(
                    TrackedPerson(
                        track_id=tid,
                        bbox=tuple(xyxy),   # (x1, y1, x2, y2)
                        confidence=conf,
                        frame_index=self._frame_count,
                    )
                )

        return tracked_persons

    @property
    def frame_count(self) -> int:
        """Total frames processed so far."""
        return self._frame_count
