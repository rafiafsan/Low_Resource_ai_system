"""
detector.py — YOLO person detection wrapper.

Wraps Ultralytics YOLO to detect ONLY the 'person' class (class index 0)
and returns standardised detection results for each video frame.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List

from utils import setup_logger

logger = setup_logger()

# COCO class index for 'person'
PERSON_CLASS_ID = 0


@dataclass
class Detection:
    """A single person detection from one frame."""
    bbox: tuple          # (x1, y1, x2, y2) in pixel coordinates
    confidence: float    # Detection confidence in [0, 1]
    class_id: int = 0    # Always 0 (person)


class PersonDetector:
    """
    Wraps Ultralytics YOLO for person-only detection.

    Args:
        model_path:  Path or name of YOLO model weights
                     (e.g. 'yolo11n.pt' or absolute path).
        conf_thresh: Minimum detection confidence threshold.
        device:      'cuda', 'cpu', or '' (auto-select).
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_thresh: float = 0.35,
        device: str = "",
    ):
        # Auto-detect device: prefer CUDA → MPS → CPU
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        logger.info(f"[Detector] Loading YOLO model '{model_path}' on device='{self.device}'")

        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as exc:
            logger.error(f"[Detector] Failed to load YOLO model: {exc}")
            raise

        self.conf_thresh = conf_thresh
        logger.info(f"[Detector] Ready — conf_threshold={conf_thresh}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run person detection on a single BGR frame.

        Args:
            frame: BGR image as numpy array (H × W × 3).

        Returns:
            List of Detection objects for every detected person.
        """
        if frame is None or frame.size == 0:
            return []

        try:
            # classes=[0] restricts YOLO to only the 'person' class
            results = self.model.predict(
                source=frame,
                classes=[PERSON_CLASS_ID],
                conf=self.conf_thresh,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            logger.warning(f"[Detector] Prediction failed on frame: {exc}")
            return []

        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                # xyxy gives (x1, y1, x2, y2) as a tensor
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls  = int(box.cls[0].cpu().numpy())

                if cls != PERSON_CLASS_ID:
                    continue  # safety guard

                detections.append(
                    Detection(
                        bbox=tuple(xyxy),       # (x1, y1, x2, y2)
                        confidence=conf,
                        class_id=cls,
                    )
                )

        return detections
