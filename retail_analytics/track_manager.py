"""
track_manager.py — Consistent Large Bounding Box trigger logic.

For each tracked person, monitors whether their bounding box has been
consistently large (i.e., person is close to the camera and clearly visible)
for a minimum number of consecutive frames.

When the trigger fires, the CURRENT frame is used — no frame history is stored.
RAM usage per track is a single integer counter (~50 bytes vs ~15 MiB previously).

Trigger condition:
    bbox_area / frame_area >= MIN_BBOX_RATIO
    for >= CONSISTENT_FRAMES consecutive frames

Once triggered, the track is marked as analyzed and never triggered again.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

try:
    from retail_analytics.utils import setup_logger
except ImportError:
    from utils import setup_logger

logger = setup_logger()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Bounding box must occupy at least this fraction of the frame area.
# 0.04 = 4% of a 1920x1080 frame = ~82,944 px² ≈ person ~200×415 pixels at 1080p.
# Increase to 0.06 for only very close persons; decrease to 0.02 for distant cameras.
MIN_BBOX_RATIO: float = 0.04

# Number of consecutive frames the large-bbox condition must hold before triggering.
# 8 frames at 25fps = ~0.32 seconds of stable, close visibility.
CONSISTENT_FRAMES: int = 8


# ---------------------------------------------------------------------------
# Per-track state (minimal footprint)
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    """Lightweight state for a single tracked person."""

    track_id: int

    # Consecutive frames where bbox_area / frame_area >= MIN_BBOX_RATIO
    consecutive_large_count: int = 0

    # Whether VLM analysis has been triggered (and we're done with this track)
    analyzed: bool = False

    # VLM result dict once available: {age_group, gender, confidence}
    analysis_result: Optional[dict] = None

    # Frame number when trigger fired
    trigger_frame_no: int = 0

    # Wall-clock timestamp when trigger fired (ISO string)
    trigger_timestamp: str = ""

    # Bounding box at trigger moment (original full-res coordinates)
    trigger_bbox: tuple = field(default_factory=tuple)

    # First and most recent frame indices this track was seen
    first_seen_frame: int = 0
    last_seen_frame: int = 0


# ---------------------------------------------------------------------------
# TrackManager
# ---------------------------------------------------------------------------

class TrackManager:
    """
    Manages per-track trigger state using the Consistent Large BBox algorithm.

    No frames are stored in memory. The trigger decision is based solely on
    the bounding box size ratio over the last N frames.

    Args:
        min_bbox_ratio:    Minimum bbox_area / frame_area to count as "large".
        consistent_frames: Consecutive large-bbox frames required to trigger.
    """

    def __init__(
        self,
        min_bbox_ratio: float = MIN_BBOX_RATIO,
        consistent_frames: int = CONSISTENT_FRAMES,
    ):
        self.min_bbox_ratio    = min_bbox_ratio
        self.consistent_frames = consistent_frames

        # Main state store: track_id → TrackState
        self._tracks: Dict[int, TrackState] = {}

        # Set of IDs that have already triggered VLM (no re-analysis ever)
        self.analyzed_track_ids: set = set()

    # ------------------------------------------------------------------
    # Core update — called every frame for every tracked person
    # ------------------------------------------------------------------

    def update(
        self,
        track_id: int,
        bbox: tuple,
        frame_shape: tuple,
        frame_index: int,
    ) -> bool:
        """
        Update the bbox-size counter for a track and check if the trigger fires.

        Args:
            track_id:    Persistent tracker ID.
            bbox:        (x1, y1, x2, y2) bounding box in full-res pixels.
            frame_shape: (height, width[, channels]) of the video frame.
            frame_index: Current frame number (used for logging / CSV).

        Returns:
            True  → trigger fires this frame (send to VLM NOW).
            False → not yet, or already analyzed.
        """
        # Register new track
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackState(
                track_id=track_id,
                first_seen_frame=frame_index,
                last_seen_frame=frame_index,
            )
            logger.debug(f"[TrackManager] New track: ID={track_id}")

        state = self._tracks[track_id]
        state.last_seen_frame = frame_index

        # Already analyzed — nothing to do
        if track_id in self.analyzed_track_ids:
            return False

        # Compute bbox size ratio
        x1, y1, x2, y2 = bbox
        bbox_area  = max(0, x2 - x1) * max(0, y2 - y1)
        frame_area = frame_shape[0] * frame_shape[1]
        ratio      = bbox_area / max(frame_area, 1)

        if ratio >= self.min_bbox_ratio:
            state.consecutive_large_count += 1
        else:
            # Person moved away or left frame — reset streak
            state.consecutive_large_count = 0

        # Trigger condition: streak reached the required length
        if state.consecutive_large_count >= self.consistent_frames:
            return True

        return False

    # ------------------------------------------------------------------
    # Actions on trigger
    # ------------------------------------------------------------------

    def mark_triggered(
        self,
        track_id: int,
        frame_index: int,
        bbox: tuple,
    ) -> None:
        """
        Mark a track as triggered (submitted to VLM).
        Called immediately after the trigger fires to prevent re-triggering.

        Args:
            track_id:    The triggered track ID.
            frame_index: Frame number of the trigger.
            bbox:        Full-res bounding box at trigger moment.
        """
        self.analyzed_track_ids.add(track_id)
        if track_id in self._tracks:
            state = self._tracks[track_id]
            state.analyzed          = True
            state.trigger_frame_no  = frame_index
            state.trigger_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            state.trigger_bbox      = bbox

    def store_result(self, track_id: int, result: dict) -> None:
        """
        Attach a VLM analysis result to a track and print the terminal report.

        Args:
            track_id: The persistent track ID.
            result:   Dict with keys: age_group, gender, confidence.
        """
        if track_id not in self._tracks:
            logger.warning(f"[TrackManager] store_result: unknown track {track_id}")
            return

        self._tracks[track_id].analysis_result = result

        # ---- Terminal output (Age Group only) ----
        sep = "=" * 36
        print(f"\n{sep}")
        print(f"Track ID: {track_id}")
        print(f"Age Group: {result.get('age_group', 'Adult')}")
        print(f"Gender: {result.get('gender', result.get('apparent_gender', 'Unknown')).title()}")
        print(f"Confidence: {result.get('confidence', 'High')}")
        print(f"{sep}\n")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_state(self, track_id: int) -> Optional[TrackState]:
        return self._tracks.get(track_id)

    def get_analysis_result(self, track_id: int) -> Optional[dict]:
        state = self._tracks.get(track_id)
        return state.analysis_result if state else None

    def get_trigger_info(self, track_id: int) -> Optional[dict]:
        """Return trigger metadata (timestamp, bbox, frame_no) for CSV writing."""
        state = self._tracks.get(track_id)
        if not state:
            return None
        return {
            "timestamp":  state.trigger_timestamp,
            "frame_no":   state.trigger_frame_no,
            "bbox":       state.trigger_bbox,
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def total_tracks(self) -> int:
        return len(self._tracks)

    @property
    def analyzed_count(self) -> int:
        return len(self.analyzed_track_ids)

    @property
    def all_track_ids(self) -> List[int]:
        return list(self._tracks.keys())
