"""Track Manager with Integrated ByteTrack and Stale State Cleanup
=============================================================
Manages ByteTrack assignment and track trail histories.
Notifies downstream event managers when tracks expire to prevent memory leaks.
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False

from config.settings import TRAIL_LENGTH


class TrackManager:
    """ByteTrack tracker and motion trail coordinator."""

    def __init__(
        self,
        trail_length: int = TRAIL_LENGTH,
        track_thresh: float = 0.35,
        track_buffer: int = 45,
        match_thresh: float = 0.80,
        frame_rate: int = 5,
    ):
        self.trail_length = trail_length
        self.history = defaultdict(lambda: deque(maxlen=self.trail_length))
        self.active_ids: Set[int] = set()
        self.lost_ids: Set[int] = set()
        self.missing_counts: Dict[int, int] = {}
        self.max_missing_frames = 15

        if SUPERVISION_AVAILABLE:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=track_thresh,
                lost_track_buffer=track_buffer,
                minimum_matching_threshold=match_thresh,
                frame_rate=15,
            )
        else:
            self.tracker = None
            print("[TrackManager] Warning: supervision library not installed. Native tracking disabled.")

    def track(
        self,
        person_detections: List[Dict],
        bag_detections: Optional[List[Dict]] = None,
    ) -> Tuple[List[Dict], List[Dict], Set[int]]:
        """Assign ByteTrack IDs to person and bag detections.

        Returns:
            (tracked_persons, tracked_bags, newly_lost_track_ids)
        """
        all_dets = []
        is_person_flags = []

        for p in person_detections:
            all_dets.append((p["bbox"], p.get("confidence", 0.9), 0, p))
            is_person_flags.append(True)

        if bag_detections:
            for b in bag_detections:
                all_dets.append((b["bbox"], b.get("confidence", 0.8), 26, b))
                is_person_flags.append(False)

        tracked_persons: List[Dict] = []
        tracked_bags: List[Dict] = []
        current_active_ids: Set[int] = set()

        if not all_dets or self.tracker is None:
            newly_lost = self.active_ids - current_active_ids
            self.lost_ids.update(newly_lost)
            self.active_ids = current_active_ids
            return tracked_persons, tracked_bags, newly_lost

        xyxy = np.array([d[0] for d in all_dets], dtype=np.float32)
        conf = np.array([d[1] for d in all_dets], dtype=np.float32)
        cls = np.array([d[2] for d in all_dets], dtype=np.int32)

        detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        tracked = self.tracker.update_with_detections(detections)

        if len(tracked) > 0 and tracked.tracker_id is not None:
            for t_idx, (t_box, track_id, t_cls) in enumerate(
                zip(tracked.xyxy, tracked.tracker_id, tracked.class_id)
            ):
                t_id = int(track_id)
                current_active_ids.add(t_id)

                best_det = None
                best_iou = -1.0
                best_is_person = True

                for (o_box, o_conf, o_cls, o_det), is_p in zip(all_dets, is_person_flags):
                    if o_cls != t_cls:
                        continue
                    ix1 = max(t_box[0], o_box[0])
                    iy1 = max(t_box[1], o_box[1])
                    ix2 = min(t_box[2], o_box[2])
                    iy2 = min(t_box[3], o_box[3])
                    iw = max(0.0, ix2 - ix1)
                    ih = max(0.0, iy2 - iy1)
                    inter = iw * ih
                    union = (
                        (t_box[2] - t_box[0]) * (t_box[3] - t_box[1])
                        + (o_box[2] - o_box[0]) * (o_box[3] - o_box[1])
                        - inter
                    )
                    iou = inter / max(1.0, union)
                    if iou > best_iou:
                        best_iou = iou
                        best_det = o_det
                        best_is_person = is_p

                if best_det is not None and best_iou > 0.3:
                    det_dict = dict(best_det)
                    det_dict["track_id"] = t_id
                    if best_is_person:
                        tracked_persons.append(det_dict)
                    else:
                        tracked_bags.append(det_dict)
                else:
                    x1, y1, x2, y2 = int(t_box[0]), int(t_box[1]), int(t_box[2]), int(t_box[3])
                    bbox = (x1, y1, x2, y2)
                    if t_cls == 0:
                        upper_center = (int((x1 + x2) / 2), int(y1))
                        bottom_center = (int((x1 + x2) / 2), int(y1 + (y2 - y1) * 0.75))
                        lower_body_midpoint = (int((x1 + x2) / 2), int(y2))
                        tracked_persons.append({
                            "track_id": t_id,
                            "bbox": bbox,
                            "confidence": 0.85,
                            "center": upper_center,
                            "bottom_center": bottom_center,
                            "person_lower_body_midpoint": lower_body_midpoint,
                        })
                    else:
                        midpoint = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        tracked_bags.append({
                            "track_id": t_id,
                            "bbox": bbox,
                            "confidence": 0.8,
                            "bag_midpoint": midpoint,
                        })

        self.update(tracked_persons)

        for tid in current_active_ids:
            self.missing_counts[tid] = 0
            self.active_ids.add(tid)

        newly_lost = set()
        for tid in list(self.active_ids):
            if tid not in current_active_ids:
                self.missing_counts[tid] = self.missing_counts.get(tid, 0) + 1
                if self.missing_counts[tid] >= self.max_missing_frames:
                    newly_lost.add(tid)
                    self.active_ids.discard(tid)
                    self.missing_counts.pop(tid, None)
                    self.history.pop(tid, None)

        self.lost_ids.update(newly_lost)

        if len(self.lost_ids) > 1000:
            self.lost_ids.clear()

        return tracked_persons, tracked_bags, newly_lost

    def update(self, detections: List[Dict]) -> None:
        """Update trail history for active detections (backward compatible)."""
        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue
            center = det.get("center")
            if center is not None:
                self.history[track_id].append(center)

    def get_trail(self, track_id: int) -> List[Tuple[int, int]]:
        """Return motion trail point history for a track ID."""
        return list(self.history[track_id])