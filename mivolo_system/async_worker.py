import logging
import queue
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
MIVOLO_ROOT = PROJECT_ROOT / "mivolo_system"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MIVOLO_ROOT))

from MiVOLO.data.misc import prepare_classification_images
from model.mi_volo import MiVOLO

logger = logging.getLogger("AsyncDemographics")


@dataclass
class TrackEvidence:
    """Accumulated age and gender observations for one tracked customer."""

    observations: List[Tuple[Optional[float], Optional[str]]] = field(default_factory=list)
    missing_frames: int = 0
    finalized: bool = False
    max_samples: int = 4

    def record(self, age: Optional[float], gender: Optional[str]) -> None:
        if age is not None or gender is not None:
            self.observations.append((age, gender))
        self.missing_frames = 0
        if len(self.observations) >= self.max_samples:
            self.finalized = True

    @property
    def average_age(self) -> Optional[float]:
        ages = [age for age, _ in self.observations if age is not None]
        return sum(ages) / len(ages) if ages else None

    @property
    def gender_votes(self) -> Counter:
        return Counter(gender.lower() for _, gender in self.observations if gender is not None)

    @property
    def final_gender(self) -> Optional[str]:
        votes = self.gender_votes
        if not votes:
            return None
        if votes.get("female", 0) >= votes.get("male", 0) and votes.get("female", 0) > 0:
            return "female"
        return votes.most_common(1)[0][0]

    @property
    def summary_label(self) -> str:
        age = self.average_age
        gender = self.final_gender
        if age is not None and gender is not None:
            return f"{gender.title()}, ~{int(round(age))}y"
        elif age is not None:
            return f"~{int(round(age))}y"
        elif gender is not None:
            return f"{gender.title()}"
        return "Profiling..."


class AsyncDemographicWorker:
    """Asynchronous background worker for running direct MiVOLO age/gender inference.

    Optimized for low-resource CPU execution:
    - Eliminates the secondary YOLOv8x face detector entirely (saves ~350MB RAM).
    - Uses the upper 25% of the YOLO person detection bounding box as the facial crop.
    - Bound queue (default 4) and max samples (default 4) prevents RAM accumulation.
    - Lazy on-demand loading of the MiVOLO checkpoint.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        detector_weights: Optional[str] = None,
        device: str = "cpu",
        max_queue_size: int = 4,
        max_samples_per_person: int = 4,
        cpu_threads: int = 1,
        lazy: bool = True,
        on_result: Optional[Callable[[int, float, str], None]] = None,
        female_threshold: float = 0.50,
    ):
        self.device = device
        self.checkpoint = checkpoint_path or str(MIVOLO_ROOT / "model" / "model_imdb_cross_person_4.22_99.46.pth.tar")
        self.max_queue_size = max_queue_size
        self.max_samples = max_samples_per_person
        self.cpu_threads = cpu_threads
        self.lazy = lazy
        self.on_result = on_result
        self.female_threshold = female_threshold

        self.queue: queue.Queue = queue.Queue(maxsize=self.max_queue_size)
        self.evidence_lock = threading.Lock()
        self.evidence_by_track: Dict[int, TrackEvidence] = {}
        self.finalized_history: Dict[int, TrackEvidence] = {}
        self.finalized_order: deque = deque(maxlen=1000)
        self.stopped = False
        self.model: Optional[MiVOLO] = None
        self.last_latency_ms: float = 0.0

        print(f"[AsyncDemographicWorker] Configured on {self.device} (Queue: {max_queue_size}, Max Samples: {max_samples_per_person}, Lazy: {lazy})")

        if not self.lazy:
            self._ensure_model_loaded()

        self.thread = threading.Thread(target=self._worker_loop, daemon=True, name="MiVOLO-WorkerThread")
        self.thread.start()

    def _ensure_model_loaded(self) -> None:
        if self.model is None:
            print(f"[AsyncDemographicWorker] Event-driven load: Initializing direct MiVOLO on {self.device} (Threads: {self.cpu_threads})...")
            t0 = time.time()
            if self.device == "cpu" and self.cpu_threads > 0:
                torch.set_num_threads(self.cpu_threads)

            self.model = MiVOLO(
                ckpt_path=self.checkpoint,
                device=self.device,
                half=False,
                disable_faces=False,
                use_persons=True,
                verbose=False,
            )
            print(f"[AsyncDemographicWorker] Direct MiVOLO loaded in {time.time() - t0:.2f}s (Face detector bypassed)!")

    @property
    def queue_size(self) -> int:
        return self.queue.qsize()

    def submit_crop(self, track_id: int, crop: np.ndarray) -> bool:
        """Submit a cropped person image for demographic inference. Non-blocking."""
        if crop is None or crop.size == 0 or self.stopped:
            return False

        with self.evidence_lock:
            ev = self.evidence_by_track.get(track_id)
            if ev and ev.finalized:
                return False

        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except (queue.Empty, ValueError):
                pass

        try:
            self.queue.put_nowait((track_id, crop.copy()))
            return True
        except queue.Full:
            return False

    def _worker_loop(self) -> None:
        """Background thread worker pulling from queue and running direct MiVOLO."""
        while not self.stopped:
            try:
                track_id, crop = self.queue.get(timeout=0.25)
            except queue.Empty:
                continue

            try:
                self._ensure_model_loaded()
                t0 = time.time()

                h, w = crop.shape[:2]
                if h < 10 or w < 10:
                    continue

                # Upper 25% of YOLO person bounding box represents the head / face region
                face_h = max(4, int(h * 0.25))
                face_crop = crop[:face_h, :]
                body_crop = crop

                with torch.inference_mode():
                    faces_input = prepare_classification_images(
                        [face_crop],
                        self.model.input_size,
                        self.model.data_config["mean"],
                        self.model.data_config["std"],
                        device=self.device,
                    )
                    person_input = prepare_classification_images(
                        [body_crop],
                        self.model.input_size,
                        self.model.data_config["mean"],
                        self.model.data_config["std"],
                        device=self.device,
                    )
                    model_input = torch.cat((faces_input, person_input), dim=1)
                    output = self.model.inference(model_input)

                    age_raw = output[:, 2].item()
                    age = age_raw * (self.model.meta.max_age - self.model.meta.min_age) + self.model.meta.avg_age
                    age = round(float(age), 1)

                    gender_probs = output[:, :2].softmax(-1)
                    female_prob = gender_probs[0, 1].item()
                    gender = "female" if female_prob >= self.female_threshold else "male"

                if age is not None or gender is not None:
                    with self.evidence_lock:
                        if track_id not in self.evidence_by_track:
                            self.evidence_by_track[track_id] = TrackEvidence(max_samples=self.max_samples)
                        self.evidence_by_track[track_id].record(age, gender)

                    if self.on_result is not None:
                        try:
                            self.on_result(track_id, age, gender)
                        except Exception as cb_err:
                            logger.warning(f"Error in on_result callback for track {track_id}: {cb_err}")

                self.last_latency_ms = (time.time() - t0) * 1000.0

            except Exception as e:
                logger.warning(f"Error during async demographic inference: {e}")
            finally:
                self.queue.task_done()

    def get_demographics(self, track_id: int) -> Optional[TrackEvidence]:
        """Return a copy of current demographic evidence for track_id."""
        with self.evidence_lock:
            return self.evidence_by_track.get(track_id) or self.finalized_history.get(track_id)

    def is_finalized(self, track_id: int) -> bool:
        """Check if track has enough observations to stop sampling."""
        with self.evidence_lock:
            ev = self.evidence_by_track.get(track_id) or self.finalized_history.get(track_id)
            return ev.finalized if ev else False

    def evict_lost_tracks(self, lost_ids: Set[int]) -> None:
        """Evict departed tracks to prevent memory leaks while retaining finalized history."""
        if not lost_ids:
            return
        with self.evidence_lock:
            for tid in lost_ids:
                ev = self.evidence_by_track.pop(tid, None)
                if ev and ev.observations:
                    if len(self.finalized_order) >= 1000:
                        oldest = self.finalized_order.popleft()
                        self.finalized_history.pop(oldest, None)
                    self.finalized_order.append(tid)
                    self.finalized_history[tid] = ev

    def shutdown(self) -> None:
        """Gracefully stop the background worker thread."""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
