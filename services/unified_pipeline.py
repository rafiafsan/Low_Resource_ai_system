"""Unified Retail Analytics Pipeline (Low-Resource Re-engineered)
===============================================================
Consolidates Footfall Counting, Event-Driven Bag Exit Validation,
Asynchronous Demographic Profiling (MiVOLO), and PostgreSQL Persistence.

Architectural Guarantees:
- Decoupled preview: Display frame rate is completely decoupled from AI inference.
- Bound memory: Single-slot latest frame buffer, no queue accumulation, bounded state.
- Event-driven processing: Bag validation and demographic inferences only fire when triggered.
- Low-latency RTSP capture: Recovers automatically without crashing application.
- Configured for Intel CPU, 4 cores, ~2 GB RAM, No GPU.
- Direct MiVOLO inference: Bypasses secondary YOLO face detector, uses upper 25% of bbox as facial representation.
- Guaranteed Demographics DB Persistence: Every tracked person's age & gender is persisted into PostgreSQL.
"""

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mivolo_system"))

from camera.rtsp_reader import RTSPReader
from config.settings import CONFIDENCE, GATE_BAND_PIXELS, TRAIL_LENGTH
from database.db_manager import DatabaseManager
from detectors.person_bag_associator import BagPersonAassociator
from detectors.person_bag_exit_validator import PersonWithBagExitValidator
from gate.event_manager import EventManager
from gate.geometry import get_side
from gate.unified_gate_manager import UnifiedGateManager
from inference.detector import create_detector
from inference.inference_scheduler import InferenceScheduler
from mivolo_system.async_worker import AsyncDemographicWorker
from monitoring.performance_monitor import PerformanceMonitor
from services.business_logic import (
    process_entry_event,
    process_exit_event,
    validate_person_bag_exit,
)
from tracking.track_manager import TrackManager
from visualization.draw_track import draw_tracks

GATE_COLORS = {
    "master": (200, 200, 200),
    "footfall": (0, 230, 255),      # Yellow / Gold
    "bag_exit": (255, 200, 0),      # Cyan
    "demographics": (50, 255, 50),  # Lime Green
}


class UnifiedRetailPipeline:
    """Decoupled, high-performance retail vision pipeline for resource-constrained edge machines."""

    def __init__(
        self,
        video_source: Optional[str] = None,
        config_path: str = "config/production.yaml",
        enable_demographics: bool = True,
        display: bool = True,
        primary_model_path: Optional[str] = None,
        bag_model_path: Optional[str] = None,
    ):
        self.config_path = config_path
        self.display = display
        self.enable_demographics = enable_demographics
        self.config = self._load_config()

        # Resolution and pacing configuration
        cam_cfg = self.config.get("camera", {})
        inf_cfg = self.config.get("inference", {})
        self.source = video_source or cam_cfg.get("rtsp_url") or cam_cfg.get("fallback_video", "test_videos/Outlet_Sample_2_Banasree.avi")
        self.cam_width = int(cam_cfg.get("width", 1280))
        self.cam_height = int(cam_cfg.get("height", 720))

        self.inf_width = int(inf_cfg.get("width", 640))
        self.inf_height = int(inf_cfg.get("height", 360))
        self.ai_target_fps = float(inf_cfg.get("target_fps", 4.0))

        if primary_model_path:
            self.config.setdefault("inference", {})["model"] = primary_model_path

        # 1. Low-Latency Camera Grabber (Latest Frame, No Queue Backlog)
        print(f"[UnifiedPipeline] Initializing RTSP reader for source: {self.source}")
        self.reader = RTSPReader(
            source=self.source,
            width=self.cam_width,
            height=self.cam_height,
            reconnect_interval_sec=float(cam_cfg.get("reconnect_interval_sec", 2.0)),
        )

        # 2. Gate Configuration
        self.gate_manager = UnifiedGateManager()

        # 3. Primary Lightweight Detector (ONNX Runtime CPU / PyTorch Fallback)
        self.detector = create_detector(self.config)

        # 4. ByteTrack and Motion Analytics
        self.track_manager = TrackManager(trail_length=int(self.config.get("tracking", {}).get("trail_length", TRAIL_LENGTH)))
        gates_cfg = self.config.get("gates", {})
        self.event_manager = EventManager(
            min_entry_dwell_seconds=float(gates_cfg.get("min_entry_dwell_seconds", 60)),
            min_exit_dwell_seconds=float(gates_cfg.get("min_exit_dwell_seconds", 10)),
            deadband_pixels=int(gates_cfg.get("deadband_pixels", 15)),
        )
        bag_cfg = self.config.get("bag_detection", {})
        self.associator = BagPersonAassociator(
            min_consecutive_frames=int(bag_cfg.get("min_consecutive_frames", 1)),
            max_bag_distance=float(bag_cfg.get("max_bag_distance", 240)),
            min_iob=float(bag_cfg.get("min_iob", 0.03)),
            association_threshold=float(bag_cfg.get("association_threshold", 0.30)),
        )
        self.exit_validator = PersonWithBagExitValidator(
            min_exit_frames=int(bag_cfg.get("min_exit_frames", 1)),
            movement_threshold=float(bag_cfg.get("movement_threshold", 10.0)),
        )
        self._sync_gate_coordinates()

        # 5. Adaptive Inference Scheduler
        self.scheduler = InferenceScheduler(
            target_fps=self.ai_target_fps,
            cpu_limit_percent=float(self.config.get("safeguards", {}).get("cpu_limit_percent", 90.0)),
        )

        # 6. Asynchronous PostgreSQL Database Writer
        db_cfg = self.config.get("database", {})
        shop_cfg = self.config.get("shop", {})
        self.db_manager = DatabaseManager(
            credentials_file=db_cfg.get("credentials_file", "config/database.json"),
            shop_id=int(shop_cfg.get("shop_id", 1)),
            camera_id=str(shop_cfg.get("camera_id", "cam_01")),
        )

        # 7. Direct MiVOLO Demographic Worker (Lazy, Face Detector Bypassed)
        demo_cfg = self.config.get("demographics", {})
        if self.enable_demographics and demo_cfg.get("enabled", True):
            self.demographic_worker: Optional[AsyncDemographicWorker] = AsyncDemographicWorker(
                checkpoint_path=demo_cfg.get("checkpoint"),
                device=demo_cfg.get("mode", "cpu"),
                max_queue_size=int(demo_cfg.get("queue_size", 4)),
                max_samples_per_person=int(demo_cfg.get("max_samples", 4)),
                cpu_threads=int(demo_cfg.get("cpu_threads", 1)),
                female_threshold=float(demo_cfg.get("female_threshold", 0.50)),
                lazy=True,
                on_result=self._on_demographic_result,
            )
        else:
            self.demographic_worker = None

        # 8. Real-Time Performance Monitor
        safeguards = self.config.get("safeguards", {})
        self.perf_monitor = PerformanceMonitor(
            ram_limit_mb=float(safeguards.get("ram_limit_mb", 1750.0)),
            cpu_limit_percent=float(safeguards.get("cpu_limit_percent", 90.0)),
        )

        # Thread-safe snapshot state for display rendering
        self._state_lock = threading.Lock()
        self.cached_persons: List[Dict] = []
        self.cached_bags: List[Dict] = []
        self.cached_total_bag_exits: int = 0
        self.cached_demo_labels: Dict[int, str] = {}

        # Tracking state
        self.monitored_for_demographics: Set[int] = set()
        self.demographic_sampling_counts: Dict[int, int] = {}
        self.recorded_db_demographics: Set[int] = set()
        self.recorded_entry_db_ids: Set[int] = set()

        # Database periodic sync timers
        self.last_metrics_sync = time.time()
        self.last_occupancy_sync = time.time()
        self.occupancy_interval = float(db_cfg.get("occupancy_interval_minutes", 20.0)) * 60.0

        # Concurrency flags
        self.running = False
        self.ai_thread: Optional[threading.Thread] = None
        self.window_name = "RetailAI - Edge Analytics"

    def _record_demographics_for_track(
        self,
        track_id: int,
        demo: Any,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        direction: str = "ENTRY",
    ) -> None:
        """Persist finalized or accumulated demographic profiling data for a track ID."""
        if track_id in self.recorded_db_demographics:
            return
        self.recorded_db_demographics.add(track_id)

        avg_age = demo.average_age
        age_val = int(round(avg_age)) if avg_age is not None else None
        age_group = "Adult"
        if age_val is not None:
            if age_val < 18:
                age_group = "Child"
            elif age_val < 30:
                age_group = "Young Adult"
            elif age_val < 70:
                age_group = "Senior"
          
        gender_val = demo.final_gender.title() if demo.final_gender else "Unknown"

        if self.db_manager.enabled:
            self.db_manager.insert_demographic(
                track_id=track_id,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                bbox=bbox,
                direction=direction,
                result_dict={
                    "age_group": age_group,
                    "exact_age": age_val,
                    "gender": gender_val,
                    "confidence": "High",
                },
            )
            self.db_manager.update_entry_demographics(track_id, age_group, gender_val)

    def _record_fallback_demographics(
        self,
        track_id: int,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        direction: str = "ENTRY",
    ) -> None:
        """Guarantee every tracking ID has age and gender recorded in PostgreSQL."""
        if track_id in self.recorded_db_demographics:
            return
        self.recorded_db_demographics.add(track_id)

        if self.db_manager.enabled:
            self.db_manager.insert_demographic(
                track_id=track_id,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                bbox=bbox,
                direction=direction,
                result_dict={
                    "age_group": "Adult",
                    "exact_age": 30,
                    "gender": "Unknown",
                    "confidence": "Estimated",
                },
            )
            self.db_manager.update_entry_demographics(track_id, "Adult", "Unknown")

    def _on_demographic_result(self, track_id: int, age: float, gender: str) -> None:
        """Instant callback invoked when MiVOLO worker finishes inference for a crop."""
        if track_id in self.recorded_db_demographics:
            return
        self.recorded_db_demographics.add(track_id)

        age_val = int(round(age)) if age is not None else None
        age_group = "Adult"
        if age_val is not None:
            if age_val < 18:
                age_group = "Child"
            elif age_val < 30:
                age_group = "Young Adult"
            elif age_val < 50:
                age_group = "Adult"
            else:
                age_group = "Senior"
        gender_val = gender.title() if gender else "Unknown"

        if self.db_manager.enabled:
            self.db_manager.insert_demographic(
                track_id=track_id,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                bbox=None,
                direction="ENTRY",
                result_dict={
                    "age_group": age_group,
                    "exact_age": age_val,
                    "gender": gender_val,
                    "confidence": "High",
                },
            )
            self.db_manager.update_entry_demographics(track_id, age_group, gender_val)

    def _load_config(self) -> Dict:
        """Load YAML configuration from file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[UnifiedPipeline] Warning: failed to load {self.config_path}: {e}")
        return {}

    def _sync_gate_coordinates(self) -> None:
        """Propagate gate configuration to active event validators."""
        bag_pts, bag_ref = self.gate_manager.get_gate("bag_exit")
        if bag_pts and bag_ref:
            self.exit_validator.gate_points = [bag_pts[0], bag_pts[1]]
            self.exit_validator.inside_reference = bag_ref

    def _draw_gates(self, frame: np.ndarray) -> None:
        """Draw virtual gate lines and directional inside reference markers."""
        for det_name in ("footfall", "bag_exit", "demographics"):
            pts, ref = self.gate_manager.get_gate(det_name)
            if not pts or not ref:
                continue

            color = GATE_COLORS.get(det_name, (255, 255, 255))
            pt1, pt2 = pts
            cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2
            cv2.putText(
                frame,
                det_name.upper(),
                (mid_x - 30, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.circle(frame, ref, 5, color, -1)
            cv2.arrowedLine(frame, (mid_x, mid_y), ref, color, 1, tipLength=0.25)

    def _ai_worker_loop(self) -> None:
        """Dedicated AI inference thread: decoupled from GUI display loop."""
        print("[UnifiedPipeline] AI inference worker thread started.")
        footfall_pts, footfall_ref = self.gate_manager.get_gate("footfall")
        demo_pts, demo_ref = self.gate_manager.get_gate("demographics")

        while self.running:
            if not self.scheduler.should_process_next_frame():
                time.sleep(0.005)
                continue

            # Check safeguards: if RAM is critical, pause demographic capture
            ram_crit, cpu_crit = self.perf_monitor.update_hardware_metrics()

            # 1. Fetch latest downscaled inference frame and native frame
            ret, inf_frame, scales, fid = self.reader.get_inference_frame(
                target_width=self.inf_width, target_height=self.inf_height
            )
            if not ret or inf_frame is None:
                time.sleep(0.01)
                continue

            _, native_frame, _, _ = self.reader.get_latest_frame()

            # Original dimensions (dynamic from active stream)
            orig_shape = self.reader.frame_shape or (self.cam_height, self.cam_width)

            # 2. Execute Detection (Person + Bag)
            t_det_start = time.time()
            raw_persons, raw_bags, inf_lat_ms = self.detector.detect(inf_frame, original_shape=orig_shape)
            self.perf_monitor.inference_ms = inf_lat_ms

            # 3. ByteTrack Multi-Object Tracking
            t_track_start = time.time()
            tracked_persons, tracked_bags, lost_ids = self.track_manager.track(raw_persons, raw_bags)
            self.perf_monitor.tracking_ms = (time.time() - t_track_start) * 1000.0

            # 4. Immediate Stale Track Eviction (Prevents memory accumulation)
            if lost_ids:
                self.event_manager.evict_lost_tracks(lost_ids)
                self.associator.evict_lost_tracks(lost_ids)
                self.exit_validator.evict_lost_tracks(lost_ids)
                if self.demographic_worker:
                    # Guarantee: Persist any unrecorded demographic observations before track eviction
                    if self.db_manager.enabled:
                        for tid in lost_ids:
                            if tid not in self.recorded_db_demographics:
                                demo = self.demographic_worker.get_demographics(tid)
                                if demo and demo.observations:
                                    self._record_demographics_for_track(tid, demo)
                                else:
                                    self._record_fallback_demographics(tid)
                    self.demographic_worker.evict_lost_tracks(lost_ids)
                else:
                    if self.db_manager.enabled:
                        for tid in lost_ids:
                            if tid not in self.recorded_db_demographics:
                                self._record_fallback_demographics(tid)
                for tid in lost_ids:
                    self.monitored_for_demographics.discard(tid)
                    self.demographic_sampling_counts.pop(tid, None)

            # 5. Dual-Gate Crossing & Event Validation
            t_gate_start = time.time()
            if footfall_pts and footfall_ref:
                for person in tracked_persons:
                    t_id = person["track_id"]
                    ev = self.event_manager.process(t_id, person["center"], footfall_pts, footfall_ref)
                    if ev:
                        ts_str = ev["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                        if ev["event_type"] == "ENTRY" and not ev.get("cancelled_exit"):
                            self.recorded_entry_db_ids.add(t_id)
                            # Pull any demographics observed so far
                            demo = self.demographic_worker.get_demographics(t_id) if self.demographic_worker else None
                            demo_dict = None
                            if demo and demo.observations:
                                avg_age = demo.average_age
                                age_val = int(round(avg_age)) if avg_age is not None else None
                                age_group = "Adult"
                                if age_val is not None:
                                    if age_val < 18:
                                        age_group = "Child"
                                    elif age_val < 30:
                                        age_group = "Young Adult"
                                    elif age_val < 50:
                                        age_group = "Adult"
                                    else:
                                        age_group = "Senior"
                                gender_val = demo.final_gender.title() if demo.final_gender else "Unknown"
                                demo_dict = {"age_group": age_group, "gender": gender_val}

                            rec = process_entry_event(t_id, ts_str, ev, demographics=demo_dict)
                            if rec and self.db_manager.enabled:
                                self.db_manager.insert_entry(**rec)
                            print(f"[UnifiedPipeline] >>> RECORDED ENTRY: Track {t_id} (Total In={self.event_manager.entry_count})")
                        elif ev["event_type"] == "EXIT" and not ev.get("cancelled_entry"):
                            rec = process_exit_event(t_id, ts_str, ev)
                            if rec and self.db_manager.enabled:
                                self.db_manager.insert_exit(**rec)
                            print(f"[UnifiedPipeline] >>> RECORDED EXIT: Track {t_id} (Total Out={self.event_manager.exit_count})")

                    # Demographics Trigger
                    if self.enable_demographics and demo_pts and demo_ref and not ram_crit:
                        side = get_side(person["center"], demo_pts[0], demo_pts[1], demo_ref)
                        if side == "INSIDE" and t_id not in self.monitored_for_demographics:
                            self.monitored_for_demographics.add(t_id)
                            self.demographic_sampling_counts[t_id] = 0
                    elif self.enable_demographics and not ram_crit and t_id in self.recorded_entry_db_ids:
                        self.monitored_for_demographics.add(t_id)
                        if t_id not in self.demographic_sampling_counts:
                            self.demographic_sampling_counts[t_id] = 0

            # Periodic state cleanup in event manager
            self.event_manager.validate_pending()
            self.perf_monitor.gate_ms = (time.time() - t_gate_start) * 1000.0

            # 6. Bag Exit Verification (Temporal Correlation)
            t_bag_start = time.time()
            total_bag_exits = self.exit_validator.get_count()
            if tracked_persons and (
                tracked_bags
                or self.associator.person_bag_associations
                or self.exit_validator.pending_associations
            ):
                associations = self.associator.associate_person_bag_exit(tracked_persons, tracked_bags)
                self.exit_validator.update_associations(associations, tracked_persons)
                exit_events = self.exit_validator.update(tracked_persons)

                if exit_events:
                    for ev in exit_events:
                        if validate_person_bag_exit(ev["person_track_id"], ev["bag_track_id"], ev):
                            print(f"[UnifiedPipeline] >>> CONFIRMED EXIT WITH BAG: {ev}")

                total_bag_exits = self.exit_validator.get_count()

            self.perf_monitor.bag_ms = (time.time() - t_bag_start) * 1000.0

            # 7. Demographics Inference & Guaranteed DB Insertion
            t_demo_start = time.time()
            if self.enable_demographics and self.demographic_worker and not ram_crit:
                for person in tracked_persons:
                    t_id = person["track_id"]
                    # Sample crops for every active person until finalized
                    if not self.demographic_worker.is_finalized(t_id):
                        count = self.demographic_sampling_counts.get(t_id, 0)
                        if count < 4:
                            self.demographic_sampling_counts[t_id] = count + 1
                            px1, py1, px2, py2 = person["bbox"]
                            if native_frame is not None:
                                crop = native_frame[max(0, py1):min(orig_shape[0], py2), max(0, px1):min(orig_shape[1], px2)]
                            else:
                                crop = inf_frame[max(0, int(py1 / scales[1])):min(inf_frame.shape[0], int(py2 / scales[1])),
                                                 max(0, int(px1 / scales[0])):min(inf_frame.shape[1], int(px2 / scales[0]))]
                            if crop.size > 0:
                                self.demographic_worker.submit_crop(t_id, crop)

                    # Guarantee: Ensure every tracking ID's age & gender is recorded into database
                    if t_id not in self.recorded_db_demographics:
                        demo = self.demographic_worker.get_demographics(t_id)
                        if demo and demo.observations:
                            self._record_demographics_for_track(t_id, demo, bbox=person["bbox"])

            self.perf_monitor.demographic_ms = (time.time() - t_demo_start) * 1000.0

            # 8. Periodic Metrics & Occupancy Syncer
            now_t = time.time()
            if self.db_manager.enabled:
                if (now_t - self.last_metrics_sync) >= 5.0:
                    self.last_metrics_sync = now_t
                    self.db_manager.upsert_daily_metrics(
                        in_count=self.event_manager.entry_count,
                        out_count=self.event_manager.exit_count,
                        person_with_bag_exit=total_bag_exits,
                        demographics_count=len(self.recorded_db_demographics),
                    )

                if (now_t - self.last_occupancy_sync) >= self.occupancy_interval:
                    self.last_occupancy_sync = now_t
                    self.db_manager.insert_occupancy(
                        valid_entries=self.event_manager.entry_count,
                        valid_exits=self.event_manager.exit_count,
                    )

            # 9. Format Overlay Labels & Publish Snapshot to GUI Thread
            demo_labels = {}
            if self.enable_demographics and self.demographic_worker:
                for person in tracked_persons:
                    tid = person["track_id"]
                    demo = self.demographic_worker.get_demographics(tid)
                    if demo and demo.observations:
                        demo_labels[tid] = demo.summary_label

            with self._state_lock:
                self.cached_persons = tracked_persons
                self.cached_bags = tracked_bags
                self.cached_total_bag_exits = total_bag_exits
                self.cached_demo_labels = demo_labels

            # Record iteration performance
            total_lat = (time.time() - t_det_start) * 1000.0
            self.scheduler.record_inference_completed(total_lat)

            # Update live stats
            self.perf_monitor.capture_fps = self.reader.capture_fps
            self.perf_monitor.ai_fps = self.scheduler.measured_ai_fps
            self.perf_monitor.db_queue_size = self.db_manager.job_queue.qsize()
            self.perf_monitor.demographic_queue_size = (
                self.demographic_worker.queue_size if self.demographic_worker else 0
            )
            self.perf_monitor.rtsp_status = (
                "ONLINE" if self.reader.is_connected else "RECONNECTING"
            )

    def run(self) -> None:
        """Main display thread: runs GUI preview at full camera rate."""
        self.running = True

        # Start background AI worker thread
        self.ai_thread = threading.Thread(target=self._ai_worker_loop, name="AI-InferenceThread", daemon=True)
        self.ai_thread.start()

        if self.display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        print("[UnifiedPipeline] Entering display loop. Decoupled preview active.")
        try:
            while self.running:
                ret, frame = self.reader.get_display_frame()
                if not ret or frame is None:
                    time.sleep(0.015)
                    continue

                if self.display:
                    # Draw virtual gate lines
                    self._draw_gates(frame)

                    # Retrieve latest cached AI state safely
                    with self._state_lock:
                        persons = list(self.cached_persons)
                        bags = list(self.cached_bags)
                        total_bag_exits = self.cached_total_bag_exits
                        demo_labels = dict(self.cached_demo_labels)

                    # Draw motion trails
                    draw_tracks(frame, persons, self.track_manager)

                    # Draw confirmed person-bag association lines
                    frame = self.associator.draw_confirmed_associations_bag_exit(frame, persons, bags)

                    # Draw person bounding boxes and demographic badges
                    for det in persons:
                        px1, py1, px2, py2 = det["bbox"]
                        t_id = det["track_id"]
                        label = f"ID: {t_id}"
                        if t_id in demo_labels:
                            label += f" | {demo_labels[t_id]}"

                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            label,
                            (px1, max(20, py1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                    # Draw bag bounding boxes
                    for bag in bags:
                        bx1, by1, bx2, by2 = bag["bbox"]
                        b_id = bag["track_id"]
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                        cv2.putText(
                            frame,
                            f"BAG: {b_id}",
                            (bx1, max(15, by1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    # Draw lightweight HUD overlay
                    profiling_count = len([t for t in self.monitored_for_demographics if not (self.demographic_worker and self.demographic_worker.is_finalized(t))])
                    pending_dwell = len(self.event_manager.pending_entries) + len(self.event_manager.pending_exits)
                    self.perf_monitor.draw_hud(
                        frame=frame,
                        entry_count=self.event_manager.entry_count,
                        exit_count=self.event_manager.exit_count,
                        bag_exit_count=total_bag_exits,
                        active_profiling_count=profiling_count,
                        valid_entry_count=self.event_manager.validated_entry_count,
                        valid_exit_count=self.event_manager.validated_exit_count,
                        pending_count=pending_dwell,
                    )

                    cv2.imshow(self.window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("[UnifiedPipeline] User requested quit.")
                        break
                    elif key == ord("r"):
                        print("[UnifiedPipeline] Reloading gates from config/gates.json...")
                        self.gate_manager.load_config()
                        self._sync_gate_coordinates()
                else:
                    time.sleep(0.02)

        finally:
            self.stop()

    def stop(self) -> None:
        """Clean shutdown sequence."""
        self.running = False
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=2.0)
        self.reader.release()
        if self.demographic_worker:
            self.demographic_worker.shutdown()
        if self.db_manager.enabled:
            # Guarantee: Ensure any remaining active track IDs have demographics persisted
            for tid in list(self.track_manager.active_ids):
                if tid not in self.recorded_db_demographics:
                    demo = self.demographic_worker.get_demographics(tid) if self.demographic_worker else None
                    if demo and demo.observations:
                        self._record_demographics_for_track(tid, demo)
                    else:
                        self._record_fallback_demographics(tid)

            self.db_manager.upsert_daily_metrics(
                in_count=self.event_manager.entry_count,
                out_count=self.event_manager.exit_count,
                person_with_bag_exit=self.cached_total_bag_exits,
                demographics_count=len(self.recorded_db_demographics),
            )
            self.db_manager.close(timeout=2.0)
        if self.display:
            cv2.destroyAllWindows()
        print("[UnifiedPipeline] Pipeline closed cleanly.")
