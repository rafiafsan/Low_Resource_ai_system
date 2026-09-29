"""
main.py — Entry point for Retail CCTV Customer Analytics System.

Pipeline:
  1. Reads video frames using OpenCV.
  2. Runs person detection and persistent tracking via YOLO + ByteTrack/BoT-SORT.
  3. Uses Consistent Large Bounding Box logic (TrackManager) to detect when a customer
     is stably close and clearly visible to the camera.
  4. Saves a single high-quality annotated snapshot of the target person.
  5. Offloads VLM demographic analysis (age group, gender, confidence) to an asynchronous
     worker thread using Ollama (qwen2.5vl:7b).
  6. Outputs formatted demographic reports to the terminal and appends to CSV.
  7. Displays real-time annotated video with tracking boxes and customer demographics.

Usage:
  python main.py
  python main.py --video "D:\\Ollama2.5_Test\\Outlet_Sample_3_Banasree.avi"
  python main.py --no-display --min-ratio 0.04 --consistent-frames 8
"""

import sys
import time
import argparse
import threading
import queue
from pathlib import Path

import cv2
import numpy as np

# Memory error guard across numpy versions
_NP_MEM_ERR = getattr(getattr(np, '_core', None), '_exceptions', None)
_NP_ARRAY_MEM_ERR = getattr(_NP_MEM_ERR, '_ArrayMemoryError', None) if _NP_MEM_ERR else None
_MEM_ERRORS = (MemoryError,) + ((type(_NP_ARRAY_MEM_ERR),) if _NP_ARRAY_MEM_ERR and isinstance(_NP_ARRAY_MEM_ERR, type) else ())

from tracker import PersonTracker
from track_manager import TrackManager, MIN_BBOX_RATIO, CONSISTENT_FRAMES
from csv_writer import CSVWriter
from vlm_client import VLMClient
from utils import (
    draw_person_box,
    draw_hud,
    ensure_dir,
    setup_logger,
    expand_bbox,
    crop_from_frame,
)

logger = setup_logger()

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_VIDEO_PATH_3 = r"D:\Ollama2.5_Test\Outlet_Sample_2_Banasree.avi"
# DEFAULT_VIDEO_PATH_2 = r"D:\Ollama2.5_Test\Outlet_Sample_2_Banasree.avi"
DEFAULT_YOLO_MODEL    = "yolo11n.pt"      # nano weights for fast person detection
DEFAULT_TRACKER       = "bytetrack.yaml"  # ByteTrack persistent tracker
DEFAULT_VLM_MODEL     = "qwen2.5vl:7b"    # Ollama vision-language model
DEFAULT_CONF_THRESH   = 0.35
DEFAULT_OUTPUT_DIR    = "output"
DEFAULT_CSV_PATH      = "output/analytics.csv"
DEFAULT_DISPLAY_SCALE = 0.85


# ---------------------------------------------------------------------------
# VLM worker thread (runs analysis without blocking the video loop)
# ---------------------------------------------------------------------------

class VLMWorker(threading.Thread):
    """
    Background worker that runs VLM demographic analysis asynchronously.
    Pulls jobs from a thread-safe queue so the main video loop runs at full speed.
    """

    def __init__(
        self,
        vlm_client: VLMClient,
        track_manager: TrackManager,
        csv_writer: CSVWriter,
    ):
        super().__init__(daemon=True, name="VLMWorker")
        self.vlm_client    = vlm_client
        self.track_manager = track_manager
        self.csv_writer    = csv_writer
        self.job_queue: queue.Queue = queue.Queue()
        self._stop_event   = threading.Event()

    def submit(
        self,
        track_id: int,
        full_frame_path: str,
        crop_path: str,
        frame_no: int,
        bbox: tuple,
    ) -> None:
        """Enqueue a VLM analysis job with full frame and person crop."""
        self.job_queue.put((track_id, full_frame_path, crop_path, frame_no, bbox))

    def stop(self) -> None:
        """Signal the worker to stop after draining the queue."""
        self._stop_event.set()

    def run(self) -> None:
        logger.info("[VLMWorker] Background worker started.")
        while not self._stop_event.is_set() or not self.job_queue.empty():
            try:
                track_id, full_path, crop_path, frame_no, bbox = self.job_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                result = self.vlm_client.analyze(
                    full_frame_path=full_path,
                    crop_path=crop_path,
                    track_id=track_id,
                )
                if result:
                    # 1. Store result in track manager (prints terminal formatted report)
                    self.track_manager.store_result(track_id, result)

                    # 2. Append demographic record to CSV
                    trigger_info = self.track_manager.get_trigger_info(track_id)
                    timestamp = trigger_info["timestamp"] if trigger_info else ""
                    self.csv_writer.write_result(
                        track_id=track_id,
                        timestamp=timestamp,
                        frame_no=frame_no,
                        bbox=bbox,
                        result_dict=result,
                    )
            except Exception as exc:
                logger.error(f"[VLMWorker] Error analyzing track {track_id}: {exc}")
            finally:
                self.job_queue.task_done()

        logger.info("[VLMWorker] Background worker stopped.")


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_video = (
        DEFAULT_VIDEO_PATH_3 if Path(DEFAULT_VIDEO_PATH_3).exists()
        else DEFAULT_VIDEO_PATH_2
    )

    parser = argparse.ArgumentParser(
        description="Retail CCTV Customer Analytics — YOLO Tracking + Consistent Large BBox + VLM"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=default_video,
        help=f"Path to input video file (default: {default_video})"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_YOLO_MODEL,
        help=f"YOLO model weights (default: {DEFAULT_YOLO_MODEL})"
    )
    parser.add_argument(
        "--tracker", "-t",
        type=str,
        default=DEFAULT_TRACKER,
        choices=["bytetrack.yaml", "botsort.yaml"],
        help=f"Tracker config (default: {DEFAULT_TRACKER})"
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default=DEFAULT_VLM_MODEL,
        help=f"Ollama VLM model tag (default: {DEFAULT_VLM_MODEL})"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESH,
        help=f"YOLO detection confidence threshold (default: {DEFAULT_CONF_THRESH})"
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=MIN_BBOX_RATIO,
        help=f"Min bbox/frame area ratio to qualify as large (default: {MIN_BBOX_RATIO})"
    )
    parser.add_argument(
        "--consistent-frames",
        type=int,
        default=CONSISTENT_FRAMES,
        help=f"Consecutive frames required to trigger VLM (default: {CONSISTENT_FRAMES})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for frames (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help=f"Output CSV path (default: {DEFAULT_CSV_PATH})"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live GUI display (headless mode)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_DISPLAY_SCALE,
        help=f"Live display scaling factor (default: {DEFAULT_DISPLAY_SCALE})"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process (0 = process entire video)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    # Set up output directories
    output_dir = ensure_dir(args.output)
    frames_dir = ensure_dir(str(output_dir / "frames"))

    logger.info("=" * 60)
    logger.info("  Retail CCTV Customer Analytics System")
    logger.info("  Consistent Large BBox Trigger Architecture")
    logger.info("=" * 60)
    logger.info(f"  Video             : {video_path}")
    logger.info(f"  YOLO Model        : {args.model}")
    logger.info(f"  Tracker           : {args.tracker}")
    logger.info(f"  VLM Model         : {args.vlm}")
    logger.info(f"  Min BBox Ratio    : {args.min_ratio * 100:.1f}% of frame area")
    logger.info(f"  Consistent Frames : {args.consistent_frames} consecutive frames")
    logger.info(f"  Output CSV        : {args.csv}")
    logger.info(f"  Frames Directory  : {frames_dir}")
    logger.info("=" * 60)

    # Initialize modules
    tracker       = PersonTracker(
        model_path=args.model,
        conf_thresh=args.conf,
        tracker_cfg=args.tracker,
    )
    track_manager = TrackManager(
        min_bbox_ratio=args.min_ratio,
        consistent_frames=args.consistent_frames,
    )
    csv_writer    = CSVWriter(output_path=args.csv)
    vlm_client    = VLMClient(model=args.vlm)

    # Start asynchronous VLM worker thread
    vlm_worker = VLMWorker(
        vlm_client=vlm_client,
        track_manager=track_manager,
        csv_writer=csv_writer,
    )
    vlm_worker.start()

    # Open video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video capture for: {video_path}")
        vlm_worker.stop()
        sys.exit(1)

    fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_no     = 0

    logger.info(
        f"Video stream opened: {fps_in:.1f} FPS, {total_frames} total frames. "
        f"{'Display window active.' if not args.no_display else 'Headless execution.'}"
    )

    fps_timer_start = time.time()
    fps_measured    = 0.0
    fps_frame_count = 0

    if not args.no_display:
        cv2.namedWindow("Retail Analytics", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Reached end of video stream.")
                break

            frame_no += 1
            if args.max_frames > 0 and frame_no > args.max_frames:
                logger.info(f"Reached max frames limit: {args.max_frames}")
                break

            fps_frame_count += 1

            # 1. Run YOLO + Persistent Tracking
            try:
                tracked_persons = tracker.update(frame)
            except _MEM_ERRORS:
                logger.warning(f"[Pipeline] Low memory in tracker at frame {frame_no}, skipping.")
                continue
            except Exception as exc:
                logger.warning(f"[Pipeline] Tracker error at frame {frame_no}: {exc}")
                continue

            # 2. Update TrackManager and test Consistent Large BBox trigger
            for person in tracked_persons:
                is_triggered = track_manager.update(
                    track_id=person.track_id,
                    bbox=person.bbox,
                    frame_shape=frame.shape,
                    frame_index=frame_no,
                )

                if is_triggered:
                    # Mark immediately so it doesn't trigger again
                    track_manager.mark_triggered(
                        track_id=person.track_id,
                        frame_index=frame_no,
                        bbox=person.bbox,
                    )

                    # 1. Full frame snapshot with TARGET PERSON highlighted
                    annotated_target_frame = draw_person_box(
                        frame=frame,
                        bbox=person.bbox,
                        track_id=person.track_id,
                        is_target=True,
                    )
                    full_frame_filename = f"track_{person.track_id}_f{frame_no}_full.jpg"
                    full_frame_path     = str(frames_dir / full_frame_filename)
                    cv2.imwrite(full_frame_path, annotated_target_frame)

                    # 2. Person crop with slight expansion padding
                    crop_bbox = expand_bbox(person.bbox, frame.shape, pad_h=0.08, pad_v=0.08)
                    crop_img  = crop_from_frame(frame, crop_bbox)
                    crop_filename = f"track_{person.track_id}_f{frame_no}_crop.jpg"
                    crop_path     = str(frames_dir / crop_filename)
                    cv2.imwrite(crop_path, crop_img)

                    # Submit to background VLM worker queue (full frame + crop)
                    vlm_worker.submit(
                        track_id=person.track_id,
                        full_frame_path=full_frame_path,
                        crop_path=crop_path,
                        frame_no=frame_no,
                        bbox=person.bbox,
                    )

                    logger.info(
                        f"[Pipeline] Track {person.track_id} triggered! "
                        f"(BBox >= {args.min_ratio*100:.1f}% for {args.consistent_frames} consecutive frames) "
                        f"→ Snapshot: {full_frame_filename}, Crop: {crop_filename}"
                    )

            # 3. Build live display frame
            if not args.no_display:
                try:
                    display_frame = frame.copy()

                    for person in tracked_persons:
                        result = track_manager.get_analysis_result(person.track_id)
                        if result:
                            gender = result.get("gender", result.get("apparent_gender", "")).title()
                            age_group = result.get("age_group", "")
                        else:
                            gender = ""
                            age_group = ""

                        display_frame = draw_person_box(
                            frame=display_frame,
                            bbox=person.bbox,
                            track_id=person.track_id,
                            age_group=age_group,
                            gender=gender,
                        )

                    # FPS measurement
                    elapsed = time.time() - fps_timer_start
                    if elapsed >= 1.0:
                        fps_measured    = fps_frame_count / elapsed
                        fps_frame_count = 0
                        fps_timer_start = time.time()

                    # HUD Overlay
                    display_frame = draw_hud(
                        frame=display_frame,
                        fps=fps_measured,
                        track_count=len(tracked_persons),
                        analyzed_count=track_manager.analyzed_count,
                        frame_no=frame_no,
                    )

                    h, w = display_frame.shape[:2]
                    disp = cv2.resize(
                        display_frame,
                        (int(w * args.scale), int(h * args.scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    cv2.imshow("Retail Analytics", disp)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):  # 'q' or ESC
                        logger.info("User requested exit.")
                        break

                except _MEM_ERRORS:
                    pass

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping gracefully...")

    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

        logger.info("Waiting for VLM worker to complete remaining analyses...")
        vlm_worker.stop()
        vlm_worker.join(timeout=180)

        logger.info("=" * 60)
        logger.info("  Processing Complete")
        logger.info(f"  Total frames processed : {frame_no}")
        logger.info(f"  Total unique tracks    : {track_manager.total_tracks}")
        logger.info(f"  VLM analyses completed : {track_manager.analyzed_count}")
        logger.info(f"  Analytics CSV          : {csv_writer.path.resolve()}")
        logger.info(f"  Saved Snapshots        : {frames_dir.resolve()}")
        logger.info("=" * 60)


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
