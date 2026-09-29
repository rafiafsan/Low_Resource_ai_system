"""Run MiVOLO age and gender detection on a CCTV stream or local video.

Examples:
    python app.py --source "rtsp://user:password@192.168.1.10:554/stream1"
    python app.py --source local
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
import cv2
import torch


# This file lives in the MiVOLO project root, which must be importable for
# ``MiVOLO`` and ``model`` imports used by the predictor.
PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_VIDEO_PATH = PROJECT_DIR/"D:\\MCP_Tracking\\Videos\\Outlet_Sample_4_Banasree.avi"
DETECTOR_WEIGHTS = PROJECT_DIR/"model"/"yolov8x_person_face.pt"
MIVOLO_CHECKPOINT = PROJECT_DIR/"model"/"model_imdb_cross_person_4.22_99.46.pth.tar"
sys.path.insert(0, str(PROJECT_DIR))

class MiVOLOConfig:
    def __init__(self, device: str):
        self.detector_weights = str(DETECTOR_WEIGHTS)
        self.checkpoint = str(MIVOLO_CHECKPOINT)
        self.device = device
        self.with_persons = True
        self.disable_faces = False
        self.draw = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiVOLO age and gender detection for CCTV or video files.")
    parser.add_argument(
        "--source",
        default="local",
        help='RTSP/HTTP CCTV URL, camera index, or "local" to use LOCAL_VIDEO_PATH.',
    )
    parser.add_argument(
        "--local-video",
        default=str(LOCAL_VIDEO_PATH),
        help="Local video file used when --source local (default: %(default)s).",
    )
    return parser.parse_args()


def resolve_source(source: str, local_video: str):
    if source.lower() == "local":
        video_path = Path(local_video)
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Local test video was not found: {video_path}. "
                "Set --local-video to an existing video file."
            )
        return str(video_path)
    return int(source) if source.isdigit() else source


def main() -> None:
    args = parse_args()
    source = resolve_source(args.source, args.local_video)

    # Delay the model import so command-line help remains available even before
    # the inference dependencies are installed.
    from MiVOLO.predictor import Predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    predictor = Predictor(MiVOLOConfig(device=device), verbose=False)
    history = defaultdict(list)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    print(f"Processing {source} on {device}. Press q in the video window to stop.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Existing MiVOLO detector, tracking, prediction, and smoothing flow.
            detected_objects = predictor.detector.track(frame)
            predictor.age_gender_model.predict(frame, detected_objects)
            current_frame_objs = detected_objects.get_results_for_tracking()

            for tracked_objects in current_frame_objs:
                for guid, data in tracked_objects.items():
                    if None not in data:
                        history[guid].append(data)

            detected_objects.set_tracked_age_gender(history)
            cv2.imshow("MiVOLO Age and Gender", detected_objects.plot())

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
