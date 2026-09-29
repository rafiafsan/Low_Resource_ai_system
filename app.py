"""MCP Tracking - Low-Resource Retail AI Analytics Engine
======================================================
Re-engineered for Windows Mini PC (Intel CPU, ~2GB RAM, No GPU).
Maintains real-time decoupled RTSP preview, ByteTrack, Footfall,
Event-driven Bag Exit validation, MiVOLO demographics, and PostgreSQL persistence.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Force reliable TCP transport for RTSP streams (prevents packet drops / tearing)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Target PC has no GPU
os.environ["YOLO_AUTOINSTALL"] = "0"  # Prevent dynamic pip invocations in frozen binary
os.environ["PYTHONUNBUFFERED"] = "1"   # Force line-buffered stdout in frozen binary

if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mivolo_system"))

import yaml

from services.unified_pipeline import UnifiedRetailPipeline


class _TeeWriter:
    """Wraps a stream and mirrors all writes to a secondary file sink.

    Used to redirect sys.stdout so that every print() call in the pipeline
    is simultaneously written to retail_ai.log regardless of TTY state.
    """

    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, data):
        if self._primary is not None:
            try:
                self._primary.write(data)
                self._primary.flush()
            except Exception:
                pass
        try:
            self._secondary.write(data)
            self._secondary.flush()
        except Exception:
            pass

    def flush(self):
        if self._primary is not None:
            try:
                self._primary.flush()
            except Exception:
                pass
        try:
            self._secondary.flush()
        except Exception:
            pass

    def fileno(self):
        if self._primary is not None:
            try:
                return self._primary.fileno()
            except Exception:
                pass
        raise io.UnsupportedOperation("fileno")

    def isatty(self):
        return False

    # Delegate any other attribute access to primary
    def __getattr__(self, name):
        return getattr(self._primary, name)


def setup_logging() -> None:
    """Configure logging + stdout tee so ALL output goes to retail_ai.log.

    Critical for frozen PyInstaller exe: sys.stdout is block-buffered (or
    None) when piped.  By opening a log file and tee-ing stdout into it we
    guarantee every print() and logging.* call is persisted immediately.
    """
    import io

    log_file = PROJECT_ROOT / "retail_ai.log"

    # Open the log file early — unbuffered so every line is flushed instantly
    try:
        log_fh = open(str(log_file), "w", encoding="utf-8", buffering=1)
    except Exception as e:
        log_fh = None
        print(f"[App] WARNING: Could not open log file {log_file}: {e}", flush=True)

    # Tee sys.stdout -> log file so all print() calls are captured
    if log_fh is not None:
        sys.stdout = _TeeWriter(sys.stdout, log_fh)
        sys.stderr = _TeeWriter(sys.stderr, log_fh)

    # Standard logging module — file + console
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if log_fh is not None:
        fh = logging.StreamHandler(log_fh)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Console handler via original stderr (before tee wrapping above)
    try:
        ch = logging.StreamHandler(sys.__stderr__)
        ch.setFormatter(fmt)
        root.addHandler(ch)
    except Exception:
        pass

    logging.info("=" * 60)
    logging.info("RetailAI v2 -- Log file: %s", log_file)
    logging.info("=" * 60)


def load_default_config(config_path: str = "config/production.yaml") -> dict:
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def parse_args():
    cfg = load_default_config()
    default_source = (
        cfg.get("camera", {}).get("rtsp_url")
        or cfg.get("camera", {}).get("fallback_video")
        or "test_videos/Outlet_Sample_1_Banasree.avi"
    )
    default_model = cfg.get("inference", {}).get("model", "models/yolov8m.onnx")

    parser = argparse.ArgumentParser(description="Retail Edge AI Analytics Engine (Low-Resource Re-engineered)")
    parser.add_argument(
        "--source",
        default=default_source,
        help="Video source: RTSP stream URL, local video file path, or USB camera index (e.g. 0).",
    )
    parser.add_argument(
        "--config",
        default="config/production.yaml",
        help="Path to production YAML configuration file.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive prompt for NVR RTSP stream and gate calibration.",
    )
    parser.add_argument(
        "--weights",
        default=default_model,
        help="Path to primary YOLO model (ONNX or PyTorch weights).",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Launch the interactive multi-detector gate calibration tool before running.",
    )
    parser.add_argument(
        "--no-demographics",
        action="store_true",
        help="Disable the asynchronous MiVOLO age/gender demographic profiling worker.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run in headless mode without opening an OpenCV preview window.",
    )

    # Backward compatibility modes
    parser.add_argument("--legacy-footfall", action="store_true", help="Run legacy CustomerCounter service.")
    parser.add_argument("--legacy-bag", action="store_true", help="Run legacy PersonBagCounter service.")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    # Interactive Wizard: Prompts when explicitly requested via --interactive
    if args.interactive:
        print("=" * 65)
        print("    Retail AI Analytics — Edge Video Tracking & Demographics")
        print("=" * 65)
        print(f"\n[Default Source]: {args.source}")
        rtsp_input = input("\nEnter NVR RTSP Stream URL or Video Path\n(Press ENTER to use default): ").strip()
        if rtsp_input:
            args.source = rtsp_input

        calib_input = input("\nDo you want to calibrate virtual gates for this camera? [y/N]: ").strip().lower()
        if calib_input in ("y", "yes"):
            args.calibrate = True

        display_input = input("\nShow live video preview window? [Y/n]: ").strip().lower()
        if display_input in ("n", "no"):
            args.no_display = True
        print("=" * 65)

    # 1. Optional pre-flight gate calibration
    if args.calibrate:
        print("[App] Launching Gate Calibration Tool...")
        from calibrate_gates import GateCalibrator

        calibrator = GateCalibrator(args.source)
        calibrator.run()

    # 2. Legacy fallback modes
    if args.legacy_footfall:
        print("[App] Running in Legacy Footfall mode...")
        from services.customer_counter import CustomerCounter

        counter = CustomerCounter(args.source)
        counter.run()
        return

    if args.legacy_bag:
        print("[App] Running in Legacy Person-Bag Counter mode...")
        from services.customer_bag_counter import PersonBagCounter

        counter = PersonBagCounter(args.source)
        counter.run()
        return

    # 3. Default: Unified Low-Resource Retail Analytics Pipeline
    print("=" * 65)
    print("  Starting Re-Engineered Unified Retail Analytics Engine")
    demo_mode = "Disabled" if args.no_demographics else "Enabled (Async MiVOLO CPU)"

    print(f" Source:            {args.source}")
    print(f" Primary Weights:   {args.weights}")
    print(f" Demographics:      {demo_mode}")
    print(f" Display Mode:      {'Headless' if args.no_display else 'Decoupled GUI Preview Window'}")
    print(f" Target Hardware:   Intel CPU (No GPU), 4 Cores, ~2GB RAM Budget")
    print("=" * 65)

    pipeline = UnifiedRetailPipeline(
        video_source=args.source,
        config_path=args.config,
        primary_model_path=args.weights,
        enable_demographics=not args.no_demographics,
        display=not args.no_display,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
