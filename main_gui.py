"""Retail Analytics v2 — Standalone Windowed GUI Launcher
=======================================================
Pure graphical application launcher with zero terminal window (console=False).
All camera inputs, gate calibrations, and live tracking metrics occur in the GUI.
"""

import os
import sys
from pathlib import Path

# Force reliable TCP transport for RTSP streams
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["YOLO_AUTOINSTALL"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mivolo_system"))

from gui.retail_gui import launch_gui


def main():
    launch_gui("config/production.yaml")


if __name__ == "__main__":
    main()
