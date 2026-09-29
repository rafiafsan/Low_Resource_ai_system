"""Standalone Interactive Gate Calibration Tool
=============================================
Allows operators to configure and preview virtual gate lines for each detector
(master, footfall, bag_exit, demographics) for different camera setups.

Usage:
    python calibrate_gates.py
    python calibrate_gates.py --source 0
    python calibrate_gates.py --source "rtsp://camera_ip:554/stream"
    python calibrate_gates.py --source "Videos/Outlet_Sample_2_Banasree.avi"
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Force reliable TCP transport for RTSP streams (prevents HEVC/H.264 packet loss)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import numpy as np

PROJECT_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import VIDEO_SOURCE
from gate.unified_gate_manager import UnifiedGateManager, VALID_DETECTORS, Point

GATE_COLORS = {
    "master": (220, 220, 220),       # Silver / White
    "footfall": (0, 230, 255),       # Yellow / Gold
    "bag_exit": (255, 200, 0),       # Cyan / Sky Blue
    "demographics": (50, 255, 50),   # Lime Green
}

DETECTOR_LABELS = {
    "master": "0: Master Gate",
    "footfall": "1: Footfall Gate",
    "bag_exit": "2: Bag Exit Gate",
    "demographics": "3: Demographics Gate",
}


class GateCalibrator:
    def __init__(self, source: str):
        self.source = int(source) if str(source).isdigit() else source
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        self.gate_manager = UnifiedGateManager()
        self.active_detector = "footfall"
        self.temp_points: Dict[str, List[Point]] = {}
        self.temp_inside: Dict[str, Optional[Point]] = {}

        # Initialize temporary points from saved manager
        self._load_from_manager()

        # Warmup read loop for RTSP streams (allows keyframe negotiation)
        self.current_frame = None
        for _ in range(40):
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                self.current_frame = frame
                break
            time.sleep(0.1)

        if self.current_frame is None:
            raise RuntimeError(f"Failed to read initial frame from video source: {self.source}")

        self.window_name = "MCP Tracking - Multi-Detector Gate Calibration"

    def _load_from_manager(self) -> None:
        """Populate working points from the saved gate manager."""
        for det in VALID_DETECTORS:
            pts, ref = self.gate_manager.get_gate(det)
            if pts:
                self.temp_points[det] = [pts[0], pts[1]]
                self.temp_inside[det] = ref
            else:
                self.temp_points[det] = []
                self.temp_inside[det] = None

    def on_mouse(self, event, x, y, flags, param) -> None:
        """Handle mouse clicks for the currently active detector."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        points = self.temp_points[self.active_detector]
        if len(points) < 2:
            points.append((x, y))
        elif self.temp_inside[self.active_detector] is None:
            self.temp_inside[self.active_detector] = (x, y)
        else:
            # 4th click resets and starts Point 1 again
            self.temp_points[self.active_detector] = [(x, y)]
            self.temp_inside[self.active_detector] = None

    def draw_ui(self, canvas: np.ndarray) -> np.ndarray:
        """Render all gates, directions, and instructions on the canvas."""
        h, w = canvas.shape[:2]

        # Draw dark top banner for HUD
        cv2.rectangle(canvas, (0, 0), (w, 85), (20, 20, 20), -1)
        cv2.line(canvas, (0, 85), (w, 85), (70, 70, 70), 1)

        # Draw detector selection pills
        x_offset = 15
        for det in VALID_DETECTORS:
            label = DETECTOR_LABELS[det]
            color = GATE_COLORS[det]
            is_active = det == self.active_detector

            box_bg = (60, 60, 60) if not is_active else (color[0] // 3, color[1] // 3, color[2] // 3)
            box_border = (120, 120, 120) if not is_active else color
            text_color = (200, 200, 200) if not is_active else (255, 255, 255)

            # Draw button box
            cv2.rectangle(canvas, (x_offset, 10), (x_offset + 175, 42), box_bg, -1)
            cv2.rectangle(canvas, (x_offset, 10), (x_offset + 175, 42), box_border, 2 if is_active else 1)
            cv2.putText(canvas, label, (x_offset + 8, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)
            x_offset += 190

        # Draw instructions text
        current_pts = self.temp_points[self.active_detector]
        current_ref = self.temp_inside[self.active_detector]
        if len(current_pts) == 0:
            step_msg = "Click Point 1: GATE START"
        elif len(current_pts) == 1:
            step_msg = "Click Point 2: GATE END"
        elif current_ref is None:
            step_msg = "Click Point 3: INSIDE REFERENCE POINT"
        else:
            step_msg = "Gate complete! [S]: Save | [R]: Reset Active | [N]: Next Frame | [Q]: Quit"

        active_color = GATE_COLORS[self.active_detector]
        cv2.putText(canvas, f"ACTIVE: {self.active_detector.upper()} -> {step_msg}",
                    (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, active_color, 2, cv2.LINE_AA)

        # Draw all configured gates
        for det in VALID_DETECTORS:
            pts = self.temp_points[det]
            ref = self.temp_inside[det]
            color = GATE_COLORS[det]
            is_active = det == self.active_detector
            thickness = 3 if is_active else 1

            # Draw gate endpoints
            for i, pt in enumerate(pts):
                cv2.circle(canvas, pt, 6 if is_active else 4, color, -1)
                cv2.putText(canvas, f"{det[:3].upper()} P{i+1}", (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # Draw gate line
            if len(pts) == 2:
                cv2.line(canvas, pts[0], pts[1], color, thickness)

                # Draw inside reference point and directional arrow
                if ref is not None:
                    cv2.circle(canvas, ref, 6 if is_active else 4, (0, 255, 0), -1)
                    cv2.putText(canvas, f"{det.upper()} INSIDE", (ref[0] + 8, ref[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)

                    # Draw direction arrow from midpoint to inside
                    mid_x = (pts[0][0] + pts[1][0]) // 2
                    mid_y = (pts[0][1] + pts[1][1]) // 2
                    cv2.arrowedLine(canvas, (mid_x, mid_y), ref, color, 2 if is_active else 1, tipLength=0.2)

        return canvas

    def save(self) -> None:
        """Save all currently configured gates to UnifiedGateManager."""
        for det in VALID_DETECTORS:
            pts = self.temp_points[det]
            ref = self.temp_inside[det]
            if len(pts) == 2 and ref is not None:
                self.gate_manager.set_gate(det, pts, ref)
            else:
                self.gate_manager.set_gate(det, None, None)

        self.gate_manager.save_config()
        print(f"[GateCalibrator] Successfully saved all gate configurations to: {self.gate_manager.config_path}")

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        print("=" * 60)
        print(" Gate Calibration Tool")
        print(" Keys:")
        print("   0: Select Master Gate")
        print("   1: Select Footfall Gate")
        print("   2: Select Bag Exit Gate")
        print("   3: Select Demographics Gate")
        print("   R: Reset current active gate")
        print("   C: Clear all gates")
        print("   N: Read next frame (for video files)")
        print("   S: Save configurations")
        print("   Q / Esc: Save and exit")
        print("=" * 60)

        try:
            while True:
                canvas = self.current_frame.copy()
                self.draw_ui(canvas)
                cv2.imshow(self.window_name, canvas)

                key = cv2.waitKey(20) & 0xFF

                # Detector switching hotkeys
                if key == ord("0"):
                    self.active_detector = "master"
                elif key == ord("1"):
                    self.active_detector = "footfall"
                elif key == ord("2"):
                    self.active_detector = "bag_exit"
                elif key == ord("3"):
                    self.active_detector = "demographics"
                elif key == ord("r"):
                    # Reset active gate
                    self.temp_points[self.active_detector] = []
                    self.temp_inside[self.active_detector] = None
                    print(f"Reset {self.active_detector} gate.")
                elif key == ord("c"):
                    # Clear all gates
                    for det in VALID_DETECTORS:
                        self.temp_points[det] = []
                        self.temp_inside[det] = None
                    print("Cleared all gates.")
                elif key == ord("n"):
                    # Next frame
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.current_frame = frame
                elif key == ord("s"):
                    self.save()
                elif key in (ord("q"), 27):  # 'q' or Esc
                    self.save()
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate gate lines for retail video analytics.")
    parser.add_argument(
        "--source",
        default=VIDEO_SOURCE,
        help="Path to video file, camera index (e.g. 0), or RTSP URL",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    calibrator = GateCalibrator(args.source)
    calibrator.run()
