"""Real-Time Performance Monitor & System Safeguards
===================================================
Tracks:
- Capture FPS, Display FPS, AI FPS
- Inference, Tracking, Gate, and Bag latencies (ms)
- Process RAM usage (MB) and Host CPU utilization (%)
- Database and Demographic queue sizes
- Automatic performance protection against memory and CPU saturation
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np
import psutil


class PerformanceMonitor:
    """Monitors system performance metrics and renders a lightweight edge HUD."""

    def __init__(
        self,
        ram_limit_mb: float = 1750.0,
        cpu_limit_percent: float = 90.0,
    ):
        self.ram_limit_mb = ram_limit_mb
        self.cpu_limit_percent = cpu_limit_percent
        self.process = psutil.Process()

        # Measured metrics
        self.capture_fps: float = 0.0
        self.display_fps: float = 0.0
        self.ai_fps: float = 0.0
        self.inference_ms: float = 0.0
        self.tracking_ms: float = 0.0
        self.gate_ms: float = 0.0
        self.bag_ms: float = 0.0
        self.demographic_ms: float = 0.0

        self.db_queue_size: int = 0
        self.demographic_queue_size: int = 0
        self.rtsp_status: str = "CONNECTING"

        # Hardware metrics
        self.ram_mb: float = 0.0
        self.cpu_percent: float = 0.0
        self._last_hw_check = 0.0

        # Display FPS counter
        self._disp_count = 0
        self._disp_start = time.time()

        # Safeguard flags
        self.ram_critical = False
        self.cpu_critical = False

    def update_display_fps(self) -> float:
        """Call on each display frame presentation."""
        now = time.time()
        self._disp_count += 1
        elapsed = now - self._disp_start
        if elapsed >= 1.0:
            self.display_fps = self._disp_count / elapsed
            self._disp_start = now
            self._disp_count = 0
        return self.display_fps

    def update_hardware_metrics(self) -> Tuple[bool, bool]:
        """Periodically sample process RAM and CPU percentage (every 1.0s)."""
        now = time.time()
        if now - self._last_hw_check >= 1.0:
            self._last_hw_check = now
            try:
                self.ram_mb = self.process.memory_info().rss / (1024.0 * 1024.0)
                self.cpu_percent = psutil.cpu_percent(interval=None)
            except Exception:
                pass

            # Safeguards
            self.ram_critical = self.ram_mb > self.ram_limit_mb
            self.cpu_critical = self.cpu_percent > self.cpu_limit_percent

        return self.ram_critical, self.cpu_critical

    def draw_hud(
        self,
        frame: np.ndarray,
        entry_count: int,
        exit_count: int,
        bag_exit_count: int,
        active_profiling_count: int = 0,
        valid_entry_count: int = 0,
        valid_exit_count: int = 0,
        pending_count: int = 0,
    ) -> None:
        """Render a sleek, lightweight semi-transparent HUD overlay on frame."""
        self.update_display_fps()
        self.update_hardware_metrics()

        hud_w, hud_h = 320, 275
        hud_x, hud_y = 12, 12

        # Draw sleek dark background rectangle
        h, w = frame.shape[:2]
        if hud_y + hud_h <= h and hud_x + hud_w <= w:
            sub = frame[hud_y : hud_y + hud_h, hud_x : hud_x + hud_w]
            overlay = np.zeros_like(sub)
            cv2.addWeighted(sub, 0.20, overlay, 0.80, 0, sub)
            frame[hud_y : hud_y + hud_h, hud_x : hud_x + hud_w] = sub
            cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (60, 60, 60), 1)

        # Header Title
        cv2.putText(
            frame,
            "RETAIL AI ANALYTICS",
            (hud_x + 12, hud_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        status_color = (50, 255, 50) if self.rtsp_status == "ONLINE" else (0, 0, 255)
        cv2.putText(
            frame,
            f"[{self.rtsp_status}]",
            (hud_x + 225, hud_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            status_color,
            1,
            cv2.LINE_AA,
        )
        cv2.line(frame, (hud_x, hud_y + 28), (hud_x + hud_w, hud_y + 28), (70, 70, 70), 1)

        # Footfall & Bag Metrics
        y = hud_y + 48
        cv2.putText(frame, "ENTRIES (IN):", (hud_x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)
        cv2.putText(frame, str(entry_count), (hud_x + 230, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (50, 255, 50), 2)

        y += 24
        cv2.putText(frame, "EXITS (OUT):", (hud_x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)
        cv2.putText(frame, str(exit_count), (hud_x + 230, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 160, 255), 2)

        y += 22
        occupancy = max(0, entry_count - exit_count)
        cv2.putText(frame, "OCCUPANCY (INSIDE):", (hud_x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 230, 180), 1)
        cv2.putText(frame, str(occupancy), (hud_x + 230, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 230, 180), 2)

        y += 24
        cv2.putText(frame, "BAG EXITS:", (hud_x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)
        cv2.putText(frame, str(bag_exit_count), (hud_x + 230, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 255), 2)

        y += 22
        cv2.putText(
            frame,
            f"DEMO PROFILING: {active_profiling_count}",
            (hud_x + 12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (170, 170, 170),
            1,
        )

        cv2.line(frame, (hud_x, y + 8), (hud_x + hud_w, y + 8), (70, 70, 70), 1)

        # Telemetry & Diagnostics
        y += 28
        perf_line1 = f"CAM: {self.capture_fps:.1f} FPS | AI: {self.ai_fps:.1f} FPS"
        cv2.putText(frame, perf_line1, (hud_x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1)

        y += 20
        ram_color = (0, 0, 255) if self.ram_critical else (220, 220, 220)
        perf_line2 = f"INF: {self.inference_ms:.0f}ms | RAM: {self.ram_mb:.0f}MB | CPU: {self.cpu_percent:.0f}%"
        cv2.putText(frame, perf_line2, (hud_x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, ram_color, 1)

        y += 20
        perf_line3 = f"QUEUES: DB={self.db_queue_size} | DEMO={self.demographic_queue_size}"
        cv2.putText(frame, perf_line3, (hud_x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
