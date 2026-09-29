"""Adaptive Inference Scheduler
==============================
Maintains consistent AI processing rate without accumulating latency.
Monitors inference execution time and CPU load, dynamically pacing inference passes.
"""

import time
import psutil
from typing import Optional


class InferenceScheduler:
    """Regulates inference invocations to meet a target FPS and prevent CPU overload."""

    def __init__(
        self,
        target_fps: float = 4.0,
        min_fps: float = 2.0,
        max_fps: float = 6.0,
        cpu_limit_percent: float = 90.0,
    ):
        self.target_fps = target_fps
        self.current_target_fps = target_fps
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.cpu_limit_percent = cpu_limit_percent

        self.last_inference_time = 0.0
        self.last_latency_ms = 0.0
        self.measured_ai_fps = 0.0
        self._fps_frame_count = 0
        self._fps_window_start = time.time()
        self._last_cpu_check = time.time()
        self.current_cpu_percent = 0.0

    def should_process_next_frame(self) -> bool:
        """Check if enough time has passed to trigger the next inference pass."""
        now = time.time()
        interval = 1.0 / max(1.0, self.current_target_fps)
        return (now - self.last_inference_time) >= interval

    def record_inference_completed(self, latency_ms: float) -> None:
        """Record completed inference pass and adjust pacing dynamically."""
        now = time.time()
        self.last_inference_time = now
        self.last_latency_ms = latency_ms

        # AI FPS counter
        self._fps_frame_count += 1
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self.measured_ai_fps = self._fps_frame_count / elapsed
            self._fps_window_start = now
            self._fps_frame_count = 0

        # Periodic CPU and latency adaptive backoff (every 2 seconds)
        if now - self._last_cpu_check >= 2.0:
            self._last_cpu_check = now
            try:
                self.current_cpu_percent = psutil.cpu_percent(interval=None)
            except Exception:
                self.current_cpu_percent = 50.0

            # If inference latency is very high or CPU is maxed, throttle AI FPS
            if self.current_cpu_percent > self.cpu_limit_percent or latency_ms > 250.0:
                self.current_target_fps = max(self.min_fps, self.current_target_fps - 0.5)
            elif self.current_cpu_percent < (self.cpu_limit_percent - 20) and latency_ms < 100.0:
                self.current_target_fps = min(self.target_fps, self.current_target_fps + 0.5)
