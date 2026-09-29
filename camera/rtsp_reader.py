"""Low-Latency RTSP Reader & Latest-Frame Buffer
================================================
Designed for low-resource edge devices (Intel CPU, ~2GB RAM, Windows).
Decouples camera ingestion from AI inference and GUI preview.

Key Characteristics:
- Strictly bounded: holds ONLY the latest frame slot (no frame accumulation).
- CAP_PROP_BUFFERSIZE = 1 where supported.
- Automatic RTSP reconnection without application crash.
- Efficient memory usage (avoids unnecessary array copies).
- Supports live RTSP streams, webcams, and local evaluation video files.
"""

import os
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

# Force reliable TCP transport for RTSP streams to avoid packet drops / tearing
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class RTSPReader:
    """Thread-safe, non-blocking camera capture using a single-slot latest-frame buffer."""

    def __init__(
        self,
        source: Union[int, str],
        width: Optional[int] = None,
        height: Optional[int] = None,
        reconnect_interval_sec: float = 2.0,
        loop_video: bool = True,
    ):
        self.raw_source = source
        self.source = int(source) if str(source).isdigit() else str(source)
        self.is_live = isinstance(self.source, int) or str(self.source).lower().startswith(
            ("rtsp://", "rtsps://", "http://", "https://")
        )
        self.target_width = width
        self.target_height = height
        self.reconnect_interval = reconnect_interval_sec
        self.loop_video = loop_video

        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._stopped = False
        self._is_connected = False
        self._reconnect_count = 0

        # Latest frame slot
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id: int = 0
        self._latest_timestamp: float = 0.0

        # Performance metrics
        self._fps_counter_time = time.time()
        self._fps_frame_count = 0
        self.capture_fps: float = 0.0

        # Start ingestion
        self._thread: Optional[threading.Thread] = None
        self._start_capture()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    def _open_stream(self) -> bool:
        """Open or reopen video stream with optimal low-latency flags."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if isinstance(self.source, str) and not self.source.strip():
            print("[RTSPReader] Warning: Source URL/path is empty.")
            return False

        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                print(f"[RTSPReader] Could not open video source: {self.source}")
                return False

            # Request buffer size 1 to minimize internal OS buffering lag
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if self.target_width and self.target_height and isinstance(self.source, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

            v_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            if v_fps <= 0 or v_fps > 60:
                v_fps = 30.0
            self._target_interval = 0.033

            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                with self._lock:
                    self._latest_frame = frame
                    self._latest_frame_id += 1
                    self._latest_timestamp = time.time()
                    self._is_connected = True
                return True
            else:
                self._is_connected = False
                return False
        except Exception as e:
            print(f"[RTSPReader] Exception during capture open: {e}")
            self._is_connected = False
            return False

    def _start_capture(self) -> None:
        """Initiate background capture thread."""
        self._open_stream()
        self._thread = threading.Thread(target=self._capture_loop, name="RTSPReader-Thread", daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        """Dedicated background capture thread: continuously drains camera buffer."""
        fail_count = 0

        while not self._stopped:
            if self.cap is None or not self.cap.isOpened() or not self._is_connected:
                time.sleep(self.reconnect_interval)
                if not self._stopped:
                    self._reconnect_count += 1
                    print(f"[RTSPReader] Attempting reconnect #{self._reconnect_count} to: {self.source}")
                    if self._open_stream():
                        print(f"[RTSPReader] Reconnect successful to: {self.source}")
                        fail_count = 0
                continue

            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                now = time.time()
                fail_count = 0

                # Slot replacement: old frame is automatically released
                with self._lock:
                    self._latest_frame = frame
                    self._latest_frame_id += 1
                    self._latest_timestamp = now
                    self._is_connected = True

                # FPS calculation
                self._fps_frame_count += 1
                if now - self._fps_counter_time >= 1.0:
                    self.capture_fps = self._fps_frame_count / (now - self._fps_counter_time)
                    self._fps_counter_time = now
                    self._fps_frame_count = 0

                # Slight pacing for local file reading to simulate realistic real-time camera rate
                if not self.is_live:
                    time.sleep(getattr(self, "_target_interval", 0.05))
            else:
                fail_count += 1
                if not self.is_live and self.loop_video and self.cap is not None:
                    # Loop video file if reached EOF
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue

                if fail_count > 5:
                    with self._lock:
                        self._is_connected = False
                    print(f"[RTSPReader] Stream read stalled or lost. Scheduling reconnect...")
                    time.sleep(self.reconnect_interval)
                    if not self._stopped:
                        self._open_stream()

    @property
    def frame_shape(self) -> Optional[Tuple[int, int]]:
        """Return (height, width) of the active camera/video frame."""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.shape[:2]
            return None

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray], int, float]:
        """Retrieve the newest available frame. Thread-safe and non-blocking.

        Returns:
            (ret, frame, frame_id, timestamp)
        """
        with self._lock:
            if not self._is_connected or self._latest_frame is None:
                return False, None, 0, 0.0
            return True, self._latest_frame, self._latest_frame_id, self._latest_timestamp

    def get_display_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return a copy of the latest frame specifically for GUI display.

        Returns:
            (ret, frame_copy)
        """
        with self._lock:
            if not self._is_connected or self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def get_inference_frame(
        self, target_width: int, target_height: int
    ) -> Tuple[bool, Optional[np.ndarray], Tuple[float, float], int]:
        """Downscale the latest frame for AI inference.

        Returns:
            (ret, resized_frame, (scale_x, scale_y), frame_id)
        """
        with self._lock:
            if not self._is_connected or self._latest_frame is None:
                return False, None, (1.0, 1.0), 0
            src = self._latest_frame
            fid = self._latest_frame_id

        h, w = src.shape[:2]
        if w == target_width and h == target_height:
            return True, src.copy(), (1.0, 1.0), fid

        resized = cv2.resize(src, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        scale_x = w / target_width
        scale_y = h / target_height
        return True, resized, (scale_x, scale_y), fid

    def release(self) -> None:
        """Cleanly shutdown capture thread and release hardware resources."""
        self._stopped = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self._latest_frame = None
        self._is_connected = False
        print("[RTSPReader] Released capture resources cleanly.")
