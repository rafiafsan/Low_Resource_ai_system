import os
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np


class ThreadedRTSPStream:
    """High-performance frame grabber for RTSP streams, webcams, and video files.

    For live streams (RTSP / USB webcams), a background thread continuously drains
    the OpenCV buffer, ensuring that read() always returns the freshest frame without
    accumulating lag. If the stream drops, it attempts automatic reconnection.

    For local video files, it reads sequentially to allow complete evaluation.
    """

    def __init__(self, source: Union[int, str], reconnect_interval: float = 2.0):
        self.raw_source = source
        self.source = int(source) if str(source).isdigit() else str(source)
        self.is_live = isinstance(self.source, int) or str(self.source).lower().startswith(
            ("rtsp://", "rtsps://", "http://", "https://")
        )
        self.reconnect_interval = reconnect_interval

        self.cap: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        self.stopped = False
        self.frame: Optional[np.ndarray] = None
        self.ret = False
        self.thread: Optional[threading.Thread] = None

        self._open_capture()

        if self.is_live:
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()

    def _open_capture(self) -> bool:
        """Open or reopen the video capture."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        if self.is_live and isinstance(self.source, str):
            # Set transport to TCP for RTSP to reduce packet loss / tearing if supported
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"[ThreadedRTSPStream] Warning: Could not open source: {self.source}")
            return False

        # Read first frame to prime the buffer
        ret, frame = self.cap.read()
        with self.lock:
            self.ret = ret
            self.frame = frame
        return ret

    def _update_loop(self) -> None:
        """Background thread loop for draining live camera buffers."""
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(self.reconnect_interval)
                if not self.stopped:
                    print(f"[ThreadedRTSPStream] Attempting reconnect to {self.source}...")
                    self._open_capture()
                continue

            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ret = True
                    self.frame = frame
            else:
                # Stream stalled or disconnected
                with self.lock:
                    self.ret = False
                time.sleep(self.reconnect_interval)
                if not self.stopped:
                    print(f"[ThreadedRTSPStream] Stream read failed. Reconnecting to {self.source}...")
                    self._open_capture()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the newest frame. Thread-safe and non-blocking for live streams."""
        if self.is_live:
            with self.lock:
                if not self.ret or self.frame is None:
                    return False, None
                return True, self.frame.copy()
        else:
            # Sequential file read
            if self.cap is None or not self.cap.isOpened():
                return False, None
            ret, frame = self.cap.read()
            return ret, frame

    def release(self) -> None:
        """Stop background worker and release capture resources."""
        self.stopped = True
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
