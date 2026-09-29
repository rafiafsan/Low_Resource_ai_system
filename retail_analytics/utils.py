"""
utils.py — Shared utility functions for the retail analytics system.

Provides image quality metrics (sharpness, brightness, face visibility),
bounding box helpers, and logging/display utilities.
"""

import cv2
import numpy as np
import logging
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(name: str = "RetailAnalytics", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a named logger with coloured console output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Image quality metrics
# ---------------------------------------------------------------------------

def compute_sharpness(image: np.ndarray) -> float:
    """
    Measure image sharpness using the variance of the Laplacian operator.
    Higher value → sharper image.  Returns a value in [0, 1] after clamping.
    """
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalise: empirically 500 is considered very sharp for CCTV footage
    return float(min(lap_var / 500.0, 1.0))


def compute_brightness(image: np.ndarray) -> float:
    """
    Estimate brightness as the mean of the Value channel in HSV space.
    Returns a value in [0, 1].
    """
    if image is None or image.size == 0:
        return 0.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if len(image.shape) == 3 else image
    v_channel = hsv[:, :, 2] if len(hsv.shape) == 3 else hsv
    mean_v = float(np.mean(v_channel))
    return mean_v / 255.0


def compute_person_size_score(bbox: tuple, frame_shape: tuple) -> float:
    """
    Compute a normalised size score for a person bounding box.
    Score = (bbox_area / frame_area), clamped to [0, 1].

    Args:
        bbox:        (x1, y1, x2, y2) bounding box in pixels.
        frame_shape: (height, width[, channels]) of the full frame.
    """
    x1, y1, x2, y2 = bbox
    bbox_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
    frame_area = frame_shape[0] * frame_shape[1]
    if frame_area == 0:
        return 0.0
    return float(min(bbox_area / frame_area, 1.0))


def compute_visibility_score(bbox: tuple, frame_shape: tuple) -> float:
    """
    Estimate how much of a person's bounding box is inside the frame.
    Full visibility = 1.0, partially cut off = lower score.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]

    # Clip to frame boundaries
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(w, x2), min(h, y2)

    visible_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    return float(min(visible_area / bbox_area, 1.0))


def compute_face_visibility(crop: np.ndarray) -> float:
    """
    Estimate face visibility in a person crop using a Haar face detector.
    Returns 1.0 if a face is detected, 0.0 otherwise.
    Gracefully returns 0.5 if the detector fails to load.
    """
    if crop is None or crop.size == 0:
        return 0.0
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        return 1.0 if len(faces) > 0 else 0.0
    except Exception:
        return 0.5  # neutral score if detection fails


# ---------------------------------------------------------------------------
# Bounding box helpers
# ---------------------------------------------------------------------------

def expand_bbox(bbox: tuple, frame_shape: tuple,
                pad_h: float = 0.25, pad_v: float = 0.15) -> tuple:
    """
    Expand a bounding box by horizontal and vertical padding percentages,
    clamped to the frame boundaries.

    Args:
        bbox:        (x1, y1, x2, y2) original bbox.
        frame_shape: (height, width[, channels]).
        pad_h:       Horizontal padding fraction (default 25 %).
        pad_v:       Vertical padding fraction (default 15 %).

    Returns:
        (x1, y1, x2, y2) expanded and clamped bbox.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]
    bw, bh = x2 - x1, y2 - y1

    nx1 = int(max(0,     x1 - bw * pad_h))
    ny1 = int(max(0,     y1 - bh * pad_v))
    nx2 = int(min(w - 1, x2 + bw * pad_h))
    ny2 = int(min(h - 1, y2 + bh * pad_v))
    return (nx1, ny1, nx2, ny2)


def crop_from_frame(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    """Return a cropped region from a frame using an (x1, y1, x2, y2) bbox."""
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2].copy()


# ---------------------------------------------------------------------------
# Overlay / drawing helpers
# ---------------------------------------------------------------------------

def draw_person_box(
    frame: np.ndarray,
    bbox: tuple,
    track_id: int,
    age_group: str = "",
    gender: str = "",
    color: tuple = (0, 200, 255),
    is_target: bool = False,
) -> np.ndarray:
    """
    Draw a bounding box with tracking metadata overlay on a frame (in-place copy).

    The label block shows:
        ID: <track_id>
        <age_group>    (after VLM analysis)
        <gender>       (after VLM analysis)
    """
    frame = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Box colour: bright cyan for normal, bright yellow for TARGET
    box_color = (0, 255, 255) if is_target else color
    thickness = 3 if is_target else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

    # Build label lines
    lines = [f"ID:{track_id}"]
    if age_group:
        lines.append(age_group)
    if gender:
        lines.append(gender)
    if is_target:
        lines.append("TARGET PERSON")

    # Draw filled label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thickness = 1
    line_height = 20
    padding = 4

    label_w = max(
        cv2.getTextSize(ln, font, font_scale, font_thickness)[0][0]
        for ln in lines
    ) + padding * 2
    label_h = line_height * len(lines) + padding

    # Place label above box, clamp to frame
    lx1 = x1
    ly1 = max(0, y1 - label_h)
    lx2 = x1 + label_w
    ly2 = y1

    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), box_color, -1)

    for i, line in enumerate(lines):
        ty = ly1 + padding + line_height * i + line_height - 4
        cv2.putText(
            frame, line,
            (lx1 + padding, ty),
            font, font_scale,
            (0, 0, 0),  # black text on coloured background
            font_thickness, cv2.LINE_AA,
        )

    return frame


def draw_hud(frame: np.ndarray, fps: float, track_count: int,
             analyzed_count: int, frame_no: int) -> np.ndarray:
    """Draw a heads-up display overlay with system metrics."""
    frame = frame.copy()
    h, w = frame.shape[:2]

    # Semi-transparent dark bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    info_str = (
        f"Frame: {frame_no}  |  FPS: {fps:.1f}  |  "
        f"Tracks: {track_count}  |  Analyzed: {analyzed_count}"
    )
    cv2.putText(
        frame, info_str,
        (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (200, 255, 200), 1, cv2.LINE_AA,
    )
    return frame


def save_debug_image(image: np.ndarray, path: str) -> bool:
    """Save an image to disk for debugging; returns True on success."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, image)
        return True
    except Exception as e:
        logging.getLogger("RetailAnalytics").warning(f"Could not save debug image to {path}: {e}")
        return False


def ensure_dir(path: str) -> Path:
    """Create directory (and parents) if it does not exist, then return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp_str() -> str:
    """Return a filesystem-safe timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
