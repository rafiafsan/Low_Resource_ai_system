"""Detector Factory for Edge Inference
====================================
Instantiates either ONNXDetector or TorchDetector based on configuration.
Ensures single-instance model lifecycle.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union

from .onnx_detector import ONNXDetector, ONNX_AVAILABLE
from .torch_detector import TorchDetector


def create_detector(config: Dict[str, Any]):
    """Instantiate the configured primary detector."""
    inf_cfg = config.get("inference", {})
    model_path = inf_cfg.get("model", "models/yolov8m.onnx")
    fallback_model = inf_cfg.get("fallback_model", "models/yolov8n.pt")
    conf_thresh = float(inf_cfg.get("confidence", 0.40))
    cpu_threads = int(inf_cfg.get("cpu_threads", 2))
    backend = inf_cfg.get("backend", "onnx").lower()

    # Determine if ONNX can be used
    if backend == "onnx" and model_path.endswith(".onnx") and os.path.exists(model_path) and ONNX_AVAILABLE:
        try:
            return ONNXDetector(
                model_path=model_path,
                conf_threshold=conf_thresh,
                cpu_threads=cpu_threads,
            )
        except Exception as e:
            print(f"[create_detector] Warning: Failed to initialize ONNX detector ({e}). Falling back to Torch...")

    # PyTorch Fallback
    target_pt = model_path if model_path.endswith(".pt") and os.path.exists(model_path) else fallback_model
    if not os.path.exists(target_pt):
        # Last resort check
        for candidate in ("models/yolov8n.pt", "models/yolov8m.pt", "models/yolov8l.pt"):
            if os.path.exists(candidate):
                target_pt = candidate
                break

    imgsz = int(inf_cfg.get("imgsz", 384))
    return TorchDetector(
        model_path=target_pt,
        conf_threshold=conf_thresh,
        cpu_threads=cpu_threads,
        imgsz=imgsz,
    )
