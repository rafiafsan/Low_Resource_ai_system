"""PyTorch YOLO Detector with Low-CPU Optimization
=================================================
Optimized PyTorch fallback using torch.inference_mode() and pinned thread count.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class TorchDetector:
    """Ultralytics PyTorch detector configured for constrained CPU execution."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.40,
        cpu_threads: int = 2,
        imgsz: int = 384,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.cpu_threads = cpu_threads
        self.imgsz = imgsz

        # Limit PyTorch CPU threads
        torch.set_num_threads(self.cpu_threads)

        print(f"[TorchDetector] Loading PyTorch model {model_path} with {self.cpu_threads} CPU threads...")
        t0 = time.time()
        self.model = YOLO(model_path)
        self.load_time_sec = time.time() - t0

        names = self.model.names
        self.person_cid = 0
        self.bag_cids = [cid for cid, name in names.items() if name in ("bag", "backpack", "handbag")]
        if not self.bag_cids:
            self.bag_cids = [24, 26]

        self.classes_to_detect = [0] + self.bag_cids
        print(f"[TorchDetector] Model loaded in {self.load_time_sec:.2f}s! Target classes: {self.classes_to_detect}")

    def detect(
        self, frame: np.ndarray, original_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[List[Dict], List[Dict], float]:
        """Run PyTorch YOLO detection pass under torch.inference_mode()."""
        t0 = time.time()
        orig_h, orig_w = original_shape if original_shape else frame.shape[:2]
        cur_h, cur_w = frame.shape[:2]
        sx = orig_w / cur_w
        sy = orig_h / cur_h

        with torch.inference_mode():
            results = self.model.predict(
                frame,
                classes=self.classes_to_detect,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                verbose=False,
                device="cpu",
            )[0]

        person_detections: List[Dict] = []
        bag_detections: List[Dict] = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.int().cpu().numpy()

            for box, score, cid in zip(boxes, confidences, class_ids):
                x1 = int(box[0] * sx)
                y1 = int(box[1] * sy)
                x2 = int(box[2] * sx)
                y2 = int(box[3] * sy)
                bbox = (x1, y1, x2, y2)

                if cid == self.person_cid:
                    upper_center = (int((x1 + x2) / 2), int(y1))
                    bottom_center = (int((x1 + x2) / 2), int(y1 + (y2 - y1) * 0.75))
                    lower_body_midpoint = (int((x1 + x2) / 2), int(y2))
                    person_detections.append({
                        "bbox": bbox,
                        "confidence": float(score),
                        "class_id": 0,
                        "class_name": "person",
                        "center": upper_center,
                        "bottom_center": bottom_center,
                        "person_lower_body_midpoint": lower_body_midpoint,
                    })
                elif cid in self.bag_cids:
                    midpoint = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    bag_detections.append({
                        "bbox": bbox,
                        "confidence": float(score),
                        "class_id": 26,
                        "class_name": "bag",
                        "bag_midpoint": midpoint,
                    })

        latency_ms = (time.time() - t0) * 1000.0
        return person_detections, bag_detections, latency_ms
