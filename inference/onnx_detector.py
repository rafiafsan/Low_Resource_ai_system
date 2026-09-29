"""Lightweight ONNX Runtime CPU YOLO Detector
=============================================
Optimized for low-resource Intel CPU machines (No GPU, 4 CPU cores, ~2GB RAM).
Runs YOLOv8 ONNX models using ONNX Runtime with strictly capped CPU threads.
Performs fast C++ accelerated NMS via cv2.dnn.NMSBoxes.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ONNXDetector:
    """High-performance CPU YOLOv8 detector using ONNX Runtime.

    Detects 'person' (COCO class 0) and bag classes (backpack=24, handbag=26).
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        cpu_threads: int = 2,
    ):
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime is required for ONNXDetector. Run: pip install onnxruntime")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.cpu_threads = cpu_threads

        # Configure ONNX Runtime session for low CPU / low RAM
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.cpu_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        print(f"[ONNXDetector] Loading model {model_path} with {self.cpu_threads} CPU threads...")
        t0 = time.time()
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.load_time_sec = time.time() - t0

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape  # e.g. [1, 3, 320, 320] or [1, 3, 640, 640]
        self.input_height = self.input_shape[2] if len(self.input_shape) >= 4 else 320
        self.input_width = self.input_shape[3] if len(self.input_shape) >= 4 else 320
        self.output_name = self.session.get_outputs()[0].name

        # Target class IDs (COCO): 0 = person, 24 = backpack, 26 = handbag, 28 = suitcase
        self.target_classes = {0: "person", 24: "bag", 26: "bag", 28: "bag"}

        print(
            f"[ONNXDetector] Model loaded in {self.load_time_sec:.2f}s! "
            f"Input size: {self.input_width}x{self.input_height}"
        )

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Preprocess OpenCV BGR frame into normalized NCHW float32 tensor."""
        h, w = frame.shape[:2]
        if w == self.input_width and h == self.input_height:
            resized = frame
            scale_x, scale_y = 1.0, 1.0
        else:
            resized = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
            scale_x = w / self.input_width
            scale_y = h / self.input_height

        # BGR -> RGB -> HWC to CHW -> normalize 0..1 -> float32 batch
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, scale_x, scale_y

    def detect(
        self, frame: np.ndarray, original_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[List[Dict], List[Dict], float]:
        """Execute detection pass and return (person_detections, bag_detections, latency_ms).

        Detections are scaled to the original frame coordinate space.
        """
        t0 = time.time()
        orig_h, orig_w = original_shape if original_shape else frame.shape[:2]

        # 1. Preprocess
        tensor, _, _ = self.preprocess(frame)

        # Scale model output coordinates directly to original target frame space
        sx = orig_w / float(self.input_width)
        sy = orig_h / float(self.input_height)

        # 2. Inference
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        # Output shape: [1, 84, N] -> transpose to [N, 84]
        preds = np.squeeze(outputs[0])
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T  # Shape: (N, 84)

        boxes_list = []
        confidences = []
        class_ids = []

        # Coordinate columns: 0,1,2,3 = cx, cy, w, h
        # Class probabilities: columns 4..83
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        # Filter by maximum score
        max_scores = np.max(class_scores, axis=1)
        valid_mask = max_scores >= self.conf_threshold

        valid_boxes = boxes_xywh[valid_mask]
        valid_scores = max_scores[valid_mask]
        valid_classes = np.argmax(class_scores[valid_mask], axis=1)

        for box, score, cid in zip(valid_boxes, valid_scores, valid_classes):
            if cid in self.target_classes:
                cx, cy, bw, bh = box
                x1 = int((cx - bw / 2.0) * sx)
                y1 = int((cy - bh / 2.0) * sy)
                w_box = int(bw * sx)
                h_box = int(bh * sy)
                boxes_list.append([x1, y1, w_box, h_box])
                confidences.append(float(score))
                class_ids.append(int(cid))

        # 3. Fast C++ NMS
        person_detections: List[Dict] = []
        bag_detections: List[Dict] = []

        if boxes_list:
            indices = cv2.dnn.NMSBoxes(boxes_list, confidences, self.conf_threshold, self.iou_threshold)
            if len(indices) > 0:
                for idx in indices.flatten():
                    x, y, w_box, h_box = boxes_list[idx]
                    cid = class_ids[idx]
                    score = confidences[idx]
                    x1 = max(0, min(orig_w - 1, x))
                    y1 = max(0, min(orig_h - 1, y))
                    x2 = max(0, min(orig_w - 1, x + w_box))
                    y2 = max(0, min(orig_h - 1, y + h_box))

                    bbox = (x1, y1, x2, y2)
                    cname = self.target_classes[cid]

                    if cname == "person":
                        upper_center = (int((x1 + x2) / 2), int(y1))
                        bottom_center = (int((x1 + x2) / 2), int(y1 + (y2 - y1) * 0.75))
                        lower_body_midpoint = (int((x1 + x2) / 2), int(y2))
                        person_detections.append({
                            "bbox": bbox,
                            "confidence": score,
                            "class_id": 0,
                            "class_name": "person",
                            "center": upper_center,
                            "bottom_center": bottom_center,
                            "person_lower_body_midpoint": lower_body_midpoint,
                        })
                    else:
                        midpoint = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        bag_detections.append({
                            "bbox": bbox,
                            "confidence": score,
                            "class_id": 26,
                            "class_name": "bag",
                            "bag_midpoint": midpoint,
                        })

        latency_ms = (time.time() - t0) * 1000.0
        return person_detections, bag_detections, latency_ms
