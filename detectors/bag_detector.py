from ultralytics import YOLO
from config.settings import BAG_CLASS_ID, MODEL_PATH, TRACKER_CONFIG, CONFIDENCE, IMAGE_SIZE
import torch

class BagDetector:

    def __init__(self):

        if not torch.cuda.is_available():
           print("CUDA is not available. Running on CPU instead.")
           device = 'cpu'
        else:
           device = 'cuda'
           print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        self.model = YOLO(MODEL_PATH)

    def detect_bags(self, frame):
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker=TRACKER_CONFIG,
                    classes = [BAG_CLASS_ID],
                    conf=CONFIDENCE,
                    imgsz=IMAGE_SIZE,
                    verbose=False,
                )
                
                detections = []
                if results and len(results) > 0 and results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    
                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = box

                        midpoint = (
                             int((x1+x2) /2),
                             int((y1+y2) /2),

                        )

                        detections.append({
                            "track_id": track_id,
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "bag_midpoint" : midpoint
                       
                        })
                return detections