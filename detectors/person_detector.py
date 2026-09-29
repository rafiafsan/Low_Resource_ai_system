from ultralytics import YOLO
from config.settings import BAG_CLASS_ID, MODEL_PATH, TRACKER_CONFIG, PERSON_CLASS_ID, CONFIDENCE, IMAGE_SIZE
import torch



class PersonDetector:

    def __init__(self):

        if not torch.cuda.is_available():
           print("CUDA is not available. Running on CPU instead.")
           device = 'cpu'
        else:
           device = 'cuda'
           print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        self.model = YOLO(MODEL_PATH)


    #model calling optimization(returning the bounding box from single model)
    def detect(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            classes=[PERSON_CLASS_ID, BAG_CLASS_ID],
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )
        person_detections = []
        bag_detections = []

        if ( results and len(results) > 0 and results[0].boxes and results[0].boxes.id is not None ):
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                class_ids = results[0].boxes.cls.int().cpu().numpy()

                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x1, y1, x2, y2 = box
                    bbox = ( int(x1), int(y1), int(x2), int(y2) )

                    if class_id == PERSON_CLASS_ID:
                      bottom_center = ( int((x1+x2)/2), int(y1 + (y2-y1)*0.75) )
                      upper_center = ( int((x1+x2)/2), int(y1) )
                      person_detections.append({
                            "track_id": track_id,
                            "bbox": bbox,
                            "center": upper_center,
                            "bottom_center": bottom_center
                    }) 
                      
                    elif class_id == BAG_CLASS_ID:
                      midpoint = ( int((x1+x2)/2), int((y1+y2)/2) )
                      bag_detections.append({
                            "track_id": track_id,
                            "bbox": bbox,
                            "bag_midpoint": midpoint
                    }) 
        return person_detections, bag_detections              
                      

                      




    




    # def detect(self, frame):
    #     #this function is using for detecting persons for footfall
    #     results = self.model.track(
    #         frame,
    #         persist=True,
    #         tracker=TRACKER_CONFIG,
    #         classes=[PERSON_CLASS_ID],
    #         conf=CONFIDENCE,
    #         imgsz=IMAGE_SIZE,
    #         verbose=False,
    #     )
        
    #     detections = []
    #     if results and len(results) > 0 and results[0].boxes and results[0].boxes.id is not None:
    #         boxes = results[0].boxes.xyxy.cpu().numpy()
    #         track_ids = results[0].boxes.id.int().cpu().numpy()
            
    #         for box, track_id in zip(boxes, track_ids):
    #             x1, y1, x2, y2 = box
    #             bottom_center_x = int((x1 + x2) / 2)
    #             bottom_center_y = int(y1 + (y2 - y1) * 0.75)

    #             upper_center_x = int((x1 + x2) / 2)
    #             upper_center_y = int(y1)

    #             # lower_body_midpoint = ( int((x1 + x2) / 2), int(y2) )

    #             detections.append({
    #                 "track_id": track_id,
    #                 "bbox": (int(x1), int(y1), int(x2), int(y2)),
    #                 "center": (upper_center_x, upper_center_y),
    #                 "bottom_center" : (bottom_center_x, bottom_center_y),
    #                 # "lower_body_midpoint" : lower_body_midpoint
    #             })
    #     return detections



    # def detect_persons(self, frame):
    #         #this fucntion is using for detecting persons for bag detection purpose
    #         results = self.model.track(
    #             frame,
    #             persist=True,
    #             tracker=TRACKER_CONFIG,
    #             classes = [PERSON_CLASS_ID],
    #             conf=CONFIDENCE,
    #             imgsz=IMAGE_SIZE,
    #             verbose=False,
    #         )
            
    #         detections = []
    #         if results and len(results) > 0 and results[0].boxes and results[0].boxes.id is not None:
    #             boxes = results[0].boxes.xyxy.cpu().numpy()
    #             track_ids = results[0].boxes.id.int().cpu().numpy()
                
    #             for box, track_id in zip(boxes, track_ids):
    #                 x1, y1, x2, y2 = box
    #                 # Lower-body midpoint
    #                 lower_body_midpoint = ( int((x1 + x2) / 2), int(y2) )
                    
    #                 detections.append({
    #                     "track_id": track_id,
    #                     "bbox": (int(x1), int(y1), int(x2), int(y2)),
    #                     "person_lower_body_midpoint" : lower_body_midpoint
    #                 })
    #         return detections
