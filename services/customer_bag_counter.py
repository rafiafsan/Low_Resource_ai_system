import os
import cv2
from datetime import datetime
from detectors.person_detector import PersonDetector
from video.camera import Camera
from video.fps import FPSCounter
from visualization.dashboard import draw_dashboard_bag_associator
from detectors.bag_detector import BagDetector
from gate.mouse_handler import MouseHandler
from detectors.person_bag_associator import BagPersonAassociator
from detectors.person_bag_exit_validator import PersonWithBagExitValidator
from visualization.draw_gate import draw_gate
from gate.gate_manager_bagperson import GateManagerBag


class PersonBagCounter:
    def __init__(self, video_source):
        self.detector = PersonDetector()
        self.camera = Camera(video_source)
        self.fps_counter = FPSCounter()
        self.bag_detector = BagDetector()
        self.associator = BagPersonAassociator()
        self.exitvalidator = PersonWithBagExitValidator()
        self.gate_manager_bag = GateManagerBag()

    def run(self):
        #draw_gate
        cv2.namedWindow("Person With Bag")
        mouse_handler = MouseHandler(self.gate_manager_bag )
        cv2.setMouseCallback("Person With Bag", mouse_handler.handle_mouse)


        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
 

            #Step 1 : detect bags with person
            # detections = self.detector.detect_persons(frame)
            # bag_detections = self.bag_detector.detect_bags(frame)

            # #call the associate function here 
            # self.associator.associate(detections, bag_detections)
            # frame = self.associator.draw_confirmed_associations( frame, detections, bag_detections )
            # total = self.associator.get_count()
            # print ("Total: ", total)


            #Step 2 : Valdiate the person's exit if with bag or not
            # person_detection = self.detector.detect(frame)
            # bag_detections = self.bag_detector.detect_bags(frame)

            person_detection, bag_detections = self.detector.detect(frame) #optimized model calling
            
            associations_bag_exit = self.associator.associate_person_bag_exit(person_detection , bag_detections)
            self.exitvalidator.update_associations(associations_bag_exit, person_detection)
            exit_events = self.exitvalidator.update(person_detection)
            frame = self.associator.draw_confirmed_associations_bag_exit( frame, person_detection, bag_detections )
            total =  self.exitvalidator.get_count()
            print("Total:", total)

            for event in exit_events:
                print("Exit with bag:", event)
            

            # Darw the detections bounding box
            for det in person_detection:
                x1, y1, x2, y2 = det["bbox"]
                track_id = det["track_id"]
                
                cv2.rectangle(frame,(x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}",(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            for det in bag_detections:
                x1, y1, x2, y2 = det["bbox"]
                track_id = det["track_id"]
                            
                cv2.rectangle(frame,(x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"ID: {track_id}",(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    

            fps = self.fps_counter.update()

            #save the gate points
            draw_gate(frame, self.gate_manager_bag.gate_points, self.gate_manager_bag.inside_reference) 
            for pt in mouse_handler.points:
                cv2.circle(frame, pt, 5, (0, 255, 255), -1)

            draw_dashboard_bag_associator(                                             
                            total,
                            frame,
                            fps                                                            
                        )         

            cv2.imshow("Person With Bag", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()
