import cv2
from detectors.person_detector import PersonDetector
from video.camera import Camera
from video.fps import FPSCounter
from tracking.track_manager import TrackManager
from gate.event_manager import EventManager
from gate.gate_manager import GateManager
from visualization.draw_gate import draw_gate
from visualization.draw_track import draw_tracks
from visualization.dashboard import draw_dashboard
from gate.mouse_handler import MouseHandler

class CustomerCounter:
    def __init__(self, video_source):
        self.detector = PersonDetector()
        self.camera = Camera(video_source)
        self.fps_counter = FPSCounter()
        self.track_manager = TrackManager()
        self.gate_manager = GateManager()
        self.event_manager = EventManager()

    def run(self):
        cv2.namedWindow("MCP Tracking")
        mouse_handler = MouseHandler(self.gate_manager)
        cv2.setMouseCallback("MCP Tracking", mouse_handler.handle_mouse)

        while True:
            ret, frame = self.camera.read()
            if not ret:
                break

            person_detections, bag_detections = self.detector.detect(frame)
            #passing the frame to the track manager for updating the track history of each detections
            self.track_manager.update(person_detections)

            for det in person_detections:
                self.event_manager.process(
                    det["track_id"], 
                    det["center"], 
                    self.gate_manager.gate_points, 
                    self.gate_manager.inside_reference
                )

            self.event_manager.validate_pending()       # NEW

            fps = self.fps_counter.update()

            draw_gate(frame, self.gate_manager.gate_points, self.gate_manager.inside_reference)
            
            for pt in mouse_handler.points:
                cv2.circle(frame, pt, 5, (0, 255, 255), -1)

            draw_tracks(frame, person_detections, self.track_manager)
            pending_count = (
                len(self.event_manager.pending_entries)
                + len(self.event_manager.pending_exits)
            )
            draw_dashboard(
                frame,
                fps,
                self.event_manager.entry_count,
                self.event_manager.exit_count,
                pending_count,
                self.event_manager.validated_entry_count,
                self.event_manager.validated_exit_count,
            )      

            cv2.imshow("MCP Tracking", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                mouse_handler.reset()

        self.camera.release()
        cv2.destroyAllWindows()
