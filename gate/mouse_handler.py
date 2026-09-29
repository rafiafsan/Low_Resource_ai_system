import cv2

class MouseHandler:
    def __init__(self, gate_manager):
        self.gate_manager = gate_manager
        self.points = []

    def handle_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                if len(self.points) == 2:
                    self.gate_manager.gate_points = [self.points[0], self.points[1]]
            elif len(self.points) == 2:
                self.points.append((x, y))
                self.gate_manager.inside_reference = (x, y)
                self.gate_manager.save_config()

    def reset(self):
        self.points = []
        self.gate_manager.gate_points = []
        self.gate_manager.inside_reference = None
        self.gate_manager.save_config()
