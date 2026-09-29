import cv2

class Camera:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()