import time

class FPSCounter:
    def __init__(self):
        self.previous = time.time()

    def update(self):
        current = time.time()
        fps = 1 / (current - self.previous)
        self.previous = current
        return fps