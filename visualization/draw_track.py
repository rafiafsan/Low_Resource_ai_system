import cv2

def draw_tracks(frame, detections, track_manager):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        track_id = det["track_id"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
