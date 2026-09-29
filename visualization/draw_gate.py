import cv2

def draw_gate(frame, gate_points, inside_reference):
    if not gate_points or len(gate_points) < 2:
        return
        
    p1, p2 = gate_points
    cv2.line(frame, p1, p2, (255, 0, 0), 2)
    
    if inside_reference:
        cv2.circle(frame, inside_reference, 5, (0, 0, 255), -1)
        cv2.putText(frame, "Inside", (inside_reference[0] + 10, inside_reference[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
