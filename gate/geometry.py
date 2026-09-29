import numpy as np
from config.settings import GATE_BAND_PIXELS

def calculate_signed_distance(point, line_start, line_end):
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    dx = x2 - x1
    dy = y2 - y1
    denominator = np.sqrt(dx * dx + dy * dy)
    if denominator == 0:
        return 0
    value = dx * (y - y1) - dy * (x - x1)
    return value / denominator

def get_side(point, line_start, line_end, inside_reference):
    current_distance = calculate_signed_distance(point, line_start, line_end)
    inside_distance = calculate_signed_distance(inside_reference, line_start, line_end)

    if abs(current_distance) < GATE_BAND_PIXELS:
        return "GATE"
    
    if inside_distance > 0:
        return "INSIDE" if current_distance > 0 else "OUTSIDE"
    
    return "INSIDE" if current_distance < 0 else "OUTSIDE"
