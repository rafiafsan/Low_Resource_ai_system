import cv2

# def draw_dashboard(frame, fps, entry_count, exit_count):
#     cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#     cv2.putText(frame, f"ENTRY: {entry_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.putText(frame, f"EXIT: {exit_count}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
import cv2


def draw_dashboard(
    frame,
    fps,
    entry_count,
    exit_count,
    pending_count=0,
    validated_entry_count=0,                         # NEW
    validated_exit_count=0                            # NEW
):
    
    # Dashboard background
    cv2.rectangle(
        frame,
        (10, 10),
        (340, 260),                                  # NEW
        (40, 40, 40),
        -1
    )

    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # ENTRY
    cv2.putText(
        frame,
        f"ENTRY: {entry_count}",
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # EXIT
    cv2.putText(
        frame,
        f"EXIT: {exit_count}",
        (25, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    # Validated ENTRY
    cv2.putText(
        frame,
        f"VALID ENTRY: {validated_entry_count}",       # NEW
        (25, 160),                                     # NEW
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,                                          # NEW
        (0, 255, 0),
        2
    )

    # Validated EXIT
    cv2.putText(
        frame,
        f"VALID EXIT: {validated_exit_count}",         # NEW
        (25, 195),                                     # NEW
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,                                          # NEW
        (0, 0, 255),
        2
    )

    # Pending validation
    cv2.putText(
        frame,
        f"PENDING: {pending_count}",
        (25, 230),                                     # NEW
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

# def draw_dashboard_bag_detector(frame, fps):
#         cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def draw_dashboard_bag_associator(total, frame, fps):
      cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
      cv2.putText(frame, f"Persons left with Shopping Bag: {total}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

