"""Visualization utilities for annotated video output."""
import cv2


def draw_annotations(frame, detections: list, show_labels: bool = True) -> None:
    """Draw bounding boxes and labels on frame in-place.

    Each detection dict:
        x1, y1, x2, y2: int — bounding box coordinates
        track_id: int — student ID
        score: int — attention score 0-100
        reasons: list[str] — behavior reasons
        is_focused: bool — True if attentive
    """
    for det in detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        track_id = det['track_id']
        score = det['score']
        reasons = det.get('reasons', [])
        is_focused = det['is_focused']

        color = (0, 255, 0) if is_focused else (0, 0, 255)
        thickness = 4 if any("长时间" in r for r in reasons) else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if show_labels:
            status = "FOCUS" if is_focused else "NOT FOCUS"
            main_reasons = reasons[:2]
            reason_text = f"({' | '.join(main_reasons)})" if main_reasons else ""
            label = f"ID:{track_id} {status}({score}) {reason_text}"
            label_y = max(20, y1 - 10)

            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, label_y - text_h - 5),
                          (x1 + text_w, label_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
