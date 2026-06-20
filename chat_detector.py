"""Chat detection for adjacent students in classroom.

Computes bbox proximity and shoulder-angle divergence to detect
pairs of students who are sitting close together and turned toward
each other (likely chatting).

EXPERIMENTAL: High false-positive rate expected. Disabled by default.
"""
import math
import numpy as np


def _compute_shoulder_angle(keypoints):
    """Compute shoulder line angle from keypoints.

    Returns angle in degrees (0 = horizontal facing forward,
    positive = right shoulder back / turned to the right).
    Returns None if shoulders have low confidence.
    """
    if keypoints is None or len(keypoints) < 17:
        return None
    left = keypoints[5]
    right = keypoints[6]
    if left[2] < 0.5 or right[2] < 0.5:
        return None
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    return math.degrees(math.atan2(dy, dx))


def _horizontal_gap(det_a, det_b):
    """Compute the horizontal gap between two bounding boxes.

    Positive = gap between them, negative = overlap.
    Returns normalized gap (gap / avg_bbox_height).
    """
    a_left, a_right = det_a['x1'], det_a['x2']
    b_left, b_right = det_b['x1'], det_b['x2']
    a_center = (a_left + a_right) / 2
    b_center = (b_left + b_right) / 2
    if a_center <= b_center:
        gap = b_left - a_right
    else:
        gap = a_left - b_right
    avg_height = ((det_a['y2'] - det_a['y1']) + (det_b['y2'] - det_b['y1'])) / 2
    if avg_height <= 0:
        return float('inf')
    return gap / avg_height


def detect_chat(detections, config):
    """Detect pairs of students who may be chatting.

    A pair is flagged when:
    1. Their bboxes are adjacent (gap < distance_threshold * avg_bbox_height)
    2. Their shoulder angles diverge (|angle_A - angle_B| > angle_threshold)

    Args:
        detections: list of viz_detection dicts, each with:
            x1, y1, x2, y2, track_id, keypoints
        config: Config instance

    Returns:
        list of (student_id_1, student_id_2) tuples for detected chat pairs
    """
    ch = config.behavior
    if not ch.chat_enabled:
        return []

    if len(detections) < 2:
        return []

    chat_pairs = []
    n = len(detections)
    for i in range(n):
        for j in range(i + 1, n):
            det_a = detections[i]
            det_b = detections[j]
            gap = _horizontal_gap(det_a, det_b)
            if gap > ch.chat_distance_threshold:
                continue
            angle_a = _compute_shoulder_angle(det_a.get('keypoints'))
            angle_b = _compute_shoulder_angle(det_b.get('keypoints'))
            if angle_a is None or angle_b is None:
                continue
            angle_diff = abs(angle_a - angle_b)
            # Handle 180° wraparound (e.g., 170° vs -170° = 20° diff)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            if angle_diff >= ch.chat_angle_threshold:
                chat_pairs.append((det_a['track_id'], det_b['track_id']))

    return chat_pairs
