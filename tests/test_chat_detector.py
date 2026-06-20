"""Regression tests for chat detection."""
import numpy as np
import pytest
from config import Config


def make_detection(track_id, x1, y1, x2, y2, shoulder_angle=0.0):
    """Create a viz_detection dict with bbox and keypoints for chat detection.

    shoulder_angle: degrees, 0=horizontal (facing forward),
                    positive=right shoulder back, negative=left shoulder back.
    """
    height = y2 - y1
    width = x2 - x1
    kpts = np.zeros((17, 3), dtype=np.float32)
    # nose
    kpts[0] = [float(x1 + x2) / 2, float(y1 + height * 0.3), 0.9]
    # left eye, right eye
    kpts[1] = [float(x1 + width * 0.4), float(y1 + height * 0.25), 0.9]
    kpts[2] = [float(x1 + width * 0.6), float(y1 + height * 0.25), 0.9]
    # shoulders (tilted by shoulder_angle)
    import math
    rad = math.radians(shoulder_angle)
    cx = float(x1 + x2) / 2
    half_w = width * 0.25
    sy = float(y1 + height * 0.4)
    kpts[5] = [cx - half_w * math.cos(rad), sy - half_w * math.sin(rad), 0.9]
    kpts[6] = [cx + half_w * math.cos(rad), sy + half_w * math.sin(rad), 0.9]
    # wrists
    kpts[9] = [float(x1 + width * 0.2), float(y1 + height * 0.7), 0.9]
    kpts[10] = [float(x1 + width * 0.8), float(y1 + height * 0.7), 0.9]
    # hips
    kpts[13] = [float(x1 + width * 0.3), float(y1 + height * 0.85), 0.9]
    kpts[14] = [float(x1 + width * 0.7), float(y1 + height * 0.85), 0.9]

    return {
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        'track_id': track_id,
        'keypoints': kpts,
        'score': 100,
        'reasons': [],
    }


@pytest.fixture
def config():
    c = Config()
    c.behavior.chat_enabled = True
    return c


def _get_chat_pairs(result):
    """Extract student ID pairs from detect_chat result."""
    if not result:
        return []
    return [(r[0], r[1]) for r in result]


class TestChatDetection:
    def test_no_chat_when_far_apart(self, config):
        """Students far apart should NOT trigger chat detection."""
        from chat_detector import detect_chat
        dets = [
            make_detection(1, 10, 10, 60, 100),
            make_detection(2, 200, 10, 250, 100),  # far away
        ]
        result = detect_chat(dets, config)
        assert len(result) == 0

    def test_no_chat_when_facing_same_direction(self, config):
        """Close students facing the same way should NOT trigger chat."""
        from chat_detector import detect_chat
        dets = [
            make_detection(1, 10, 10, 60, 100, shoulder_angle=0.0),
            make_detection(2, 65, 10, 115, 100, shoulder_angle=0.0),
        ]
        result = detect_chat(dets, config)
        assert len(result) == 0

    def test_chat_when_close_and_facing(self, config):
        """Close students facing each other SHOULD trigger chat detection."""
        from chat_detector import detect_chat
        dets = [
            make_detection(1, 10, 10, 60, 100, shoulder_angle=30.0),
            make_detection(2, 65, 10, 115, 100, shoulder_angle=-30.0),
        ]
        result = detect_chat(dets, config)
        assert len(result) >= 1, f"Expected chat detection, got {result}"

    def test_no_chat_single_student(self, config):
        """Single student should produce no chat pairs."""
        from chat_detector import detect_chat
        dets = [
            make_detection(1, 10, 10, 60, 100),
        ]
        result = detect_chat(dets, config)
        assert len(result) == 0

    def test_no_chat_when_disabled(self, config):
        """When chat_enabled=False, no detection even if conditions met."""
        from chat_detector import detect_chat
        config.behavior.chat_enabled = False
        dets = [
            make_detection(1, 10, 10, 60, 100, shoulder_angle=30.0),
            make_detection(2, 65, 10, 115, 100, shoulder_angle=-30.0),
        ]
        result = detect_chat(dets, config)
        assert len(result) == 0

    def test_no_chat_empty_detections(self, config):
        """Empty detection list should return empty result."""
        from chat_detector import detect_chat
        result = detect_chat([], config)
        assert len(result) == 0

    def test_three_students_two_chatting(self, config):
        """3 students: 2 adjacent chatting, 1 alone. Only 1 chat pair."""
        from chat_detector import detect_chat
        dets = [
            make_detection(1, 10, 10, 60, 100, shoulder_angle=30.0),
            make_detection(2, 65, 10, 115, 100, shoulder_angle=-30.0),
            make_detection(3, 200, 10, 250, 100, shoulder_angle=0.0),
        ]
        result = detect_chat(dets, config)
        pairs = _get_chat_pairs(result)
        assert (1, 2) in pairs or (2, 1) in pairs, f"Expected (1,2) chat, got {pairs}"
        assert not any(3 in p for p in pairs), f"Student 3 should not chat: {pairs}"

    def test_chat_pair_returns_penalty_for_both(self, config):
        """Each student in a chat pair gets a chat reason added."""
        from chat_detector import detect_chat
        dets = [
            make_detection(1, 10, 10, 60, 100, shoulder_angle=30.0),
            make_detection(2, 65, 10, 115, 100, shoulder_angle=-30.0),
        ]
        result = detect_chat(dets, config)
        # result should be [(student_id_1, student_id_2), ...]
        assert (1, 2) in result or (2, 1) in result
