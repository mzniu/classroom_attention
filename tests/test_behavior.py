"""Regression tests for attention scoring with synthetic keypoints."""
import numpy as np
import pytest
from behavior import calculate_attention_score, StudentStateTracker
from config import Config


def make_kpt(x, y, conf=0.9):
    """Create a YOLO-format keypoint [x, y, confidence]."""
    return np.array([x, y, conf], dtype=np.float32)


def make_keypoints(overrides=None):
    """Create a standard upright-sitting 17-keypoint COCO array.

    Indices: 0=nose, 1=left_eye, 2=right_eye, 5=left_shoulder,
    6=right_shoulder, 9=left_wrist, 10=right_wrist, 13=left_hip, 14=right_hip.
    """
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = [0.5, 0.15, 0.9]   # nose
    kpts[1] = [0.47, 0.12, 0.9]  # left eye
    kpts[2] = [0.53, 0.12, 0.9]  # right eye
    kpts[5] = [0.35, 0.35, 0.9]  # left shoulder
    kpts[6] = [0.65, 0.35, 0.9]  # right shoulder
    kpts[9] = [0.25, 0.55, 0.9]  # left wrist
    kpts[10] = [0.75, 0.55, 0.9] # right wrist
    kpts[13] = [0.35, 0.65, 0.9] # left hip
    kpts[14] = [0.65, 0.65, 0.9] # right hip

    if overrides:
        for idx, val in overrides.items():
            kpts[idx] = np.array(val, dtype=np.float32)
    return kpts


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def tracker():
    return StudentStateTracker()


class TestAttentionScoring:
    def test_perfect_posture_scores_high(self, config, tracker):
        """Upright sitting student should score >= 90."""
        kpts = make_keypoints()
        score, reasons = calculate_attention_score(
            kpts, 100, config, tracker, 1, 30.0)
        assert score >= 90, f"Expected >=90, got {score}, reasons: {reasons}"
        assert len(reasons) == 0

    def test_head_down_detected(self, config, tracker):
        """Nose below shoulders should trigger 短暂低头."""
        kpts = make_keypoints({
            0: [0.5, 0.50, 0.9],  # nose below shoulder center (0.35)
        })
        score, reasons = calculate_attention_score(
            kpts, 100, config, tracker, 2, 30.0)
        assert score < 90, f"Expected <90 for head down, got {score}"
        assert any("短暂低头" in r for r in reasons), f"Reasons: {reasons}"

    def test_shoulder_tilt_detected(self, config, tracker):
        """Tilted shoulders should trigger 侧身."""
        kpts = make_keypoints({
            5: [0.35, 0.55, 0.9],  # left shoulder dropped
            6: [0.65, 0.35, 0.9],  # right shoulder normal
        })
        score, reasons = calculate_attention_score(
            kpts, 100, config, tracker, 3, 30.0)
        assert score < 90, f"Expected <90 for tilt, got {score}"
        assert any("侧身" in r for r in reasons), f"Reasons: {reasons}"

    def test_hand_below_hip(self, config, tracker):
        """Hand below hip should trigger 手部异常."""
        kpts = make_keypoints({
            9: [0.35, 0.80, 0.9],  # left wrist below hip
        })
        score, reasons = calculate_attention_score(
            kpts, 100, config, tracker, 4, 30.0)
        assert score < 90, f"Expected <90 for low hand, got {score}"
        assert any("手部" in r for r in reasons), f"Reasons: {reasons}"

    def test_long_term_head_down_severe_penalty(self, config):
        """Repeated head-down frames should trigger 长时间低头 with severe penalty."""
        tracker = StudentStateTracker()
        kpts = make_keypoints({
            0: [0.5, 0.50, 0.9],  # head down
        })
        # Simulate ~3.3 seconds at 30fps (100 frames)
        for _frame in range(100):
            score, reasons = calculate_attention_score(
                kpts, 100, config, tracker, 5, 30.0)
        assert score <= 30, f"Expected severe penalty, got {score}"
        assert any("长时间低头" in r for r in reasons), f"Reasons: {reasons}"

    def test_low_confidence_keypoints_no_penalty(self, config, tracker):
        """Low-confidence keypoints should NOT trigger false positives."""
        kpts = make_keypoints({
            0: [0.5, 0.15, 0.1],   # low confidence nose
            5: [0.35, 0.35, 0.1],  # low confidence left shoulder
            6: [0.65, 0.35, 0.1],  # low confidence right shoulder
        })
        score, reasons = calculate_attention_score(
            kpts, 100, config, tracker, 6, 30.0)
        assert score >= 80, f"Low conf triggered false positive: score={score}, reasons={reasons}"

    def test_score_clamped_0_to_100(self, config, tracker):
        """Score must stay in [0, 100] with extreme input."""
        kpts = make_keypoints({
            0: [0.5, 1.0, 0.9],    # extreme head down
            5: [0.1, 0.9, 0.9],    # extreme tilt
            6: [0.9, 0.1, 0.9],
        })
        score, _ = calculate_attention_score(
            kpts, 100, config, tracker, 7, 30.0)
        assert 0 <= score <= 100, f"Score {score} out of bounds"

    def test_none_keypoints_returns_zero(self, config, tracker):
        """None input should return (0, [])."""
        score, reasons = calculate_attention_score(
            None, 100, config, tracker, 8, 30.0)
        assert score == 0
        assert reasons == []

    def test_insufficient_keypoints_returns_zero(self, config, tracker):
        """Fewer than 17 keypoints should return (0, [])."""
        kpts = np.zeros((10, 3), dtype=np.float32)
        score, reasons = calculate_attention_score(
            kpts, 100, config, tracker, 9, 30.0)
        assert score == 0
        assert reasons == []
