"""Regression tests for attention scoring with synthetic keypoints."""
import numpy as np
import pytest
from behavior import calculate_attention_score, StudentStateTracker
from config import Config


def make_kpt(x, y, conf=0.9):
    """Create a YOLO-format keypoint [x, y, confidence]."""
    return np.array([x, y, conf], dtype=np.float32)


def make_keypoints(overrides=None, scale=1.0):
    """Create a standard upright-sitting 17-keypoint COCO array.

    Indices: 0=nose, 1=left_eye, 2=right_eye, 5=left_shoulder,
    6=right_shoulder, 9=left_wrist, 10=right_wrist, 13=left_hip, 14=right_hip.

    Args:
        overrides: dict of index -> [x, y, conf] to override specific keypoints.
        scale: multiply x,y coordinates by this factor (use bbox_height for pixel coords).
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
    kpts[:, 0] *= scale
    kpts[:, 1] *= scale
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
        kpts = make_keypoints(
            {0: [0.5, 0.50, 0.9]},  # nose below shoulder center (0.35)
            scale=100)  # pixel coords for bbox_height=100
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
        kpts = make_keypoints(
            {0: [0.5, 0.50, 0.9]},  # head down
            scale=100)  # pixel coords for bbox_height=100
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


class TestMediaPipeConversion:
    def test_mediapipe_to_coco_conversion(self):
        """MediaPipe landmarks should convert to COCO keypoint format."""
        from behavior import mediapipe_landmarks_to_coco_keypoints

        class MockLandmark:
            def __init__(self, x, y, visibility):
                self.x = x
                self.y = y
                self.visibility = visibility

        landmarks = [MockLandmark(0, 0, 0)] * 33
        landmarks[0] = MockLandmark(0.5, 0.15, 0.9)    # nose
        landmarks[2] = MockLandmark(0.47, 0.12, 0.9)   # left eye
        landmarks[5] = MockLandmark(0.53, 0.12, 0.9)   # right eye
        landmarks[11] = MockLandmark(0.35, 0.35, 0.9)  # left shoulder
        landmarks[12] = MockLandmark(0.65, 0.35, 0.9)  # right shoulder

        kpts = mediapipe_landmarks_to_coco_keypoints(landmarks)
        assert kpts.shape == (17, 3)
        assert float(kpts[0][0]) == pytest.approx(0.5)
        assert float(kpts[0][1]) == pytest.approx(0.15)
        assert float(kpts[5][0]) == pytest.approx(0.35)
        assert float(kpts[6][0]) == pytest.approx(0.65)

    def test_mediapipe_to_coco_none_returns_zeros(self):
        """None input should return zero-filled array."""
        from behavior import mediapipe_landmarks_to_coco_keypoints
        kpts = mediapipe_landmarks_to_coco_keypoints(None)
        assert kpts.shape == (17, 3)
        assert np.all(kpts == 0)

    def test_v1_long_term_head_down(self):
        """v1 with StudentStateTracker should detect long-term head down
        using MediaPipe landmarks converted to COCO keypoints."""
        from behavior import (
            mediapipe_landmarks_to_coco_keypoints,
            StudentStateTracker,
            calculate_attention_score,
        )
        from config import Config

        class MockLandmark:
            def __init__(self, x, y, visibility):
                self.x = x
                self.y = y
                self.visibility = visibility

        landmarks = [MockLandmark(0, 0, 0)] * 33
        landmarks[0] = MockLandmark(0.5, 0.50, 0.9)   # nose low
        landmarks[2] = MockLandmark(0.47, 0.47, 0.9)
        landmarks[5] = MockLandmark(0.53, 0.47, 0.9)
        landmarks[11] = MockLandmark(0.35, 0.35, 0.9)  # shoulder
        landmarks[12] = MockLandmark(0.65, 0.35, 0.9)
        landmarks[15] = MockLandmark(0.25, 0.55, 0.9)  # wrist
        landmarks[16] = MockLandmark(0.75, 0.55, 0.9)
        landmarks[23] = MockLandmark(0.35, 0.65, 0.9)  # hip
        landmarks[24] = MockLandmark(0.65, 0.65, 0.9)

        config = Config()
        tracker = StudentStateTracker()

        kpts = mediapipe_landmarks_to_coco_keypoints(landmarks)
        # Convert normalized coords to pixel coords (as ca.py does)
        bbox_h = 100
        kpts[:, 0] *= bbox_h
        kpts[:, 1] *= bbox_h
        for _ in range(100):
            score, reasons = calculate_attention_score(
                kpts, bbox_h, config, tracker, 1, 30.0)

        assert score <= 30, f"Expected severe penalty, got {score}"
        assert any("长时间低头" in r for r in reasons)


class TestHandRaise:
    def test_hand_raise_detected_no_penalty(self, config, tracker):
        """Wrist above shoulder AND above nose = hand raised, no penalty."""
        kpts = make_keypoints({
            9: [0.35, 0.10, 0.9],   # left wrist high above head
        }, scale=100)
        score, reasons = calculate_attention_score(
            kpts, 100, config, tracker, 10, 30.0)
        assert score >= 90, f"Hand raise should NOT penalize, got {score}"
        assert any("举手" in r for r in reasons), f"Expected 举手 in {reasons}"

    def test_hand_raise_right_hand(self, config, tracker):
        """Right hand raised should also be detected."""
        kpts = make_keypoints({
            10: [0.65, 0.10, 0.9],  # right wrist high above head
        }, scale=100)
        _, reasons = calculate_attention_score(
            kpts, 100, config, tracker, 11, 30.0)
        assert any("举手" in r for r in reasons), f"Reasons: {reasons}"

    def test_no_hand_raise_when_normal(self, config, tracker):
        """Normal hand position (wrist below shoulder) should NOT trigger."""
        kpts = make_keypoints(scale=100)
        _, reasons = calculate_attention_score(
            kpts, 100, config, tracker, 12, 30.0)
        assert not any("举手" in r for r in reasons), f"Unexpected 举手: {reasons}"

    def test_hand_raise_wrist_below_nose_ignored(self, config, tracker):
        """Wrist above shoulder but below nose = not raised (arm not high enough)."""
        kpts = make_keypoints({
            # nose at y=15 (scaled), shoulder at y=35, wrist at y=25
            9: [0.35, 0.25, 0.9],   # wrist between nose and shoulder
        }, scale=100)
        _, reasons = calculate_attention_score(
            kpts, 100, config, tracker, 13, 30.0)
        assert not any("举手" in r for r in reasons), \
            f"Wrist not above nose, should not trigger: {reasons}"

    def test_hand_raise_both_hands(self, config, tracker):
        """Both hands raised should still only note it once."""
        kpts = make_keypoints({
            9: [0.35, 0.10, 0.9],    # left wrist high
            10: [0.65, 0.10, 0.9],   # right wrist high
        }, scale=100)
        _, reasons = calculate_attention_score(
            kpts, 100, config, tracker, 14, 30.0)
        assert any("举手" in r for r in reasons), f"Reasons: {reasons}"

    def test_hand_raise_low_confidence_ignored(self, config, tracker):
        """Low-confidence wrist should NOT trigger hand raise."""
        kpts = make_keypoints({
            9: [0.35, 0.10, 0.4],   # low confidence
        }, scale=100)
        _, reasons = calculate_attention_score(
            kpts, 100, config, tracker, 15, 30.0)
        assert not any("举手" in r for r in reasons), \
            f"Low conf should not trigger: {reasons}"

    def test_hand_raise_disabled(self, config, tracker):
        """When hand_raise_enabled=False, no detection."""
        config.behavior.hand_raise_enabled = False
        kpts = make_keypoints({
            9: [0.35, 0.10, 0.9],   # wrist high
        }, scale=100)
        _, reasons = calculate_attention_score(
            kpts, 100, config, tracker, 16, 30.0)
        assert not any("举手" in r for r in reasons), \
            f"Disabled but still detected: {reasons}"
