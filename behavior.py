"""Behavior analysis engine for classroom attention detection.

Works with YOLOv8-pose keypoints (COCO 17-keypoint format).
Each keypoint is [x, y, confidence] where x,y are pixel coordinates.
"""
from collections import defaultdict, deque
import numpy as np
from config import Config

# MediaPipe Pose (33 landmarks) → COCO (17 keypoints) index mapping.
# Source: MediaPipe Pose landmark model card.
_MP_TO_COCO = {
    0: 0,    # NOSE
    1: 2,    # LEFT_EYE
    2: 5,    # RIGHT_EYE
    5: 11,   # LEFT_SHOULDER
    6: 12,   # RIGHT_SHOULDER
    9: 15,   # LEFT_WRIST
    10: 16,  # RIGHT_WRIST
    13: 23,  # LEFT_HIP
    14: 24,  # RIGHT_HIP
}


def mediapipe_landmarks_to_coco_keypoints(landmarks) -> "np.ndarray":
    """Convert MediaPipe Pose landmarks to COCO 17-keypoint format.

    Args:
        landmarks: List of 33 MediaPipe Pose landmarks (each with x, y, visibility),
                   or None.

    Returns:
        (17, 3) numpy array [x, y, confidence], zero-filled where no mapping exists.
    """
    kpts = np.zeros((17, 3), dtype=np.float32)
    if landmarks is None:
        return kpts

    for coco_idx, mp_idx in _MP_TO_COCO.items():
        lm = landmarks[mp_idx]
        kpts[coco_idx] = [lm.x, lm.y, lm.visibility]
    return kpts


class StudentStateTracker:
    """Tracks per-student behavioral state across frames."""

    def __init__(self, eye_detector=None):
        self.head_position = defaultdict(lambda: deque(maxlen=30))
        self.eye_openness = defaultdict(lambda: deque(maxlen=30))
        self.gaze_position = defaultdict(lambda: deque(maxlen=30))
        self.head_down_timer = defaultdict(float)
        self.eye_closed_timer = defaultdict(float)
        self.stillness_timer = defaultdict(float)
        self.eye_detector = eye_detector

    def update(self, student_id: int, keypoints, fps: float,
               face_crop=None) -> float:
        """Update student state from keypoints. Returns current time in seconds."""
        current_time = len(self.head_position[student_id]) / fps
        if keypoints is not None and len(keypoints) > 0:
            nose = keypoints[0]
            left_eye = keypoints[1]
            right_eye = keypoints[2]
            if nose[2] > 0.5:
                self.head_position[student_id].append(nose[:2])
            ear = self._get_ear(left_eye, right_eye, face_crop)
            self.eye_openness[student_id].append(ear)
            if left_eye[2] > 0.5 and right_eye[2] > 0.5:
                gaze_x = (left_eye[0] + right_eye[0]) / 2
                gaze_y = (left_eye[1] + right_eye[1]) / 2
                self.gaze_position[student_id].append((gaze_x, gaze_y))
        return current_time

    def _get_ear(self, left_eye, right_eye, face_crop=None) -> float:
        """Get EAR using Face Mesh when available, else confidence proxy."""
        if self.eye_detector is not None and face_crop is not None:
            try:
                ear = self.eye_detector.get_ear(face_crop)
                if ear is not None:
                    return ear
            except Exception:
                pass
        return self._calculate_ear(left_eye, right_eye)

    @staticmethod
    def _calculate_ear(left_eye, right_eye) -> float:
        """Simplified eye aspect ratio using confidence as proxy."""
        try:
            if left_eye[2] > 0.5 and right_eye[2] > 0.5:
                return (left_eye[2] + right_eye[2]) / 2
            return 1.0
        except Exception:
            return 1.0

    def check_long_term_behaviors(self, student_id: int, keypoints, fps: float,
                                   config: Config) -> list:
        """Check for sustained behaviors (head down, eye closed, stillness)."""
        behaviors = []
        bh = config.behavior

        # Long-term head down
        if keypoints is not None and len(keypoints) > 0:
            nose = keypoints[0]
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            if (nose[2] > 0.5 and left_shoulder[2] > 0.5
                    and right_shoulder[2] > 0.5):
                shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
                if nose[1] - shoulder_center_y > bh.head_down_threshold:
                    self.head_down_timer[student_id] += 1 / fps
                    if self.head_down_timer[student_id] >= bh.head_down_duration:
                        behaviors.append(
                            f"长时间低头({self.head_down_timer[student_id]:.1f}s)")
                else:
                    self.head_down_timer[student_id] = 0
            else:
                self.head_down_timer[student_id] = 0

        # Long-term eye closed
        if len(self.eye_openness[student_id]) > 0:
            ear = self.eye_openness[student_id][-1]
            if ear < bh.eye_closed_threshold:
                self.eye_closed_timer[student_id] += 1 / fps
                if self.eye_closed_timer[student_id] >= bh.eye_closed_duration:
                    behaviors.append(
                        f"闭眼({self.eye_closed_timer[student_id]:.1f}s)")
            else:
                self.eye_closed_timer[student_id] = 0
        else:
            self.eye_closed_timer[student_id] = 0

        # Long-term stillness / staring blankly
        if len(self.head_position[student_id]) >= 5:
            recent = list(self.head_position[student_id])[-5:]
            head_movement = 0
            for i in range(1, len(recent)):
                dx = recent[i][0] - recent[i - 1][0]
                dy = recent[i][1] - recent[i - 1][1]
                head_movement += np.sqrt(dx * dx + dy * dy)

            if head_movement < bh.stillness_threshold:
                self.stillness_timer[student_id] += 1 / fps
                if self.stillness_timer[student_id] >= bh.stillness_duration:
                    behaviors.append(
                        f"发呆({self.stillness_timer[student_id]:.1f}s)")
            else:
                self.stillness_timer[student_id] = 0
        else:
            self.stillness_timer[student_id] = 0

        return behaviors


def calculate_attention_score(keypoints, bbox_height: float, config: Config,
                               state_tracker: StudentStateTracker,
                               student_id: int, fps: float,
                               face_crop=None) -> tuple:
    """Calculate attention score from keypoints.

    Args:
        keypoints: numpy array (17, 3) of COCO keypoints [x, y, conf]
        bbox_height: bounding box height in pixels
        config: Config instance
        state_tracker: StudentStateTracker instance
        student_id: tracked student ID
        fps: video frames per second
        face_crop: optional BGR face image crop for real EAR calculation

    Returns:
        (score, reasons) where score is 0-100 and reasons is list of strings
    """
    if keypoints is None or len(keypoints) < 17:
        return 0, []

    score = 100
    reasons = []
    bh = config.behavior

    try:
        state_tracker.update(student_id, keypoints, fps, face_crop)
        long_term_behaviors = state_tracker.check_long_term_behaviors(
            student_id, keypoints, fps, config)

        for behavior in long_term_behaviors:
            if "长时间低头" in behavior:
                score -= bh.head_down_penalty
            elif "闭眼" in behavior:
                score -= bh.eye_closed_penalty
            elif "发呆" in behavior:
                score -= bh.stillness_penalty
            reasons.append(behavior)

        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hand = keypoints[9]
        right_hand = keypoints[10]
        left_hip = keypoints[13]
        right_hip = keypoints[14]

        # Short-term head down
        if (not any("长时间低头" in r for r in reasons)
                and nose[2] > 0.5
                and left_shoulder[2] > 0.5
                and right_shoulder[2] > 0.5):
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            head_drop = (nose[1] - shoulder_center_y) * bbox_height
            if head_drop > bh.head_down_threshold * bbox_height:
                score -= bh.short_head_down_penalty
                reasons.append("短暂低头")

        # Shoulder tilt
        if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
            shoulder_vec = np.array([
                right_shoulder[0] - left_shoulder[0],
                right_shoulder[1] - left_shoulder[1]
            ])
            angle = np.degrees(np.arctan2(abs(shoulder_vec[1]),
                                           abs(shoulder_vec[0])))
            if angle > bh.shoulder_tilt_threshold:
                score -= bh.shoulder_tilt_penalty
                reasons.append(f"侧身({int(angle)}°)")

        # Hand below hip
        hand_low = False
        if left_hand[2] > 0.5 and left_hip[2] > 0.5:
            if left_hand[1] > left_hip[1] + bh.hand_below_hip_threshold:
                hand_low = True
        if right_hand[2] > 0.5 and right_hip[2] > 0.5:
            if right_hand[1] > right_hip[1] + bh.hand_below_hip_threshold:
                hand_low = True
        if hand_low:
            score -= bh.hand_below_hip_penalty
            reasons.append("手部异常")

    except Exception:
        return max(0, score), reasons

    return max(0, min(100, score)), reasons
