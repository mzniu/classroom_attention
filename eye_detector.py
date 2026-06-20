"""Eye Aspect Ratio calculation using MediaPipe Face Mesh landmarks.

The EAR metric quantifies eye openness as the ratio of vertical to horizontal
eye landmark distances. Values near 0 indicate closed eyes; open eyes typically
produce EAR > 0.2.
"""
import numpy as np

# MediaPipe Face Mesh eye contour indices (from the 468-landmark topology).
# Each set of 6 points traces an eye: left-corner -> top -> right-corner -> bottom.
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


def calculate_ear_from_landmarks(eye_points: np.ndarray) -> float:
    """Compute Eye Aspect Ratio from 6 eye contour points.

    Points are ordered clockwise: [p1, p2, p3, p4, p5, p6]
      p1 - left corner (horizontal endpoint)
      p2 - upper-left point
      p3 - upper-right point
      p4 - right corner (horizontal endpoint)
      p5 - lower-right point
      p6 - lower-left point

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Args:
        eye_points: (6, 2) numpy array of (x, y) coordinates.

    Returns:
        EAR value as float. Returns 1.0 for degenerate (zero-width) eyes.
    """
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])
    h = np.linalg.norm(eye_points[0] - eye_points[3])
    if h < 1e-6:
        return 1.0
    return (v1 + v2) / (2.0 * h)


class EyeDetector:
    """Wraps MediaPipe Face Mesh for per-student EAR calculation."""

    def __init__(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options as base
        except ImportError:
            raise ImportError(
                "MediaPipe is required for EyeDetector. "
                "Install with: pip install mediapipe"
            )
        options = vision.FaceLandmarkerOptions(
            base_options=base.BaseOptions(
                model_asset_path='face_landmarker.task'),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def get_ear(self, face_image) -> float | None:
        """Run Face Landmarker on a face crop and return average EAR.

        Args:
            face_image: BGR image crop (numpy array) of a single face.

        Returns:
            Average left+right EAR as float, or None if detection fails.
        """
        import cv2
        import mediapipe as mp
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.face_landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        h, w = face_image.shape[:2]

        def extract_eye_points(indices):
            points = []
            for idx in indices:
                lm = landmarks[idx]
                points.append([lm.x * w, lm.y * h])
            return np.array(points, dtype=np.float32)

        left_ear = calculate_ear_from_landmarks(
            extract_eye_points(LEFT_EYE_INDICES))
        right_ear = calculate_ear_from_landmarks(
            extract_eye_points(RIGHT_EYE_INDICES))
        return (left_ear + right_ear) / 2.0

    def close(self):
        """Release Face Landmarker resources."""
        try:
            self.face_landmarker.close()
        except Exception:
            pass
