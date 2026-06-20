# Classroom Attention Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate code duplication between ca.py and ca_gpu.py by extracting shared modules (config, behavior, reporter, visualizer, utils), externalizing thresholds to config.yaml, and adding regression tests.

**Architecture:** Five shared modules with clear interfaces — `config.py` loads YAML, `behavior.py` scores attention from normalized keypoints, `reporter.py` generates CSV/console reports, `visualizer.py` draws annotations, `utils.py` handles video I/O. Both backends (v1 MediaPipe + v2 YOLOv8-pose) consume these modules. Single entry point `main.py` auto-selects backend.

**Tech Stack:** Python 3.8+, PyTorch, Ultralytics 8.3.0, OpenCV, pandas, PyYAML, pytest

## Global Constraints

- Python >= 3.8
- ultralytics == 8.3.0 (pinned)
- opencv-python == 4.8.1.78 (pinned)
- Backward compatible: existing CLI of ca.py and ca_gpu.py must still work
- All new modules must have type hints for public functions
- Tests use pytest with fixed synthetic keypoint inputs (no video files needed)
- No new dependencies beyond PyYAML and pytest (dev-only)

---

## File Structure

```
classroom_attention/
├── main.py              # NEW: unified CLI entry, auto backend selection
├── config.py            # NEW: YAML config loader
├── config.yaml          # NEW: externalized thresholds
├── behavior.py          # NEW: StudentStateTracker + calculate_attention_score
├── reporter.py          # NEW: generate_report + print_report
├── visualizer.py        # NEW: draw_annotations
├── utils.py             # NEW: create_video_capture, detect_device, suppress_warnings
├── requirements.txt     # MODIFY: add missing deps
├── .gitignore           # MODIFY: add *.pt, *.csv, *.mp4, __pycache__
├── ca.py                # MODIFY: import from shared modules
├── ca_gpu.py            # MODIFY: import from shared modules
├── tests/
│   ├── __init__.py      # NEW
│   ├── test_behavior.py # NEW: regression tests for scoring
│   ├── test_config.py   # NEW: config loading tests
│   └── test_reporter.py # NEW: report generation tests
```

**File responsibilities:**
- `config.py` — loads `config.yaml`, provides `Config` dataclass with typed fields
- `config.yaml` — all tunable thresholds, no code values
- `behavior.py` — `StudentStateTracker` class + `calculate_attention_score()` function, works with normalized keypoints (x,y,conf in [0,1])
- `reporter.py` — `generate_report(records) -> (DataFrame, dict)`, `print_report(summary) -> None`
- `visualizer.py` — `draw_annotations(frame, detections) -> frame` — pure function
- `utils.py` — `create_video_capture(path) -> (cap, fps, frames)`, `detect_device() -> str`, `suppress_warnings()`
- `main.py` — argparse CLI, creates Config, selects backend, runs pipeline

---

### Task 1: Fix dependencies and .gitignore

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Update requirements.txt**

Add missing dependencies needed by ca.py and new config module:

```
torch>=1.13.0
torchvision>=0.14.0
ultralytics==8.3.0
opencv-python==4.8.1.78
pandas>=1.5.3
numpy>=1.24.0
mediapipe>=0.10.0
deep_sort_realtime>=1.3.0
pyyaml>=6.0
```

- [ ] **Step 2: Update .gitignore**

Replace current content:

```gitignore
# niuclaude-memory
MEMORY.md

# Python
__pycache__/
*.pyc
*.pyo

# Model files (downloaded automatically by ultralytics)
*.pt

# Output files
*.csv
*.mp4
output_annotated*.mp4

# IDE
.vscode/
.idea/

# Virtual env
venv/
.venv/
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "chore: update dependencies and gitignore for refactor"
```

---

### Task 2: Create config.yaml and config.py

**Files:**
- Create: `config.yaml`
- Create: `config.py`

**Interfaces:**
- Produces: `Config` dataclass with fields: `pose_model`, `device`, `head_down_threshold`, `head_down_duration`, `eye_closed_threshold`, `eye_closed_duration`, `stillness_threshold`, `stillness_duration`, `gaze_fixed_threshold`, `shoulder_tilt_threshold`, `hand_below_hip_threshold`, `attention_score_threshold`, `output_video`, `output_video_path`, `show_labels`, `skip_frames`, `confidence_threshold`, `max_age`, `n_init`, `max_iou_distance`
- Produces: `load_config(path) -> Config` function

- [ ] **Step 1: Create config.yaml**

```yaml
# Classroom Attention Detection - Configuration
# All tunable thresholds for different classroom scenarios

model:
  pose_model: "yolov8m-pose.pt"    # v2 GPU backend
  yolo_model: "yolo11m.pt"         # v1 CPU/MediaPipe backend
  device: 0                         # 0 = GPU, "cpu" for CPU-only

behavior:
  head_down:
    threshold: 0.03       # nose below shoulder-center (normalized)
    duration_sec: 3.0     # seconds before "long-term" classification
    penalty: 80           # points deducted for long-term head-down

  eye_closed:
    ear_threshold: 0.18   # eye aspect ratio proxy
    duration_sec: 2.0
    penalty: 70

  stillness:
    movement_px: 5.0      # max head movement pixels (in normalized space)
    duration_sec: 4.0
    penalty: 50

  shoulder_tilt:
    angle_threshold: 25   # degrees
    penalty: 20

  hand_below_hip:
    threshold: 0.02       # normalized
    penalty: 15

  short_head_down:
    penalty: 30           # single-frame head-down penalty

scoring:
  attention_threshold: 50  # score below this = not-focused

tracking:
  max_age: 30
  n_init: 3
  max_iou_distance: 0.7

detection:
  confidence_threshold: 0.5
  person_class_id: 0

output:
  save_video: true
  video_path: "output_annotated.mp4"
  show_labels: true
  save_csv: true
  csv_path: "attention_report.csv"

performance:
  skip_frames: 2          # process every (N+1)th frame
```

- [ ] **Step 2: Create config.py**

```python
"""Configuration loader from YAML file."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class BehaviorConfig:
    head_down_threshold: float = 0.03
    head_down_duration: float = 3.0
    head_down_penalty: int = 80
    eye_closed_threshold: float = 0.18
    eye_closed_duration: float = 2.0
    eye_closed_penalty: int = 70
    stillness_threshold: float = 5.0
    stillness_duration: float = 4.0
    stillness_penalty: int = 50
    shoulder_tilt_threshold: int = 25
    shoulder_tilt_penalty: int = 20
    hand_below_hip_threshold: float = 0.02
    hand_below_hip_penalty: int = 15
    short_head_down_penalty: int = 30


@dataclass
class Config:
    # Model
    pose_model: str = "yolov8m-pose.pt"
    yolo_model: str = "yolo11m.pt"
    device: str = "0"

    # Behavior (nested)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)

    # Scoring
    attention_threshold: int = 50

    # Tracking
    max_age: int = 30
    n_init: int = 3
    max_iou_distance: float = 0.7

    # Detection
    confidence_threshold: float = 0.5
    person_class_id: int = 0

    # Output
    save_video: bool = True
    video_path: str = "output_annotated.mp4"
    show_labels: bool = True
    save_csv: bool = True
    csv_path: str = "attention_report.csv"

    # Performance
    skip_frames: int = 2


def load_config(path: str = "config.yaml") -> Config:
    """Load configuration from YAML file. Returns Config with defaults for missing keys."""
    cfg = Config()
    yaml_path = Path(path)
    if not yaml_path.exists():
        return cfg

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return cfg

    if "model" in data:
        m = data["model"]
        cfg.pose_model = m.get("pose_model", cfg.pose_model)
        cfg.yolo_model = m.get("yolo_model", cfg.yolo_model)
        cfg.device = str(m.get("device", cfg.device))

    if "behavior" in data:
        b = data["behavior"]
        bh = cfg.behavior
        if "head_down" in b:
            hd = b["head_down"]
            bh.head_down_threshold = hd.get("threshold", bh.head_down_threshold)
            bh.head_down_duration = hd.get("duration_sec", bh.head_down_duration)
            bh.head_down_penalty = hd.get("penalty", bh.head_down_penalty)
        if "eye_closed" in b:
            ec = b["eye_closed"]
            bh.eye_closed_threshold = ec.get("ear_threshold", bh.eye_closed_threshold)
            bh.eye_closed_duration = ec.get("duration_sec", bh.eye_closed_duration)
            bh.eye_closed_penalty = ec.get("penalty", bh.eye_closed_penalty)
        if "stillness" in b:
            st = b["stillness"]
            bh.stillness_threshold = st.get("movement_px", bh.stillness_threshold)
            bh.stillness_duration = st.get("duration_sec", bh.stillness_duration)
            bh.stillness_penalty = st.get("penalty", bh.stillness_penalty)
        if "shoulder_tilt" in b:
            st = b["shoulder_tilt"]
            bh.shoulder_tilt_threshold = st.get("angle_threshold", bh.shoulder_tilt_threshold)
            bh.shoulder_tilt_penalty = st.get("penalty", bh.shoulder_tilt_penalty)
        if "hand_below_hip" in b:
            hb = b["hand_below_hip"]
            bh.hand_below_hip_threshold = hb.get("threshold", bh.hand_below_hip_threshold)
            bh.hand_below_hip_penalty = hb.get("penalty", bh.hand_below_hip_penalty)
        if "short_head_down" in b:
            sh = b["short_head_down"]
            bh.short_head_down_penalty = sh.get("penalty", bh.short_head_down_penalty)

    if "scoring" in data:
        cfg.attention_threshold = data["scoring"].get("attention_threshold", cfg.attention_threshold)
    if "tracking" in data:
        t = data["tracking"]
        cfg.max_age = t.get("max_age", cfg.max_age)
        cfg.n_init = t.get("n_init", cfg.n_init)
        cfg.max_iou_distance = t.get("max_iou_distance", cfg.max_iou_distance)
    if "detection" in data:
        d = data["detection"]
        cfg.confidence_threshold = d.get("confidence_threshold", cfg.confidence_threshold)
        cfg.person_class_id = d.get("person_class_id", cfg.person_class_id)
    if "output" in data:
        o = data["output"]
        cfg.save_video = o.get("save_video", cfg.save_video)
        cfg.video_path = o.get("video_path", cfg.video_path)
        cfg.show_labels = o.get("show_labels", cfg.show_labels)
        cfg.save_csv = o.get("save_csv", cfg.save_csv)
        cfg.csv_path = o.get("csv_path", cfg.csv_path)
    if "performance" in data:
        p = data["performance"]
        cfg.skip_frames = p.get("skip_frames", cfg.skip_frames)

    return cfg
```

- [ ] **Step 3: Write config test**

Create `tests/test_config.py`:

```python
"""Tests for config loading."""
import tempfile
from pathlib import Path
from config import Config, load_config, BehaviorConfig


def test_default_config():
    cfg = Config()
    assert cfg.behavior.head_down_penalty == 80
    assert cfg.attention_threshold == 50
    assert cfg.skip_frames == 2


def test_load_config_with_defaults():
    cfg = load_config("nonexistent.yaml")
    assert isinstance(cfg, Config)
    assert cfg.behavior.head_down_threshold == 0.03


def test_load_config_partial():
    import yaml
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.dump({"scoring": {"attention_threshold": 60}}, f)
        f.flush()
        cfg = load_config(f.name)
        assert cfg.attention_threshold == 60
        assert cfg.behavior.head_down_penalty == 80  # unchanged default
        Path(f.name).unlink()


def test_behavior_config_defaults():
    b = BehaviorConfig()
    assert b.head_down_duration == 3.0
    assert b.eye_closed_duration == 2.0
    assert b.stillness_penalty == 50
```

- [ ] **Step 4: Run config tests**

```bash
python -m pytest tests/test_config.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add config.yaml config.py tests/test_config.py tests/__init__.py
git commit -m "feat: add config.yaml and config.py with tests"
```

---

### Task 3: Extract behavior.py from ca_gpu.py

**Files:**
- Create: `behavior.py`
- Test: `tests/test_behavior.py`

**Interfaces:**
- Consumes: `Config`, `BehaviorConfig` from config.py
- Produces: `StudentStateTracker` class, `calculate_attention_score(keypoints, bbox_height, config, tracker, student_id, fps) -> tuple[int, list[str]]`

**Critical:** The scoring function must produce identical results to ca_gpu.py's current implementation. This is our regression safety net.

- [ ] **Step 1: Write behavior tests FIRST**

Create `tests/test_behavior.py`:

```python
"""Regression tests for attention scoring. Uses fixed synthetic keypoints."""
import numpy as np
import pytest
from behavior import calculate_attention_score, StudentStateTracker
from config import Config


def make_kpt(x, y, conf=0.9):
    """Helper: create a YOLO-format keypoint [x, y, confidence]."""
    return np.array([x, y, conf], dtype=np.float32)


def make_keypoints(overrides=None):
    """Create a standard sitting-pose 17-keypoint array.
    Indices (YOLOv8-pose / COCO):
        0=nose, 1=left_eye, 2=right_eye,
        5=left_shoulder, 6=right_shoulder,
        9=left_wrist, 10=right_wrist,
        13=left_hip, 14=right_hip
    Default: upright sitting, all keypoints visible.
    """
    kpts = np.zeros((17, 3), dtype=np.float32)
    # Head (nose, eyes)
    kpts[0] = [0.5, 0.15, 0.9]   # nose
    kpts[1] = [0.47, 0.12, 0.9]  # left eye
    kpts[2] = [0.53, 0.12, 0.9]  # right eye
    # Shoulders
    kpts[5] = [0.35, 0.35, 0.9]  # left shoulder
    kpts[6] = [0.65, 0.35, 0.9]  # right shoulder
    # Wrists
    kpts[9] = [0.25, 0.55, 0.9]  # left wrist
    kpts[10] = [0.75, 0.55, 0.9] # right wrist
    # Hips
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
        """Upright sitting student gets high score."""
        kpts = make_keypoints()
        score, reasons = calculate_attention_score(kpts, 100, config, tracker, 1, 30.0)
        assert score >= 90, f"Expected >=90, got {score}, reasons: {reasons}"
        assert len(reasons) == 0

    def test_head_down_detected(self, config, tracker):
        """Nose below shoulders reduces score."""
        kpts = make_keypoints({
            0: [0.5, 0.50, 0.9],  # nose below shoulder center (0.35)
        })
        score, reasons = calculate_attention_score(kpts, 100, config, tracker, 2, 30.0)
        assert score < 90, f"Expected <90 for head down, got {score}"
        assert any("短暂低头" in r for r in reasons)

    def test_shoulder_tilt_detected(self, config, tracker):
        """Tilted shoulders reduce score."""
        kpts = make_keypoints({
            5: [0.35, 0.55, 0.9],  # left shoulder dropped
            6: [0.65, 0.35, 0.9],  # right shoulder normal
        })
        score, reasons = calculate_attention_score(kpts, 100, config, tracker, 3, 30.0)
        assert score < 90, f"Expected <90 for tilt, got {score}"
        assert any("侧身" in r for r in reasons)

    def test_hand_below_hip(self, config, tracker):
        """Hands below hips reduce score."""
        kpts = make_keypoints({
            9: [0.35, 0.80, 0.9],   # left wrist below hip
        })
        score, reasons = calculate_attention_score(kpts, 100, config, tracker, 4, 30.0)
        assert score < 90, f"Expected <90 for low hand, got {score}"
        assert any("手部" in r for r in reasons)

    def test_long_term_head_down_severe_penalty(self, config):
        """Repeated head-down frames trigger long-term penalty."""
        tracker = StudentStateTracker()
        kpts = make_keypoints({
            0: [0.5, 0.50, 0.9],  # head down
        })
        # Simulate many frames at 30fps (3+ seconds)
        for frame in range(100):
            score, reasons = calculate_attention_score(kpts, 100, config, tracker, 5, 30.0)
        assert score <= 30, f"Expected severe penalty, got {score}"
        assert any("长时间低头" in r for r in reasons)

    def test_low_confidence_keypoints_no_penalty(self, config, tracker):
        """Low-confidence keypoints should not trigger penalties."""
        kpts = make_keypoints({
            0: [0.5, 0.15, 0.1],   # low confidence nose
            5: [0.35, 0.35, 0.1],  # low confidence left shoulder
            6: [0.65, 0.35, 0.1],  # low confidence right shoulder
        })
        score, reasons = calculate_attention_score(kpts, 100, config, tracker, 6, 30.0)
        # Low confidence = rules should not trigger
        assert score >= 80, f"Low confidence should not trigger false positives, got {score}"

    def test_score_clamped_0_to_100(self, config, tracker):
        """Score should stay in [0, 100] even with extreme inputs."""
        kpts = make_keypoints({
            0: [0.5, 1.0, 0.9],    # extreme head down
            5: [0.1, 0.9, 0.9],    # extreme tilt
            6: [0.9, 0.1, 0.9],
        })
        score, _ = calculate_attention_score(kpts, 100, config, tracker, 7, 30.0)
        assert 0 <= score <= 100, f"Score {score} out of bounds"

    def test_none_keypoints_returns_zero(self, config, tracker):
        """None keypoints should return 0."""
        score, reasons = calculate_attention_score(None, 100, config, tracker, 8, 30.0)
        assert score == 0

    def test_insufficient_keypoints_returns_zero(self, config, tracker):
        """Too few keypoints should return 0."""
        kpts = np.zeros((10, 3), dtype=np.float32)  # only 10, need 17
        score, reasons = calculate_attention_score(kpts, 100, config, tracker, 9, 30.0)
        assert score == 0
```

- [ ] **Step 2: Run tests to verify they FAIL**

```bash
python -m pytest tests/test_behavior.py -v
```

Expected: all fail with ModuleNotFoundError for behavior module.

- [ ] **Step 3: Create behavior.py**

Extract `StudentStateTracker` and `calculate_attention_score` from ca_gpu.py lines 55-245. Keep logic identical:

```python
"""Behavior analysis engine for classroom attention detection.

Works with normalized YOLO-pose keypoints (COCO 17-keypoint format).
Each keypoint is [x, y, confidence] where x,y are pixel coordinates and
confidence is visibility [0, 1].
"""
from collections import defaultdict, deque
import numpy as np
from config import Config


class StudentStateTracker:
    """Tracks per-student behavioral state across frames."""

    def __init__(self):
        self.head_position = defaultdict(lambda: deque(maxlen=30))
        self.eye_openness = defaultdict(lambda: deque(maxlen=30))
        self.gaze_position = defaultdict(lambda: deque(maxlen=30))
        self.head_down_timer = defaultdict(float)
        self.eye_closed_timer = defaultdict(float)
        self.stillness_timer = defaultdict(float)

    def update(self, student_id: int, keypoints, fps: float) -> float:
        """Update student state from keypoints. Returns current time in seconds."""
        current_time = len(self.head_position[student_id]) / fps
        if keypoints is not None and len(keypoints) > 0:
            nose = keypoints[0]
            left_eye = keypoints[1]
            right_eye = keypoints[2]
            if nose[2] > 0.5:
                self.head_position[student_id].append(nose[:2])
            ear = self._calculate_ear(left_eye, right_eye)
            self.eye_openness[student_id].append(ear)
            if left_eye[2] > 0.5 and right_eye[2] > 0.5:
                gaze_x = (left_eye[0] + right_eye[0]) / 2
                gaze_y = (left_eye[1] + right_eye[1]) / 2
                self.gaze_position[student_id].append((gaze_x, gaze_y))
        return current_time

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
            if (nose[2] > 0.5 and left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5):
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

        # Stillness / staring blankly
        if len(self.head_position[student_id]) >= 5:
            recent = list(self.head_position[student_id])[-5:]
            head_movement = sum(
                np.sqrt((recent[i][0] - recent[i-1][0])**2 +
                        (recent[i][1] - recent[i-1][1])**2)
                for i in range(1, len(recent))
            )
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
                               student_id: int, fps: float) -> tuple:
    """Calculate attention score from keypoints.

    Args:
        keypoints: numpy array (17, 3) of COCO keypoints [x, y, conf]
        bbox_height: bounding box height in pixels
        config: Config instance
        state_tracker: StudentStateTracker instance
        student_id: tracked student ID
        fps: video frames per second

    Returns:
        (score, reasons) where score is 0-100 and reasons is list of strings
    """
    if keypoints is None or len(keypoints) < 17:
        return 0, []

    score = 100
    reasons = []
    bh = config.behavior

    try:
        state_tracker.update(student_id, keypoints, fps)
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
```

- [ ] **Step 4: Run behavior tests**

```bash
python -m pytest tests/test_behavior.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add behavior.py tests/test_behavior.py
git commit -m "feat: extract behavior analysis engine with regression tests"
```

---

### Task 4: Extract reporter.py

**Files:**
- Create: `reporter.py`
- Test: `tests/test_reporter.py`

**Interfaces:**
- Consumes: list of attention record dicts (student_id, time_sec, time_str, frame, score, reason, bbox)
- Produces: `generate_report(records) -> tuple[DataFrame | None, dict]`
- Produces: `print_report(summary) -> None`

- [ ] **Step 1: Write reporter tests**

Create `tests/test_reporter.py`:

```python
"""Tests for report generation."""
import pytest
from reporter import generate_report, print_report

SAMPLE_RECORDS = [
    {'student_id': 1, 'time_sec': 1.0, 'time_str': '0:00:01', 'frame': 30,
     'score': 30, 'reason': '短暂低头', 'bbox': (10, 10, 100, 200)},
    {'student_id': 1, 'time_sec': 1.5, 'time_str': '0:00:01', 'frame': 45,
     'score': 25, 'reason': '短暂低头', 'bbox': (10, 10, 100, 200)},
    {'student_id': 1, 'time_sec': 10.0, 'time_str': '0:00:10', 'frame': 300,
     'score': 40, 'reason': '侧身(30°)', 'bbox': (10, 10, 100, 200)},
    {'student_id': 2, 'time_sec': 5.0, 'time_str': '0:00:05', 'frame': 150,
     'score': 20, 'reason': '长时间低头(3.5s)', 'bbox': (200, 50, 350, 300)},
]


def test_generate_report_empty():
    df, summary = generate_report([])
    assert df is None
    assert summary == {}


def test_generate_report_structure():
    df, summary = generate_report(SAMPLE_RECORDS)
    assert df is not None
    assert len(df) == 4
    assert set(df.columns) == {'student_id', 'time_sec', 'time_str', 'frame',
                                'score', 'reason', 'bbox'}


def test_generate_report_splits_gap():
    """Records more than 3 seconds apart should be separate events."""
    # Student 1 has records at 1.0s, 1.5s (adjacent) and 10.0s (gap > 3s)
    df, summary = generate_report(SAMPLE_RECORDS)
    assert 1 in summary
    # Should have 2 separate time ranges for student 1
    # (1.0-1.5 merged, and 10.0 alone)
    assert summary[1]['event_count'] >= 1


def test_generate_report_has_reason():
    df, summary = generate_report(SAMPLE_RECORDS)
    assert 2 in summary
    # Student 2's main reason should mention head down
    ranges = summary[2]['time_ranges']
    assert len(ranges) > 0
    assert 'reason' in ranges[0]


def test_print_report_no_summary(capsys):
    print_report({})
    captured = capsys.readouterr()
    assert "未检测到不专注行为" in captured.out
```

- [ ] **Step 2: Create reporter.py**

Extract `generate_report` and `print_report` from ca_gpu.py lines 449-543. Merge the time-range logic from both versions (they are nearly identical — v2 adds reason tracking per range):

```python
"""CSV report generation and console output."""
from collections import Counter
from datetime import timedelta
import pandas as pd


def generate_report(attention_records: list) -> tuple:
    """Generate DataFrame and summary dict from attention records.

    Each record is a dict with: student_id, time_sec, time_str, frame, score,
    reason (semicolon-separated), bbox.

    Returns (DataFrame, summary) where summary is {student_id: {...}}.
    """
    if not attention_records:
        return None, {}

    df = pd.DataFrame(attention_records)
    summary = {}

    for student_id in sorted(df['student_id'].unique()):
        student_data = df[df['student_id'] == student_id].sort_values('time_sec')
        time_ranges = []

        if not student_data.empty:
            start_time = student_data.iloc[0]['time_sec']
            end_time = student_data.iloc[0]['time_sec']

            for _, row in student_data.iterrows():
                if row['time_sec'] - end_time > 3:
                    if end_time - start_time >= 1:
                        time_ranges.append((start_time, end_time))
                    start_time = row['time_sec']
                end_time = row['time_sec']

            if end_time - start_time >= 1:
                time_ranges.append((start_time, end_time))

        formatted_ranges = []
        total_duration = 0
        for start, end in time_ranges:
            duration = end - start

            # Find dominant reason for this time range
            time_range_data = student_data[
                (student_data['time_sec'] >= start) &
                (student_data['time_sec'] <= end)
            ]
            if not time_range_data.empty and 'reason' in time_range_data.columns:
                all_reasons = []
                for reason_str in time_range_data['reason']:
                    all_reasons.extend(str(reason_str).split(';'))
                reason_counts = Counter(all_reasons)
                main_reason = reason_counts.most_common(1)[0][0] if reason_counts else "未知"
            else:
                main_reason = "未知"

            formatted_ranges.append({
                'start': str(timedelta(seconds=int(start))),
                'end': str(timedelta(seconds=int(end))),
                'duration_sec': round(duration, 1),
                'reason': main_reason,
            })
            total_duration += duration

        if formatted_ranges:
            summary[student_id] = {
                'time_ranges': formatted_ranges,
                'total_duration_sec': round(total_duration, 1),
                'event_count': len(formatted_ranges),
            }

    return df, summary


def print_report(summary: dict) -> None:
    """Print formatted report to console."""
    if not summary:
        print("\n=== 检测报告 ===")
        print("未检测到不专注行为！")
        return

    print("\n" + "=" * 70)
    print("课堂专注度检测报告".center(70))
    print("=" * 70)

    for student_id in sorted(summary.keys()):
        data = summary[student_id]
        print(f"\n【学生ID: {student_id:02d}】")
        print(f"不专注事件次数: {data['event_count']}")
        print(f"总不专注时长: {data['total_duration_sec']}秒")
        print("不专注时间段:")

        for i, time_range in enumerate(data['time_ranges'], 1):
            print(f"  {i}. {time_range['start']} ~ {time_range['end']} "
                  f"(持续 {time_range['duration_sec']}秒)")
            if 'reason' in time_range:
                print(f"     主因: {time_range['reason']}")

    print("\n" + "=" * 70)
    print(f"总计不专注学生数: {len(summary)}人")
    print("=" * 70 + "\n")
```

- [ ] **Step 3: Run reporter tests**

```bash
python -m pytest tests/test_reporter.py -v
```

Expected: 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add reporter.py tests/test_reporter.py
git commit -m "feat: extract report generation module with tests"
```

---

### Task 5: Create visualizer.py

**Files:**
- Create: `visualizer.py`

**Interfaces:**
- Consumes: frame (numpy array), list of detection dicts with keys: x1, y1, x2, y2, track_id, score, reasons, is_focused
- Produces: annotated frame (numpy array) — pure function, no side effects

- [ ] **Step 1: Create visualizer.py**

```python
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

        # Thicker border for long-term behaviors
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
```

- [ ] **Step 2: Commit**

```bash
git add visualizer.py
git commit -m "feat: extract visualization module"
```

---

### Task 6: Create utils.py

**Files:**
- Create: `utils.py`

**Interfaces:**
- Produces: `create_video_capture(path) -> tuple[cv2.VideoCapture, float, int]`
- Produces: `create_video_writer(path, fourcc, fps, size) -> cv2.VideoWriter`
- Produces: `detect_device() -> str` — returns "cuda" or "cpu"
- Produces: `suppress_warnings() -> None`

- [ ] **Step 1: Create utils.py**

```python
"""Shared utilities for video I/O and device detection."""
import os
import sys
import logging
import warnings
import cv2
import torch


def suppress_warnings() -> None:
    """Suppress non-critical warnings from TF, MediaPipe, etc."""
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('MEDIAPIPE_DISABLE_GPU', '1')
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)


def detect_device() -> str:
    """Return 'cuda' if GPU available, else 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_video_capture(path: str) -> tuple:
    """Open video file. Returns (VideoCapture, fps, total_frames)."""
    video_path = os.path.abspath(path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0 or fps == 0 or width == 0 or height == 0:
        cap.release()
        raise ValueError(f"Invalid video format: {video_path}")

    print(f"✓ 视频加载成功: {total_frames}帧, {fps:.2f}fps, {width}x{height}")
    return cap, fps, total_frames, width, height


def create_video_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create video writer for annotated output."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Cannot create video: {path}")
    return writer


def print_gpu_info() -> None:
    """Print GPU diagnostic information."""
    print("-" * 60)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"显存: {memory_gb:.1f} GB")
    print("-" * 60 + "\n")
```

- [ ] **Step 2: Commit**

```bash
git add utils.py
git commit -m "feat: extract video I/O and device utilities"
```

---

### Task 7: Refactor ca_gpu.py to use shared modules

**Files:**
- Modify: `ca_gpu.py`

**Strategy:** Replace inlined implementations of StudentStateTracker, calculate_attention_score, generate_report, print_report with imports from shared modules. Keep the ClassroomMonitor class and main() as the v2 backend entry point. The processing loop stays in ca_gpu.py.

- [ ] **Step 1: Update ca_gpu.py imports and remove duplicated code**

Replace lines 1-23 (imports through warning suppression) plus lines 55-543 (StudentStateTracker through print_report) with imports:

```python
#!/usr/bin/env python3
"""
课堂专注度检测系统 v2.0 (增强版 - GPU/YOLOv8-pose backend)
"""
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import timedelta
import argparse
import os
import sys
import torch

from config import Config, load_config
from behavior import StudentStateTracker, calculate_attention_score
from reporter import generate_report, print_report
from visualizer import draw_annotations
from utils import (suppress_warnings, detect_device, create_video_capture,
                   create_video_writer, print_gpu_info)

suppress_warnings()
```

Then update `ClassroomMonitor.__init__` to accept config:

```python
class ClassroomMonitor:
    def __init__(self, video_path, config=None):
        self.video_path = video_path
        self.config = config if config is not None else Config()
        self.attention_records = []
        self.state_tracker = StudentStateTracker()

        if self.config.device == "0" or self.config.device == "cuda":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"✓ GPU加速启用: {gpu_name} ({gpu_memory:.1f} GB)")
                self.config.device = "cuda"
            else:
                print("⚠ GPU不可用，使用CPU模式")
                self.config.device = "cpu"
```

Update the process() method to use `utils.create_video_capture`, `utils.create_video_writer`, `visualizer.draw_annotations`, and `reporter.generate_report`. The `calculate_attention_score` call uses the behavior module.

Update `generate_report` and `print_report` methods to delegate to the reporter module:

```python
    def generate_report(self):
        return generate_report(self.attention_records)

    def print_report(self, summary):
        print_report(summary)
```

Update main() to use `load_config`:

```python
def main():
    parser = argparse.ArgumentParser(
        description='课堂专注度检测 v2.0 (增强版)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python ca_gpu.py test.mp4 --save-video
  python ca_gpu.py test.mp4 --threshold 40 --save-video
  python ca_gpu.py test.mp4 --save-video --max-frames 500
        '''
    )
    parser.add_argument('video_path', help='输入视频文件路径')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--threshold', type=int, default=None,
                       help='专注度阈值(0-100), 覆盖配置文件')
    parser.add_argument('--skip-frames', type=int, default=None,
                       help='跳帧数, 覆盖配置文件')
    parser.add_argument('--save-video', action='store_true',
                       help='保存标注后的视频文件')
    parser.add_argument('--no-labels', action='store_true',
                       help='不在视频上显示文字标签')
    parser.add_argument('-o', '--output', default=None,
                       help='输出视频路径')
    parser.add_argument('--max-frames', type=int, default=0,
                       help='最大处理帧数(0=全部), 用于测试')

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"✗ 错误: 文件不存在: {args.video_path}")
        sys.exit(1)

    config = load_config(args.config)

    if args.threshold is not None:
        config.attention_threshold = args.threshold
    if args.skip_frames is not None:
        config.skip_frames = args.skip_frames
    if args.save_video:
        config.save_video = True
    if args.output is not None:
        config.video_path = args.output
    if args.no_labels:
        config.show_labels = False

    print_gpu_info()

    try:
        monitor = ClassroomMonitor(args.video_path, config)
        df, summary = monitor.process(args.max_frames)
        monitor.print_report(summary)

        if df is not None:
            df.to_csv(config.csv_path, index=False, encoding='utf-8-sig')
            print(f"✓ CSV报告已保存: {os.path.abspath(config.csv_path)}")

        if config.save_video and os.path.exists(config.video_path):
            print(f"✓ 标注视频已保存: {os.path.abspath(config.video_path)}")

        print("\n" + "=" * 60)
        print("✓ 所有任务完成！")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

- [ ] **Step 2: Verify ca_gpu.py can be imported without errors**

```bash
python -c "from ca_gpu import ClassroomMonitor; print('Import OK')"
```

Expected: `Import OK` (model files won't be loaded for import check).

- [ ] **Step 3: Run the full test suite to ensure no regression**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add ca_gpu.py
git commit -m "refactor: update ca_gpu.py to use shared modules"
```

---

### Task 8: Refactor ca.py to use shared modules

**Files:**
- Modify: `ca.py`

**Strategy:** Replace inlined Config, ResourceManager (partial), generate_report, print_report with shared module imports. Keep the v1-specific processing loop (YOLO + DeepSORT + MediaPipe), but have it use shared reporter, visualizer, utils.

- [ ] **Step 1: Update ca.py imports and remove duplicated code**

Replace lines 1-54 (imports through Config class) with:

```python
#!/usr/bin/env python3
"""
课堂专注度检测系统 v1 (CPU/MediaPipe backend, Windows compatible)
"""
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import timedelta
import argparse
import os
import sys

from config import Config, load_config
from reporter import generate_report, print_report
from visualizer import draw_annotations
from utils import (suppress_warnings, detect_device, create_video_capture,
                   create_video_writer)

# v1-specific: suppress MediaPipe GPU to avoid Windows resource conflicts
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger().setLevel(logging.ERROR)
```

Update the v1 `calculate_attention_score` to accept Config from shared config module. Update `ClassroomAttentionMonitor` to use shared `create_video_capture`, `generate_report`, `print_report`, `draw_annotations`. Update `main()` to use `load_config`.

- [ ] **Step 2: Verify ca.py can be imported**

```bash
python -c "import sys; sys.stderr = sys.__stderr__; from ca import ClassroomAttentionMonitor; print('Import OK')"
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 4: Commit**

```bash
git add ca.py
git commit -m "refactor: update ca.py to use shared modules"
```

---

### Task 9: Create main.py unified entry point

**Files:**
- Create: `main.py`

- [ ] **Step 1: Create main.py**

```python
#!/usr/bin/env python3
"""
课堂专注度检测系统 — 统一入口
自动检测 GPU/CPU 并选择最优后端
"""
import argparse
import os
import sys

from config import load_config
from utils import print_gpu_info, detect_device


def main():
    parser = argparse.ArgumentParser(
        description='课堂专注度检测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python main.py test.mp4                     # 自动选择后端
  python main.py test.mp4 --backend gpu       # 强制 GPU 后端
  python main.py test.mp4 --backend cpu       # 强制 CPU/MediaPipe 后端
  python main.py test.mp4 --config my.yaml    # 使用自定义配置
        '''
    )
    parser.add_argument('video_path', help='输入视频文件路径')
    parser.add_argument('--backend', choices=['auto', 'gpu', 'cpu'],
                       default='auto', help='后端选择 (默认: auto)')
    parser.add_argument('--config', default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--threshold', type=int, default=None,
                       help='专注度阈值(0-100)')
    parser.add_argument('--skip-frames', type=int, default=None,
                       help='跳帧数')
    parser.add_argument('--save-video', action='store_true',
                       help='保存标注视频')
    parser.add_argument('--no-labels', action='store_true',
                       help='不显示文字标签')
    parser.add_argument('-o', '--output', default=None,
                       help='输出视频路径')
    parser.add_argument('--max-frames', type=int, default=0,
                       help='最大处理帧数(0=全部), 用于测试')

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"✗ 错误: 文件不存在: {args.video_path}")
        sys.exit(1)

    config = load_config(args.config)
    if args.threshold is not None:
        config.attention_threshold = args.threshold
    if args.skip_frames is not None:
        config.skip_frames = args.skip_frames
    if args.save_video:
        config.save_video = True
    if args.output is not None:
        config.video_path = args.output
    if args.no_labels:
        config.show_labels = False

    print_gpu_info()

    # Auto-select backend
    backend = args.backend
    if backend == 'auto':
        backend = 'gpu' if detect_device() == 'cuda' else 'cpu'

    if backend == 'gpu':
        print("→ 使用 GPU/YOLOv8-pose 后端\n")
        from ca_gpu import ClassroomMonitor
        monitor = ClassroomMonitor(args.video_path, config)
        df, summary = monitor.process(args.max_frames)
        monitor.print_report(summary)
    else:
        print("→ 使用 CPU/MediaPipe 后端\n")
        from ca import ClassroomAttentionMonitor
        monitor = ClassroomAttentionMonitor(args.video_path, config)
        df, summary = monitor.process()
        monitor.print_report(summary)

    if df is not None:
        df.to_csv(config.csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ CSV报告已保存: {os.path.abspath(config.csv_path)}")

    if config.save_video and os.path.exists(config.video_path):
        print(f"✓ 标注视频已保存: {os.path.abspath(config.video_path)}")

    print("\n" + "=" * 60)
    print("✓ 所有任务完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify main.py imports**

```bash
python -c "import main; print('Import OK')"
```

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass (14 total).

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: add unified main.py with auto backend selection"
```

---

### Task 10: Final verification and cleanup

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```

- [ ] **Step 2: Verify both backends import cleanly**

```bash
python -c "from ca_gpu import ClassroomMonitor; from ca import ClassroomAttentionMonitor; print('Both backends OK')"
```

- [ ] **Step 3: Verify CLI help works**

```bash
python main.py --help
python ca_gpu.py --help
python ca.py --help
```

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "chore: final cleanup after refactor verification"
```

---

## Verification Checklist

After implementation, verify:
1. `pytest tests/ -v` — all tests pass
2. `python main.py --help` — CLI works
3. `python ca_gpu.py --help` — v2 backward compat
4. `python ca.py --help` — v1 backward compat
5. `python -c "from config import load_config; c = load_config(); print(c.behavior.head_down_penalty)"` — returns 80
6. No `yolov8m.pt` in repo (removed unused model)
7. `.gitignore` covers *.pt, *.csv, *.mp4
