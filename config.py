"""Configuration loader from YAML file."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class BehaviorConfig:
    """Tunable thresholds for all behavior detection rules."""
    head_down_threshold: float = 0.08
    head_down_duration: float = 3.0
    head_down_penalty: int = 80
    eye_closed_threshold: float = 0.18
    eye_closed_duration: float = 2.0
    eye_closed_penalty: int = 70
    stillness_threshold: float = 25.0
    stillness_duration: float = 4.0
    stillness_penalty: int = 50
    shoulder_tilt_threshold: int = 25
    shoulder_tilt_penalty: int = 20
    hand_below_hip_threshold: float = 0.02
    hand_below_hip_penalty: int = 15
    short_head_down_penalty: int = 30
    hand_raise_enabled: bool = True
    chat_enabled: bool = False
    chat_distance_threshold: float = 0.15
    chat_angle_threshold: float = 40.0
    chat_penalty: int = 15


@dataclass
class Config:
    """System configuration with sensible defaults."""
    # Model
    pose_model: str = "yolov8m-pose.pt"
    yolo_model: str = "yolo11m.pt"
    device: str = "0"

    # Behavior thresholds
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
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Output
    save_video: bool = True
    video_path: str = "output_annotated.mp4"
    show_labels: bool = True
    save_csv: bool = True
    csv_path: str = "attention_report.csv"

    # Camera
    camera_id: int = None  # None = video file, int = camera device ID

    # Performance
    skip_frames: int = 2


def load_config(path: str = "config.yaml") -> Config:
    """Load configuration from YAML file with fallback to defaults."""
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
        if "hand_raise" in b:
            hr = b["hand_raise"]
            bh.hand_raise_enabled = hr.get("enabled", bh.hand_raise_enabled)
        if "chat" in b:
            ch = b["chat"]
            bh.chat_enabled = ch.get("enabled", bh.chat_enabled)
            bh.chat_distance_threshold = ch.get("distance_threshold", bh.chat_distance_threshold)
            bh.chat_angle_threshold = ch.get("angle_threshold", bh.chat_angle_threshold)
            bh.chat_penalty = ch.get("penalty", bh.chat_penalty)

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
