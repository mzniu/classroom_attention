"""Shared utilities for video I/O and device detection."""
import os
import logging
import warnings
import cv2


def _get_torch():
    """Lazy-load torch (not a hard dependency for all modules)."""
    try:
        import torch
        return torch
    except ImportError:
        return None


def suppress_warnings() -> None:
    """Suppress non-critical warnings from TF, MediaPipe, etc."""
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('MEDIAPIPE_DISABLE_GPU', '1')
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)


def detect_device() -> str:
    """Return 'cuda' if GPU available, else 'cpu'."""
    torch = _get_torch()
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_video_capture(path: str) -> tuple:
    """Open video file. Returns (VideoCapture, fps, total_frames, width, height)."""
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


def create_camera_capture(device_id: int = 0) -> tuple:
    """Open camera device. Returns (VideoCapture, fps, width, height)."""
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        # Windows: try DirectShow backend
        cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise ValueError(f"Cannot open camera device: {device_id}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps == 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        cap.release()
        raise ValueError(f"Invalid camera resolution: {width}x{height}")

    print(f"✓ 摄像头已连接: {width}x{height} @ {fps:.0f}fps")
    return cap, fps, width, height


def create_video_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create video writer for annotated output."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Cannot create video: {path}")
    return writer


def print_gpu_info() -> None:
    """Print GPU diagnostic information."""
    torch = _get_torch()
    print("-" * 60)
    if torch is not None:
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"显存: {memory_gb:.1f} GB")
    else:
        print("PyTorch: 未安装")
        print("CUDA可用: False")
    print("-" * 60 + "\n")
