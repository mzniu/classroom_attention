"""Shared utilities for video I/O and device detection."""
import os
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
