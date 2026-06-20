"""Tests for utility functions."""
import os
import numpy as np
from utils import suppress_warnings, detect_device, create_video_writer


def test_suppress_warnings_sets_env_vars():
    """suppress_warnings sets expected environment variables."""
    suppress_warnings()
    assert os.environ.get('TF_CPP_MIN_LOG_LEVEL') == '3'
    assert os.environ.get('MEDIAPIPE_DISABLE_GPU') == '1'


def test_detect_device_returns_string():
    """detect_device returns 'cuda' or 'cpu'."""
    result = detect_device()
    assert result in ('cuda', 'cpu')


def test_print_gpu_info_runs(capsys):
    """print_gpu_info runs without error."""
    from utils import print_gpu_info
    print_gpu_info()
    captured = capsys.readouterr()
    assert 'PyTorch' in captured.out or 'CUDA' in captured.out


def test_create_video_writer_creates_file(tmp_path):
    """create_video_writer creates a valid .mp4 file."""
    import cv2
    output = str(tmp_path / "test.mp4")
    writer = create_video_writer(output, fps=30.0, width=640, height=480)
    assert writer.isOpened()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    writer.write(frame)
    writer.release()
    assert os.path.exists(output)
    assert os.path.getsize(output) > 0
