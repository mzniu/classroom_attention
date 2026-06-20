"""Tests for eye detector module."""
import numpy as np
import pytest
from eye_detector import calculate_ear_from_landmarks


def make_eye_points(top_gap=0.04):
    """Create synthetic 6-point eye contour.

    Points ordered: [p1(left), p2(top-left), p3(top-right), p4(right),
                     p5(bottom-right), p6(bottom-left)]
    """
    cx, cy = 0.37, 0.42
    return np.array([
        [0.35, cy],            # p1: left corner
        [0.36, cy - top_gap],  # p2: top-left
        [0.38, cy - top_gap],  # p3: top-right
        [0.39, cy],            # p4: right corner
        [0.38, cy + top_gap],  # p5: bottom-right
        [0.36, cy + top_gap],  # p6: bottom-left
    ], dtype=np.float32)


def test_ear_open_eye():
    """Open eye (wide vertical gap) should have EAR > 0.2."""
    eye_points = make_eye_points(top_gap=0.04)
    ear = calculate_ear_from_landmarks(eye_points)
    assert ear > 0.2, f"Open eye EAR should be > 0.2, got {ear:.3f}"


def test_ear_closed_eye():
    """Closed eye (narrow vertical gap) should have EAR < 0.15."""
    eye_points = make_eye_points(top_gap=0.002)
    ear = calculate_ear_from_landmarks(eye_points)
    assert ear < 0.15, f"Closed eye EAR should be < 0.15, got {ear:.3f}"


def test_ear_completely_closed():
    """Completely closed eye (zero vertical gap) should have near-zero EAR."""
    eye_points = make_eye_points(top_gap=0.0)
    ear = calculate_ear_from_landmarks(eye_points)
    assert ear < 0.01, f"Completely closed eye EAR should be ~0, got {ear:.3f}"


def test_ear_scaling_invariant():
    """EAR should be invariant to uniform scale."""
    small = np.array([
        [0.35, 0.41], [0.355, 0.39], [0.365, 0.39],
        [0.37, 0.41], [0.365, 0.43], [0.355, 0.43],
    ], dtype=np.float32)
    ear_small = calculate_ear_from_landmarks(small)
    large = small * 100
    ear_large = calculate_ear_from_landmarks(large)
    assert abs(ear_small - ear_large) < 0.01


def test_ear_zero_horizontal_returns_one():
    """Degenerate case: zero horizontal distance returns 1.0."""
    eye_points = np.array([
        [0.35, 0.41], [0.35, 0.39], [0.35, 0.39],
        [0.35, 0.41], [0.35, 0.43], [0.35, 0.43],
    ], dtype=np.float32)
    ear = calculate_ear_from_landmarks(eye_points)
    # Horizontal distance is 0, so EAR defaults to 1.0
    assert ear == 1.0
