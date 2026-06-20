"""Tests for visualization module."""
import cv2
import numpy as np
from visualizer import draw_annotations


def make_frame(width=640, height=480):
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_draw_annotations_no_detections():
    """Frame unchanged with empty detections."""
    frame = make_frame()
    before = frame.copy()
    draw_annotations(frame, [], show_labels=True)
    assert np.array_equal(frame, before)


def test_draw_annotations_focused_student():
    """Green box for focused student."""
    frame = make_frame()
    dets = [{'x1': 50, 'y1': 50, 'x2': 150, 'y2': 250,
             'track_id': 1, 'score': 85, 'reasons': [],
             'is_focused': True}]
    draw_annotations(frame, dets, show_labels=True)
    # Check right edge midpoint for green border
    assert np.any(np.all(frame[50:250, 149] == [0, 255, 0], axis=1))


def test_draw_annotations_not_focused_student():
    """Red box for not-focused student."""
    frame = make_frame()
    dets = [{'x1': 50, 'y1': 50, 'x2': 150, 'y2': 250,
             'track_id': 2, 'score': 35, 'reasons': ['短暂低头'],
             'is_focused': False}]
    draw_annotations(frame, dets, show_labels=True)
    # Check right edge midpoint for red border
    assert np.any(np.all(frame[50:250, 149] == [0, 0, 255], axis=1))


def test_draw_annotations_long_term_thick_border():
    """Long-term behavior gets thicker border (4px vs 2px)."""
    frame = make_frame()
    dets = [{'x1': 50, 'y1': 50, 'x2': 150, 'y2': 250,
             'track_id': 3, 'score': 10, 'reasons': ['长时间低头(3.5s)'],
             'is_focused': False}]
    draw_annotations(frame, dets, show_labels=True)
    # 4px border: pixel 3px inside left edge should still be red
    assert np.any(np.all(frame[50:250, 53] == [0, 0, 255], axis=1))


def test_draw_annotations_multiple_students():
    """Multiple detections all drawn."""
    frame = make_frame()
    dets = [
        {'x1': 50, 'y1': 50, 'x2': 150, 'y2': 250, 'track_id': 1,
         'score': 85, 'reasons': [], 'is_focused': True},
        {'x1': 200, 'y1': 100, 'x2': 350, 'y2': 300, 'track_id': 2,
         'score': 30, 'reasons': ['闭眼(2.5s)'], 'is_focused': False},
    ]
    draw_annotations(frame, dets, show_labels=True)
    # Both green and red should exist
    assert np.any(np.all(frame[50:250, 149] == [0, 255, 0], axis=1))
    assert np.any(np.all(frame[100:300, 349] == [0, 0, 255], axis=1))
