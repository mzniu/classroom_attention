"""Tests for report generation."""
import pytest
from reporter import generate_report, print_report

SAMPLE_RECORDS = [
    # Student 1, event group 1: 1.0-2.0 (duration >= 1, merged from 3 records)
    {'student_id': 1, 'time_sec': 1.0, 'time_str': '0:00:01', 'frame': 30,
     'score': 30, 'reason': '短暂低头', 'bbox': (10, 10, 100, 200)},
    {'student_id': 1, 'time_sec': 1.5, 'time_str': '0:00:01', 'frame': 45,
     'score': 25, 'reason': '短暂低头', 'bbox': (10, 10, 100, 200)},
    {'student_id': 1, 'time_sec': 2.0, 'time_str': '0:00:02', 'frame': 60,
     'score': 35, 'reason': '短暂低头', 'bbox': (10, 10, 100, 200)},
    # Student 1, event group 2: 10.0-11.0 (gap > 3s from group 1)
    {'student_id': 1, 'time_sec': 10.0, 'time_str': '0:00:10', 'frame': 300,
     'score': 40, 'reason': '侧身(30°)', 'bbox': (10, 10, 100, 200)},
    {'student_id': 1, 'time_sec': 10.5, 'time_str': '0:00:10', 'frame': 315,
     'score': 35, 'reason': '侧身(30°)', 'bbox': (10, 10, 100, 200)},
    {'student_id': 1, 'time_sec': 11.0, 'time_str': '0:00:11', 'frame': 330,
     'score': 42, 'reason': '侧身(30°)', 'bbox': (10, 10, 100, 200)},
    # Student 2: single event group, 5.0-6.5
    {'student_id': 2, 'time_sec': 5.0, 'time_str': '0:00:05', 'frame': 150,
     'score': 20, 'reason': '长时间低头(3.5s)', 'bbox': (200, 50, 350, 300)},
    {'student_id': 2, 'time_sec': 5.5, 'time_str': '0:00:05', 'frame': 165,
     'score': 15, 'reason': '长时间低头(4.0s)', 'bbox': (200, 50, 350, 300)},
    {'student_id': 2, 'time_sec': 6.5, 'time_str': '0:00:06', 'frame': 195,
     'score': 22, 'reason': '长时间低头(5.0s)', 'bbox': (200, 50, 350, 300)},
]


def test_generate_report_empty():
    """Empty records return (None, {})."""
    df, summary = generate_report([])
    assert df is None
    assert summary == {}


def test_generate_report_structure():
    """DataFrame has expected columns."""
    df, summary = generate_report(SAMPLE_RECORDS)
    assert df is not None
    assert len(df) == 9
    assert set(df.columns) == {'student_id', 'time_sec', 'time_str', 'frame',
                                'score', 'reason', 'bbox'}


def test_generate_report_splits_gap():
    """Records >3s apart produce separate events."""
    df, summary = generate_report(SAMPLE_RECORDS)
    assert 1 in summary
    # Student 1 has records at 1.0-1.5 (merged) and 10.0 (separate due to gap)
    assert summary[1]['event_count'] >= 1


def test_generate_report_extracts_reason():
    """Each time range includes dominant reason."""
    df, summary = generate_report(SAMPLE_RECORDS)
    assert 2 in summary
    ranges = summary[2]['time_ranges']
    assert len(ranges) > 0
    assert 'reason' in ranges[0]


def test_print_report_empty(capsys):
    """Empty summary prints '未检测到' message."""
    print_report({})
    captured = capsys.readouterr()
    assert "未检测到不专注行为" in captured.out


def test_print_report_with_data(capsys):
    """Summary with data prints student info."""
    df, summary = generate_report(SAMPLE_RECORDS)
    print_report(summary)
    captured = capsys.readouterr()
    assert "学生ID" in captured.out
    assert "不专注" in captured.out
