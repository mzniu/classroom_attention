"""CSV report generation and console output."""
from collections import Counter
from datetime import timedelta
import pandas as pd


def generate_report(attention_records: list) -> tuple:
    """Generate DataFrame and summary dict from attention records.

    Each record dict has: student_id, time_sec, time_str, frame, score,
    reason (semicolon-separated), bbox.

    Returns (DataFrame | None, summary dict).
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
