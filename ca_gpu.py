#!/usr/bin/env python3
"""
课堂专注度检测系统 v2.0 (GPU/YOLOv8-pose backend)
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
                   create_camera_capture, create_video_writer, print_gpu_info)

suppress_warnings()


class ClassroomMonitor:
    def __init__(self, video_path=None, config=None):
        self.video_path = video_path  # None for camera mode
        self.config = config if config is not None else Config()
        self.is_camera = self.config.camera_id is not None
        self.attention_records = []
        self.state_tracker = StudentStateTracker()

        device = self.config.device
        if device in ("0", "cuda") and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU加速启用: {gpu_name} ({gpu_memory:.1f} GB)")
            self.config.device = "cuda"
        else:
            print("⚠ 使用CPU模式")
            self.config.device = "cpu"

    def process(self, max_frames=0):
        print("\n" + "=" * 60)
        print("课堂专注度检测系统 v2.0".center(60))
        print("=" * 60 + "\n")

        print("步骤1: 加载YOLOv8-pose模型...")
        yolo = YOLO(self.config.pose_model)
        if self.config.device == "cuda":
            yolo.to("cuda")
        print("✓ 模型加载成功\n")

        if self.is_camera:
            print("步骤2: 连接摄像头...")
            cap, fps, width, height = create_camera_capture(self.config.camera_id)
            total_frames = float('inf')
            print("✓ 摄像头连接成功\n")
        else:
            print("步骤2: 加载视频文件...")
            cap, fps, total_frames, width, height = create_video_capture(self.video_path)

        if max_frames > 0:
            total_frames = min(total_frames, max_frames)

        print(f"✓ {'摄像头' if self.is_camera else '视频'}: "
              f"{'实时' if self.is_camera else f'{total_frames}帧'}, "
              f"{fps:.2f}fps, {width}x{height}\n")

        video_writer = None
        if self.config.save_video:
            writer_fps = fps / (self.config.skip_frames + 1)
            video_writer = create_video_writer(self.config.video_path, writer_fps, width, height)
            print(f"✓ 视频输出: {os.path.abspath(self.config.video_path)}\n")

        print("步骤3: 开始GPU加速检测...")
        print("行为: 低头(短暂/长期) | 闭眼 | 发呆 | 侧身 | 手部异常")
        if self.is_camera:
            print("提示: 按 ESC 键退出\n")
        else:
            print()

        frame_idx = 0
        processed_count = 0

        try:
            while True:
                if max_frames > 0 and frame_idx >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % (self.config.skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                results = yolo.track(
                    frame,
                    classes=[0],
                    conf=self.config.confidence_threshold,
                    persist=True,
                    tracker="bytetrack.yaml",
                    device=self.config.device,
                    verbose=False
                )

                detections = []
                if results[0].boxes and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    keypoints = results[0].keypoints
                    orig_frame = results[0].orig_img

                    for i, box in enumerate(boxes):
                        track_id = int(box.id[0]) if box.id is not None else 0
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox_height = y2 - y1

                        if keypoints and i < len(keypoints.data):
                            kpts = keypoints.data[i].cpu().numpy()
                            attention_score, reasons = calculate_attention_score(
                                kpts, bbox_height, self.config,
                                self.state_tracker, track_id, fps
                            )

                            is_not_focused = attention_score < self.config.attention_threshold
                            detections.append({
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'track_id': track_id,
                                'score': attention_score,
                                'reasons': reasons,
                                'is_focused': not is_not_focused,
                            })

                            if is_not_focused:
                                self.attention_records.append({
                                    'student_id': track_id,
                                    'time_sec': round(frame_idx / fps, 2),
                                    'time_str': str(timedelta(seconds=int(frame_idx / fps))),
                                    'frame': frame_idx,
                                    'score': attention_score,
                                    'reason': ';'.join(reasons),
                                    'bbox': (x1, y1, x2, y2)
                                })

                    draw_annotations(orig_frame, detections, self.config.show_labels)
                    display_frame = orig_frame
                    if video_writer:
                        video_writer.write(orig_frame)
                else:
                    display_frame = frame
                    if video_writer:
                        video_writer.write(frame)

                if self.is_camera:
                    cv2.imshow('Classroom Attention (ESC to exit)', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        print("\n\n用户按 ESC 退出...")
                        break
                elif processed_count % 50 == 0:
                    progress = (frame_idx / total_frames) * 100
                    detected_people = len(results[0].boxes) if results[0].boxes else 0
                    not_focus_count = sum(1 for r in self.attention_records
                                          if r['frame'] == frame_idx)
                    print(f"  --> 进度: {progress:.1f}% [{frame_idx}/{total_frames}] | "
                          f"检测到: {detected_people}人 | 不专注: {not_focus_count}人")

                processed_count += 1
                frame_idx += 1

        except KeyboardInterrupt:
            print("\n\n用户中断，正在保存...")

        except Exception as e:
            print(f"\n处理出错: {e}")
            import traceback
            traceback.print_exc()

        finally:
            try:
                cap.release()
                if video_writer:
                    video_writer.release()
                    print(f"\n✓ 标注视频已保存: {os.path.abspath(self.config.video_path)}")
                if self.is_camera:
                    cv2.destroyAllWindows()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"清理资源时出错: {e}")

        return self.generate_report()

    def generate_report(self):
        return generate_report(self.attention_records)

    def print_report(self, summary):
        print_report(summary)


def main():
    parser = argparse.ArgumentParser(
        description='课堂专注度检测 v2.0 (GPU/YOLOv8-pose)',
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
    parser.add_argument('--threshold', type=int, default=None, help='专注度阈值(0-100)')
    parser.add_argument('--skip-frames', type=int, default=None, help='跳帧数')
    parser.add_argument('--save-video', action='store_true', help='保存标注后的视频文件')
    parser.add_argument('--no-labels', action='store_true', help='不在视频上显示文字标签')
    parser.add_argument('-o', '--output', default=None, help='输出视频路径')
    parser.add_argument('--max-frames', type=int, default=0, help='最大处理帧数(0=全部)')

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


if __name__ == "__main__":
    main()
