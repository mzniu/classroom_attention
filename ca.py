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
import warnings
import logging

from config import Config, load_config
from reporter import generate_report, print_report
from visualizer import draw_annotations
from utils import suppress_warnings, create_video_capture

# v1-specific: suppress stderr for MediaPipe on Windows
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)
sys.stderr = open(os.devnull, 'w')
mp_pose = mp.solutions.pose


class ResourceManager:
    """Manage model resources for v1 pipeline."""

    @staticmethod
    def create_yolo(config):
        try:
            model = YOLO(config.yolo_model)
            print("✓ YOLO模型加载成功")
            return model
        except Exception as e:
            print(f"✗ YOLO加载失败: {e}")
            return None

    @staticmethod
    def create_tracker(config):
        try:
            tracker = DeepSort(
                max_age=config.max_age,
                n_init=config.n_init,
                max_iou_distance=config.max_iou_distance
            )
            print("✓ DeepSORT跟踪器初始化成功")
            return tracker
        except Exception as e:
            print(f"✗ DeepSORT初始化失败: {e}")
            return None

    @staticmethod
    def create_pose_estimator(config):
        try:
            estimator = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=0,
                min_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence
            )
            print("✓ MediaPipe姿态估计器初始化成功")
            return estimator
        except Exception as e:
            print(f"✗ MediaPipe初始化失败: {e}")
            return None


def calculate_attention_score(landmarks, config):
    """Calculate attention score from MediaPipe landmarks."""
    if not landmarks:
        return 0

    score = 100
    try:
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        bh = config.behavior

        # Head down
        if (nose.visibility > 0.5 and left_shoulder.visibility > 0.5
                and right_shoulder.visibility > 0.5):
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            if nose.y - shoulder_center_y > bh.head_down_threshold:
                score -= 60

        # Shoulder tilt
        if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
            shoulder_vector = np.array([
                right_shoulder.x - left_shoulder.x,
                right_shoulder.y - left_shoulder.y
            ])
            angle = np.degrees(np.arctan2(abs(shoulder_vector[1]),
                                           abs(shoulder_vector[0])))
            if angle > bh.shoulder_tilt_threshold:
                score -= 30

        # Hand below hip
        hand_below_hip = False
        if (left_hand.visibility > 0.5 and left_hip.visibility > 0.5
                and left_hand.y > left_hip.y + bh.hand_below_hip_threshold):
            hand_below_hip = True
        if (right_hand.visibility > 0.5 and right_hip.visibility > 0.5
                and right_hand.y > right_hip.y + bh.hand_below_hip_threshold):
            hand_below_hip = True
        if hand_below_hip:
            score -= 20

    except Exception:
        return max(0, score)

    return max(0, min(100, score))


class ClassroomAttentionMonitor:
    """v1 classroom attention monitor (YOLO + DeepSORT + MediaPipe)."""

    def __init__(self, video_path, config=None):
        self.video_path = video_path
        self.config = config if config is not None else Config()
        self.attention_records = []

    def process(self, output_path="output_annotated.mp4"):
        print("\n" + "=" * 60)
        print("课堂专注度检测系统启动".center(60))
        print("=" * 60 + "\n")

        print("步骤1: 初始化模型资源...")
        yolo = ResourceManager.create_yolo(self.config)
        tracker = ResourceManager.create_tracker(self.config)
        cap, fps, total_frames, width, height = create_video_capture(self.video_path)

        if None in [yolo, tracker, cap]:
            print("\n✗ 关键资源初始化失败，程序退出")
            return None, None

        print("\n步骤2: 开始视频处理...")
        print("提示: 按 Ctrl+C 可安全中断\n")

        video_writer = None
        if self.config.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vid_fps = fps / (self.config.skip_frames + 1)
            video_writer = cv2.VideoWriter(output_path, fourcc, vid_fps, (width, height))

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % (self.config.skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                results = yolo(frame, classes=[self.config.person_class_id],
                              conf=self.config.confidence_threshold, verbose=False)

                dets = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    dets.append(([x1, y1, x2 - x1, y2 - y1], conf, 'student'))

                tracks = tracker.update_tracks(dets, frame=frame)

                viz_detections = []
                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id
                    bbox = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, bbox)

                    student_crop = frame[y1:y2, x1:x2]
                    if student_crop.size == 0:
                        continue

                    with mp_pose.Pose(
                        static_image_mode=True,
                        model_complexity=0,
                        min_detection_confidence=self.config.min_detection_confidence
                    ) as pose_estimator:
                        rgb_crop = cv2.cvtColor(student_crop, cv2.COLOR_BGR2RGB)
                        pose_results = pose_estimator.process(rgb_crop)

                        if pose_results and pose_results.pose_landmarks:
                            landmarks = pose_results.pose_landmarks.landmark
                            score = calculate_attention_score(landmarks, self.config)

                            is_not_focused = score < self.config.attention_threshold
                            viz_detections.append({
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'track_id': track_id,
                                'score': score,
                                'reasons': [],
                                'is_focused': not is_not_focused,
                            })

                            if is_not_focused:
                                self.attention_records.append({
                                    'student_id': int(track_id),
                                    'time_sec': round(frame_idx / fps, 2),
                                    'time_str': str(timedelta(seconds=int(frame_idx / fps))),
                                    'frame': frame_idx,
                                    'score': score,
                                    'reason': '',
                                    'bbox': (x1, y1, x2, y2)
                                })

                draw_annotations(frame, viz_detections, True)

                if video_writer:
                    video_writer.write(frame)

                if frame_idx % 100 == 0:
                    print(f"  已处理 {frame_idx}/{total_frames} 帧...", end='\r')

                frame_idx += 1

        except KeyboardInterrupt:
            print("\n\n用户中断处理，正在保存已有结果...")

        except Exception as e:
            print(f"\n处理出错: {e}")
            import traceback
            traceback.print_exc()

        finally:
            print("\n步骤3: 释放资源...")
            if 'cap' in locals():
                cap.release()
            if video_writer:
                video_writer.release()
            print("✓ 资源已释放\n")

        return self.generate_report()

    def generate_report(self):
        return generate_report(self.attention_records)

    def print_report(self, summary):
        print_report(summary)


def main():
    sys.stderr = sys.__stderr__

    parser = argparse.ArgumentParser(
        description='课堂专注度检测系统 v1 (CPU/MediaPipe)'
    )
    parser.add_argument('video_path', help='输入视频文件路径')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--threshold', type=int, default=None, help='专注度阈值(0-100)')
    parser.add_argument('--skip-frames', type=int, default=None, help='跳帧数')

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"错误: 文件不存在: {args.video_path}")
        sys.exit(1)

    config = load_config(args.config)
    if args.threshold is not None:
        config.attention_threshold = args.threshold
    if args.skip_frames is not None:
        config.skip_frames = args.skip_frames

    print("\n" + "=" * 60)
    print("课堂专注度检测系统 v1".center(60))
    print("=" * 60 + "\n")

    try:
        monitor = ClassroomAttentionMonitor(args.video_path, config)
        df, summary = monitor.process()

        sys.stderr = open(os.devnull, 'w')
        monitor.print_report(summary)

        if df is not None:
            config.csv_path = "attention_report.csv"
            df.to_csv(config.csv_path, index=False, encoding='utf-8-sig')
            print(f"✓ 报告已保存至: {os.path.abspath(config.csv_path)}")

        print("=" * 60)
        print("✓ 处理完成！")
        print("=" * 60 + "\n")

    except Exception as e:
        sys.stderr = sys.__stderr__
        print(f"\n程序出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
