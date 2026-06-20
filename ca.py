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
from utils import suppress_warnings, create_video_capture, create_camera_capture
from behavior import (
    StudentStateTracker, calculate_attention_score,
    mediapipe_landmarks_to_coco_keypoints
)
from chat_detector import detect_chat
from eye_detector import EyeDetector

# v1-specific: suppress MediaPipe/TF warnings on Windows
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)


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
    def create_pose_landmarker(config):
        """Create MediaPipe PoseLandmarker (tasks API, mediapipe>=0.10)."""
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options as base

            options = vision.PoseLandmarkerOptions(
                base_options=base.BaseOptions(
                    model_asset_path='pose_landmarker_lite.task'),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=config.min_detection_confidence,
                min_pose_presence_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )
            landmarker = vision.PoseLandmarker.create_from_options(options)
            print("✓ MediaPipe PoseLandmarker 初始化成功")
            return landmarker
        except Exception as e:
            print(f"✗ MediaPipe PoseLandmarker 初始化失败: {e}")
            return None


class ClassroomAttentionMonitor:
    """v1 classroom attention monitor (YOLO + DeepSORT + MediaPipe)."""

    def __init__(self, video_path=None, config=None):
        self.video_path = video_path  # None for camera mode
        self.config = config if config is not None else Config()
        self.is_camera = self.config.camera_id is not None
        self.attention_records = []
        try:
            self.eye_detector = EyeDetector()
            print("✓ MediaPipe Face Mesh 初始化成功 (真实闭眼检测)")
        except Exception as e:
            self.eye_detector = None
            print(f"⚠ Face Mesh 不可用，使用置信度代理: {e}")
        self.state_tracker = StudentStateTracker(
            eye_detector=self.eye_detector)

    def process(self, output_path="output_annotated.mp4"):
        print("\n" + "=" * 60)
        print("课堂专注度检测系统启动".center(60))
        print("=" * 60 + "\n")

        # Suppress MediaPipe stderr noise during processing
        _stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

        print("步骤1: 初始化模型资源...")
        yolo = ResourceManager.create_yolo(self.config)
        tracker = ResourceManager.create_tracker(self.config)
        pose_landmarker = ResourceManager.create_pose_landmarker(self.config)

        if self.is_camera:
            cap, fps, width, height = create_camera_capture(self.config.camera_id)
            total_frames = float('inf')
        else:
            cap, fps, total_frames, width, height = create_video_capture(
                self.video_path)

        if None in [yolo, tracker, cap, pose_landmarker]:
            print("\n✗ 关键资源初始化失败，程序退出")
            return None, None

        print("\n步骤2: 开始视频处理...")
        if self.is_camera:
            print("提示: 按 ESC 键退出\n")
        else:
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

                    rgb_crop = cv2.cvtColor(student_crop, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                    result = pose_landmarker.detect(mp_image)

                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks[0]
                        kpts = mediapipe_landmarks_to_coco_keypoints(landmarks)
                        # Convert normalized coords [0-1] to pixel coords
                        crop_w, crop_h = x2 - x1, y2 - y1
                        kpts[:, 0] *= crop_w
                        kpts[:, 1] *= crop_h
                        bbox_height = y2 - y1
                        score, reasons = calculate_attention_score(
                            kpts, bbox_height, self.config,
                            self.state_tracker, int(track_id), fps,
                            face_crop=student_crop
                        )

                        is_not_focused = score < self.config.attention_threshold
                        viz_detections.append({
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'track_id': track_id,
                            'score': score,
                            'reasons': reasons,
                            'is_focused': not is_not_focused,
                            'keypoints': kpts,
                        })

                        if is_not_focused:
                            self.attention_records.append({
                                'student_id': int(track_id),
                                'time_sec': round(frame_idx / fps, 2),
                                'time_str': str(timedelta(seconds=int(frame_idx / fps))),
                                'frame': frame_idx,
                                'score': score,
                                'reason': ';'.join(reasons),
                                'bbox': (x1, y1, x2, y2)
                            })

                chat_pairs = detect_chat(viz_detections, self.config)
                for sid_a, sid_b in chat_pairs:
                    for det in viz_detections:
                        if det['track_id'] in (sid_a, sid_b):
                            det['score'] = max(0, det['score'] -
                                               self.config.behavior.chat_penalty)
                            if '聊天' not in det['reasons']:
                                det['reasons'].append('聊天')
                            det['is_focused'] = det['score'] >= self.config.attention_threshold

                draw_annotations(frame, viz_detections, True)

                if video_writer:
                    video_writer.write(frame)

                if self.is_camera:
                    cv2.imshow('Classroom Attention (ESC to exit)', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        print("\n\n用户按 ESC 退出...")
                        break
                    cv2.setWindowTitle(
                        'Classroom Attention (ESC to exit)',
                        f'Classroom Attention - Frame {frame_idx}')
                elif frame_idx % 100 == 0:
                    print(f"  已处理 {frame_idx}/{total_frames} 帧...", end='\r')

                frame_idx += 1

        except KeyboardInterrupt:
            print("\n\n用户中断处理，正在保存已有结果...")

        except Exception as e:
            print(f"\n处理出错: {e}")
            import traceback
            traceback.print_exc()

        finally:
            sys.stderr = _stderr
            print("\n步骤3: 释放资源...")
            if 'cap' in locals():
                cap.release()
            if video_writer:
                video_writer.release()
            if self.is_camera:
                cv2.destroyAllWindows()
            if self.eye_detector is not None:
                self.eye_detector.close()
            print("✓ 资源已释放\n")

        return self.generate_report()

    def generate_report(self):
        return generate_report(self.attention_records)

    def print_report(self, summary):
        print_report(summary)


def main():
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
        print(f"\n程序出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
