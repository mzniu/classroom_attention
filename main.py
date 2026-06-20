#!/usr/bin/env python3
"""
课堂专注度检测系统 -- 统一入口
Auto-detects GPU/CPU and selects optimal backend.
"""
import argparse
import os
import sys

from config import load_config
from utils import print_gpu_info, detect_device


def main():
    parser = argparse.ArgumentParser(
        description='课堂专注度检测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python main.py test.mp4                     # 自动选择后端
  python main.py test.mp4 --backend gpu       # 强制 GPU 后端
  python main.py test.mp4 --backend cpu       # 强制 CPU/MediaPipe 后端
  python main.py test.mp4 --config my.yaml    # 使用自定义配置
  python main.py --camera 0 --backend cpu     # 实时摄像头
        '''
    )
    parser.add_argument('video_path', nargs='?', default=None,
                       help='输入视频文件路径 (使用 --camera 时可选)')
    parser.add_argument('--camera', type=int, default=None,
                       help='摄像头设备ID (例如: 0), 启用实时摄像头模式')
    parser.add_argument('--backend', choices=['auto', 'gpu', 'cpu'],
                       default='auto', help='后端选择 (默认: auto)')
    parser.add_argument('--config', default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--threshold', type=int, default=None,
                       help='专注度阈值(0-100)')
    parser.add_argument('--skip-frames', type=int, default=None,
                       help='跳帧数')
    parser.add_argument('--save-video', action='store_true',
                       help='保存标注视频')
    parser.add_argument('--no-labels', action='store_true',
                       help='不显示文字标签')
    parser.add_argument('-o', '--output', default=None,
                       help='输出视频路径')
    parser.add_argument('--max-frames', type=int, default=0,
                       help='最大处理帧数(0=全部), 用于测试')

    args = parser.parse_args()

    # Validate input source
    if args.camera is None and args.video_path is None:
        print("✗ 错误: 必须指定视频文件路径或 --camera 参数")
        sys.exit(1)
    if args.camera is not None and args.video_path is not None:
        print("✗ 错误: 不能同时指定视频文件和 --camera 参数")
        sys.exit(1)
    if args.video_path is not None and not os.path.exists(args.video_path):
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
    if args.camera is not None:
        config.camera_id = args.camera

    print_gpu_info()

    backend = args.backend
    if backend == 'auto':
        backend = 'gpu' if detect_device() == 'cuda' else 'cpu'

    input_path = args.video_path  # None when camera mode

    if backend == 'gpu':
        print("--> 使用 GPU/YOLOv8-pose 后端\n")
        from ca_gpu import ClassroomMonitor
        monitor = ClassroomMonitor(input_path, config)
        df, summary = monitor.process(args.max_frames)
        monitor.print_report(summary)
    else:
        print("--> 使用 CPU/MediaPipe 后端\n")
        from ca import ClassroomAttentionMonitor
        monitor = ClassroomAttentionMonitor(input_path, config)
        df, summary = monitor.process()
        monitor.print_report(summary)

    if df is not None:
        df.to_csv(config.csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ CSV报告已保存: {os.path.abspath(config.csv_path)}")

    if config.save_video and os.path.exists(config.video_path):
        print(f"✓ 标注视频已保存: {os.path.abspath(config.video_path)}")

    print("\n" + "=" * 60)
    print("✓ 所有任务完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
