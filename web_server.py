"""Web UI server for classroom attention monitoring.

Provides a FastAPI + WebSocket backend that streams annotated video frames
and attention statistics to a browser-based dashboard.
"""
import asyncio
import base64
import json
import os
import sys
import threading
from datetime import timedelta

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from config import Config, load_config
from behavior import (
    StudentStateTracker, calculate_attention_score,
    mediapipe_landmarks_to_coco_keypoints
)
from chat_detector import detect_chat
from visualizer import draw_annotations
from utils import suppress_warnings

suppress_warnings()
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="Classroom Attention Monitor")

# Global state
_active_streams: dict[int, dict] = {}
_stream_lock = threading.Lock()
_config = load_config()


def _load_model():
    """Load YOLO and pose models for streaming."""
    yolo = YOLO(_config.yolo_model)
    tracker = DeepSort(
        max_age=_config.max_age,
        n_init=_config.n_init,
        max_iou_distance=_config.max_iou_distance
    )
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import base_options as base
    options = vision.PoseLandmarkerOptions(
        base_options=base.BaseOptions(
            model_asset_path='pose_landmarker_lite.task'),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=5,
        min_pose_detection_confidence=_config.min_detection_confidence,
        min_pose_presence_confidence=_config.min_detection_confidence,
        min_tracking_confidence=_config.min_tracking_confidence,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    return yolo, tracker, landmarker


def _process_frame(frame, frame_idx, fps, yolo, tracker, landmarker,
                   state_tracker):
    """Process a single frame and return detections + annotated image."""
    viz_frame = frame.copy()
    results = yolo(frame, classes=[_config.person_class_id],
                   conf=_config.confidence_threshold, verbose=False)

    dets = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        dets.append(([x1, y1, x2 - x1, y2 - y1], conf, 'student'))

    tracks = tracker.update_tracks(dets, frame=frame)
    detections = []

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
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            kpts = mediapipe_landmarks_to_coco_keypoints(landmarks)
            crop_w, crop_h = x2 - x1, y2 - y1
            kpts[:, 0] *= crop_w
            kpts[:, 1] *= crop_h
            bbox_height = y2 - y1
            score, reasons = calculate_attention_score(
                kpts, bbox_height, _config,
                state_tracker, int(track_id), fps,
                face_crop=student_crop
            )
            detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'track_id': track_id,
                'score': score,
                'reasons': reasons,
                'is_focused': score >= _config.attention_threshold,
                'keypoints': kpts,
            })

    chat_pairs = detect_chat(detections, _config)
    for sid_a, sid_b in chat_pairs:
        for det in detections:
            if det['track_id'] in (sid_a, sid_b):
                det['score'] = max(0, det['score'] - _config.behavior.chat_penalty)
                if '聊天' not in det['reasons']:
                    det['reasons'].append('聊天')
                det['is_focused'] = det['score'] >= _config.attention_threshold

    draw_annotations(viz_frame, detections, _config.show_labels)
    return viz_frame, detections


def _encode_frame(frame):
    """Encode a frame as base64 JPEG."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buffer).decode('utf-8')


@app.get("/")
async def root():
    """Serve the dashboard page."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding='utf-8'))
    return HTMLResponse("<h1>Classroom Attention Monitor</h1><p>Dashboard not found.</p>")


@app.get("/api/stats")
async def get_stats():
    """Return current attention statistics."""
    with _stream_lock:
        result = {
            'active_streams': len(_active_streams),
            'streams': {
                sid: {'frame': s['frame_idx'], 'students': s.get('student_count', 0)}
                for sid, s in _active_streams.items()
            }
        }
    return JSONResponse(result)


@app.get("/api/config")
async def get_config():
    """Return current configuration."""
    with _stream_lock:
        result = {
            'attention_threshold': _config.attention_threshold,
            'behavior': {
                'head_down_threshold': _config.behavior.head_down_threshold,
                'eye_closed_threshold': _config.behavior.eye_closed_threshold,
                'stillness_threshold': _config.behavior.stillness_threshold,
                'shoulder_tilt_threshold': _config.behavior.shoulder_tilt_threshold,
            },
            'skip_frames': _config.skip_frames,
            'chat_enabled': _config.behavior.chat_enabled,
            'hand_raise_enabled': _config.behavior.hand_raise_enabled,
        }
    return JSONResponse(result)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for live frame streaming."""
    await ws.accept()
    stream_id = id(ws)

    # Suppress stderr noise during streaming
    _stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')

    try:
        # Receive configuration from client
        data = await ws.receive_json()
        video_source = data.get('video_source', '0')  # default: camera 0
        skip_frames = data.get('skip_frames', _config.skip_frames)

        # Determine if video source is camera (int) or file (str)
        try:
            camera_id = int(video_source)
            is_camera = True
            cap = cv2.VideoCapture(camera_id)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = float('inf')
        except ValueError:
            is_camera = False
            if not os.path.exists(video_source):
                await ws.send_json({'error': f'Video not found: {video_source}'})
                return
            cap = cv2.VideoCapture(video_source)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        await ws.send_json({
            'status': 'loading',
            'message': 'Loading models...',
            'resolution': [width, height],
            'fps': fps,
            'total_frames': total_frames if total_frames != float('inf') else None,
        })

        # Load models
        yolo, tracker, landmarker = _load_model()
        state_tracker = StudentStateTracker()

        with _stream_lock:
            _active_streams[stream_id] = {
                'frame_idx': 0,
                'student_count': 0,
                'video_source': video_source,
            }

        await ws.send_json({'status': 'streaming', 'message': 'Stream started'})

        frame_idx = 0
        stats_interval = max(1, int(fps))  # send stats every ~1 second

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue

            viz_frame, detections = _process_frame(
                frame, frame_idx, fps, yolo, tracker, landmarker, state_tracker
            )

            payload = {
                'type': 'frame',
                'frame_idx': frame_idx,
                'image': _encode_frame(viz_frame),
                'students': [
                    {
                        'id': d['track_id'],
                        'score': d['score'],
                        'focused': d['is_focused'],
                        'reasons': d['reasons'],
                    }
                    for d in detections
                ],
            }

            with _stream_lock:
                if stream_id in _active_streams:
                    _active_streams[stream_id]['frame_idx'] = frame_idx
                    _active_streams[stream_id]['student_count'] = len(detections)

            await ws.send_json(payload)
            frame_idx += 1

            # Small sleep to avoid flooding the client
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({'error': str(e)})
        except Exception:
            pass
    finally:
        sys.stderr = _stderr
        with _stream_lock:
            _active_streams.pop(stream_id, None)
        try:
            cap.release()
        except Exception:
            pass


def main():
    """Start the web server."""
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(
        description='Classroom Attention Web UI'
    )
    parser.add_argument('--host', default='0.0.0.0', help='Bind address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    args = parser.parse_args()

    global _config
    _config = load_config(args.config)

    print("\n" + "=" * 60)
    print(" Classroom Attention - Web UI ".center(60))
    print("=" * 60)
    print(f"\n  Dashboard: http://localhost:{args.port}")
    print(f"  API docs:  http://localhost:{args.port}/docs")
    print(f"  WebSocket: ws://localhost:{args.port}/ws")
    print("\n  Press Ctrl+C to stop\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
