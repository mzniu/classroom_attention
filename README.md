# 课堂专注度检测系统 (Classroom Attention Detection System)

基于姿态估计的智能课堂行为分析系统，自动识别学生专注状态并生成可视化报告。

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)

---

## 核心功能

- **长时间低头检测** - 持续低头超过3秒触发严重警告
- **闭眼检测** - 捕捉打瞌睡行为（持续2秒以上）
- **发呆检测** - 头部静止超4秒判定为发呆
- **短暂低头/侧身/手部异常** - 即时姿态异常检测
- **多目标跟踪** - 自动识别并跟踪每个学生
- **视频标注输出** - 红框标记不专注学生，生成可回放视频
- **CSV详细报告** - 包含时间戳、行为类型、持续时长

---

## 系统架构

```
classroom_attention/
├── main.py              # 统一入口, 自动选择GPU/CPU后端
├── config.yaml          # 外部配置文件, 所有阈值可调
├── config.py            # 配置加载器 (YAML → dataclass)
├── behavior.py          # 行为分析引擎 (评分 + 状态追踪)
├── reporter.py          # CSV报告生成 + 控制台输出
├── visualizer.py        # 标注绘制 (纯函数)
├── utils.py             # 视频I/O, GPU检测, 警告抑制
├── ca_gpu.py            # GPU后端 (YOLOv8-pose + ByteTrack)
├── ca.py                # CPU后端 (YOLO + MediaPipe + DeepSORT)
└── tests/               # 20个回归测试
```

**双后端设计**：
| | GPU 后端 (`ca_gpu.py`) | CPU 后端 (`ca.py`) |
|---|---|---|
| 姿态获取 | YOLOv8-pose 一体化 | YOLO + MediaPipe 两步 |
| 追踪算法 | ByteTrack | DeepSORT |
| 行为检测 | 长期+短期 (6种) | 即时姿态 (3种) |
| 推荐场景 | NVIDIA GPU | 无GPU/Windows兼容 |

**数据流**：`视频输入 → 人物检测+姿态估计 → 行为评分 → 状态追踪 → CSV报告 + 标注视频`

---

## 安装部署

### 环境要求
- **Python**: 3.9+
- **GPU**: NVIDIA GPU (推荐) 或 CPU模式
- **操作系统**: Windows 10/11 / Linux / macOS

### 快速安装

```bash
# 1. 克隆项目
git clone https://github.com/mzniu/classroom_attention.git
cd classroom_attention

# 2. 安装依赖
pip install -r requirements.txt
```

GPU用户额外安装 PyTorch CUDA 版本：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

---

## 使用方法

### 统一入口（推荐）

```bash
# 自动检测GPU/CPU并选择最优后端
python main.py classroom_video.mp4

# 强制使用GPU后端
python main.py classroom_video.mp4 --backend gpu --save-video

# 强制使用CPU后端（无GPU环境）
python main.py classroom_video.mp4 --backend cpu

# 使用自定义配置
python main.py classroom_video.mp4 --config my_config.yaml --save-video
```

### 直接调用后端

```bash
# GPU 后端 (YOLOv8-pose)
python ca_gpu.py classroom_video.mp4 --save-video

# CPU 后端 (MediaPipe, Windows兼容)
python ca.py classroom_video.mp4
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `video_path` | 输入视频文件路径 | (必填) |
| `--backend` | 后端选择: auto / gpu / cpu | auto |
| `--config` | 配置文件路径 | config.yaml |
| `--threshold` | 专注度阈值(0-100), 越低越严格 | 50 |
| `--skip-frames` | 跳帧数(0=处理所有帧) | 2 |
| `--save-video` | 保存标注视频 | 不保存 |
| `-o, --output` | 输出视频文件名 | output_annotated.mp4 |
| `--no-labels` | 不显示文字标签 | 显示 |
| `--max-frames` | 测试模式: 只处理前N帧 | 0(全部) |

### 命令示例

```bash
# 快速测试前500帧
python main.py video.mp4 --save-video --max-frames 500

# 严格模式, 检测所有细微行为
python main.py video.mp4 --threshold 70 --skip-frames 1 --save-video

# 只显示边框不显示文字
python main.py video.mp4 --save-video --no-labels

# 使用自定义配置
python main.py video.mp4 --config strict_config.yaml --save-video
```

---

## 输出结果

### 1. 可视化标注视频
- **红色粗边框**: 长时间不专注（低头/闭眼/发呆）
- **红色细边框**: 短期不专注
- **绿色边框**: 专注状态
- **标签信息**: 学生ID、状态、分数、行为原因

### 2. CSV详细报告 (`attention_report.csv`)
```csv
student_id,time_sec,time_str,frame,score,reason,bbox
12,15.67,0:00:15,783,25,"长时间低头(3.2s);发呆(4.1s)","(125,340,189,456)"
7,15.67,0:00:15,783,35,"闭眼(2.4s)","(234,298,267,389)"
```

### 3. 控制台汇总报告
```
======================================================================
                  课堂专注度检测报告
======================================================================

【学生ID: 12】
不专注事件次数: 5
总不专注时长: 87.3秒
不专注时间段:
  1. 0:00:15 ~ 0:01:23 (持续 68.0秒)
     主因: 长时间低头(3.2s)
  2. 0:02:15 ~ 0:02:34 (持续 19.3秒)
     主因: 发呆(4.1s)

======================================================================
总计不专注学生数: 8人
======================================================================
```

---

## 配置指南

所有行为阈值通过 `config.yaml` 管理，无需修改代码：

```yaml
behavior:
  head_down:
    threshold: 0.03       # 低头检测灵敏度
    duration_sec: 3.0     # 多长时间算"长时间低头"
    penalty: 80           # 扣分权重

  eye_closed:
    ear_threshold: 0.18   # 眼睛开合度阈值
    duration_sec: 2.0
    penalty: 70

  stillness:
    movement_px: 5.0      # 头部移动阈值(像素)
    duration_sec: 4.0
    penalty: 50

scoring:
  attention_threshold: 50  # 低于此分判定为不专注

performance:
  skip_frames: 2           # 每N+1帧处理1帧
```

**不同场景的推荐配置**：
- **严格模式**: `attention_threshold: 70`, `head_down.duration_sec: 2.0`
- **宽松模式**: `attention_threshold: 35`, `head_down.duration_sec: 5.0`
- **高性能**: `skip_frames: 5`

---

## 性能指标

| 硬件配置 | 处理速度 | 显存占用 |
|----------|----------|----------|
| RTX 4060 Ti 16GB | ~250 FPS | ~4GB |
| RTX 3060 12GB | ~200 FPS | ~3.5GB |
| CPU (i7-12700K) | ~5 FPS | - |
| CPU + 跳帧5 | ~30 FPS | - |

---

## 测试

项目包含20个回归测试，覆盖核心模块：

```bash
# 运行全部测试
pytest tests/ -v

# 分模块运行
pytest tests/test_behavior.py -v   # 行为分析 (9个)
pytest tests/test_config.py -v     # 配置加载 (5个)
pytest tests/test_reporter.py -v   # 报告生成 (6个)
```

---

## 已知限制

- 遮挡超过1秒可能导致ID切换
- 教室超过50人时建议降低分辨率或增大跳帧
- 躺卧、大幅度转身可能导致关键点丢失

---

## 许可证

Apache License 2.0

## 联系方式

- **项目地址**: [https://github.com/mzniu/classroom_attention](https://github.com/mzniu/classroom_attention)
- **问题反馈**: [Issues页面](https://github.com/mzniu/classroom_attention/issues)
- **邮箱**: aindy@126.com
