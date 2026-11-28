# 视觉检测对接文档

本文档面向决策/控制层和后续模型接入方。感知层只做**视觉检测**：输入 RGB/仿真状态，输出结构化事件；不控制机器人、不写关节、不决定动作。

当前支持：

- `ground_truth`：仿真 FK 真值，用于测试和演示；
- `color`：固定机位 + 清单 ROI 的传统颜色/轮廓基线，用于合成图像；
- `none`：关闭感知。

后续 YOLO/关键点/MediaPipe 等只需要实现一个 `HandDetector`，不需要改决策/控制代码。

## 1. 快速开始（仿真）

```python
from g1_arena import ArenaSimulation
from perception import create_perception

with ArenaSimulation() as sim:
    perception = create_perception(
        "color",                  # 或 "ground_truth" / "none"
        camera="interaction_neg",
        width=1280,
        height=720,
    )

    # 演示：走到互动带中间观众前，再挥手
    sim.execute("walk", target=[-6.68, -2.45, 0.79], speed=1.0)
    sim.step(600)
    sim.execute("wave", duration=4.0)

    for _ in range(480):
        sim.step()
        for event in perception.poll(sim):
            print(event.to_dict())
            sim.submit_event(event)   # 只写入 last_event；动作映射由控制层负责
```

`run_demo.py` 已封装同样的流程：

```powershell
# 颜色基线（需要 opencv-python-headless）
.\.venv\Scripts\python.exe run_demo.py --action interaction --perception color

# 仿真真值
.\.venv\Scripts\python.exe run_demo.py --action interaction --perception ground_truth
```

## 2. 公开接口

### 2.1 PerceptionSource

```python
from typing import Protocol
from g1_arena import ArenaSimulation
from g1_arena.events import SimulationEvent

class PerceptionSource(Protocol):
    def poll(self, sim: ArenaSimulation) -> list[SimulationEvent]: ...
    def reset(self) -> None: ...
```

- `poll(sim)`：返回本帧检测到的事件列表，可能为空；建议 15–30 FPS 调用，不需要每个 120 Hz 物理步都调。
- `reset()`：新 episode / 仿真 reset 后调用；内部也会检测时间回跳并自动重置。
- 决策层只依赖这两个方法，不依赖具体检测器。

### 2.2 create_perception

```python
perception = create_perception(
    "color",
    manifest=None,          # 默认 assets/unitree_g1/scene_manifest.json
    config=None,            # HighFiveConfig；默认 0.30 m/0.10 s/1.0 s
    camera="interaction_neg",
    width=1280,
    height=720,
)
```

可选 `source`：

| source | 说明 | 依赖 |
|---|---|---|
| `none` / `off` | 空实现，始终返回 `[]` | 无 |
| `ground_truth` / `gt` | 仿真 FK + 清单手锚点 | 无 |
| `color` / `baseline` | 固定机位颜色/轮廓基线 | `requirements-cv.txt` |

未知 source 抛 `ValueError`；`color` 缺 OpenCV 抛 `RuntimeError`，并提示安装命令。

### 2.3 HandDetector

后续模型实现这个 Protocol：

```python
import numpy as np
from perception import HandDetection

class HandDetector(Protocol):
    def detect(self, frame: np.ndarray, timestamp: float) -> list[HandDetection]: ...
```

`HandDetection` 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `fan_id` | `str` | 观众 ID，例如 `fan_int_neg_3`；未知 ID 会被跳过 |
| `world` | `tuple[float,float,float]` | 观众掌心世界坐标，米 |
| `confidence` | `float` | 0–1；低于 `HighFiveConfig.min_confidence` 不发事件 |
| `hand` | `str` | 观众手标签，默认 `raised` |
| `bbox` | `tuple[int,int,int,int] \| None` | 可选的 2D 框 |
| `track_id` | `int \| None` | 可选跟踪 ID |
| `camera` | `str` | 可选的相机名 |
| `robot_hand` | `str` | 可选；`VisionPerception` 会用 FK 的最近掌心覆盖 |

检测器负责把 2D 框/关键点转成世界坐标（相机标定或互动平面），并给出 `fan_id`。没有 world 坐标时不要伪造；应先补标定/平面假设。

### 2.4 原始检测流（给 CCC）

如果对接方要做“是否举牌/是否伸手/是否击掌”的语义，不要从 `high_five` 事件反推；直接用原始检测：

```python
from perception import create_detector

detector = create_detector("color", camera="interaction_neg")
frame = detector.detect(sim)      # DetectionFrame
# 或真实相机：frame = detector.process(frame_rgb, timestamp)

for hand in frame.hands:
    print(hand.fan_id, hand.bbox, hand.world, hand.confidence)
for sign in frame.signs:
    print(sign.fan_id, sign.bbox, sign.world, sign.confidence)
```

- `DetectorSource.detect(sim)` / `process(frame, timestamp)` 只输出手/牌，不判断语义。
- `DetectionFrame` 包含 `hands`、`signs`、`timestamp`、`camera`、`width`、`height`。
- `HandDetection`：`fan_id`、`world`、`confidence`、`hand`、`bbox`、`track_id`。
- `SignDetection`：`fan_id`、`world`、`bbox`、`confidence`。
- 合成数据的 `scene_manifest.json` 还带 `hand_reaching` / `sign_raised` 真值，可用于 CCC 的语义测试。
- 职责拆分和 CCC 的接入示例见 [ccc-handoff.md](ccc-handoff.md)。

### 2.5 VisionPerception

```python
from perception import VisionPerception

perception = VisionPerception(
    detector,                 # 任意 HandDetector
    manifest=scene_manifest,  # 可选
    config=HighFiveConfig(source="vision", mode="my_model"),
    camera="interaction_neg",
    width=1280,
    height=720,
)
```

两种调用方式：

```python
# 仿真：内部 render + G1 FK 掌心
events = perception.poll(sim)

# 真实相机：调用方给 RGB 和机器人掌心
events = perception.process(frame_rgb, timestamp, robot_palms)
```

`robot_palms` 是 `{"left": np.ndarray(3), "right": np.ndarray(3)}`；真机用本体 FK/手部检测提供。

### 2.6 GroundTruthPerception / ColorContourDetector

- `GroundTruthPerception.poll(sim)`：用清单的观众手锚点和 G1 FK 掌心，不读图像；用于测试和对照。
- `ColorContourDetector`：固定机位 + 清单 ROI；手用 HSV 肤色 + 轮廓，应援牌用颜色/边缘矩形。**不是通用检测器**，只用于合成数据和固定安装场景。

## 3. 事件契约

```json
{
  "event": "high_five",
  "confidence": 0.92,
  "timestamp": 4.69,
  "source": "vision",
  "position": [-6.68, -3.02, 1.3],
  "intensity": null,
  "metadata": {
    "fan_id": "fan_int_neg_3",
    "hand": "raised",
    "robot_hand": "right",
    "distance_m": 0.236,
    "mode": "color_baseline",
    "bbox": [653, 346, 693, 364],
    "track_id": null,
    "camera": "interaction_neg",
    "fan_hand": [-6.68, -3.02, 1.3],
    "robot_palm": [-6.73, -2.93, 1.09]
  }
}
```

- `event`：当前只有 `high_five`；`audience_sign_detected` 预留。
- `position`：世界坐标，米；当前是观众手锚点/检测位置。
- `timestamp`：仿真时间或采集时间，秒。
- `source`：`ground_truth` 或 `vision`。
- `metadata`：自定义字段都放这里；`SimulationEvent.from_dict` 会丢弃未知顶层字段。
- `confidence >= 0.5` 才会被 `submit_event` 当作有效事件；感知层默认阈值也是 0.5。
- `submit_event` 目前只映射 `crowd_cheering` / `audience_wave` / 安全事件；`high_five` 只进入 `last_event`，**不动作**。

### 触发条件

`HighFiveConfig` 默认：

| 参数 | 默认 | 含义 |
|---|---|---|
| `distance_m` | 0.30 | 机器人掌心到观众掌心距离 |
| `release_m` | 0.45 | 退开这么远后才允许下一次 |
| `min_duration_s` | 0.10 | 需持续接近的时间 |
| `cooldown_s` | 1.0 | 同一观众冷却 |
| `min_confidence` | 0.5 | 低于此值不产生事件 |

一次接近只发一次；退开 + 冷却后可再次触发。

## 4. 真实相机接入

感知层不打开相机。集成方负责：

1. 采集 RGB（建议 `np.uint8` HxWx3）。
2. 提供单调递增的 `timestamp`。
3. 提供机器人掌心 `robot_palms`（本体 FK 或手部检测）。
4. 检测器把 2D 观测转成世界坐标（相机标定 / 互动平面）。
5. 调用 `VisionPerception.process(...)`，把事件交给决策层。

```python
frame = camera.read_rgb()
robot_palms = robot.get_palm_positions()
for event in perception.process(frame, timestamp, robot_palms):
    decision.submit(event)
```

相机内参/外参从 `scene_manifest.json` 的 `cameras` 读取；换分辨率时用 `fovy_deg` 重算 `f`，不要直接沿用旧 `K`。

## 5. 接入新检测器

1. 新建模块（例如 `perception/detectors/yolo.py`），实现 `HandDetector.detect`。
2. 不要改 `g1_arena/`；不要在 `perception/__init__.py` eager import torch/ultralytics。
3. 每帧返回 `HandDetection` 列表；`fan_id` 用跟踪/分类关联到 `scene_manifest.json` 的观众。
4. 2D→3D 用相机标定 + 互动平面；拿不到 3D 时先不要发 `high_five`，可以只做 2D 检测/可视化。
5. 在 `create_perception` 注册新 source，或直接把 detector 传给 `VisionPerception`。
6. 用合成数据集评估：`scripts/render_cv_dataset.py` + `scripts/eval_baseline.py`；真实数据另行评估，不与仿真指标混报。

示例：

```python
class MyDetector:
    def detect(self, frame, timestamp):
        # 返回 list[HandDetection]
        return []

perception = VisionPerception(MyDetector(), camera="interaction_neg")
```

## 6. 验收与指标

- 检测：手 bbox P/R/F1、关键点误差、牌子 F1；用 `perception.evaluation`。
- 事件：`high_five` 事件级 recall/FP、时间偏移、处理耗时；不要用帧级 mAP 代替事件验收。
- 当前合成基线参考（8 帧 640×360 seed 1，原始检测含不伸手样本）：手 P/R/F1=0.75/0.94/0.83，牌 F1=0.76，candidate=1.0，检测器 runtime p50≈1.0 ms；960×540 序列 1 TP/0 FP/0 FN，处理 p50≈45 ms（含渲染）。事件路径默认只考虑伸手的观众。
- 命令：
  ```powershell
  .\.venv\Scripts\python.exe scripts\render_cv_dataset.py --frames 8 --width 640 --height 360 --seed 1
  .\.venv\Scripts\python.exe scripts\eval_baseline.py --dataset output\cv_dataset
  .\.venv\Scripts\python.exe scripts\eval_baseline.py --sequence --width 960 --height 540
  ```

## 7. 故障排查

| 现象 | 检查 |
|---|---|
| `color` 启动报 OpenCV | `pip install -r requirements-cv.txt` |
| 一直无事件 | 手是否在 ROI/画面内；距离是否 <0.30 m；是否持续 ≥0.10 s；是否在冷却 |
| `last_event` 有 `high_five` 但机器人不动 | 预期：动作映射在控制层，当前只写 `last_event` |
| 事件 metadata 少了字段 | 自定义字段必须放 `metadata`；顶层未知字段会被丢弃 |
| 渲染尺寸不对 | `ArenaSimulation.render` 按 `(width,height)` 缓存；感知的 `width/height` 要和实际帧一致 |
| 清单/场景不一致 | 重新运行 `scripts/generate_scene.py`，检查 `scene_hash` |
| 未知 `fan_id` | 检测器关联错误；会被 `manifest.fan()` 跳过 |

## 8. 边界

- 当前只保证合成场景；真实 WWE/体育馆素材因版权没有纳入训练/评估。
- `color` 基线依赖固定机位和清单 ROI，不是通用检测器。
- `high_five` 只进入 `last_event`；`Action.HIGH_FIVE` / 决策映射未实现。
- 只实现 `-y` 侧互动前排；`+y` 侧后续镜像。
