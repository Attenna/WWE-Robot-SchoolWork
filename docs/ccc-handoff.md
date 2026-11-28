# BBB → CCC 检测交接

微信群分工：

| 角色 | 负责 |
|---|---|
| AAA | 搭擂台/场景 |
| **BBB（本层）** | 相机/清单、合成数据、手与应援牌的**原始视觉检测**、检测接口 |
| CCC | 声音检测 + “是否举牌/是否伸手/是否击掌”的**语义与操作** |
| XXX | 寻路 |

本层只输出检测原语，不判断“举牌/伸手/击掌”，也不控制机器人。

## 1. CCC 拿什么

```python
from g1_arena import ArenaSimulation
from perception import create_detector

with ArenaSimulation() as sim:
    detector = create_detector(
        "color",                  # 或 ground_truth / none
        camera="interaction_neg",
        width=1280,
        height=720,
    )
    detections = detector.detect(sim)   # DetectionFrame
    for hand in detections.hands:
        print(hand.fan_id, hand.bbox, hand.world, hand.confidence)
    for sign in detections.signs:
        print(sign.fan_id, sign.bbox, sign.world, sign.confidence)
```

真实相机接入用：

```python
detections = detector.process(frame_rgb, timestamp)
```

`DetectionFrame` 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `timestamp` | `float` | 仿真/采集时间 |
| `camera` | `str` | 相机名 |
| `width` / `height` | `int` | 帧尺寸 |
| `hands` | `tuple[HandDetection, ...]` | 原始手检测 |
| `signs` | `tuple[SignDetection, ...]` | 原始牌子检测 |

`HandDetection`：`fan_id`、`world`、`confidence`、`hand`、`bbox`、`track_id`、`camera`。
`SignDetection`：`fan_id`、`world`、`bbox`、`confidence`、`camera`。

检测源：

| source | 说明 |
|---|---|
| `color` | 固定机位 + 清单 ROI 的颜色/轮廓基线；合成数据用 |
| `ground_truth` | 清单锚点伪造检测，给 CCC 语义逻辑做单元测试/对照 |
| `none` | 空实现 |

## 2. CCC 负责的语义

建议 CCC 在检测流上输出自己的事件/状态，例如：

- `sign_raised`：牌子是否举过头顶/离开胸前；
- `hand_reaching`：手是否越过护栏/朝机器人伸出；
- `high_five`：手-手接触/接近意图（可结合声音/击掌声）；
- 这些状态可以再用 `SimulationEvent` 提交给决策层。

判定输入建议：

- `SignDetection.bbox` 与 `world.z`：牌子相对头部/护栏的高度；
- `HandDetection.bbox`、`world.y/z`、`track_id`：手的位置和运动方向；
- `fan_id`：固定场景里由 ROIR/清单先验稳定给出；
- 时间序列：至少 3–5 帧的上升/接近趋势，不要只看单帧；
- 置信度：检测置信度 + 时序平滑。

当前仓库的 `HighFiveStateMachine`（`perception/high_five.py`）是**可选参考/demo**，不是 BBB 的正式语义交付；CCC 可以复用、改写或替换。`VisionPerception` 只是把检测接到这个 demo 的适配器。

## 3. 合成数据里的手势状态

`scene_manifest.json` 的每个互动观众有：

```json
{
  "id": "fan_int_neg_3",
  "hand_reaching": true,
  "sign_raised": true,
  "hand": [-6.68, -3.02, 1.3],
  "sign": [-6.68, -3.05, 1.85]
}
```

- `fan_int_neg_3` 是正样本（举牌 + 伸手），`run_demo --action interaction` 使用它。
- 其余观众覆盖 `sign_raised` / `hand_reaching` 的正负组合；合成数据集 `labels.jsonl` 也包含这两个字段，CCC 可以直接拿来做语义规则/评估。
- 注意：`hand_reaching=false` 的手在护栏后/下方，视觉上可能不可见；这正好是“没伸手”的负样本。

## 4. 边界与约定

- BBB 只保证检测：bbox、world、confidence、`fan_id`；不保证语义正确。
- 语义事件不要直接写关节；通过 `SimulationEvent` 交给决策/控制。
- `high_five` 当前只进入 `last_event`，动作映射待控制层；这与 CCC 的语义层可以解耦。
- 颜色基线是固定机位 + 清单 ROI 的合成场景基线，不是通用检测器；换机位/真实场景要换检测器。
- 真实 WWE/版权素材不在当前范围；只使用合成数据。

## 5. 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# 29 passed

.\.venv\Scripts\python.exe scripts\render_cv_dataset.py --frames 8 --width 640 --height 360 --seed 1
.\.venv\Scripts\python.exe scripts\eval_baseline.py --dataset output\cv_dataset
.\.venv\Scripts\python.exe run_demo.py --action interaction --perception color --headless --seconds 8
```

参考指标（8 帧 640×360 seed 1，原始检测含不伸手样本）：手 P≈0.75 / R≈0.94 / F1≈0.83；牌 P/R/F1≈0.81/0.71/0.76；事件 1 TP/0 FP/0 FN。
