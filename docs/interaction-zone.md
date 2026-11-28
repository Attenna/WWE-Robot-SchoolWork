# 互动区（interaction zone）定义与 CV 约定

本文档定义《智能交互机器人项目介绍》中“展示模拟观众的手和观众的牌子”的准确位置，避免把它理解成看台座位区。

## 定义

互动区是**入场通道边缘到第一排护栏之间、机器人可以离开通道走进去的地带**。观众站在护栏后，把手和应援牌伸到互动区边界；机器人在互动区内与观众击掌。

- 互动区不是看台座位，不在观众席内部。
- 当前先实现截图中的 `-y` 侧；`+y` 侧后续按同一套参数镜像。
- 通道、看台、互动前排的坐标只以 `assets/unitree_g1/scene_manifest.json` 为准；CV 和数据集不要硬编码，测试校验清单与生成脚本一致。

## 当前几何

坐标系与项目一致：x 沿通道指向擂台，y 指向侧方观众席，z 向上，单位米。

| 对象 | 当前值 | 说明 |
|---|---|---|
| 入场通道 | x `[-11.45, -3.75]`，y `[-1.10, 1.10]`，台面 z≈0.08 | 原场景不变 |
| 互动带 | x `[-9.30, -4.80]`，y `[-3.10, -1.20]`，z `[0, 2.10]` | 机器人可进入；CV ROI |
| 互动护栏 | y=`-3.10`，高 1.0 | 观众伸手的边界 |
| 站立观众 | y=`-3.65`，6 人 | `fan_int_neg_0` … `fan_int_neg_5` |
| 手 | y=`-3.02`，z=`1.30` | 独立 `*_hand` geom |
| 应援牌 | y=`-3.05`，z=`1.85` | 独立 `*_sign` geom |
| CV 相机 | `interaction_neg` | 覆盖互动前排和机器人 |

互动带长度由通道长度推导：`INTERACTION_X_MIN = AISLE_X_MIN + 2.15`，`INTERACTION_X_MAX = AISLE_X_MAX - 1.05`。人数按约 0.8 m 间距计算。最新版通道加长后，重新运行生成脚本即可，CV 不需要改坐标。

## 场景清单

`scripts/generate_scene.py` 同时生成：

- `assets/unitree_g1/arena.xml`：MuJoCo 场景。
- `assets/unitree_g1/scene_manifest.json`：给 CV/数据集的唯一坐标来源。

清单包含：

```json
{
  "schema_version": 1,
  "scene_hash": "…",
  "interaction_zone": {
    "side": "neg",
    "x": [-9.3, -4.8],
    "y": [-3.1, -1.2],
    "z": [0.0, 2.1],
    "barricade_y": -3.1,
    "hand_y": -3.02,
    "hand_z": 1.3,
    "sign_y": -3.05,
    "sign_z": 1.85
  },
  "fans": [
    {
      "id": "fan_int_neg_0",
      "hand_side": "raised",
      "hand": [-8.9, -3.02, 1.3],
      "sign": [-8.9, -3.05, 1.85],
      "hand_geom": "fan_int_neg_0_hand",
      "sign_geom": "fan_int_neg_0_sign"
    }
  ],
  "cameras": {
    "interaction_neg": {
      "pos": [-10.4, -0.9, 2.8],
      "fovy_deg": 65,
      "width": 1280,
      "height": 720,
      "R_wc": [[…]],
      "t_wc": [-10.4, -0.9, 2.8],
      "K": [[…]]
    }
  }
}
```

`scene_hash` 是 `arena.xml` 的 SHA-256。数据集清单应记录同一个 hash；场景一变就重新生成数据和相机 ROI。

## 相机与投影约定

MuJoCo 相机局部坐标是 **-z 朝前、+y 朝上**，像素 v 向下：

```python
p_c = R_wc.T @ (p_w - t_wc)
u = W / 2 + f * p_c[0] / -p_c[2]
v = H / 2 - f * p_c[1] / -p_c[2]
f = 0.5 * H / tan(radians(fovy) / 2)
```

不要直接套用 OpenCV 的 z-forward 公式。真机相机需要另外提供畸变参数和 `T_world_cam`；仿真相机无畸变。

## CV 事件（Ground Truth 已实现）

CV 只提交结构化事件，不直接写关节。`perception.GroundTruthPerception` 已经用清单和 G1 掌心 FK 跑通 `high_five` 事件；视觉适配器后续实现同样的 `poll(sim)` 接口。

- `high_five`：机器人掌心与观众掌心距离 <0.30 m、相对接近速度 >0、持续 ≥0.1 s、同一观众冷却 1 s；一次经过只发一次。参数集中在 `perception.HighFiveConfig`。
- `audience_sign_detected`：需要机器人对牌子内容反应时再启用；文字放 `metadata.text`（当前未实现）。

自定义字段必须放在 `metadata`，`SimulationEvent.from_dict` 会丢弃未知顶层字段：

```json
{
  "event": "high_five",
  "confidence": 0.92,
  "timestamp": 12.4,
  "source": "ground_truth",
  "position": [-6.68, -3.02, 1.3],
  "metadata": {
    "fan_id": "fan_int_neg_3",
    "hand": "raised",
    "robot_hand": "right",
    "distance_m": 0.24,
    "fan_hand": [-6.68, -3.02, 1.3],
    "robot_palm": [-6.73, -2.93, 1.09],
    "mode": "ground_truth"
  }
}
```

`perception.VisionPerception` 已实现适配器接口：检测器返回 `HandDetection`（`fan_id`、`world`、`confidence`、可选 `bbox`/`track_id`），适配器把结果送入同一个 `HighFiveStateMachine`，并在 `metadata` 里追加 `track_id`、`bbox`、`camera`，`source` 为 `vision`。当前还没有具体检测模型/基线，模型部分后续接入。

当前 `submit_event` 只映射 `crowd_cheering` / `audience_wave` / 安全事件；`high_five` 只进入 `last_event`，动作映射由决策/控制层添加。`emergency_stop` / `obstacle_detected` 的安全锁存优先于任何 CV 事件。

```python
from perception import GroundTruthPerception

perception = GroundTruthPerception()
for event in perception.poll(sim):
    print(event.to_dict())
    sim.submit_event(event)
```

## 合成数据集

`scripts/render_cv_dataset.py` 从 `interaction_neg` 渲染 RGB，内部用 MuJoCo segmentation 生成手/牌的可见 mask，并把清单里的 3D 锚点投到 2D：

```powershell
.\.venv\Scripts\python.exe scripts\render_cv_dataset.py --frames 20 --width 960 --height 540
```

输出 `output/cv_dataset/`：

- `images/000000.png`：RGB（可加 `--save-depth` / `--save-segmentation` 保存 `.npy`）。
- `labels.jsonl`：每帧 6 名观众的手/牌标签，schema `perception.dataset.v1`。
- `dataset_manifest.json`：`scene_hash`、相机、分辨率、seed、标签 schema。

单个锚点标签：

```json
{
  "world": [-6.68, -3.02, 1.3],
  "bbox": [165, 115, 171, 121],
  "keypoint": [168.0, 118.1],
  "keypoint_visible": true,
  "visible": true,
  "pixels": 25
}
```

`high_five_candidate` 是单帧距离 ≤0.30 m 的候选（含 `fan_id`、`distance_m`、`robot_hand`），不是最终事件；事件还需要状态机的接近/持续时间/冷却。数据集默认写到 `output/`，不入库；场景一变就按新的 `scene_hash` 重新生成。

```python
from perception.dataset import load_dataset

manifest, frames = load_dataset("output/cv_dataset")
```

## 验收

- 清单与 XML 的 `scene_hash` 一致；每个 `fan_int_neg_*` 的 `hand` / `sign` geom 存在。
- 每只手/牌子都在互动带内，且手在护栏靠通道一侧；`interaction_neg` 能看到全部手/牌锚点。
- 合成数据事件的召回、误报和延迟按事件级统计，不按帧级 mAP。
- 真实相机数据单独报告；仿真指标不能替代真实场景指标。

## 后续

1. ~~`perception/` 接口 + `GroundTruthPerception`~~（已完成）。
2. ~~合成数据渲染 + 标签 schema~~（已完成，见 `scripts/render_cv_dataset.py`）。
3. ~~`VisionPerception` 事件适配器~~（已完成，见 `perception/vision.py`）；下一步接具体手/牌检测模型或传统基线。
4. `high_five` 事件接入决策/控制，新增 `Action.HIGH_FIVE` 或临时映射到 `WAVE`。
5. 镜像 `+y` 侧，并按最新版通道长度重新生成清单。
