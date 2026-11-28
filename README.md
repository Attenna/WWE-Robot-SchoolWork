# 宇树 G1｜WWE 风格入场互动仿真底座

基于 **MuJoCo 4.0.1 + Python** 的可运行源码，对应《智能交互机器人项目介绍》中的“模块 01：仿真环境与机器人本体”，并提供事件、状态和动作接口供其他模块集成。

## 已实现

- 真实 Unitree G1 29 自由度 MJCF 模型、网格、关节和执行器定义。
- 四边形擂台：约 6.1 m 围绳内径、三层围绳、四角立柱、护垫、帆布台面、围裙和台阶。
- 入场舞台、灯光架、发光入场通道。
- 擂台两侧各 3 排观众席，共 48 个坐姿观众；入场通道旁另有 6 名站立互动观众，带独立手部与应援牌。
- 互动带：通道边缘到第一排护栏之间、机器人可以进入的击掌区域，带绿色 LED 边界。
- 动作：行走、挥手、庆祝、鞠躬、待机、安全停止。
- 4 个固定相机：全景、入口、场边、互动前排。
- 结构化事件输入、机器人状态读取、动作执行和 RGB 图像输出。
- CV 感知层：场景清单、G1 掌心 FK、击掌状态机和 `high_five` 事件（仿真 ground truth 适配器）。
- `VisionPerception`：检测器 Protocol + 视觉事件适配器；手部世界坐标和 robot palm 由检测器/FK 提供。
- 合成 CV 数据集：RGB + 手/应援牌 bbox/keypoint/可见性标签 + 可选 depth/segmentation，清单记录 `scene_hash`。
- 传统颜色/轮廓基线（无 torch）：固定机位 + 清单 ROI，OpenCV 肤色找手、颜色/边缘找应援牌；含数据集与事件指标评估。
- 急停／障碍事件优先，锁定后普通互动事件不会覆盖安全停止。

> **动作边界**：当前行走和动作采用运动学脚本，直接设置关节角及基座位姿，保证演示稳定、路线可重复。它不是基于接触力平衡的动态双足步行策略。后续接入 RL、MPC 或导航模块时，可替换 `G1MotionController`，场景和接口无需重写。

开发、接口契约与 Git 协作规范见 [AGENTS.md](AGENTS.md)。

## 目录

```text
WWE-Robot-SchoolWork/
├── AGENTS.md                    # 开发约定、接口契约与 Git 协作规范
├── run_demo.py                  # 图形界面与无界面演示入口
├── requirements.txt
├── requirements-cv.txt          # 可选：opencv-python-headless（传统基线）
├── pyproject.toml
├── docs/
│   ├── interaction-zone.md      # 互动区定义、坐标与 CV 约定
│   └── real-data-and-wwe.md     # 真实手部数据、WWE 素材与版权/评测协议
├── g1_arena/
│   ├── simulation.py            # 统一仿真接口
│   ├── controller.py            # G1 动作脚本与安全状态
│   └── events.py                # 统一事件数据结构
├── perception/
│   ├── manifest.py              # 场景清单、互动带、相机投影
│   ├── robot.py                 # G1 掌心 FK 近似
│   ├── high_five.py             # 击掌状态机与事件
│   ├── dataset.py               # 合成数据集标签 schema 与读取
│   ├── vision.py                # 检测器 Protocol + 视觉事件适配器
│   ├── baseline.py              # 传统颜色/轮廓基线（可选 OpenCV）
│   ├── evaluation.py            # bbox/事件指标
│   ├── real_eval.py             # 真实评估集 schema
│   └── ground_truth.py          # 仿真 ground truth 适配器
├── scripts/
│   ├── generate_scene.py        # 可参数化修改的场馆生成源码
│   ├── render_cv_dataset.py     # 合成 CV 数据集渲染器
│   ├── eval_baseline.py         # 基线数据集/事件评估
│   ├── prepare_real_eval.py     # CC 图片、50 帧 clip 清单、标注骨架
│   └── extract_clip_frames.py   # 从本地有权限视频按 clip 计划抽帧
├── assets/unitree_g1/
│   ├── arena.xml                # 完整仿真场景
│   ├── scene_manifest.json      # 互动区、观众手/牌、相机标定清单
│   ├── g1.xml                   # G1 机器人模型
│   ├── assets/                  # 官方网格
│   └── LICENSE                  # G1 模型 BSD-3-Clause 许可证
└── tests/
    ├── test_smoke.py            # 基础验收测试
    ├── test_perception.py       # 感知层契约与击掌测试
    ├── test_dataset.py          # 合成数据集渲染与标签测试
    ├── test_evaluation.py       # 检测/事件指标测试
    ├── test_baseline.py         # 传统基线数据集验收（需 OpenCV）
    └── test_real_eval.py        # 真实评估集 schema 与 clip 计划测试
```

## 直接运行（Windows）

项目已附带虚拟环境时，在本目录打开终端：

```powershell
.\.venv\Scripts\python.exe run_demo.py
```

默认流程：机器人从入口出发，沿通道行走，到达擂台外侧后挥手，再庆祝。

重新创建依赖环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_demo.py
```

### 按键

| 按键 | 功能 |
|---|---|
| Enter | 重新开始完整入场流程 |
| W | 挥手 |
| C | 庆祝 |
| B | 鞠躬 |
| 空格 | 停止 |
| R | 复位并解除安全锁定 |

### 单独动作与截图

```powershell
.\.venv\Scripts\python.exe run_demo.py --action wave
.\.venv\Scripts\python.exe run_demo.py --action walk
.\.venv\Scripts\python.exe run_demo.py --action entrance --headless --seconds 5 --screenshot output\entrance.png
```

可选相机：`overview`、`entrance`、`ringside`、`interaction_neg`。其中 `interaction_neg` 是互动前排的 CV 视角，适合检查观众手部、应援牌和机器人接近过程。

### 感知层（Ground Truth）

`--action interaction` 会让机器人走到互动带的中间观众前，随后挥手；`--perception ground_truth` 用清单里的观众手锚点和 G1 掌心 FK 计算距离，产生一次 `high_five` 事件。事件目前只打印并写入 `last_event`，动作映射留给决策/控制层。

```powershell
# 图形界面：走到互动带、挥手、打印 high_five 事件
.\.venv\Scripts\python.exe run_demo.py --action interaction --perception ground_truth

# 无界面：8 秒跑完整个击掌流程并截图
.\.venv\Scripts\python.exe run_demo.py --action interaction --perception ground_truth --headless --seconds 8 --camera interaction_neg --screenshot output\high_five.png
```

感知层的公开入口：

```python
from perception import GroundTruthPerception

perception = GroundTruthPerception()
for event in perception.poll(sim):
    print(event.to_dict())
    sim.submit_event(event)  # 事件进入 last_event；动作映射由决策/控制层添加
```

### 合成数据集

```powershell
# 20 帧，960×540，输出到 output/cv_dataset
.\.venv\Scripts\python.exe scripts\render_cv_dataset.py --frames 20 --width 960 --height 540

# 需要 depth / segmentation 时
.\.venv\Scripts\python.exe scripts\render_cv_dataset.py --frames 20 --save-depth --save-segmentation
```

输出结构：

```text
output/cv_dataset/
├── images/                 # RGB；可选 *_depth.npy / *_seg.npy
├── labels.jsonl            # 每帧 6 名观众的手/牌标签
└── dataset_manifest.json   # scene_hash、相机、尺寸、seed、标签 schema
```

标签 schema 为 `perception.dataset.v1`，每个锚点包含 `world`、`bbox`、`keypoint`、`keypoint_visible`、`visible`、`pixels`；每帧还包含机器人掌心、`high_five_candidate`（距离 ≤0.30 m 时）。数据集默认写到 `output/`，不入库；场景变更后重新生成。

```python
from perception.dataset import load_dataset

manifest, frames = load_dataset("output/cv_dataset")
```

### 传统基线（无 torch）

需要可选 OpenCV：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cv.txt
```

```powershell
# 数据集检测指标
.\.venv\Scripts\python.exe scripts\eval_baseline.py --dataset output\cv_dataset --out output\cv_eval_dataset.json
# 仿真序列事件指标
.\.venv\Scripts\python.exe scripts\eval_baseline.py --sequence --width 960 --height 540 --out output\cv_eval_sequence.json
```

基线是**固定机位 + 清单 ROI 先验**：在每名观众手/牌的投影位置开 ROI，手用 HSV 肤色 + 轮廓，应援牌用颜色/边缘矩形；没有 torch。世界坐标使用清单锚点，所以它验证的是“数据集 + 事件管线”，不是通用 3D 估计。参考结果（8 帧 640×360，seed 1）：

| 指标 | 结果 |
|---|---|
| 手 bbox P / R / F1 | 1.00 / 1.00 / 1.00 |
| 应援牌 P / R / F1 | 0.72 / 0.65 / 0.68 |
| 单帧 high-five candidate | 1.00 / 1.00 / 1.00 |
| 检测器 runtime p50 | ≈1.1 ms |

序列模式（960×540）得到 1 个 TP、0 FP/FN；处理耗时 p50≈45 ms（含 MuJoCo 渲染），事件时间偏移 0（同步仿真，不代表真实相机延迟）。

### 真实评估集脚手架

```powershell
# 3 张 CC 授权的 WWE 观众/粉丝牌图 → output/real_samples，附 sources.json
.\.venv\Scripts\python.exe scripts\prepare_real_eval.py fetch-cc

# 50 帧 clip 清单 → output/real_eval/clips.json
.\.venv\Scripts\python.exe scripts\prepare_real_eval.py plan

# 如果你有本地、有权限的视频，按 clip 计划抽帧（不下载视频）
.\.venv\Scripts\python.exe scripts\extract_clip_frames.py --clip-id shield_2017 --video D:\path\shield.mp4

# 从本地抽帧目录生成标注骨架，再用 CVAT/Label Studio 或手填 JSONL
.\.venv\Scripts\python.exe scripts\prepare_real_eval.py init-labels --frames-dir output/real_eval/frames --clip-id shield_2017
.\.venv\Scripts\python.exe scripts\prepare_real_eval.py check
```

50 帧来自 4 个官方入场/360° 片段（Shield 2017、Edge 1999、Jey Uso & Sami Zayn、WWE 360°），只存标注和来源，不提交视频/帧。schema `real-eval.v1` 支持手 bbox/5/21 关键点、可见性、手套/护腕、应援牌 bbox/文字、击掌引用。详见 [docs/real-data-and-wwe.md](docs/real-data-and-wwe.md)。

### 视觉事件适配器

`VisionPerception` 不绑定具体模型；检测器实现 `detect(frame, timestamp) -> list[HandDetection]` 即可。`HandDetection.world` 必须是世界坐标（由相机标定/深度得到），`fan_id` 用于关联互动观众。

```python
from perception import HandDetection, VisionPerception

class MyDetector:
    def detect(self, frame, timestamp):
        return [HandDetection(fan_id="fan_int_neg_3", world=(-6.68, -3.02, 1.3), confidence=0.9)]

perception = VisionPerception(MyDetector(), camera="interaction_neg")
for event in perception.poll(sim):
    sim.submit_event(event)
```

仿真里机器人掌心由 FK 提供；真机部署时把 `robot_palms` 换成机器人本体/手部检测的位姿。事件与 `GroundTruthPerception` 完全同构，决策层不需要区分来源。

## 后续集成

```python
from g1_arena import ArenaSimulation

with ArenaSimulation() as sim:
    # 导航模块：给定目标点，单位为米。
    sim.execute("walk", target=[-4.15, 0.0, 0.87], speed=0.65)

    # 感知模块只提交结构化事件，不直接写关节。
    sim.submit_event({
        "event": "crowd_cheering",
        "confidence": 0.87,
        "timestamp": 12.40,
        "source": "audio",
    })

    sim.step(120)  # 前进 1 秒（默认 120 Hz）
    state = sim.get_state()
    rgb = sim.render(camera="overview", width=1280, height=720)
```

### 统一约定

- 坐标：x 轴沿通道指向擂台，y 轴指向侧方观众席，z 轴向上。
- 单位：米、秒、弧度；四元数顺序为 `w, x, y, z`。
- 动作：`idle`、`walk`、`wave`、`celebrate`、`bow`、`stop`。
- 示例事件：`crowd_cheering`、`audience_wave`、`obstacle_detected`、`emergency_stop`；CV 还可提交 `high_five`（一次击掌）。
- 感知层只报告事件；决策层选择行为；控制层执行动作。`high_five` 的动作映射待控制层添加。
- `emergency_stop` 和 `obstacle_detected` 会锁定安全停止，调用 `reset()` 后才恢复普通动作。

### 互动区（CV 约定）

互动区不是看台座位，而是入场通道边缘到第一排护栏之间、机器人可以离开通道走进去的地带。当前场景先实现截图中的 `-y` 侧：6 名站立观众在护栏后，手和应援牌是独立几何体，机器人在绿色 LED 边界内与观众击掌。

互动带坐标、观众手/牌锚点和 `interaction_neg` 相机参数统一由 `assets/unitree_g1/scene_manifest.json` 给出；CV 和数据集不要硬编码。完整定义和 `high_five` 事件约定见 [docs/interaction-zone.md](docs/interaction-zone.md)。

```powershell
# 从互动前排视角截一张图
.\.venv\Scripts\python.exe run_demo.py --headless --seconds 1 --camera interaction_neg --screenshot output\interaction_neg.png
```

## 修改场景与测试

修改 `scripts/generate_scene.py` 中的尺寸、座位或材质参数，然后重新生成 XML 和清单：

```powershell
.\.venv\Scripts\python.exe scripts\generate_scene.py
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
```

## 资料与模型来源

- 擂台结构参考：[Wrestling ring — Configuration and construction](https://en.wikipedia.org/wiki/Wrestling_ring)。采用方形台面、三层围绳、四角立柱和护垫，不包含八角笼。
- 真实手部数据、WWE 公开素材、版权边界和真实评估协议见 [docs/real-data-and-wwe.md](docs/real-data-and-wwe.md)。
- G1 模型：[MuJoCo Menagerie — Unitree G1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1)，BSD-3-Clause，原许可证已随模型保留。
- Python 接口：[MuJoCo Python 文档](https://mujoco.readthedocs.io/en/stable/python.html)。

Python 的 `mujoco` 包自带运行库；项目无需依赖 `D:\Mujoco` 的 DLL 搜索路径，但两者版本已统一为 4.0.1。
