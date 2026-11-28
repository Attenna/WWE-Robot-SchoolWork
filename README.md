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
- 急停／障碍事件优先，锁定后普通互动事件不会覆盖安全停止。

> **动作边界**：当前行走和动作采用运动学脚本，直接设置关节角及基座位姿，保证演示稳定、路线可重复。它不是基于接触力平衡的动态双足步行策略。后续接入 RL、MPC 或导航模块时，可替换 `G1MotionController`，场景和接口无需重写。

开发、接口契约与 Git 协作规范见 [AGENTS.md](AGENTS.md)。

## 目录

```text
WWE-Robot-SchoolWork/
├── AGENTS.md                    # 开发约定、接口契约与 Git 协作规范
├── run_demo.py                  # 图形界面与无界面演示入口
├── requirements.txt
├── pyproject.toml
├── docs/
│   └── interaction-zone.md      # 互动区定义、坐标与 CV 约定
├── g1_arena/
│   ├── simulation.py            # 统一仿真接口
│   ├── controller.py            # G1 动作脚本与安全状态
│   └── events.py                # 统一事件数据结构
├── perception/
│   ├── manifest.py              # 场景清单、互动带、相机投影
│   ├── robot.py                 # G1 掌心 FK 近似
│   ├── high_five.py             # 击掌状态机与事件
│   └── ground_truth.py          # 仿真 ground truth 适配器
├── scripts/
│   └── generate_scene.py        # 可参数化修改的场馆生成源码
├── assets/unitree_g1/
│   ├── arena.xml                # 完整仿真场景
│   ├── scene_manifest.json      # 互动区、观众手/牌、相机标定清单
│   ├── g1.xml                   # G1 机器人模型
│   ├── assets/                  # 官方网格
│   └── LICENSE                  # G1 模型 BSD-3-Clause 许可证
└── tests/
    ├── test_smoke.py            # 基础验收测试
    └── test_perception.py       # 感知层契约与击掌测试
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
- G1 模型：[MuJoCo Menagerie — Unitree G1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1)，BSD-3-Clause，原许可证已随模型保留。
- Python 接口：[MuJoCo Python 文档](https://mujoco.readthedocs.io/en/stable/python.html)。

Python 的 `mujoco` 包自带运行库；项目无需依赖 `D:\Mujoco` 的 DLL 搜索路径，但两者版本已统一为 4.0.1。
