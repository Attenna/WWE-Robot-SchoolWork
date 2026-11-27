# 宇树 G1｜WWE 风格入场互动仿真底座

基于 **MuJoCo 4.0.1 + Python** 的可运行源码，对应《智能交互机器人项目介绍》中的“模块 01：仿真环境与机器人本体”，并提供事件、状态和动作接口供其他模块集成。

## 已实现

- 真实 Unitree G1 29 自由度 MJCF 模型、网格、关节和执行器定义。
- 四边形擂台：约 6.1 m 围绳内径、三层围绳、四角立柱、护垫、帆布台面、围裙和台阶。
- 入场舞台、灯光架、发光入场通道。
- 擂台两侧各 3 排观众席，共 48 个简化观众。
- 动作：行走、挥手、庆祝、鞠躬、待机、安全停止。
- 3 个固定相机：全景、入口、场边。
- 结构化事件输入、机器人状态读取、动作执行和 RGB 图像输出。
- 急停／障碍事件优先，锁定后普通互动事件不会覆盖安全停止。

> **动作边界**：当前行走和动作采用运动学脚本，直接设置关节角及基座位姿，保证演示稳定、路线可重复。它不是基于接触力平衡的动态双足步行策略。后续接入 RL、MPC 或导航模块时，可替换 `G1MotionController`，场景和接口无需重写。

## 目录

```text
g1_wwe_sim/
├── run_demo.py                  # 图形界面与无界面演示入口
├── requirements.txt
├── pyproject.toml
├── g1_arena/
│   ├── config.py                # 通道、起点、终点、速度与 40 秒配置
│   ├── simulation.py            # 统一仿真接口
│   ├── controller.py            # G1 动作脚本与安全状态
│   └── events.py                # 统一事件数据结构
├── scripts/
│   └── generate_scene.py        # 可参数化修改的场馆生成源码
├── assets/unitree_g1/
│   ├── arena.xml                # 完整仿真场景
│   ├── g1.xml                   # G1 机器人模型
│   ├── assets/                  # 官方网格
│   └── LICENSE                  # G1 模型 BSD-3-Clause 许可证
└── tests/test_smoke.py          # 基础验收测试
```

## 直接运行（Windows）

项目已附带虚拟环境时，在本目录打开终端：

```powershell
.\.venv\Scripts\python.exe run_demo.py
```

默认流程：机器人从入口出发，以 **0.5 m/s** 沿 **20 m** 有效路径行走，**40 s** 到达擂台外侧，然后挥手 4 s、庆祝 3 s。

基础参数集中在 `g1_arena/config.py`：

```python
ROBOT_START = (-24.15, 0.0, 0.87)
RING_APPROACH = (-4.15, 0.0, 0.87)
WALK_SPEED = 0.5
WALK_DURATION_SECONDS = 40.0
```

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
.\.venv\Scripts\python.exe run_demo.py --action entrance --headless --seconds 40 --screenshot output\entrance.png
```

可选相机：`overview`、`entrance`、`ringside`。

## 后续集成

```python
from g1_arena import ArenaSimulation

with ArenaSimulation() as sim:
    # 使用 config.py 默认起点、终点和 0.5 m/s 速度，40 s 到达。
    sim.execute("walk")

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
- 示例事件：`crowd_cheering`、`audience_wave`、`obstacle_detected`、`emergency_stop`。
- 感知层只报告事件；决策层选择行为；控制层执行动作。
- `emergency_stop` 和 `obstacle_detected` 会锁定安全停止，调用 `reset()` 后才恢复普通动作。

## 修改场景与测试

优先修改 `g1_arena/config.py` 中的路径和速度；观众席、材质等外观参数位于 `scripts/generate_scene.py`。修改后重新生成：

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
