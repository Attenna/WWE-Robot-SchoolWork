# AGENTS.md

本文件面向在本仓库中工作的编码智能体，也适用于人类协作者。开始修改前先读本文件与 `README.md`。

## 项目定位

- 仓库：`WWE-Robot-SchoolWork`
- 当前内容：模块 01「仿真环境与机器人本体」的可运行底座（foundation）。
- 技术栈：MuJoCo 4.0.1 + Python ≥ 3.10，场景与动作均在本地离线运行。
- 动作层是运动学脚本（直接写关节角与浮动基座位姿），**不是**动态平衡步态；替换 `G1MotionController` 即可接入 RL / MPC / 导航，场景与事件接口不应因此改动。

## 目录与职责

```text
WWE-Robot-SchoolWork/
├── run_demo.py               # 图形界面 / 无界面演示入口
├── docs/interaction-zone.md  # 互动区定义、坐标与 CV 约定
├── g1_arena/
│   ├── simulation.py         # ArenaSimulation：场景、状态、动作、渲染的统一入口
│   ├── controller.py         # G1MotionController：29 关节 + 浮动基座的运动学脚本
│   └── events.py             # SimulationEvent：感知/决策/仿真之间唯一的事件结构
├── perception/               # CV 感知层：清单、掌心 FK、击掌、数据集、ground truth 适配器
├── scripts/generate_scene.py # 场馆生成源码（尺寸、座位、材质的唯一事实来源）
├── scripts/render_cv_dataset.py # 合成 CV 数据集渲染器（输出到 output/，不入库）
├── assets/unitree_g1/
│   ├── arena.xml             # 完整仿真场景（由 generate_scene.py 生成）
│   ├── scene_manifest.json   # 互动区、观众手/牌、相机标定清单（同脚本生成）
│   ├── g1.xml                # G1 机器人模型
│   └── assets/               # 官方网格，勿手工修改
├── tests/test_smoke.py       # 基础验收测试
└── output/                   # 本地运行产物，已被 .gitignore 忽略
```

## 环境与命令

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt

python run_demo.py                 # 图形界面
python run_demo.py --action entrance --headless --seconds 5 --screenshot output/entrance.png
python run_demo.py --action interaction --perception ground_truth --headless --seconds 8
python scripts/render_cv_dataset.py --frames 20 --width 960 --height 540
python -m pytest -q                # 提交前必须通过
python scripts/generate_scene.py   # 重新生成 arena.xml 与 scene_manifest.json
```

- `mujoco==4.0.1` 是固定版本，升级需单独提交并说明原因。
- 无显示环境渲染可尝试 `MUJOCO_GL=egl`（依赖系统 EGL 库）。
- 若本机 Python 版本过新导致 `mujoco` 无轮子，优先使用 3.10–3.12 的虚拟环境，不要降级/替换仿真库。

## 接口契约（改动需同步文档与测试）

- 坐标：x 沿入场通道指向擂台，y 指向侧方观众席（`±y` 各一侧，当前互动前排位于 `-y` 侧），z 向上；单位米、秒、弧度；四元数顺序 `w, x, y, z`。
- 动作：`idle`、`walk`、`wave`、`celebrate`、`bow`、`stop`；`walk` 接受 `[x, y]` 或 `[x, y, z]`。
- 事件：`crowd_cheering`、`audience_wave`、`obstacle_detected`、`emergency_stop`；`confidence >= 0.5` 才生效。
- CV 可提交 `high_five`（一次击掌，`metadata` 含 `fan_id`、`hand`、`robot_hand`、`distance_m`）；当前只进入 `last_event`，动作映射由决策/控制层添加。
- `emergency_stop` / `obstacle_detected` 会锁存安全停止，只有 `reset()` 能解除；锁存期间普通事件不得覆盖。
- 相机名：`overview`、`entrance`、`ringside`、`interaction_neg`；默认步长 1/120 s。
- `ArenaSimulation` 对外方法保持稳定：`execute`、`submit_event`、`step`、`get_state`、`render`、`reset`、`close`。
- `G1MotionController` 需保留 `__init__(model)` 与 `update(dt) -> qpos` 契约，便于整体替换。

## 修改规则

- 场馆几何只改 `scripts/generate_scene.py`，然后重新生成 `assets/unitree_g1/arena.xml` 与 `assets/unitree_g1/scene_manifest.json`，三者必须在同一个提交里保持一致；不要直接手改生成出的 XML/JSON。
- 互动区坐标、观众手/牌锚点和 CV 相机参数以 `scene_manifest.json` 为唯一事实来源；CV 和数据集不得硬编码，测试只校验清单与生成脚本一致。
- `assets/unitree_g1/assets/` 是上游官方网格，禁止重导出或覆盖；`LICENSE` 必须保留。
- `g1_arena/` 保持轻依赖，不要引入 torch、ROS 等重框架；新模块通过接口组合，不要耦合进仿真核心。
- 改动公开行为时，同步更新 `README.md`、本文件中的契约和 `tests/test_smoke.py`。
- 文档与注释使用中文，标识符与提交类型使用英文；不提交密钥、个人路径、截图和 `output/` 产物。
- 无法在本机验证的改动（例如需要图形界面）要在提交说明或 PR 中明确写出未验证项与原因。

## Git 协作规范

### 分支

- `main` 始终保持可运行；行为或结构改动一律走分支，文档/脚手架小修可直接提交 `main`。
- 分支命名：`feat/<主题>`、`fix/<主题>`、`chore/<主题>`、`docs/<主题>`，主题用小写短横线，例如 `feat/audio-perception`。
- 一个分支只做一件事；不要在同一分支混入无关重构或格式化。

### 提交

- 遵循 Conventional Commits：`feat:`、`fix:`、`chore:`、`docs:`、`test:`、`refactor:`。
- 标题用祈使句、说明「做了什么」，正文写「为什么」与验证方式；示例：

  ```text
  feat: add audio event adapter for crowd cheering

  将音频置信度映射为 SimulationEvent，便于感知模块独立接入。
  验证：python -m pytest -q
  ```

- 每次提交都应可运行；不要提交半成品或注释掉的调试代码。
- 场景 XML 与生成脚本、接口与测试、README 与契约描述，必须成对更新。

### 同步与推送

- 开工前 `git fetch origin && git rebase origin/main`；推送前再同步一次，避免制造无意义的合并提交。
- 禁止对 `main` 或其他共享分支 force-push；禁止改写已推送的历史。
- 冲突用 rebase 解决，解决后必须重新跑测试再推送。
- 大文件（>10 MB）先沟通；单文件不得超过 GitHub 100 MB 限制，优先复用现有网格。

### Pull Request

- 面向 `main` 提 PR，标题同提交规范，描述至少包含：改了什么、为什么、如何验证。
- 视觉/场景改动附 `output/` 下的截图（截图不入库，贴到 PR 描述里）。
- 行为变更需至少一人 review；作者不要自行合并未经 review 的行为变更。
- 合并方式默认 squash，保持 `main` 历史线性；合并后删除分支。

### 多智能体并行

- 每个任务独立分支、独立工作区；不要在同一分支上让多个智能体并发提交。
- 智能体开始前先读 `AGENTS.md` 与相关源码，结束前报告：改动文件、验证命令与结果、未完成项。
- 发现与本任务无关的问题只记录、不顺手修改，避免扩大 diff。
