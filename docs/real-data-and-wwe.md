# 真实数据与 WWE 公开素材调研

> **当前决策（2026-11）**：版权素材谈不起，课程交付继续用合成数据；视觉检测接口优先。本文和真实评估脚手架保留为可选资料，不下载/不再分发 WWE 版权内容，也不作为当前验收依赖。

面向 CV 负责人：合成数据已经能跑通事件，但真实 WWE/体育馆场景有很多合成数据没有的难点。本文整理可用的公开手部数据、WWE 公开素材、版权边界和下一步验证方案。

## 结论（先看这里）

- **训练用开放数据集，WWE 广播素材只做真实域评估和域随机化参考**。WWE 官方录像有版权，不要下载/截图入库，也不要在公开仓库再分发；用官方链接、短片段做内部评估，训练/再分发需授权。
- **先做 50–100 帧真实评估集，再决定是否上模型**。不要先下 100K 数据集或训大模型；先看当前颜色基线在真实 WWE 入场/击掌/观众牌画面上掉多少。
- 公开数据优先级：
  1. **TV-Hand / COCO-Hand**：电视/自然场景手检测，最接近广播域；先核对许可证。
  2. **EgoHands / 100DOH**：真实手检测，适合预训练/微调。
  3. **11k Hands / FreiHAND / CMU Panoptic**：手形/关键点/姿态，不适合直接做人群检测。
  4. **InterHand2.6M**：双手交互/击掌姿态，适合高五的手-手关系。
  5. **CrowdHuman / S-Hock**：人群检测/体育场观众上下文，不是手框。
- WWE 素材优先级：官方 “Entrances through the crowd” 播放列表、The Shield/Edge/Jey Uso 的穿人群入场、WWE 360° 入场、WWE fan signs 图集、Wikimedia Commons 的 CC 粉丝牌照片。
- **广播镜头不能直接产出世界坐标事件**：镜头切换、未知标定、运动模糊会让 3D/世界坐标不可靠。WWE 素材用于 2D 手/牌检测和域评估；“高五事件”的几何仍用固定机位 + 标定 + 机器人手位姿。

## 开放手部数据

| 数据集 | 内容 | 许可/获取 | 适合 | 注意 |
|---|---|---|---|---|
| [11k Hands](https://sites.google.com/view/11khands) | 11,076 张，190 人，1600×1200，手背/手心，白底 | 页面写明 **free for reasonable academic fair use**；632 MB | 手形、肤色、手部分割 | 白底摆拍，不是人群/遮挡 |
| [FreiHAND](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html) | 32,560 训练样本（4× 增强）、3,960 评估；21 关键点、3D、分割 | **research only，禁止商用**；3.7 GB | 关键点/姿态/分割 | 绿幕、224×224，不是检测场景 |
| [EgoHands](http://vision.soic.indiana.edu/projects/egohands/) | 48 段第一视角视频、4,800 帧、手框/分割 | 原始研究用途；[Roboflow 衍生版](https://universe.roboflow.com/brad-dwyer/egohands-public) 标 **CC BY 4.0** | 真实手检测预训练 | 第一视角、桌面交互，不是广播/人群 |
| [TV-Hand](https://cs.stonybrook.edu/~cvl/projects/hand_det_attention) | 9.5K 帧电视画面，手 mask/方向；2.3 GB | 项目页未写明确许可，**使用前核对** | 最接近广播域的手检测 | 图像来自电视剧，版权/许可要确认 |
| [COCO-Hand](https://cs.stonybrook.edu/~cvl/projects/hand_det_attention) | 25K COCO 图像的手标注；1.2 GB | 标注来自项目；COCO 图像许可逐张不同，**核对** | 自然场景手检测 | 不是“手部特写”数据集 |
| [100DOH](https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/) | 100K 帧、手-物体接触、手检测模型 | 页面提供下载；使用前核对许可 | 真实手检测/交互 | 数据来自 YouTube，体量大 |
| [CMU Panoptic Hand](http://domedb.perception.cs.cmu.edu/handdb.html) | 14,817 张无约束手部图像、关键点 | 研究用途，核对页面 | 手关键点 | 演播室多视角，不是人群 |
| [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) | 2.6M 帧单/双手交互、3D 姿态 | **CC BY-NC 4.0**（非商用）；5 fps 图像约 80 GB | 双手交互/高五姿态 | 演播室捕获，不是广播人群 |
| [CrowdHuman](https://www.crowdhuman.org/) | 15K/4.3K/5K 人群图像、人体框 | 研究用途，核对 | 粉丝人体检测/ROI | 没有手框 |
| [S-Hock](https://openaccess.thecvf.com/content_cvpr_2015/papers/Conigliaro_The-S-Hock_Dataset_2015_CVPR_paper.pdf) | 体育场观众、细粒度动作（鼓掌、手叉腰等） | 研究用途，核对 | 观众上下文/动作 | 没有手框 |
| [Kinetics-400](https://github.com/cvdfoundation/kinetics-dataset) | 400 类短视频，含 **high fiving** 类 | YouTube 链接；许可随原视频，**不可再分发** | 高五动作/时序预训练 | 下载/训练遵守 YouTube 条款 |

## WWE 公开素材

官方/公开链接（只放链接，不把录像或截图提交到仓库）：

| 素材 | 能帮什么 | 版权/使用 |
|---|---|---|
| [WWE “Entrances through the crowd” 播放列表](https://www.wwe.com/videos/playlists/entrances-through-the-crowd) | 选手穿观众席入场，粉丝伸手/击掌/举牌，和 G1 入场通道最接近 | WWE 版权；官方链接/短片段内部评估 |
| [The Shield 2017 穿人群入场](https://youtube.com/watch?v=Lf50KrdASTI) | 长镜头穿人群、粉丝伸手、护栏互动 | WWE 官方 YouTube；不下载/不再分发 |
| [Edge 1999 穿人群入场](https://wwe.com/videos/edge-enters-through-the-crowd-raw-jan-4-1999) | 早期广播画质/灯光/人群密度 | WWE 版权 |
| [Jey Uso & Sami Zayn 穿人群入场（FOX 片段）](https://fox.com/watch/clip/fmc-0yhjpk1afm6z3wci/jey-uso-and-sami-zayn-enter-through-the-crowd-on-raw-in-green-bay-wisconsin-wwe-on-fox) | 现代广播高光/近景/人群互动 | FOX/WWE；订阅/授权 |
| [WWE 360° 欧洲现场入场](https://wwe.com/worldwide/article/superstars-360-entrances-wwe-live-events-europe-november-2017) | 360° 人群、机位/视角参考 | WWE 版权 |
| [WWE 最佳粉丝牌图集](https://www.wwe.com/shows/raw/2012-05-14/best-fan-signs-photos) | 真实应援牌尺寸、文字、颜色、遮挡 | WWE 版权；只做参考 |
| [Wikimedia: WWE Smackdown FANS with signs!](https://commons.wikimedia.org/wiki/File:WWE_Smackdown_FANS_with_signs!_(3489904123).jpg) | 真实观众举牌，1280×852 | **CC BY 2.0**；作者 jenn / happyskrappy，需署名 |
| [Wikimedia: Sign Guy Dudley](https://commons.wikimedia.org/wiki/File:Sign_Guy_Dudley.jpg) | 真实粉丝/牌子 | **CC BY-SA 2.0**；需署名/相同方式共享 |
| [Wikimedia: WWE Sign Guy](https://commons.wikimedia.org/wiki/File:WWE_Sign_Guy_(8466430893).jpg) | 第一排粉丝/牌子 | 打开文件页核对许可后再用 |
| [Wikimedia: Wrestlemania Sign](https://commons.wikimedia.org/wiki/File:Wrestlemania_Sign_(8467525728).jpg) | 大型牌子/场馆背景 | 打开文件页核对许可后再用 |
| [Internet Archive wrestling 搜索](https://archive.org/search?query=wrestling) | 老录像/独立赛事 | 每个 item 的 rights 不同，**不要假定 public domain** |

> 注意：YouTube 下载、WWE Network/FOX 订阅内容、广播截图都受平台条款和版权约束。公开仓库只保留链接和标注，不提交视频/帧；需要训练或再分发时走授权。

## 从 WWE 素材能学到什么

1. **高五发生在护栏/通道边，不在座位深处**。这与互动区设计一致；官方穿人群入场就是现实版参考。
2. **手很小、很快、经常被遮挡**：粉丝伸手 1–3 帧、镜头切换、运动模糊、护栏/牌子遮挡；事件级检测需要时间容差，不能只看单帧。
3. **应援牌是强上下文也是遮挡物**：真实牌子大、文字多、颜色杂，常遮住手/脸；检测牌子有助于排除手部误检，也可做 OCR 互动。
4. **手的外观差异大**：护腕、胶带、手套、袖子、泡沫手指、不同肤色、戒指/手链；HSV 肤色基线在真实画面会明显掉点，学习模型更稳。
5. **广播域很难**：720p/1080i 压缩、场馆聚光灯、闪光灯、镜头切换、广角/近景、观众密度高；合成数据需要加运动模糊、压缩、闪光、视角/尺度变化。
6. **360° 入场素材**可帮助决定固定相机的高度/FOV/多机位；广播近景可帮助估计真实手部像素尺度。
7. **音频线索**（击掌的“啪”声）在真实场景可能比纯视觉更稳；可作为后续多模态融合，不在本模块范围。

## 真实评估集方案（脚手架已落地）

目标：50–100 帧，来自 3–5 个官方入场/击掌片段 + 2–3 张 CC 粉丝牌照片；只存标注和来源，不存帧。

### 50 帧 clip 计划

| clip_id | 内容 | 采样帧数 | 时间范围（近似） |
|---|---|---|---|
| `shield_2017` | The Shield 穿人群入场（2017） | 12 | 00:20–02:20 |
| `edge_1999` | Edge 穿人群入场（1999） | 12 | 00:20–02:20 |
| `jey_sami` | Jey Uso & Sami Zayn 穿人群入场 | 12 | 00:20–02:20 |
| `wwe_360` | WWE 360° 入场（2017） | 14 | 00:30–02:30 |
| **合计** | | **50** | |

时间范围是近似值，第一次浏览后按实际镜头调整；只从有权限的来源抽帧，不提交视频/帧。

### 工具

```powershell
# 1) 下载 3 张 CC 授权的 WWE 粉丝牌/观众图到 output/real_samples，并写 sources.json
.\.venv\Scripts\python.exe scripts\prepare_real_eval.py fetch-cc

# 2) 生成 50 帧 clip 清单 output/real_eval/clips.json
.\.venv\Scripts\python.exe scripts\prepare_real_eval.py plan

# 3) 如果你有本地、有权限的视频，按 clip 计划抽帧（不下载视频）
.\.venv\Scripts\python.exe scripts\extract_clip_frames.py --clip-id shield_2017 --video D:\path\shield.mp4

# 4) 从本地抽帧目录生成标注骨架（每张图一行 JSONL）
.\.venv\Scripts\python.exe scripts\prepare_real_eval.py init-labels --frames-dir output/real_eval/frames --clip-id shield_2017

# 5) 检查标注、图片存在性和高五引用
.\.venv\Scripts\python.exe scripts\prepare_real_eval.py check
```

`fetch-cc` 下载的是：
- `wwe_smackdown_fans_signs.jpg`（CC BY 2.0，jenn / happyskrappy）
- `sign_guy_dudley.jpg`（CC BY-SA 2.0，Mr Jonathan Ice）
- `wwe_sign_guy.jpg`（CC BY-SA 2.0，Matt Brink）

它们在 `output/real_samples/`，不入库；`sources.json` 里有作者/许可/来源。它们用于真实风格/牌子/观众密度参考，不替代 50 帧 clip 评估集。

### 标注 schema

代码：`perception/real_eval.py`，schema 名 `real-eval.v1`。每帧一个 JSONL 对象：

```json
{
  "schema_version": 1,
  "clip_id": "shield_2017",
  "source_url": "https://youtube.com/watch?v=Lf50KrdASTI",
  "timecode_s": 42.5,
  "image": "frame_0001.png",
  "width": 1280,
  "height": 720,
  "shot": "medium",
  "blur": "mild",
  "hands": [
    {
      "hand_id": "fan_hand_1",
      "bbox": [640, 300, 700, 360],
      "side": "right",
      "visibility": "partial",
      "keypoints": [[660, 320, 1], [670, 330, 1], [680, 340, 1], [690, 350, 1], [700, 360, 1]],
      "keypoint_format": 5,
      "role": "fan",
      "glove": false,
      "wristband": true
    }
  ],
  "signs": [
    {"bbox": [500, 180, 760, 300], "text": "IT'S ALIVE!!!", "occludes_hands": true}
  ],
  "high_five": {
    "present": true,
    "fan_hand_id": "fan_hand_1",
    "wrestler_hand_id": "wrestler_hand_1"
  },
  "notes": ""
}
```

- `keypoint_format` 支持 5 或 21；`keypoints` 为空表示只标框。
- `visibility`：`full` / `partial` / `occluded`；`glove`/`wristband` 记录真实域难点。
- `shot`：`wide` / `medium` / `close` / `unknown`；`blur`：`none` / `mild` / `heavy` / `unknown`。
- `high_five.present=true` 时 `fan_hand_id` / `wrestler_hand_id` 必须存在。
- `check` 会校验 bbox、图片存在性、keypoint 数量和高五引用。

标注字段：

- 手：bbox、5 点或 21 点关键点、左/右、可见性（完全/部分/遮挡）、是否戴手套/护腕。
- 牌子：bbox、文字（可选）、是否遮挡手/脸。
- 场景：镜头类型（广角/中景/近景）、模糊/压缩/闪光、观众密度。
- 事件：是否发生击掌接触（掌心接触 1–2 帧内），接触帧时间戳。

流程：

1. 按 clip 选帧，不按连续帧堆量；每个 clip 取 10–20 帧，覆盖广角/近景/遮挡/不同肤色。
2. 两人标注，10% 双标；按 clip 划分 train/val/test，避免同场景泄漏。
3. 先跑当前颜色基线，记录手 bbox recall/precision、牌子 F1、击掌事件 recall/FP。
4. 如果手 recall <0.8 或牌子 F1 <0.7，再决定训练/微调模型；否则保留基线做课程演示。

## 训练路线（按需）

- **阶段 1（当前）**：合成数据 + 颜色基线，无 torch；只验证管线和事件。
- **阶段 2（真实域）**：用 TV-Hand / COCO-Hand / EgoHands / 100DOH 预训练或微调手检测/关键点；合成数据 + 少量真实标注混合；在 WWE 评估集上单独报告。
- **阶段 3（高五姿态）**：InterHand2.6M 做双手交互/接触表征；Kinetics 的 high fiving 类做时序上下文；只作为辅助，不直接替代手检测。
- **阶段 4（牌子）**：如果演示需要读牌子，接检测/OCR；否则把牌子当遮挡/上下文，不单独训练。
- 训练数据用开放数据集 + 合成；WWE 广播只做评估/域随机化参考，除非拿到明确授权。

## 版权与隐私

- WWE 官方录像、FOX/WWE Network、YouTube 视频均受版权/平台条款约束；不提交、不再分发、不批量下载。
- CC 图片使用需署名、保留许可链接；CC BY-SA 的衍生作品需相同方式共享；只放 `output/`，不入库。
- 数据集许可：11k Hands 学术合理使用；FreiHAND 仅研究、禁商用；InterHand2.6M 非商用；TV-Hand/COCO-Hand/100DOH 使用前逐项核对。
- 观众人脸/身份：公开演示前模糊人脸，不做身份识别/生物特征用途。

## 下一步

1. 从 3 个官方入场片段 + 2 张 CC 粉丝牌照片做 50 帧真实评估集；先跑当前基线。
2. 根据真实评测决定是否下载 TV-Hand（2.3 GB）或 EgoHands/COCO-Hand 做手检测微调。
3. 把 WWE 观察转成合成数据域随机化：手套/护腕、肤色、牌子文字、运动模糊、JPEG 压缩、闪光、镜头角度。
4. 真实评测确认需要模型后，再开 `feat/high-five-yolo`；不要先训模型再找数据。
