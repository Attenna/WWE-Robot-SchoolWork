"""视觉事件适配器：把检测器的 3D 手部观测接到击掌状态机。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from g1_arena import ArenaSimulation
from g1_arena.events import SimulationEvent

from .high_five import HighFiveConfig, HighFiveStateMachine
from .manifest import SceneManifest, load_manifest
from .robot import hand_positions


@dataclass(frozen=True)
class HandDetection:
    """检测器输出的观众手部观测；world 为世界坐标（米）。

    检测器负责把 2D 框/关键点通过相机标定或深度转成 world。
    ``fan_id`` 来自跟踪/分类；未知 ID 会被 VisionPerception 跳过。
    """

    fan_id: str
    world: tuple[float, float, float]
    confidence: float
    hand: str = "raised"
    bbox: tuple[int, int, int, int] | None = None
    track_id: int | None = None
    camera: str = ""


class HandDetector(Protocol):
    """视觉模型/传统检测器需要实现的接口。"""

    def detect(self, frame: np.ndarray, timestamp: float) -> list[HandDetection]:
        ...


class VisionPerception:
    """把检测器结果送入 HighFiveStateMachine；机器人掌心由仿真 FK 提供。

    真机部署时把 ``robot_palms`` 换成机器人本体/手部检测的结果即可；
    事件格式与 GroundTruthPerception 完全一致，决策层不需要区分来源。
    """

    def __init__(
        self,
        detector: HandDetector,
        manifest: SceneManifest | None = None,
        config: HighFiveConfig | None = None,
        camera: str = "interaction_neg",
        width: int = 1280,
        height: int = 720,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height 必须大于 0")
        self.detector = detector
        self.manifest = manifest or load_manifest()
        self.manifest.camera(camera)
        self.config = config or HighFiveConfig(source="vision", mode="vision")
        self.state = HighFiveStateMachine(self.config)
        self.camera = camera
        self.width = width
        self.height = height
        self._last_time = 0.0

    def reset(self) -> None:
        self.state.reset()
        self._last_time = 0.0

    def poll(self, sim: ArenaSimulation) -> list[SimulationEvent]:
        timestamp = float(sim.data.time)
        if timestamp < self._last_time:
            self.reset()
        self._last_time = timestamp
        frame = sim.render(self.camera, self.width, self.height)
        palms = hand_positions(sim.model, sim.data)
        return self.process(frame, timestamp, palms)

    def process(
        self,
        frame: np.ndarray,
        timestamp: float,
        robot_palms: dict[str, np.ndarray],
    ) -> list[SimulationEvent]:
        timestamp = float(timestamp)
        if timestamp < self._last_time:
            self.reset()
        self._last_time = timestamp
        if not robot_palms:
            raise ValueError("robot_palms 不能为空")

        detections = self.detector.detect(frame, timestamp)
        events: list[SimulationEvent] = []
        for detection in detections:
            if detection.confidence < self.config.min_confidence:
                continue
            try:
                fan = self.manifest.fan(detection.fan_id)
            except KeyError:
                continue

            world = np.asarray(detection.world, dtype=float)
            robot_hand, palm = min(
                robot_palms.items(),
                key=lambda item: float(np.linalg.norm(np.asarray(item[1]) - world)),
            )
            distance = float(np.linalg.norm(np.asarray(palm) - world))
            event = self.state.update(
                timestamp=timestamp,
                fan_id=detection.fan_id,
                distance=distance,
                position=detection.world,
                hand=detection.hand,
                robot_hand=robot_hand,
                confidence=detection.confidence,
            )
            if event is None:
                continue

            event.metadata.update(
                {
                    "bbox": list(detection.bbox) if detection.bbox is not None else None,
                    "track_id": detection.track_id,
                    "camera": detection.camera or self.camera,
                    "fan_hand": [float(value) for value in fan.hand],
                    "robot_palm": [float(value) for value in palm],
                }
            )
            events.append(event)
        return events
