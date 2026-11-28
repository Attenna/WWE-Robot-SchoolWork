"""用场景清单和 G1 FK 产生击掌事件的仿真感知适配器。"""

from __future__ import annotations

import numpy as np

from g1_arena import ArenaSimulation
from g1_arena.events import SimulationEvent

from .high_five import HighFiveConfig, HighFiveStateMachine
from .manifest import SceneManifest, load_manifest
from .robot import hand_positions


class GroundTruthPerception:
    """不依赖图像的仿真 ground truth；用于跑通事件闭环和给视觉基线做对照。

    视觉模型接入后应实现同样的 ``poll(sim) -> list[SimulationEvent]`` 接口，
    决策层无需区分事件来自 ground truth 还是视觉。
    """

    def __init__(
        self,
        manifest: SceneManifest | None = None,
        config: HighFiveConfig | None = None,
    ) -> None:
        self.manifest = manifest or load_manifest()
        self.state = HighFiveStateMachine(config or HighFiveConfig())
        self._last_time = 0.0

    def reset(self) -> None:
        self.state.reset()
        self._last_time = 0.0

    def poll(self, sim: ArenaSimulation) -> list[SimulationEvent]:
        timestamp = float(sim.data.time)
        if timestamp < self._last_time:
            self.reset()
        self._last_time = timestamp

        palms = hand_positions(sim.model, sim.data)
        events: list[SimulationEvent] = []
        for fan in self.manifest.fans:
            if not fan.hand_reaching:
                continue
            if not self.manifest.zone.contains(fan.hand):
                continue
            fan_hand = np.asarray(fan.hand, dtype=float)
            robot_hand, palm = min(
                palms.items(),
                key=lambda item: float(np.linalg.norm(item[1] - fan_hand)),
            )
            distance = float(np.linalg.norm(palm - fan_hand))
            event = self.state.update(
                timestamp=timestamp,
                fan_id=fan.id,
                distance=distance,
                position=fan.hand,
                hand=fan.hand_side,
                robot_hand=robot_hand,
                confidence=1.0,
            )
            if event is not None:
                event.metadata["fan_hand"] = [float(value) for value in fan.hand]
                event.metadata["robot_palm"] = [float(value) for value in palm]
                events.append(event)
        return events
