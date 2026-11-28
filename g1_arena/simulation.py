"""可复用的 MuJoCo 场馆仿真接口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

from .controller import Action, G1MotionController
from .events import SimulationEvent


class ArenaSimulation:
    """场景、状态和动作的统一入口。

    坐标系：x 轴沿入场通道指向擂台，y 轴指向左侧观众席，z 轴向上；单位为米、秒、弧度。
    默认采用运动学脚本模式，以保证路线与动作演示可重复。
    """

    def __init__(self, scene_path: str | Path | None = None, timestep: float = 1.0 / 120.0) -> None:
        root = Path(__file__).resolve().parents[1]
        self.scene_path = Path(scene_path) if scene_path else root / "assets" / "unitree_g1" / "arena.xml"
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path.resolve()))
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.controller = G1MotionController(self.model)
        self.last_event: SimulationEvent | None = None
        self.safety_latched = False
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.reset()

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def action(self) -> Action:
        return self.controller.action

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.controller.reset()
        self.data.qpos[:] = self.controller.qpos
        self.data.qpos[:3] = self.controller.base_position
        self.data.qpos[3:7] = self.controller.base_quaternion
        mujoco.mj_forward(self.model, self.data)
        self.last_event = None
        self.safety_latched = False

    def execute(
        self,
        action: Action | str,
        *,
        target: Iterable[float] | None = None,
        duration: float | None = None,
        speed: float = 0.55,
    ) -> None:
        requested = Action(action)
        if self.safety_latched and requested != Action.STOP:
            return
        self.controller.execute(requested, target=target, duration=duration, speed=speed)

    def submit_event(self, event: SimulationEvent | dict[str, Any]) -> SimulationEvent:
        """示例事件适配器；正式项目中可由独立决策模块替代本映射。"""
        parsed = event if isinstance(event, SimulationEvent) else SimulationEvent.from_dict(event)
        self.last_event = parsed
        if parsed.event in {"emergency_stop", "obstacle_detected"} and parsed.confidence >= 0.5:
            self.safety_latched = True
            self.execute(Action.STOP)
        elif not self.safety_latched and parsed.confidence >= 0.5:
            if parsed.event == "crowd_cheering":
                self.execute(Action.CELEBRATE)
            elif parsed.event == "audience_wave":
                self.execute(Action.WAVE)
        return parsed

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            self.data.qpos[:] = self.controller.update(self.timestep)
            self.data.time += self.timestep
            mujoco.mj_forward(self.model, self.data)

    def get_state(self) -> dict[str, Any]:
        joint_positions = {
            name: float(self.data.qpos[address])
            for name, address in self.controller.joint_addresses.items()
        }
        return {
            "time": float(self.data.time),
            "action": self.action.value,
            "position": self.data.qpos[:3].tolist(),
            "quaternion_wxyz": self.data.qpos[3:7].tolist(),
            "linear_velocity": self.controller.linear_velocity.tolist(),
            "joint_positions": joint_positions,
            "last_event": self.last_event.to_dict() if self.last_event else None,
        }

    def render(self, camera: str = "overview", width: int = 1280, height: int = 720) -> np.ndarray:
        key = (int(width), int(height))
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderers[key] = renderer
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def __enter__(self) -> "ArenaSimulation":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
