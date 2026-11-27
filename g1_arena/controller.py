"""G1 的可替换脚本动作层。

当前实现直接写入关节角和浮动基座位姿，用于稳定的场景与流程演示。
它不是经过训练的动力学步行策略；后续可在本类位置替换为 RL/MPC 控制器。
"""

from __future__ import annotations

from enum import Enum
import math
from typing import Iterable

import mujoco
import numpy as np

from .config import RING_APPROACH, ROBOT_START, WALK_SPEED, WALK_STEP_FREQUENCY


class Action(str, Enum):
    IDLE = "idle"
    WALK = "walk"
    WAVE = "wave"
    CELEBRATE = "celebrate"
    BOW = "bow"
    STOP = "stop"


class G1MotionController:
    """管理 G1 的 29 个关节和浮动基座，输出一帧目标 qpos。"""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id < 0:
            raise ValueError("G1 模型缺少 stand keyframe")
        self.home = model.key_qpos[key_id].copy()
        self.qpos = self.home.copy()
        self.joint_addresses = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id): int(model.jnt_qposadr[joint_id])
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
        }
        self.action = Action.IDLE
        self.action_time = 0.0
        self.action_duration = math.inf
        self.speed = WALK_SPEED
        self.base_position = np.array(ROBOT_START, dtype=float)
        self.base_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.walk_start = self.base_position.copy()
        self.walk_target = self.base_position.copy()
        self.previous_position = self.base_position.copy()
        self.linear_velocity = np.zeros(3)

    def reset(self) -> None:
        self.action = Action.IDLE
        self.action_time = 0.0
        self.action_duration = math.inf
        self.base_position = np.array(ROBOT_START, dtype=float)
        self.base_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.previous_position = self.base_position.copy()
        self.linear_velocity.fill(0.0)
        self.qpos = self.home.copy()

    def execute(
        self,
        action: Action | str,
        *,
        target: Iterable[float] | None = None,
        duration: float | None = None,
        speed: float | None = None,
    ) -> None:
        self.action = Action(action)
        self.action_time = 0.0
        self.speed = max(0.05, WALK_SPEED if speed is None else float(speed))
        self.action_duration = math.inf
        if self.action == Action.WALK:
            target_array = np.array(RING_APPROACH if target is None else list(target), dtype=float)
            if target_array.size == 2:
                target_array = np.array([target_array[0], target_array[1], self.base_position[2]])
            if target_array.shape != (3,):
                raise ValueError("target 必须是 [x, y] 或 [x, y, z]")
            self.walk_start = self.base_position.copy()
            self.walk_target = target_array
            distance = float(np.linalg.norm(self.walk_target[:2] - self.walk_start[:2]))
            self.action_duration = distance / self.speed
            if distance > 1e-6:
                direction = self.walk_target[:2] - self.walk_start[:2]
                yaw = math.atan2(direction[1], direction[0])
                self.base_quaternion = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        elif self.action in (Action.WAVE, Action.CELEBRATE, Action.BOW):
            self.action_duration = duration or {Action.WAVE: 4.0, Action.CELEBRATE: 3.0, Action.BOW: 2.5}[self.action]

    def _joint(self, name: str) -> int:
        return self.joint_addresses[name]

    def _apply_walk(self) -> None:
        phase = 2.0 * math.pi * WALK_STEP_FREQUENCY * self.action_time
        swing = math.sin(phase)
        left_lift = max(0.0, math.sin(phase))
        right_lift = max(0.0, -math.sin(phase))
        joints = {
            "left_hip_pitch_joint": -0.24 * swing,
            "right_hip_pitch_joint": 0.24 * swing,
            "left_knee_joint": 0.10 + 0.30 * left_lift,
            "right_knee_joint": 0.10 + 0.30 * right_lift,
            "left_ankle_pitch_joint": 0.12 * swing - 0.12 * left_lift,
            "right_ankle_pitch_joint": -0.12 * swing - 0.12 * right_lift,
            "left_hip_roll_joint": 0.035,
            "right_hip_roll_joint": -0.035,
            "left_shoulder_pitch_joint": self.home[self._joint("left_shoulder_pitch_joint")] + 0.18 * swing,
            "right_shoulder_pitch_joint": self.home[self._joint("right_shoulder_pitch_joint")] - 0.18 * swing,
        }
        for name, value in joints.items():
            self.qpos[self._joint(name)] = value
        self.qpos[2] += 0.012 * (1.0 - math.cos(2.0 * phase))

    def _apply_wave(self) -> None:
        self.qpos[self._joint("right_shoulder_pitch_joint")] = -0.45
        self.qpos[self._joint("right_shoulder_roll_joint")] = -1.30
        self.qpos[self._joint("right_shoulder_yaw_joint")] = 0.15
        self.qpos[self._joint("right_elbow_joint")] = 0.45
        self.qpos[self._joint("right_wrist_yaw_joint")] = 0.65 * math.sin(2.0 * math.pi * 1.8 * self.action_time)

    def _apply_celebrate(self) -> None:
        bounce = 0.025 * math.sin(2.0 * math.pi * 2.0 * self.action_time)
        self.qpos[self._joint("left_shoulder_pitch_joint")] = -0.40
        self.qpos[self._joint("right_shoulder_pitch_joint")] = -0.40
        self.qpos[self._joint("left_shoulder_roll_joint")] = 1.30
        self.qpos[self._joint("right_shoulder_roll_joint")] = -1.30
        self.qpos[self._joint("left_elbow_joint")] = 0.35
        self.qpos[self._joint("right_elbow_joint")] = 0.35
        self.qpos[2] += bounce

    def _apply_bow(self) -> None:
        progress = min(1.0, self.action_time / 0.6, max(0.0, (self.action_duration - self.action_time) / 0.6))
        self.qpos[self._joint("waist_pitch_joint")] = 0.42 * progress

    def update(self, dt: float) -> np.ndarray:
        self.previous_position = self.base_position.copy()
        self.action_time += dt
        self.qpos[:] = self.home
        self.qpos[:3] = self.base_position
        self.qpos[3:7] = self.base_quaternion

        if self.action == Action.WALK:
            distance = float(np.linalg.norm(self.walk_target[:2] - self.walk_start[:2]))
            alpha = 1.0 if distance < 1e-6 else min(1.0, self.action_time / self.action_duration)
            self.base_position = self.walk_start + (self.walk_target - self.walk_start) * alpha
            self.qpos[:3] = self.base_position
            self._apply_walk()
        elif self.action == Action.WAVE:
            self._apply_wave()
        elif self.action == Action.CELEBRATE:
            self._apply_celebrate()
        elif self.action == Action.BOW:
            self._apply_bow()

        # 消除长时间累加浮点误差，保证第 40 秒整进入 idle，而非多走一帧。
        if self.action_time + 1e-9 >= self.action_duration:
            if self.action == Action.WALK:
                self.base_position = self.walk_target.copy()
                self.qpos[:3] = self.base_position
            self.action = Action.IDLE
            self.action_time = 0.0
            self.action_duration = math.inf

        self.linear_velocity = (self.base_position - self.previous_position) / dt
        return self.qpos.copy()
