"""G1 掌心位置估计（仿真 FK）。"""

from __future__ import annotations

import mujoco
import numpy as np


# 从 wrist_roll_link 原点到掌心的大致偏移；G1 没有 hand site，
# 用该偏移近似掌心，便于 ground-truth 感知和数据集标注。
PALM_OFFSET = 0.14


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"G1 模型缺少 body：{name}")
    return body_id


def hand_positions(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
    """返回左右掌心世界坐标，键为 left/right。"""
    positions: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        body_id = _body_id(model, f"{side}_wrist_roll_link")
        rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
        positions[side] = np.asarray(data.xpos[body_id], dtype=float) + rotation @ np.array(
            [PALM_OFFSET, 0.0, 0.0], dtype=float
        )
    return positions
