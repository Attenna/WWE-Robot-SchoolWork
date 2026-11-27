"""项目基础配置：统一场景、控制器和演示入口使用的参数。"""

from __future__ import annotations

import math

# 机器人从加长通道起点行至擂台外侧：20 m / 0.5 m/s = 40 s。
ROBOT_START = (-24.15, 0.0, 0.87)
RING_APPROACH = (-4.15, 0.0, 0.87)
WALK_SPEED = 0.5
WALK_DURATION_SECONDS = 40.0
WALK_STEP_FREQUENCY = 0.85
WALK_DISTANCE = math.dist(ROBOT_START[:2], RING_APPROACH[:2])

if not math.isclose(WALK_DISTANCE / WALK_SPEED, WALK_DURATION_SECONDS, abs_tol=1e-9):
    raise ValueError("基础配置不满足 40 秒行走时间要求")

# 通道几何范围；起点位于舞台前方，终点与擂台外侧衔接。
AISLE_START_X = -25.85
AISLE_END_X = -3.75
AISLE_CENTER_X = (AISLE_START_X + AISLE_END_X) / 2.0
AISLE_HALF_LENGTH = (AISLE_END_X - AISLE_START_X) / 2.0
STAGE_X = -26.45
STAGE_SCREEN_X = STAGE_X - 0.95
STAGE_TRUSS_X = STAGE_X - 0.25
