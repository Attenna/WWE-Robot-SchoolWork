"""击掌事件状态机：距离、接近、持续时间和冷却。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from g1_arena.events import SimulationEvent


@dataclass(frozen=True)
class HighFiveConfig:
    """击掌检测阈值；标注、ground truth 和视觉适配器应共用同一组值。"""

    distance_m: float = 0.30
    release_m: float = 0.45
    min_duration_s: float = 0.10
    cooldown_s: float = 1.0
    approach_tolerance_m: float = 0.02
    min_confidence: float = 0.5
    source: str = "ground_truth"
    mode: str = "ground_truth"

    def __post_init__(self) -> None:
        if self.distance_m <= 0.0:
            raise ValueError("distance_m 必须大于 0")
        if self.release_m <= self.distance_m:
            raise ValueError("release_m 必须大于 distance_m")
        if self.min_duration_s < 0.0 or self.cooldown_s < 0.0:
            raise ValueError("时间阈值不能为负")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence 必须位于 [0, 1]")


@dataclass
class _FanState:
    near_since: float | None = None
    last_distance: float | None = None
    last_event_at: float = -math.inf
    released: bool = True


class HighFiveStateMachine:
    """按观众 ID 维护击掌状态；一次接近只发一次事件，退开后需冷却。"""

    def __init__(self, config: HighFiveConfig | None = None) -> None:
        self.config = config or HighFiveConfig()
        self._states: dict[str, _FanState] = {}
        self._last_time: float | None = None

    def reset(self) -> None:
        self._states.clear()
        self._last_time = None

    def update(
        self,
        *,
        timestamp: float,
        fan_id: str,
        distance: float,
        position: Any,
        hand: str = "raised",
        robot_hand: str = "right",
        confidence: float = 1.0,
    ) -> SimulationEvent | None:
        timestamp = float(timestamp)
        distance = float(distance)
        if self._last_time is not None and timestamp < self._last_time:
            self.reset()
        self._last_time = timestamp

        state = self._states.setdefault(fan_id, _FanState())
        if distance > self.config.release_m:
            state.near_since = None
            state.last_distance = distance
            state.released = True
            return None

        approaching = (
            state.last_distance is None
            or distance <= state.last_distance + self.config.approach_tolerance_m
        )
        state.last_distance = distance

        if distance > self.config.distance_m:
            state.near_since = None
            return None

        if state.near_since is None:
            state.near_since = timestamp
        if not state.released:
            return None
        if not approaching:
            return None
        if timestamp - state.near_since < self.config.min_duration_s:
            return None
        if timestamp - state.last_event_at < self.config.cooldown_s:
            return None

        confidence = min(1.0, max(0.0, float(confidence)))
        if confidence < self.config.min_confidence:
            return None

        event = SimulationEvent(
            event="high_five",
            confidence=confidence,
            timestamp=timestamp,
            source=self.config.source,
            position=[float(value) for value in position],
            metadata={
                "fan_id": fan_id,
                "hand": hand,
                "robot_hand": robot_hand,
                "distance_m": round(distance, 4),
                "mode": self.config.mode,
            },
        )
        state.last_event_at = timestamp
        state.released = False
        state.near_since = None
        return event
