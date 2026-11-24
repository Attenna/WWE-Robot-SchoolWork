"""统一事件结构：视觉、声音和决策模块只通过该结构与仿真层交换信息。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SimulationEvent:
    """与项目介绍约定一致的事件格式，单位为秒和米。"""

    event: str
    confidence: float
    timestamp: float
    source: str
    position: list[float] | None = None
    intensity: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event:
            raise ValueError("event 不能为空")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence 必须位于 [0, 1]")
        if self.position is not None and len(self.position) != 3:
            raise ValueError("position 必须是 [x, y, z]")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SimulationEvent":
        known = {
            key: value[key]
            for key in ("event", "confidence", "timestamp", "source", "position", "intensity")
            if key in value
        }
        known["metadata"] = value.get("metadata", {})
        return cls(**known)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
