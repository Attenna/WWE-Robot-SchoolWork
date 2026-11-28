"""视觉检测公开对接入口测试。"""

from __future__ import annotations

import pytest

from g1_arena import ArenaSimulation
from perception import (
    NullPerception,
    PerceptionSource,
    VisionPerception,
    create_perception,
)


def test_factory_none_and_ground_truth() -> None:
    none = create_perception("none")
    assert isinstance(none, NullPerception)
    assert isinstance(none, PerceptionSource)
    with ArenaSimulation() as sim:
        assert none.poll(sim) == []

    ground_truth = create_perception("ground_truth")
    assert isinstance(ground_truth, PerceptionSource)


def test_factory_color_baseline() -> None:
    try:
        perception = create_perception("color")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert isinstance(perception, VisionPerception)
    assert isinstance(perception, PerceptionSource)


def test_factory_unknown_source() -> None:
    with pytest.raises(ValueError):
        create_perception("yolo-not-installed")
