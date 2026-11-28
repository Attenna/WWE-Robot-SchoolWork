"""检测/事件指标函数测试。"""

from __future__ import annotations

from g1_arena.events import SimulationEvent
from perception.evaluation import (
    bbox_iou,
    latency_summary,
    match_detections,
    match_events,
)


def test_bbox_iou() -> None:
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert 0.3 < bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) < 0.4


def test_match_detections() -> None:
    ground_truth = [("a", (0, 0, 10, 10)), ("b", (20, 20, 30, 30))]
    detections = [("a", (1, 1, 11, 11)), ("a", (50, 50, 60, 60))]
    counts = match_detections(detections, ground_truth, iou_threshold=0.3)
    assert counts.tp == 1
    assert counts.fp == 1
    assert counts.fn == 1
    assert 0.0 < counts.precision < 1.0
    assert 0.0 < counts.recall < 1.0


def _event(event: str, timestamp: float, fan_id: str) -> SimulationEvent:
    return SimulationEvent(
        event=event,
        confidence=1.0,
        timestamp=timestamp,
        source="test",
        position=[0.0, 0.0, 0.0],
        metadata={"fan_id": fan_id},
    )


def test_match_events_and_latency() -> None:
    truth = [_event("high_five", 1.1, "fan_a"), _event("high_five", 3.0, "fan_b")]
    predicted = [_event("high_five", 1.0, "fan_a"), _event("high_five", 3.0, "fan_c")]
    counts, latencies = match_events(predicted, truth, tolerance_s=0.5)
    assert counts.tp == 1
    assert counts.fp == 1
    assert counts.fn == 1
    assert len(latencies) == 1
    assert abs(latencies[0] + 0.1) < 1e-9
    summary = latency_summary(latencies)
    assert summary["p50_ms"] == -100.0
    assert summary["mean_ms"] == -100.0
