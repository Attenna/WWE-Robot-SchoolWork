"""感知层契约与 ground-truth 击掌测试。"""

from __future__ import annotations

import pytest

from g1_arena import Action, ArenaSimulation, SimulationEvent
from perception import (
    GroundTruthPerception,
    HandDetection,
    HighFiveConfig,
    HighFiveStateMachine,
    VisionPerception,
    load_manifest,
)


def test_manifest_contract() -> None:
    manifest = load_manifest()
    assert manifest.schema_version == 1
    assert manifest.scene_hash
    assert manifest.zone.contains([-6.68, -3.02, 1.3])
    assert manifest.zone.contains([-9.20, -2.00, 1.00])
    assert not manifest.zone.contains([-6.68, -3.20, 1.30])

    fan = manifest.fans[len(manifest.fans) // 2]
    assert fan.hand_side == "raised"
    assert manifest.fan(fan.id) == fan
    with pytest.raises(KeyError):
        manifest.fan("fan_int_neg_missing")

    camera = manifest.camera("interaction_neg")
    assert camera.width > 0 and camera.height > 0
    for item in manifest.fans:
        for point in (item.hand, item.sign):
            u, v, depth = camera.project(point)
            assert depth > 0.0
            assert 0.0 <= u < camera.width
            assert 0.0 <= v < camera.height
    with pytest.raises(KeyError):
        manifest.camera("nope")


def test_high_five_state_machine() -> None:
    config = HighFiveConfig(
        distance_m=0.30,
        release_m=0.45,
        min_duration_s=0.10,
        cooldown_s=1.0,
    )
    machine = HighFiveStateMachine(config)
    position = [-6.68, -3.02, 1.3]

    assert machine.update(timestamp=0.00, fan_id="fan", distance=0.80, position=position) is None
    assert machine.update(timestamp=0.10, fan_id="fan", distance=0.40, position=position) is None
    assert machine.update(timestamp=0.20, fan_id="fan", distance=0.28, position=position) is None
    assert machine.update(timestamp=0.25, fan_id="fan", distance=0.26, position=position) is None

    event = machine.update(timestamp=0.31, fan_id="fan", distance=0.24, position=position)
    assert event is not None
    assert event.event == "high_five"
    assert event.confidence == 1.0
    assert event.position == position
    assert event.metadata["fan_id"] == "fan"
    assert event.metadata["distance_m"] == 0.24
    assert event.metadata["hand"] == "raised"
    assert event.metadata["robot_hand"] == "right"

    # 仍然贴近时不能重复发；退到 release 之外后才允许下一次。
    assert machine.update(timestamp=0.40, fan_id="fan", distance=0.24, position=position) is None
    assert machine.update(timestamp=1.40, fan_id="fan", distance=0.24, position=position) is None
    assert machine.update(timestamp=1.50, fan_id="fan", distance=0.60, position=position) is None
    assert machine.update(timestamp=1.60, fan_id="fan", distance=0.28, position=position) is None
    second = machine.update(timestamp=1.72, fan_id="fan", distance=0.25, position=position)
    assert second is not None
    assert second.timestamp == 1.72


def test_high_five_confidence_gate_and_config_validation() -> None:
    machine = HighFiveStateMachine(HighFiveConfig(min_confidence=0.6))
    position = [-6.68, -3.02, 1.3]
    assert machine.update(timestamp=0.0, fan_id="fan", distance=0.28, position=position, confidence=0.4) is None
    assert machine.update(timestamp=0.15, fan_id="fan", distance=0.24, position=position, confidence=0.4) is None
    event = machine.update(timestamp=0.30, fan_id="fan", distance=0.22, position=position, confidence=0.9)
    assert event is not None
    assert event.confidence == 0.9

    with pytest.raises(ValueError):
        HighFiveConfig(distance_m=0.30, release_m=0.30)
    with pytest.raises(ValueError):
        HighFiveConfig(min_confidence=1.5)


def test_ground_truth_perception_emits_one_high_five() -> None:
    with ArenaSimulation() as sim:
        perception = GroundTruthPerception()
        fan = perception.manifest.fans[len(perception.manifest.fans) // 2]
        sim.execute(
            Action.WALK,
            target=[fan.hand[0], -2.45, 0.79],
            speed=1.0,
        )
        sim.step(600)
        sim.execute(Action.WAVE, duration=4.0)

        events = []
        for _ in range(80):
            sim.step()
            events.extend(perception.poll(sim))

        assert len(events) == 1
        event = events[0]
        assert event.event == "high_five"
        assert event.source == "ground_truth"
        assert event.confidence == 1.0
        assert event.position == list(fan.hand)
        assert event.metadata["fan_id"] == fan.id
        assert event.metadata["hand"] == "raised"
        assert event.metadata["robot_hand"] == "right"
        assert event.metadata["distance_m"] <= 0.30

        sim.submit_event(event)
        state = sim.get_state()
        assert state["last_event"]["event"] == "high_five"
        assert state["last_event"]["metadata"]["fan_id"] == fan.id
        assert sim.action == Action.WAVE
        assert not sim.safety_latched


class _FakeDetector:
    def __init__(self, detection: HandDetection) -> None:
        self.detection = detection
        self.frames: list[tuple[tuple[int, ...], float]] = []

    def detect(self, frame, timestamp: float) -> list[HandDetection]:
        self.frames.append((tuple(frame.shape), timestamp))
        return [self.detection]


def test_vision_perception_with_fake_detector() -> None:
    with ArenaSimulation() as sim:
        manifest = load_manifest()
        fan = manifest.fans[len(manifest.fans) // 2]
        detector = _FakeDetector(
            HandDetection(
                fan_id=fan.id,
                world=fan.hand,
                confidence=0.92,
                bbox=(10, 20, 30, 40),
                track_id=7,
            )
        )
        vision = VisionPerception(
            detector,
            manifest=manifest,
            width=320,
            height=240,
        )
        sim.execute(Action.WALK, target=[fan.hand[0], -2.45, 0.79], speed=1.0)
        sim.step(600)
        sim.execute(Action.WAVE, duration=4.0)

        events = []
        for _ in range(40):
            sim.step()
            events.extend(vision.poll(sim))

        assert len(events) == 1
        event = events[0]
        assert event.event == "high_five"
        assert event.source == "vision"
        assert event.metadata["fan_id"] == fan.id
        assert event.metadata["bbox"] == [10, 20, 30, 40]
        assert event.metadata["track_id"] == 7
        assert event.metadata["camera"] == "interaction_neg"
        assert event.metadata["distance_m"] <= 0.30
        assert detector.frames
        assert detector.frames[0][0] == (240, 320, 3)


def test_submit_event_accepts_high_five_without_safety_latch() -> None:
    with ArenaSimulation() as sim:
        event = SimulationEvent(
            event="high_five",
            confidence=0.9,
            timestamp=1.0,
            source="vision",
            position=[-6.68, -3.02, 1.3],
            metadata={"fan_id": "fan_int_neg_3", "distance_m": 0.18},
        )
        sim.submit_event(event)
        assert sim.get_state()["last_event"]["event"] == "high_five"
        assert sim.action == Action.IDLE
        assert not sim.safety_latched
