"""无需图形窗口的基础验收测试。"""

from __future__ import annotations

import math

from g1_arena import Action, ArenaSimulation, SimulationEvent


def test_model_and_scene() -> None:
    with ArenaSimulation() as sim:
        assert sim.model.nq == 36
        assert sim.model.nu == 29
        assert sim.model.ngeom >= 300
        assert sim.model.ncam == 3


def test_walk_reaches_aisle_end() -> None:
    with ArenaSimulation() as sim:
        start_x = sim.get_state()["position"][0]
        sim.execute(Action.WALK, target=[-6.0, 0.0, 0.87], speed=0.8)
        for _ in range(600):
            sim.step()
        state = sim.get_state()
        assert state["action"] == "idle"
        assert math.isclose(state["position"][0], -6.0, abs_tol=1e-6)
        assert state["position"][0] > start_x


def test_wave_and_return_to_idle() -> None:
    with ArenaSimulation() as sim:
        sim.execute(Action.WAVE, duration=0.2)
        sim.step(6)
        changed = sim.get_state()["joint_positions"]["right_shoulder_roll_joint"]
        assert abs(changed + 1.30) < 1e-6
        sim.step(20)
        assert sim.action == Action.IDLE


def test_event_mapping_and_safety_priority() -> None:
    with ArenaSimulation() as sim:
        sim.submit_event(SimulationEvent("crowd_cheering", 0.9, 0.0, "audio"))
        assert sim.action == Action.CELEBRATE
        sim.submit_event({"event": "emergency_stop", "confidence": 1.0, "timestamp": 0.1, "source": "test"})
        assert sim.action == Action.STOP
        sim.submit_event({"event": "audience_wave", "confidence": 1.0, "timestamp": 0.2, "source": "test"})
        assert sim.action == Action.STOP
        sim.reset()
        assert not sim.safety_latched
