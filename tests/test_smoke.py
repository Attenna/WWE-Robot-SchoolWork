"""无需图形窗口的基础验收测试。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from g1_arena import Action, ArenaSimulation, SimulationEvent
from scripts.generate_scene import (
    INTERACTION_BARRICADE_Y,
    INTERACTION_FAN_COUNT,
    INTERACTION_X_MAX,
    INTERACTION_X_MIN,
    INTERACTION_ZONE_Y_MAX,
    INTERACTION_ZONE_Y_MIN,
)


ROOT = Path(__file__).resolve().parents[1]
ARENA_PATH = ROOT / "assets" / "unitree_g1" / "arena.xml"
MANIFEST_PATH = ROOT / "assets" / "unitree_g1" / "scene_manifest.json"


def project_point(point: list[float], camera: dict[str, object]) -> tuple[float, float, float]:
    """按 manifest 的 MuJoCo 相机约定投影到像素。"""
    rotation = np.asarray(camera["R_wc"], dtype=float)
    translation = np.asarray(camera["t_wc"], dtype=float)
    camera_point = rotation.T @ (np.asarray(point, dtype=float) - translation)
    assert camera_point[2] < 0.0
    focal = 0.5 * float(camera["height"]) / math.tan(math.radians(float(camera["fovy_deg"])) / 2.0)
    u = 0.5 * float(camera["width"]) + focal * camera_point[0] / -camera_point[2]
    v = 0.5 * float(camera["height"]) - focal * camera_point[1] / -camera_point[2]
    return float(u), float(v), float(-camera_point[2])


def test_model_and_scene() -> None:
    with ArenaSimulation() as sim:
        assert sim.model.nq == 36
        assert sim.model.nu == 29
        assert sim.model.ngeom >= 300
        camera_names = {sim.model.camera(index).name for index in range(sim.model.ncam)}
        assert {"overview", "entrance", "ringside", "interaction_neg"} <= camera_names


def test_interaction_front_row_manifest() -> None:
    xml = ARENA_PATH.read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["scene_hash"] == hashlib.sha256(xml.encode("utf-8")).hexdigest()

    zone = manifest["interaction_zone"]
    assert zone["side"] == "neg"
    assert zone["x"] == [round(INTERACTION_X_MIN, 3), round(INTERACTION_X_MAX, 3)]
    assert zone["y"] == [INTERACTION_ZONE_Y_MIN, INTERACTION_ZONE_Y_MAX]
    assert zone["barricade_y"] == INTERACTION_BARRICADE_Y
    assert len(manifest["fans"]) == INTERACTION_FAN_COUNT
    assert manifest["audience"] == {"seated": 48, "interaction": INTERACTION_FAN_COUNT}

    with ArenaSimulation() as sim:
        geom_names = {sim.model.geom(index).name for index in range(sim.model.ngeom)}
        required_geoms = {
            "barricade_int_neg",
            "barricade_int_neg_top",
            "interaction_zone_edge_aisle",
            "interaction_zone_edge_barricade",
            "interaction_zone_end_west",
            "interaction_zone_end_east",
        }
        assert required_geoms <= geom_names
        assert any(fan["hand_reaching"] for fan in manifest["fans"])
        assert any(not fan["hand_reaching"] for fan in manifest["fans"])
        assert any(fan["sign_raised"] for fan in manifest["fans"])
        assert any(not fan["sign_raised"] for fan in manifest["fans"])
        for fan in manifest["fans"]:
            for suffix in ("legs", "body", "head", "arm", "hand", "sign", "sign_handle"):
                assert f"{fan['id']}_{suffix}" in geom_names
            hand = fan["hand"]
            sign = fan["sign"]
            if fan["hand_reaching"]:
                assert zone["y"][0] < hand[1] < zone["y"][1]
                assert 1.0 <= hand[2] <= 1.5
                assert hand[1] > zone["barricade_y"]
            if fan["sign_raised"]:
                assert 1.7 <= sign[2] <= 2.0

        camera = manifest["cameras"]["interaction_neg"]
        for fan in manifest["fans"]:
            for point in (fan["hand"], fan["sign"]):
                u, v, depth = project_point(point, camera)
                assert depth > 0.0
                assert 0.0 <= u < camera["width"]
                assert 0.0 <= v < camera["height"]


def test_render_multiple_resolutions() -> None:
    with ArenaSimulation() as sim:
        small = sim.render("overview", 320, 240)
        large = sim.render("overview", 640, 480)
        assert small.shape == (240, 320, 3)
        assert large.shape == (480, 640, 3)


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
