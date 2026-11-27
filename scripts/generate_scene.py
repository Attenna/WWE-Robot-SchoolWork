"""生成四边擂台、入场通道、舞台和两侧观众席的 MJCF 场景。"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_arena.config import (  # noqa: E402
    AISLE_CENTER_X,
    AISLE_HALF_LENGTH,
    STAGE_SCREEN_X,
    STAGE_TRUSS_X,
    STAGE_X,
)

OUTPUT = ROOT / "assets" / "unitree_g1" / "arena.xml"


def geom(name: str, body: str) -> str:
    return f'    <geom name="{name}" {body}/>'


def camera(name: str, position: tuple[float, float, float], target: tuple[float, float, float], fovy: float) -> str:
    """MuJoCo 相机局部 -z 指向目标，局部 +y 朝上。"""
    eye = np.array(position, dtype=float)
    forward = np.array(target, dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    back = -forward
    up = np.cross(back, right)
    axes = " ".join(f"{value:.8f}" for value in np.concatenate([right, up]))
    pos = " ".join(str(value) for value in position)
    return f'    <camera name="{name}" pos="{pos}" xyaxes="{axes}" fovy="{fovy}"/>'


def audience_member(name: str, x: float, y: float, base_z: float, color: str, wave: bool) -> list[str]:
    """用低成本几何体构造坐姿观众，不参与物理碰撞。"""
    common = 'contype="0" conaffinity="0" group="1"'
    items = [
        geom(name + "_seat", f'type="box" pos="{x:.2f} {y:.2f} {base_z + 0.28:.2f}" size="0.24 0.25 0.06" rgba="0.10 0.12 0.18 1" {common}'),
        geom(name + "_body", f'type="capsule" pos="{x:.2f} {y:.2f} {base_z + 0.68:.2f}" size="0.17 0.25" rgba="{color}" {common}'),
        geom(name + "_head", f'type="sphere" pos="{x:.2f} {y:.2f} {base_z + 1.10:.2f}" size="0.14" rgba="0.72 0.52 0.38 1" {common}'),
    ]
    for side in (-1, 1):
        arm_y = y + side * 0.19
        end_z = base_z + (1.45 if wave and side == 1 else 0.55)
        items.append(
            geom(
                f"{name}_arm_{side}",
                f'type="capsule" fromto="{x:.2f} {arm_y:.2f} {base_z + 0.88:.2f} {x + 0.05:.2f} {arm_y + side * 0.08:.2f} {end_z:.2f}" size="0.055" rgba="{color}" {common}',
            )
        )
    return items


def build() -> str:
    out = [
        '<mujoco model="G1 interactive entrance arena">',
        '  <include file="g1.xml"/>',
        '  <option timestep="0.008333333" integrator="implicitfast"/>',
        '  <visual>',
        '    <headlight diffuse="0.55 0.55 0.55" ambient="0.28 0.28 0.34" specular="0.4 0.4 0.4"/>',
        '    <rgba haze="0.025 0.035 0.07 1"/>',
        '    <global azimuth="135" elevation="-24" offwidth="1280" offheight="720"/>',
        '    <quality shadowsize="4096" offsamples="4"/>',
        '    <scale framelength="0.5" framewidth="0.008"/>',
        '  </visual>',
        '  <asset>',
        '    <texture type="skybox" builtin="gradient" rgb1="0.035 0.05 0.10" rgb2="0.005 0.008 0.02" width="512" height="1024"/>',
        '    <texture type="2d" name="floor_grid" builtin="checker" rgb1="0.035 0.045 0.07" rgb2="0.055 0.065 0.09" width="256" height="256"/>',
        '    <material name="arena_floor" texture="floor_grid" texrepeat="16 16" reflectance="0.05"/>',
        '    <material name="canvas" rgba="0.76 0.80 0.88 1"/>',
        '    <material name="apron" rgba="0.015 0.025 0.07 1"/>',
        '    <material name="steel" rgba="0.26 0.30 0.38 1" specular="0.7"/>',
        '    <material name="blue_led" rgba="0.02 0.25 0.95 1" emission="0.8"/>',
        '    <material name="red_led" rgba="0.85 0.025 0.06 1" emission="0.6"/>',
        '  </asset>',
        '  <worldbody>',
        '    <light name="key" pos="-3 -3 9" dir="0 0 -1" directional="true" diffuse="0.75 0.82 1"/>',
        '    <light name="fill" pos="-5 5 7" dir="0 0 -1" directional="true" diffuse="0.50 0.55 0.70"/>',
        '    <light name="ring_spot" pos="0 0 7" dir="0 0 -1" directional="false" attenuation="0.05 0.02 0.005" diffuse="1 0.92 0.80"/>',
        geom("arena_floor", 'type="plane" pos="0 0 -0.01" size="40 24 0.1" material="arena_floor"'),
        '    <!-- 加长入场通道：有效行走路径 20 m，默认 40 s 完成 -->',
        geom("entrance_aisle", f'type="box" pos="{AISLE_CENTER_X:.2f} 0 0.035" size="{AISLE_HALF_LENGTH:.2f} 1.10 0.045" rgba="0.10 0.13 0.22 1"'),
        geom("aisle_center", f'type="box" pos="{AISLE_CENTER_X:.2f} 0 0.083" size="{AISLE_HALF_LENGTH - 0.05:.2f} 0.055 0.006" material="blue_led" contype="0" conaffinity="0"'),
        geom("aisle_edge_left", f'type="box" pos="{AISLE_CENTER_X:.2f} 1.12 0.10" size="{AISLE_HALF_LENGTH:.2f} 0.045 0.10" material="blue_led" contype="0" conaffinity="0"'),
        geom("aisle_edge_right", f'type="box" pos="{AISLE_CENTER_X:.2f} -1.12 0.10" size="{AISLE_HALF_LENGTH:.2f} 0.045 0.10" material="red_led" contype="0" conaffinity="0"'),
        geom("entrance_stage", f'type="box" pos="{STAGE_X:.2f} 0 0.16" size="1.70 2.60 0.16" rgba="0.07 0.09 0.15 1"'),
        geom("stage_screen", f'type="box" pos="{STAGE_SCREEN_X:.2f} 0 2.05" size="0.12 1.55 0.90" material="blue_led" contype="0" conaffinity="0"'),
        geom("stage_screen_inner", f'type="box" pos="{STAGE_SCREEN_X + 0.15:.2f} 0 2.05" size="0.025 1.35 0.72" rgba="0.015 0.025 0.07 1" contype="0" conaffinity="0"'),
        geom("stage_logo_vertical", f'type="box" pos="{STAGE_SCREEN_X + 0.19:.2f} 0 2.05" size="0.018 0.13 0.50" material="red_led" contype="0" conaffinity="0"'),
        geom("stage_logo_horizontal", f'type="box" pos="{STAGE_SCREEN_X + 0.20:.2f} 0 2.05" size="0.018 0.58 0.13" material="red_led" contype="0" conaffinity="0"'),
    ]
    for side in (-1, 1):
        out.append(geom(f"stage_truss_{side}", f'type="box" pos="{STAGE_TRUSS_X:.2f} {side * 1.90} 1.85" size="0.13 0.13 1.85" material="steel" contype="0" conaffinity="0"'))
    out.extend([
        geom("stage_truss_top", f'type="box" pos="{STAGE_TRUSS_X:.2f} 0 3.65" size="0.13 2.03 0.13" material="steel" contype="0" conaffinity="0"'),
        '    <!-- 20 英尺级四边擂台；台面含围裙约 6.7 米见方 -->',
        geom("ring_base", 'type="box" pos="0 0 0.50" size="3.35 3.35 0.50" material="apron"'),
        geom("ring_canvas", 'type="box" pos="0 0 1.015" size="3.35 3.35 0.025" material="canvas" friction="0.9 0.02 0.01"'),
        geom("canvas_center", 'type="cylinder" pos="0 0 1.047" size="0.85 0.008" rgba="0.04 0.12 0.36 1" contype="0" conaffinity="0"'),
        geom("canvas_center_inner", 'type="cylinder" pos="0 0 1.057" size="0.65 0.008" rgba="0.80 0.06 0.08 1" contype="0" conaffinity="0"'),
    ])
    rope_colors = ["0.85 0.04 0.06 1", "0.90 0.90 0.92 1", "0.03 0.20 0.85 1"]
    rope_heights = [1.55, 2.00, 2.45]
    for level, (z, color) in enumerate(zip(rope_heights, rope_colors), start=1):
        out.extend([
            geom(f"rope_north_{level}", f'type="cylinder" fromto="-3.05 3.05 {z} 3.05 3.05 {z}" size="0.027" rgba="{color}" contype="0" conaffinity="0"'),
            geom(f"rope_south_{level}", f'type="cylinder" fromto="-3.05 -3.05 {z} 3.05 -3.05 {z}" size="0.027" rgba="{color}" contype="0" conaffinity="0"'),
            geom(f"rope_west_{level}", f'type="cylinder" fromto="-3.05 -3.05 {z} -3.05 3.05 {z}" size="0.027" rgba="{color}" contype="0" conaffinity="0"'),
            geom(f"rope_east_{level}", f'type="cylinder" fromto="3.05 -3.05 {z} 3.05 3.05 {z}" size="0.027" rgba="{color}" contype="0" conaffinity="0"'),
        ])
    for x in (-3.18, 3.18):
        for y in (-3.18, 3.18):
            corner = f"{x}_{y}"
            out.append(geom(f"post_{corner}", f'type="cylinder" pos="{x} {y} 1.27" size="0.075 1.27" material="steel"'))
            for level, z in enumerate(rope_heights, start=1):
                out.append(geom(f"turnbuckle_{corner}_{level}", f'type="box" pos="{x * 0.955:.3f} {y * 0.955:.3f} {z}" size="0.13 0.13 0.09" rgba="0.04 0.06 0.12 1" contype="0" conaffinity="0"'))
    out.extend([
        geom("step_low", 'type="box" pos="-3.95 -2.45 0.12" size="0.55 0.50 0.12" material="steel"'),
        geom("step_mid", 'type="box" pos="-3.72 -2.45 0.31" size="0.38 0.50 0.12" material="steel"'),
        geom("step_high", 'type="box" pos="-3.52 -2.45 0.50" size="0.22 0.50 0.12" material="steel"'),
        '    <!-- 两侧阶梯观众席与护栏 -->',
    ])
    colors = ["0.15 0.35 0.80 1", "0.75 0.08 0.12 1", "0.12 0.55 0.35 1", "0.65 0.45 0.12 1"]
    for side in (-1, 1):
        out.append(geom(f"barricade_{side}", f'type="box" pos="-1.3 {side * 4.35} 0.55" size="5.6 0.06 0.55" rgba="0.12 0.16 0.25 1" contype="0" conaffinity="0"'))
        for row in range(3):
            y = side * (5.35 + row * 1.15)
            height = 0.16 + row * 0.32
            out.append(geom(f"stand_{side}_{row}", f'type="box" pos="-1.35 {y:.2f} {height / 2:.2f}" size="5.25 0.55 {height / 2:.2f}" rgba="0.10 0.12 0.19 1" contype="0" conaffinity="0"'))
            for col in range(8):
                x = -5.55 + col * 1.20
                out.extend(audience_member(f"fan_{side}_{row}_{col}", x, y, height, colors[(row + col) % len(colors)], wave=(row + col) % 4 == 0))
    out.extend([
        camera("overview", (-31.0, -22.0, 14.0), (-11.5, 0.0, 0.9), 54),
        camera("entrance", (-29.0, 5.5, 4.5), (-22.0, 0.0, 1.0), 55),
        camera("ringside", (6.8, -8.0, 4.8), (-0.5, 0.0, 1.0), 52),
        '  </worldbody>',
        '</mujoco>',
        '',
    ])
    return "\n".join(out)


if __name__ == "__main__":
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"已生成场景：{OUTPUT}")
