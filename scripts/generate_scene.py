"""生成四边擂台、入场通道、舞台、两侧观众席和互动前排的 MJCF 场景。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "unitree_g1" / "arena.xml"
OUTPUT_MANIFEST = ROOT / "assets" / "unitree_g1" / "scene_manifest.json"

AISLE_CENTER_X = -7.60
AISLE_HALF_LENGTH = 3.85
AISLE_HALF_WIDTH = 1.10
AISLE_X_MIN = AISLE_CENTER_X - AISLE_HALF_LENGTH
AISLE_X_MAX = AISLE_CENTER_X + AISLE_HALF_LENGTH
AISLE_TOP_Z = 0.08

# 互动带：通道边缘到第一排护栏之间、机器人可以离开通道走进去的地带。
# 两侧各有一侧看台；当前先实现截图里的 -y 侧，后续可按 side 参数镜像。
INTERACTION_X_MIN = AISLE_X_MIN + 2.15  # 避开入口舞台
INTERACTION_X_MAX = AISLE_X_MAX - 1.05  # 避开擂台台阶
INTERACTION_ZONE_Y_MIN = -3.10
INTERACTION_ZONE_Y_MAX = -1.20
INTERACTION_BARRICADE_Y = INTERACTION_ZONE_Y_MIN
INTERACTION_FAN_Y = -3.65
INTERACTION_HAND_Y = -3.02
INTERACTION_HAND_Z = 1.30
INTERACTION_SIGN_Y = -3.05
INTERACTION_SIGN_Z = 1.85
INTERACTION_ZONE_Z_MAX = 2.10
INTERACTION_FAN_COUNT = max(4, int((INTERACTION_X_MAX - INTERACTION_X_MIN) / 0.80) + 1)

AUDIENCE_COLORS = ["0.15 0.35 0.80 1", "0.75 0.08 0.12 1", "0.12 0.55 0.35 1", "0.65 0.45 0.12 1"]
SIGN_COLORS = ["0.95 0.75 0.10 1", "0.90 0.15 0.20 1", "0.15 0.55 0.95 1", "0.15 0.80 0.45 1"]


@dataclass(frozen=True)
class CameraSpec:
    """固定相机；同时负责生成 MJCF 和给 CV 的标定清单。"""

    name: str
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    fovy: float
    width: int = 1280
    height: int = 720

    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回相机局部 x/y/z 轴在世界坐标中的方向；相机沿 -z 看。"""
        eye = np.array(self.position, dtype=float)
        forward = np.array(self.target, dtype=float) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        back = -forward
        up = np.cross(back, right)
        return right, up, back

    def xml(self) -> str:
        right, up, _ = self.axes()
        xyaxes = " ".join(f"{value:.8f}" for value in np.concatenate([right, up]))
        pos = " ".join(str(value) for value in self.position)
        return f'    <camera name="{self.name}" pos="{pos}" xyaxes="{xyaxes}" fovy="{self.fovy}"/>'

    def manifest(self) -> dict[str, object]:
        right, up, back = self.axes()
        rotation = np.column_stack([right, up, back])
        focal = 0.5 * self.height / np.tan(np.deg2rad(self.fovy) / 2.0)
        intrinsics = np.array(
            [
                [focal, 0.0, self.width / 2.0],
                [0.0, focal, self.height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return {
            "pos": [float(value) for value in self.position],
            "target": [float(value) for value in self.target],
            "fovy_deg": float(self.fovy),
            "width": self.width,
            "height": self.height,
            "xyaxes": [float(value) for value in np.concatenate([right, up])],
            "R_wc": rotation.tolist(),
            "t_wc": [float(value) for value in self.position],
            "K": intrinsics.tolist(),
            "intrinsics_note": "K uses this width/height; for other resolutions recompute f from fovy_deg.",
            "projection": (
                "p_c = R_wc.T @ (p_w - t_wc); "
                "u = cx + f * p_c.x / -p_c.z; "
                "v = cy - f * p_c.y / -p_c.z"
            ),
        }


CAMERAS = [
    CameraSpec("overview", (-13.5, -12.5, 9.0), (-2.2, 0.0, 0.9), 52),
    CameraSpec("entrance", (-13.8, 4.2, 4.0), (-6.3, 0.0, 1.0), 55),
    CameraSpec("ringside", (6.8, -8.0, 4.8), (-0.5, 0.0, 1.0), 52),
    CameraSpec("interaction_neg", (-10.4, -0.9, 2.8), (-6.1, -3.1, 1.05), 65),
]


def geom(name: str, body: str) -> str:
    return f'    <geom name="{name}" {body}/>'


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


def interaction_fan_xs() -> list[float]:
    span = INTERACTION_X_MAX - INTERACTION_X_MIN
    if INTERACTION_FAN_COUNT <= 1:
        return [0.5 * (INTERACTION_X_MIN + INTERACTION_X_MAX)]
    margin = 0.40
    step = (span - 2.0 * margin) / (INTERACTION_FAN_COUNT - 1)
    return [INTERACTION_X_MIN + margin + index * step for index in range(INTERACTION_FAN_COUNT)]


def interaction_fan(name: str, x: float, color: str, sign_color: str) -> list[str]:
    """站立互动观众；手和应援牌是独立 geom，便于 CV 分割与标注。"""
    common = 'contype="0" conaffinity="0" group="1"'
    y = INTERACTION_FAN_Y
    return [
        geom(name + "_legs", f'type="capsule" fromto="{x:.2f} {y:.2f} 0.04 {x:.2f} {y:.2f} 0.58" size="0.12" rgba="0.08 0.10 0.16 1" {common}'),
        geom(name + "_body", f'type="capsule" pos="{x:.2f} {y:.2f} 0.95" size="0.17 0.30" rgba="{color}" {common}'),
        geom(name + "_head", f'type="sphere" pos="{x:.2f} {y:.2f} 1.55" size="0.14" rgba="0.72 0.52 0.38 1" {common}'),
        geom(name + "_arm", f'type="capsule" fromto="{x:.2f} {y:.2f} 1.25 {x:.2f} {INTERACTION_HAND_Y + 0.04:.2f} 1.28" size="0.055" rgba="{color}" {common}'),
        geom(name + "_hand", f'type="sphere" pos="{x:.2f} {INTERACTION_HAND_Y:.2f} {INTERACTION_HAND_Z:.2f}" size="0.075" rgba="0.78 0.58 0.42 1" {common}'),
        geom(name + "_sign", f'type="box" pos="{x:.2f} {INTERACTION_SIGN_Y:.2f} {INTERACTION_SIGN_Z:.2f}" size="0.22 0.02 0.15" rgba="{sign_color}" {common}'),
        geom(name + "_sign_handle", f'type="capsule" fromto="{x:.2f} {INTERACTION_HAND_Y:.2f} 1.33 {x:.2f} {INTERACTION_SIGN_Y:.2f} 1.70" size="0.02" rgba="0.35 0.32 0.28 1" {common}'),
    ]


def interaction_group() -> list[str]:
    center = 0.5 * (INTERACTION_X_MIN + INTERACTION_X_MAX)
    half_length = 0.5 * (INTERACTION_X_MAX - INTERACTION_X_MIN)
    zone_center_y = 0.5 * (INTERACTION_ZONE_Y_MIN + INTERACTION_ZONE_Y_MAX)
    zone_half_y = 0.5 * (INTERACTION_ZONE_Y_MAX - INTERACTION_ZONE_Y_MIN)
    out = [
        '    <!-- 入场通道旁的互动前排：站立观众、手/应援牌、互动带边界 -->',
        geom("barricade_int_neg", f'type="box" pos="{center:.2f} {INTERACTION_BARRICADE_Y:.2f} 0.50" size="{half_length:.2f} 0.06 0.50" rgba="0.12 0.16 0.25 1" contype="0" conaffinity="0"'),
        geom("barricade_int_neg_top", f'type="box" pos="{center:.2f} {INTERACTION_BARRICADE_Y:.2f} 1.02" size="{half_length:.2f} 0.07 0.03" material="steel" contype="0" conaffinity="0"'),
        geom("interaction_zone_edge_aisle", f'type="box" pos="{center:.2f} {INTERACTION_ZONE_Y_MAX:.2f} 0.02" size="{half_length:.2f} 0.03 0.006" material="green_led" contype="0" conaffinity="0"'),
        geom("interaction_zone_edge_barricade", f'type="box" pos="{center:.2f} {INTERACTION_BARRICADE_Y:.2f} 0.02" size="{half_length:.2f} 0.03 0.006" material="green_led" contype="0" conaffinity="0"'),
        geom("interaction_zone_end_west", f'type="box" pos="{INTERACTION_X_MIN:.2f} {zone_center_y:.2f} 0.02" size="0.03 {zone_half_y:.2f} 0.006" material="green_led" contype="0" conaffinity="0"'),
        geom("interaction_zone_end_east", f'type="box" pos="{INTERACTION_X_MAX:.2f} {zone_center_y:.2f} 0.02" size="0.03 {zone_half_y:.2f} 0.006" material="green_led" contype="0" conaffinity="0"'),
    ]
    for index, x in enumerate(interaction_fan_xs()):
        out.extend(
            interaction_fan(
                f"fan_int_neg_{index}",
                x,
                AUDIENCE_COLORS[index % len(AUDIENCE_COLORS)],
                SIGN_COLORS[index % len(SIGN_COLORS)],
            )
        )
    return out


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
        '    <material name="green_led" rgba="0.05 0.90 0.35 1" emission="0.7"/>',
        '  </asset>',
        '  <worldbody>',
        '    <light name="key" pos="-3 -3 9" dir="0 0 -1" directional="true" diffuse="0.75 0.82 1"/>',
        '    <light name="fill" pos="-5 5 7" dir="0 0 -1" directional="true" diffuse="0.50 0.55 0.70"/>',
        '    <light name="ring_spot" pos="0 0 7" dir="0 0 -1" directional="false" attenuation="0.05 0.02 0.005" diffuse="1 0.92 0.80"/>',
        geom("arena_floor", 'type="plane" pos="0 0 -0.01" size="30 24 0.1" material="arena_floor"'),
        '    <!-- 入场通道与舞台：x 轴正方向通向擂台 -->',
        geom("entrance_aisle", f'type="box" pos="{AISLE_CENTER_X} 0 0.035" size="{AISLE_HALF_LENGTH} {AISLE_HALF_WIDTH} 0.045" rgba="0.10 0.13 0.22 1"'),
        geom("aisle_center", f'type="box" pos="{AISLE_CENTER_X} 0 0.083" size="{AISLE_HALF_LENGTH} 0.055 0.006" material="blue_led" contype="0" conaffinity="0"'),
        geom("aisle_edge_left", f'type="box" pos="{AISLE_CENTER_X} 1.12 0.10" size="{AISLE_HALF_LENGTH} 0.045 0.10" material="blue_led" contype="0" conaffinity="0"'),
        geom("aisle_edge_right", f'type="box" pos="{AISLE_CENTER_X} -1.12 0.10" size="{AISLE_HALF_LENGTH} 0.045 0.10" material="red_led" contype="0" conaffinity="0"'),
        geom("entrance_stage", 'type="box" pos="-11.60 0 0.16" size="1.70 2.60 0.16" rgba="0.07 0.09 0.15 1"'),
        geom("stage_screen", 'type="box" pos="-12.55 0 2.05" size="0.12 1.55 0.90" material="blue_led" contype="0" conaffinity="0"'),
        geom("stage_screen_inner", 'type="box" pos="-12.40 0 2.05" size="0.025 1.35 0.72" rgba="0.015 0.025 0.07 1" contype="0" conaffinity="0"'),
        geom("stage_logo_vertical", 'type="box" pos="-12.36 0 2.05" size="0.018 0.13 0.50" material="red_led" contype="0" conaffinity="0"'),
        geom("stage_logo_horizontal", 'type="box" pos="-12.35 0 2.05" size="0.018 0.58 0.13" material="red_led" contype="0" conaffinity="0"'),
    ]
    for side in (-1, 1):
        out.append(geom(f"stage_truss_{side}", f'type="box" pos="-11.85 {side * 1.90} 1.85" size="0.13 0.13 1.85" material="steel" contype="0" conaffinity="0"'))
    out.extend([
        geom("stage_truss_top", 'type="box" pos="-11.85 0 3.65" size="0.13 2.03 0.13" material="steel" contype="0" conaffinity="0"'),
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
    for side in (-1, 1):
        out.append(geom(f"barricade_{side}", f'type="box" pos="-1.3 {side * 4.35} 0.55" size="5.6 0.06 0.55" rgba="0.12 0.16 0.25 1" contype="0" conaffinity="0"'))
        for row in range(3):
            y = side * (5.35 + row * 1.15)
            height = 0.16 + row * 0.32
            out.append(geom(f"stand_{side}_{row}", f'type="box" pos="-1.35 {y:.2f} {height / 2:.2f}" size="5.25 0.55 {height / 2:.2f}" rgba="0.10 0.12 0.19 1" contype="0" conaffinity="0"'))
            for col in range(8):
                x = -5.55 + col * 1.20
                out.extend(
                    audience_member(
                        f"fan_{side}_{row}_{col}",
                        x,
                        y,
                        height,
                        AUDIENCE_COLORS[(row + col) % len(AUDIENCE_COLORS)],
                        wave=(row + col) % 4 == 0,
                    )
                )
    out.extend(interaction_group())
    out.extend(cam.xml() for cam in CAMERAS)
    out.extend([
        '  </worldbody>',
        '</mujoco>',
        '',
    ])
    return "\n".join(out)


def build_manifest(xml: str, scene_hash: str | None = None) -> dict[str, object]:
    fans = []
    for index, x in enumerate(interaction_fan_xs()):
        fans.append(
            {
                "id": f"fan_int_neg_{index}",
                "hand_side": "raised",
                "x": round(x, 3),
                "body": [round(x, 3), INTERACTION_FAN_Y, 0.95],
                "hand": [round(x, 3), INTERACTION_HAND_Y, INTERACTION_HAND_Z],
                "sign": [round(x, 3), INTERACTION_SIGN_Y, INTERACTION_SIGN_Z],
                "hand_geom": f"fan_int_neg_{index}_hand",
                "sign_geom": f"fan_int_neg_{index}_sign",
            }
        )
    return {
        "schema_version": 1,
        "generated_by": "scripts/generate_scene.py",
        "scene": "G1 interactive entrance arena",
        "scene_hash": scene_hash or hashlib.sha256(xml.encode("utf-8")).hexdigest(),
        "scene_hash_note": "SHA-256 of the UTF-8 XML normalized to LF newlines.",
        "coordinate_system": "MuJoCo world; x toward ring; y toward side stands; z up; meters; quaternion wxyz",
        "aisle": {
            "x": [round(AISLE_X_MIN, 3), round(AISLE_X_MAX, 3)],
            "y": [round(-AISLE_HALF_WIDTH, 3), round(AISLE_HALF_WIDTH, 3)],
            "top_z": AISLE_TOP_Z,
        },
        "interaction_zone": {
            "side": "neg",
            "x": [round(INTERACTION_X_MIN, 3), round(INTERACTION_X_MAX, 3)],
            "y": [INTERACTION_ZONE_Y_MIN, INTERACTION_ZONE_Y_MAX],
            "z": [0.0, INTERACTION_ZONE_Z_MAX],
            "barricade_y": INTERACTION_BARRICADE_Y,
            "fan_y": INTERACTION_FAN_Y,
            "hand_y": INTERACTION_HAND_Y,
            "hand_z": INTERACTION_HAND_Z,
            "sign_y": INTERACTION_SIGN_Y,
            "sign_z": INTERACTION_SIGN_Z,
        },
        "audience": {"seated": 48, "interaction": len(fans)},
        "fans": fans,
        "cameras": {cam.name: cam.manifest() for cam in CAMERAS},
    }


if __name__ == "__main__":
    xml = build()
    OUTPUT.write_text(xml, encoding="utf-8", newline="\r\n")
    manifest = build_manifest(xml)
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\r\n")
    print(f"已生成场景：{OUTPUT}")
    print(f"已生成清单：{OUTPUT_MANIFEST}")
