"""读取场景清单，并提供互动带与相机投影的只读视图。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = ROOT / "assets" / "unitree_g1" / "scene_manifest.json"


def _vec3(value: Any, field: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是三个数字") from exc
    if len(values) != 3:
        raise ValueError(f"{field} 必须是三个数字")
    return values


def _vec2(value: Any, field: str) -> tuple[float, float]:
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是两个数字") from exc
    if len(values) != 2:
        raise ValueError(f"{field} 必须是两个数字")
    if values[0] > values[1]:
        raise ValueError(f"{field} 的下界不能大于上界")
    return values


@dataclass(frozen=True)
class InteractionZone:
    """互动带的世界坐标范围；不是看台座位区。"""

    side: str
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]
    barricade_y: float
    fan_y: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractionZone":
        return cls(
            side=str(data.get("side", "neg")),
            x=_vec2(data["x"], "interaction_zone.x"),
            y=_vec2(data["y"], "interaction_zone.y"),
            z=_vec2(data["z"], "interaction_zone.z"),
            barricade_y=float(data["barricade_y"]),
            fan_y=float(data["fan_y"]),
        )

    def contains(self, point: Any, margin: float = 0.0) -> bool:
        x, y, z = (float(item) for item in point)
        return (
            self.x[0] - margin <= x <= self.x[1] + margin
            and self.y[0] - margin <= y <= self.y[1] + margin
            and self.z[0] - margin <= z <= self.z[1] + margin
        )


@dataclass(frozen=True)
class FanAnchor:
    """一名互动观众的掌心/应援牌锚点和对应 geom 名。"""

    id: str
    hand: tuple[float, float, float]
    sign: tuple[float, float, float]
    hand_geom: str
    sign_geom: str
    hand_side: str = "raised"
    hand_reaching: bool = True
    sign_raised: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FanAnchor":
        return cls(
            id=str(data["id"]),
            hand=_vec3(data["hand"], f"fan.{data['id']}.hand"),
            sign=_vec3(data["sign"], f"fan.{data['id']}.sign"),
            hand_geom=str(data["hand_geom"]),
            sign_geom=str(data["sign_geom"]),
            hand_side=str(data.get("hand_side", "raised")),
            hand_reaching=bool(data.get("hand_reaching", True)),
            sign_raised=bool(data.get("sign_raised", True)),
        )


@dataclass(frozen=True)
class CameraCalibration:
    """MuJoCo 固定相机参数；像素 v 向下，相机沿 -z 看。"""

    name: str
    pos: tuple[float, float, float]
    target: tuple[float, float, float]
    fovy_deg: float
    width: int
    height: int
    R_wc: tuple[tuple[float, float, float], ...]
    t_wc: tuple[float, float, float]
    K: tuple[tuple[float, float, float], ...]

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "CameraCalibration":
        try:
            rotation = tuple(
                _vec3(row, f"camera.{name}.R_wc[{index}]")
                for index, row in enumerate(data["R_wc"])
            )
            intrinsics = tuple(
                _vec3(row, f"camera.{name}.K[{index}]")
                for index, row in enumerate(data["K"])
            )
        except TypeError as exc:
            raise ValueError(f"camera.{name}.R_wc/K 必须是 3x3 数组") from exc
        if len(rotation) != 3 or len(intrinsics) != 3:
            raise ValueError(f"camera.{name}.R_wc/K 必须是 3x3 数组")
        return cls(
            name=name,
            pos=_vec3(data["pos"], f"camera.{name}.pos"),
            target=_vec3(data["target"], f"camera.{name}.target"),
            fovy_deg=float(data["fovy_deg"]),
            width=int(data["width"]),
            height=int(data["height"]),
            R_wc=rotation,
            t_wc=_vec3(data["t_wc"], f"camera.{name}.t_wc"),
            K=intrinsics,
        )

    def project(
        self,
        point: Any,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[float, float, float]:
        """返回 (u, v, depth)；depth 是相机前方距离。

        width/height 用于实际渲染分辨率；默认使用清单中的标称分辨率。
        """
        rotation = np.asarray(self.R_wc, dtype=float)
        camera_point = rotation.T @ (np.asarray(point, dtype=float) - np.asarray(self.t_wc, dtype=float))
        if camera_point[2] >= 0.0:
            raise ValueError(f"点 {point} 在相机 {self.name} 后方")
        image_width = self.width if width is None else int(width)
        image_height = self.height if height is None else int(height)
        if image_width <= 0 or image_height <= 0:
            raise ValueError("投影分辨率必须为正")
        focal = 0.5 * image_height / math.tan(math.radians(self.fovy_deg) / 2.0)
        u = 0.5 * image_width + focal * camera_point[0] / -camera_point[2]
        v = 0.5 * image_height - focal * camera_point[1] / -camera_point[2]
        return float(u), float(v), float(-camera_point[2])


@dataclass(frozen=True)
class SceneManifest:
    """scene_manifest.json 的只读对象模型。"""

    path: Path
    schema_version: int
    scene_hash: str
    zone: InteractionZone
    fans: tuple[FanAnchor, ...]
    cameras: dict[str, CameraCalibration]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SceneManifest":
        manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
        if not manifest_path.is_file():
            raise FileNotFoundError(f"找不到场景清单：{manifest_path}")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = int(data.get("schema_version", 0))
        if version != 1:
            raise ValueError(f"不支持的场景清单版本：{version}")
        zone = InteractionZone.from_dict(data["interaction_zone"])
        fans = tuple(FanAnchor.from_dict(item) for item in data["fans"])
        cameras = {
            str(name): CameraCalibration.from_dict(str(name), item)
            for name, item in data["cameras"].items()
        }
        for fan in fans:
            if fan.hand_reaching and not zone.contains(fan.hand):
                raise ValueError(f"{fan.id} 的伸手锚点不在互动带内：{fan.hand}")
        return cls(
            path=manifest_path,
            schema_version=version,
            scene_hash=str(data.get("scene_hash", "")),
            zone=zone,
            fans=fans,
            cameras=cameras,
        )

    def fan(self, fan_id: str) -> FanAnchor:
        for fan in self.fans:
            if fan.id == fan_id:
                return fan
        raise KeyError(f"未知互动观众：{fan_id}")

    def camera(self, name: str) -> CameraCalibration:
        try:
            return self.cameras[name]
        except KeyError:
            raise KeyError(f"未知相机：{name}") from None


def load_manifest(path: str | Path | None = None) -> SceneManifest:
    return SceneManifest.load(path)
