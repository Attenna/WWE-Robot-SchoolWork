"""合成数据集的标签 schema 与读取工具。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


LABEL_SCHEMA = "perception.dataset.v1"
DATASET_MANIFEST = "dataset_manifest.json"


def _vec3(value: Any, field: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是三个数字") from exc
    if len(values) != 3:
        raise ValueError(f"{field} 必须是三个数字")
    return values


def _point(value: Any, field: str) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是两个数字") from exc
    if len(values) != 2:
        raise ValueError(f"{field} 必须是两个数字")
    return values


def _bbox(value: Any, field: str) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    try:
        values = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是四个整数") from exc
    if len(values) != 4:
        raise ValueError(f"{field} 必须是四个整数")
    x1, y1, x2, y2 = values
    if x2 < x1 or y2 < y1:
        raise ValueError(f"{field} 的右下角不能小于左上角")
    return x1, y1, x2, y2


@dataclass(frozen=True)
class AnchorLabel:
    """一个手或应援牌锚点的可见性、框和关键点。"""

    world: tuple[float, float, float]
    bbox: tuple[int, int, int, int] | None
    keypoint: tuple[float, float] | None
    keypoint_visible: bool
    visible: bool
    pixels: int

    @classmethod
    def from_dict(cls, data: dict[str, Any], field: str) -> "AnchorLabel":
        return cls(
            world=_vec3(data["world"], f"{field}.world"),
            bbox=_bbox(data.get("bbox"), f"{field}.bbox"),
            keypoint=_point(data.get("keypoint"), f"{field}.keypoint"),
            keypoint_visible=bool(data.get("keypoint_visible", False)),
            visible=bool(data.get("visible", data.get("bbox") is not None)),
            pixels=int(data.get("pixels", 0)),
        )


@dataclass(frozen=True)
class FanLabel:
    fan_id: str
    hand: AnchorLabel
    sign: AnchorLabel
    hand_reaching: bool = True
    sign_raised: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FanLabel":
        fan_id = str(data["fan_id"])
        return cls(
            fan_id=fan_id,
            hand=AnchorLabel.from_dict(data["hand"], f"{fan_id}.hand"),
            sign=AnchorLabel.from_dict(data["sign"], f"{fan_id}.sign"),
            hand_reaching=bool(data.get("hand_reaching", True)),
            sign_raised=bool(data.get("sign_raised", True)),
        )


@dataclass(frozen=True)
class FrameLabel:
    index: int
    image: str
    timestamp: float
    camera: str
    width: int
    height: int
    scene_hash: str
    robot_base: tuple[float, float, float]
    robot_palms: dict[str, tuple[float, float, float]]
    fans: tuple[FanLabel, ...]
    high_five_candidate: dict[str, Any] | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameLabel":
        robot = data.get("robot", {})
        palms = {
            str(side): _vec3(value, f"robot.palms.{side}")
            for side, value in robot.get("palms", {}).items()
        }
        return cls(
            index=int(data["index"]),
            image=str(data["image"]),
            timestamp=float(data["timestamp"]),
            camera=str(data["camera"]),
            width=int(data["width"]),
            height=int(data["height"]),
            scene_hash=str(data.get("scene_hash", "")),
            robot_base=_vec3(robot.get("base", [0.0, 0.0, 0.0]), "robot.base"),
            robot_palms=palms,
            fans=tuple(FanLabel.from_dict(item) for item in data["fans"]),
            high_five_candidate=data.get("high_five_candidate"),
        )


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: int
    label_schema: str
    dataset: str
    scene_hash: str
    camera: str
    width: int
    height: int
    frames: int
    seed: int
    saved_depth: bool
    saved_segmentation: bool
    images_dir: str
    labels: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetManifest":
        schema = str(data.get("label_schema", ""))
        if schema != LABEL_SCHEMA:
            raise ValueError(f"不支持的标签 schema：{schema}")
        return cls(
            schema_version=int(data["schema_version"]),
            label_schema=schema,
            dataset=str(data["dataset"]),
            scene_hash=str(data["scene_hash"]),
            camera=str(data["camera"]),
            width=int(data["width"]),
            height=int(data["height"]),
            frames=int(data["frames"]),
            seed=int(data["seed"]),
            saved_depth=bool(data.get("saved_depth", False)),
            saved_segmentation=bool(data.get("saved_segmentation", False)),
            images_dir=str(data.get("images_dir", "images")),
            labels=str(data.get("labels", "labels.jsonl")),
        )


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到数据集清单：{manifest_path}")
    return DatasetManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))


def load_frame_labels(path: str | Path) -> list[FrameLabel]:
    labels_path = Path(path)
    if not labels_path.is_file():
        raise FileNotFoundError(f"找不到标签文件：{labels_path}")
    labels: list[FrameLabel] = []
    for line_number, line in enumerate(labels_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            labels.append(FrameLabel.from_dict(json.loads(line)))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{labels_path}:{line_number} 标签格式错误：{exc}") from exc
    return labels


def load_dataset(directory: str | Path) -> tuple[DatasetManifest, list[FrameLabel]]:
    root = Path(directory)
    manifest = load_dataset_manifest(root / DATASET_MANIFEST)
    labels = load_frame_labels(root / manifest.labels)
    if len(labels) != manifest.frames:
        raise ValueError(f"标签数量 {len(labels)} 与清单 frames={manifest.frames} 不一致")
    return manifest, labels
