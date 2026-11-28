"""真实评估集 schema：手/应援牌/击掌标注和 clip 清单。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


REAL_EVAL_SCHEMA = "real-eval.v1"
KEYPOINT_COUNTS = {5, 21}
VISIBILITIES = {"full", "partial", "occluded"}
SHOT_TYPES = {"wide", "medium", "close", "unknown"}
BLUR_LEVELS = {"none", "mild", "heavy", "unknown"}
HAND_SIDES = {"left", "right", "unknown"}
HAND_ROLES = {"fan", "wrestler", "unknown"}


def _bbox(value: Any, field: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是四个整数") from exc
    if len(values) != 4:
        raise ValueError(f"{field} 必须是四个整数")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{field} 的右下角必须大于左上角")
    return values


def _keypoints(value: Any, field: str) -> tuple[tuple[float, float, int], ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field} 必须是列表")
    points = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) not in (2, 3):
            raise ValueError(f"{field}[{index}] 必须是 [u,v] 或 [u,v,visibility]")
        u, v = float(item[0]), float(item[1])
        visibility = int(item[2]) if len(item) == 3 else 1
        if visibility not in (0, 1):
            raise ValueError(f"{field}[{index}] visibility 只能是 0/1")
        points.append((u, v, visibility))
    return tuple(points)


@dataclass(frozen=True)
class RealHand:
    hand_id: str
    bbox: tuple[int, int, int, int]
    side: str = "unknown"
    visibility: str = "full"
    keypoints: tuple[tuple[float, float, int], ...] = ()
    keypoint_format: int = 0
    glove: bool = False
    wristband: bool = False
    role: str = "fan"

    @classmethod
    def from_dict(cls, data: dict[str, Any], field: str) -> "RealHand":
        hand_id = str(data.get("hand_id", "")).strip()
        if not hand_id:
            raise ValueError(f"{field}.hand_id 不能为空")
        side = str(data.get("side", "unknown"))
        visibility = str(data.get("visibility", "full"))
        role = str(data.get("role", "fan"))
        if side not in HAND_SIDES:
            raise ValueError(f"{field}.side 必须是 {sorted(HAND_SIDES)}")
        if visibility not in VISIBILITIES:
            raise ValueError(f"{field}.visibility 必须是 {sorted(VISIBILITIES)}")
        if role not in HAND_ROLES:
            raise ValueError(f"{field}.role 必须是 {sorted(HAND_ROLES)}")
        keypoints = _keypoints(data.get("keypoints", []), f"{field}.keypoints")
        keypoint_format = int(data.get("keypoint_format", 0))
        if keypoints and keypoint_format not in KEYPOINT_COUNTS:
            raise ValueError(f"{field}.keypoint_format 必须是 5 或 21")
        if keypoints and len(keypoints) != keypoint_format:
            raise ValueError(f"{field}.keypoints 数量 {len(keypoints)} 与 keypoint_format 不一致")
        return cls(
            hand_id=hand_id,
            bbox=_bbox(data["bbox"], f"{field}.bbox"),
            side=side,
            visibility=visibility,
            keypoints=keypoints,
            keypoint_format=keypoint_format,
            glove=bool(data.get("glove", False)),
            wristband=bool(data.get("wristband", False)),
            role=role,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hand_id": self.hand_id,
            "bbox": list(self.bbox),
            "side": self.side,
            "visibility": self.visibility,
            "keypoints": [list(point) for point in self.keypoints],
            "keypoint_format": self.keypoint_format,
            "glove": self.glove,
            "wristband": self.wristband,
            "role": self.role,
        }


@dataclass(frozen=True)
class RealSign:
    bbox: tuple[int, int, int, int]
    text: str = ""
    occludes_hands: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any], field: str) -> "RealSign":
        return cls(
            bbox=_bbox(data["bbox"], f"{field}.bbox"),
            text=str(data.get("text", "")),
            occludes_hands=bool(data.get("occludes_hands", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "text": self.text,
            "occludes_hands": self.occludes_hands,
        }


@dataclass(frozen=True)
class RealHighFive:
    present: bool
    fan_hand_id: str = ""
    wrestler_hand_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], field: str) -> "RealHighFive":
        present = bool(data.get("present", False))
        fan_hand_id = str(data.get("fan_hand_id", ""))
        wrestler_hand_id = str(data.get("wrestler_hand_id", ""))
        if present and (not fan_hand_id or not wrestler_hand_id):
            raise ValueError(f"{field} 为 true 时必须给出 fan_hand_id 和 wrestler_hand_id")
        return cls(present=present, fan_hand_id=fan_hand_id, wrestler_hand_id=wrestler_hand_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "fan_hand_id": self.fan_hand_id,
            "wrestler_hand_id": self.wrestler_hand_id,
        }


@dataclass(frozen=True)
class RealFrameLabel:
    schema_version: int
    clip_id: str
    source_url: str
    timecode_s: float
    image: str
    width: int
    height: int
    shot: str
    blur: str
    hands: tuple[RealHand, ...]
    signs: tuple[RealSign, ...]
    high_five: RealHighFive | None
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RealFrameLabel":
        schema_version = int(data.get("schema_version", 0))
        if schema_version != 1:
            raise ValueError(f"不支持的 real-eval schema：{schema_version}")
        shot = str(data.get("shot", "unknown"))
        blur = str(data.get("blur", "unknown"))
        if shot not in SHOT_TYPES:
            raise ValueError(f"shot 必须是 {sorted(SHOT_TYPES)}")
        if blur not in BLUR_LEVELS:
            raise ValueError(f"blur 必须是 {sorted(BLUR_LEVELS)}")
        high_five_data = data.get("high_five")
        return cls(
            schema_version=schema_version,
            clip_id=str(data["clip_id"]),
            source_url=str(data["source_url"]),
            timecode_s=float(data["timecode_s"]),
            image=str(data["image"]),
            width=int(data["width"]),
            height=int(data["height"]),
            shot=shot,
            blur=blur,
            hands=tuple(
                RealHand.from_dict(item, f"hands[{index}]")
                for index, item in enumerate(data.get("hands", []))
            ),
            signs=tuple(
                RealSign.from_dict(item, f"signs[{index}]")
                for index, item in enumerate(data.get("signs", []))
            ),
            high_five=(
                RealHighFive.from_dict(high_five_data, "high_five")
                if isinstance(high_five_data, dict)
                else None
            ),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "clip_id": self.clip_id,
            "source_url": self.source_url,
            "timecode_s": self.timecode_s,
            "image": self.image,
            "width": self.width,
            "height": self.height,
            "shot": self.shot,
            "blur": self.blur,
            "hands": [hand.to_dict() for hand in self.hands],
            "signs": [sign.to_dict() for sign in self.signs],
            "high_five": self.high_five.to_dict() if self.high_five else None,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class RealClip:
    clip_id: str
    title: str
    url: str
    usage_note: str
    duration_s: float
    time_range_s: tuple[float, float]
    sample_count: int
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RealClip":
        time_range = tuple(float(value) for value in data["time_range_s"])
        if len(time_range) != 2 or time_range[0] < 0 or time_range[1] <= time_range[0]:
            raise ValueError("time_range_s 必须是递增的 [start, end]")
        sample_count = int(data["sample_count"])
        if sample_count < 0:
            raise ValueError("sample_count 不能为负")
        duration = float(data.get("duration_s", 0.0))
        if duration < 0:
            raise ValueError("duration_s 不能为负")
        return cls(
            clip_id=str(data["clip_id"]),
            title=str(data["title"]),
            url=str(data["url"]),
            usage_note=str(data.get("usage_note", "")),
            duration_s=duration,
            time_range_s=time_range,
            sample_count=sample_count,
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "title": self.title,
            "url": self.url,
            "usage_note": self.usage_note,
            "duration_s": self.duration_s,
            "time_range_s": list(self.time_range_s),
            "sample_count": self.sample_count,
            "notes": self.notes,
        }


def load_real_clips(path: str | Path) -> list[RealClip]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    clips = [RealClip.from_dict(item) for item in data["clips"]]
    total = int(data.get("total_frames", sum(clip.sample_count for clip in clips)))
    if total != sum(clip.sample_count for clip in clips):
        raise ValueError("total_frames 与各 clip sample_count 之和不一致")
    return clips


def load_real_frames(path: str | Path) -> list[RealFrameLabel]:
    labels_path = Path(path)
    labels = []
    for line_number, line in enumerate(labels_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            labels.append(RealFrameLabel.from_dict(json.loads(line)))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{labels_path}:{line_number} 标注格式错误：{exc}") from exc
    return labels


def save_real_frames(path: str | Path, frames: list[RealFrameLabel]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for frame in frames:
            handle.write(json.dumps(frame.to_dict(), ensure_ascii=False) + "\n")


def frame_skeleton(
    clip: RealClip,
    image: str,
    timecode_s: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "clip_id": clip.clip_id,
        "source_url": clip.url,
        "timecode_s": float(timecode_s),
        "image": image,
        "width": int(width),
        "height": int(height),
        "shot": "unknown",
        "blur": "unknown",
        "hands": [],
        "signs": [],
        "high_five": None,
        "notes": "",
    }


def validate_frame(frame: RealFrameLabel, frames_root: str | Path | None = None) -> list[str]:
    errors: list[str] = []
    if frame.width <= 0 or frame.height <= 0:
        errors.append("width/height 必须大于 0")
    if frame.timecode_s < 0:
        errors.append("timecode_s 不能为负")
    if not frame.source_url.startswith(("http://", "https://")):
        errors.append("source_url 必须是 http(s) 链接")
    if frames_root is not None and not (Path(frames_root) / frame.image).is_file():
        errors.append(f"找不到图片：{frame.image}")

    hand_ids = {hand.hand_id for hand in frame.hands}
    if len(hand_ids) != len(frame.hands):
        errors.append("hand_id 不能重复")
    for hand_index, hand in enumerate(frame.hands):
        x1, y1, x2, y2 = hand.bbox
        if frame.width and frame.height and (x1 < 0 or y1 < 0 or x2 > frame.width or y2 > frame.height):
            errors.append(f"hands[{hand_index}] bbox 超出图像")
        if hand.keypoints and len(hand.keypoints) != hand.keypoint_format:
            errors.append(f"hands[{hand_index}] 关键点数量与 keypoint_format 不一致")
    for sign_index, sign in enumerate(frame.signs):
        x1, y1, x2, y2 = sign.bbox
        if frame.width and frame.height and (x1 < 0 or y1 < 0 or x2 > frame.width or y2 > frame.height):
            errors.append(f"signs[{sign_index}] bbox 超出图像")

    if frame.high_five and frame.high_five.present:
        if frame.high_five.fan_hand_id not in hand_ids:
            errors.append("high_five.fan_hand_id 不在 hands 中")
        if frame.high_five.wrestler_hand_id not in hand_ids:
            errors.append("high_five.wrestler_hand_id 不在 hands 中")
    return errors
