"""真实评估集 schema、clip 计划和标注骨架测试。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from perception.real_eval import (
    RealFrameLabel,
    load_real_frames,
    save_real_frames,
    validate_frame,
)
from scripts.prepare_real_eval import CLIPS, check_labels, init_labels


def _sample_frame() -> dict:
    return {
        "schema_version": 1,
        "clip_id": "shield_2017",
        "source_url": "https://youtube.com/watch?v=Lf50KrdASTI",
        "timecode_s": 42.5,
        "image": "frame.png",
        "width": 640,
        "height": 360,
        "shot": "medium",
        "blur": "mild",
        "hands": [
            {
                "hand_id": "fan_hand",
                "bbox": [10, 20, 40, 60],
                "side": "right",
                "visibility": "partial",
                "keypoints": [[20, 30, 1], [25, 35, 1], [30, 40, 0], [35, 45, 1], [40, 50, 1]],
                "keypoint_format": 5,
                "role": "fan",
                "wristband": True,
            },
            {
                "hand_id": "wrestler_hand",
                "bbox": [50, 20, 80, 60],
                "side": "left",
                "visibility": "full",
                "keypoints": [],
                "keypoint_format": 0,
                "role": "wrestler",
                "glove": True,
            },
        ],
        "signs": [
            {"bbox": [100, 20, 200, 100], "text": "IT'S ALIVE!!!", "occludes_hands": True}
        ],
        "high_five": {
            "present": True,
            "fan_hand_id": "fan_hand",
            "wrestler_hand_id": "wrestler_hand",
        },
        "notes": "test",
    }


def test_real_eval_roundtrip(tmp_path: Path) -> None:
    frame = RealFrameLabel.from_dict(_sample_frame())
    assert frame.clip_id == "shield_2017"
    assert len(frame.hands) == 2
    assert frame.high_five is not None and frame.high_five.present
    assert validate_frame(frame) == []

    labels_path = tmp_path / "labels.jsonl"
    save_real_frames(labels_path, [frame])
    loaded = load_real_frames(labels_path)
    assert len(loaded) == 1
    assert loaded[0].to_dict() == frame.to_dict()


def test_real_eval_rejects_bad_shapes() -> None:
    data = _sample_frame()
    data["hands"][0]["bbox"] = [40, 60, 10, 20]
    with pytest.raises(ValueError):
        RealFrameLabel.from_dict(data)

    data = _sample_frame()
    data["hands"][0]["keypoints"] = [[1, 2, 1]]
    with pytest.raises(ValueError):
        RealFrameLabel.from_dict(data)

    data = _sample_frame()
    data["hands"][0]["side"] = "middle"
    with pytest.raises(ValueError):
        RealFrameLabel.from_dict(data)


def test_real_eval_high_five_references() -> None:
    data = _sample_frame()
    data["high_five"]["fan_hand_id"] = "missing"
    frame = RealFrameLabel.from_dict(data)
    errors = validate_frame(frame)
    assert any("fan_hand_id" in error for error in errors)


def test_clip_plan_is_fifty_frames() -> None:
    assert len(CLIPS) == 4
    assert sum(clip.sample_count for clip in CLIPS) == 50
    assert len({clip.clip_id for clip in CLIPS}) == len(CLIPS)
    for clip in CLIPS:
        assert clip.url.startswith("https://")
        assert clip.time_range_s[1] > clip.time_range_s[0]


def test_init_labels_and_check(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    Image.new("RGB", (64, 48), color=(30, 30, 30)).save(frames_dir / "frame_0001.png")
    labels_path = tmp_path / "labels.jsonl"
    count = init_labels(frames_dir, labels_path, "shield_2017")
    assert count == 1
    frames = load_real_frames(labels_path)
    assert len(frames) == 1
    assert frames[0].clip_id == "shield_2017"
    assert frames[0].width == 64
    assert frames[0].height == 48

    result = check_labels(labels_path, frames_dir)
    assert result["frames"] == 1
    assert result["errors"] == []
