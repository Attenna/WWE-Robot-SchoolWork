"""合成数据集渲染与标签 schema 测试。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from perception.dataset import LABEL_SCHEMA, load_dataset
from perception.manifest import load_manifest
from scripts.render_cv_dataset import render_dataset


def test_render_dataset_small(tmp_path: Path) -> None:
    result = render_dataset(
        tmp_path,
        frames=3,
        width=320,
        height=240,
        seed=1,
    )
    assert result["frames"] == 3
    assert result["label_schema"] == LABEL_SCHEMA
    assert result["camera"] == "interaction_neg"

    scene = load_manifest()
    manifest, labels = load_dataset(tmp_path)
    assert manifest.frames == 3
    assert manifest.scene_hash == scene.scene_hash
    assert manifest.width == 320
    assert manifest.height == 240
    assert len(labels) == 3

    visible_hands = 0
    for label in labels:
        assert label.width == 320
        assert label.height == 240
        assert label.scene_hash == scene.scene_hash
        assert len(label.fans) == len(scene.fans)
        image_path = tmp_path / label.image
        assert image_path.is_file()
        with Image.open(image_path) as image:
            assert image.size == (320, 240)

        for fan in label.fans:
            for anchor in (fan.hand, fan.sign):
                if anchor.bbox is not None:
                    x1, y1, x2, y2 = anchor.bbox
                    assert 0 <= x1 < x2 <= 320
                    assert 0 <= y1 < y2 <= 240
                if anchor.keypoint_visible:
                    assert anchor.bbox is not None
                    x1, y1, x2, y2 = anchor.bbox
                    u, v = anchor.keypoint
                    assert x1 - 1 <= u <= x2 + 1
                    assert y1 - 1 <= v <= y2 + 1
                if anchor.visible:
                    visible_hands += 1
        if label.high_five_candidate is not None:
            assert label.high_five_candidate["distance_m"] <= 0.30
            assert label.high_five_candidate["robot_hand"] in {"left", "right"}

    assert visible_hands > 0
