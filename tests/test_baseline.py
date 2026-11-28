"""传统颜色/轮廓基线在合成数据集上的快速验收。"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("cv2")

from perception.baseline import ColorContourDetector  # noqa: E402
from perception.evaluation import evaluate_dataset  # noqa: E402
from perception.manifest import load_manifest  # noqa: E402
from scripts.render_cv_dataset import render_dataset  # noqa: E402


def test_color_baseline_on_small_dataset(tmp_path: Path) -> None:
    render_dataset(tmp_path, frames=4, width=640, height=360, seed=1)
    detector = ColorContourDetector(
        load_manifest(),
        camera="interaction_neg",
        width=640,
        height=360,
    )
    metrics = evaluate_dataset(tmp_path, detector, iou_threshold=0.3)

    assert metrics["hands"]["tp"] > 0
    assert metrics["hands"]["precision"] >= 0.8
    assert metrics["hands"]["recall"] >= 0.8
    assert metrics["signs"]["f1"] >= 0.4
    assert metrics["high_five_candidate"]["tp"] > 0
