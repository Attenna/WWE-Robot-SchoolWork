"""CCC 原始检测流接口测试。"""

from __future__ import annotations

import numpy as np

from g1_arena import ArenaSimulation
from perception import (
    DetectionFrame,
    DetectorSource,
    FrameDetectionSource,
    HandDetection,
    SignDetection,
    create_detector,
    load_manifest,
)


class _FakeDetector:
    def detect(self, frame, timestamp):
        return [
            HandDetection(
                fan_id="fan_int_neg_3",
                world=(-6.68, -3.02, 1.3),
                confidence=0.9,
                bbox=(10, 20, 30, 40),
            )
        ]

    def detect_signs(self, frame):
        return [
            SignDetection(
                fan_id="fan_int_neg_3",
                world=(-6.68, -3.05, 1.85),
                bbox=(40, 10, 90, 50),
                confidence=0.8,
            )
        ]


def test_ground_truth_detector() -> None:
    detector = create_detector("ground_truth")
    assert isinstance(detector, DetectorSource)
    with ArenaSimulation() as sim:
        frame = detector.detect(sim)
    assert isinstance(frame, DetectionFrame)
    assert len(frame.hands) == 6
    assert len(frame.signs) == 6
    assert {hand.hand for hand in frame.hands} == {"raised", "down"}


def test_null_detector() -> None:
    detector = create_detector("none")
    with ArenaSimulation() as sim:
        frame = detector.detect(sim)
    assert frame.hands == ()
    assert frame.signs == ()


def test_frame_detection_source() -> None:
    source = FrameDetectionSource(
        _FakeDetector(),
        manifest=load_manifest(),
        width=320,
        height=240,
    )
    frame = source.process(np.zeros((240, 320, 3), dtype=np.uint8), 1.5)
    assert frame.timestamp == 1.5
    assert frame.width == 320
    assert frame.height == 240
    assert len(frame.hands) == 1
    assert len(frame.signs) == 1
    assert frame.signs[0].camera == ""


def test_color_detector_factory() -> None:
    try:
        detector = create_detector("color")
    except RuntimeError as exc:
        assert "opencv" in str(exc).lower()
        return
    assert isinstance(detector, FrameDetectionSource)
