"""原始手/应援牌检测流：BBB 交给 CCC 的检测原语。

这一层不判断“是否举牌/是否伸手/是否击掌”，只输出每帧的手、牌 bbox/world/
confidence。语义和声音融合由 CCC 的决策层负责。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from g1_arena import ArenaSimulation

from .manifest import SceneManifest, load_manifest
from .vision import HandDetection, SignDetection


@dataclass(frozen=True)
class DetectionFrame:
    """一帧原始检测结果。"""

    timestamp: float
    camera: str
    width: int
    height: int
    hands: tuple[HandDetection, ...]
    signs: tuple[SignDetection, ...]


@runtime_checkable
class DetectorSource(Protocol):
    """给 CCC 的原始检测接口。"""

    def detect(self, sim: ArenaSimulation) -> DetectionFrame:
        ...

    def reset(self) -> None:
        ...


class FrameDetectionSource:
    """在仿真里渲染一帧，再调用 HandDetector；也支持外部传入真实帧。"""

    def __init__(
        self,
        detector,
        *,
        manifest: SceneManifest | None = None,
        camera: str = "interaction_neg",
        width: int = 1280,
        height: int = 720,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height 必须大于 0")
        self.detector = detector
        self.manifest = manifest or load_manifest()
        self.manifest.camera(camera)
        self.camera = camera
        self.width = int(width)
        self.height = int(height)

    def reset(self) -> None:
        reset = getattr(self.detector, "reset", None)
        if callable(reset):
            reset()

    def process(self, frame: np.ndarray, timestamp: float) -> DetectionFrame:
        hands = tuple(self.detector.detect(frame, timestamp))
        detect_signs = getattr(self.detector, "detect_signs", None)
        signs = tuple(detect_signs(frame)) if callable(detect_signs) else ()
        return DetectionFrame(
            timestamp=float(timestamp),
            camera=self.camera,
            width=self.width,
            height=self.height,
            hands=hands,
            signs=signs,
        )

    def detect(self, sim: ArenaSimulation) -> DetectionFrame:
        timestamp = float(sim.data.time)
        frame = sim.render(self.camera, self.width, self.height)
        return self.process(frame, timestamp)


class GroundTruthDetectorSource:
    """用清单锚点伪造检测结果，给 CCC 的语义逻辑做单元测试/对照。"""

    def __init__(
        self,
        manifest: SceneManifest | None = None,
        camera: str = "interaction_neg",
    ) -> None:
        self.manifest = manifest or load_manifest()
        calibration = self.manifest.camera(camera)
        self.camera = camera
        self.width = calibration.width
        self.height = calibration.height

    def reset(self) -> None:
        return None

    def detect(self, sim: ArenaSimulation) -> DetectionFrame:
        hands = tuple(
            HandDetection(
                fan_id=fan.id,
                world=fan.hand,
                confidence=1.0,
                hand="raised" if fan.hand_reaching else "down",
                camera=self.camera,
            )
            for fan in self.manifest.fans
        )
        signs = tuple(
            SignDetection(
                fan_id=fan.id,
                world=fan.sign,
                bbox=None,
                confidence=1.0,
                camera=self.camera,
            )
            for fan in self.manifest.fans
        )
        return DetectionFrame(
            timestamp=float(sim.data.time),
            camera=self.camera,
            width=self.width,
            height=self.height,
            hands=hands,
            signs=signs,
        )


class NullDetectorSource:
    """关闭检测时的空实现。"""

    def __init__(self, camera: str = "interaction_neg", width: int = 1280, height: int = 720) -> None:
        self.camera = camera
        self.width = int(width)
        self.height = int(height)

    def reset(self) -> None:
        return None

    def detect(self, sim: ArenaSimulation) -> DetectionFrame:
        return DetectionFrame(
            timestamp=float(sim.data.time),
            camera=self.camera,
            width=self.width,
            height=self.height,
            hands=(),
            signs=(),
        )


def create_detector(
    source: str = "color",
    *,
    manifest: SceneManifest | None = None,
    camera: str = "interaction_neg",
    width: int = 1280,
    height: int = 720,
) -> DetectorSource:
    """创建原始检测源：none / ground_truth / color。"""
    source_name = source.strip().lower()
    if source_name in {"none", "off"}:
        return NullDetectorSource(camera=camera, width=width, height=height)
    if source_name not in {"ground_truth", "gt", "color", "baseline"}:
        raise ValueError(
            f"未知检测源：{source!r}；可选：none/off、ground_truth/gt、color/baseline"
        )

    scene = manifest or load_manifest()
    if source_name in {"ground_truth", "gt"}:
        return GroundTruthDetectorSource(scene, camera=camera)

    try:
        from .baseline import ColorContourDetector
    except ImportError as exc:
        raise RuntimeError(
            "颜色基线需要 opencv-python-headless：pip install -r requirements-cv.txt"
        ) from exc
    detector = ColorContourDetector(
        scene,
        camera=camera,
        width=width,
        height=height,
    )
    return FrameDetectionSource(
        detector,
        manifest=scene,
        camera=camera,
        width=width,
        height=height,
    )
