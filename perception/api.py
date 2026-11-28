"""视觉检测的公开对接入口。

决策/控制层只需要依赖 ``PerceptionSource``：
  - ``poll(sim) -> list[SimulationEvent]``
  - ``reset()``

检测器实现 ``HandDetector``：
  - ``detect(frame, timestamp) -> list[HandDetection]``

具体来源（ground truth、颜色基线、后续 YOLO/关键点）由 ``create_perception``
选择；上层不需要知道检测器细节。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from g1_arena import ArenaSimulation
from g1_arena.events import SimulationEvent

from .ground_truth import GroundTruthPerception
from .high_five import HighFiveConfig
from .manifest import SceneManifest, load_manifest
from .vision import VisionPerception


@runtime_checkable
class PerceptionSource(Protocol):
    """视觉/仿真感知源对决策层的唯一接口。"""

    def poll(self, sim: ArenaSimulation) -> list[SimulationEvent]:
        ...

    def reset(self) -> None:
        ...


class NullPerception:
    """关闭感知时的空实现；保持上层调用形状一致。"""

    def poll(self, sim: ArenaSimulation) -> list[SimulationEvent]:
        return []

    def reset(self) -> None:
        return None


def create_perception(
    source: str = "ground_truth",
    *,
    manifest: SceneManifest | None = None,
    config: HighFiveConfig | None = None,
    camera: str = "interaction_neg",
    width: int = 1280,
    height: int = 720,
) -> PerceptionSource:
    """按名称创建感知源。

    source:
      - ``none`` / ``off``：空实现；
      - ``ground_truth`` / ``gt``：仿真 FK 真值，用于测试/演示；
      - ``color`` / ``baseline``：固定机位 + 清单 ROI 的颜色/轮廓基线。

    后续模型只需要实现 ``HandDetector``，再在这里注册，不要改决策/控制代码。
    """
    source_name = source.strip().lower()
    if source_name in {"none", "off"}:
        return NullPerception()
    if source_name not in {"ground_truth", "gt", "color", "baseline"}:
        raise ValueError(
            f"未知感知源：{source!r}；可选：none/off、ground_truth/gt、color/baseline"
        )

    scene = manifest or load_manifest()
    if source_name in {"ground_truth", "gt"}:
        return GroundTruthPerception(scene, config)

    if source_name in {"color", "baseline"}:
        try:
            from .baseline import ColorContourDetector
        except ImportError as exc:
            raise RuntimeError(
                "颜色基线需要 opencv-python-headless："
                "pip install -r requirements-cv.txt"
            ) from exc
        detector = ColorContourDetector(
            scene,
            camera=camera,
            width=width,
            height=height,
        )
        high_five_config = config or HighFiveConfig(source="vision", mode="color_baseline")
        return VisionPerception(
            detector,
            manifest=scene,
            config=high_five_config,
            camera=camera,
            width=width,
            height=height,
        )
