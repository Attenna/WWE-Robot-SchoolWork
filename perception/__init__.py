"""CV 感知层：场景清单、击掌事件和仿真 ground-truth 适配器。"""

from .ground_truth import GroundTruthPerception
from .high_five import HighFiveConfig, HighFiveStateMachine
from .manifest import (
    CameraCalibration,
    FanAnchor,
    InteractionZone,
    SceneManifest,
    load_manifest,
)
from .robot import hand_positions

__all__ = [
    "CameraCalibration",
    "FanAnchor",
    "GroundTruthPerception",
    "HighFiveConfig",
    "HighFiveStateMachine",
    "InteractionZone",
    "SceneManifest",
    "hand_positions",
    "load_manifest",
]
