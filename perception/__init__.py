"""CV 感知层：场景清单、击掌事件、合成数据和仿真 ground-truth 适配器。"""

from .api import NullPerception, PerceptionSource, create_perception
from .dataset import DatasetManifest, FrameLabel, load_dataset, load_frame_labels
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
from .vision import HandDetection, HandDetector, VisionPerception

__all__ = [
    "CameraCalibration",
    "DatasetManifest",
    "FanAnchor",
    "FrameLabel",
    "GroundTruthPerception",
    "HandDetection",
    "HandDetector",
    "HighFiveConfig",
    "HighFiveStateMachine",
    "InteractionZone",
    "NullPerception",
    "PerceptionSource",
    "SceneManifest",
    "VisionPerception",
    "create_perception",
    "hand_positions",
    "load_dataset",
    "load_frame_labels",
    "load_manifest",
]
