"""检测框与事件级指标；供传统基线和后续模型共用。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image

from g1_arena.events import SimulationEvent

from .dataset import load_dataset
from .high_five import HighFiveConfig


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        total = self.tp + self.fp
        return self.tp / total if total else 0.0

    @property
    def recall(self) -> float:
        total = self.tp + self.fn
        return self.tp / total if total else 0.0

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def bbox_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def match_detections(
    detections: Sequence[tuple[str, tuple[int, int, int, int]]],
    ground_truth: Sequence[tuple[str, tuple[int, int, int, int]]],
    iou_threshold: float = 0.3,
) -> Counts:
    used: set[int] = set()
    true_positives = 0
    for fan_id, bbox in ground_truth:
        best_index = None
        best_iou = 0.0
        for index, (pred_fan, pred_bbox) in enumerate(detections):
            if index in used or pred_fan != fan_id:
                continue
            iou = bbox_iou(bbox, pred_bbox)
            if iou >= iou_threshold and iou > best_iou:
                best_index = index
                best_iou = iou
        if best_index is not None:
            used.add(best_index)
            true_positives += 1
    return Counts(
        tp=true_positives,
        fp=len(detections) - len(used),
        fn=len(ground_truth) - true_positives,
    )


def _event_fan(event: SimulationEvent) -> str:
    return str(event.metadata.get("fan_id", ""))


def match_events(
    predicted: Sequence[SimulationEvent],
    ground_truth: Sequence[SimulationEvent],
    tolerance_s: float = 0.5,
) -> tuple[Counts, list[float]]:
    used: set[int] = set()
    true_positives = 0
    latencies: list[float] = []
    for event in sorted(predicted, key=lambda item: item.timestamp):
        best = None
        for index, truth in enumerate(ground_truth):
            if index in used or _event_fan(event) != _event_fan(truth):
                continue
            delta = float(event.timestamp) - float(truth.timestamp)
            if abs(delta) <= tolerance_s and (best is None or abs(delta) < abs(best[1])):
                best = (index, delta)
        if best is not None:
            used.add(best[0])
            true_positives += 1
            latencies.append(best[1])
    return (
        Counts(
            tp=true_positives,
            fp=len(predicted) - true_positives,
            fn=len(ground_truth) - len(used),
        ),
        latencies,
    )


def latency_summary(latencies: Sequence[float]) -> dict[str, float | None]:
    if not latencies:
        return {"p50_ms": None, "p95_ms": None, "mean_ms": None}
    values = np.asarray(latencies, dtype=float) * 1000.0
    return {
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "mean_ms": round(float(values.mean()), 2),
    }


def evaluate_dataset(
    directory: str | Path,
    detector: Any,
    iou_threshold: float = 0.3,
    limit: int | None = None,
) -> dict[str, Any]:
    root = Path(directory)
    manifest, frames = load_dataset(root)
    if limit is not None:
        frames = frames[:limit]
    distance_threshold = HighFiveConfig().distance_m

    hand_gt: list[tuple[str, tuple[int, int, int, int]]] = []
    hand_pred: list[tuple[str, tuple[int, int, int, int]]] = []
    sign_gt: list[tuple[str, tuple[int, int, int, int]]] = []
    sign_pred: list[tuple[str, tuple[int, int, int, int]]] = []
    candidate_counts = Counts()
    runtime_seconds: list[float] = []

    for frame_label in frames:
        image = np.asarray(Image.open(root / frame_label.image).convert("RGB"))
        started = time.perf_counter()
        hands = detector.detect(image, frame_label.timestamp)
        signs = detector.detect_signs(image) if hasattr(detector, "detect_signs") else []
        runtime_seconds.append(time.perf_counter() - started)

        for fan in frame_label.fans:
            if fan.hand.visible and fan.hand.bbox is not None:
                hand_gt.append((fan.fan_id, fan.hand.bbox))
            if fan.sign.visible and fan.sign.bbox is not None:
                sign_gt.append((fan.fan_id, fan.sign.bbox))
        for detection in hands:
            if detection.bbox is not None:
                hand_pred.append((detection.fan_id, detection.bbox))
        for detection in signs:
            sign_pred.append((detection.fan_id, detection.bbox))

        ground_truth_positive = frame_label.high_five_candidate is not None
        predicted_positive = any(
            float(np.linalg.norm(np.asarray(detection.world) - np.asarray(palm))) <= distance_threshold
            for detection in hands
            for palm in frame_label.robot_palms.values()
        )
        if ground_truth_positive and predicted_positive:
            candidate_counts.tp += 1
        elif ground_truth_positive:
            candidate_counts.fn += 1
        elif predicted_positive:
            candidate_counts.fp += 1

    return {
        "dataset": {
            "path": str(root),
            "frames": len(frames),
            "camera": manifest.camera,
            "width": manifest.width,
            "height": manifest.height,
            "scene_hash": manifest.scene_hash,
        },
        "hands": match_detections(hand_pred, hand_gt, iou_threshold).to_dict(),
        "signs": match_detections(sign_pred, sign_gt, iou_threshold).to_dict(),
        "high_five_candidate": {
            **candidate_counts.to_dict(),
            "frames": len(frames),
            "ground_truth_positives": sum(1 for frame in frames if frame.high_five_candidate is not None),
        },
        "detector_runtime_ms": latency_summary(runtime_seconds),
    }
