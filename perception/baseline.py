"""固定机位 + 清单先验的传统颜色/轮廓基线（无 torch）。

这不是通用手部检测器：它假设现场机位和互动观众位置固定，用
scene_manifest 的投影位置作为 ROI 先验，再在 ROI 内用 HSV 肤色/边缘轮廓
找手和应援牌。目的先验证数据集与事件指标，而不是替代真实模型。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .manifest import SceneManifest, load_manifest
from .vision import HandDetection, SignDetection


SKIN_LOWER = np.array([0, 25, 50], dtype=np.uint8)
SKIN_UPPER = np.array([35, 210, 255], dtype=np.uint8)


class ColorContourDetector:
    """基于清单 ROI 的颜色/轮廓基线；``detect`` 供 VisionPerception 使用。"""

    def __init__(
        self,
        manifest: SceneManifest | None = None,
        camera: str = "interaction_neg",
        width: int = 1280,
        height: int = 720,
        min_confidence: float = 0.5,
        detect_non_reaching: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height 必须大于 0")
        self.manifest = manifest or load_manifest()
        self.camera = self.manifest.camera(camera)
        self.camera_name = camera
        self.width = int(width)
        self.height = int(height)
        self.min_confidence = float(min_confidence)
        self.detect_non_reaching = bool(detect_non_reaching)

        scale = (self.width / 1280.0) ** 2
        self._hand_radius = max(12, int(28 * self.width / 1280.0))
        self._hand_min_area = max(8, int(30 * scale))
        self._hand_max_area = max(200, int(6000 * scale))
        self._hand_area_ref = max(1.0, 600.0 * scale)
        self._sign_radius = max(14, int(30 * self.width / 1280.0))
        self._sign_min_area = max(40, int(250 * scale))
        self._sign_max_area = max(500, int(40000 * scale))
        self._sign_area_ref = max(1.0, 3000.0 * scale)

        hand_fans = self.manifest.fans if self.detect_non_reaching else [
            fan for fan in self.manifest.fans if fan.hand_reaching
        ]
        self._hand_priors = self._priors([(fan.id, fan.hand) for fan in hand_fans], self._hand_radius)
        self._sign_priors = self._priors([(fan.id, fan.sign) for fan in self.manifest.fans], self._sign_radius)

    def _priors(self, items: list[tuple[str, tuple[float, float, float]]], margin: int):
        priors = []
        for fan_id, world in items:
            try:
                u, v, depth = self.camera.project(world, width=self.width, height=self.height)
            except ValueError:
                continue
            if -margin <= u < self.width + margin and -margin <= v < self.height + margin:
                priors.append((fan_id, world, float(u), float(v), depth))
        return priors

    def _check_frame(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame 必须是 HxWx3 RGB")
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(f"frame 尺寸 {frame.shape[1]}x{frame.shape[0]} 与检测器 {self.width}x{self.height} 不一致")

    def _crop(self, image: np.ndarray, u: float, v: float, radius: int):
        x1 = max(0, int(round(u)) - radius)
        y1 = max(0, int(round(v)) - radius)
        x2 = min(self.width, int(round(u)) + radius + 1)
        y2 = min(self.height, int(round(v)) + radius + 1)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return image[y1:y2, x1:x2], x1, y1

    def detect(self, frame: np.ndarray, timestamp: float) -> list[HandDetection]:
        self._check_frame(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        kernel = np.ones((3, 3), dtype=np.uint8)
        detections: list[HandDetection] = []

        for fan_id, world, u, v, _ in self._hand_priors:
            crop = self._crop(hsv, u, v, self._hand_radius)
            if crop is None:
                continue
            roi, x1, y1 = crop
            mask = cv2.inRange(roi, SKIN_LOWER, SKIN_UPPER)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            expected = (u - x1, v - y1)
            candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self._hand_min_area or area > self._hand_max_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w / 2.0, y + h / 2.0)
                distance = math.hypot(center[0] - expected[0], center[1] - expected[1])
                if distance > self._hand_radius * 1.2:
                    continue
                candidates.append((distance, -area, contour))
            if not candidates:
                continue

            _, _, contour = min(candidates, key=lambda item: (item[0], item[1]))
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (int(x1 + x), int(y1 + y), int(x1 + x + w), int(y1 + y + h))
            confidence = min(1.0, 0.5 + 0.5 * float(w * h) / self._hand_area_ref)
            if confidence < self.min_confidence:
                continue
            detections.append(
                HandDetection(
                    fan_id=fan_id,
                    world=world,
                    confidence=float(confidence),
                    bbox=bbox,
                    camera=self.camera_name,
                )
            )
        return detections

    def detect_signs(self, frame: np.ndarray) -> list[SignDetection]:
        self._check_frame(frame)
        kernel = np.ones((3, 3), dtype=np.uint8)
        detections: list[SignDetection] = []

        for fan_id, world, u, v, _ in self._sign_priors:
            crop = self._crop(frame, u, v, self._sign_radius)
            if crop is None:
                continue
            roi, x1, y1 = crop
            # 应援牌在观众头顶；只保留 ROI 上半部，避免把头部/衣服当成牌子。
            row_limit = min(roi.shape[0], max(1, int(self._sign_radius * 1.80)))
            roi = roi[:row_limit, :]
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            mask = (
                ((saturation > 70) & (value > 60))
                | ((value > 180) & (saturation < 60))
            ).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            expected = (u - x1, v - y1)
            candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self._sign_min_area or area > self._sign_max_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w <= 0 or h <= 0:
                    continue
                aspect = w / h
                if not 0.6 <= aspect <= 3.5:
                    continue
                if area / float(w * h) < 0.35:
                    continue
                center = (x + w / 2.0, y + h / 2.0)
                distance = math.hypot(center[0] - expected[0], center[1] - expected[1])
                if distance > self._sign_radius * 1.5:
                    continue
                candidates.append((distance, -area, contour))
            if not candidates:
                continue

            _, _, contour = min(candidates, key=lambda item: (item[0], item[1]))
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (int(x1 + x), int(y1 + y), int(x1 + x + w), int(y1 + y + h))
            confidence = min(1.0, 0.5 + 0.5 * float(w * h) / self._sign_area_ref)
            if confidence < self.min_confidence:
                continue
            detections.append(
                SignDetection(
                    fan_id=fan_id,
                    world=world,
                    bbox=bbox,
                    confidence=float(confidence),
                    camera=self.camera_name,
                )
            )
        return detections
