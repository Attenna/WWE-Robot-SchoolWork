"""渲染带手/应援牌标签的合成 CV 数据集。

示例：
  python scripts/render_cv_dataset.py --frames 20 --width 960 --height 540
  python scripts/render_cv_dataset.py --frames 100 --camera interaction_neg --save-depth
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g1_arena import Action, ArenaSimulation  # noqa: E402
from perception import load_manifest  # noqa: E402
from perception.dataset import DATASET_MANIFEST, LABEL_SCHEMA  # noqa: E402
from perception.high_five import HighFiveConfig  # noqa: E402
from perception.robot import hand_positions  # noqa: E402


DEFAULT_OUTPUT = ROOT / "output" / "cv_dataset"
FPS = 30.0
GEOM_OBJECT_TYPE = 5  # mjtObj.mjOBJ_GEOM
SHIRT_COLORS = np.array(
    [
        [0.15, 0.35, 0.80, 1.0],
        [0.75, 0.08, 0.12, 1.0],
        [0.12, 0.55, 0.35, 1.0],
        [0.65, 0.45, 0.12, 1.0],
        [0.55, 0.20, 0.65, 1.0],
        [0.15, 0.60, 0.75, 1.0],
    ],
    dtype=float,
)
SIGN_COLORS = np.array(
    [
        [0.95, 0.75, 0.10, 1.0],
        [0.90, 0.15, 0.20, 1.0],
        [0.15, 0.55, 0.95, 1.0],
        [0.15, 0.80, 0.45, 1.0],
        [0.95, 0.35, 0.10, 1.0],
        [0.85, 0.85, 0.80, 1.0],
    ],
    dtype=float,
)


def geom_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if value < 0:
        raise ValueError(f"模型缺少 geom：{name}")
    return int(value)


def randomize_fan_colors(model: mujoco.MjModel, manifest, rng: np.random.Generator) -> None:
    for fan in manifest.fans:
        shirt = SHIRT_COLORS[int(rng.integers(len(SHIRT_COLORS)))]
        sign = SIGN_COLORS[int(rng.integers(len(SIGN_COLORS)))]
        for suffix in ("body", "arm"):
            model.geom_rgba[geom_id(model, f"{fan.id}_{suffix}")] = shirt
        model.geom_rgba[geom_id(model, fan.sign_geom)] = sign


def place_robot(
    sim: ArenaSimulation,
    manifest,
    rng: np.random.Generator,
    *,
    near: bool,
    wave: bool,
) -> None:
    """沿互动带随机放机器人；near 时面向观众，far 时留在通道上。"""
    zone = manifest.zone
    if near:
        # 靠近某一名观众，保证数据集中有足够的正样本。
        anchor = manifest.fans[int(rng.integers(len(manifest.fans)))]
        # wave 姿态下右掌相对基座约 +0.29 m（+x），把基座放在观众左侧对齐。
        x = float(anchor.hand[0]) - 0.29 + rng.uniform(-0.03, 0.03)
        y = zone.y[0] + rng.uniform(0.48, 0.60)
        z = 0.79
        yaw = 0.0  # 面向 +x 沿通道走，右手向 -y 侧观众伸出
    else:
        x = rng.uniform(zone.x[0], zone.x[1])
        y = rng.uniform(-0.20, 0.20)
        z = 0.87
        yaw = 0.0

    sim.controller.action = Action.IDLE
    sim.controller.action_time = 0.0
    sim.controller.base_position = np.array([x, y, z], dtype=float)
    sim.controller.base_quaternion = np.array(
        [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=float
    )
    sim.controller.qpos[:] = sim.controller.home
    sim.controller.qpos[:3] = sim.controller.base_position
    sim.controller.qpos[3:7] = sim.controller.base_quaternion
    sim.data.qpos[:] = sim.controller.qpos
    mujoco.mj_forward(sim.model, sim.data)

    if near and wave:
        sim.execute(Action.WAVE, duration=4.0)
        sim.step(12)
    else:
        sim.execute(Action.IDLE)
        sim.step(1)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def anchor_label(
    world: tuple[float, float, float],
    geom_name: str,
    model: mujoco.MjModel,
    segmentation: np.ndarray,
    camera,
    width: int,
    height: int,
) -> dict[str, Any]:
    gid = geom_id(model, geom_name)
    mask = (segmentation[:, :, 0] == gid) & (segmentation[:, :, 1] == GEOM_OBJECT_TYPE)
    pixels = int(mask.sum())
    bbox = mask_bbox(mask) if pixels >= 4 else None
    keypoint: list[float] | None = None
    keypoint_visible = False
    try:
        u, v, _ = camera.project(world, width=width, height=height)
    except ValueError:
        u = v = float("nan")
    if math.isfinite(u) and math.isfinite(v) and 0.0 <= u < width and 0.0 <= v < height:
        keypoint = [float(u), float(v)]
        px = min(width - 1, max(0, int(round(u))))
        py = min(height - 1, max(0, int(round(v))))
        keypoint_visible = bool(
            segmentation[py, px, 0] == gid
            and segmentation[py, px, 1] == GEOM_OBJECT_TYPE
        )
    return {
        "world": [float(value) for value in world],
        "bbox": list(bbox) if bbox is not None else None,
        "keypoint": keypoint,
        "keypoint_visible": keypoint_visible,
        "visible": bbox is not None,
        "pixels": pixels,
    }


def augment_rgb(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    image = Image.fromarray(rgb)
    image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.88, 1.12)))
    image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.92, 1.08)))
    array = np.asarray(image, dtype=np.float32)
    array += rng.normal(0.0, float(rng.uniform(0.0, 4.0)), array.shape)
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def render_dataset(
    output: str | Path,
    *,
    frames: int = 20,
    camera: str = "interaction_neg",
    width: int = 1280,
    height: int = 720,
    seed: int = 0,
    save_depth: bool = False,
    save_segmentation: bool = False,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    if frames <= 0:
        raise ValueError("frames 必须大于 0")
    if width <= 0 or height <= 0:
        raise ValueError("width/height 必须大于 0")

    manifest = load_manifest(manifest_path)
    calibration = manifest.camera(camera)
    root = Path(output)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = root / "labels.jsonl"
    rng = np.random.default_rng(seed)
    distance_threshold = HighFiveConfig().distance_m
    labels: list[dict[str, Any]] = []

    with ArenaSimulation() as sim:
        if mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            raise ValueError(f"模型缺少相机：{camera}")
        renderer = mujoco.Renderer(sim.model, height=height, width=width)
        base_light = sim.model.light_diffuse.copy()
        try:
            for index in range(frames):
                sim.reset()
                randomize_fan_colors(sim.model, manifest, rng)
                sim.model.light_diffuse[:] = np.clip(
                    base_light + rng.normal(0.0, 0.05, base_light.shape),
                    0.0,
                    1.0,
                )
                slot = index % 4
                if slot == 0:
                    near, wave = True, True
                elif slot == 1:
                    near, wave = True, False
                elif slot == 2:
                    near, wave = False, False
                else:
                    near = bool(rng.random() < 0.5)
                    wave = bool(rng.random() < 0.5)
                place_robot(sim, manifest, rng, near=near, wave=wave)

                renderer.update_scene(sim.data, camera=camera)
                rgb = renderer.render().copy()
                depth = None
                if save_depth:
                    renderer.enable_depth_rendering()
                    depth = renderer.render().copy()
                    renderer.disable_depth_rendering()
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()

                rgb = augment_rgb(rgb, rng)
                image_name = f"images/{index:06d}.png"
                Image.fromarray(rgb).save(root / image_name)
                if save_depth and depth is not None:
                    np.save(images_dir / f"{index:06d}_depth.npy", depth)
                if save_segmentation:
                    np.save(images_dir / f"{index:06d}_seg.npy", segmentation)

                palms = {side: [float(v) for v in value] for side, value in hand_positions(sim.model, sim.data).items()}
                fan_labels = []
                for fan in manifest.fans:
                    fan_labels.append(
                        {
                            "fan_id": fan.id,
                            "hand": anchor_label(fan.hand, fan.hand_geom, sim.model, segmentation, calibration, width, height),
                            "sign": anchor_label(fan.sign, fan.sign_geom, sim.model, segmentation, calibration, width, height),
                        }
                    )

                candidate = None
                for fan in manifest.fans:
                    for side, palm in palms.items():
                        distance = float(np.linalg.norm(np.asarray(palm) - np.asarray(fan.hand)))
                        if candidate is None or distance < candidate["distance_m"]:
                            candidate = {
                                "fan_id": fan.id,
                                "distance_m": round(distance, 4),
                                "robot_hand": side,
                            }
                if candidate is not None and candidate["distance_m"] > distance_threshold:
                    candidate = None

                labels.append(
                    {
                        "index": index,
                        "image": image_name,
                        "timestamp": round(index / FPS, 6),
                        "camera": camera,
                        "width": width,
                        "height": height,
                        "scene_hash": manifest.scene_hash,
                        "robot": {
                            "base": [float(value) for value in sim.controller.base_position],
                            "palms": palms,
                        },
                        "fans": fan_labels,
                        "high_five_candidate": candidate,
                    }
                )
        finally:
            renderer.close()

    with labels_path.open("w", encoding="utf-8", newline="\n") as handle:
        for label in labels:
            handle.write(json.dumps(label, ensure_ascii=False) + "\n")

    dataset_manifest = {
        "schema_version": 1,
        "label_schema": LABEL_SCHEMA,
        "dataset": "g1_interaction_hands",
        "scene_hash": manifest.scene_hash,
        "camera": camera,
        "width": width,
        "height": height,
        "frames": frames,
        "seed": seed,
        "saved_depth": bool(save_depth),
        "saved_segmentation": bool(save_segmentation),
        "images_dir": "images",
        "labels": "labels.jsonl",
    }
    (root / DATASET_MANIFEST).write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="渲染带手/应援牌标签的合成 CV 数据集")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--camera", default="interaction_neg")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--save-segmentation", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    result = render_dataset(
        args.out,
        frames=args.frames,
        camera=args.camera,
        width=args.width,
        height=args.height,
        seed=args.seed,
        save_depth=args.save_depth,
        save_segmentation=args.save_segmentation,
        manifest_path=args.manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"数据集已生成：{args.out.resolve()}")


if __name__ == "__main__":
    main()
