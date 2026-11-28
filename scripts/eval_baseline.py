"""评估传统颜色/轮廓基线：数据集检测指标 + 仿真序列事件指标。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g1_arena import Action, ArenaSimulation  # noqa: E402
from perception import (  # noqa: E402
    GroundTruthPerception,
    VisionPerception,
    load_manifest,
)
from perception.dataset import load_dataset_manifest  # noqa: E402
from perception.evaluation import (  # noqa: E402
    evaluate_dataset,
    latency_summary,
    match_events,
)

try:
    from perception.baseline import ColorContourDetector
except ImportError as exc:  # pragma: no cover - 依赖缺失时给明确提示
    raise SystemExit("需要 opencv-python-headless：pip install -r requirements-cv.txt") from exc


DEFAULT_DATASET = ROOT / "output" / "cv_dataset"


def evaluate_sequence(args: argparse.Namespace) -> dict:
    with ArenaSimulation() as sim:
        manifest = load_manifest()
        fan = manifest.fans[len(manifest.fans) // 2]
        detector = ColorContourDetector(
            manifest,
            camera=args.camera,
            width=args.width,
            height=args.height,
        )
        ground_truth = GroundTruthPerception(manifest)
        vision = VisionPerception(
            detector,
            manifest=manifest,
            camera=args.camera,
            width=args.width,
            height=args.height,
        )

        sim.execute(Action.WALK, target=[fan.hand[0], -2.45, 0.79], speed=1.0)
        sim.step(600)
        sim.execute(Action.WAVE, duration=4.0)

        truth_events = []
        predicted_events = []
        poll_seconds: list[float] = []
        for step in range(480):
            sim.step()
            truth_events.extend(ground_truth.poll(sim))
            if step % 4 == 0:
                started = time.perf_counter()
                predicted_events.extend(vision.poll(sim))
                poll_seconds.append(time.perf_counter() - started)

    counts, latencies = match_events(predicted_events, truth_events, tolerance_s=args.tolerance)
    return {
        "mode": "sequence",
        "camera": args.camera,
        "width": args.width,
        "height": args.height,
        "fan_id": fan.id,
        "events": counts.to_dict(),
        "event_offset_ms": latency_summary(latencies),
        "processing_ms": latency_summary(poll_seconds),
        "predicted_events": [event.to_dict() for event in predicted_events],
        "ground_truth_events": [event.to_dict() for event in truth_events],
    }


def evaluate_dataset_mode(args: argparse.Namespace) -> dict:
    dataset_manifest = load_dataset_manifest(args.dataset / "dataset_manifest.json")
    manifest = load_manifest()
    detector = ColorContourDetector(
        manifest,
        camera=dataset_manifest.camera,
        width=dataset_manifest.width,
        height=dataset_manifest.height,
    )
    return evaluate_dataset(
        args.dataset,
        detector,
        iou_threshold=args.iou,
        limit=args.limit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="评估传统颜色/轮廓手部基线")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sequence", action="store_true", help="在仿真交互序列上评事件指标")
    parser.add_argument("--camera", default="interaction_neg")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "cv_eval.json")
    args = parser.parse_args()

    result = evaluate_sequence(args) if args.sequence else evaluate_dataset_mode(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"结果已保存：{args.out.resolve()}")


if __name__ == "__main__":
    main()
