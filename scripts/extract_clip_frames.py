"""从本地、有权限的视频按 clip 计划抽帧；不下载任何视频。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_real_eval import CLIPS  # noqa: E402

try:
    import cv2
except ImportError as exc:  # pragma: no cover - 可选依赖
    raise SystemExit("需要 opencv-python-headless：pip install -r requirements-cv.txt") from exc


def extract_clip(video: Path, clip_id: str, output: Path, overwrite: bool = False) -> int:
    clip = next((item for item in CLIPS if item.clip_id == clip_id), None)
    if clip is None:
        raise SystemExit(f"未知 clip_id：{clip_id}")
    if not video.is_file():
        raise SystemExit(f"找不到视频：{video}")
    output.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"无法打开视频：{video}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if frame_count > 0 else 0.0
        start, end = clip.time_range_s
        if duration > 0:
            end = min(end, max(start + 0.1, duration - 0.1))
        if end <= start:
            raise SystemExit(f"clip 时间范围无效：{start}–{end}")

        times = np.linspace(start, end, clip.sample_count, endpoint=False)
        times += (end - start) / (clip.sample_count * 2.0)
        written = 0
        for index, timestamp in enumerate(times):
            target = output / f"{clip.clip_id}_{index:04d}.png"
            if target.exists() and not overwrite:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))
            ok, frame = capture.read()
            if not ok:
                continue
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(target)
            written += 1
        return written
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="从本地视频按 clip 计划抽帧")
    parser.add_argument("--clip-id", choices=[clip.clip_id for clip in CLIPS], required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "real_eval" / "frames")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    count = extract_clip(args.video, args.clip_id, args.out, overwrite=args.overwrite)
    print(f"已抽取 {count} 帧到 {args.out.resolve()}")
    print("下一步：用 init-labels 生成标注骨架，再用 CVAT/Label Studio 或 JSONL 标注。")


if __name__ == "__main__":
    main()
