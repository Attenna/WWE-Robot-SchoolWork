"""准备真实评估集：CC 图片下载、clip 清单、标注骨架和检查。

示例：
  python scripts/prepare_real_eval.py fetch-cc
  python scripts/prepare_real_eval.py plan
  python scripts/prepare_real_eval.py init-labels --frames-dir output/real_eval/frames --clip-id shield_2017
  python scripts/prepare_real_eval.py check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.real_eval import (  # noqa: E402
    RealClip,
    frame_skeleton,
    load_real_frames,
    validate_frame,
)


DEFAULT_REAL_EVAL = ROOT / "output" / "real_eval"
DEFAULT_CC_DIR = ROOT / "output" / "real_samples"
USER_AGENT = "WWE-Robot-SchoolWork/0.1 (research; contact: repository owner)"

CLIPS = [
    RealClip(
        clip_id="shield_2017",
        title="The Shield entrance through the crowd (2017)",
        url="https://youtube.com/watch?v=Lf50KrdASTI",
        usage_note="Official WWE YouTube; internal evaluation only; do not redistribute frames.",
        duration_s=161.0,
        time_range_s=(20.0, 140.0),
        sample_count=12,
        notes="Crowd walk and barricade interaction; modern broadcast.",
    ),
    RealClip(
        clip_id="edge_1999",
        title="Edge enters through the crowd (1999)",
        url="https://wwe.com/videos/edge-enters-through-the-crowd-raw-jan-4-1999",
        usage_note="WWE official; internal evaluation only; do not redistribute frames.",
        duration_s=0.0,
        time_range_s=(20.0, 140.0),
        sample_count=12,
        notes="Older broadcast; lower resolution and different lighting.",
    ),
    RealClip(
        clip_id="jey_sami",
        title="Jey Uso & Sami Zayn enter through the crowd",
        url=(
            "https://fox.com/watch/clip/fmc-0yhjpk1afm6z3wci/"
            "jey-uso-and-sami-zayn-enter-through-the-crowd-on-raw-in-green-bay-wisconsin-wwe-on-fox"
        ),
        usage_note="FOX/WWE; subscription/authorization required; internal evaluation only.",
        duration_s=180.0,
        time_range_s=(20.0, 140.0),
        sample_count=12,
        notes="Modern broadcast close-ups and crowd density.",
    ),
    RealClip(
        clip_id="wwe_360",
        title="WWE 360° entrance (Europe 2017)",
        url="https://wwe.com/worldwide/article/superstars-360-entrances-wwe-live-events-europe-november-2017",
        usage_note="WWE official; internal evaluation only; pick one 360° clip.",
        duration_s=0.0,
        time_range_s=(30.0, 150.0),
        sample_count=14,
        notes="Panoramic/multi-view reference for camera placement.",
    ),
]

CC_IMAGES = [
    {
        "filename": "wwe_smackdown_fans_signs.jpg",
        "title": "WWE Smackdown FANS with signs!",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/f7/WWE_Smackdown_FANS_with_signs%21_%283489904123%29.jpg",
        "author": "jenn (happyskrappy)",
        "license": "CC BY 2.0",
        "license_url": "https://creativecommons.org/licenses/by/2.0/",
        "source_page": "https://commons.wikimedia.org/wiki/File:WWE_Smackdown_FANS_with_signs!_(3489904123).jpg",
    },
    {
        "filename": "sign_guy_dudley.jpg",
        "title": "Sign Guy Dudley",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/59/Sign_Guy_Dudley.jpg",
        "author": "Mr Jonathan Ice",
        "license": "CC BY-SA 2.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/2.0/",
        "source_page": "https://commons.wikimedia.org/wiki/File:Sign_Guy_Dudley.jpg",
    },
    {
        "filename": "wwe_sign_guy.jpg",
        "title": "WWE Sign Guy",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/fa/WWE_Sign_Guy_%288466430893%29.jpg",
        "author": "Matt Brink",
        "license": "CC BY-SA 2.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/2.0/",
        "source_page": "https://commons.wikimedia.org/wiki/File:WWE_Sign_Guy_(8466430893).jpg",
    },
]


def fetch_cc(output: Path, force: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for item in CC_IMAGES:
        target = output / item["filename"]
        if force or not target.exists():
            request = Request(item["url"], headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=60) as response:
                target.write_bytes(response.read())
        entries.append({**item, "path": str(target)})
    sources = {
        "note": "CC images for internal style/domain review; keep attribution and license links when sharing.",
        "images": entries,
    }
    (output / "sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return sources


def write_plan(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    total = sum(clip.sample_count for clip in CLIPS)
    plan = {
        "schema_version": 1,
        "total_frames": total,
        "clips": [clip.to_dict() for clip in CLIPS],
        "note": "Time ranges are approximate; adjust after a first viewing. Do not commit video frames.",
    }
    (output / "clips.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return plan


def init_labels(frames_dir: Path, labels_path: Path, clip_id: str, force: bool = False) -> int:
    if labels_path.exists() and not force:
        raise SystemExit(f"标签文件已存在：{labels_path}（加 --force 覆盖）")
    clip = next((item for item in CLIPS if item.clip_id == clip_id), None)
    if clip is None:
        raise SystemExit(f"未知 clip_id：{clip_id}；可选：{', '.join(item.clip_id for item in CLIPS)}")
    images = sorted(
        path for path in frames_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ) if frames_dir.is_dir() else []
    if not images:
        raise SystemExit(f"没有找到图片：{frames_dir}")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8", newline="\n") as handle:
        for path in images:
            with Image.open(path) as image:
                width, height = image.size
            skeleton = frame_skeleton(clip, path.name, 0.0, width, height)
            handle.write(json.dumps(skeleton, ensure_ascii=False) + "\n")
    return len(images)


def check_labels(labels_path: Path, frames_root: Path) -> dict:
    frames = load_real_frames(labels_path)
    errors: list[str] = []
    for index, frame in enumerate(frames):
        for error in validate_frame(frame, frames_root):
            errors.append(f"frame {index} ({frame.image}): {error}")
    return {
        "labels": str(labels_path),
        "frames": len(frames),
        "hands": sum(len(frame.hands) for frame in frames),
        "signs": sum(len(frame.signs) for frame in frames),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="准备真实 WWE 评估集")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch-cc", help="下载 CC 授权的 WWE 粉丝牌图片和署名")
    fetch.add_argument("--out", type=Path, default=DEFAULT_CC_DIR)
    fetch.add_argument("--force", action="store_true")

    plan = sub.add_parser("plan", help="生成 50 帧 clip 采样清单")
    plan.add_argument("--out", type=Path, default=DEFAULT_REAL_EVAL)

    init = sub.add_parser("init-labels", help="从本地抽帧目录生成标注骨架")
    init.add_argument("--frames-dir", type=Path, default=DEFAULT_REAL_EVAL / "frames")
    init.add_argument("--out", type=Path, default=DEFAULT_REAL_EVAL / "labels.jsonl")
    init.add_argument("--clip-id", default="shield_2017")
    init.add_argument("--force", action="store_true")

    check = sub.add_parser("check", help="检查标注和图片")
    check.add_argument("--labels", type=Path, default=DEFAULT_REAL_EVAL / "labels.jsonl")
    check.add_argument("--frames-root", type=Path, default=DEFAULT_REAL_EVAL / "frames")

    args = parser.parse_args()
    if args.command == "fetch-cc":
        sources = fetch_cc(args.out, force=args.force)
        print(json.dumps(sources, ensure_ascii=False, indent=2))
    elif args.command == "plan":
        result = write_plan(args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "init-labels":
        count = init_labels(args.frames_dir, args.out, args.clip_id, force=args.force)
        print(f"已生成 {count} 条标注骨架：{args.out.resolve()}")
    elif args.command == "check":
        result = check_labels(args.labels, args.frames_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["errors"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
