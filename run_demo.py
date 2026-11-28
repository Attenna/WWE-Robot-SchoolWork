"""运行 G1 入场互动演示。

示例：
  python run_demo.py
  python run_demo.py --action wave
  python run_demo.py --headless --seconds 3 --screenshot output/wave.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
from PIL import Image

from g1_arena import Action, ArenaSimulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="宇树 G1 WWE 风格入场互动仿真")
    parser.add_argument("--action", choices=["entrance", "walk", "wave", "celebrate", "bow", "idle", "interaction"], default="entrance")
    parser.add_argument("--perception", choices=["none", "ground_truth"], default="none", help="感知适配器；ground_truth 用仿真 FK 产生 high_five 事件")
    parser.add_argument("--headless", action="store_true", help="不打开窗口，仅运行仿真")
    parser.add_argument("--seconds", type=float, default=8.0, help="无界面模式的仿真时长")
    parser.add_argument("--screenshot", type=Path, help="结束时保存相机图像")
    parser.add_argument("--camera", choices=["overview", "entrance", "ringside", "interaction_neg"], default="overview")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def interaction_target() -> list[float]:
    """从清单取中间一名观众，走到护栏前的互动带。"""
    from perception import load_manifest

    manifest = load_manifest()
    fan = manifest.fans[len(manifest.fans) // 2]
    return [float(fan.hand[0]), float(manifest.zone.y[0] + 0.65), 0.79]


def make_perception(args: argparse.Namespace):
    if args.perception == "ground_truth":
        from perception import GroundTruthPerception

        return GroundTruthPerception()
    return None


def poll_perception(sim: ArenaSimulation, perception) -> None:
    if perception is None:
        return
    for event in perception.poll(sim):
        print(json.dumps(event.to_dict(), ensure_ascii=False))
        sim.submit_event(event)


def start_action(sim: ArenaSimulation, action: str) -> None:
    if action == "entrance":
        sim.execute(Action.WALK, target=[-4.15, 0.0, 0.87], speed=0.65)
    elif action == "interaction":
        sim.execute(Action.WALK, target=interaction_target(), speed=0.8)
    elif action == "walk":
        sim.execute(Action.WALK)
    elif action != "idle":
        sim.execute(action)


def run_headless(sim: ArenaSimulation, args: argparse.Namespace) -> None:
    perception = make_perception(args)
    interaction_stage = 0
    start_action(sim, args.action)
    for _ in range(round(args.seconds / sim.timestep)):
        sim.step()
        if args.action == "interaction" and interaction_stage == 0 and sim.action == Action.IDLE:
            sim.execute(Action.WAVE, duration=4.0)
            interaction_stage = 1
        poll_perception(sim, perception)
    print(json.dumps(sim.get_state(), ensure_ascii=False, indent=2))
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        image = sim.render(args.camera, args.width, args.height)
        Image.fromarray(image).save(args.screenshot)
        print(f"图像已保存：{args.screenshot.resolve()}")


def run_viewer(sim: ArenaSimulation, args: argparse.Namespace) -> None:
    perception = make_perception(args)
    start_action(sim, args.action)
    mode = args.action
    demo_stage = 0
    interaction_stage = 0
    previous_action = sim.action

    def on_key(key: int) -> None:
        nonlocal demo_stage, interaction_stage, mode
        if key in (ord(" "),):
            sim.execute(Action.STOP)
        elif key in (ord("W"), ord("w")):
            sim.execute(Action.WAVE)
        elif key in (ord("C"), ord("c")):
            sim.execute(Action.CELEBRATE)
        elif key in (ord("B"), ord("b")):
            sim.execute(Action.BOW)
        elif key in (ord("R"), ord("r")):
            sim.reset()
            interaction_stage = 0
            if mode == "interaction":
                start_action(sim, "interaction")
        elif key == 257:  # Enter
            sim.reset()
            start_action(sim, "entrance")
            mode = "entrance"
            demo_stage = 0
            interaction_stage = 1  # Enter 明确切到入场流程，禁用 interaction 自动挥手

    print("按键：Enter=完整入场，W=挥手，C=庆祝，B=鞠躬，空格=停止，R=复位；关闭窗口退出。")
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
        while viewer.is_running():
            frame_start = time.perf_counter()
            sim.step()
            if mode == "interaction" and interaction_stage == 0 and sim.action == Action.IDLE:
                sim.execute(Action.WAVE, duration=4.0)
                interaction_stage = 1
            poll_perception(sim, perception)
            if mode == "entrance":
                if previous_action != Action.IDLE and sim.action == Action.IDLE:
                    demo_stage += 1
                    if demo_stage == 1:
                        sim.execute(Action.WAVE, duration=4.0)
                    elif demo_stage == 2:
                        sim.execute(Action.CELEBRATE, duration=3.0)
                previous_action = sim.action
            viewer.sync()
            remaining = sim.timestep - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    args = parse_args()
    with ArenaSimulation() as sim:
        if args.headless or args.screenshot:
            run_headless(sim, args)
        else:
            run_viewer(sim, args)


if __name__ == "__main__":
    main()
