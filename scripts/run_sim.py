#!/usr/bin/env python3
"""Run a recorded G1 standing, walking or calibration-reaching demonstration."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import time

from _bootstrap import configure_render, new_run_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", choices=("stand", "walk", "reach"), default="reach")
    parser.add_argument("--seconds", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--views", nargs="+", choices=("ego", "workspace", "side"), default=["ego", "workspace"])
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--gl", choices=("auto", "egl", "osmesa", "glfw"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0.02 <= args.seconds <= 60:
        parser.error("--seconds must be between 0.02 and 60")
    if not math.isclose(args.seconds / 0.002, round(args.seconds / 0.002), abs_tol=1e-8):
        parser.error("--seconds must be a multiple of the 0.002 s physics step")
    try:
        configure_render(args.gl, viewer=args.viewer)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    from hwap.actions import CommandChunk, Keyframe, Pose
    from hwap.environment import HumanoidEnvironment
    from hwap.recording import EpisodeRecorder

    output = args.output or new_run_path("demo_" + args.demo)
    with ExitStack() as stack:
        env = stack.enter_context(HumanoidEnvironment(views=tuple(args.views), budget_s=args.seconds))
        env.reset(args.seed)
        recorder = EpisodeRecorder(env, output)
        viewer = None
        if args.viewer:
            import mujoco.viewer
            viewer = stack.enter_context(mujoco.viewer.launch_passive(env.sim.model, env.sim.data))
        wall_start, sim_start = time.monotonic(), float(env.sim.data.time)

        def capture(current_env, command):
            recorder.capture(current_env, command)
            if viewer is not None and viewer.is_running():
                viewer.sync()
                delay = wall_start + float(current_env.sim.data.time) - sim_start - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

        while env.episode_steps * env.sim.model.opt.timestep < args.seconds - 1e-9:
            if viewer is not None and not viewer.is_running():
                break
            remaining = args.seconds - env.episode_steps * env.sim.model.opt.timestep
            duration = min(2.5, round(remaining / 0.002) * 0.002)
            if args.demo == "reach":
                # This is explicitly a geometric calibration demo, not a visual Agent.
                start = Pose(tuple(env.sim.palm_position("right")))
                goal = Pose(tuple(env.goal))
                chunk = CommandChunk("calibration_reach", (
                    Keyframe(0, right_wrist=start),
                    Keyframe(duration * 0.6, right_wrist=goal),
                    Keyframe(duration, right_wrist=goal)), frame="world")
            else:
                velocity = (0.04, 0.0, 0.0) if args.demo == "walk" else (0.0, 0.0, 0.0)
                chunk = CommandChunk(args.demo, (
                    Keyframe(0, velocity=velocity), Keyframe(duration, velocity=velocity)))
            start_row = len(recorder.rows) - 1
            outcome = env.execute(chunk, callback=capture)
            recorder.capture(env, None, force_image=True)
            recorder.chunks.append({"chunk": chunk.to_dict(), "start_row": start_row,
                                    "end_row": len(recorder.rows) - 1, "outcome": outcome})
            if outcome["failure"]:
                break
        result = recorder.finish({"run_kind": "simulation_demo", "demo": args.demo,
                                  "agent": "replay_calibration" if args.demo == "reach" else "scripted_control",
                                  "llm_calls": 0, "world_model_calls": 0})
    print(json.dumps({"episode": str(output.resolve()), "metrics": result["metrics"],
                      "run_kind": result["run_kind"]}, indent=2))
    if result["metrics"]["failure"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
