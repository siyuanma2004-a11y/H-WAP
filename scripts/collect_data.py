#!/usr/bin/env python3
"""Collect synchronized randomized command rollouts without a model API.

``--mode wrist`` (default) keeps the original small right-wrist jitter.
``--mode wm`` writes richer world-model data: both palms, hand closure,
multi-keyframe chunks, optional gaze (only when the backend reports head
support) and opt-in small base velocities, in the heading_at_start frame.
"""
import argparse
import json
import math
from pathlib import Path

from _bootstrap import configure_render, new_run_path, write_json

WORKSPACE_Z = (0.83, 1.30)
VELOCITY_LOW = (-0.10, -0.06, -0.20)  # below the action-contract limits;
VELOCITY_HIGH = (0.03, 0.06, 0.20)  # little forward motion: the start pose faces the table


def heading_relative(point, base):
    import numpy as np
    x, y, yaw = base
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ (np.asarray(point) - [x, y, 0])


def gaze_supported(env):
    caps = env.observe(render=False)["capabilities"]
    return bool(caps.get("gaze") or caps.get("head") or caps.get("head_torso_reference"))


def gaze_pose(rng, max_yaw, max_pitch):
    """Orientation-only head request; returns None if the action schema needs a head position."""
    import numpy as np
    from hwap.actions import Pose
    yaw, pitch = rng.uniform(-max_yaw, max_yaw), rng.uniform(0, max_pitch)  # positive pitch looks down
    q = [np.cos(yaw/2)*np.cos(pitch/2), -np.sin(yaw/2)*np.sin(pitch/2),
         np.cos(yaw/2)*np.sin(pitch/2), np.sin(yaw/2)*np.cos(pitch/2)]
    try:
        return Pose(None, tuple(q))
    except (TypeError, ValueError):
        return None


def wm_chunk(env, rng, args, index, state):
    """One random chunk; ``state`` carries the last hand closures and gaze so chunks join continuously."""
    import numpy as np
    from hwap.actions import CommandChunk, Keyframe, Pose
    base = env.sim.base_pose
    previous = {side: heading_relative(env.sim.palm_position(side), base) for side in ("left", "right")}
    times = np.round(np.linspace(0, args.chunk_seconds, args.keyframes) / 0.002) * 0.002
    hands = {side: (state[side], float(rng.uniform(0, 1))) for side in ("left", "right")}
    velocity = tuple(rng.uniform(VELOCITY_LOW, VELOCITY_HIGH)) if args.locomotion else (0.0, 0.0, 0.0)
    keys = []
    for k, t in enumerate(times):
        alpha, fields = k / (len(times) - 1), {}
        for side in ("left", "right"):
            if k:
                target = previous[side] + rng.uniform(-args.jitter_m, args.jitter_m, 3)
                target[2] = np.clip(target[2], *WORKSPACE_Z)
                target[:2] = np.clip(target[:2], [0.05, -0.6], [0.6, 0.6])
                previous[side] = target
            fields[side + "_wrist"] = Pose(tuple(previous[side]))
            fields[side + "_hand"] = (1 - alpha) * hands[side][0] + alpha * hands[side][1]
        if state["gaze"] is not None:
            fields["head"] = state["gaze"] if k == 0 else gaze_pose(rng, args.gaze_yaw, args.gaze_pitch)
            state["gaze"] = fields["head"]
        # Base velocity ramps through interior keyframes only (needs >= 3 keyframes).
        keys.append(Keyframe(float(t), velocity=velocity if 0 < k < len(times) - 1 else (0.0, 0.0, 0.0), **fields))
    state.update(left=hands["left"][1], right=hands["right"][1])
    return CommandChunk(f"random_wm_{index}", tuple(keys), frame="heading_at_start")


def wrist_chunk(env, rng, args, index):
    import numpy as np
    from hwap.actions import CommandChunk, Keyframe, Pose
    start = env.sim.palm_position("right")
    target = start + rng.uniform(-args.jitter_m, args.jitter_m, 3)
    # Stay above the table; the controller/evaluator still checks contacts.
    target[2] = np.clip(target[2], *WORKSPACE_Z)
    return CommandChunk(f"random_wrist_{index}", (
        Keyframe(0, right_wrist=Pose(tuple(start))),
        Keyframe(args.chunk_seconds, right_wrist=Pose(tuple(target)))), frame="world")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--task", default="reach")
    parser.add_argument("--mode", choices=("wrist", "wm"), default="wrist")
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    parser.add_argument("--keyframes", type=int, default=3, help="wm mode: keyframes per chunk")
    parser.add_argument("--jitter-m", type=float, default=0.025, help="per-keyframe step; wm mode allows up to 0.15")
    parser.add_argument("--locomotion", action="store_true", help="wm mode: small random base velocities")
    parser.add_argument("--no-gaze", action="store_true", help="wm mode: never send head requests")
    parser.add_argument("--gaze-yaw", type=float, default=0.4)
    parser.add_argument("--gaze-pitch", type=float, default=0.35)
    parser.add_argument("--views", nargs="+", choices=("ego", "workspace", "side"), default=["ego", "workspace"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--export-dataset", type=Path)
    parser.add_argument("--wm-export", type=Path, help="also write a clip-level world-model dataset here")
    parser.add_argument("--clip-frames", type=int, default=9)
    parser.add_argument("--clip-stride", type=int, default=4)
    parser.add_argument("--splits", type=Path, help="JSON map of task:seed / task / default -> split")
    parser.add_argument("--video", action="store_true", help="wm export: also encode per-clip mp4")
    parser.add_argument("--gl", choices=("auto", "egl", "osmesa", "glfw"), default="auto")
    args = parser.parse_args()
    if args.episodes < 1 or not 1 <= args.chunks <= 100 or args.seed_start < 0:
        parser.error("Use positive episodes/chunks and a non-negative seed")
    limit = 0.15 if args.mode == "wm" else 0.05
    if not math.isfinite(args.jitter_m) or not 0 < args.jitter_m <= limit:
        parser.error(f"--jitter-m must be in (0, {limit}] for --mode {args.mode}")
    if not math.isfinite(args.chunk_seconds) or not 0.02 <= args.chunk_seconds <= 3:
        parser.error("--chunk-seconds must be in [0.02, 3]")
    if not math.isclose(args.chunk_seconds / 0.002, round(args.chunk_seconds / 0.002), abs_tol=1e-8):
        parser.error("--chunk-seconds must be a multiple of the physics step")
    if not 2 <= args.keyframes <= 8 or args.chunk_seconds / (args.keyframes - 1) < 0.02:
        parser.error("--keyframes must be in [2, 8] and spaced at least 0.02 s")
    if args.mode == "wrist" and (args.locomotion or args.keyframes != 3):
        parser.error("--locomotion/--keyframes require --mode wm")
    try:
        configure_render(args.gl)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    import numpy as np
    from hwap.environment import HumanoidEnvironment
    from hwap.recording import EpisodeRecorder

    output = (args.output or new_run_path("random_collection")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    agent = "scripted_random_wrist" if args.mode == "wrist" else "scripted_random_wm"
    report = {"run_kind": "random_action_collection", "agent": agent, "mode": args.mode, "task": args.task,
              "llm_calls": 0, "episodes": [], "jitter_m": args.jitter_m, "locomotion": args.locomotion}
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        rng = np.random.default_rng(seed + 1_000_003)
        with HumanoidEnvironment(args.task, views=tuple(args.views), budget_s=args.chunks * args.chunk_seconds) as env:
            env.reset(seed)
            use_gaze = args.mode == "wm" and not args.no_gaze and gaze_supported(env)
            if use_gaze and gaze_pose(rng, 0, 0) is None:
                use_gaze, report["gaze_note"] = False, "action schema requires a head position; gaze skipped"
            report["gaze"] = use_gaze
            recorder = EpisodeRecorder(env, output / f"{args.task}_seed{seed}")
            state = {"left": 0.0, "right": 0.0, "gaze": gaze_pose(rng, 0, 0) if use_gaze else None}
            for index in range(args.chunks):
                chunk = wrist_chunk(env, rng, args, index) if args.mode == "wrist" else wm_chunk(env, rng, args, index, state)
                outcome = recorder.execute(chunk)
                if outcome["failure"]:
                    break
            episode = recorder.finish({"run_kind": report["run_kind"], "agent": agent,
                                       "llm_calls": 0, "world_model_calls": 0})
            report["episodes"].append({"seed": seed, "episode": str(recorder.output),
                                       "chunks": len(episode["chunks"]), "metrics": episode["metrics"]})
    if args.export_dataset:
        from hwap.dataset import export_dataset
        report["dataset"] = export_dataset(output, args.export_dataset.resolve(), splits=args.splits)
    if args.wm_export:
        from hwap.dataset import export_world_model_dataset
        report["world_model_dataset"] = export_world_model_dataset(
            output, args.wm_export.resolve(), clip_frames=args.clip_frames, stride=args.clip_stride,
            views=args.views, splits=args.splits, video=args.video)
    write_json(output / "collection_report.json", report)
    print(json.dumps({"output": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
