"""Run, replay and summarize small reproducible MuJoCo experiments."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from .actions import CommandChunk, Keyframe, Pose
from .agents import ReplayAgent, VisionChatAgent
from .environment import HumanoidEnvironment
from .planning import choose_oracle, local_variants
from .recording import EpisodeRecorder, replay_episode, write_json


def calibration_chunk(env):
    # State-informed calibration reference; not a visual or learned policy.
    current = Pose(tuple(env.sim.palm_position("right")))
    return CommandChunk("reach_green_marker", (Keyframe(0,right_wrist=current),
        Keyframe(1.5,right_wrist=Pose(tuple(env.goal))), Keyframe(2.5,right_wrist=Pose(tuple(env.goal)))), frame="world")


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        with HumanoidEnvironment(args.task, views=tuple(args.views), budget_s=args.budget_s,
                backend=args.backend,sonic_model_dir=args.sonic_model_dir) as env:
            env.reset(seed)
            if args.agent_config:
                agent = VisionChatAgent(args.agent_config)
            elif args.chunks:
                raw = json.loads(Path(args.chunks).read_text())
                agent = ReplayAgent([CommandChunk.parse(c) for c in raw])
            else:
                if args.task != "reach":
                    raise ValueError("pick_place needs supplied chunks or a configured visual agent")
                agent = ReplayAgent([calibration_chunk(env)])
            path = root/f"{args.method}_{args.task}_seed{seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
            recorder = EpisodeRecorder(env,path)
            feedback, searches, imagination_calls = None, [], 0
            error = None
            try:
                for _ in range(args.max_chunks):
                    public_feedback = None if feedback is None else {k:feedback[k] for k in ("duration_s","success","failure")}
                    chunk = agent.propose(env.observe(), public_feedback)
                    if chunk is None:
                        break
                    if args.method == "oracle":
                        chunk, candidates = choose_oracle(env, local_variants(chunk))
                        imagination_calls += len(candidates)
                        searches.append(candidates)
                    feedback = recorder.execute(chunk)
                    if feedback["success"] or feedback["failure"]:
                        break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            result = recorder.finish({"method":args.method,"agent":agent.name,"llm_calls":agent.calls,
                "llm_tokens":agent.tokens,"world_model_calls":imagination_calls,"searches":searches,"run_error":error,
                "agent_model":getattr(agent,"model",None),"agent_provider":getattr(agent,"provider",None),
                "agent_history":getattr(agent,"history",[]),
                "critic":"privileged_geometric_reference" if args.method == "oracle" else None})
            row = {"seed":seed,"task":args.task,"method":args.method,"backend":env.controller.backend_name,"agent":agent.name,
                **result["metrics"],"wall_time_s":result["wall_time_s"],"llm_calls":agent.calls,
                "llm_tokens":agent.tokens,"world_model_calls":imagination_calls,"run_error":error,"episode":str(path)}
            rows.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
    summary = {"episodes":len(rows),"successes":sum(r["success"] for r in rows),
        "success_rate":float(np.mean([r["success"] for r in rows])),"runs":rows}
    write_json(root/f"summary_{args.method}.json",summary)
    with (root/f"summary_{args.method}.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    if any(r["run_error"] for r in rows):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(prog="python -m hwap")
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--task",choices=["reach","pick_place"],default="reach")
    runner.add_argument("--method",choices=["direct","oracle"],default="direct")
    runner.add_argument("--seeds",type=int,nargs="+",default=[0])
    runner.add_argument("--views",nargs="+",default=["ego"])
    runner.add_argument("--output",default="runs/calibration")
    runner.add_argument("--budget-s",type=float,default=20)
    runner.add_argument("--max-chunks",type=int,default=8)
    runner.add_argument("--backend",choices=["unitree","sonic"],default="unitree")
    runner.add_argument("--sonic-model-dir",default="models/sonic_v1_1")
    group=runner.add_mutually_exclusive_group()
    group.add_argument("--chunks")
    group.add_argument("--agent-config")
    replay = sub.add_parser("replay");replay.add_argument("episode")
    replay.add_argument("--render",action="store_true")
    export = sub.add_parser("export");export.add_argument("runs");export.add_argument("output")
    video = sub.add_parser("video");video.add_argument("episode")
    video.add_argument("--view",default="ego");video.add_argument("--output")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "replay":
        print(json.dumps(replay_episode(args.episode,args.render),indent=2))
    elif args.command == "video":
        from .video import export_video
        print(json.dumps(export_video(args.episode,args.view,args.output),indent=2))
    else:
        from .dataset import export_dataset
        print(json.dumps(export_dataset(args.runs,args.output),indent=2))


if __name__ == "__main__":
    main()
