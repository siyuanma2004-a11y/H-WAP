#!/usr/bin/env python3
"""Run a pinned direct/oracle experiment, summarize results and verify replay."""
import argparse
import json
import math
import re
from pathlib import Path
import statistics
import subprocess
import sys

from _bootstrap import ROOT, configure_render, new_run_path, write_json


def validate(config):
    allowed = {"schema_version", "name", "task", "methods", "seeds", "views", "backend",
               "budget_s", "max_chunks", "agent", "agent_config", "verify_replay"}
    if set(config) - allowed or config.get("schema_version") != 1:
        raise ValueError("Unknown experiment fields or schema_version")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config.get("name", "experiment")):
        raise ValueError("name must be a simple identifier, without directory separators")
    if config.get("task") not in {"reach", "pick_place"}:
        raise ValueError("Choose reach or pick_place")
    methods = config.get("methods", [])
    if not methods or len(set(methods)) != len(methods) or set(methods) - {"direct", "oracle"}:
        raise ValueError("methods must contain distinct direct/oracle entries")
    seeds = config.get("seeds", [])
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct non-negative integers")
    views = config.get("views", [])
    if not views or set(views) - {"ego", "workspace", "side"} or len(set(views)) != len(views):
        raise ValueError("Choose distinct ego/workspace/side views")
    if config.get("agent") not in {"calibration", "vision"}:
        raise ValueError("Explicitly select calibration or vision")
    if config["agent"] == "calibration" and config["task"] != "reach":
        raise ValueError("Calibration only implements reach; pick_place needs a visual Agent or explicit chunks")
    if config.get("backend", "unitree") not in {"unitree", "sonic"}:
        raise ValueError("Choose unitree or sonic")
    budget = config.get("budget_s", 20)
    if not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("budget_s must be positive and finite")
    if type(config.get("max_chunks", 8)) is not int or not 1 <= config.get("max_chunks", 8) <= 100:
        raise ValueError("max_chunks must be between 1 and 100")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/smoke.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--agent-config", type=Path, help="Explicitly use a configured visual model")
    parser.add_argument("--sonic-model-dir", type=Path, default=ROOT / "models/sonic_v1_1")
    parser.add_argument("--gl", choices=("auto", "egl", "osmesa", "glfw"), default="auto")
    parser.add_argument("--export-dataset", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.agent_config:
        config["agent"], config["agent_config"] = "vision", str(args.agent_config.resolve())
    try:
        validate(config)
        configure_render(args.gl)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    agent_config = None
    if config["agent"] == "vision":
        if not config.get("agent_config"):
            parser.error("A visual Agent needs --agent-config or agent_config in the experiment JSON")
        agent_config = Path(config["agent_config"])
        if not agent_config.is_absolute():
            agent_config = args.config.resolve().parent / agent_config
        if not agent_config.is_file():
            parser.error("Agent configuration does not exist: " + str(agent_config))
    output = (args.output or new_run_path(config.get("name", "experiment"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", {"config": config, "python": sys.executable,
               "agent_config": None if agent_config is None else str(agent_config.resolve())})
    report = {"status": "running", "experiment_kind": "calibration_smoke" if config["agent"] == "calibration" else "visual_agent",
              "oracle_critic": "privileged_geometric_reference", "methods": {}, "replays": []}
    try:
        for method in config["methods"]:
            command = [sys.executable, "-m", "hwap", "run", "--task", config["task"],
                       "--method", method, "--backend", config.get("backend", "unitree"),
                       "--seeds", *map(str, config["seeds"]), "--views", *config["views"],
                       "--budget-s", str(config.get("budget_s", 20)),
                       "--max-chunks", str(config.get("max_chunks", 8)), "--output", str(output),
                       "--sonic-model-dir", str(args.sonic_model_dir.resolve())]
            if agent_config:
                command += ["--agent-config", str(agent_config.resolve())]
            subprocess.run(command, cwd=ROOT, check=True)
            summary = json.loads((output / f"summary_{method}.json").read_text())
            rows = summary["runs"]
            report["methods"][method] = {
                "episodes": len(rows), "success_rate": summary["success_rate"],
                "mean_goal_error_m": statistics.mean(row["goal_error_m"] for row in rows),
                "mean_wall_time_s": statistics.mean(row["wall_time_s"] for row in rows),
                "falls": sum(row["failure"] == "fall" for row in rows),
                "llm_calls": sum(row["llm_calls"] for row in rows),
                "world_model_calls": sum(row["world_model_calls"] for row in rows)}
            if config.get("verify_replay", True):
                from hwap.recording import replay_episode
                for row in rows:
                    replay = replay_episode(row["episode"])
                    report["replays"].append({"episode": row["episode"], **replay})
        if args.export_dataset:
            from hwap.dataset import export_dataset
            report["dataset"] = export_dataset(output, args.export_dataset.resolve())
        report["status"] = "complete"
    except Exception as error:
        report["status"], report["error"] = "failed", f"{type(error).__name__}: {error}"
    write_json(output / "experiment_report.json", report)
    print(json.dumps({"output": str(output), **report}, indent=2))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
