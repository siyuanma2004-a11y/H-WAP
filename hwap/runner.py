"""Harness experiments from JSON configs; also the simulator evaluator for the gate.

Config (schema_version 2), paths relative to the config file:
  {"schema_version": 2, "name": str, "tasks": [..], "seeds": [..], "views": [..],
   "backend": "unitree"|"sonic", "budget_s": float, "width": int, "height": int,
   "planner": {"kind": "llm"|"calibration", "llm_config": path},
   "critic": {"kind": "vlm"|"privileged"|"none", "llm_config": path (default: planner's)},
   "world_model": {"kind": "oracle"|"remote"|"none", ...world_model_from_config fields},
   "store": {"seed": path, "path": path|null (default <output>/store)},
   "harness": {HarnessConfig fields},
   "reflection": {"enabled": bool, "regression_set": ["task:seed", ..]}}
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from .actions import CommandChunk, Keyframe, Pose
from .harness import HarnessConfig, LLMPlanner, ScriptedPlanner, reflect, run_episode


ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"schema_version", "name", "tasks", "seeds", "views", "backend", "budget_s", "width", "height",
          "planner", "critic", "world_model", "store", "harness", "reflection", "sonic_model_dir"}


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if set(config) - FIELDS or config.get("schema_version") != 2:
        raise ValueError(f"unknown harness experiment fields or schema_version: {sorted(set(config)-FIELDS)}")
    if not config.get("tasks") or not config.get("seeds") or any(type(s) is not int or s < 0 for s in config["seeds"]):
        raise ValueError("tasks and non-negative integer seeds are required")
    for section, key in (("planner", "llm_config"), ("critic", "llm_config"), ("store", "seed"), ("store", "path"),
                         ("world_model", "config")):
        value = config.get(section, {}).get(key)
        if value and not Path(value).is_absolute():
            config[section][key] = str((path.parent/value).resolve())
    HarnessConfig.from_dict(config.get("harness", {}))
    if config.get("planner", {}).get("kind", "calibration") not in {"llm", "calibration"}:
        raise ValueError("planner.kind must be llm or calibration")
    if config.get("critic", {}).get("kind", "none") not in {"vlm", "privileged", "none"}:
        raise ValueError("critic.kind must be vlm, privileged or none")
    if config.get("critic", {}).get("kind") == "vlm" and not (config["critic"].get("llm_config") or config.get("planner", {}).get("llm_config")):
        raise ValueError("a VLM critic needs an llm_config")
    return config


def calibration_planner(env):
    """State-informed replay calibration (reads the privileged goal); never a research result."""
    def plan(context, observation, step):
        if step:
            return None
        start = Pose(tuple(env.sim.palm_position("right")))
        goal = Pose(tuple(env.goal))
        return CommandChunk("calibration_reach", (Keyframe(0, right_wrist=start), Keyframe(1.5, right_wrist=goal),
                                                  Keyframe(2.5, right_wrist=goal)), frame="world")
    planner = ScriptedPlanner(plan)
    planner.name = "replay_calibration"
    return planner


class Components:
    """LLM clients, world model, critic and store shared across a batch."""

    def __init__(self, config, store_root):
        from .llm import client_from_config
        from .world_model import world_model_from_config
        self.config = config
        planner = config.get("planner", {"kind": "calibration"})
        self.planner_client = client_from_config(planner["llm_config"]) if planner.get("kind") == "llm" else None
        critic = config.get("critic", {"kind": "none"})
        self.critic_client = None
        if critic.get("kind") == "vlm":
            path = critic.get("llm_config") or planner.get("llm_config")
            self.critic_client = self.planner_client if path == planner.get("llm_config") and self.planner_client else client_from_config(path)
        wm = config.get("world_model", {"kind": "none"})
        self.world_model = None if wm.get("kind", "none") == "none" else world_model_from_config(wm)
        self.harness = HarnessConfig.from_dict(config.get("harness", {}))
        self.store = None
        if config.get("store"):
            from .store import VersionedStore
            self.store = VersionedStore.from_seed(store_root, config["store"].get("seed", ROOT/"harness_seed"))

    def memory_skills_guidance(self, store=None):
        store = store or self.store
        if store is None:
            return None, None, {}
        from .improvement import Guidance
        from .memory import MemoryBank
        from .skills import SkillLibrary
        return MemoryBank(store), SkillLibrary(store), Guidance(store).all()

    def make(self, env, store=None):
        from .critic import PrivilegedCritic, VLMCritic
        memory, skills, guidance = self.memory_skills_guidance(store)
        if self.planner_client is not None:
            planner = LLMPlanner(self.planner_client, guidance.get("planner", "")+"\n"+guidance.get("perception", ""),
                                 allow_gaze=self.harness.gaze == "llm")
        else:
            planner = calibration_planner(env)
        kind = self.config.get("critic", {}).get("kind", "none")
        critic = PrivilegedCritic() if kind == "privileged" else VLMCritic(self.critic_client, guidance.get("critic", "")) if kind == "vlm" else None
        return planner, critic, memory, skills


def run_one(config, components, task, seed, output, store=None):
    from .environment import HumanoidEnvironment
    from .recording import EpisodeRecorder
    with HumanoidEnvironment(task, views=tuple(config.get("views", ["ego"])), budget_s=config.get("budget_s", 20),
                             width=config.get("width", 320), height=config.get("height", 240),
                             backend=config.get("backend", "unitree"), sonic_model_dir=config.get("sonic_model_dir")) as env:
        env.reset(seed)
        planner, critic, memory, skills = components.make(env, store)
        calls_before = (components.planner_client.calls, components.planner_client.tokens) if components.planner_client else (0, 0)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        recorder = EpisodeRecorder(env, Path(output)/f"{config.get('name', 'harness')}_{task}_seed{seed}_{stamp}")
        extra = run_episode(env, planner, recorder, components.harness, components.world_model, critic, memory, skills)
        if components.planner_client:
            extra["llm_calls"] = components.planner_client.calls-calls_before[0]
            extra["llm_tokens"] = components.planner_client.tokens-calls_before[1]
        extra["method"] = config.get("name", "harness")
        extra["agent"] = planner.name
        return recorder.finish(extra), recorder.output


def make_sim_evaluator(config, components, work_dir):
    """evaluate(store_path, ["task:seed", ..]) -> success rate, for TestingGate."""
    def evaluate(store_path, task_set):
        from .store import VersionedStore
        store = VersionedStore(store_path)
        per_task = {}
        for item in task_set:
            task, seed = item.rsplit(":", 1)
            record, _ = run_one(config, components, task, int(seed), Path(work_dir)/Path(store_path).name, store)
            per_task[item] = bool(record["metrics"]["success"]) and not record.get("run_error")
        return {"success_rate": float(np.mean(list(per_task.values()))) if per_task else 0.0,
                "per_task": per_task, "n": len(per_task)}
    return evaluate


def run_experiment(config_path, output):
    config = load_config(config_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/"harness_experiment.json").write_text(json.dumps(config, indent=2)+"\n")
    store_root = Path(config.get("store", {}).get("path") or output/"store") if config.get("store") else None
    components = Components(config, store_root)
    gate = None
    reflection = config.get("reflection", {})
    if reflection.get("enabled"):
        if components.store is None or components.planner_client is None:
            raise ValueError("reflection needs a store and an LLM planner")
        from .improvement import TestingGate
        evaluator = make_sim_evaluator(config, components, output/"gate_runs")
        gate = TestingGate(components.store, evaluator, ablate=components.harness.gate == "none")
    rows = []
    for task in config["tasks"]:
        for seed in config["seeds"]:
            record, path = run_one(config, components, task, seed, output)
            row = {"task": task, "seed": seed, "method": config.get("name"), **record["metrics"],
                   "run_error": record.get("run_error"), "wall_time_s": record["wall_time_s"],
                   **{k: record.get(k) for k in ("llm_calls", "llm_tokens", "world_model_calls", "critic_calls",
                                                 "sim_checks", "invalid_chunks", "gaze_chunks")},
                   "store_head": components.store.head() if components.store else None, "episode": str(path)}
            if gate is not None:
                memory, skills, _ = components.memory_skills_guidance()
                result = reflect(components.planner_client, {**record, "task": task}, memory, skills, gate,
                                 reflection.get("regression_set", []), [f"{task}:{seed}"])
                (path/"reflection.json").write_text(json.dumps(result, indent=2, default=str)+"\n")
                row["accepted_changes"] = sum(d["status"].startswith("accepted") for d in result["decisions"])
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = {"name": config.get("name"), "episodes": len(rows), "success_rate": float(np.mean([r["success"] for r in rows])),
               "llm_log": components.planner_client.log if components.planner_client else [], "runs": rows}
    (output/"harness_summary.json").write_text(json.dumps(summary, indent=2, default=str)+"\n")
    if store_root is not None:
        summary["store"] = str(store_root)
    return summary
