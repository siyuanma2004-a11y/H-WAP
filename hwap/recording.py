"""Episode records: commands, executed states, calibration, images and replay."""
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from .images import png_bytes
from .provenance import source_hashes


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


class EpisodeRecorder:
    def __init__(self, env, output):
        self.env, self.output = env, Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.rows, self.chunks, self.frames = [], [], []
        self.started = time.monotonic()
        self.initial_qpos = env.sim.data.qpos.copy()
        self.initial_qvel = env.sim.data.qvel.copy()
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        self.metadata = {"schema_version": 1, "task": env.task, "seed": env.seed, "views": list(env.views),
                         "backend": env.controller.backend_name, "git_commit": commit, "goal": env.goal.tolist(),
                         "control_backend":env.backend,"sonic_model_dir":None if env.sonic_model_dir is None else str(env.sonic_model_dir),
                         "budget_s": env.budget_s, "physics_dt": float(env.sim.model.opt.timestep), "control_dt": 0.02,
                         "initial_qpos": self.initial_qpos.tolist(), "initial_qvel": self.initial_qvel.tolist()}
        self.metadata["robot_body_ids"] = env.robot_body_ids.tolist()
        self.metadata["source_sha256"] = source_hashes()
        self.metadata["sonic_model_revision"] = getattr(env.controller,"model_revision",None)
        self.metadata["robot_parent_ids"] = env.sim.model.body_parentid[env.robot_body_ids].tolist()
        self.capture(env, None, force_image=True)

    def capture(self, env, command, force_image=False):
        d = env.sim.data
        self.rows.append({"time": float(d.time), "qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "ctrl": d.ctrl.copy(),
            "base_pose": np.array(env.sim.base_pose), "left_wrist": env.sim.palm_position("left"),
            "right_wrist": env.sim.palm_position("right"), "robot_body_positions": d.xpos[env.robot_body_ids].copy(),
            "privileged_object_position": d.xpos[env.object_id].copy()})
        if force_image or len(self.rows)%5 == 0:
            index = len(self.rows)-1
            paths = {}
            for view, rgb in env.render().items():
                folder = self.output/"frames"/view
                folder.mkdir(parents=True, exist_ok=True)
                p = folder/f"{index:06d}.png"
                p.write_bytes(png_bytes(rgb))
                paths[view] = str(p.relative_to(self.output))
            self.frames.append({"row": index, "time": float(d.time), "images": paths, "cameras": env.camera_parameters()})

    def execute(self, chunk):
        start = len(self.rows)-1
        outcome = self.env.execute(chunk, callback=self.capture)
        self.capture(self.env, None, force_image=True)
        self.chunks.append({"chunk": chunk.to_dict(), "start_row": start, "end_row": len(self.rows)-1, "outcome": outcome})
        return outcome

    def finish(self, extra=None):
        arrays = {key: np.stack([r[key] for r in self.rows]) for key in self.rows[0]}
        np.savez_compressed(self.output/"trajectory.npz", **arrays)
        self.metadata.update({"wall_time_s": time.monotonic()-self.started, "rows": len(self.rows),
            "frames": self.frames, "chunks": self.chunks, "metrics": self.env.metrics(), **(extra or {})})
        if hasattr(self.env.controller,"inference_calls"):
            self.metadata["controller_inference"] = {"calls":self.env.controller.inference_calls,
                "seconds":self.env.controller.inference_seconds,"provider":"CPUExecutionProvider"}
        write_json(self.output/"episode.json", self.metadata)
        return self.metadata


def replay_episode(path, render=False, atol=1e-9):
    from .actions import CommandChunk
    from .environment import HumanoidEnvironment
    path = Path(path)
    record = json.loads((path/"episode.json").read_text())
    if "source_sha256" in record and record["source_sha256"] != source_hashes():
        raise ValueError("working source differs from the recorded episode; restore its recorded revision/files before replay")
    arrays = np.load(path/"trajectory.npz", allow_pickle=False)
    with HumanoidEnvironment(record["task"], views=tuple(record["views"]), budget_s=record["budget_s"],
            backend=record.get("control_backend","unitree"),sonic_model_dir=record.get("sonic_model_dir")) as env:
        env.reset(record["seed"])
        np.testing.assert_allclose(env.sim.data.qpos, record["initial_qpos"], atol=atol, rtol=0)
        maximum_error = 0.0
        for item in record["chunks"]:
            env.execute(CommandChunk.parse(item["chunk"]))
            error = float(np.max(np.abs(env.sim.data.qpos-arrays["qpos"][item["end_row"]])))
            maximum_error = max(maximum_error, error)
            if render:
                env.render()
        if maximum_error > atol:
            raise RuntimeError(f"replay divergence: {maximum_error} > {atol}")
        return {"max_qpos_error": maximum_error, "metrics": env.metrics(), "chunks": len(record["chunks"])}
