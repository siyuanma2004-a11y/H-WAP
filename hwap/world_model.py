"""World-model interface used by the planner.

imagine(env, chunk, views, samples) -> {"model", "views", "timestamps", "frames":
[{view: rgb}] per sample, "public": {...}, "privileged": {...}, "disagreement"}.
Only "frames", "timestamps" and "public" may reach the agent or a visual critic.
"privileged" holds evaluator truth for the labelled oracle-critic reference.
"""
import base64
import json
import os
import urllib.request
import numpy as np
from .images import decode_png, png_bytes


def _robot_public(env):
    return {k: v for k, v in env.observe(render=False)["robot"].items()}


def sim_check(env, chunk, frame_every=5, render=False, views=None):
    """Roll the chunk in the simulator and restore: feasibility + whole-body poses.

    Frames are captured every `frame_every` control ticks (10 Hz by default),
    matching EpisodeRecorder. Returns robot body positions and camera
    calibration per frame so pose-image conditions can be rendered.
    """
    state = env.snapshot()
    rows = []
    saved_views = env.views
    if views is not None:
        env.views = tuple(views)

    def capture(environment, command):
        now = float(environment.sim.data.time)
        if rows and rows[-1]["time"] == now:
            return
        if command is None or environment.controller.tick % (10*frame_every) == 0:
            rows.append({"time": now,
                "robot_body_positions": environment.sim.data.xpos[environment.robot_body_ids].copy(),
                "cameras": environment.camera_parameters(),
                "images": environment.render() if render else None})
    try:
        start = float(env.sim.data.time)
        capture(env, None)
        outcome = env.execute(chunk, callback=capture)
        capture(env, None)
        final_robot = _robot_public(env)
    except ValueError as error:
        return {"feasible": False, "reason": str(error), "rows": [], "public": {"failure": "invalid_chunk"}}
    finally:
        env.restore(state)
        env.views = saved_views
    failure = outcome["failure"]
    return {"feasible": failure is None, "reason": failure, "rows": rows, "start_time": start,
            "public": {"duration_s": outcome["duration_s"], "failure": failure, "final_robot": final_robot},
            "privileged": outcome}


def command_pose_images(env, chunk, views, times):
    """Pose images drawn from the commanded targets only (the 'from commands'
    skeleton ablation). Shoulders, base pose and cameras come from the current
    state, so no simulated future enters the condition."""
    from .conditioning import (KinematicsModel, command_pose_image, command_pose_points, command_vector,
                               integrate_base_path)
    anchor = np.array(env.sim.base_pose)
    shoulders = KinematicsModel().shoulders_relative(env.sim.data.qpos)
    cameras = env.camera_parameters()
    commands = [command_vector(chunk, t, anchor) for t in times]
    velocities = [chunk.sample(t).velocity for t in times]
    path = integrate_base_path(anchor, np.array([0.0]+list(times)), np.array([chunk.sample(0).velocity]+velocities))
    images = {v: [] for v in views}
    for index, (vector, mask) in enumerate(commands):
        points = command_pose_points(vector, mask, path[index+1], shoulders, path[:index+2])
        for v in views:
            images[v].append(command_pose_image(points, cameras[v]))
    return images


def _disagreement(samples, views):
    """Mean per-pixel std (0..1) across samples of the final frame of each view."""
    if len(samples) < 2:
        return None
    values = [np.std(np.stack([s[-1][v].astype(float)/255 for s in samples]), axis=0).mean() for v in views]
    return float(np.mean(values))


class OracleWorldModel:
    """Simulator imagination: the paper's upper bound for a learned world model.

    Extra samples perturb the movable object's planar position by `perturb_m`
    to emulate world-model uncertainty; sample 0 is the unperturbed rollout.
    """
    name = "oracle_simulator"

    def __init__(self, perturb_m=0.0, seed=0, frame_every=5):
        self.perturb_m, self.frame_every = float(perturb_m), int(frame_every)
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def imagine(self, env, chunk, views=None, samples=1):
        views = tuple(views or env.views)
        sequences, first = [], None
        for index in range(max(1, int(samples))):
            state = env.snapshot()
            try:
                if index and self.perturb_m and hasattr(env, "object_qadr"):
                    env.sim.data.qpos[env.object_qadr:env.object_qadr+2] += self.rng.uniform(-self.perturb_m, self.perturb_m, 2)
                result = sim_check(env, chunk, self.frame_every, render=True, views=views)
            finally:
                env.restore(state)
            self.calls += 1
            sequences.append([row["images"] for row in result["rows"]])
            first = first or result
        return {"model": self.name, "views": list(views), "samples": len(sequences),
                "timestamps": [row["time"]-first.get("start_time", 0.0) for row in first["rows"]],
                "frames": sequences, "public": first["public"], "privileged": first.get("privileged", {}),
                "disagreement": _disagreement(sequences, views), "body_rows": first["rows"]}


class RemoteVideoWorldModel:
    """Client for a learned action-conditioned video model served over HTTP.

    Request JSON: {"model", "views", "history": {view: [png_b64]}, "pose_conditions":
    {view: [png_b64 per future frame]}, "timestamps", "chunk", "samples"}.
    Response JSON: {"samples": [{view: [png_b64 per future frame]}], "usage": {...}}.
    Pose images come from sim_check (whole-body pose, "sim") or from the command
    targets ("command"); this is the proposal's skeleton-source ablation.
    """
    name = "remote_video_world_model"

    def __init__(self, config):
        self.config = config
        self.url = config["base_url"].rstrip("/")+"/imagine"
        if not self.url.startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
            raise ValueError("use HTTPS or a local inference endpoint")
        self.key = os.environ.get(config.get("api_key_env", "H_WAP_WM_KEY"))
        self.pose_source = config.get("pose_source", "sim")
        if self.pose_source not in {"sim", "command"}:
            raise ValueError("pose_source must be sim or command")
        self.timeout = float(config.get("timeout_s", 300))
        self.calls = 0

    def pose_conditions(self, env, chunk, views, frame_every=5):
        from .dataset import skeleton_image
        check = sim_check(env, chunk, frame_every, render=False, views=views)
        ids = env.robot_body_ids.tolist()
        parents = env.sim.model.body_parentid[env.robot_body_ids].tolist()
        if self.pose_source == "command":
            return command_pose_images(env, chunk, views, [row["time"]-check["start_time"] for row in check["rows"][1:]]), check
        images = {v: [skeleton_image(row["robot_body_positions"], ids, parents, row["cameras"][v])
                      for row in check["rows"][1:]] for v in views}
        return images, check

    def imagine(self, env, chunk, views=None, samples=1):
        views = tuple(views or env.views)
        current = env.render()
        conditions, check = self.pose_conditions(env, chunk, views)
        encode = lambda rgb: base64.b64encode(png_bytes(rgb)).decode()
        body = {"model": self.config.get("model"), "views": list(views), "samples": int(samples),
                "history": {v: [encode(current[v])] for v in views},
                "pose_conditions": {v: [encode(i) for i in conditions[v]] for v in views},
                "timestamps": [row["time"]-check.get("start_time", 0.0) for row in check["rows"][1:]],
                "chunk": chunk.to_dict()}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer "+self.key
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        self.calls += 1
        sequences = [[{v: decode_png(base64.b64decode(s[v][i])) for v in views} for i in range(len(s[views[0]]))]
                     for s in result["samples"]]
        return {"model": self.config.get("model", self.name), "views": list(views), "samples": len(sequences),
                "timestamps": body["timestamps"], "frames": sequences,
                "public": {"duration_s": chunk.duration, "failure": check["public"].get("failure")},
                "privileged": {}, "disagreement": _disagreement(sequences, views), "usage": result.get("usage", {})}


def world_model_from_config(config):
    kind = config.get("kind", "oracle")
    if kind == "oracle":
        return OracleWorldModel(config.get("perturb_m", 0.0), config.get("seed", 0), config.get("frame_every", 5))
    if kind == "remote":
        return RemoteVideoWorldModel(config)
    raise ValueError("unknown world model kind")
