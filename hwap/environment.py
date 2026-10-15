"""Task episodes with explicit privileged evaluation and restorable state."""
import copy
from dataclasses import replace
import mujoco
import numpy as np
from agent_g1.simulation import G1Simulation
from .actions import EGO_CAMERA_AXIS, EGO_CAMERA_OFFSET, Keyframe, Pose
from .control import ChunkController, WAIST_JOINT_NAMES
from .tasks import get_task


VIEWS = {"ego": "egocentric_camera", "workspace": "workspace_camera", "side": "side_camera"}
RENDERED_GEOM_GROUPS = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)  # mujoco.Renderer defaults


class HumanoidEnvironment:
    def __init__(self, task="reach", views=("ego",), width=320, height=240, budget_s=20, backend="unitree", sonic_model_dir=None):
        if not views or set(views)-set(VIEWS):
            raise ValueError("unknown camera view")
        if width < 16 or height < 16 or budget_s <= 0:
            raise ValueError("invalid image size or episode budget")
        self.task_spec = get_task(task)
        self.sim = G1Simulation(spec_hook=self.task_spec.scene_hook if self.task_spec.occluders else None)
        if backend == "unitree":
            self.controller = ChunkController(self.sim)
        elif backend == "sonic":
            from .sonic import SonicOnnxController
            if sonic_model_dir is None:
                raise ValueError("SONIC model directory is required")
            self.controller = SonicOnnxController(self.sim,sonic_model_dir)
        else:
            raise ValueError("unknown control backend")
        self.backend,self.sonic_model_dir = backend,sonic_model_dir
        self.task, self.views = task, tuple(views)
        self.width, self.height, self.budget_s = width, height, budget_s
        self.renderer = None
        def belongs_to_robot(body):
            while body:
                if body == self.sim.pelvis_body_id:
                    return True
                body = int(self.sim.model.body_parentid[body])
            return False
        self.robot_body_ids = np.array([b for b in range(self.sim.model.nbody) if belongs_to_robot(b)], dtype=int)
        self.rng = np.random.default_rng()
        self.goal_id = self.sim._required_id(mujoco.mjtObj.mjOBJ_SITE, "goal_marker")
        self.object_id = self.sim._required_id(mujoco.mjtObj.mjOBJ_BODY, "target_object")
        self.object_joint = self.sim._required_id(mujoco.mjtObj.mjOBJ_JOINT, "target_object_joint")
        self.object_qadr = self.sim.model.jnt_qposadr[self.object_joint]
        self.object_vadr = self.sim.model.jnt_dofadr[self.object_joint]
        self.occluder_ids = np.array([self.sim._required_id(mujoco.mjtObj.mjOBJ_BODY, name) for name, _, _ in self.task_spec.occluders], dtype=int)
        self.torso_id = self.sim._required_id(mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.waist_qadr = self.sim.model.jnt_qposadr[[self.sim._required_id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in WAIST_JOINT_NAMES]]
        self.layout = None
        self.episode_steps, self.hold_steps, self.lifted, self.failure = 0, 0, False, None

    def reset(self, seed=0):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.sim.reset(seed=None)
        self.controller.reset()
        # Privileged layout: evaluator and recordings only, never in observe().
        self.layout = self.task_spec.sample_layout(self.rng)
        self.goal = np.array(self.layout["goal"])
        self.sim.model.site_pos[self.goal_id] = self.goal
        self.sim.data.qpos[self.object_qadr:self.object_qadr+2] = self.layout["object_xy"]
        for body, (name, _, _) in zip(self.occluder_ids, self.task_spec.occluders):
            self.sim.model.body_pos[body] = self.layout["occluders"][name]
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.initial_object_z = float(self.sim.data.xpos[self.object_id, 2])
        # Stabilize the controller before the episode. This state is recorded.
        for _ in range(100):
            self.controller.step(Keyframe(0))
        self.episode_steps, self.hold_steps, self.lifted, self.failure = 0, 0, False, None
        return self.observe(render=False)

    def snapshot(self):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        physics = np.empty(mujoco.mj_stateSize(self.sim.model, spec))
        mujoco.mj_getState(self.sim.model, self.sim.data, physics, spec)
        return {"physics": physics, "controller": self.controller.snapshot(),
                "rng": copy.deepcopy(self.rng.bit_generator.state),
                "task_state": (self.episode_steps, self.hold_steps, self.lifted, self.failure)}

    def restore(self, snapshot):
        mujoco.mj_setState(self.sim.model, self.sim.data, snapshot["physics"], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.controller.restore(snapshot["controller"])
        self.rng.bit_generator.state = copy.deepcopy(snapshot["rng"])
        self.episode_steps, self.hold_steps, self.lifted, self.failure = snapshot["task_state"]

    def _world_pose(self, pose, anchor, frame):
        if pose is None or frame == "world":
            return pose
        x, y, yaw = anchor
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0,0,1]])
        position = None if pose.position is None else tuple(rotation @ np.array(pose.position) + [x, y, 0])
        q = None
        if pose.quaternion is not None:
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)]), np.asarray(pose.quaternion))
        return Pose(position, None if q is None else tuple(q))

    def validate_chunk(self, chunk):
        dt = self.sim.model.opt.timestep
        steps = int(round(chunk.duration/dt))
        if not np.isclose(steps*dt, chunk.duration, atol=1e-9):
            raise ValueError("chunk duration must be a multiple of the physics timestep")
        if (self.episode_steps+steps)*dt > self.budget_s+1e-9:
            raise ValueError("chunk exceeds the episode simulation-time budget")
        anchor = self.sim.base_pose
        for k in chunk.keyframes:
            if (k.head is not None and not self.controller.supports_head) or (k.waist_height is not None and not self.controller.supports_waist):
                raise ValueError("the selected backend cannot execute the head/waist request")
            if np.any(np.asarray(k.velocity)!=0) and not self.controller.supports_velocity:
                raise ValueError("the selected backend has no velocity planner")
            if k.head is not None and k.head.position is not None and not self.controller.supports_head_position:
                raise ValueError("this backend only supports orientation-only head (torso gaze) commands")
            head = self._world_pose(k.head,anchor,chunk.frame)
            if head is not None and head.position is not None:
                p = np.array(head.position)
                if np.linalg.norm(p[:2]-np.array(anchor[:2])) > 0.4 or not 1.0 <= p[2] <= 1.7:
                    raise ValueError("torso/head reference exceeds the experimental workspace")
            for side in ("left", "right"):
                pose = self._world_pose(getattr(k, side+"_wrist"), anchor, chunk.frame)
                if pose is not None:
                    p = np.array(pose.position)
                    if np.linalg.norm(p[:2]-np.array(anchor[:2])) > 0.9 or not 0.55 <= p[2] <= 1.55:
                        raise ValueError("wrist target exceeds the experimental workspace")
        return steps, anchor

    def execute(self, chunk, callback=None):
        steps, anchor = self.validate_chunk(chunk)
        start_time = float(self.sim.data.time)
        for i in range(steps):
            command = chunk.sample((i+1)*self.sim.model.opt.timestep)
            command = replace(command, left_wrist=self._world_pose(command.left_wrist, anchor, chunk.frame),
                              right_wrist=self._world_pose(command.right_wrist, anchor, chunk.frame),
                              head=self._world_pose(command.head,anchor,chunk.frame))
            self.controller.step(command)
            self.episode_steps += 1
            self._evaluate()
            if callback is not None and self.controller.tick % 10 == 0:
                callback(self, command)
            if self.failure:
                break
        return {"duration_s": float(self.sim.data.time-start_time), **self.metrics()}

    def _evaluate(self):
        d = self.sim.data
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            self.failure = "non_finite_state"
        elif not 0.60 < self.sim.pelvis_height < 1.0:
            self.failure = "fall"
        elif self.sim.table_collision_contacts():
            self.failure = "robot_table_collision"
        elif self.sim.non_hand_object_collision_contacts():
            self.failure = "non_hand_object_collision"
        elif len(self.occluder_ids) and self.occluder_contacts():
            self.failure = "robot_occluder_collision"
        obj = d.xpos[self.object_id]
        self.lifted = self.lifted or obj[2] > self.initial_object_z+0.06
        if self.task_spec.family == "reach":
            good = np.linalg.norm(self.sim.palm_position(self.task_spec.hand)-self.goal) < 0.045
        else:
            speed = np.linalg.norm(d.qvel[self.object_vadr:self.object_vadr+3])
            good = self.lifted and np.linalg.norm(obj-self.goal) < 0.045 and speed < 0.1
        self.hold_steps = self.hold_steps+1 if good and not self.failure else 0

    def occluder_contacts(self):
        robot, props = set(self.robot_body_ids.tolist()), set(self.occluder_ids.tolist())
        m, d = self.sim.model, self.sim.data
        n = d.ncon
        pairs = zip(m.geom_bodyid[d.contact.geom1[:n]], m.geom_bodyid[d.contact.geom2[:n]])
        return [(int(a), int(b)) for a, b in pairs if (a in robot and b in props) or (b in robot and a in props)]

    def task_target(self):
        """Privileged point the task needs to see: the marker, or the cube for pick_place."""
        return self.goal.copy() if self.task_spec.family == "reach" else self.sim.data.xpos[self.object_id].copy()

    def target_visibility(self, view="ego", point=None):
        """Privileged evaluator check: frustum projection plus an mj_ray occlusion test."""
        m, d = self.sim.model, self.sim.data
        point = self.task_target() if point is None else np.asarray(point, dtype=float)
        camera = self.sim._required_id(mujoco.mjtObj.mjOBJ_CAMERA, VIEWS[view])
        origin, rotation = d.cam_xpos[camera].copy(), d.cam_xmat[camera].reshape(3, 3)
        local = (point-origin) @ rotation
        depth = -local[2]
        f = self.height/(2*np.tan(np.deg2rad(m.cam_fovy[camera])/2))
        u, v = (self.width/2+f*local[0]/depth, self.height/2-f*local[1]/depth) if depth > 1e-6 else (np.nan, np.nan)
        in_frustum = bool(depth > 0.02 and 0 <= u < self.width and 0 <= v < self.height)
        distance = float(np.linalg.norm(point-origin))
        geom = np.zeros(1, dtype=np.int32)
        exclude = self.object_id if self.task_spec.family == "pick_place" else -1
        hit = mujoco.mj_ray(m, d, origin, (point-origin)/distance, RENDERED_GEOM_GROUPS, 1, exclude, geom)
        occluded = bool(0 <= hit < distance-0.02)
        return {"view": view, "in_frustum": in_frustum, "occluded": occluded, "visible": in_frustum and not occluded,
                "pixel": [float(u), float(v)], "blocking_geom": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(geom[0])) if occluded else None}

    def metrics(self):
        position = self.sim.palm_position(self.task_spec.hand) if self.task_spec.family == "reach" else self.sim.data.xpos[self.object_id]
        return {"success": bool(self.hold_steps >= 100 and not self.failure), "failure": self.failure,
                "goal_error_m": float(np.linalg.norm(position-self.goal)), "lifted": bool(self.lifted),
                "pelvis_height_m": self.sim.pelvis_height, "physics_steps": self.episode_steps,
                "target_visible_ego": self.target_visibility("ego")["visible"]}

    def head_state(self):
        d = self.sim.data
        waist = d.qpos[self.waist_qadr]
        return {"quaternion": d.xquat[self.torso_id].tolist(), "torso_position": d.xpos[self.torso_id].tolist(),
                "waist": {"yaw": float(waist[0]), "roll": float(waist[1]), "pitch": float(waist[2])}}

    def capabilities(self):
        c = self.controller
        gaze = None if c.gaze_limits is None else {**{k: list(v) for k, v in c.gaze_limits.items()},
            "while_walking": {k: list(v) for k, v in c.walking_gaze_limits.items()},
            "reference": "head quaternion = torso_link orientation; waist angles relative to the pelvis; positive pitch looks down",
            "note": "requests outside the limits are clipped; a forward lean slowly creeps the base forward",
            "position": c.supports_head_position, "camera_offset_in_torso": list(EGO_CAMERA_OFFSET),
            "camera_axis_in_torso": list(EGO_CAMERA_AXIS)}
        return {"backend": c.backend_name, "head_torso_reference": c.supports_head, "head_position": c.supports_head_position,
                "gaze": gaze, "velocity": c.supports_velocity, "waist_height": c.supports_waist}

    def observe(self, render=True):
        robot = {"base_pose": list(self.sim.base_pose), "pelvis_height": self.sim.pelvis_height,
                 "left_wrist": self.sim.palm_position("left").tolist(), "right_wrist": self.sim.palm_position("right").tolist(),
                 "head": self.head_state()}
        # Task instruction and robot proprioception are public; layouts, hidden
        # object coordinates, simulator state and evaluator truth are not.
        return {"task": self.task, "capabilities": self.capabilities(), "instruction": self.task_spec.instruction,
                "robot": robot, "cameras":self.camera_parameters(), "images": self.render() if render else {}}

    def camera_parameters(self):
        result = {}
        for view in self.views:
            camera = self.sim._required_id(mujoco.mjtObj.mjOBJ_CAMERA, VIEWS[view])
            f = self.height/(2*np.tan(np.deg2rad(self.sim.model.cam_fovy[camera])/2))
            result[view] = {"position": self.sim.data.cam_xpos[camera].tolist(),
                "rotation_world_from_camera": self.sim.data.cam_xmat[camera].reshape(3,3).tolist(),
                "intrinsics": [[f,0,self.width/2],[0,f,self.height/2],[0,0,1]], "width": self.width, "height": self.height}
        return result

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.sim.model, height=self.height, width=self.width)
        frames = {}
        for view in self.views:
            self.renderer.update_scene(self.sim.data, camera=VIEWS[view])
            frames[view] = self.renderer.render().copy()
        return frames

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
