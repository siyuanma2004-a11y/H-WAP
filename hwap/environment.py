"""Task episodes with explicit privileged evaluation and restorable state."""
import copy
from dataclasses import replace
import mujoco
import numpy as np
from agent_g1.simulation import G1Simulation
from .actions import Keyframe, Pose
from .control import ChunkController


VIEWS = {"ego": "egocentric_camera", "workspace": "workspace_camera", "side": "side_camera"}


class HumanoidEnvironment:
    def __init__(self, task="reach", views=("ego",), width=320, height=240, budget_s=20, backend="unitree", sonic_model_dir=None):
        if task not in {"reach", "pick_place"} or not views or set(views)-set(VIEWS):
            raise ValueError("unknown task or camera view")
        if width < 16 or height < 16 or budget_s <= 0:
            raise ValueError("invalid image size or episode budget")
        self.sim = G1Simulation()
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
        self.episode_steps, self.hold_steps, self.lifted, self.failure = 0, 0, False, None

    def reset(self, seed=0):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.sim.reset(seed=None)
        self.controller.reset()
        self.goal = np.array([0.30, -0.25, 1.05]) if self.task == "reach" else np.array([0.60, -0.10, 0.775])
        self.goal[:2] += self.rng.uniform(-0.015, 0.015, 2)
        self.sim.model.site_pos[self.goal_id] = self.goal
        self.sim.data.qpos[self.object_qadr:self.object_qadr+2] = [0.45, 0.0] + self.rng.uniform(-0.015, 0.015, 2)
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
        position = rotation @ np.array(pose.position) + [x, y, 0]
        q = None
        if pose.quaternion is not None:
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)]), np.asarray(pose.quaternion))
        return Pose(tuple(position), None if q is None else tuple(q))

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
            head = self._world_pose(k.head,anchor,chunk.frame)
            if head is not None:
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
        obj = d.xpos[self.object_id]
        self.lifted = self.lifted or obj[2] > self.initial_object_z+0.06
        if self.task == "reach":
            good = np.linalg.norm(self.sim.palm_position("right")-self.goal) < 0.045
        else:
            speed = np.linalg.norm(d.qvel[self.object_vadr:self.object_vadr+3])
            good = self.lifted and np.linalg.norm(obj-self.goal) < 0.045 and speed < 0.1
        self.hold_steps = self.hold_steps+1 if good and not self.failure else 0

    def metrics(self):
        position = self.sim.palm_position("right") if self.task == "reach" else self.sim.data.xpos[self.object_id]
        return {"success": bool(self.hold_steps >= 100 and not self.failure), "failure": self.failure,
                "goal_error_m": float(np.linalg.norm(position-self.goal)), "lifted": bool(self.lifted),
                "pelvis_height_m": self.sim.pelvis_height, "physics_steps": self.episode_steps}

    def observe(self, render=True):
        robot = {"base_pose": list(self.sim.base_pose), "pelvis_height": self.sim.pelvis_height,
                 "left_wrist": self.sim.palm_position("left").tolist(), "right_wrist": self.sim.palm_position("right").tolist()}
        # Task instruction and robot proprioception are public; hidden object
        # coordinates, simulator state and evaluator truth are not.
        return {"task": self.task, "capabilities":{"backend":self.controller.backend_name,
                "head_torso_reference":self.controller.supports_head,"velocity":self.controller.supports_velocity,
                "waist_height":self.controller.supports_waist},
                "instruction": "Touch the green marker with your right palm" if self.task == "reach" else "Move the blue cube onto the green marker",
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
