"""Cartesian tracking and a single 500 Hz physics / 50 Hz control loop."""
import copy
import mujoco
import numpy as np
from agent_g1.arms import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
from agent_g1.locomotion import LoadedG1LocomotionPolicy


WAIST_JOINT_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
# Gaze limits (rad) relative to the pelvis. The walking policy was trained with a
# fixed upper body. Standing 4.5 s holds at every corner of GAZE_LIMITS and
# walking at every corner of WALKING_GAZE_LIMITS did not fall (pelvis > 0.76 m);
# roll 0.15 combined with yaw 0.6 and a backward lean fell. A forward lean makes
# the stepping policy creep forward (about 0.16 m/s per rad), so looks that
# lean toward the table should be brief.
GAZE_LIMITS = {"yaw": (-0.6, 0.6), "roll": (-0.08, 0.08), "pitch": (-0.2, 0.3)}
WALKING_GAZE_LIMITS = {"yaw": (-0.4, 0.4), "roll": (-0.08, 0.08), "pitch": (-0.1, 0.2)}
GAZE_RATE = 0.02  # rad per 50 Hz control tick


def waist_angles(relative):
    """Decompose R = Rz(yaw) Rx(roll) Ry(pitch), the G1 waist joint chain."""
    return (float(np.arctan2(-relative[0, 1], relative[1, 1])), float(np.arcsin(np.clip(relative[2, 1], -1, 1))),
            float(np.arctan2(-relative[2, 0], relative[2, 2])))


class ChunkController:
    backend_name = "unitree_12dof_cartesian_ik"
    supports_head = True  # orientation-only torso gaze through the waist joints
    supports_head_position = False
    supports_velocity = True
    supports_waist = False
    gaze_limits = GAZE_LIMITS
    walking_gaze_limits = WALKING_GAZE_LIMITS

    def __init__(self, simulation):
        self.sim = simulation
        self.walk = LoadedG1LocomotionPolicy.from_official(simulation, backend="numpy")
        self.config = self.walk.config
        self.tick = 0
        self.action = np.zeros(12)
        self.arm_maps = {}
        for side, names in (("left", LEFT_ARM_JOINT_NAMES), ("right", RIGHT_ARM_JOINT_NAMES)):
            joints = np.array([simulation._required_id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in names])
            actuators = np.array([simulation._required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names])
            self.arm_maps[side] = (joints, actuators)
        self.waist_joints = np.array([simulation._required_id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in WAIST_JOINT_NAMES])
        self.waist_actuators = np.array([simulation._required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in WAIST_JOINT_NAMES])
        simulation.model.actuator_gainprm[:12, 0] = self.config.kps
        simulation.model.actuator_gainprm[:12, 1:] = 0
        simulation.model.actuator_biasprm[:12, 0] = 0
        simulation.model.actuator_biasprm[:12, 1] = -np.array(self.config.kps)
        simulation.model.actuator_biasprm[:12, 2] = -np.array(self.config.kds)

    def reset(self):
        self.tick = 0
        self.action[:] = 0
        self.walk.policy_model.reset_memory()
        self.sim.data.ctrl[:12] = self.config.default_angles

    def snapshot(self):
        p = self.walk.policy_model
        return copy.deepcopy((self.tick, self.action, p.hidden_state, p.cell_state))

    def restore(self, state):
        self.tick, self.action, hidden, cell = copy.deepcopy(state)
        self.walk.policy_model.hidden_state[:] = hidden
        self.walk.policy_model.cell_state[:] = cell

    def _arm(self, side, pose):
        if pose is None:
            return
        model, data = self.sim.model, self.sim.data
        site = getattr(self.sim, side+"_palm_site_id")
        joints, actuators = self.arm_maps[side]
        dofs, addresses = model.jnt_dofadr[joints], model.jnt_qposadr[joints]
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site)
        error, jac = np.array(pose.position)-data.site_xpos[site], jp[:, dofs]
        if pose.quaternion is not None:
            current, inverse, difference, rotation = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3)
            mujoco.mju_mat2Quat(current, data.site_xmat[site])
            mujoco.mju_negQuat(inverse, current)
            mujoco.mju_mulQuat(difference, np.asarray(pose.quaternion), inverse)
            mujoco.mju_quat2Vel(rotation, difference, 1.0)
            error, jac = np.concatenate([error, 0.3*rotation]), np.vstack([jac, 0.3*jr[:, dofs]])
        increment = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(len(error))*0.004, error)
        target = data.qpos[addresses] + np.clip(increment*0.5, -0.035, 0.035)
        data.ctrl[actuators] = np.clip(target, model.jnt_range[joints, 0], model.jnt_range[joints, 1])

    def _head(self, pose, walking=False):
        # No head command holds the last waist target (the home pose by default).
        if pose is None:
            return
        limits = self.walking_gaze_limits if walking else self.gaze_limits
        if pose.position is not None or pose.quaternion is None:
            raise ValueError("this backend supports orientation-only head (torso gaze) commands")
        data = self.sim.data
        desired = np.zeros(9)
        mujoco.mju_quat2Mat(desired, np.asarray(pose.quaternion))
        relative = data.xmat[self.sim.pelvis_body_id].reshape(3, 3).T @ desired.reshape(3, 3)
        goal = np.array([np.clip(a, *limits[k]) for a, k in zip(waist_angles(relative), ("yaw", "roll", "pitch"))])
        current = data.ctrl[self.waist_actuators]
        data.ctrl[self.waist_actuators] = current+np.clip(goal-current, -GAZE_RATE, GAZE_RATE)

    def _hand(self, side, amount):
        if amount is None:
            return
        model, data = self.sim.model, self.sim.data
        for suffix in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"):
            name = f"{side}_hand_{suffix}_joint"
            joint = self.sim._required_id(mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator = self.sim._required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            lo, hi = model.jnt_range[joint]
            opened = np.clip(0, lo, hi)
            closed = lo if abs(lo) > abs(hi) else hi
            goal = opened+float(amount)*0.65*(closed-opened)
            data.ctrl[actuator] += np.clip(goal-data.ctrl[actuator], -0.04, 0.04)

    def step(self, command):
        if command.waist_height is not None:
            raise ValueError("waist-height commands require SONIC")
        if self.tick % self.config.control_decimation == 0:
            observation = self.walk._build_observation(np.asarray(command.velocity), self.action)
            raw = self.walk.policy_model(observation)
            if not np.isfinite(raw).all():
                raise RuntimeError("non-finite walking action")
            self.action = np.clip(raw, -1, 1)
            self.sim.data.ctrl[:12] = np.array(self.config.default_angles)+self.config.action_scale*self.action
            self._head(command.head, walking=bool(np.any(np.asarray(command.velocity) != 0)))
            self._arm("left", command.left_wrist)
            self._arm("right", command.right_wrist)
            self._hand("left", command.left_hand)
            self._hand("right", command.right_hand)
        self.sim.step(self.sim.data.ctrl.copy())
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.tick += 1
