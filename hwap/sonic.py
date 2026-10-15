"""SONIC v1.1 ONNX simulation adapter.

Matches the pinned upstream observation registry and policy_parameters.hpp.
This deterministic Python/CPU adapter is for simulator validation. It does not
replace the official TensorRT/SDK deployment or connect to physical hardware.
The initial implementation uses a stationary lower-body reference; nonzero
velocity and waist-height requests are rejected until planner integration.
"""
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from agent_g1.arms import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
from agent_g1.locomotion import G1PolicyConfig, G1_LEG_JOINT_NAMES
from .control import ChunkController


ISAAC_TO_HARDWARE = np.array([0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28])
HARDWARE_TO_ISAAC = np.argsort(ISAAC_TO_HARDWARE)
LOWER_ISAAC_INDICES = np.array([0,1,3,4,6,7,9,10,13,14,17,18])
DEFAULT_ANGLES = np.array([-.312,0,0,.669,-.363,0]*2+[0,0,0]+[.2,.2,0,.6,0,0,0]+[.2,-.2,0,.6,0,0,0])
BODY_NAMES = G1_LEG_JOINT_NAMES+("waist_yaw_joint","waist_roll_joint","waist_pitch_joint")+LEFT_ARM_JOINT_NAMES+RIGHT_ARM_JOINT_NAMES
ARMATURE = np.array([.025101925,.025101925,.010177520,.025101925,.003609725,.003609725]*2+
    [.010177520,.003609725,.003609725]+[.003609725]*5+[.00425]*2+[.003609725]*5+[.00425]*2)
EFFORT = np.array([139,139,88,139,25,25]*2+[88,25,25]+[25]*5+[5]*2+[25]*5+[5]*2)
BASE_KP = ARMATURE*(20*np.pi)**2
BASE_KD = 4*ARMATURE*(20*np.pi)
GAIN_MULTIPLIER = np.ones(29)
GAIN_MULTIPLIER[[4,5,10,11,13,14]] = 2
GAIN_MULTIPLIER[[4,10]] *= 1.5  # upstream v1.1 ankle-pitch deployment tuning


def quaternion_matrix(quaternion):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix,np.asarray(quaternion,dtype=float))
    return matrix.reshape(3,3)


def relative_quaternion(parent,child):
    inverse, result = np.empty(4),np.empty(4)
    mujoco.mju_negQuat(inverse,np.asarray(parent,dtype=float))
    mujoco.mju_mulQuat(result,inverse,np.asarray(child,dtype=float))
    return result


class SonicOnnxController(ChunkController):
    backend_name = "sonic_v1_1_onnx_standing"
    supports_head = True
    supports_head_position = True
    gaze_limits = None  # head is a SONIC torso reference, not waist-joint gaze
    supports_velocity = False
    supports_waist = False

    def __init__(self, simulation, model_dir):
        import onnxruntime as ort
        self.sim, self.config = simulation,G1PolicyConfig.create_official()
        self.model_dir = Path(model_dir)
        manifest = json.loads((self.model_dir/"manifest.json").read_text())
        for filename in ("model_encoder.onnx","model_decoder.onnx","observation_config.yaml"):
            path = self.model_dir/filename
            wanted = manifest["files"][filename]
            if path.stat().st_size != wanted["size"] or hashlib.sha256(path.read_bytes()).hexdigest()!=wanted["sha256"]:
                raise ValueError("SONIC artifact checksum mismatch: "+filename)
        self.model_revision = manifest["revision"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(str(self.model_dir/"model_encoder.onnx"),options,providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(self.model_dir/"model_decoder.onnx"),options,providers=["CPUExecutionProvider"])
        if self.encoder.get_inputs()[0].shape[-1] != 1751 or self.decoder.get_inputs()[0].shape[-1] != 994:
            raise ValueError("model I/O does not match SONIC v1.1 1751/994 contract")
        self.encoder_input = self.encoder.get_inputs()[0].name
        self.decoder_input = self.decoder.get_inputs()[0].name
        self.joints = np.array([simulation._required_id(mujoco.mjtObj.mjOBJ_JOINT,n) for n in BODY_NAMES])
        self.actuators = np.array([simulation._required_id(mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in BODY_NAMES])
        self.qadr, self.vadr = simulation.model.jnt_qposadr[self.joints],simulation.model.jnt_dofadr[self.joints]
        simulation.model.dof_armature[self.vadr] = ARMATURE
        simulation.model.actuator_gainprm[self.actuators,0] = BASE_KP*GAIN_MULTIPLIER
        simulation.model.actuator_gainprm[self.actuators,1:] = 0
        simulation.model.actuator_biasprm[self.actuators,0] = 0
        simulation.model.actuator_biasprm[self.actuators,1] = -BASE_KP*GAIN_MULTIPLIER
        simulation.model.actuator_biasprm[self.actuators,2] = -BASE_KD*GAIN_MULTIPLIER
        self.tick, self.action = 0,np.zeros(29)
        self.history = deque(maxlen=10)
        self.inference_seconds, self.inference_calls = 0.0,0

    def reset(self):
        d,m = self.sim.data,self.sim.model
        d.qpos[self.qadr] = DEFAULT_ANGLES
        d.qvel[:] = 0
        d.ctrl[self.actuators] = DEFAULT_ANGLES
        mujoco.mj_forward(m,d)
        self.tick, self.action = 0,np.zeros(29)
        self.history.clear()
        self.inference_seconds,self.inference_calls = 0.0,0
        self.reference_positions, self.reference_quaternions = [],[]
        for name,offset in (("left_wrist_yaw_link",[.18,-.025,0]),("right_wrist_yaw_link",[.18,.025,0]),("torso_link",[0,0,.35])):
            body = self.sim._required_id(mujoco.mjtObj.mjOBJ_BODY,name)
            self.reference_positions.append(d.xpos[body]+d.xmat[body].reshape(3,3)@offset)
            self.reference_quaternions.append(d.xquat[body].copy())
        self.reference_positions = np.array(self.reference_positions)
        self.reference_quaternions = np.array(self.reference_quaternions)
        self.reference_heading = self.sim.base_pose[2]
        first = self._history_row()
        self.history.extend([copy.deepcopy(first) for _ in range(10)])

    def snapshot(self):
        return copy.deepcopy((self.tick,self.action,list(self.history),self.inference_seconds,self.inference_calls))

    def restore(self,state):
        self.tick,self.action,rows,self.inference_seconds,self.inference_calls = copy.deepcopy(state)
        self.history = deque(rows,maxlen=10)

    def _history_row(self):
        d = self.sim.data
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.sim.model,d,mujoco.mjtObj.mjOBJ_BODY,self.sim.pelvis_body_id,velocity,1)
        gravity = quaternion_matrix(d.qpos[3:7]).T@np.array([0,0,-1.])
        return {"omega":velocity[:3].copy(),"q":(d.qpos[self.qadr]-DEFAULT_ANGLES)[HARDWARE_TO_ISAAC],
            "dq":d.qvel[self.vadr][HARDWARE_TO_ISAAC].copy(),"action":self.action.copy(),"gravity":gravity}

    def encoder_observation(self,command):
        d = self.sim.data
        features = np.zeros(1751,dtype=np.float32)
        # Upstream encoder_mode_4 is [mode_id,0,0,0], not one-hot.
        features[0] = 1
        yaw_error = self.reference_heading-self.sim.base_pose[2]
        anchor = np.array([np.cos(yaw_error),-np.sin(yaw_error),np.sin(yaw_error),np.cos(yaw_error),0,0])
        features[644:650] = anchor
        reference_isaac = DEFAULT_ANGLES[HARDWARE_TO_ISAAC]
        features[650:770] = np.tile(reference_isaac[LOWER_ISAAC_INDICES],10)
        # The following 120 lower-body velocity values remain zero (standing).
        positions, quaternions = self.reference_positions.copy(),self.reference_quaternions.copy()
        for index,pose,delta in ((0,command.left_wrist,[.08,-.025,0]),(1,command.right_wrist,[.08,.025,0]),(2,command.head,[0,0,0])):
            if pose is not None:
                q = pose.quaternion if pose.quaternion is not None else quaternions[index]
                if pose.position is not None:  # orientation-only head keeps its reference position
                    positions[index] = np.array(pose.position)+quaternion_matrix(q)@delta
                quaternions[index] = q
        R = quaternion_matrix(d.qpos[3:7])
        features[890:899] = ((positions-d.qpos[:3])@R).reshape(-1)
        features[899:911] = np.concatenate([relative_quaternion(d.qpos[3:7],q) for q in quaternions])
        return features

    def step(self,command):
        if np.any(np.asarray(command.velocity)!=0) or command.waist_height is not None:
            raise ValueError("SONIC standing adapter needs planner integration for velocity/waist commands")
        if self.tick%10==0:
            self.history.append(self._history_row())
            start = time.perf_counter()
            features = self.encoder_observation(command)
            token = self.encoder.run(None,{self.encoder_input:features[None]})[0].reshape(-1)
            obs = np.concatenate([token]+[np.concatenate([row[key] for row in self.history]) for key in ("omega","q","dq","action","gravity")]).astype(np.float32)
            if obs.shape!=(994,) or not np.isfinite(obs).all():
                raise RuntimeError("invalid SONIC decoder observation")
            self.action = self.decoder.run(None,{self.decoder_input:obs[None]})[0].reshape(-1).astype(float)
            self.inference_seconds += time.perf_counter()-start
            self.inference_calls += 1
            if self.action.shape!=(29,) or not np.isfinite(self.action).all():
                raise RuntimeError("invalid SONIC action")
            target = DEFAULT_ANGLES+0.25*EFFORT/BASE_KP*self.action[ISAAC_TO_HARDWARE]
            self.sim.data.ctrl[self.actuators] = np.clip(target,self.sim.model.jnt_range[self.joints,0],self.sim.model.jnt_range[self.joints,1])
            self._hand("left",command.left_hand)
            self._hand("right",command.right_hand)
        self.sim.step(self.sim.data.ctrl.copy())
        mujoco.mj_forward(self.sim.model,self.sim.data)
        self.tick += 1
