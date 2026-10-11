"""Unitree official 12-DoF G1 locomotion policy adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence
import zipfile

import mujoco
import numpy as np

from .simulation import G1Simulation, yaw_from_quaternion


G1_LEG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_UNITREE_RL_GYM_ROOT = REPO_ROOT / "models" / "unitree_rl_gym"
OFFICIAL_G1_POLICY_PATH = OFFICIAL_UNITREE_RL_GYM_ROOT / "motion.pt"
OFFICIAL_UNITREE_RL_GYM_LICENSE = OFFICIAL_UNITREE_RL_GYM_ROOT / "LICENSE"


@dataclass(frozen=True)
class LocomotionResult:
    """Measured outcome of one planar base relocation."""

    requested_x: float
    requested_y: float
    requested_yaw: float
    achieved_x: float
    achieved_y: float
    achieved_yaw: float
    translation_error_m: float
    yaw_error_rad: float
    duration_s: float
    physics_steps: int
    collision_free: bool
    pelvis_stable: bool
    success: bool
    source: str
    table_collision_count: int = 0
    non_hand_object_collision_count: int = 0
    minimum_pelvis_height_m: float = float("nan")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "translation_error_m",
            float(
                np.hypot(
                    self.achieved_x - self.requested_x,
                    self.achieved_y - self.requested_y,
                )
            ),
        )
        object.__setattr__(
            self,
            "yaw_error_rad",
            abs(
                (self.achieved_yaw - self.requested_yaw + np.pi)
                % (2.0 * np.pi)
                - np.pi
            ),
        )

    @property
    def requested_pose(self) -> np.ndarray:
        return np.asarray(
            [self.requested_x, self.requested_y, self.requested_yaw]
        )

    @property
    def achieved_pose(self) -> np.ndarray:
        return np.asarray(
            [self.achieved_x, self.achieved_y, self.achieved_yaw]
        )


@dataclass(frozen=True)
class G1PolicyConfig:
    """Deployment constants from Unitree's official G1 config."""

    num_obs: int
    num_actions: int
    simulation_dt: float
    control_decimation: int
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    action_scale: float
    cmd_scale: tuple[float, float, float]
    default_angles: tuple[float, ...]
    kps: tuple[float, ...]
    kds: tuple[float, ...]
    gait_period_s: float = 0.8

    @property
    def control_dt(self) -> float:
        return self.simulation_dt * self.control_decimation

    @classmethod
    def create_official(cls) -> "G1PolicyConfig":
        return cls(
            num_obs=47,
            num_actions=12,
            simulation_dt=0.002,
            control_decimation=10,
            ang_vel_scale=0.25,
            dof_pos_scale=1.0,
            dof_vel_scale=0.05,
            action_scale=0.25,
            cmd_scale=(2.0, 2.0, 0.25),
            default_angles=(
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
            ),
            kps=(
                100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
                100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
            ),
            kds=(2.0, 2.0, 2.0, 4.0, 2.0, 2.0) * 2,
        )


class G1PolicyModel(Protocol):
    def __call__(self, observation: Sequence[float]) -> Sequence[float]:
        ...


class _TorchScriptG1Policy:
    def __init__(self, module) -> None:
        self.module = module

    def __call__(self, observation: Sequence[float]) -> np.ndarray:
        import torch

        tensor = torch.from_numpy(
            np.asarray(observation, dtype=np.float32)
        ).unsqueeze(0)
        with torch.no_grad():
            output = self.module(tensor)
        return output.detach().numpy().squeeze(0)


class NumPyG1Policy:
    """NumPy inference for Unitree's exported LSTM actor."""

    _STORAGE_SHAPES = {
        "2": (32, 64),
        "3": (32,),
        "4": (12, 32),
        "5": (12,),
        "6": (256, 47),
        "7": (256, 64),
        "8": (256,),
        "9": (256,),
    }

    def __init__(self, archive_path: Path) -> None:
        self.archive_path = Path(archive_path)
        self.parameters: dict[str, np.ndarray] = {}
        with zipfile.ZipFile(self.archive_path, "r") as archive:
            names = set(archive.namelist())
            required = {
                f"policy_lstm_1/data/{key}" for key in self._STORAGE_SHAPES
            }
            if not required.issubset(names):
                raise ValueError(
                    "Unitree G1 policy archive is missing required storages"
                )
            for storage_id, shape in self._STORAGE_SHAPES.items():
                raw = archive.read(f"policy_lstm_1/data/{storage_id}")
                expected_bytes = int(np.prod(shape)) * np.dtype("<f4").itemsize
                if len(raw) != expected_bytes:
                    raise ValueError(
                        "Unitree G1 policy storage has an unexpected size: "
                        f"{storage_id}"
                    )
                self.parameters[storage_id] = (
                    np.frombuffer(raw, dtype="<f4")
                    .reshape(shape)
                    .astype(np.float64)
                )
        self.hidden_state = np.zeros((1, 1, 64), dtype=np.float64)
        self.cell_state = np.zeros((1, 1, 64), dtype=np.float64)

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-value))

    def reset_memory(self) -> None:
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def __call__(self, observation: Sequence[float]) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float64).reshape(1, 1, -1)
        if x.shape != (1, 1, 47):
            raise ValueError("Unitree G1 policy observation must have 47 values")
        gates = (
            np.einsum("i,ji->j", x[0, 0], self.parameters["6"])
            + np.einsum(
                "i,ji->j", self.hidden_state[0, 0], self.parameters["7"]
            )
            + self.parameters["8"]
            + self.parameters["9"]
        )
        input_gate, forget_gate, candidate, output_gate = np.split(
            gates, 4, axis=-1
        )
        input_gate = self._sigmoid(input_gate)
        forget_gate = self._sigmoid(forget_gate)
        candidate = np.tanh(candidate)
        output_gate = self._sigmoid(output_gate)
        next_cell = forget_gate * self.cell_state[0, 0] + input_gate * candidate
        next_hidden = output_gate * np.tanh(next_cell)
        self.cell_state[0, 0] = next_cell
        self.hidden_state[0, 0] = next_hidden

        hidden = (
            np.einsum("i,ji->j", next_hidden, self.parameters["2"])
            + self.parameters["3"]
        )
        activated = np.where(
            hidden > 0.0,
            hidden,
            np.expm1(np.minimum(hidden, 0.0)),
        ).astype(np.float32)
        action = (
            np.einsum("i,ji->j", activated, self.parameters["4"])
            + self.parameters["5"]
        )
        return action.astype(np.float32)


class LoadedG1LocomotionPolicy:
    """Pose servo around Unitree's official G1 walking actor."""

    source = "unitree_rl_gym_g1_12dof_torchscript"

    def __init__(
        self,
        simulation: G1Simulation,
        policy_model: G1PolicyModel | None,
        policy_path: Path = OFFICIAL_G1_POLICY_PATH,
        config: G1PolicyConfig | None = None,
        success_tolerance_m: float = 0.02,
        success_tolerance_yaw: float = 0.05,
        max_forward_velocity_m_s: float = 0.15,
        max_lateral_velocity_m_s: float = 0.10,
        max_yaw_rate_rad_s: float = 0.30,
        command_gain: float = 1.5,
        yaw_command_gain: float = 1.5,
    ) -> None:
        self.simulation = simulation
        self.policy_model = policy_model
        self.policy_path = Path(policy_path)
        self.config = config or G1PolicyConfig.create_official()
        self.success_tolerance_m = float(success_tolerance_m)
        self.success_tolerance_yaw = float(success_tolerance_yaw)
        self.max_forward_velocity_m_s = float(max_forward_velocity_m_s)
        self.max_lateral_velocity_m_s = float(max_lateral_velocity_m_s)
        self.max_yaw_rate_rad_s = float(max_yaw_rate_rad_s)
        self.command_gain = float(command_gain)
        self.yaw_command_gain = float(yaw_command_gain)
        self._validate_model_mapping()

    @classmethod
    def from_official(
        cls,
        simulation: G1Simulation,
        load: bool = True,
        backend: str = "numpy",
    ) -> "LoadedG1LocomotionPolicy":
        if not OFFICIAL_G1_POLICY_PATH.is_file():
            raise FileNotFoundError(
                f"Unitree G1 policy is missing: {OFFICIAL_G1_POLICY_PATH}"
            )
        policy_model = None
        if load:
            if backend == "numpy":
                policy_model = NumPyG1Policy(OFFICIAL_G1_POLICY_PATH)
            elif backend == "torch":
                try:
                    import torch
                except ImportError as error:
                    raise RuntimeError(
                        "PyTorch is required for the TorchScript backend"
                    ) from error
                module = torch.jit.load(
                    str(OFFICIAL_G1_POLICY_PATH), map_location="cpu"
                )
                module.eval()
                policy_model = _TorchScriptG1Policy(module)
            else:
                raise ValueError("backend must be 'numpy' or 'torch'")
        return cls(simulation, policy_model, OFFICIAL_G1_POLICY_PATH)

    def _validate_model_mapping(self) -> None:
        model = self.simulation.model
        if model.nq < 19 or model.nv < 18 or model.nu < 12:
            raise ValueError("G1 model must expose at least twelve leg DoFs")
        for expected_index, name in enumerate(G1_LEG_JOINT_NAMES, start=1):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            if joint_id != expected_index or actuator_id != expected_index - 1:
                raise ValueError(
                    "Unitree G1 policy requires official leg joint ordering; "
                    f"mismatch at {name}"
                )
            if (
                model.jnt_qposadr[joint_id] != 6 + expected_index
                or model.jnt_dofadr[joint_id] != 5 + expected_index
            ):
                raise ValueError(
                    f"G1 leg joint {name} has non-contiguous state address"
                )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = quaternion
        return np.asarray(
            [
                2.0 * (-qz * qx + qw * qy),
                -2.0 * (qz * qy + qw * qx),
                1.0 - 2.0 * (qw * qw + qz * qz),
            ],
            dtype=np.float64,
        )

    def _velocity_command(
        self,
        achieved_x: float,
        achieved_y: float,
        achieved_yaw: float,
        target_x: float,
        target_y: float,
        target_yaw: float,
    ) -> np.ndarray:
        error_x = target_x - achieved_x
        error_y = target_y - achieved_y
        yaw_error = self._wrap_angle(target_yaw - achieved_yaw)
        body_forward = np.cos(achieved_yaw) * error_x + np.sin(
            achieved_yaw
        ) * error_y
        body_lateral = -np.sin(achieved_yaw) * error_x + np.cos(
            achieved_yaw
        ) * error_y
        return np.asarray(
            [
                np.clip(
                    self.command_gain * body_forward,
                    -self.max_forward_velocity_m_s,
                    self.max_forward_velocity_m_s,
                ),
                np.clip(
                    self.command_gain * body_lateral,
                    -self.max_lateral_velocity_m_s,
                    self.max_lateral_velocity_m_s,
                ),
                np.clip(
                    self.yaw_command_gain * yaw_error,
                    -self.max_yaw_rate_rad_s,
                    self.max_yaw_rate_rad_s,
                ),
            ],
            dtype=np.float64,
        )

    def _build_observation(
        self, command: np.ndarray, previous_action: np.ndarray
    ) -> np.ndarray:
        config = self.config
        qpos = self.simulation.data.qpos[7 : 7 + config.num_actions].copy()
        qvel = self.simulation.data.qvel[6 : 6 + config.num_actions].copy()
        omega = self.simulation.data.qvel[3:6].copy()
        gravity = self._gravity_orientation(self.simulation.data.qpos[3:7])
        phase = (
            self.simulation.data.time % config.gait_period_s
        ) / config.gait_period_s
        observation = np.zeros(config.num_obs, dtype=np.float64)
        observation[:3] = omega * config.ang_vel_scale
        observation[3:6] = gravity
        observation[6:9] = command * np.asarray(config.cmd_scale)
        observation[9 : 9 + config.num_actions] = (
            qpos - np.asarray(config.default_angles)
        ) * config.dof_pos_scale
        observation[
            9 + config.num_actions : 9 + 2 * config.num_actions
        ] = qvel * config.dof_vel_scale
        observation[
            9 + 2 * config.num_actions : 9 + 3 * config.num_actions
        ] = previous_action
        observation[-2:] = [
            np.sin(2.0 * np.pi * phase),
            np.cos(2.0 * np.pi * phase),
        ]
        return observation

    def move_to(
        self,
        target_x: float,
        target_y: float,
        target_yaw: float,
        timeout_s: float,
        step_callback: Callable[[str], None] | None = None,
        settle_control_steps: int = 10,
    ) -> LocomotionResult:
        """Walk to a planar pose using official policy velocity commands."""
        target = np.asarray([target_x, target_y, target_yaw], dtype=np.float64)
        if not np.isfinite(target).all():
            raise ValueError("target_x, target_y, and target_yaw must be finite")
        if timeout_s <= 0.0 or not np.isfinite(timeout_s):
            raise ValueError("timeout_s must be positive and finite")
        if settle_control_steps < 1:
            raise ValueError("settle_control_steps must be at least one")
        if self.policy_model is None:
            raise RuntimeError("a loaded policy model is required")

        config = self.config
        maximum_steps = int(
            np.ceil(timeout_s / config.simulation_dt - 1.0e-12)
        )
        original_gain = self.simulation.model.actuator_gainprm[:12].copy()
        original_bias = self.simulation.model.actuator_biasprm[:12].copy()
        original_upper_ctrl = self.simulation.data.ctrl[12:].copy()
        action = np.zeros(config.num_actions, dtype=np.float64)
        completed_steps = 0
        table_collision_count = 0
        non_hand_object_collision_count = 0
        minimum_pelvis_height = float("inf")
        settled_updates = 0

        try:
            self.simulation.model.actuator_gainprm[:12, 0] = config.kps
            self.simulation.model.actuator_gainprm[:12, 1:] = 0.0
            self.simulation.model.actuator_biasprm[:12, 0] = 0.0
            self.simulation.model.actuator_biasprm[:12, 1] = -np.asarray(
                config.kps
            )
            self.simulation.model.actuator_biasprm[:12, 2] = -np.asarray(
                config.kds
            )
            self.simulation.data.ctrl[:12] = config.default_angles

            for _ in range(maximum_steps):
                if completed_steps % config.control_decimation == 0:
                    achieved_x, achieved_y, achieved_yaw = (
                        self.simulation.base_pose
                    )
                    translation_error = float(
                        np.hypot(
                            achieved_x - target_x, achieved_y - target_y
                        )
                    )
                    yaw_error = abs(
                        self._wrap_angle(achieved_yaw - target_yaw)
                    )
                    within_tolerance = (
                        translation_error <= self.success_tolerance_m
                        and yaw_error <= self.success_tolerance_yaw
                    )
                    if within_tolerance:
                        command = np.zeros(3, dtype=np.float64)
                        settled_updates += 1
                    else:
                        command = self._velocity_command(
                            achieved_x,
                            achieved_y,
                            achieved_yaw,
                            target_x,
                            target_y,
                            target_yaw,
                        )
                        settled_updates = 0
                    observation = self._build_observation(command, action)
                    raw_action = np.asarray(
                        self.policy_model(observation), dtype=np.float64
                    ).reshape(-1)
                    if raw_action.shape != (config.num_actions,) or not (
                        np.isfinite(raw_action).all()
                    ):
                        raise ValueError(
                            "Unitree G1 policy must return twelve finite actions"
                        )
                    action = np.clip(raw_action, -1.0, 1.0)
                    self.simulation.data.ctrl[:12] = (
                        np.asarray(config.default_angles)
                        + config.action_scale * action
                    )

                self.simulation.data.qfrc_applied[:] = 0.0
                self.simulation.step(self.simulation.data.ctrl.copy())
                completed_steps += 1
                minimum_pelvis_height = min(
                    minimum_pelvis_height, self.simulation.pelvis_height
                )
                table_contacts = self.simulation.table_collision_contacts()
                non_hand_contacts = (
                    self.simulation.non_hand_object_collision_contacts()
                )
                table_collision_count += len(table_contacts)
                non_hand_object_collision_count += len(non_hand_contacts)
                if step_callback is not None:
                    step_callback("base_relocate")
                if table_contacts or non_hand_contacts:
                    break
                if (
                    settled_updates >= settle_control_steps
                    and completed_steps % config.control_decimation == 0
                ):
                    break
        finally:
            self.simulation.model.actuator_gainprm[:12] = original_gain
            self.simulation.model.actuator_biasprm[:12] = original_bias
            self.simulation.data.ctrl[:12] = config.default_angles
            self.simulation.data.ctrl[12:] = original_upper_ctrl
            self.simulation.data.qfrc_applied[:] = 0.0

        achieved_x, achieved_y, achieved_yaw = self.simulation.base_pose
        translation_error = float(
            np.hypot(achieved_x - target_x, achieved_y - target_y)
        )
        yaw_error = abs(self._wrap_angle(achieved_yaw - target_yaw))
        final_pelvis_height = self.simulation.pelvis_height
        pelvis_stable = (
            0.70 < minimum_pelvis_height < 0.92
            and 0.70 < final_pelvis_height < 0.92
        )
        collision_free = (
            table_collision_count == 0
            and non_hand_object_collision_count == 0
        )
        success = (
            collision_free
            and pelvis_stable
            and translation_error <= self.success_tolerance_m
            and yaw_error <= self.success_tolerance_yaw
        )
        return LocomotionResult(
            requested_x=float(target_x),
            requested_y=float(target_y),
            requested_yaw=float(target_yaw),
            achieved_x=achieved_x,
            achieved_y=achieved_y,
            achieved_yaw=achieved_yaw,
            translation_error_m=translation_error,
            yaw_error_rad=yaw_error,
            duration_s=float(completed_steps * config.simulation_dt),
            physics_steps=completed_steps,
            collision_free=collision_free,
            pelvis_stable=pelvis_stable,
            success=success,
            source=self.source,
            table_collision_count=table_collision_count,
            non_hand_object_collision_count=non_hand_object_collision_count,
            minimum_pelvis_height_m=minimum_pelvis_height,
        )
