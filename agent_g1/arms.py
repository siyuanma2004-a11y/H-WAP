"""Symmetric position control for both seven-DoF G1 arms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np

from .simulation import G1Simulation


LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class DualArmControlResult:
    left_target_qpos: np.ndarray
    right_target_qpos: np.ndarray
    left_achieved_qpos: np.ndarray
    right_achieved_qpos: np.ndarray
    left_position_error_rad: float
    right_position_error_rad: float
    pelvis_height: float
    physics_steps: int
    success: bool


class DualArmPositionController:
    """Command both G1 arms without touching legs, waist, or hands."""

    def __init__(self, simulation: G1Simulation) -> None:
        self.simulation = simulation
        self.left_joint_ids = self._joint_ids(LEFT_ARM_JOINT_NAMES)
        self.right_joint_ids = self._joint_ids(RIGHT_ARM_JOINT_NAMES)
        self.left_actuator_ids = self._actuator_ids(LEFT_ARM_JOINT_NAMES)
        self.right_actuator_ids = self._actuator_ids(RIGHT_ARM_JOINT_NAMES)

    def _joint_ids(self, names: Sequence[str]) -> tuple[int, ...]:
        ids = tuple(
            mujoco.mj_name2id(
                self.simulation.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            for name in names
        )
        if -1 in ids:
            raise ValueError("G1 model is missing a named arm joint")
        return ids

    def _actuator_ids(self, names: Sequence[str]) -> tuple[int, ...]:
        ids = tuple(
            mujoco.mj_name2id(
                self.simulation.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            for name in names
        )
        if -1 in ids:
            raise ValueError("G1 model is missing a named arm actuator")
        return ids

    def _validate_target(
        self, values: Sequence[float], joint_ids: Sequence[int], side: str
    ) -> np.ndarray:
        target = np.asarray(values, dtype=np.float64)
        if target.shape != (len(joint_ids),):
            raise ValueError(
                f"{side} arm target must contain {len(joint_ids)} values"
            )
        if not np.isfinite(target).all():
            raise ValueError(f"{side} arm target must be finite")
        lower = self.simulation.model.jnt_range[joint_ids, 0]
        upper = self.simulation.model.jnt_range[joint_ids, 1]
        return np.clip(target, lower, upper)

    def set_joint_targets(
        self,
        left_target: Sequence[float],
        right_target: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Write position commands for all fourteen arm joints."""
        left = self._validate_target(
            left_target, self.left_joint_ids, "left"
        )
        right = self._validate_target(
            right_target, self.right_joint_ids, "right"
        )
        for actuator_id, value in zip(self.left_actuator_ids, left):
            self.simulation.data.ctrl[actuator_id] = value
        for actuator_id, value in zip(self.right_actuator_ids, right):
            self.simulation.data.ctrl[actuator_id] = value
        return left, right

    def move_to_targets(
        self,
        left_target: Sequence[float],
        right_target: Sequence[float],
        max_steps: int = 800,
        settle_steps: int = 300,
        tolerance_rad: float = 0.025,
        consecutive_success_steps: int = 25,
    ) -> DualArmControlResult:
        """Ramp both arm targets and step MuJoCo until they settle."""
        if max_steps <= settle_steps:
            raise ValueError("max_steps must exceed settle_steps")
        if tolerance_rad <= 0.0 or consecutive_success_steps < 1:
            raise ValueError("tolerance and consecutive steps must be positive")

        left_goal = self._validate_target(
            left_target, self.left_joint_ids, "left"
        )
        right_goal = self._validate_target(
            right_target, self.right_joint_ids, "right"
        )
        left_start = self.simulation.joint_positions(LEFT_ARM_JOINT_NAMES)
        right_start = self.simulation.joint_positions(RIGHT_ARM_JOINT_NAMES)
        ramp_steps = max_steps - settle_steps
        completed_steps = 0
        success_count = 0

        for step_index in range(max_steps):
            alpha = min(1.0, step_index / ramp_steps)
            self.set_joint_targets(
                left_start + alpha * (left_goal - left_start),
                right_start + alpha * (right_goal - right_start),
            )
            self.simulation.step(self.simulation.data.ctrl.copy())
            completed_steps += 1
            left_error = float(
                np.max(
                    np.abs(
                        self.simulation.joint_positions(
                            LEFT_ARM_JOINT_NAMES
                        )
                        - left_goal
                    )
                )
            )
            right_error = float(
                np.max(
                    np.abs(
                        self.simulation.joint_positions(
                            RIGHT_ARM_JOINT_NAMES
                        )
                        - right_goal
                    )
                )
            )
            if left_error <= tolerance_rad and right_error <= tolerance_rad:
                success_count += 1
            else:
                success_count = 0
            if (
                step_index >= ramp_steps
                and success_count >= consecutive_success_steps
            ):
                break

        left_achieved = self.simulation.joint_positions(LEFT_ARM_JOINT_NAMES)
        right_achieved = self.simulation.joint_positions(RIGHT_ARM_JOINT_NAMES)
        left_error = float(np.max(np.abs(left_achieved - left_goal)))
        right_error = float(np.max(np.abs(right_achieved - right_goal)))
        return DualArmControlResult(
            left_target_qpos=left_goal,
            right_target_qpos=right_goal,
            left_achieved_qpos=left_achieved,
            right_achieved_qpos=right_achieved,
            left_position_error_rad=left_error,
            right_position_error_rad=right_error,
            pelvis_height=self.simulation.pelvis_height,
            physics_steps=completed_steps,
            success=(
                left_error <= tolerance_rad
                and right_error <= tolerance_rad
                and 0.70 < self.simulation.pelvis_height < 0.92
            ),
        )
