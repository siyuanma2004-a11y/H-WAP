"""Hardware-bound high-level Unitree locomotion adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .locomotion import LocomotionResult, yaw_from_quaternion


@dataclass(frozen=True)
class UnitreeSafetyLimits:
    max_forward_velocity_m_s: float = 0.15
    max_lateral_velocity_m_s: float = 0.10
    max_yaw_rate_rad_s: float = 0.30
    position_gain: float = 2.0
    yaw_gain: float = 2.0


class UnitreeHighLevelTransport(Protocol):
    def set_velocity(
        self, vx_m_s: float, vy_m_s: float, yaw_rate_rad_s: float
    ) -> None:
        ...

    def base_pose(self) -> tuple[float, float, float]:
        ...


class UnitreeHighLevelLocomotion:
    """Drive a verified high-level Unitree service toward a planar pose."""

    source = "unitree_high_level_service"

    def __init__(
        self,
        transport: UnitreeHighLevelTransport,
        control_period_s: float = 0.1,
        safety_limits: UnitreeSafetyLimits = UnitreeSafetyLimits(),
        success_tolerance_m: float = 0.03,
        success_tolerance_yaw: float = 0.10,
    ) -> None:
        if control_period_s <= 0.0 or not np.isfinite(control_period_s):
            raise ValueError("control_period_s must be positive and finite")
        if success_tolerance_m <= 0.0 or success_tolerance_yaw <= 0.0:
            raise ValueError("success tolerances must be positive")
        limits = np.asarray(
            [
                safety_limits.max_forward_velocity_m_s,
                safety_limits.max_lateral_velocity_m_s,
                safety_limits.max_yaw_rate_rad_s,
                safety_limits.position_gain,
                safety_limits.yaw_gain,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(limits).all() or np.any(limits[:3] <= 0.0):
            raise ValueError("safety limits must be positive and finite")
        self.transport = transport
        self.control_period_s = float(control_period_s)
        self.safety_limits = safety_limits
        self.success_tolerance_m = float(success_tolerance_m)
        self.success_tolerance_yaw = float(success_tolerance_yaw)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def move_to(
        self,
        target_x: float,
        target_y: float,
        target_yaw: float,
        timeout_s: float,
        step_callback: Callable[[str], None] | None = None,
    ) -> LocomotionResult:
        target = np.asarray([target_x, target_y, target_yaw], dtype=np.float64)
        if not np.isfinite(target).all():
            raise ValueError("target_x, target_y, and target_yaw must be finite")
        if timeout_s <= 0.0 or not np.isfinite(timeout_s):
            raise ValueError("timeout_s must be positive and finite")

        command_count = int(
            np.ceil(timeout_s / self.control_period_s - 1.0e-12)
        )
        completed_steps = 0
        try:
            for _ in range(command_count):
                measured = np.asarray(
                    self.transport.base_pose(), dtype=np.float64
                )
                error_x = target_x - float(measured[0])
                error_y = target_y - float(measured[1])
                measured_yaw = (
                    yaw_from_quaternion(measured[3:7])
                    if measured.size >= 4
                    else float(measured[2])
                )
                yaw_error = self._wrap_angle(target_yaw - measured_yaw)
                body_forward = np.cos(measured_yaw) * error_x + np.sin(
                    measured_yaw
                ) * error_y
                body_lateral = -np.sin(measured_yaw) * error_x + np.cos(
                    measured_yaw
                ) * error_y
                limits = self.safety_limits
                vx = float(
                    np.clip(
                        limits.position_gain * body_forward,
                        -limits.max_forward_velocity_m_s,
                        limits.max_forward_velocity_m_s,
                    )
                )
                vy = float(
                    np.clip(
                        limits.position_gain * body_lateral,
                        -limits.max_lateral_velocity_m_s,
                        limits.max_lateral_velocity_m_s,
                    )
                )
                yaw_rate = float(
                    np.clip(
                        limits.yaw_gain * yaw_error,
                        -limits.max_yaw_rate_rad_s,
                        limits.max_yaw_rate_rad_s,
                    )
                )
                self.transport.set_velocity(vx, vy, yaw_rate)
                completed_steps += 1
                if step_callback is not None:
                    step_callback("base_relocate")

                achieved = np.asarray(
                    self.transport.base_pose(), dtype=np.float64
                )
                translation_error = float(
                    np.hypot(achieved[0] - target_x, achieved[1] - target_y)
                )
                achieved_yaw = (
                    yaw_from_quaternion(achieved[3:7])
                    if achieved.size >= 4
                    else float(achieved[2])
                )
                final_yaw_error = abs(
                    self._wrap_angle(achieved_yaw - target_yaw)
                )
                if (
                    translation_error <= self.success_tolerance_m
                    and final_yaw_error <= self.success_tolerance_yaw
                ):
                    break
        finally:
            self.transport.set_velocity(0.0, 0.0, 0.0)

        achieved = np.asarray(self.transport.base_pose(), dtype=np.float64)
        achieved_yaw = (
            yaw_from_quaternion(achieved[3:7])
            if achieved.size >= 4
            else float(achieved[2])
        )
        translation_error = float(
            np.hypot(achieved[0] - target_x, achieved[1] - target_y)
        )
        yaw_error = abs(self._wrap_angle(achieved_yaw - target_yaw))
        success = (
            translation_error <= self.success_tolerance_m
            and yaw_error <= self.success_tolerance_yaw
        )
        return LocomotionResult(
            requested_x=float(target_x),
            requested_y=float(target_y),
            requested_yaw=float(target_yaw),
            achieved_x=float(achieved[0]),
            achieved_y=float(achieved[1]),
            achieved_yaw=float(achieved_yaw),
            translation_error_m=translation_error,
            yaw_error_rad=yaw_error,
            duration_s=float(completed_steps * self.control_period_s),
            physics_steps=completed_steps,
            collision_free=False,
            pelvis_stable=False,
            success=success,
            source=self.source,
        )
