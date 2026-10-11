import numpy as np
import pytest

from agent_g1.arms import (
    DualArmPositionController,
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
)
from agent_g1.simulation import G1Simulation


def test_dual_arm_controller_commands_both_seven_dof_arms():
    simulation = G1Simulation()
    simulation.reset(seed=7)
    controller = DualArmPositionController(simulation)
    left_target = np.asarray([0.35, 0.35, 0.0, 1.0, 0.0, 0.0, 0.0])
    right_target = np.asarray([0.35, -0.35, 0.0, 1.0, 0.0, 0.0, 0.0])

    result = controller.move_to_targets(left_target, right_target)

    assert result.success
    assert result.left_position_error_rad < 0.03
    assert result.right_position_error_rad < 0.03
    assert result.physics_steps > 0
    assert 0.70 < result.pelvis_height < 0.92
    np.testing.assert_allclose(
        simulation.joint_positions(LEFT_ARM_JOINT_NAMES), left_target, atol=0.03
    )
    np.testing.assert_allclose(
        simulation.joint_positions(RIGHT_ARM_JOINT_NAMES), right_target, atol=0.03
    )


def test_dual_arm_control_preserves_leg_commands():
    simulation = G1Simulation()
    simulation.reset(seed=7)
    leg_controls = simulation.data.ctrl[:12].copy()
    controller = DualArmPositionController(simulation)

    left_target = simulation.joint_positions(LEFT_ARM_JOINT_NAMES)
    left_target[0] += 0.1
    right_target = simulation.joint_positions(RIGHT_ARM_JOINT_NAMES)
    right_target[0] += 0.1
    controller.set_joint_targets(left_target, right_target)

    np.testing.assert_allclose(simulation.data.ctrl[:12], leg_controls)
    assert simulation.data.ctrl[controller.left_actuator_ids[0]] == pytest.approx(
        left_target[0]
    )
    assert simulation.data.ctrl[controller.right_actuator_ids[0]] == pytest.approx(
        right_target[0]
    )
