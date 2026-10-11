from pathlib import Path
import sys

import numpy as np
import pytest

from agent_g1.locomotion import (
    G1PolicyConfig,
    LoadedG1LocomotionPolicy,
    NumPyG1Policy,
    OFFICIAL_G1_POLICY_PATH,
    OFFICIAL_UNITREE_RL_GYM_LICENSE,
)
from agent_g1.simulation import G1Simulation
from agent_g1.arms import (
    DualArmPositionController,
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
)


def test_unitree_policy_artifacts_and_official_constants_are_available():
    assert OFFICIAL_G1_POLICY_PATH.is_file()
    assert OFFICIAL_UNITREE_RL_GYM_LICENSE.is_file()
    assert "BSD 3-Clause License" in OFFICIAL_UNITREE_RL_GYM_LICENSE.read_text()

    config = G1PolicyConfig.create_official()

    assert config.num_obs == 47
    assert config.num_actions == 12
    assert config.simulation_dt == pytest.approx(0.002)
    assert config.control_decimation == 10
    assert config.control_dt == pytest.approx(0.02)
    assert config.default_angles == pytest.approx(
        [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2
    )
    assert config.kps == pytest.approx([100, 100, 100, 150, 40, 40] * 2)
    assert config.kds == pytest.approx([2, 2, 2, 4, 2, 2] * 2)


def test_numpy_policy_loader_runs_without_importing_torch():
    torch_was_loaded = "torch" in sys.modules
    policy = NumPyG1Policy(OFFICIAL_G1_POLICY_PATH)
    observation = np.zeros(47, dtype=np.float32)

    first_action = policy(observation)
    second_action = policy(observation)

    assert first_action.shape == (12,)
    assert second_action.shape == (12,)
    assert np.isfinite(first_action).all()
    assert np.isfinite(second_action).all()
    assert not np.array_equal(first_action, second_action)
    if not torch_was_loaded:
        assert "torch" not in sys.modules


def test_official_policy_physically_walks_to_nearby_base_pose():
    simulation = G1Simulation()
    simulation.reset(seed=7)
    skill = LoadedG1LocomotionPolicy.from_official(
        simulation, load=True, backend="numpy"
    )

    result = skill.move_to(
        target_x=0.04,
        target_y=0.0,
        target_yaw=0.0,
        timeout_s=2.0,
        settle_control_steps=5,
    )

    assert result.success
    assert result.source == "unitree_rl_gym_g1_12dof_torchscript"
    assert result.physics_steps > 10
    assert result.duration_s < 2.0
    assert result.translation_error_m <= 0.02
    assert result.yaw_error_rad <= 0.05
    assert result.collision_free
    assert result.pelvis_stable
    assert result.table_collision_count == 0
    assert result.non_hand_object_collision_count == 0


def test_official_policy_preserves_dual_arm_commands_while_walking():
    simulation = G1Simulation()
    simulation.reset(seed=7)
    controller = DualArmPositionController(simulation)
    left_target = simulation.joint_positions(LEFT_ARM_JOINT_NAMES)
    right_target = simulation.joint_positions(RIGHT_ARM_JOINT_NAMES)
    left_target[0] += 0.1
    right_target[0] += 0.1
    controller.set_joint_targets(left_target, right_target)
    upper_body_commands = simulation.data.ctrl[12:].copy()
    skill = LoadedG1LocomotionPolicy.from_official(
        simulation, load=True, backend="numpy"
    )

    result = skill.move_to(
        target_x=0.04,
        target_y=0.0,
        target_yaw=0.0,
        timeout_s=2.0,
        settle_control_steps=5,
    )

    assert result.success
    np.testing.assert_allclose(
        simulation.data.ctrl[12:], upper_body_commands
    )
