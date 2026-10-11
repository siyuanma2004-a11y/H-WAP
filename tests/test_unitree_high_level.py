import pytest

from agent_g1.unitree_high_level import (
    UnitreeHighLevelLocomotion,
    UnitreeSafetyLimits,
)


class FakeUnitreeTransport:
    def __init__(self):
        self.pose = (0.0, 0.0, 0.0)
        self.commands = []
        self.fail_after_commands = None

    def set_velocity(self, vx_m_s, vy_m_s, yaw_rate_rad_s):
        self.commands.append((vx_m_s, vy_m_s, yaw_rate_rad_s))
        if (
            self.fail_after_commands is not None
            and len(self.commands) >= self.fail_after_commands
        ):
            raise RuntimeError("DDS connection lost")
        x, y, yaw = self.pose
        self.pose = (
            x + 0.1 * vx_m_s,
            y + 0.1 * vy_m_s,
            yaw + 0.1 * yaw_rate_rad_s,
        )

    def base_pose(self):
        return self.pose


def test_unitree_adapter_converges_with_bounded_commands_and_stops():
    transport = FakeUnitreeTransport()
    locomotion = UnitreeHighLevelLocomotion(transport, control_period_s=0.1)

    result = locomotion.move_to(0.24, 0.16, 0.12, timeout_s=3.0)

    assert result.success
    assert result.source == "unitree_high_level_service"
    assert result.achieved_pose == pytest.approx([0.24, 0.16, 0.12], abs=0.03)
    assert result.translation_error_m <= 0.03
    assert result.yaw_error_rad <= 0.10
    assert transport.commands[-1] == (0.0, 0.0, 0.0)
    assert all(
        abs(command[0]) <= 0.15 + 1e-12
        and abs(command[1]) <= 0.10 + 1e-12
        and abs(command[2]) <= 0.30 + 1e-12
        for command in transport.commands[:-1]
    )


def test_unitree_adapter_reports_timeout_and_stops():
    transport = FakeUnitreeTransport()
    locomotion = UnitreeHighLevelLocomotion(transport, control_period_s=0.1)

    result = locomotion.move_to(0.50, 0.0, 0.0, timeout_s=0.4)

    assert not result.success
    assert result.physics_steps == 4
    assert result.duration_s == pytest.approx(0.4)
    assert result.translation_error_m > 0.03
    assert transport.commands[-1] == (0.0, 0.0, 0.0)


def test_unitree_adapter_stops_when_transport_raises():
    transport = FakeUnitreeTransport()
    transport.fail_after_commands = 1
    locomotion = UnitreeHighLevelLocomotion(transport, control_period_s=0.1)

    with pytest.raises(RuntimeError, match="DDS connection lost"):
        locomotion.move_to(0.2, 0.0, 0.0, timeout_s=1.0)

    assert transport.commands == [
        (0.15, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]


def test_unitree_adapter_validates_inputs_and_limits():
    transport = FakeUnitreeTransport()

    with pytest.raises(ValueError, match="control_period_s must be positive"):
        UnitreeHighLevelLocomotion(transport, control_period_s=0.0)
    with pytest.raises(ValueError, match="safety limits must be positive"):
        UnitreeHighLevelLocomotion(
            transport,
            safety_limits=UnitreeSafetyLimits(max_forward_velocity_m_s=0.0),
        )

    locomotion = UnitreeHighLevelLocomotion(transport)
    with pytest.raises(ValueError, match="finite"):
        locomotion.move_to(float("nan"), 0.0, 0.0, timeout_s=1.0)
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        locomotion.move_to(0.0, 0.0, 0.0, timeout_s=0.0)
