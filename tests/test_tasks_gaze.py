"""Torso gaze, look_at, the task registry and active-perception visibility."""
import json
import numpy as np
import pytest
from hwap.actions import CommandChunk, Keyframe, Pose, look_at, quaternion_multiply, yaw_pitch_quaternion
from hwap.environment import HumanoidEnvironment
from hwap.tasks import SPLITS, TASKS, task_splits


def gaze(yaw, pitch, roll=0.0):
    roll_q = (np.cos(roll/2), np.sin(roll/2), 0, 0)
    q = quaternion_multiply(yaw_pitch_quaternion(yaw, 0), quaternion_multiply(roll_q, yaw_pitch_quaternion(0, pitch)))
    return Pose(None, q)


def look(head, duration=1.0, frame="heading_at_start", velocity=(0, 0, 0)):
    return CommandChunk("look", (Keyframe(0, head=head, velocity=velocity), Keyframe(duration, head=head, velocity=velocity)), frame=frame)


def reach(env, vias=()):
    hand = env.task_spec.hand
    side = hand+"_wrist"
    goal = Pose(tuple(env.goal))
    keys = [Keyframe(0, **{side: Pose(tuple(env.sim.palm_position(hand)))})]
    keys += [Keyframe(0.6*(i+1), **{side: Pose(v)}) for i, v in enumerate(vias)]
    keys += [Keyframe(2.4 if vias else 1.5, **{side: goal}), Keyframe(3.0 if vias else 2.5, **{side: goal})]
    outcome = env.execute(CommandChunk("reach", tuple(keys), frame="world"))
    if not outcome["success"] and not outcome["failure"]:
        outcome = env.execute(CommandChunk("hold", (Keyframe(0, **{side: goal}), Keyframe(0.6, **{side: goal})), frame="world"))
    return outcome


def green_pixels(env):
    rgb = env.render()["ego"].astype(int)
    return int(np.sum((rgb[..., 1] > 120) & (rgb[..., 0] < 90) & (rgb[..., 2] < 110)))


def test_orientation_only_head_schema():
    q = yaw_pitch_quaternion(0.3, 0.1)
    chunk = CommandChunk.parse({"name": "g", "frame": "heading_at_start", "keyframes": [
        {"t": 0, "head": {"position": None, "quaternion": [1, 0, 0, 0]}}, {"t": 1, "head": {"position": None, "quaternion": list(q)}}]})
    middle = chunk.sample(0.5).head
    assert middle.position is None and np.isclose(np.linalg.norm(middle.quaternion), 1)
    with pytest.raises(ValueError, match="orientation-only"):
        Pose(None)
    with pytest.raises(ValueError, match="wrist poses require"):
        Keyframe(0, right_wrist=Pose(None, q))
    with pytest.raises(ValueError, match="all be orientation-only"):
        CommandChunk("mixed", (Keyframe(0, head=Pose(None, q)), Keyframe(1, head=Pose((0, 0, 1.2), q))))


def test_unitree_gaze_capability_and_rejected_head_position():
    with HumanoidEnvironment(budget_s=5) as env:
        env.reset(0)
        observation = env.observe(render=False)
        g = observation["capabilities"]["gaze"]
        assert g["yaw"] == [-0.6, 0.6] and g["pitch"] == [-0.2, 0.3] and g["roll"] == [-0.08, 0.08]
        assert g["while_walking"]["yaw"] == [-0.4, 0.4] and not observation["capabilities"]["head_position"]
        assert set(observation["robot"]["head"]) == {"quaternion", "torso_position", "waist"}
        before = env.snapshot()
        with pytest.raises(ValueError, match="orientation-only"):
            env.execute(look(Pose((0, 0, 1.2), (1, 0, 0, 0))))
        np.testing.assert_array_equal(before["physics"], env.snapshot()["physics"])
        pose = Pose(tuple(env.sim.palm_position("right")))
        env.execute(CommandChunk("hold", (Keyframe(0, right_wrist=pose), Keyframe(0.4, right_wrist=pose)), frame="world"))
        assert not env.sim.data.ctrl[env.controller.waist_actuators].any()


def test_standing_gaze_sweep_tracks_and_stays_upright():
    with HumanoidEnvironment(budget_s=10) as env:
        env.reset(0)
        for yaw, pitch, roll in ((0.6, -0.2, 0.08), (-0.6, 0.0, -0.08), (0.0, 0.3, 0.0), (0.0, 0.0, 0.0)):
            outcome = env.execute(look(gaze(yaw, pitch, roll), 1.5))
            assert outcome["failure"] is None and outcome["pelvis_height_m"] > 0.74
            waist = env.head_state()["waist"]
            assert abs(waist["yaw"]-yaw) < 0.12 and abs(waist["pitch"]-pitch) < 0.12
        # Requests beyond the limits are clipped, not rejected.
        env.execute(look(gaze(1.2, 0.0), 1.0))
        assert env.sim.data.ctrl[env.controller.waist_actuators[0]] == pytest.approx(0.6)


def test_walking_gaze_uses_the_tighter_envelope():
    with HumanoidEnvironment(budget_s=10) as env:
        env.reset(0)
        outcome = env.execute(look(gaze(0.6, 0.0), 2.0, velocity=(-0.1, 0, 0)))
        assert outcome["failure"] is None and outcome["pelvis_height_m"] > 0.74
        assert env.sim.data.ctrl[env.controller.waist_actuators[0]] == pytest.approx(0.4)


def test_look_at_points_ego_camera_at_world_target():
    target = (0.50, 0.30, 1.00)
    with HumanoidEnvironment(width=64, height=48, budget_s=5) as env:
        env.reset(1)
        assert np.hypot(*np.subtract(env.target_visibility(point=target)["pixel"], (32, 24))) > 10
        robot = env.observe(render=False)["robot"]
        env.execute(look(Pose(None, look_at(target, robot["head"]["torso_position"])), 1.5, frame="world"))
        u, v = env.target_visibility(point=target)["pixel"]
        assert abs(u-32) < 5 and abs(v-24) < 5


def test_registry_keeps_legacy_layouts_and_hides_them():
    table = task_splits()
    assert set(table["splits"]) == set(SPLITS) and all(table["splits"][s] for s in ("seen", "new_layout", "zero_shot"))
    assert {"reach", "pick_place", "reach_hidden_side", "reach_occluded"} <= set(TASKS)
    rng = np.random.default_rng(7)
    goal = np.array([0.30, -0.25, 1.05]); goal[:2] += rng.uniform(-0.015, 0.015, 2)
    obj = [0.45, 0.0]+rng.uniform(-0.015, 0.015, 2)
    layout = TASKS["reach"].sample_layout(np.random.default_rng(7))
    np.testing.assert_array_equal(layout["goal"], goal)
    np.testing.assert_array_equal(layout["object_xy"], obj)
    with pytest.raises(ValueError, match="unknown task"):
        HumanoidEnvironment("reach_typo")
    with HumanoidEnvironment("reach_occluded", width=32, height=24) as env:
        env.reset(3)
        observation = env.observe(render=False)
        text = json.dumps(observation)
        assert "layout" not in observation and "goal" not in text and "occluder" not in text
        assert not any(repr(float(x)) in text for x in env.goal)


def test_hidden_side_target_needs_gaze_and_is_reachable():
    for task in ("reach_hidden_side", "reach_hidden_side_left"):
        with HumanoidEnvironment(task, width=64, height=48, budget_s=8) as env:
            env.reset(2)
            assert env.task_spec.needs_active_perception
            initial = env.target_visibility()
            assert not initial["in_frustum"] and not env.metrics()["target_visible_ego"]
            state = env.snapshot()
            robot, limits = env.observe(render=False)["robot"], env.capabilities()["gaze"]
            q = look_at(env.goal, robot["head"]["torso_position"], limits["yaw"], limits["pitch"], base_yaw=robot["base_pose"][2])
            env.execute(look(Pose(None, q), frame="world"))
            assert env.target_visibility()["visible"]
            env.restore(state)
            assert reach(env)["success"]


def test_occluded_target_is_blocked_until_the_robot_leans_over_the_post():
    with HumanoidEnvironment("reach_occluded", width=160, height=120, budget_s=8) as env:
        env.reset(4)
        initial = env.target_visibility()
        assert initial["in_frustum"] and initial["occluded"] and initial["blocking_geom"] == "occluder_geom"
        before = green_pixels(env)
        state = env.snapshot()
        env.execute(look(gaze(0.6, 0.0)))
        assert not env.target_visibility()["visible"]  # turning alone does not see past the post
        env.restore(state)
        outcome = env.execute(look(gaze(0.0, 0.3)))
        assert outcome["failure"] is None and env.target_visibility()["visible"]
        assert green_pixels(env) > before+3
        env.restore(state)
        assert reach(env, ((0.12, -0.30, 1.0), (0.25, -0.30, 1.12), (env.goal[0]-0.05, -0.25, env.goal[2]+0.07)))["success"]
