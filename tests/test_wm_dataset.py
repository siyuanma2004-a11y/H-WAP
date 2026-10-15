"""World-model clip export: action vectors, pose images, timing and splits."""
import json
import numpy as np

from hwap.actions import CommandChunk, Keyframe, Pose
from hwap.conditioning import (ACTION_DIM, COLORS, SLICES, active_chunk, command_pose_image, commanded_vectors,
                               integrate_base_path, quaternion_multiply, yaw_quaternion)
from hwap.dataset import export_dataset, export_world_model_dataset
from hwap.environment import HumanoidEnvironment
from hwap.recording import EpisodeRecorder


def test_commanded_vectors_convert_heading_frame_and_mask_unset_fields():
    q = tuple(np.array([0.9, 0.1, 0.2, 0.3])/np.linalg.norm([0.9, 0.1, 0.2, 0.3]))
    first = CommandChunk("a", (Keyframe(0, right_wrist=Pose((0.3, -0.2, 1.0), q), right_hand=0.0, velocity=(0.1, 0, 0.2)),
                               Keyframe(0.1, right_wrist=Pose((0.4, -0.1, 1.1), q), right_hand=1.0, velocity=(0.1, 0, 0.2))))
    second = CommandChunk("b", (Keyframe(0, left_wrist=Pose((0.2, 0.2, 0.9))), Keyframe(0.1, left_wrist=Pose((0.2, 0.3, 0.9)))),
                          frame="world")
    record = {"chunks": [{"chunk": first.to_dict(), "start_row": 0, "end_row": 5},
                         {"chunk": second.to_dict(), "start_row": 5, "end_row": 10}]}
    time = np.arange(11)*0.02
    base = np.tile([1.0, 2.0, 0.7], (11, 1))
    base[5] = [9, 9, 3]  # the second chunk is world-frame; its anchor must be ignored
    vectors, mask = commanded_vectors(record, {"time": time, "base_pose": base}, [0, 3, 5, 6, 10])
    assert active_chunk(record["chunks"], 5)["start_row"] == 0  # shared row = finished chunk
    c, s = np.cos(0.7), np.sin(0.7)
    for i, row in enumerate([0, 3, 5]):
        key = first.sample(time[row])
        expected = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ key.right_wrist.position + [1, 2, 0]
        np.testing.assert_allclose(vectors[i, SLICES["right_palm_position"]], expected, atol=1e-12)
        np.testing.assert_allclose(vectors[i, SLICES["right_palm_quaternion"]],
                                   quaternion_multiply(yaw_quaternion(0.7), key.right_wrist.quaternion), atol=1e-12)
        assert vectors[i, SLICES["right_hand"]][0] == key.right_hand
        assert not mask[i, SLICES["left_palm_position"]].any() and not mask[i, SLICES["head_quaternion"]].any()
        assert mask[i, SLICES["base_velocity"]].all() and not mask[i, SLICES["waist_height"]].any()
    np.testing.assert_allclose(vectors[3, SLICES["left_palm_position"]], second.sample(0.02).left_wrist.position)
    np.testing.assert_allclose(vectors[4, SLICES["left_palm_position"]], [0.2, 0.3, 0.9])
    assert not mask[3, SLICES["right_palm_position"]].any() and vectors.shape[1] == ACTION_DIM


def test_orientation_only_head_is_masked_as_quaternion_only():
    gaze = (np.cos(0.2), 0, 0, np.sin(0.2))
    chunk = CommandChunk("look", (Keyframe(0, head=Pose(None, gaze)), Keyframe(0.1, head=Pose(None, gaze))))
    record = {"chunks": [{"chunk": chunk.to_dict(), "start_row": 0, "end_row": 5}]}
    vectors, mask = commanded_vectors(record, {"time": np.arange(6)*0.02, "base_pose": np.tile([0, 0, 0.5], (6, 1))}, [2])
    assert mask[0, SLICES["head_quaternion"]].all() and not mask[0, SLICES["head_position"]].any()
    np.testing.assert_allclose(vectors[0, SLICES["head_quaternion"]], quaternion_multiply(yaw_quaternion(0.5), gaze))


def test_command_pose_image_projects_known_point():
    camera = {"position": [0, 0, 0], "rotation_world_from_camera": np.eye(3).tolist(),
              "intrinsics": [[50, 0, 32], [0, 50, 24], [0, 0, 1]], "width": 64, "height": 48}
    image = command_pose_image({"right_palm": [0.1, 0.04, -1.0]}, camera)
    assert tuple(image[22, 37]) == COLORS["right"] and image[:, :30].sum() == 0
    assert command_pose_image({"right_palm": [0, 0, 1.0]}, camera).sum() == 0  # behind the camera


def test_integrated_base_path_follows_heading_frame_velocity():
    path = integrate_base_path([0, 0, np.pi/2], [0, 0.5, 1.0], [[0.1, 0, 0], [0.1, 0, 0], [0, 0, 0]])
    np.testing.assert_allclose(path[-1], [0, 0.1, np.pi/2], atol=1e-9)


def record_episode(runs, seed):
    with HumanoidEnvironment(width=64, height=48) as env:
        env.reset(seed)
        recorder = EpisodeRecorder(env, runs/f"reach_seed{seed}")
        start = env.sim.palm_position("right")
        recorder.execute(CommandChunk("lift", (Keyframe(0, right_wrist=Pose(tuple(start)), right_hand=0.0),
            Keyframe(0.6, right_wrist=Pose(tuple(start+[0, 0, 0.03])), right_hand=0.5)), frame="world"))
        x, y, yaw = env.sim.base_pose
        local = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]) @ (env.sim.palm_position("right")-[x, y, 0])
        recorder.execute(CommandChunk("forward", (Keyframe(0, right_wrist=Pose(tuple(local))),
            Keyframe(0.6, right_wrist=Pose(tuple(local+[0.02, 0, 0]))))))
        recorder.finish()


def test_world_model_export_clips_actions_and_splits(tmp_path):
    runs = tmp_path/"runs"
    for seed in (3, 5):
        record_episode(runs, seed)
    result = export_world_model_dataset(runs, tmp_path/"wm", clip_frames=4, stride=2,
                                        splits={"reach:3": "heldout_layout", "default": "train"})
    manifest = json.loads((tmp_path/"wm/manifest.json").read_text())
    clips = [json.loads(line) for line in (tmp_path/"wm/clips.jsonl").read_text().splitlines()]
    assert result["clips"] == len(clips) > 0 and manifest["action_dim"] == ACTION_DIM
    assert {c["group"]: c["split"] for c in clips} == {"reach:3": "heldout_layout", "reach:5": "train"}
    assert manifest["counts"]["heldout_layout"] + manifest["counts"]["train"] == len(clips)
    for clip in clips:
        data = np.load(tmp_path/"wm"/clip["actions"])
        record = json.loads((runs/clip["episode"]/"episode.json").read_text())
        arrays = np.load(runs/clip["episode"]/"trajectory.npz")
        assert data["timestamps_s"][0] == 0 and np.all(np.diff(data["timestamps_s"]) > 0)
        expected, mask = commanded_vectors(record, arrays, data["rows"])
        np.testing.assert_allclose(data["commanded"], expected)
        assert (data["commanded_mask"] == mask).all() and mask[:, SLICES["right_palm_position"]].all()
        assert not mask[:, SLICES["left_palm_position"]].any()
        # Achieved vectors are FK of recorded qpos and must match the recorded palm.
        np.testing.assert_allclose(data["achieved"][:, SLICES["right_palm_position"]], arrays["right_wrist"][data["rows"]], atol=1e-9)
        assert isinstance(clip["camera_static"]["ego"], bool)
        assert data["camera_ego_intrinsics"].shape == (4, 3, 3)
        for key in ("frames", "achieved_conditions", "command_conditions"):
            assert all((tmp_path/"wm"/p).is_file() for p in clip[key]["ego"])
    legacy = export_dataset(runs, tmp_path/"legacy", horizon=2, splits={"reach": "test"})
    assert set(legacy["counts"]) >= {"test"} and legacy["counts"]["test"] == legacy["windows"]
