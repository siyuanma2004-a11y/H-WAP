"""Verify that exported windows preserve executable actions and their timing."""
import json
import numpy as np

from hwap.actions import CommandChunk, Keyframe, Pose
from hwap.dataset import export_dataset
from hwap.environment import HumanoidEnvironment
from hwap.recording import EpisodeRecorder


def test_export_aligns_raw_commands_across_chunk_boundaries(tmp_path):
    runs = tmp_path / "runs"
    with HumanoidEnvironment(width=64, height=48) as env:
        env.reset(17)
        recorder = EpisodeRecorder(env, runs / "episode")
        pose = Pose(tuple(env.sim.palm_position("right")))
        for index in range(2):
            chunk = CommandChunk(f"hold_{index}", (
                Keyframe(0, right_wrist=pose), Keyframe(0.4, right_wrist=pose)), frame="world")
            recorder.execute(chunk)
        recorder.finish()
    result = export_dataset(runs, tmp_path / "dataset", horizon=2)
    manifest = json.loads((tmp_path / "dataset/manifest.json").read_text())
    assert result["windows"] > 0 and manifest["command_conditioning_available"]
    names = set()
    for sample in manifest["samples"]:
        assert sample["command_chunks"]
        assert len(sample["future_timestamps_s"]) == 2
        assert np.all(np.diff(sample["future_timestamps_s"]) >= 0)
        end = sample["future_timestamps_s"][-1]
        for item in sample["command_chunks"]:
            chunk = CommandChunk.parse(item["chunk"])
            assert item["chunk_start_relative_s"] < end
            assert item["chunk_start_relative_s"] + chunk.duration > 0
            names.add(chunk.name)
        assert sample["group"] == "reach:17"
    assert names == {"hold_0", "hold_1"}
