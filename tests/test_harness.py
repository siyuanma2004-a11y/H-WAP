"""Harness loop, world-model interface, critics and LLM adapters without network calls."""
import json
from pathlib import Path
import numpy as np
import pytest
from hwap.actions import CommandChunk, Keyframe, Pose
from hwap.critic import PrivilegedCritic, VLMCritic
from hwap.environment import HumanoidEnvironment
from hwap.harness import (HarnessConfig, HarnessTools, LLMPlanner, ScriptedPlanner, normalize_chunk,
                          reflect, run_episode, structured_variants)
from hwap.llm import LLMClient
from hwap.recording import EpisodeRecorder, replay_episode
from hwap.world_model import OracleWorldModel, sim_check


ROOT = Path(__file__).resolve().parents[1]
PRIVILEGED = ("goal", "goal_error_m", "lifted", "object_position", "layout")


class FakeClient(LLMClient):
    provider = "fake"

    def __init__(self, replies):
        super().__init__({"model": "fake"})
        self.replies, self.requests = list(replies), []

    def _request(self, system, blocks, schema):
        self.requests.append({"system": system, "blocks": blocks, "schema": schema})
        reply = self.replies.pop(0)
        return (reply if isinstance(reply, str) else json.dumps(reply)), {"input_tokens": 3, "output_tokens": 2}, {}


def reach_chunk(env, name="reach"):
    start, goal = Pose(tuple(env.sim.palm_position("right"))), Pose(tuple(env.goal))
    return CommandChunk(name, (Keyframe(0, right_wrist=start), Keyframe(1.5, right_wrist=goal), Keyframe(2.5, right_wrist=goal)), frame="world")


def assert_public(text):
    for key in PRIVILEGED:
        assert f'"{key}"' not in text


def test_oracle_imagination_restores_state_and_reports_disagreement():
    with HumanoidEnvironment(views=("ego",), width=48, height=36) as env:
        env.reset(3)
        before = env.sim.data.qpos.copy()
        chunk = reach_chunk(env)
        imagined = OracleWorldModel(perturb_m=0.02).imagine(env, chunk, samples=2)
        np.testing.assert_array_equal(env.sim.data.qpos, before)
        assert imagined["samples"] == 2 and imagined["disagreement"] is not None
        assert len(imagined["frames"][0]) == len(imagined["timestamps"]) >= 20
        assert np.all(np.diff(imagined["timestamps"]) > 0)
        assert imagined["privileged"]["goal_error_m"] < 0.045
        assert_public(json.dumps(imagined["public"]))
        check = sim_check(env, chunk)
        assert check["feasible"] and check["rows"][0]["robot_body_positions"].shape[1] == 3
        np.testing.assert_array_equal(env.sim.data.qpos, before)


def test_full_loop_with_oracle_search_records_and_replays(tmp_path):
    config = HarnessConfig(max_chunks=2, refine_rounds=1, local_search=3, samples=1)
    with HumanoidEnvironment(views=("ego", "workspace"), width=48, height=36, budget_s=10) as env:
        env.reset(4)
        planner = ScriptedPlanner(lambda context, observation, step: reach_chunk(env) if step == 0 else None)
        recorder = EpisodeRecorder(env, tmp_path/"episode")
        extra = run_episode(env, planner, recorder, config, OracleWorldModel(), PrivilegedCritic())
        record = recorder.finish(extra)
    assert record["run_error"] is None and record["metrics"]["success"]
    step = record["steps"][0]
    assert step["search"]["chosen"] in range(3) and len(step["search"]["scores"]) == 3
    assert record["world_model_calls"] == 4 and record["critic_privileged"]
    assert "right_wrist_error_m" in step["imagined_vs_real"]
    assert any((tmp_path/"episode/imagination").rglob("*.png"))
    assert replay_episode(tmp_path/"episode")["max_qpos_error"] == 0


def test_llm_planner_converts_gaze_and_repairs_invalid_output():
    with HumanoidEnvironment(views=("ego",), width=32, height=24) as env:
        env.reset(0)
        observation = env.observe()
        wrist = observation["robot"]["right_wrist"]
        good = {"schema_version": 1, "name": "look", "frame": "world", "keyframes": [
            {"t": 0, "right_wrist": {"position": wrist}, "gaze": {"yaw": 0.0, "pitch": 0.0}},
            {"t": 1.0, "right_wrist": {"position": wrist}, "gaze": {"yaw": 0.2, "pitch": 0.1}}]}
        client = FakeClient(["not json", "```json\n"+json.dumps(good)+"\n```"])
        planner = LLMPlanner(client)
        context = {"instruction": observation["instruction"], "observation": {"robot": observation["robot"]}}
        chunk = planner.propose(context, observation)
    assert client.calls == 2 and "rejected" in client.requests[1]["blocks"][-1]["text"]
    assert chunk.keyframes[1].head.position is None and abs(np.linalg.norm(chunk.keyframes[1].head.quaternion)-1) < 1e-9
    assert any(b["type"] == "image" for b in client.requests[0]["blocks"])
    fixed = normalize_chunk(good, allow_gaze=False)
    assert all(k.head is None for k in fixed.keyframes)
    assert "Gaze is fixed" in LLMPlanner(client, allow_gaze=False).system


def test_structured_variants_keep_start_continuous():
    with HumanoidEnvironment(views=("ego",), width=32, height=24) as env:
        env.reset(1)
        chunk = reach_chunk(env)
        variants = structured_variants(chunk, 4, np.random.default_rng(0))
    assert variants[0] is chunk and len({v.name for v in variants}) == 4
    for v in variants[1:]:
        assert v.keyframes[0] == chunk.keyframes[0]
        assert 0 < np.linalg.norm(np.subtract(v.keyframes[2].right_wrist.position, chunk.keyframes[2].right_wrist.position)) <= 0.03*np.sqrt(3)+1e-9


def test_vlm_critic_sees_frames_but_not_privileged_truth():
    with HumanoidEnvironment(views=("ego",), width=32, height=24) as env:
        env.reset(2)
        chunk = reach_chunk(env)
        imagined = OracleWorldModel().imagine(env, chunk)
    client = FakeClient([{"accept": False, "score": 1.4, "feedback": "lower the wrist", "issues": ["high"]},
                         {"scores": [0.2, 0.9], "reason": "second"}])
    critic = VLMCritic(client, "guidance text")
    review = critic.review("touch the marker", imagined, chunk)
    assert review["score"] == 1.0 and not review["accept"]
    request = client.requests[0]
    assert request["schema"]["required"] == ["accept", "score", "feedback", "issues"]
    assert sum(b["type"] == "image" for b in request["blocks"]) == 3
    assert_public(json.dumps([b.get("text", "") for b in request["blocks"]]))
    assert critic.rank("touch", [imagined, imagined], [chunk, chunk]) == [0.2, 0.9]


def test_tool_surface_matches_appendix_and_hides_truth(tmp_path):
    with HumanoidEnvironment(views=("ego",), width=32, height=24) as env:
        env.reset(0)
        tools = HarnessTools(env, EpisodeRecorder(env, tmp_path/"ep"), OracleWorldModel())
        names = {spec["name"] for spec in tools.specs()}
        assert names == {"get_observation", "sim_check", "imagine", "execute", "run_policy", "memory_search",
                         "memory_write", "skill_list", "skill_load", "skill_write", "run_regression", "choose_view"}
        observation = tools.call("get_observation", {})
        assert_public(json.dumps(observation))
        chunk = reach_chunk(env).to_dict()
        assert tools.call("sim_check", {"chunk": chunk})["feasible"]
        assert tools.call("imagine", {"chunk": chunk})["frames"] > 0
        assert tools.call("execute", {"chunk": chunk})["success"]


def test_reflection_routes_changes_through_gate(tmp_path):
    from hwap.improvement import TestingGate
    from hwap.memory import MemoryBank
    from hwap.skills import SkillLibrary
    from hwap.store import VersionedStore
    store = VersionedStore.from_seed(tmp_path/"store", ROOT/"harness_seed")
    reply = {"changes": [{"kind": "memory_write", "target": "body", "payload": {"text": "Right palm reaches z=1.05 m comfortably."},
                          "rationale": "worked", "trigger_episodes": ["ep0"]}]}
    record = {"task": "reach", "seed": 0, "metrics": {"success": True, "failure": None, "goal_error_m": 0.01}, "chunks": []}
    evaluate = lambda path, tasks: {"success_rate": 1.0 if (Path(path)/"memory/body").exists() else 0.5}
    result = reflect(FakeClient([reply]), record, MemoryBank(store), SkillLibrary(store), TestingGate(store, evaluate), ["reach:9"], ["reach:0"])
    assert result["decisions"][0]["status"] == "accepted"
    assert MemoryBank(store).search("palm")[0]["kind"] == "body"
    ungated = reflect(FakeClient([reply]), record, MemoryBank(store), SkillLibrary(store), TestingGate(store, None, ablate=True))
    assert ungated["decisions"][0]["status"] == "accepted_ungated"


def test_harness_configs_validate_and_smoke_runs(tmp_path):
    from hwap.runner import load_config, run_experiment
    for path in sorted((ROOT/"configs/harness").glob("*.json")):
        load_config(path)
    config = json.loads((ROOT/"configs/harness/smoke_oracle.json").read_text())
    config.update({"seeds": [0], "width": 48, "height": 36, "store": {"seed": str(ROOT/"harness_seed")}})
    config["harness"].update({"samples": 1, "local_search": 2})
    (tmp_path/"smoke.json").write_text(json.dumps(config))
    summary = run_experiment(tmp_path/"smoke.json", tmp_path/"out")
    row = summary["runs"][0]
    assert summary["success_rate"] == 1.0 and row["run_error"] is None and row["store_head"]
    episode = json.loads((Path(row["episode"])/"episode.json").read_text())
    assert episode["agent"] == "replay_calibration" and episode["skills_loaded"]


def test_anthropic_request_shape(monkeypatch):
    pytest.importorskip("anthropic")
    from hwap.llm import AnthropicClient
    captured = {}

    class Response:
        stop_reason, stop_details, model, _request_id = "end_turn", None, "claude-opus-5-5", "req_1"
        content = [type("Block", (), {"type": "text", "text": '{"ok": true}'})()]
        usage = type("Usage", (), {"input_tokens": 5, "output_tokens": 1})()

    client = AnthropicClient({"provider": "anthropic"})
    monkeypatch.setattr(client.client.beta.messages, "create", lambda **kw: captured.update(kw) or Response())
    result = client.complete("system", [{"type": "text", "text": "hi"}, {"type": "image", "rgb": np.zeros((4, 4, 3), np.uint8), "label": "ego"}],
                             schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False})
    assert result["text"] == '{"ok": true}' and client.tokens == 6
    assert captured["model"] == "claude-opus-5-5" and captured["fallbacks"] == "default"
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert captured["messages"][0]["content"][2]["source"]["media_type"] == "image/png"


@pytest.mark.parametrize("pose_source", ["sim", "command"])
def test_remote_world_model_request_and_decoding(monkeypatch, pose_source):
    import base64
    import io
    from hwap.images import png_bytes
    from hwap.world_model import RemoteVideoWorldModel
    seen = {}

    def respond(request, timeout):
        body = json.loads(request.data)
        seen.update(body)
        frames = len(body["pose_conditions"]["ego"])
        sample = {"ego": [base64.b64encode(png_bytes(np.full((24, 32, 3), 10*i, np.uint8))).decode() for i in range(frames)]}
        return io.BytesIO(json.dumps({"samples": [sample, sample]}).encode())

    monkeypatch.setattr("hwap.world_model.urllib.request.urlopen", respond)
    with HumanoidEnvironment(views=("ego",), width=32, height=24) as env:
        env.reset(0)
        before = env.sim.data.qpos.copy()
        model = RemoteVideoWorldModel({"base_url": "http://127.0.0.1:9", "model": "wm", "pose_source": pose_source})
        imagined = model.imagine(env, reach_chunk(env), samples=2)
        np.testing.assert_array_equal(env.sim.data.qpos, before)
    assert seen["samples"] == 2 and len(seen["history"]["ego"]) == 1
    assert len(seen["pose_conditions"]["ego"]) == len(seen["timestamps"]) == len(imagined["frames"][0]) > 5
    conditions = [base64.b64decode(c) for c in seen["pose_conditions"]["ego"]]
    assert any(c != conditions[0] for c in conditions[1:])  # the commanded/simulated pose moves
    assert imagined["disagreement"] == 0.0 and imagined["frames"][0][3]["ego"][0, 0, 0] == 30
    assert_public(json.dumps({k: seen[k] for k in ("timestamps", "chunk")}))


def test_look_chunks_and_view_ranking_use_gaze():
    from hwap.harness import look_chunk, rank_views
    with HumanoidEnvironment(views=("ego",), width=32, height=24) as env:
        env.reset(0)
        observation = env.observe(render=False)
        candidates = [look_chunk(observation, 0.0, 0.0), look_chunk(observation, 0.3, 0.2)]
        for chunk in candidates:
            env.validate_chunk(chunk)
        ranking = rank_views(env, OracleWorldModel(perturb_m=0.03), candidates, samples=2)
    assert {r["chunk"] for r in ranking} == {c.name for c in candidates}
    assert all(r["disagreement"] >= 0 for r in ranking) and ranking[0]["score"] >= ranking[1]["score"]
    assert max(r["motion_rad"] for r in ranking) > 0.3
