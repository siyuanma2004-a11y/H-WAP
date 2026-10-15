"""Versioned memory/skill store and the self-edit testing gate (no simulator)."""
import json
from pathlib import Path
import pytest

from hwap.actions import CommandChunk
from hwap.improvement import Change, Guidance, TestingGate, apply_change, parse_changes, reflection_prompt
from hwap.memory import MemoryBank
from hwap.skills import SkillLibrary
from hwap.store import VersionedStore

SEED = Path(__file__).resolve().parents[1]/"harness_seed"


@pytest.fixture
def store(tmp_path):
    return VersionedStore.from_seed(tmp_path/"store", SEED)


def test_seed_store_is_versioned_and_rolls_back(store):
    assert (store.root/".git").is_dir() and store.log()[0]["event"] == "seeded"
    first = store.head()
    store.write("guidance/planner.md", "edited\n")
    sha = store.commit("edit planner", {"why": "test"})
    assert sha != first and store.read("guidance/planner.md") == "edited\n"
    assert store.log()[0]["evidence"] == {"why": "test"}
    store.rollback(sha, "bad edit")
    assert store.read("guidance/planner.md") != "edited\n"
    assert store.log()[0]["event"] == "rollback"
    # The changelog is untracked so rejected/pending events never alter content.
    assert "changelog.jsonl" not in store._git("ls-files").split()


def test_paths_and_names_cannot_escape(store):
    for bad in ("../x", "/etc/passwd", "a/../../b", ".git/config", "changelog.jsonl", ""):
        with pytest.raises(ValueError):
            store.path(bad)
    library = SkillLibrary(store)
    for bad in ("../evil", "Upper", "a_b", "a--b"):
        with pytest.raises(ValueError):
            library.write(bad, "d", "body")
    with pytest.raises(ValueError):
        library.write("ok-skill", "d", "body", scripts={"../x.py": "print(1)"})
    with pytest.raises(ValueError):
        MemoryBank(store).delete("../../guidance/planner")


def test_memory_search_ranking_and_render(store):
    memory = MemoryBank(store)
    drawer = memory.write("world", "Kitchen drawer 2 sticks. Pull from the bottom of the handle.",
                          tags=["drawer"], task="open_drawer", place="kitchen")
    reach = memory.write("body", "At waist 0.75 m the right wrist reaches 0.55 m forward.", tags=["reach"])
    memory.write("perception", "Crouch to waist 0.55 m to look inside deep cabinets.", tags=["cabinet"])
    assert [e["id"] for e in memory.search("drawer handle")][0] == drawer
    assert memory.search(task="open_drawer")[0]["id"] == drawer
    assert memory.search(tags=["reach"])[0]["id"] == reach
    assert memory.search("nothing matches zzz") == []
    assert len(memory.search()) == 3 and len(memory.search(kinds=["body"])) == 1
    assert memory.render(memory.search("drawer"))[:8] == "[world] "
    memory.write("world", "Kitchen drawer 2 sticks. Pull from the bottom of the handle.")
    assert len(memory.consolidate()) == 1
    memory.delete(reach)
    assert memory.search(tags=["reach"]) == []
    with pytest.raises(ValueError):
        memory.write("unknown", "x")


def test_skill_library_load_match_use_and_script(store):
    library = SkillLibrary(store)
    assert library.match("touch the green marker with your right palm")[0]["name"] == "reach-visible-marker"
    skill = library.load("reach-visible-marker")
    assert skill["scripts"] == ["skills/reach-visible-marker/scripts/straight_reach.py"]
    for example in skill["examples"]:
        CommandChunk.parse(json.loads(store.read(example)))
    assert "reach-visible-marker" in library.describe()
    library.record_use("reach-visible-marker", True)
    library.record_use("reach-visible-marker", False)
    assert library.load("reach-visible-marker")["uses"] == 2
    assert library.list()[0]["successes"] == 1
    keyframes = library.run_script("reach-visible-marker", "straight_reach.py",
                                   {"start": [0, -0.23, 0.62], "target": [0.3, -0.25, 1.05]})
    chunk = CommandChunk.parse({"name": "r", "frame": "world", "keyframes": keyframes})
    assert chunk.duration == pytest.approx(2.5)
    library.write("bad-script", "returns nothing useful", "body", scripts={"x.py": "print('[]')"})
    with pytest.raises(ValueError):
        library.run_script("bad-script", "x.py", {})
    library.write("slow-script", "sleeps", "body", scripts={"x.py": "import time; time.sleep(5)"})
    with pytest.raises(RuntimeError):
        library.run_script("slow-script", "x.py", {}, timeout=0.5)


def fake_evaluate(path, tasks):
    """Success depends on whether the store has a memory tagged 'helps'/'hurts'."""
    found = MemoryBank(VersionedStore(path)).search(tags=["helps", "hurts"])
    helps, hurts = any("helps" in e["tags"] for e in found), any("hurts" in e["tags"] for e in found)
    per_task = {t: float((t.startswith("trigger") and helps) or (t.startswith("reg") and not hurts)) for t in tasks}
    return {"success_rate": sum(per_task.values())/len(tasks), "per_task": per_task, "n": len(tasks)}


def memory_change(tag):
    return Change("memory_write", "world", {"text": f"Lesson {tag}.", "tags": [tag]}, "test", ("ep1",))


def test_gate_accepts_rejects_pending_and_ablation(store):
    gate = TestingGate(store, fake_evaluate)
    head = store.head()
    rejected = gate.review(memory_change("neutral"), ["reg_a"], ["trigger_a"])
    assert rejected.status == "rejected" and store.head() == head
    assert MemoryBank(store).search(tags=["neutral"]) == []
    hurts = gate.review(memory_change("hurts"), ["reg_a"], ["trigger_a"])
    assert hurts.status == "rejected" and "regression" in hurts.reason
    accepted = gate.review(memory_change("helps"), ["reg_a", "reg_b"], ["trigger_a"])
    assert accepted.status == "accepted" and accepted.commit == store.head()
    assert accepted.candidate["trigger"]["success_rate"] == 1.0
    assert MemoryBank(store).search(tags=["helps"])
    change = Change("guidance_edit", "planner", {"text": "Never exceed 0.3 m/s."}, "safety")
    pending = gate.review(change, ["reg_a"], ["trigger_a"], require_human=True)
    assert pending.status == "pending_human" and "0.3" not in Guidance(store).read("planner")
    gate.approve(change, "operator")
    assert "0.3" in Guidance(store).read("planner")
    ungated = TestingGate(store, fake_evaluate, ablate=True).review(memory_change("hurts"), ["reg_a"], ["trigger_a"])
    assert ungated.status == "accepted_ungated" and MemoryBank(store).search(tags=["hurts"])
    events = [e["event"] for e in store.log(50)]
    assert events.count("rejected") == 2 and "pending_human" in events
    with pytest.raises(ValueError):
        gate.review(memory_change("helps"), ["reg_a"], [])


def test_changes_are_validated_and_applied(store):
    changes = parse_changes(json.dumps({"changes": [
        {"kind": "skill_write", "target": "open-drawer", "payload": {"description": "Open a drawer.",
         "body": "Pull straight back 20 cm.", "scripts": {"pull.py": "print(1)"}}, "rationale": "r",
         "trigger_episodes": ["ep43"]},
        {"kind": "skill_edit", "target": "open-drawer", "payload": {"body": "Pull 25 cm."}},
        {"kind": "guidance_edit", "target": "critic", "payload": {"text": "Check the handle."}}]}))
    for change in changes:
        apply_change(store, change)
    assert SkillLibrary(store).load("open-drawer")["body"] == "Pull 25 cm."
    assert Guidance(store).read("critic").strip() == "Check the handle."
    for bad in ({"changes": [{"kind": "rm_rf", "target": "x"}]},
                {"changes": [{"kind": "memory_write", "target": "world", "payload": {}}]},
                {"changes": [{"kind": "memory_write", "target": "world", "payload": {"text": "x", "extra": 1}}]},
                {"changes": [{"kind": "guidance_edit", "target": "../x", "payload": {"text": "x"}}]},
                {"changes": [{"kind": "skill_write", "target": "a", "payload": {"description": "d", "body": "b"},
                              "trigger_episodes": "ep1"}]},
                {"changes": []} | {"other": 1}):
        with pytest.raises(ValueError):
            parse_changes(bad)


def test_reflection_prompt_excludes_privileged_values(store):
    record = {"task": "reach", "seed": 3, "goal": [0.3, -0.25, 1.05],
              "metrics": {"success": False, "failure": None, "goal_error_m": 0.12},
              "chunks": [{"chunk": {"name": "c"}, "outcome": {"duration_s": 2.5, "success": False,
                                                                 "failure": None, "goal_error_m": 0.12}}]}
    text = reflection_prompt(record, MemoryBank(store).search(), SkillLibrary(store).list())
    assert "reach-visible-marker" in text and "goal_error" not in text and "1.05" not in text
