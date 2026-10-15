"""Self-edits (memory, skills, guidance) accepted only through a testing gate.

The gate evaluates the unchanged store and a candidate copy with the change
applied. ``evaluate(store_snapshot_path, task_set)`` is injected so the same
gate can use imagination, simulation or a fake in tests. Accepted changes are
committed with their evidence; rejected and pending changes are only logged.
"""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import tempfile
from .memory import KINDS, MemoryBank
from .skills import SkillLibrary, skill_name
from .store import VersionedStore

CHANGE_KINDS = ("memory_write", "memory_delete", "skill_write", "skill_edit", "guidance_edit")
GUIDANCE = ("planner", "critic", "perception")
PAYLOAD_FIELDS = {
    "memory_write": ({"text"}, {"tags", "task", "objects", "place", "evidence"}),
    "memory_delete": (set(), set()),
    "skill_write": ({"description", "body"}, {"examples", "scripts"}),
    "skill_edit": (set(), {"description", "body", "examples", "scripts"}),
    "guidance_edit": ({"text"}, set()),
}


@dataclass(frozen=True)
class Change:
    kind: str
    target: str
    payload: dict = field(default_factory=dict)
    rationale: str = ""
    trigger_episodes: tuple = ()

    def __post_init__(self):
        if self.kind not in CHANGE_KINDS:
            raise ValueError("change kind must be one of "+", ".join(CHANGE_KINDS))
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("change target is required")
        if not isinstance(self.payload, dict) or not isinstance(self.rationale, str):
            raise ValueError("payload must be an object and rationale a string")
        required, optional = PAYLOAD_FIELDS[self.kind]
        keys = set(self.payload)
        if required-keys or keys-required-optional:
            raise ValueError(f"{self.kind} payload needs {sorted(required)} and allows {sorted(optional)}")
        if self.kind == "skill_edit" and not keys:
            raise ValueError("skill_edit must change at least one field")
        if self.kind == "memory_write" and self.target not in KINDS:
            raise ValueError("memory_write target is the memory kind")
        if self.kind in ("skill_write", "skill_edit"):
            skill_name(self.target)
        if self.kind == "guidance_edit" and self.target not in GUIDANCE:
            raise ValueError("guidance_edit target must be one of "+", ".join(GUIDANCE))
        for key in ("text", "description", "body"):
            if key in self.payload and (not isinstance(self.payload[key], str) or not self.payload[key].strip()):
                raise ValueError(f"payload {key} must be non-empty text")
        for key in ("examples", "scripts"):
            files = self.payload.get(key, {})
            if not isinstance(files, dict) or not all(isinstance(v, str) for v in files.values()):
                raise ValueError(f"payload {key} must map file names to text")
        if isinstance(self.trigger_episodes, str) or not all(isinstance(e, str) for e in self.trigger_episodes):
            raise ValueError("trigger episodes must be strings")
        object.__setattr__(self, "trigger_episodes", tuple(self.trigger_episodes))

    def to_dict(self):
        return asdict(self)


class Guidance:
    """Harness guidance read by the planner, critic and perception prompts."""

    def __init__(self, store):
        self.store = store

    def _rel(self, name):
        if name not in GUIDANCE:
            raise ValueError("unknown guidance file")
        return f"guidance/{name}.md"

    def read(self, name):
        rel = self._rel(name)
        return self.store.read(rel) if self.store.exists(rel) else ""

    def write(self, name, text):
        if not isinstance(text, str) or len(text) > 20000:
            raise ValueError("guidance must be text of at most 20000 characters")
        self.store.write(self._rel(name), text.rstrip()+"\n")

    def all(self):
        return {name: self.read(name) for name in GUIDANCE}

    def render(self):
        return "\n\n".join(f"## {name} guidance\n{text.strip()}" for name, text in self.all().items() if text.strip())


def apply_change(store, change):
    """Apply a validated change to the store files (no commit). Returns the affected id."""
    p = change.payload
    if change.kind == "memory_write":
        return MemoryBank(store).write(change.target, p["text"], p.get("tags", ()), p.get("task"),
                                       p.get("evidence") or list(change.trigger_episodes), p.get("objects", ()),
                                       p.get("place"))
    if change.kind == "memory_delete":
        MemoryBank(store).delete(change.target)
    elif change.kind == "skill_write":
        SkillLibrary(store).write(change.target, p["description"], p["body"], p.get("examples"), p.get("scripts"))
    elif change.kind == "skill_edit":
        library = SkillLibrary(store)
        current = library.load(change.target)
        library.write(change.target, p.get("description", current["description"]), p.get("body", current["body"]),
                      p.get("examples"), p.get("scripts"))
    else:
        Guidance(store).write(change.target, p["text"])
    return change.target


@dataclass
class Decision:
    status: str  # accepted, rejected, pending_human, accepted_ungated
    reason: str
    change: dict
    baseline: dict = None
    candidate: dict = None
    commit: str = None

    def to_dict(self):
        return asdict(self)


class TestingGate:
    __test__ = False  # not a pytest class

    def __init__(self, store, evaluate, ablate=False):
        self.store, self.evaluate, self.ablate = store, evaluate, ablate

    def _score(self, path, tasks):
        if not tasks:
            return None
        result = dict(self.evaluate(Path(path), list(tasks)))
        if not 0 <= float(result["success_rate"]) <= 1:
            raise ValueError("evaluate must return success_rate in [0, 1]")
        return result

    def review(self, change, regression_set, trigger_set, min_trigger_gain=0.0, tolerance=0.0, require_human=False):
        sets = {"regression_set": list(regression_set), "trigger_set": list(trigger_set)}
        if require_human:
            self.store.append_log({"event": "pending_human", "change": change.to_dict(), **sets})
            return Decision("pending_human", "safety-related change needs human approval", change.to_dict())
        if self.ablate:
            apply_change(self.store, change)
            sha = self.store.commit(f"{change.kind}: {change.target} (ungated)",
                                    {"gate": "disabled", "change": change.to_dict()})
            return Decision("accepted_ungated", "testing gate disabled (ablation)", change.to_dict(), commit=sha)
        if not trigger_set:
            raise ValueError("a gated change needs a non-empty trigger set")
        with tempfile.TemporaryDirectory(prefix="hwap_gate_") as work:
            base = self.store.snapshot(Path(work)/"baseline")
            cand = self.store.snapshot(Path(work)/"candidate")
            apply_change(VersionedStore(cand), change)
            baseline = {"regression": self._score(base, regression_set), "trigger": self._score(base, trigger_set)}
            candidate = {"regression": self._score(cand, regression_set), "trigger": self._score(cand, trigger_set)}
        rate = lambda s: None if s is None else float(s["success_rate"])
        regression_ok = baseline["regression"] is None or rate(candidate["regression"]) >= rate(baseline["regression"])-tolerance
        gain = rate(candidate["trigger"])-rate(baseline["trigger"])
        evidence = {"change": change.to_dict(), **sets, "baseline": baseline, "candidate": candidate,
                    "trigger_gain": gain, "min_trigger_gain": min_trigger_gain, "tolerance": tolerance}
        if regression_ok and gain > min_trigger_gain:
            apply_change(self.store, change)
            sha = self.store.commit(f"{change.kind}: {change.target}", evidence)
            return Decision("accepted", f"trigger gain {gain:+.3f}, regression held", change.to_dict(), baseline, candidate, sha)
        reason = "regression got worse" if not regression_ok else f"trigger gain {gain:+.3f} <= {min_trigger_gain}"
        self.store.append_log({"event": "rejected", "reason": reason, **evidence})
        return Decision("rejected", reason, change.to_dict(), baseline, candidate)

    def approve(self, change, approver, note=""):
        """Apply a pending safety-related change after explicit human approval."""
        if not isinstance(approver, str) or not approver.strip():
            raise ValueError("approval needs the approver's name")
        apply_change(self.store, change)
        sha = self.store.commit(f"{change.kind}: {change.target} (human approved)",
                                {"change": change.to_dict(), "approver": approver, "note": note})
        return Decision("accepted", "human approved", change.to_dict(), commit=sha)


PUBLIC_OUTCOME = ("duration_s", "success", "failure")


def reflection_prompt(episode_record, memory_entries, skills):
    """Build the reflection request; only public observations/feedback are included."""
    metrics = episode_record.get("metrics", {})
    episode = {"task": episode_record.get("task"), "seed": episode_record.get("seed"),
               "success": metrics.get("success"), "failure": metrics.get("failure"),
               "chunks": [{"chunk": c["chunk"], "outcome": {k: c["outcome"].get(k) for k in PUBLIC_OUTCOME}}
                          for c in episode_record.get("chunks", [])],
               "agent_notes": [h.get("notes") for h in episode_record.get("agent_history", []) if h.get("notes")]}
    memory = "\n".join(f"[{e['kind']}] {' '.join(e['text'].split())} ({e['id']})" for e in memory_entries) or "(none)"
    skill_text = "\n".join(f"- {s['name']}: {s['description']}" for s in skills) or "(none)"
    return "\n".join([
        "Reflect on this humanoid episode. Propose at most 5 changes that would help future attempts.",
        "Return ONLY JSON: {\"changes\": [{\"kind\", \"target\", \"payload\", \"rationale\", \"trigger_episodes\"}]}.",
        "kind/target/payload: memory_write target=<" + "|".join(KINDS) + "> payload={text, tags?, task?, objects?, place?};",
        "memory_delete target=<memory id> payload={}; skill_write target=<kebab-name> payload={description, body,",
        "examples?, scripts?}; skill_edit target=<existing skill> payload={description?, body?, examples?, scripts?};",
        "guidance_edit target=<" + "|".join(GUIDANCE) + "> payload={text} (replaces the whole file).",
        "Write concrete, reusable lessons with numbers. Do not record hidden simulator values; they were not observed.",
        "Every change is tested on a regression set before it is kept.",
        "", "## Episode", json.dumps(episode, ensure_ascii=False),
        "", "## Current memory", memory, "", "## Current skills", skill_text])


def parse_changes(value, limit=10):
    """Strictly validate LLM-proposed changes (JSON text, list, or {"changes": [...]})."""
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        if set(value) != {"changes"}:
            raise ValueError("expected exactly {\"changes\": [...]}")
        value = value["changes"]
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"changes must be a list of at most {limit}")
    changes = []
    for item in value:
        if not isinstance(item, dict) or set(item)-{"kind", "target", "payload", "rationale", "trigger_episodes"}:
            raise ValueError("invalid change fields")
        episodes = item.get("trigger_episodes", [])
        if not isinstance(episodes, list):
            raise ValueError("trigger_episodes must be a list")
        changes.append(Change(item.get("kind"), item.get("target"), item.get("payload", {}),
                              item.get("rationale", ""), tuple(episodes)))
    return changes
