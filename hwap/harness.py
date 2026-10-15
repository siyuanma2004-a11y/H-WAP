"""H-WAP agent harness: the proposal's full loop (section 5) with ablation switches.

    propose -> [sim_check] -> imagine -> critic -> revise (K rounds)
            -> local search (N variants, critic rank) -> execute -> compare
    after the episode: reflect -> proposed memory/skill changes -> testing gate

Every component is optional so baselines/ablations are configuration, not code:
direct LLM agent (no world model), WAP port (no memory/skills, fixed gaze),
oracle imagination, VLM vs privileged critic, gaze modes, memory/skills on/off.
"""
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import time
import numpy as np
from .actions import CommandChunk, Pose
from .images import png_bytes
from .llm import parse_json
from .world_model import sim_check


PUBLIC_OBSERVATION = ("task", "instruction", "robot", "capabilities", "cameras")


@dataclass
class HarnessConfig:
    max_chunks: int = 8
    refine_rounds: int = 2          # K, global imagine-critique-revise rounds
    local_search: int = 1           # N candidates around the accepted chunk (1 disables)
    search_scale_m: float = 0.03
    samples: int = 1                # world-model samples per imagination
    use_sim_check: bool = True      # SONIC/IK-in-sim feasibility before imagining
    gaze: str = "llm"               # llm | fixed (no gaze control ablation)
    use_memory: bool = True
    use_skills: bool = True
    reflect: bool = False           # middle loop: write memory/skills after the episode
    gate: str = "test"              # test | none (TestingGate ablate=True)
    save_imagination: bool = True
    seed: int = 0
    feedback_fields: tuple = ("duration_s", "success", "failure")

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown harness fields: {sorted(unknown)}")
        config = cls(**value)
        if config.gaze not in {"llm", "fixed"} or config.gate not in {"test", "none"}:
            raise ValueError("gaze must be llm/fixed and gate test/none")
        if not 1 <= config.max_chunks <= 100 or config.refine_rounds < 0 or config.local_search < 1 or config.samples < 1:
            raise ValueError("invalid harness budgets")
        return config


def public_observation(observation):
    return {k: observation[k] for k in PUBLIC_OBSERVATION if k in observation}


def gaze_quaternion(yaw, pitch):
    """Head orientation for gaze yaw (left +) and pitch (down +), robot heading frame."""
    qz = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
    qy = np.array([np.cos(pitch/2), 0, np.sin(pitch/2), 0])
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = qy
    return (w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2)


def normalize_chunk(raw, base_yaw=0.0, allow_gaze=True):
    """Accept the LLM-friendly `gaze: {yaw, pitch}` keyframe field and convert it to `head`."""
    raw = json.loads(json.dumps(raw))
    for key in raw.get("keyframes", []):
        gaze = key.pop("gaze", None)
        if gaze is not None:
            yaw = float(gaze.get("yaw", 0.0)) + (base_yaw if raw.get("frame") == "world" else 0.0)
            key["head"] = {"position": None, "quaternion": list(gaze_quaternion(yaw, float(gaze.get("pitch", 0.0))))}
        if not allow_gaze:
            key.pop("head", None)
    return CommandChunk.parse(raw)


def structured_variants(chunk, count, rng, scale=0.03, gaze_rad=0.12):
    """Candidate 0 is the chunk itself; others shift every later wrist target by one
    random offset (keeping t=0 continuous) and jitter gaze yaw when gaze is set."""
    candidates = [chunk]
    for index in range(1, count):
        offsets = {side: rng.uniform(-scale, scale, 3) for side in ("left_wrist", "right_wrist")}
        dyaw = rng.uniform(-gaze_rad, gaze_rad)
        keys = []
        for i, key in enumerate(chunk.keyframes):
            changes = {}
            for side, offset in offsets.items():
                pose = getattr(key, side)
                if pose is not None and i > 0:
                    changes[side] = Pose(tuple(np.array(pose.position)+offset), pose.quaternion)
            if key.head is not None and key.head.quaternion is not None and i > 0:
                q = np.array(key.head.quaternion)
                rz = np.array([np.cos(dyaw/2), 0, 0, np.sin(dyaw/2)])
                w1, x1, y1, z1 = rz
                w2, x2, y2, z2 = q
                rotated = (w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2)
                changes["head"] = Pose(key.head.position, rotated) if key.head.position is not None else replace(key.head, quaternion=rotated)
            keys.append(replace(key, **changes))
        candidates.append(replace(chunk, name=f"{chunk.name}_v{index}", keyframes=tuple(keys)))
    return candidates


def look_chunk(observation, yaw, pitch, duration=1.0, name=None):
    """Head-only chunk that holds both measured palms and turns the gaze."""
    robot = observation["robot"]
    keys = [{"t": t, "left_wrist": {"position": robot["left_wrist"]}, "right_wrist": {"position": robot["right_wrist"]},
             "gaze": {"yaw": y, "pitch": p}} for t, y, p in ((0.0, 0.0, 0.0), (duration, yaw, pitch))]
    return normalize_chunk({"schema_version": 1, "name": name or f"look_y{yaw:+.2f}_p{pitch:+.2f}", "frame": "world",
                            "keyframes": keys}, robot["base_pose"][2])


def rank_views(env, world_model, candidates, samples=3, motion_cost=0.001):
    """Active perception step 3: score each candidate look by the disagreement among
    imagined samples in the resulting view (where the uncertain world state becomes
    visible), minus a cost per radian of gaze motion. With the oracle, sample
    diversity comes from OracleWorldModel(perturb_m>0)."""
    results = []
    for chunk in candidates:
        imagined = world_model.imagine(env, chunk, samples=samples)
        head = chunk.keyframes[-1].head
        angle = 0.0 if head is None or head.quaternion is None else 2*np.arccos(min(1.0, abs(head.quaternion[0])))
        gain = imagined["disagreement"] or 0.0
        results.append({"chunk": chunk.name, "disagreement": gain, "motion_rad": float(angle), "score": float(gain-motion_cost*angle)})
    return sorted(results, key=lambda r: -r["score"])


PLANNER_SYSTEM = """You are the planner of an LLM agent that operates a simulated Unitree G1 humanoid the way a VR
teleoperator does. You never command joints. You write one short command chunk at a time as JSON:
{"schema_version":1,"name":str,"frame":"heading_at_start"|"world","keyframes":[...]} with 2-8 keyframes,
t starting at 0 and ending within 3 s. Keyframe fields (all optional except t; any used field must be set at t=0):
- left_wrist/right_wrist: {"position":[x,y,z] metres,"quaternion":[w,x,y,z] or null} for the palm point 10 cm past the wrist.
  heading_at_start: XY relative to the base at chunk start (x forward, y left), Z above the floor. world: global metres.
- left_hand/right_hand: closure 0 (open) .. 1 (closed).
- velocity: [vx, vy, yaw_rate] in the current heading frame (limits 0.15, 0.10, 0.30).
- gaze: {"yaw": rad left+, "pitch": rad down+} relative to the robot heading; moves the head camera.
Obey observation.capabilities: do not use fields the backend does not support. Start every wrist target at its
measured position at t=0. Prefer short, conservative chunks; you will see the outcome and plan the next one.
Return JSON only."""


class LLMPlanner:
    name = "llm_planner"

    def __init__(self, client, guidance="", allow_gaze=True):
        self.client, self.guidance, self.allow_gaze = client, guidance, allow_gaze
        system = PLANNER_SYSTEM if allow_gaze else PLANNER_SYSTEM.replace(
            '- gaze: {"yaw": rad left+, "pitch": rad down+} relative to the robot heading; moves the head camera.\n',
            "- Gaze is fixed in this configuration; never set gaze or head.\n")
        self.system = system+("\n\nHarness guidance:\n"+guidance if guidance else "")

    @property
    def calls(self):
        return self.client.calls

    @property
    def tokens(self):
        return self.client.tokens

    def _ask(self, payload, observation, purpose):
        content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
        content += [{"type": "image", "rgb": rgb, "label": view} for view, rgb in observation.get("images", {}).items()]
        base_yaw = observation["robot"]["base_pose"][2]
        error = None
        for _ in range(2):
            if error:
                content = content+[{"type": "text", "text": "Your previous chunk was rejected: "+error+". Return a corrected chunk."}]
            text = self.client.complete(self.system, content, purpose=purpose)["text"]
            try:
                return normalize_chunk(parse_json(text), base_yaw, self.allow_gaze)
            except (ValueError, KeyError, TypeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        raise ValueError("planner did not produce a valid chunk: "+error)

    def propose(self, context, observation):
        return self._ask({**context, "request": "write the next command chunk"}, observation, "propose")

    def revise(self, context, observation, chunk, feedback):
        return self._ask({**context, "previous_chunk": chunk.to_dict(), "critic_feedback": feedback,
                          "request": "revise the chunk to address the feedback"}, observation, "revise")

    def fix(self, context, observation, chunk, error):
        return self._ask({**context, "rejected_chunk": chunk.to_dict(), "validation_error": error,
                          "request": "the harness rejected this chunk; return a valid one"}, observation, "fix")


class ScriptedPlanner:
    """Deterministic stand-in: `fn(context, observation, step) -> CommandChunk or None`."""
    name = "scripted_planner"

    def __init__(self, fn):
        self.fn, self.step, self.calls, self.tokens = fn, 0, 0, 0

    def propose(self, context, observation):
        self.step += 1
        return self.fn(context, observation, self.step-1)

    def revise(self, context, observation, chunk, feedback):
        return chunk

    def fix(self, context, observation, chunk, error):
        raise ValueError(error)


def _save_imagination(folder, imagination):
    folder.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(imagination["frames"][0] if imagination["frames"] else []):
        for view, rgb in frame.items():
            (folder/f"{view}_{index:03d}.png").write_bytes(png_bytes(rgb))
    return str(folder)


def compare_imagined_real(imagination, observation):
    """Imagined-vs-real gap from public quantities only (proposal metric 'match')."""
    result = {}
    final = imagination["public"].get("final_robot")
    if final:
        for side in ("left_wrist", "right_wrist"):
            if side in final:
                result[side+"_error_m"] = float(np.linalg.norm(np.array(final[side])-np.array(observation["robot"][side])))
    frames = imagination["frames"][0] if imagination["frames"] else []
    if frames:
        for view, rgb in observation.get("images", {}).items():
            if view in frames[-1]:
                result[view+"_final_frame_l1"] = float(np.mean(np.abs(frames[-1][view].astype(float)-rgb.astype(float)))/255)
    return result


def summarize_chunk(chunk):
    return {"name": chunk.name, "frame": chunk.frame, "duration_s": chunk.duration, "keyframes": len(chunk.keyframes)}


def run_episode(env, planner, recorder, config, world_model=None, critic=None, memory=None, skills=None, instruction=None):
    """Run one episode. Returns the extra fields merged into episode.json."""
    rng = np.random.default_rng(config.seed)
    instruction = instruction or env.observe(render=False)["instruction"]
    memory_text, skill_text, loaded = "", "", []
    if memory is not None and config.use_memory:
        memory_text = memory.render(memory.search(instruction, task=env.task))
    if skills is not None and config.use_skills:
        skill_text = skills.describe()
        loaded = [skills.load(match["name"]) for match in skills.match(instruction)]
    transcript, steps, error = [], [], None
    counters = {"world_model_calls": 0, "critic_calls": 0, "sim_checks": 0, "invalid_chunks": 0, "gaze_chunks": 0}
    feedback = None
    try:
        for index in range(config.max_chunks):
            observation = env.observe()
            context = {"instruction": instruction, "observation": public_observation(observation),
                       "last_feedback": feedback, "episode_so_far": transcript[-6:],
                       "memory": memory_text, "skills_available": skill_text,
                       "skills_loaded": [{k: s[k] for k in ("name", "body")} for s in loaded]}
            chunk = planner.propose(context, observation)
            if chunk is None:
                break
            step = {"index": index, "proposed": chunk.to_dict(), "rounds": [], "search": None}
            for attempt in range(3):
                try:
                    env.validate_chunk(chunk)
                    break
                except ValueError as exc:
                    counters["invalid_chunks"] += 1
                    if attempt == 2:
                        raise
                    chunk = planner.fix(context, observation, chunk, str(exc))
            imagination = None
            if world_model is not None:
                for round_index in range(config.refine_rounds+1):
                    record = {"chunk": summarize_chunk(chunk)}
                    if config.use_sim_check:
                        check = sim_check(env, chunk)
                        counters["sim_checks"] += 1
                        record["sim_check"] = check["public"]
                        if not check["feasible"] and round_index < config.refine_rounds:
                            chunk = planner.revise(context, observation, chunk, "simulated execution failed: "+str(check["reason"]))
                            step["rounds"].append(record)
                            continue
                    imagination = world_model.imagine(env, chunk, samples=config.samples)
                    counters["world_model_calls"] += imagination["samples"]
                    review = critic.review(instruction, imagination, chunk) if critic is not None else {"accept": True, "feedback": ""}
                    counters["critic_calls"] += critic is not None
                    record.update({"review": review, "disagreement": imagination.get("disagreement")})
                    if config.save_imagination:
                        record["frames"] = _save_imagination(recorder.output/"imagination"/f"step{index:02d}_round{round_index}", imagination)
                    step["rounds"].append(record)
                    if review["accept"] or round_index == config.refine_rounds:
                        break
                    chunk = planner.revise(context, observation, chunk, review["feedback"])
                    env.validate_chunk(chunk)
                if config.local_search > 1:
                    candidates = structured_variants(chunk, config.local_search, rng, config.search_scale_m)
                    feasible, imaginations = [], []
                    for candidate in candidates:
                        try:
                            env.validate_chunk(candidate)
                        except ValueError:
                            continue
                        if config.use_sim_check:
                            counters["sim_checks"] += 1
                            if not sim_check(env, candidate)["feasible"]:
                                continue
                        feasible.append(candidate)
                        imaginations.append(world_model.imagine(env, candidate, samples=config.samples))
                        counters["world_model_calls"] += imaginations[-1]["samples"]
                    if feasible:
                        scores = critic.rank(instruction, imaginations, feasible) if critic is not None else [0.0]*len(feasible)
                        counters["critic_calls"] += critic is not None
                        best = int(np.argmax(scores))
                        chunk, imagination = feasible[best], imaginations[best]
                        step["search"] = {"candidates": [summarize_chunk(c) for c in feasible], "scores": scores, "chosen": best,
                                          "privileged_reference": [i["privileged"].get("goal_error_m") for i in imaginations]}
            counters["gaze_chunks"] += any(k.head is not None for k in chunk.keyframes)
            feedback = recorder.execute(chunk)
            public_feedback = {k: feedback[k] for k in config.feedback_fields if k in feedback}
            after = env.observe()
            step.update({"executed": chunk.to_dict(), "feedback": public_feedback})
            if imagination is not None:
                step["imagined_vs_real"] = compare_imagined_real(imagination, after)
            steps.append(step)
            transcript.append({"chunk": summarize_chunk(chunk), "feedback": public_feedback,
                               "robot_after": after["robot"], "imagined_vs_real": step.get("imagined_vs_real")})
            feedback = public_feedback
            if feedback.get("success") or feedback.get("failure"):
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {"harness": asdict(config), "planner": planner.name, "steps": steps, "run_error": error,
            "world_model": None if world_model is None else world_model.name,
            "critic": None if critic is None else critic.name,
            "critic_privileged": bool(getattr(critic, "privileged", False)),
            "memory_entries_loaded": memory_text.count("\n")+bool(memory_text),
            "skills_loaded": [s["name"] for s in loaded], "llm_calls": planner.calls, "llm_tokens": planner.tokens,
            **counters}


class HarnessTools:
    """Appendix A tool surface, for a tool-calling agent. Specs are JSON schemas;
    `call` dispatches and returns JSON-serialisable results (images summarised)."""

    def __init__(self, env, recorder, world_model=None, critic=None, memory=None, skills=None, gate=None, evaluate_change=None):
        self.env, self.recorder, self.world_model, self.critic = env, recorder, world_model, critic
        self.memory, self.skills, self.gate, self.evaluate_change = memory, skills, gate, evaluate_change

    SPECS = {
        "get_observation": {"description": "Current camera images and robot state.", "properties": {}},
        "sim_check": {"description": "Run a chunk in the simulator for feasibility and whole-body pose; state is restored.",
                      "properties": {"chunk": {"type": "object"}}, "required": ["chunk"]},
        "imagine": {"description": "World-model rollout of a chunk with sample disagreement.",
                    "properties": {"chunk": {"type": "object"}, "samples": {"type": "integer"}}, "required": ["chunk"]},
        "execute": {"description": "Execute a chunk on the robot; returns outcome and new observation.",
                    "properties": {"chunk": {"type": "object"}}, "required": ["chunk"]},
        "run_policy": {"description": "Optional learned teleop-format policy (not configured by default).",
                       "properties": {"name": {"type": "string"}, "instruction": {"type": "string"}, "max_seconds": {"type": "number"}},
                       "required": ["name", "instruction"]},
        "memory_search": {"description": "Search memory notes.", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "memory_write": {"description": "Write a memory note.", "properties": {"kind": {"type": "string"}, "text": {"type": "string"},
                         "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["kind", "text"]},
        "skill_list": {"description": "List skill names and descriptions.", "properties": {}},
        "skill_load": {"description": "Load a skill's full instructions.", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        "skill_write": {"description": "Create or replace a skill.", "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                        "body": {"type": "string"}}, "required": ["name", "description", "body"]},
        "choose_view": {"description": "Rank candidate gaze directions by imagined information gain.",
                        "properties": {"candidates": {"type": "array", "items": {"type": "object", "properties": {
                            "yaw": {"type": "number"}, "pitch": {"type": "number"}}}}, "samples": {"type": "integer"}},
                        "required": ["candidates"]},
        "run_regression": {"description": "Test a proposed self-edit with the testing gate before accepting it.",
                           "properties": {"change": {"type": "object"}}, "required": ["change"]},
    }

    def specs(self):
        return [{"name": name, "description": spec["description"], "input_schema": {"type": "object",
                 "properties": spec["properties"], "required": spec.get("required", [])}} for name, spec in self.SPECS.items()]

    def _chunk(self, value):
        return normalize_chunk(value, self.env.sim.base_pose[2])

    def call(self, name, args):
        if name not in self.SPECS:
            raise ValueError("unknown tool "+name)
        if name == "get_observation":
            observation = self.env.observe()
            return {**public_observation(observation), "images": sorted(observation["images"])}
        if name == "sim_check":
            result = sim_check(self.env, self._chunk(args["chunk"]))
            return {"feasible": result["feasible"], **result["public"]}
        if name == "imagine":
            result = self.world_model.imagine(self.env, self._chunk(args["chunk"]), samples=int(args.get("samples", 1)))
            return {"public": result["public"], "disagreement": result["disagreement"], "frames": len(result["frames"][0]) if result["frames"] else 0}
        if name == "execute":
            outcome = self.recorder.execute(self._chunk(args["chunk"]))
            return {k: outcome[k] for k in ("duration_s", "success", "failure")}
        if name == "choose_view":
            observation = self.env.observe(render=False)
            chunks = [look_chunk(observation, c.get("yaw", 0.0), c.get("pitch", 0.0)) for c in args["candidates"]]
            return {"ranking": rank_views(self.env, self.world_model, chunks, int(args.get("samples", 3)))}
        if name == "run_policy":
            return {"error": "no learned policy is configured; all main results are reported without it"}
        if name == "memory_search":
            return {"entries": self.memory.search(args["query"])}
        if name == "memory_write":
            return {"id": self.memory.write(args["kind"], args["text"], tags=tuple(args.get("tags", ())))}
        if name == "skill_list":
            return {"skills": self.skills.list()}
        if name == "skill_load":
            return self.skills.load(args["name"])
        if name == "skill_write":
            self.skills.write(args["name"], args["description"], args["body"])
            return {"written": args["name"]}
        from .improvement import parse_changes
        change = parse_changes({"changes": [args["change"]]})[0]
        return self.gate.review(change, *self.evaluate_change).to_dict()


def reflect(client, record, memory, skills, gate, regression_set=(), trigger_set=()):
    """Middle loop: ask the LLM for memory/skill changes and pass each through the
    testing gate (TestingGate(ablate=True) is the 'no testing gate' ablation)."""
    from .improvement import parse_changes, reflection_prompt
    entries = memory.search(record.get("task", ""), task=record.get("task")) if memory is not None else []
    prompt = reflection_prompt(record, entries, skills.list() if skills is not None else [])
    result = client.complete("You reflect on a robot episode and propose improvements to memory and skills. Return JSON only.",
                             [{"type": "text", "text": prompt}], purpose="reflect")
    changes = parse_changes(parse_json(result["text"]))
    decisions = [gate.review(change, regression_set, trigger_set or (record.get("task"),)).to_dict() for change in changes]
    return {"changes": len(changes), "decisions": decisions, "llm_record": result["record"]}
