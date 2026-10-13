"""Oracle branching restores physics, controller history, RNG and evaluator."""
from dataclasses import replace
import numpy as np
from .actions import CommandChunk, Pose


class OracleWorldModel:
    name = "oracle_simulator"

    def imagine(self, env, chunk, render=True):
        state = env.snapshot()
        try:
            outcome = env.execute(chunk)
            observation = env.observe(render=render)
            return {"outcome": outcome, "observation": observation}
        finally:
            env.restore(state)


def local_variants(chunk, offsets=(-0.03, 0.0, 0.03)):
    candidates = []
    for offset in offsets:
        keys = []
        for i, key in enumerate(chunk.keyframes):
            pose = key.right_wrist
            if pose is not None and i > 0:
                p = np.array(pose.position)
                p[0] += offset
                pose = Pose(tuple(p), pose.quaternion)
            keys.append(replace(key, right_wrist=pose))
        candidates.append(replace(chunk, name=f"{chunk.name}_dx{offset:+.3f}", keyframes=tuple(keys)))
    return candidates


def choose_oracle(env, candidates):
    """Privileged geometric scorer: a planning upper reference, not a VLM result."""
    world = OracleWorldModel()
    records = []
    for chunk in candidates:
        imagined = world.imagine(env, chunk, render=False)
        result = imagined["outcome"]
        score = result["goal_error_m"] + (10 if result["failure"] else 0)
        records.append({"chunk": chunk.to_dict(), "score": score, "outcome": result})
    best = int(np.argmin([r["score"] for r in records]))
    return candidates[best], records
