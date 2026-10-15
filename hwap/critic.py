"""Critics judge imagined rollouts.

PrivilegedCritic reads evaluator truth and is only a labelled upper reference.
VLMCritic sees imagined frames, the instruction and public outcome only.
"""
import json
from .llm import parse_json


REVIEW_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["accept", "score", "feedback", "issues"],
    "properties": {"accept": {"type": "boolean"}, "score": {"type": "number"},
                   "feedback": {"type": "string"}, "issues": {"type": "array", "items": {"type": "string"}}}}
RANK_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["scores", "reason"],
    "properties": {"scores": {"type": "array", "items": {"type": "number"}}, "reason": {"type": "string"}}}


def key_frames(imagination, count=3, sample=0):
    """First, middle and last imagined frames of one sample, as content blocks."""
    frames = imagination["frames"][sample] if imagination["frames"] else []
    if not frames:
        return []
    picks = sorted({0, len(frames)//2, len(frames)-1})[:count]
    blocks = []
    for i in picks:
        t = imagination["timestamps"][i] if i < len(imagination["timestamps"]) else None
        for view in imagination["views"]:
            blocks.append({"type": "image", "rgb": frames[i][view], "label": f"{view} imagined t={t:.2f}s" if t is not None else view})
    return blocks


class PrivilegedCritic:
    name = "privileged_geometric_reference"
    privileged = True

    def review(self, task, imagination, chunk=None):
        p = imagination.get("privileged", {})
        error = p.get("goal_error_m", float("inf"))
        failure = p.get("failure")
        return {"accept": bool(p.get("success")) or (failure is None and error < 0.045), "score": -(error+(10 if failure else 0)),
                "feedback": f"privileged goal error {error:.3f} m, failure={failure}", "issues": []}

    def rank(self, task, imaginations, chunks=None):
        return [self.review(task, i)["score"] for i in imaginations]


class VLMCritic:
    name = "vlm_critic"
    privileged = False

    def __init__(self, client, guidance=""):
        self.client, self.guidance = client, guidance
        self.system = ("You are the critic in a humanoid world-action planner. You watch imagined camera frames of "
            "a Unitree G1 executing a short command chunk and judge whether it makes progress on the instruction. "
            "Look for contacts, collisions, the hand reaching the right spot, occlusion of the target and falls. "
            "Imagined frames may be wrong; say when the view cannot show what matters. Return JSON only.\n"+guidance)

    def review(self, task, imagination, chunk=None):
        content = [{"type": "text", "text": json.dumps({"instruction": task, "chunk": None if chunk is None else chunk.to_dict(),
            "imagined_outcome": imagination["public"], "sample_disagreement": imagination.get("disagreement"),
            "request": "accept if executing this chunk is likely to succeed or clearly advance the task; "
                       "score in [0,1]; feedback must say concretely how to change the keyframes"})}]
        content += key_frames(imagination)
        result = self.client.complete(self.system, content, REVIEW_SCHEMA, purpose="critic_review")
        review = parse_json(result["text"])
        review["score"] = float(min(1, max(0, review["score"])))
        return review

    def rank(self, task, imaginations, chunks=None):
        content = [{"type": "text", "text": json.dumps({"instruction": task, "candidates": len(imaginations),
            "request": "score every candidate in [0,1] in the given order; higher is better"})}]
        for index, imagination in enumerate(imaginations):
            content.append({"type": "text", "text": json.dumps({"candidate": index, "imagined_outcome": imagination["public"],
                "chunk": None if chunks is None else chunks[index].to_dict()})})
            content += key_frames(imagination, count=2)
        result = self.client.complete(self.system, content, RANK_SCHEMA, purpose="critic_rank")
        scores = parse_json(result["text"])["scores"]
        if len(scores) != len(imaginations):
            raise ValueError("critic returned the wrong number of scores")
        return [float(s) for s in scores]
