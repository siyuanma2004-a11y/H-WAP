"""Registered tasks: seeded layouts, scene props, instructions and splits.

Layouts are privileged evaluator state. Agents receive only the instruction,
the public description and their own camera images.
"""
from dataclasses import dataclass, field
import mujoco
import numpy as np


SPLITS = ("seen", "new_layout", "compositional", "zero_shot")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    family: str  # "reach": palm touches the goal marker; "pick_place": cube onto the marker
    instruction: str
    split: str
    goal: tuple
    goal_jitter: tuple  # uniform half-widths for x, y[, z]
    object_xy: tuple = (0.45, 0.0)
    object_jitter: tuple = (0.015, 0.015)
    hand: str = "right"
    needs_active_perception: bool = False
    # (name, offset from the goal, box half-size); props move with the goal.
    occluders: tuple = ()
    description: str = ""
    tags: tuple = field(default_factory=tuple)

    def __post_init__(self):
        if self.family not in {"reach", "pick_place"} or self.split not in SPLITS or self.hand not in {"left", "right"}:
            raise ValueError("invalid task family, split or hand")

    def sample_layout(self, rng):
        # Draw order matches the original reach/pick_place reset exactly.
        goal = np.array(self.goal, dtype=float)
        jitter = np.asarray(self.goal_jitter, dtype=float)
        goal[:len(jitter)] += rng.uniform(-jitter, jitter)
        obj = np.asarray(self.object_xy, dtype=float)+rng.uniform(-np.asarray(self.object_jitter), np.asarray(self.object_jitter))
        return {"goal": goal.tolist(), "object_xy": obj.tolist(),
                "occluders": {name: (goal+np.asarray(offset)).tolist() for name, offset, _ in self.occluders}}

    def scene_hook(self, spec):
        for name, offset, size in self.occluders:
            body = spec.worldbody.add_body(name=name, pos=list(np.asarray(self.goal)+np.asarray(offset)))
            body.add_geom(name=name+"_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size),
                          rgba=[0.55, 0.45, 0.35, 1], friction=[1, 0.005, 0.0001])

    def public(self):
        return {"name": self.name, "instruction": self.instruction, "description": self.description}


TASKS = {}


def register(spec):
    if spec.name in TASKS:
        raise ValueError("duplicate task: "+spec.name)
    TASKS[spec.name] = spec
    return spec


def get_task(name):
    if name not in TASKS:
        raise ValueError("unknown task: "+str(name))
    return TASKS[name]


def task_splits():
    """{split: [task names]} plus per-task metadata for evaluation tables."""
    table = {split: [] for split in SPLITS}
    for spec in TASKS.values():
        table[spec.split].append(spec.name)
    return {"splits": table, "tasks": {name: {"split": s.split, "family": s.family, "hand": s.hand,
            "needs_active_perception": s.needs_active_perception, "tags": list(s.tags)} for name, s in TASKS.items()}}


REACH = "Touch the green marker with your right palm"
register(TaskSpec("reach", "reach", REACH, "seen", (0.30, -0.25, 1.05), (0.015, 0.015),
                  description="Marker floats above the table, visible from the default view.", tags=("reach",)))
register(TaskSpec("pick_place", "pick_place", "Move the blue cube onto the green marker", "seen",
                  (0.60, -0.10, 0.775), (0.015, 0.015), description="Cube and goal on the table.", tags=("pick_place",)))
register(TaskSpec("reach_wide", "reach", REACH, "new_layout", (0.30, -0.25, 1.05), (0.06, 0.06, 0.05),
                  description="Reach with a wider marker layout than the seen split.", tags=("reach",)))
# Right of the default ego frustum; turning the torso (waist yaw) brings it into view.
register(TaskSpec("reach_hidden_side", "reach", "Find the green marker and touch it with your right palm", "seen",
                  (0.16, -0.40, 1.00), (0.02, 0.02, 0.02), needs_active_perception=True,
                  description="The marker may start outside the camera view.", tags=("reach", "gaze_search")))
register(TaskSpec("reach_hidden_side_left", "reach", "Find the green marker and touch it with your left palm",
                  "zero_shot", (0.16, 0.42, 1.00), (0.02, 0.02, 0.02), hand="left", needs_active_perception=True,
                  description="The marker may start outside the camera view.", tags=("reach", "gaze_search", "left_hand")))
# A post at the table edge hides a marker from the upright view. Leaning forward
# (waist pitch >= ~0.15) moves the camera past it; turning alone does not.
# The arm reaches it around the right side of the post (lift, then forward).
register(TaskSpec("reach_occluded", "reach", "Find the green marker behind the post and touch it with your right palm",
                  "seen", (0.40, -0.22, 0.95), (0.015, 0.015), needs_active_perception=True,
                  occluders=(("occluder", (-0.10, 0.09, 0.0075), (0.02, 0.035, 0.2125)),),
                  description="A post on the table can hide the marker from the current view.",
                  tags=("reach", "occlusion")))
