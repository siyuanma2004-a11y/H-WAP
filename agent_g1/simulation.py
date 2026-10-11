"""MuJoCo simulation wrapper for the migrated G1 tabletop scene."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = REPO_ROOT / "models" / "unitree_g1" / "agent_g1_tabletop.xml"


def yaw_from_quaternion(quaternion: np.ndarray) -> float:
    """Extract world yaw from a MuJoCo ``[w, x, y, z]`` quaternion."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    w, x, y, z = quaternion
    return float(
        np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    )


class G1Simulation:
    """Small named-entity wrapper around the G1 table scene."""

    def __init__(self) -> None:
        self.model = self._load_model()
        self.data = mujoco.MjData(self.model)
        self.pelvis_body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        self.left_palm_site_id = self._required_id(
            mujoco.mjtObj.mjOBJ_SITE, "left_palm_site"
        )
        self.right_palm_site_id = self._required_id(
            mujoco.mjtObj.mjOBJ_SITE, "right_palm_site"
        )

    @staticmethod
    def _load_model() -> mujoco.MjModel:
        """Compile the scene and attach Cartesian palm references."""
        spec = mujoco.MjSpec.from_file(str(SCENE_PATH))
        for side in ("left", "right"):
            wrist = spec.body(f"{side}_wrist_yaw_link")
            wrist.add_site(
                name=f"{side}_palm_site",
                pos=[0.10, 0.0, 0.0],
                size=[0.008, 0.008, 0.008],
                rgba=[1.0, 0.2, 0.1, 1.0],
            )
        return spec.compile()

    def _required_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id == -1:
            raise ValueError(f"G1 scene is missing {name}")
        return object_id

    def reset(self, seed: int | None = None) -> None:
        """Reset to the neutral scene home keyframe."""
        if seed is not None:
            np.random.seed(seed)
        home_key_id = self._required_id(mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_key_id)
        mujoco.mj_forward(self.model, self.data)

    def set_initial_base_pose(
        self, base_x: float, base_y: float, base_yaw: float
    ) -> None:
        """Set a finite initial floating-base pose before a rollout."""
        values = np.asarray([base_x, base_y, base_yaw], dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("initial base pose must be finite [x, y, yaw]")
        self.data.qpos[0:2] = values[:2]
        rotation = np.asarray(
            [np.cos(base_yaw / 2.0), 0.0, 0.0, np.sin(base_yaw / 2.0)]
        )
        self.data.qpos[3:7] = rotation
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl: Sequence[float]) -> None:
        """Advance one physics step with all 43 position commands."""
        ctrl_array = np.asarray(ctrl, dtype=np.float64)
        if ctrl_array.shape != (self.model.nu,):
            raise ValueError(
                f"expected ctrl shape {(self.model.nu,)}, got {ctrl_array.shape}"
            )
        self.data.ctrl[:] = ctrl_array
        mujoco.mj_step(self.model, self.data)

    @property
    def base_pose(self) -> tuple[float, float, float]:
        """Return measured world-frame ``[x, y, yaw]``."""
        position = self.data.xpos[self.pelvis_body_id]
        return float(position[0]), float(position[1]), yaw_from_quaternion(
            self.data.qpos[3:7]
        )

    @property
    def pelvis_height(self) -> float:
        return float(self.data.xpos[self.pelvis_body_id, 2])

    def palm_position(self, side: str) -> np.ndarray:
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        site_id = (
            self.left_palm_site_id
            if side == "left"
            else self.right_palm_site_id
        )
        return self.data.site_xpos[site_id].copy()

    def joint_positions(self, joint_names: Sequence[str]) -> np.ndarray:
        """Read named scalar joints in the supplied order."""
        values = []
        for name in joint_names:
            joint_id = self._required_id(mujoco.mjtObj.mjOBJ_JOINT, name)
            values.append(self.data.qpos[self.model.jnt_qposadr[joint_id]])
        return np.asarray(values, dtype=np.float64)

    def _contact_bodies(self) -> list[tuple[str, str, float]]:
        contacts = []
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            body_1 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.model.geom_bodyid[contact.geom1],
            )
            body_2 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.model.geom_bodyid[contact.geom2],
            )
            contacts.append((body_1, body_2, float(contact.dist)))
        return contacts

    def table_collision_contacts(self) -> list[tuple[str, str, float]]:
        """Return contacts between the table and any G1 body."""
        contacts = []
        for body_1, body_2, distance in self._contact_bodies():
            if "table" not in {body_1, body_2}:
                continue
            other_body = body_2 if body_1 == "table" else body_1
            if other_body and (
                other_body.startswith(("left_", "right_", "pelvis"))
            ):
                contacts.append((body_1, body_2, distance))
        return contacts

    def non_hand_object_collision_contacts(
        self,
    ) -> list[tuple[str, str, float]]:
        """Return object contacts outside either Dex3 hand."""
        contacts = []
        for body_1, body_2, distance in self._contact_bodies():
            if "target_object" not in {body_1, body_2}:
                continue
            other_body = body_2 if body_1 == "target_object" else body_1
            if other_body in {None, "target_object", "table"}:
                continue
            if other_body.startswith(("left_hand_", "right_hand_")):
                continue
            contacts.append((body_1, body_2, distance))
        return contacts
