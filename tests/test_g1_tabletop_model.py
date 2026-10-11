from pathlib import Path

import mujoco
import numpy as np
import pytest


MODEL_DIR = Path(__file__).parents[1] / "models" / "unitree_g1"
BASE_MODEL = MODEL_DIR / "g1.xml"
HAND_MODEL = MODEL_DIR / "g1_with_hands.xml"
TABLETOP_MODEL = MODEL_DIR / "agent_g1_tabletop.xml"


def object_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, obj_type, name)
    if object_id == -1:
        pytest.fail(f"missing {obj_type.name.lower()}: {name}")
    return object_id


def test_base_g1_model_compiles_with_official_actuators():
    model = mujoco.MjModel.from_xml_path(str(BASE_MODEL))

    assert model.nu == 29
    assert object_id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    ) != -1


def test_hand_model_compiles_with_three_finger_hand_actuators():
    model = mujoco.MjModel.from_xml_path(str(HAND_MODEL))

    assert model.nq == 50
    assert model.nv == 49
    assert model.nu == 43
    for side in ("left", "right"):
        object_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_hand_thumb_0_joint"
        )
        object_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_hand_middle_1_joint"
        )


def test_tabletop_scene_compiles_with_g1_hands_table_and_object():
    model = mujoco.MjModel.from_xml_path(str(TABLETOP_MODEL))

    for body_name in ("pelvis", "floor", "table", "target_object"):
        object_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    for geom_name in ("table_top", "target_object_geom"):
        object_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    for camera_name in ("egocentric_camera", "side_camera", "workspace_camera"):
        object_id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

    assert model.nu == 43
    assert model.nq == 57
    assert model.nv == 55


def test_g1_has_two_official_three_finger_dexterous_hands():
    model = mujoco.MjModel.from_xml_path(str(TABLETOP_MODEL))

    expected_fingers = {
        "thumb_0",
        "thumb_1",
        "thumb_2",
        "index_0",
        "index_1",
        "middle_0",
        "middle_1",
    }
    for side in ("left", "right"):
        hand_joints = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(model.njnt)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            .startswith(f"{side}_hand_")
        }
        hand_actuators = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            for actuator_id in range(model.nu)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            .startswith(f"{side}_hand_")
        }

        assert len(hand_joints) == 7
        assert len(hand_actuators) == 7
        assert {
            joint_name.removeprefix(f"{side}_hand_").removesuffix("_joint")
            for joint_name in hand_joints
        } == expected_fingers


def test_home_keyframe_steps_without_instability():
    model = mujoco.MjModel.from_xml_path(str(TABLETOP_MODEL))
    data = mujoco.MjData(model)
    keyframe_id = object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)

    pelvis_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    object_body_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    pelvis_height = data.xpos[pelvis_id][2]
    object_height = data.xpos[object_body_id][2]

    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    assert 0.65 < pelvis_height < 0.92
    assert object_height > 0.73
