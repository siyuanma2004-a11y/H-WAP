from dataclasses import replace
import numpy as np
import pytest
import mujoco
from hwap.actions import CommandChunk, Keyframe, Pose
from hwap.environment import HumanoidEnvironment
from hwap.planning import OracleWorldModel
from hwap.recording import EpisodeRecorder, replay_episode


def hold(env,duration=0.2):
    pose=Pose(tuple(env.sim.palm_position("right")))
    return CommandChunk("hold",(Keyframe(0,right_wrist=pose),Keyframe(duration,right_wrist=pose)),frame="world")


def test_action_schema_rejects_unsafe_and_ambiguous_input():
    with pytest.raises(ValueError,match="finite"):
        Pose((float("nan"),0,1))
    with pytest.raises(ValueError,match="unit norm"):
        Pose((0,0,1),(0,0,0,0))
    with pytest.raises(ValueError,match="velocity"):
        Keyframe(0,velocity=(1,0,0))
    with pytest.raises(ValueError,match="initialized"):
        CommandChunk("bad",(Keyframe(0),Keyframe(1,right_wrist=Pose((0.3,0,1)))))


def test_quaternion_interpolation_and_sparse_hold():
    first=Pose((0.2,0,1),(1,0,0,0))
    second=Pose((0.4,0,1),(-1,0,0,0))
    c=CommandChunk("x",(Keyframe(0,right_wrist=first),Keyframe(1,right_wrist=second),Keyframe(2)))
    np.testing.assert_allclose(c.sample(0.5).right_wrist.position,(0.3,0,1))
    np.testing.assert_allclose(c.sample(0.5).right_wrist.quaternion,(1,0,0,0))
    assert c.sample(1.5).right_wrist.position==second.position


def test_egocentric_camera_is_rigidly_attached_to_robot():
    with HumanoidEnvironment() as env:
        env.reset(1)
        cam=env.sim._required_id(mujoco.mjtObj.mjOBJ_CAMERA,"egocentric_camera")
        assert env.sim.model.cam_bodyid[cam]!=0
        before=env.sim.data.cam_xpos[cam].copy()
        env.sim.data.qpos[0]+=0.2
        mujoco.mj_forward(env.sim.model,env.sim.data)
        np.testing.assert_allclose(env.sim.data.cam_xpos[cam]-before,(0.2,0,0),atol=1e-10)


def test_reach_marker_is_inside_egocentric_camera_frame():
    with HumanoidEnvironment() as env:
        for seed in range(5):
            env.reset(seed)
            cam=env.camera_parameters()['ego']
            local=(env.goal-np.array(cam['position']))@np.array(cam['rotation_world_from_camera'])
            assert -local[2]>0
            f=cam['intrinsics'][0][0]
            u=cam['width']/2+f*local[0]/(-local[2])
            v=cam['height']/2-f*local[1]/(-local[2])
            assert 8<u<cam['width']-8 and 8<v<cam['height']-8


def test_oracle_restores_state_and_reproduces_future():
    with HumanoidEnvironment() as env:
        env.reset(4)
        before=env.snapshot()
        world=OracleWorldModel()
        first=world.imagine(env,hold(env),render=False)
        after=env.snapshot()
        np.testing.assert_array_equal(before["physics"],after["physics"])
        for a,b in zip(before["controller"][1:],after["controller"][1:]):
            np.testing.assert_array_equal(a,b)
        assert before["task_state"]==after["task_state"]
        second=world.imagine(env,hold(env),render=False)
        assert first["outcome"]==second["outcome"]


def test_rejected_action_does_not_mutate_simulator():
    with HumanoidEnvironment(budget_s=0.1) as env:
        env.reset(2)
        before=env.snapshot()
        with pytest.raises(ValueError,match="budget"):
            env.execute(hold(env,0.2))
        np.testing.assert_array_equal(before["physics"],env.snapshot()["physics"])


def test_episode_roundtrip_replays_physics_and_records_calibration(tmp_path):
    with HumanoidEnvironment(views=("ego","workspace"),width=64,height=48) as env:
        env.reset(5)
        recorder=EpisodeRecorder(env,tmp_path/"ep")
        recorder.execute(hold(env,0.2))
        result=recorder.finish()
        assert result["frames"][0]["cameras"]["ego"]["width"]==64
        assert set(env.observe(render=False)["robot"])=={"base_pose","pelvis_height","left_wrist","right_wrist"}
    replay=replay_episode(tmp_path/"ep")
    assert replay["max_qpos_error"]<1e-9


def test_sonic_reference_is_invariant_to_global_heading_and_translation():
    # Test the upstream teleop frame convention without loading model weights.
    from hwap.sonic import SonicOnnxController,quaternion_matrix
    from agent_g1.simulation import G1Simulation
    sim=G1Simulation()
    c=object.__new__(SonicOnnxController)
    c.sim=sim
    c.reference_positions=np.array([[.3,.2,1.0],[.3,-.2,1.0],[0,0,1.4]])
    c.reference_quaternions=np.array([[1.,0,0,0]]*3)
    c.reference_heading=0
    initial=c.encoder_observation(Keyframe(0))
    yaw=.7;q=np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)])
    rotation=quaternion_matrix(q);shift=np.array([.4,-.2,0])
    sim.data.qpos[:3]=rotation@sim.data.qpos[:3]+shift
    sim.data.qpos[3:7]=q
    mujoco.mj_forward(sim.model,sim.data)
    c.reference_positions=c.reference_positions@rotation.T+shift
    c.reference_quaternions=np.array([q]*3)
    c.reference_heading=yaw
    transformed=c.encoder_observation(Keyframe(0))
    np.testing.assert_allclose(initial,transformed,atol=1e-6)
    assert transformed[0]==1 and not transformed[1:4].any()


def test_recorded_source_mismatch_is_rejected_before_replay(tmp_path):
    import json
    p=tmp_path/'episode';p.mkdir()
    (p/'episode.json').write_text(json.dumps({'source_sha256':{'hwap/environment.py':'wrong'}}))
    with pytest.raises(ValueError,match='source differs'):
        replay_episode(p)
