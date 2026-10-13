import base64
import io
import json
from pathlib import Path
import numpy as np
import pytest
from hwap.agents import VisionChatAgent
from hwap.environment import HumanoidEnvironment
from hwap.images import png_bytes
from scripts.codex_vision_bridge import cli_command,event_usage,extract_inputs


def test_bridge_accepts_only_inline_observation_images(tmp_path):
    with pytest.raises(ValueError,match='inline PNG'):
        extract_inputs([{'role':'user','content':[{'type':'image_url','image_url':{'url':'https://example.com/image.png'}}]}],tmp_path)
    encoded=base64.b64encode(png_bytes(np.zeros((16,16,3),np.uint8))).decode()
    texts,images=extract_inputs([{'role':'user','content':[{'type':'text','text':'Camera: ego'},
        {'type':'image_url','image_url':{'url':'data:image/png;base64,'+encoded}}]}],tmp_path)
    assert texts==['user: Camera: ego'] and images[0].read_bytes().startswith(b'\x89PNG')


def test_model_invocation_is_pinned_ephemeral_and_read_only(tmp_path):
    command=cli_command('codex',tmp_path,tmp_path/'schema.json',[tmp_path/'ego.png'])
    assert command[command.index('--model')+1]=='gpt-6.1-sol'
    assert '--ephemeral' in command and '--ignore-user-config' in command
    assert command[command.index('--sandbox')+1]=='read-only'
    assert 'features.shell_tool=false' in command and 'features.multi_agent=false' in command
    assert command[-1]=='-'
    assert not any('auth.json' in arg for arg in command)


def test_tool_activity_and_incomplete_model_turn_are_rejected():
    good=json.dumps({'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':5}})
    assert event_usage(good)['output_tokens']==5
    with pytest.raises(RuntimeError,match='non-inference'):
        event_usage(json.dumps({'type':'item.completed','item':{'type':'command_execution'}})+'\n'+good)
    with pytest.raises(RuntimeError,match='did not complete'):
        event_usage('{}')


def test_visual_request_excludes_privileged_feedback_and_records_provenance(tmp_path,monkeypatch):
    config=tmp_path/'agent.json'
    config.write_text(json.dumps({'base_url':'http://127.0.0.1:18561/v1','model':'gpt-6.1-sol',
        'provider':'codex_cli_bridge','prompt_profile':'reach_stationary_world'}))
    with HumanoidEnvironment() as env:
        env.reset(0)
        observation=env.observe(render=False)
        observation['goal']=env.goal.tolist()  # Future evaluator additions must not cross the boundary.
        observation['images']={'ego':np.zeros((16,16,3),np.uint8)}
        wrist=observation['robot']['right_wrist']
        raw={'schema_version':1,'name':'hold','frame':'world','keyframes':[
            {'t':0,'right_wrist':{'position':wrist}}, {'t':.2,'right_wrist':{'position':wrist}}]}
        def response(request,timeout):
            request_body=json.loads(request.data)
            public=json.loads(request_body['messages'][1]['content'][0]['text'])
            assert 'goal' not in public['observation'] and 'object_position' not in public['observation']
            assert set(public['feedback'])=={'success','failure','duration_s'}
            assert request_body['model']=='gpt-6.1-sol'
            return io.BytesIO(json.dumps({'choices':[{'message':{'content':json.dumps(raw)}}],
                'usage':{'total_tokens':15},'hwap_bridge':{'tool_calls':0,'model':'gpt-6.1-sol'}}).encode())
        monkeypatch.setattr('hwap.agents.urllib.request.urlopen',response)
        agent=VisionChatAgent(config)
        chunk=agent.propose(observation,{'duration_s':.2,'success':False,'failure':None,'goal_error_m':.5,'lifted':False})
        assert agent.calls==1 and agent.tokens==15 and agent.name=='codex_cli_chatgpt'
        assert agent.history[0]['bridge']['tool_calls']==0
        assert len(agent.history[0]['image_sha256']['ego'])==64
        env.validate_chunk(chunk)
