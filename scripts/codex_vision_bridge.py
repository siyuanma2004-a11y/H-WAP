#!/usr/bin/env python3
"""Private loopback chat-completions adapter for locally authenticated Codex.

No auth.json parsing, credential copying, or undocumented backend requests.
Only codex exec, image inputs and schema-constrained final output are used.
"""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import threading
import time

MAX_BODY=4*1024*1024
MODEL='gpt-6.1-sol'
DISABLED_FEATURES=('shell_tool','unified_exec','apps','plugins','browser_use','computer_use',
    'multi_agent','code_mode_host','skill_search','view_image','sleep_tool','goals','workspace_dependencies')


def extract_inputs(messages,directory):
    texts,images=[],[]
    for message in messages:
        role=message.get('role')
        if role not in ('system','user','assistant'):
            raise ValueError('unsupported message role')
        content=message.get('content')
        if isinstance(content,str):
            texts.append(role+': '+content)
            continue
        if not isinstance(content,list):
            raise ValueError('content must be text or multimodal blocks')
        for block in content:
            if block.get('type')=='text':
                texts.append(role+': '+str(block['text']))
            elif block.get('type')=='image_url':
                url=block['image_url']['url']
                if not url.startswith('data:image/png;base64,'):
                    raise ValueError('only inline PNG observations are accepted')
                data=base64.b64decode(url.split(',',1)[1],validate=True)
                if not data.startswith(b'\x89PNG\r\n\x1a\n') or len(data)>1024*1024:
                    raise ValueError('invalid or oversized observation image')
                if len(images)>=3:
                    raise ValueError('at most three observation views are accepted')
                path=directory/f'observation_{len(images)}.png'
                path.write_bytes(data);images.append(path)
            else:
                raise ValueError('unsupported content block')
    if not images or sum(len(t) for t in texts)>24000:
        raise ValueError('images required and prompt text must be at most 24000 characters')
    return texts,images


def cli_command(codex,directory,schema,images,model=MODEL,reasoning='low',transport='http'):
    if model!=MODEL or transport not in ('http','auto'):
        raise ValueError('unsupported model or transport')
    cmd=[codex,'--no-daemon','-a','never','exec','--ephemeral','--ignore-user-config',
        '--skip-git-repo-check','--sandbox','read-only','--model',model,
        '-c',f'model_reasoning_effort="{reasoning}"','-c','web_search="disabled"',
        '-c','features.skip_host_skill_discovery=true']
    if transport=='http':
        # A custom provider avoids overriding reserved built-in IDs. The CLI
        # handles its normal cached OpenAI/ChatGPT auth; no base URL or token is set.
        for setting in ('model_provider="hwap_chatgpt_http"',
                'model_providers.hwap_chatgpt_http.name="H-WAP ChatGPT HTTP"',
                'model_providers.hwap_chatgpt_http.requires_openai_auth=true',
                'model_providers.hwap_chatgpt_http.wire_api="responses"',
                'model_providers.hwap_chatgpt_http.supports_websockets=false'):
            cmd+=['-c',setting]
    for feature in DISABLED_FEATURES:cmd+=['-c',f'features.{feature}=false']
    cmd+=['--output-schema',str(schema),'--output-last-message',str(directory/'response.json'),'--json']
    for image in images:cmd+=['--image',str(image)]
    return cmd+['-']


def event_usage(raw):
    usage={};tool_calls=[]
    for line in raw.splitlines():
        try:event=json.loads(line)
        except json.JSONDecodeError:continue
        if event.get('type')=='turn.completed':usage=event.get('usage',{})
        item=event.get('item',{})
        if item.get('type') not in (None,'error','agent_message','reasoning'):
            tool_calls.append(item.get('type'))
    if tool_calls:
        raise RuntimeError('model attempted non-inference tools: '+','.join(tool_calls))
    if not usage:
        raise RuntimeError('Codex did not complete a model turn')
    return usage


class Backend:
    def __init__(self,args):
        self.codex=shutil.which(args.codex)
        if not self.codex:raise RuntimeError('Codex CLI is not available')
        self.schema=Path(args.schema).resolve()
        json.loads(self.schema.read_text())
        self.token=Path(args.token_file).read_text().strip()
        if len(self.token)<32:raise ValueError('bridge token must have at least 32 characters')
        self.logs=Path(args.log_dir).resolve();self.logs.mkdir(parents=True,exist_ok=True)
        self.timeout,self.reasoning=args.timeout_s,args.reasoning
        self.transport=getattr(args,'transport','http')
        self.lock=threading.Lock()
        self.calls=0

    def infer(self,body):
        if body.get('model')!=MODEL:raise ValueError('this bridge is pinned to '+MODEL)
        if not self.lock.acquire(blocking=False):raise RuntimeError('one model request is already running')
        directory=self.logs/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'_'+secrets.token_hex(3))
        try:
            directory.mkdir(mode=0o700)
            texts,images=extract_inputs(body.get('messages',[]),directory)
            prompt=('PURE VISUAL INFERENCE: use only the supplied observation text and attached images. '
                'Do not use tools, read other files, browse, or inspect this machine. This is not a coding task. '
                'Generate one simulated robot command that satisfies the output schema. '
                'No hidden object/goal coordinates are available. The images are ordered as the Camera labels in the text.\n\n'+'\n\n'.join(texts))
            (directory/'prompt.txt').write_text(prompt)
            command=cli_command(self.codex,directory,self.schema,images,reasoning=self.reasoning,transport=self.transport)
            started=time.monotonic()
            env=os.environ.copy()
            for name in ('OPENAI_API_KEY','CODEX_API_KEY'):env.pop(name,None)
            process=subprocess.Popen(command,cwd=directory,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,text=True,env=env,start_new_session=True)
            try:stdout,stderr=process.communicate(prompt,timeout=self.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.communicate()
                raise RuntimeError('Codex inference exceeded its wall-time limit')
            (directory/'events.jsonl').write_text(stdout)
            (directory/'stderr.log').write_text(stderr)
            if process.returncode:raise RuntimeError('Codex inference failed; inspect the local call log')
            usage=event_usage(stdout)
            chunk=json.loads((directory/'response.json').read_text())
            content=json.dumps(chunk,allow_nan=False)
            elapsed=time.monotonic()-started;self.calls+=1
            metadata={'model':MODEL,'reasoning_effort':self.reasoning,'elapsed_s':elapsed,'tool_calls':0,
                'image_count':len(images),'image_sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in images],
                'usage':usage,'local_call_id':directory.name,'transport':'codex_exec_chatgpt'}
            metadata.update({'responses_transport':self.transport,
                'model_provider':'hwap_chatgpt_http' if self.transport=='http' else 'openai',
                'bridge_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'schema_sha256':hashlib.sha256(self.schema.read_bytes()).hexdigest()})
            (directory/'call.json').write_text(json.dumps(metadata,indent=2)+'\n')
            print(json.dumps({'call':directory.name,'model':MODEL,'seconds':round(elapsed,2),'images':len(images)}),flush=True)
            prompt_tokens=int(usage.get('input_tokens',0));completion_tokens=int(usage.get('output_tokens',0))
            return {'id':directory.name,'object':'chat.completion','model':MODEL,
                'choices':[{'index':0,'message':{'role':'assistant','content':content},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':prompt_tokens,'completion_tokens':completion_tokens,
                    'total_tokens':prompt_tokens+completion_tokens},'hwap_bridge':metadata}
        finally:self.lock.release()


def make_handler(backend):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def send_json(self,status,value):
            payload=json.dumps(value,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            if self.path!='/health':return self.send_json(404,{'error':'unknown route'})
            self.send_json(200,{'status':'ready','model':MODEL,'completed_calls':backend.calls})
        def do_POST(self):
            if self.path!='/v1/chat/completions':return self.send_json(404,{'error':'unknown route'})
            if not secrets.compare_digest(self.headers.get('Authorization',''),'Bearer '+backend.token):
                return self.send_json(401,{'error':'bridge authentication required'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=MAX_BODY:raise ValueError('request body outside allowed size')
                result=backend.infer(json.loads(self.rfile.read(size)))
                self.send_json(200,result)
            except (ValueError,KeyError,TypeError) as error:self.send_json(400,{'error':'invalid request: '+str(error)})
            except (RuntimeError,OSError) as error:self.send_json(502,{'error':str(error)})
    return Handler


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=18561)
    parser.add_argument('--codex',default='codex')
    parser.add_argument('--schema',default=str(Path(__file__).resolve().parents[1]/'configs/action_chunk.schema.json'))
    parser.add_argument('--token-file',required=True)
    parser.add_argument('--log-dir',default='runs/codex_bridge')
    parser.add_argument('--timeout-s',type=float,default=180)
    parser.add_argument('--reasoning',choices=['low','medium','high'],default='low')
    parser.add_argument('--transport',choices=['http','auto'],default='http')
    args=parser.parse_args();backend=Backend(args)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(backend))
    print(json.dumps({'status':'ready','listen':f'127.0.0.1:{args.port}','model':MODEL}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
