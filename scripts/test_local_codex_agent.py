#!/usr/bin/env python3
"""Mac-side launcher: local ChatGPT-auth Codex + SSH-forwarded train rollout."""
import argparse
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import shlex
import subprocess
import tempfile
import threading
from types import SimpleNamespace
from codex_vision_bridge import Backend,make_handler


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ssh-host',default='train')
    parser.add_argument('--seeds',type=int,nargs='+',default=[0])
    parser.add_argument('--views',choices=['ego','workspace','side'],nargs='+',default=['ego','workspace'])
    parser.add_argument('--output',default='runs/codex_local_smoke')
    parser.add_argument('--max-chunks',type=int,default=2)
    parser.add_argument('--budget-s',type=float,default=5)
    parser.add_argument('--reasoning',choices=['low','medium','high'],default='low')
    args=parser.parse_args()
    if not 1<=args.max_chunks<=4 or not 0<args.budget_s<=12:parser.error('smoke test needs 1–4 chunks and at most 12 simulation seconds')
    root=Path(__file__).resolve().parents[1]
    ssh=['ssh','-o','ControlMaster=no','-o','ControlPath=none','-o','BatchMode=yes',
        '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
    with tempfile.TemporaryDirectory(prefix='hwap-codex-bridge-') as temporary:
        private=Path(temporary);token=private/'token'
        token.write_text(secrets.token_urlsafe(36));os.chmod(token,0o600)
        remote_token='.setup/codex-bridge-'+secrets.token_hex(8)+'.token'
        backend=Backend(SimpleNamespace(codex='codex',schema=root/'configs/action_chunk.schema.json',token_file=token,
            log_dir=root/'runs/codex_bridge',timeout_s=180,reasoning=args.reasoning))
        server=ThreadingHTTPServer(('127.0.0.1',18561),make_handler(backend))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            upload='cd "$HOME/H-WAP" && umask 077 && mkdir -p .setup && cat > '+shlex.quote(remote_token)
            subprocess.run(ssh+[args.ssh_host,upload],input=token.read_bytes(),check=True)
            options=['--token-file',remote_token,'--task','reach','--method','direct','--seeds']+[str(s) for s in args.seeds]
            options+=['--views']+args.views+['--max-chunks',str(args.max_chunks),'--budget-s',str(args.budget_s),'--output',args.output]
            remote='cd "$HOME/H-WAP" && source scripts/activate_server.sh && python scripts/run_codex_agent.py '+shlex.join(options)
            command=ssh+['-o','ExitOnForwardFailure=yes','-R','127.0.0.1:18561:127.0.0.1:18561',args.ssh_host,remote]
            print(json.dumps({'model':'gpt-6.1-sol','host':args.ssh_host,'views':args.views,'seeds':args.seeds,'output':args.output}),flush=True)
            result=subprocess.run(command)
            if result.returncode:raise SystemExit(result.returncode)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)
            cleanup='cd "$HOME/H-WAP" && rm -f -- '+shlex.quote(remote_token)
            try:subprocess.run(ssh+[args.ssh_host,cleanup],timeout=15,check=False)
            except subprocess.TimeoutExpired:print('Bridge stopped; temporary remote bridge capability cleanup timed out.',flush=True)


if __name__=='__main__':main()
