#!/usr/bin/env python3
"""Run a train rollout through a private localhost SSH-forwarded Codex bridge."""
import argparse
import os
from pathlib import Path
import runpy
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--token-file',default='.setup/codex-bridge.token')
args,rest=parser.parse_known_args()
os.environ['H_WAP_BRIDGE_TOKEN']=Path(args.token_file).read_text().strip()
sys.argv=['hwap','run','--agent-config','configs/agent.codex-local.json']+rest
runpy.run_module('hwap',run_name='__main__')
