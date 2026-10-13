#!/usr/bin/env bash
# Source this file from Bash to activate the server environment.
H_WAP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$H_WAP_ROOT/.venv/bin/activate"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$H_WAP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
