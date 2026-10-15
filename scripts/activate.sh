#!/usr/bin/env bash
# Source this script: source scripts/activate.sh
H_WAP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
H_WAP_VENV="${H_WAP_VENV:-$H_WAP_ROOT/.venv}"
if [[ ! -f "$H_WAP_VENV/bin/activate" ]]; then
    printf 'Environment missing. Run: bash "%s/scripts/setup_env.sh"\n' "$H_WAP_ROOT" >&2
    return 1 2>/dev/null || exit 1
fi
source "$H_WAP_VENV/bin/activate"
if [[ -z "${MUJOCO_GL:-}" ]]; then
    if [[ "$(uname -s)" == "Darwin" || -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
        export MUJOCO_GL=glfw
    else
        export MUJOCO_GL=egl
    fi
fi
export PYTHONPATH="$H_WAP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
