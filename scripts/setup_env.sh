#!/usr/bin/env bash
set -euo pipefail
H_WAP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
H_WAP_PYTHON="${H_WAP_PYTHON:-}"
H_WAP_VENV="${H_WAP_VENV:-$H_WAP_ROOT/.venv}"
H_WAP_GL=auto
H_WAP_SONIC=0
H_WAP_TORCH=0
H_WAP_VIDEO=0
H_WAP_LLM=0
H_WAP_CHECK=1
usage() {
    cat <<'USAGE'
Usage: bash scripts/setup_env.sh [options]
  --python PATH       Python 3.10-3.12 executable (3.11 recommended)
  --gl BACKEND        auto, egl, osmesa, or glfw
  --with-sonic        Install the optional ONNX Runtime dependencies
  --with-torch        Install the optional TorchScript backend
  --with-video        Install a bundled ffmpeg executable
  --with-llm          Install the Claude API SDK for the harness planner/critic
  --skip-check        Install only, without the rendering/physics check
Existing environments are reused. No environment or system driver is deleted.
USAGE
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) H_WAP_PYTHON="${2:?--python needs an executable}"; shift 2 ;;
        --gl) H_WAP_GL="${2:?--gl needs a backend}"; shift 2 ;;
        --with-sonic) H_WAP_SONIC=1; shift ;;
        --with-torch) H_WAP_TORCH=1; shift ;;
        --with-video) H_WAP_VIDEO=1; shift ;;
        --with-llm) H_WAP_LLM=1; shift ;;
        --skip-check) H_WAP_CHECK=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage; exit 2 ;;
    esac
done
case "$H_WAP_GL" in auto|egl|osmesa|glfw) ;; *) usage; exit 2 ;; esac
if [[ ! -x "$H_WAP_VENV/bin/python" ]]; then
    if [[ -z "$H_WAP_PYTHON" ]]; then
        for H_WAP_CANDIDATE in "$H_WAP_ROOT/.runtime/bin/python3.11" \
            "$HOME/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11" \
            python3.11 python3.12 python3; do
            if command -v "$H_WAP_CANDIDATE" >/dev/null 2>&1; then
                H_WAP_PYTHON="$H_WAP_CANDIDATE"
                break
            fi
        done
    fi
    if [[ -z "$H_WAP_PYTHON" ]]; then
        printf 'Install Python 3.11, then pass --python /path/to/python3.11.\n' >&2
        exit 1
    fi
    "$H_WAP_PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,13), "Use Python 3.10-3.12"'
    "$H_WAP_PYTHON" -m venv "$H_WAP_VENV"
fi
"$H_WAP_VENV/bin/python" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,13), "Existing venv needs Python 3.10-3.12"'
"$H_WAP_VENV/bin/python" -m pip install -r "$H_WAP_ROOT/requirements.txt"
if [[ "$H_WAP_SONIC" == 1 ]]; then
    "$H_WAP_VENV/bin/python" -m pip install -r "$H_WAP_ROOT/requirements-inference.txt"
fi
if [[ "$H_WAP_TORCH" == 1 ]]; then
    "$H_WAP_VENV/bin/python" -m pip install -r "$H_WAP_ROOT/requirements-torch.txt"
fi
if [[ "$H_WAP_VIDEO" == 1 ]]; then
    "$H_WAP_VENV/bin/python" -m pip install -r "$H_WAP_ROOT/requirements-video.txt"
fi
if [[ "$H_WAP_LLM" == 1 ]]; then
    "$H_WAP_VENV/bin/python" -m pip install -r "$H_WAP_ROOT/requirements-llm.txt"
fi
"$H_WAP_VENV/bin/python" -m pip check
if [[ "$H_WAP_CHECK" == 1 ]]; then
    "$H_WAP_VENV/bin/python" "$H_WAP_ROOT/scripts/check_env.py" --gl "$H_WAP_GL"
fi
printf '\nReady. Activate with: source "%s/scripts/activate.sh"\n' "$H_WAP_ROOT"
