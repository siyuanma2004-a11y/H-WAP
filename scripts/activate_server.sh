#!/usr/bin/env bash
# Source this file from Bash to activate the server environment.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/activate.sh"
