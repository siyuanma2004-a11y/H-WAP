#!/usr/bin/env bash
set -euo pipefail
H_WAP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$H_WAP_ROOT/scripts/activate.sh"
cd "$H_WAP_ROOT"
python scripts/check_env.py
python -m pytest tests -q
python scripts/run_experiment.py --config configs/experiments/smoke.json
