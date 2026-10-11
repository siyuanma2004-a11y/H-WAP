# Agent-G1 Simulation Environment

This document explains how to reproduce the Python environment used to compile and test the MuJoCo G1 models in this repository.

## Expected layout

The temporary setup assumes that both projects are sibling directories:

```text
projects/
├── Agent-G1/
└── RSI_locomanip/
```

`RSI_locomanip` is used only as the temporary Python environment source. Its documentation, environment-variable files, research code, results, and ideas are not part of this repository.

## Verified versions

| Dependency | Verified version |
|---|---:|
| Python | 3.9.6 |
| MuJoCo | 3.3.7 |
| NumPy | 2.0.2 |
| pytest | 8.4.2 |

## Option A: link the existing RSI_locomanip environment

From the `Agent-G1` repository root:

```bash
cd /path/to/Agent-G1
ln -s ../RSI_locomanip/.venv .venv
```

Activate it when needed:

```bash
source .venv/bin/activate
```

The symlink is intentionally local and ignored by Git. Do not commit `.env` files or copy the RSI_locomanip virtual environment into this repository.

## Option B: create a standalone environment

If the temporary linked environment is unavailable, create an equivalent local environment:

```bash
cd /path/to/Agent-G1
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "mujoco==3.3.7" "numpy==2.0.2" "pytest==8.4.2"
```

## Environment check and model tests

Run this single command from the repository root:

```bash
cd /Users/Admin/projects/Agent-G1 && \
  .venv/bin/python -c 'import sys, mujoco, numpy, pytest; print("Python:", sys.version.split()[0]); print("Executable:", sys.executable); print("MuJoCo:", mujoco.__version__); print("NumPy:", numpy.__version__); print("pytest:", pytest.__version__)' && \
  .venv/bin/python -m pytest tests -q
```

Expected result:

```text
Python: 3.9.6
Executable: .../.venv/bin/python
MuJoCo: 3.3.7
NumPy: 2.0.2
pytest: 8.4.2
...
5 passed
```
