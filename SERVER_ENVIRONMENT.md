# H-WAP Server Environment

Configured: 2026-10-13 (Asia/Shanghai).

## Connect and activate

```bash
ssh trainserver_gy
# Fallback: ssh train
cd ~/H-WAP
source scripts/activate_server.sh
python -m pytest tests -q
```

Server directory: `/home/train/H-WAP`. GPU: RTX 4090, 24 GiB.

All 199 original project files, including Git history and the proposal, were copied and verified using SHA-256. The local project was retained. The macOS virtual-environment symlink was replaced with an independent Linux environment.

## Versions

| Component | Version |
|---|---|
| Python | 3.11.16 |
| MuJoCo | 3.3.7 |
| NumPy | 2.0.2 |
| pytest | 8.4.2 |
| PyTorch | 2.7.1 |
| PyTorch CUDA runtime | 12.6 |

`.runtime/` contains the stable project-owned Python runtime. `.venv/` contains application packages. Existing environments and the server's system CUDA installation were not modified. The activation script defaults to `MUJOCO_GL=egl` for headless rendering and adds the project to `PYTHONPATH`.

## Verified

- All 15 existing tests passed under the stable Python runtime.
- `pip check` passed.
- CUDA tensor computation passed on the RTX 4090.
- EGL rendering passed at 320 x 240 RGB: `.setup/mujoco-headless.png`.
- TorchScript locomotion passed with translation error 0.009546 m, stable pelvis and no detected collisions.
- Original files were verified at migration. Subsequent framework edits are intentional and tracked on `codex/hwap-mujoco-framework`.

## Records

- `requirements-server.lock`: complete pip package versions.
- `.setup/validation.json`: final environment verification.
- `.setup/source-manifest.json`: original file SHA-256 manifest.
- `.setup/repacked-package-provenance.json`: exact versions and source distributions of PyTorch/CUDA/Triton installation packages reused from server caches. Installed RECORD hashes were verified; a source with a mismatching hash was rejected. Packages were installed independently, without linking to other environments.
- `.setup/runtime-packages.json`: stable Python runtime package versions.
- `.setup/wheels/`: retained offline installation packages.

To repeat environment validation:

```bash
cd ~/H-WAP
source scripts/activate_server.sh
python -m pytest tests -q
# .setup/check_environment.py checks the original migration snapshot and is no longer a current-source check.
python -m pip check
```

## Scope

The current repository's MuJoCo simulation, Unitree locomotion, dual-arm control, CUDA PyTorch and headless rendering are configured. The H-WAP experiment framework and experimental SONIC ONNX adapter are described in [EXPERIMENT_FRAMEWORK.md](EXPERIMENT_FRAMEWORK.md). ONNX Runtime 1.22.1 and PyYAML 6.0.2 are installed. TensorRT/C++ hardware deployment and Isaac Lab were not installed.
