# H-WAP

H-WAP is a MuJoCo simulation and control project for the Unitree G1 humanoid.
It provides the G1 model with three-finger hands, a tabletop object scene,
Unitree's official 12-DoF locomotion policy, and symmetric seven-DoF dual-arm
position control.

## Quick test

From the repository root:

```bash
python -m pytest tests -v
```

The current suite verifies model compilation, stable walking, collision-free
nearby relocation, high-level Unitree transport behavior, and dual-arm
position convergence.

## Environment

See [ENVIRONMENT.md](ENVIRONMENT.md) for the temporary linked-environment
setup, standalone installation instructions, verified dependency versions, and
the combined environment/test command.

## Reproducible experiment framework

See [EXPERIMENT_FRAMEWORK.md](EXPERIMENT_FRAMEWORK.md) for the train environment, action chunks, direct/Oracle runs, replay, camera-conditioned dataset export and the experimental SONIC adapter.
