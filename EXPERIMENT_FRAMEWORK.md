# H-WAP MuJoCo experiment framework

Configured on train at `/home/train/H-WAP`, 2026-10-13.

For a fresh local checkout, use `bash scripts/setup_env.sh` and the commands in
[README.md](README.md). The local starter scripts do not require the train
server, its SSH alias, or the previous Mac virtual environment.

## Start and reproduce

```bash
ssh train
cd ~/H-WAP
source scripts/activate_server.sh
python -m pip check
python -m pytest tests -q
python -m hwap run --task reach --method direct --seeds 0 1 2 3 4 --views ego workspace --output runs/new_pilot
python -m hwap run --task reach --method oracle --seeds 0 1 2 3 4 --views ego workspace --output runs/new_pilot
python -m hwap replay runs/new_pilot/<episode-directory>
python -m hwap export runs/new_pilot datasets/new_pilot
python -m hwap video runs/new_pilot/<episode-directory> --view ego
```

Export destinations must be new directories. Runs retain unique episode directories; summary files describe the most recent invocation of that method. Use a new output directory for each experimental batch.

## Implemented sequence

1. Independent Python 3.11 / MuJoCo EGL / CUDA PyTorch environment.
2. Unified 500 Hz physics and 50 Hz control, Cartesian palm targets, symmetric hands, validated versioned JSON chunks, explicit coordinate frames and backend capabilities.
3. Robot-mounted egocentric camera, optional workspace/side views, camera calibration, seeded reach and pick/place task reset and evaluator.
4. Recorded actions, full measured trajectories, RGB frames, calibration, source hashes, independent privileged evaluation and exact simulator replay.
5. Direct execution and state-restoring simulator Oracle local search. The Oracle currently uses a privileged geometric scorer; it is not a visual critic or a learned world model.
6. Configurable multimodal chat-completions Agent adapter; default runs use a state-informed replay calibration reference.
7. Episode-level dataset export with robot-only skeleton conditions. Task/seed groups stay in one split across planning methods.
8. Experimental pinned SONIC v1.1 ONNX standing adapter, pending stability validation. No Isaac Lab installation is required for this route.

## Action contract

A chunk has `schema_version: 1`, `name`, `frame`, and 2–64 ordered keyframes, starting at 0 and lasting at most 3 s. Keyframes can contain `left_wrist`, `right_wrist`, `left_hand`, `right_hand`, `velocity`, `head`, and `waist_height`. Each pose uses metres and optional scalar-first unit quaternion `[w,x,y,z]`. A wrist pose is the palm site (10 cm beyond wrist yaw), not the wrist joint origin. Used sparse fields must be initialized in the first keyframe; omitted later values hold their previous value. Position interpolation is linear, quaternion interpolation is SLERP.

`heading_at_start`: pose XY origin and heading are frozen at chunk start; Z is above the floor. `world`: global pose coordinates. Velocity is always in the current heading frame. The Unitree backend supports small body velocities, arm poses and hands, but rejects head/waist requests. The experimental SONIC standing backend supports arm/torso references and hands but rejects locomotion/waist-height requests until a lower-body planner is integrated. G1 has no independent neck; head control means a torso reference.

## Observation and evaluation boundary

The Agent sees configured RGB views, the task instruction and measured robot proprioception. It receives duration/success/failure feedback. Object position, target distance, simulator state, lift flags and geometric search scores remain in privileged evaluation records. The calibration reference reads the target marker coordinate, and is explicitly labeled `replay_calibration`.

Reach success requires the right palm within 4.5 cm for 0.2 s. Pick/place success additionally requires a prior lift above 6 cm and the object within 4.5 cm of its goal at speed below 0.1 m/s for 0.2 s. Falls, robot/table contact and non-hand object contact end the rollout. Pick/place has an evaluator but no validated grasp policy yet.

## Recorded artifacts

- `runs/pilot_v1/`: original five-seed direct and Oracle calibration pilot; both 5/5, endpoint error about 1.1 cm. Ten episodes replayed with zero qpos difference. These are framework checks, not H-WAP research results.
- Per episode: `episode.json`, `trajectory.npz`, and `frames/<view>/*.png`.
- `datasets/pilot_v1/manifest.json`: 120 windows, train 96 / validation 0 / test 24. This is a small pilot, not sufficient training data or a complete validation split.
- Dataset future conditions use achieved robot FK projected into the recorded camera. They validate synchronization and image conditioning only. A learned world model needs command-conditioned predicted/executable future poses, an explicit treatment of future camera motion, broader data and held-out seeds/tasks.

## Real visual Agent

Copy `configs/agent.example.json` to a configuration with a real vision model endpoint. Supply credentials through its configured environment variable, never in tracked files. Then:

```bash
python -m hwap run --task reach --method direct --agent-config /path/to/agent.json --views ego --seeds 10 11 12 --output runs/visual_agent
```

The repository also includes a local Codex vision bridge; its earlier smoke-test
records are described in [CODEX_VISION_AGENT.md](CODEX_VISION_AGENT.md).
An actual endpoint or local bridge must be explicitly configured for each new
visual-model run. LLM call/token counts and controller/planner costs are
recorded separately where available.

## SONIC provenance and scope

Official source: NVIDIA `GR00T-WholeBodyControl`, commit `b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`, vendored runtime subset under `third_party/GR00T-WholeBodyControl` (excluded from Git).

Official model: `nvidia/GEAR-SONIC`, revision `6733128a3d8a523b1418b06bca3cdf61c8b0987f`, variant `sonic_v1_1`, control 50 Hz, latent 64, encoder input 1751, decoder input 994. Artifact sizes/SHA256 are in `configs/sonic_v1_1.manifest.json` and the installed model directory. Every load verifies them. Weights are fully downloaded and verified on the Mac; upload to train was interrupted by network stalls, so the remote weight directory is incomplete and should not yet be used.

The Python ONNX adapter follows upstream joint order, gains, action scale, observation registry, histories, teleop mode encoding and VR point offsets. It uses CPUExecutionProvider, reserving GPU memory for later visual/world-model work. It is a simulation adapter; the official TensorRT/C++ hardware deployment has not been installed.

```bash
python -m hwap run --backend sonic --sonic-model-dir models/sonic_v1_1 --task reach --seeds 0 --views ego workspace --output runs/sonic_validation
```

Do not promote SONIC to the default before standing/target tracking/replay tests pass in this project's robot and hand model.

## Harness layer

The proposal's full loop (world model, critic, local search, gaze, memory/skills, testing gate) is in
`hwap/harness.py` and is configured by `configs/harness/*.json`; see [INFRASTRUCTURE.md](INFRASTRUCTURE.md).

## Next milestones

1. Validate SONIC standing and dual-arm/torso tracking; then connect its lower-body motion planner. Keep Unitree+IK as the reproducible reference.
2. Deploy a real visual Agent and measure direct vs Oracle with the same seeds, task budgets and camera access; replace privileged scoring with a visual critic for the actual Agent planning experiment.
3. Build a reliable grasp primitive and diversify reach/pick-place configurations; add simple obstructed viewpoints after success on the unobstructed tasks.
4. Expand dataset collection, establish all three grouped splits and command-conditioned future robot/camera previews. Only then train/evaluate a learned video world model on the 4090; larger models can use the 48 GB server or A800.
5. Add learned-world-model search, structured failure memory and reusable verified skills once the direct/Oracle comparison is stable.

## Local starter scripts

- `scripts/setup_env.sh`: isolated CPU simulation environment, optional SONIC,
  TorchScript and video dependencies.
- `scripts/check_env.py`: dependency report, model compilation, physics hold and
  three-view RGB probe; diagnostics are isolated in a child process.
- `scripts/run_sim.py`: recorded stand, walk and calibration reach demonstrations,
  with an optional desktop viewer.
- `scripts/run_experiment.py`: JSON experiment presets, direct/oracle runs,
  aggregate metrics and replay verification.
- `scripts/collect_data.py`: synchronized small random wrist rollouts and optional
  grouped dataset export.
- `scripts/download_sonic.py`: pinned downloads and size/SHA256 validation.

Dataset windows now additionally preserve raw command chunks, offsets relative
to the window, and future image timestamps. Achieved skeleton conditions remain
explicitly labeled as measured; this does not supply a trained video world model
or an inference-time future-pose generator.
