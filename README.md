# H-WAP

面向 Unitree G1 的 humanoid world-action-planning 研究项目。目前先提供可复现的 **MuJoCo 仿真基础设施**：G1 三指双手、桌面场景、500 Hz 物理 / 50 Hz 控制、关键帧动作、相机、记录、回放和数据导出。

默认控制器使用 Unitree 行走策略的 NumPy 实现及双臂笛卡尔 IK，可在 CPU 上运行。SONIC 是可选的实验后端，目前只实现站立参考，尚未验证完整行走和腰高规划。项目目前没有训练好的视频世界模型。

## 一次安装

推荐 Python 3.11；支持 Python 3.10–3.12。无需 Isaac Lab、CUDA 或 LLM API 即可运行默认演示。

```bash
bash scripts/setup_env.sh
source scripts/activate.sh
python scripts/check_env.py
```

安装器复用或创建项目的 `.venv`，不修改系统 Python。若需指定解释器：

```bash
bash scripts/setup_env.sh --python /path/to/python3.11
```

无窗口 Linux 默认使用 EGL；桌面查看使用 GLFW。系统图形依赖、macOS 及可选依赖见 [ENVIRONMENT.md](ENVIRONMENT.md)。

## 启动仿真

```bash
# 无窗口伸手演示：保存图像、动作和轨迹
python scripts/run_sim.py --demo reach

# 站立 / 小速度行走
python scripts/run_sim.py --demo stand --seconds 3
python scripts/run_sim.py --demo walk --seconds 3

# 有桌面显示服务时打开 G1 查看器
python scripts/run_sim.py --demo reach --viewer --gl glfw
```

`reach` 演示读取目标真值以校准控制链，记录为 `replay_calibration`；它不是视觉 Agent 的研究结果。`stand` 和 `walk` 检查控制与稳定性，其任务成功率不作为伸手任务结果。

每次默认生成新的 `runs/demo_<mode>_<timestamp>/`，包括：

- `episode.json`：任务、seed、环境版本、相机标定、动作、指标和源码哈希。
- `trajectory.npz`：时间、qpos/qvel、实际控制、腕部位置和机器人骨架。
- `frames/<view>/*.png`：同步 RGB 图像。

使用 `--output NEW_DIRECTORY` 可指定新目录；已有回合不会被覆盖。

## 批量实验与回放

```bash
# 相同任务、seed、相机下运行 direct 和 oracle，并验证每回合回放
python scripts/run_experiment.py --config configs/experiments/smoke.json

# 回放终端打印的具体回合目录
python -m hwap replay runs/<batch>/<episode>

# 一次完成环境检查、测试及六个回合的校准实验
bash scripts/smoke_test.sh
```

批次写出 `experiment.json`、每个方法的 JSON/CSV 汇总及 `experiment_report.json`。Oracle 目前使用模拟器和特权几何评分，应作为规划上限单独报告；真实视觉 critic 尚未实现。

真实视觉模型实验模板是 `configs/experiments/vision_reach.json`。先配置自己的视觉 endpoint，再显式传入：

```bash
python scripts/run_experiment.py --config configs/experiments/vision_reach.json \
  --agent-config /path/to/agent.json
```

凭据通过环境变量提供。默认脚本不调用模型 API。已有本机 Codex 桥接的使用记录见 [CODEX_VISION_AGENT.md](CODEX_VISION_AGENT.md)。

## 收集数据与导出

```bash
python scripts/collect_data.py --episodes 10 --chunks 3 --seed-start 100 \
  --output runs/collection_001 --export-dataset datasets/collection_001

python -m hwap export runs/<batch> datasets/<new_dataset>
```

采集器生成小幅随机腕部轨迹，保留失败回合，记录三维动作与实际图像。导出窗口包括 RGB、实测机器人骨架、原始 command chunk、相对动作时间及未来帧时间戳；按 task/seed 分组，避免同 seed 的不同规划方法跨 split。

这些是早期同步数据。实测未来骨架是训练标签/条件，推理时仍需可生成的动作条件或 SONIC 预演；脚本没有将这些数据标记为可直接训练完整世界模型。

## H-WAP harness（proposal 完整循环）

`python -m hwap harness <config> --output <dir>` 运行 proposal §5 的循环：提出 chunk → sim_check → 世界模型想象 →
critic → 修订（K 轮）→ 局部搜索（N 个候选）→ 执行 → 比较想象与真实，回合后可选反思（记忆/技能，经测试闸门）。
基线和消融都是 `configs/harness/` 中的配置：

```bash
python -m hwap harness configs/harness/smoke_oracle.json --output runs/harness_smoke   # 无需 API
bash scripts/setup_env.sh --with-llm                                                    # Claude planner/critic
python -m hwap harness configs/harness/hwap_full.json --output runs/hwap_full
```

记忆、技能和 guidance 存放在版本化 git 仓库中（默认 `<output>/store`，由 `harness_seed/` 初始化）。各模块与 proposal 的对照、
仍缺的基础设施见 [INFRASTRUCTURE.md](INFRASTRUCTURE.md)；部署说明和尚缺内容的详细说明见 [still_requirements.md](still_requirements.md)。

## 可选模块

```bash
# SONIC ONNX 推理依赖与约 200 MB 的固定版本权重
bash scripts/setup_env.sh --with-sonic
python scripts/download_sonic.py
python scripts/download_sonic.py --verify-only

# 原始 TorchScript 行走后端
bash scripts/setup_env.sh --with-torch

# 视频导出（系统已有 ffmpeg 时无需此安装）
bash scripts/setup_env.sh --with-video
python -m hwap video runs/<batch>/<episode> --view ego
```

SONIC 下载器检查每个文件的字节数和 SHA256，完成后写出适配器所需的 manifest。已有损坏文件需显式使用 `--repair` 才会替换。完整 SONIC 控制链、可靠抓取、学习式世界模型和更多主动感知场景仍是后续里程碑（见 INFRASTRUCTURE.md）。

## 开发

```bash
python -m pytest tests -q
make check
make demo
make pilot
```

VS Code 配置已包含 `.venv` 解释器、pytest、演示调试和常用任务。桌面终端执行 `code /path/to/H-WAP` 打开项目；无显示服务的远程会话使用 Remote SSH 或无窗口脚本。

动作协议和已有研究框架的详细说明见 [EXPERIMENT_FRAMEWORK.md](EXPERIMENT_FRAMEWORK.md)。
