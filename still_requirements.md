# still_requirements：部署说明与尚缺内容

本文写给接手部署和后续开发的人，包括三部分：

1. 本仓库这次新增了哪些内容；
2. 怎样在一台新机器或 train 服务器上部署、验证和运行；
3. 为完成论文 proposal（`Research Proposal: Agentic Humanoid World Action Planner (H-WAP)` v0.4），代码库还缺哪些基础设施。

训练和正式实验不在本文范围内。模块与 proposal 章节的逐项对照见 [INFRASTRUCTURE.md](INFRASTRUCTURE.md)。

---

## 1. 这次新增的内容

| 类别 | 文件 | 作用 |
|---|---|---|
| 规划循环 | `hwap/harness.py` | proposal §5 的完整循环：提出 chunk → `sim_check` → 世界模型想象 → critic → 修订（K 轮）→ 局部搜索（N 个候选）→ 执行 → 想象与真实对比；回合后反思。另有附录 A 的工具接口 `HarnessTools`、注视动作 `look_chunk`、按信息增益排序视角的 `rank_views` |
| 世界模型 | `hwap/world_model.py` | `OracleWorldModel`（仿真想象，论文中的上界，可扰动多次采样）、`sim_check`（回滚式可行性检查和全身位姿）、`RemoteVideoWorldModel`（学习式视频模型的 HTTP 客户端） |
| Critic | `hwap/critic.py` | `VLMCritic` 只看想象帧和公开结果；`PrivilegedCritic` 读取评估真值，只作带标签的参考 |
| LLM 后端 | `hwap/llm.py` | Claude（官方 `anthropic` SDK）和 OpenAI 兼容端点（含本机 Codex 桥接），每次调用都记录审计信息 |
| 实验入口 | `hwap/runner.py`, `configs/harness/*.json` | `python -m hwap harness <config> --output <dir>`；基线和消融都通过配置切换 |
| 自我改进 | `hwap/store.py`, `hwap/memory.py`, `hwap/skills.py`, `hwap/improvement.py`, `harness_seed/` | 版本化 git 存储、Markdown 记忆、SKILL.md 技能、测试闸门、guidance 文件、反思提示和改动解析 |
| 主动感知 | `hwap/actions.py`, `hwap/control.py`, `hwap/environment.py` | head 可以只给朝向；默认后端用腰部三轴转动躯干上的 ego 相机；`look_at`；目标可见性判断 |
| 任务 | `hwap/tasks.py` | 任务注册表和划分（seen / new_layout / zero_shot / compositional），新增 3 个需要主动感知的任务 |
| 世界模型数据 | `hwap/conditioning.py`, `hwap/dataset.py`, `scripts/collect_data.py` | 片段级导出（`python -m hwap export-wm`），包括 27 维指令/实测动作向量、两种 pose image、逐帧相机参数；新增 `--mode wm` 随机采集 |
| 环境脚本 | `scripts/setup_env.sh`, `scripts/activate.sh`, `scripts/check_env.py`, `scripts/run_sim.py`, `scripts/run_experiment.py`, `scripts/download_sonic.py`, `scripts/smoke_test.sh`, `Makefile`, `.vscode/` | 本地一键安装、环境检查、演示、批量实验、SONIC 权重下载 |
| 依赖 | `requirements*.txt` | 按用途拆分：仿真、Torch、视频、SONIC 推理、LLM |
| 测试 | `tests/test_harness.py`, `test_memory_skills.py`, `test_tasks_gaze.py`, `test_wm_dataset.py`, `test_dataset_commands.py` | 共 61 项，不需要网络、GPU 或 API |

---

## 2. 部署说明

### 2.1 系统要求

- Linux（Ubuntu 22.04/24.04 已验证）或 macOS；Python 3.10–3.12，推荐 3.11。
- 默认路线只用 CPU，不需要 CUDA、Isaac Lab 或模型 API。
- 无显示器的 Linux 服务器用 EGL 渲染，需要系统图形库：

  ```bash
  sudo apt-get install libegl1 libgl1-mesa-dri libglfw3
  # 只有使用 OSMesa 软件渲染时才需要：
  sudo apt-get install libosmesa6
  ```

- NVIDIA 机器需要系统已有正常的驱动和 EGL；安装器不会改动驱动。
- `make` 可选；没有时直接运行 Makefile 中对应的命令。

### 2.2 安装

```bash
git clone https://github.com/siyuanma2004-a11y/H-WAP.git
cd H-WAP
git checkout codex/hwap-mujoco-framework

bash scripts/setup_env.sh                 # 创建或复用 .venv，安装 MuJoCo 3.3.7 / NumPy 2.0.2 / pytest，并做环境检查
source scripts/activate.sh                # 激活环境，自动设置 MUJOCO_GL 和 PYTHONPATH
python scripts/check_env.py               # 依赖、模型编译、物理保持、三路相机渲染；结果写入 .setup/
```

指定解释器：`bash scripts/setup_env.sh --python /path/to/python3.11`；指定环境目录：`H_WAP_VENV=/path bash scripts/setup_env.sh`。

可选模块（可以组合）：

| 选项 | 安装内容 | 用途 |
|---|---|---|
| `--with-llm` | `anthropic==1.12.0` | Claude 作为 planner / critic / 反思 |
| `--with-sonic` | `onnxruntime==1.22.1`, `PyYAML` | SONIC ONNX 后端；随后运行 `python scripts/download_sonic.py`（约 200 MB，逐文件校验 SHA256，`--verify-only` 只校验，`--repair` 替换损坏文件） |
| `--with-torch` | `torch==2.7.1` | 原始 TorchScript 行走后端，用于对照 |
| `--with-video` | `imageio-ffmpeg` | 系统没有 ffmpeg 时导出 mp4 |

### 2.3 凭据

凭据不进入仓库。`.gitignore` 已经排除 `.env*`、`configs/agent.local*.json`、`runs/`、`datasets/`。

- **Claude**：设置 `ANTHROPIC_API_KEY`，或者用 `ant auth login` 登录（SDK 会自动读取）。
  配置示例在 `configs/agent.anthropic.example.json`，默认模型 `claude-opus-5-5`，`effort: medium`。
  请求使用结构化 JSON 输出，被拒答时由服务端自动换模型重试（`fallbacks: "default"`），拒答和 `max_tokens` 截断会作为错误报告。
- **OpenAI 兼容端点**：复制 `configs/agent.example.json` 为 `configs/agent.local.json`，在 `api_key_env` 指定的环境变量中提供密钥。只接受 HTTPS 或本机地址。
- **本机 Codex 桥接**：见 [CODEX_VISION_AGENT.md](CODEX_VISION_AGENT.md)。

### 2.4 验证部署

```bash
python -m pytest tests -q                                   # 应为 61 passed，约 25 s
bash scripts/smoke_test.sh                                  # 环境检查 + 测试 + direct/oracle 校准实验 + 回放
python -m hwap harness configs/harness/smoke_oracle.json --output runs/harness_smoke
python -m hwap replay runs/harness_smoke/<episode 目录>      # 应输出 max_qpos_error = 0.0
```

`smoke_oracle.json` 不调用任何 API：planner 是读取目标真值的校准参考，critic 是特权参考，所以它只验证管线能跑通，不是研究结果。

### 2.5 运行入口

```bash
# 仿真演示
python scripts/run_sim.py --demo reach | stand | walk [--viewer --gl glfw]

# 规划循环（需要 --with-llm 和 Claude 凭据）
python -m hwap harness configs/harness/direct_llm.json --output runs/direct_llm   # 基线：无世界模型
python -m hwap harness configs/harness/wap_port.json   --output runs/wap_port     # WAP 直接移植：无记忆/技能，固定注视
python -m hwap harness configs/harness/hwap_full.json  --output runs/hwap_full    # 完整版
python -m hwap harness configs/harness/learned_wm.json --output runs/learned_wm   # 需要先部署世界模型服务（见 2.7）

# 世界模型数据
python scripts/collect_data.py --mode wm --episodes 20 --chunks 3 --views ego workspace side --output runs/wm_raw
python -m hwap export-wm runs/wm_raw datasets/wm_v1 --clip-frames 9 --stride 4 --video [--splits splits.json]

# 旧版 direct/oracle 批量实验和导出
python scripts/run_experiment.py --config configs/experiments/smoke.json
python -m hwap export runs/<batch> datasets/<name>
python -m hwap video runs/<batch>/<episode> --view ego
```

每次运行都要用新的输出目录，已有目录不会被覆盖。

harness 配置（`schema_version: 2`）中可以切换的字段：

| 字段 | 取值 | 对应的消融或基线 |
|---|---|---|
| `planner.kind` | `llm` / `calibration` | 校准参考只用于冒烟测试 |
| `critic.kind` | `vlm` / `privileged` / `none` | VLM critic，或特权上界 |
| `world_model.kind` | `oracle` / `remote` / `none` | 有无世界模型；Oracle 与学习式模型对比 |
| `world_model.pose_source` | `sim` / `command` | 骨架来自 SONIC-in-sim 还是来自指令 |
| `harness.gaze` | `llm` / `fixed` | 有无注视控制 |
| `harness.use_memory` / `use_skills` | bool | 有无记忆、有无技能 |
| `harness.gate` | `test` / `none` | 有无测试闸门 |
| `harness.refine_rounds` / `local_search` / `samples` | 整数 | K、N、世界模型采样数 |
| `planner.llm_config` | 配置文件路径 | 比较不同 LLM |

### 2.6 产物与存储

- `runs/<output>/<episode>/episode.json`：
  - 任务、seed、源码哈希、相机标定
  - 每步的提出、修订、critic 意见、候选分数、想象与真实的差距
  - LLM、世界模型和 critic 的调用计数
- `runs/<output>/<episode>/imagination/`：每轮想象的帧。
- `trajectory.npz`、`frames/<view>/`：实际执行的轨迹和图像，用于精确回放。
- `runs/<output>/store/`：记忆、技能和 guidance 的独立 git 仓库，从 `harness_seed/` 初始化。
  - 被接受的改动会提交，提交信息带证据；被拒绝和待人工批准的改动记在 `changelog.jsonl`。
  - 配置里的 `store.path` 可以指向跨实验共享的存储，用于持续学习。
- `harness_summary.json`：本批次各回合的指标和每次 LLM 调用的审计记录。

### 2.7 学习式世界模型服务的接口约定

`RemoteVideoWorldModel` 向 `<base_url>/imagine` 发送 POST 请求（只允许 HTTPS 或本机地址；如果设置了 `H_WAP_WM_KEY`，会带 Bearer 认证头）：

```json
请求：{"model": str, "views": [view], "samples": int,
       "history": {view: [png_base64]},
       "pose_conditions": {view: [每个未来帧一张 png_base64]},
       "timestamps": [秒], "chunk": {command chunk}}
响应：{"samples": [{view: [每个未来帧一张 png_base64]}], "usage": {...}}
```

服务端要为每个 `pose_conditions` 帧返回一帧预测图像。训练数据的字段与 `export-wm` 导出的片段一一对应。

### 2.8 train 服务器

服务器使用说明见 [SERVER_ENVIRONMENT.md](SERVER_ENVIRONMENT.md)。连接后执行：

```bash
cd ~/H-WAP
source scripts/activate_server.sh
python -m pytest tests -q
```

注意：

- train 上的 SONIC 权重目录不完整，不能直接使用；需要重新运行 `download_sonic.py`，或从已校验的副本同步后用 `--verify-only` 检查。
- 服务器环境要补装本次新增的依赖：`pip install -r requirements-llm.txt`。

### 2.9 常见问题

| 现象 | 处理 |
|---|---|
| `mujoco.FatalError` / EGL 初始化失败 | 确认已安装 libegl1；用 `python scripts/check_env.py --skip-render` 先排除渲染以外的问题；没有 GPU 的机器改用 `MUJOCO_GL=osmesa` |
| macOS 查看器打不开 | 必须用 `.venv/bin/mjpython scripts/run_sim.py --viewer --gl glfw` |
| `replay` 报 source differs | 回放必须使用录制时的源码和模型；先切换回录制时的提交 |
| 前倾注视后机器人往前蹭、撞桌 | 已知限制（见 3.2 第 3 条）；前倾的观察动作要短 |
| GitHub HTTPS 偶尔超时 | 加上 `git -c http.version=HTTP/1.1` 后重试 |

---

## 3. 为完成 proposal 尚缺的内容

按对 CVPR 2027（11 月 16 日提交）的影响排序。每项包括现状、缺少什么、建议的接入点和验收标准。

### 3.1 阻塞主表的（需要尽快决策）

1. **HumanoidArena / Isaac Lab 环境后端**
   - 现状：proposal 的主基准是 HumanoidArena（Isaac Lab，7 个任务，SR/AFR 指标，ACT/DP/FM/pi0.5 基线）；仓库只有 MuJoCo 桌面场景。
   - 缺少：一个与 `HumanoidEnvironment` 接口相同的环境类（`reset/observe/execute/validate_chunk/snapshot/restore/render/camera_parameters/metrics`），以及 HumanoidArena 自己的评估协议。
   - 难点：Oracle 想象和 `sim_check` 都依赖 `snapshot/restore`，Isaac Lab 下需要实现可靠的状态克隆，或者另开一个并行环境实例来想象。
   - 需要决策：论文主表用 HumanoidArena 还是 MuJoCo 自建场景。若用 MuJoCo，需要在论文中说明与 HumanoidArena 不可直接比较。
   - 验收：HumanoidArena 至少 1 个任务能被 `python -m hwap harness` 驱动，并输出 SR/AFR。

2. **完整的 SONIC 控制链**
   - 现状：`hwap/sonic.py` 只有站立参考；`velocity` 和 `waist_height` 会被拒绝；train 上的权重不完整。
   - 缺少：
     - SONIC 运动学规划器（速度 0–6 m/s、方向、风格、骨盆高度 0.3–0.8 m，每 100 ms 重规划）；
     - 站立和手臂跟踪的稳定性验证；
     - 腰高 / 蹲下（"桌下/高架"类场景必需）。
   - 验收：在本项目的机器人和手部模型下，SONIC 后端通过站立、目标跟踪、行走、蹲起和精确回放测试，之后再改为默认后端。现在可复现的参考后端是 Unitree+IK。

3. **自建主动感知场景**
   - 现状：只有桌面变体（`reach_hidden_side`、`reach_hidden_side_left`、`reach_occluded`）；`compositional` 划分没有任务。
   - 缺少 proposal §7.1 中的场景：
     - find-and-fetch（铰接柜门、抽屉）
     - under-and-over（桌下、高架，依赖腰高）
     - 搬运时被自己手臂遮挡的放置
     - 深度判断
     - 多房间搜索
     - 长程整理
   - 每个场景需要 MJCF 模型、布局随机化、成功评估器和可见性检查，并在 `hwap/tasks.py` 中注册。
   - 验收：每个场景至少 10 个 seed，初始视野内看不到目标，执行一次有效观察后能看到，评估器在人工或脚本给出的成功轨迹上判为成功。

4. **抓取原语与命名手型**
   - 现状：`pick_place` 有评估器，但还没有验证过的抓取方式；手只有 0–1 开合度。
   - 缺少：proposal §3.1 的命名手型（open / power / pinch），以及能稳定抓起方块的手指映射和接触参数。
   - 验收：脚本化抓取在 10 个 seed 上成功率达到可报告的水平（建议 ≥ 80%），回放精确。

### 3.2 影响完整版方法和消融的

1. **学习式世界模型服务端**
   - 现状：客户端协议已定（2.7 节）；`export-wm` 能导出训练片段。
   - 缺少：基于 Wan2.2-TI2V-5B（或 OSCAR / Cosmos 备选）的推理服务、多样本输出、显存和延迟预算。
   - 训练本身不在本文范围内。

2. **外循环调度和持续学习**
   - 现状：已有 `guidance_edit` 改动类型、测试闸门和反思接口。
   - 缺少：
     - "每 M 回合根据跨回合模式修订 planner/critic/perception guidance" 的调度；
     - continual split（同一任务族 50 个以上回合）的批量运行脚本和学习曲线统计；
     - 闸门评估的预算控制：每个改动都要在回归集和触发集上各跑两遍，LLM 调用量可观。

3. **注视与行走策略的耦合**
   - 现状：默认后端的行走策略训练时上身固定，前倾会让基座以约 0.16 m/s/rad 前移；不加指令时 6 s 内也有约 −0.13 m 侧移和 −0.22 rad 偏航。
   - 缺少：基座位置保持（在站立时对漂移做闭环补偿），或者在 SONIC 接入后改由 SONIC 处理。

4. **主动感知第 2 步（想象预览）**
   - 依赖 Route A ego 世界模型。
   - 现在的 `rank_views` 用 Oracle 样本分歧度作代理，Oracle 的分歧来自人为扰动物体位置。论文中需要说明这一点，或者用学习式模型的真实样本分歧替换。

5. **tool-calling 形式的 agent**
   - 现状：`HarnessTools` 提供了附录 A 的工具 schema 和调度。
   - 缺少：把它接到 Claude 工具调用循环中（由 LLM 自己决定何时 `imagine`、何时 `choose_view`）。现在的主循环是由代码控制的固定流程。
   - 如果论文要宣称 "agent 自主调用工具"，这一项需要补上。

6. **指标汇总与出表**
   - 现状：每个回合的指标都已记录。
   - 缺少：一个跨批次汇总脚本，生成主表、消融表、学习曲线、"每次观察带来的增益" 和 "观察改变决策的比例"、技能使用统计（加载次数、加载后成功率、编辑次数），并给出置信区间。

### 3.3 基线与数据

1. **人类 VR 遥操作通路**
   - 缺少：PICO → command chunk 流的录制和回放（仿真中可先用键盘或手柄代替）。
   - 用途：人类上界基线，以及 proposal 所说的"免费配对数据"。

2. **策略基线**
   - 缺少：ACT / Diffusion Policy / Flow Matching / pi0.5 在 HumanoidArena 数据上的训练评测接入，以及可选的 `run_policy` 工具（现在调用会返回"未配置"）。

3. **世界模型训练数据的规模和划分**
   - 现状：`collect_data.py --mode wm` 是小幅随机动作，一部分回合因桌面碰撞提前结束（保留了失败回合）。
   - 缺少：
     - 覆盖新任务和场景的、更有结构的采集策略（例如回放 LLM 和脚本策略的成功轨迹，加上随机扰动）；
     - 固定第三人称相机位置的最终方案（Route B）；
     - 在 `--splits` 中明确留出 seed、任务和布局。

### 3.4 上真机前（ICML 阶段）

1. **延迟隐藏**：当前 chunk 执行时并行规划下一个。现在仿真在推理时暂停，离线实验可以接受，真机不行。
2. **安全**：插值器中的速度和加速度限幅、急停按钮、SONIC 安全监控接入。安全相关的 guidance 改动需要人工批准，`TestingGate(require_human=True)` 已经支持。
3. **真实传感器**：头部相机、腕部相机（是否配备仍待确认）、固定第三人称相机的标定和时间同步，以及与仿真记录格式一致的录制器。

### 3.5 proposal 中仍待确认的问题

proposal §12 的开放问题与代码直接相关的有：

- 是否有腕部相机：决定观测视角配置。
- WAP 代码是否开源：决定是否需要按论文复现 critic 和搜索细节。
- chunk 长度和关键帧数：现在限制为 ≤ 3 s、2–64 个关键帧，LLM 提示建议 2–8 个。
- Wan 5B 还是 14B：决定 2.7 节服务端的显存和延迟预算。
- 主表选用哪些 HumanoidArena 任务和自建场景：决定 3.1 第 1、3 条的范围。
