# H-WAP 基础设施对照（proposal v0.4 → 代码库）

本文对照 `Research Proposal: Agentic Humanoid World Action Planner (H-WAP)` v0.4，
列出论文所需的**基础设施**（不含训练和正式实验）现在的状态。
状态标记：✅ 已有/已补齐，🟡 部分可用（有接口或替身，正式组件后续接入），❌ 仍缺，需要决策或外部资源。

## 1. 总览

| Proposal 组件 | 状态 | 代码位置 | 说明 |
|---|---|---|---|
| §3.1 command chunk（腕/手/速度/头/腰高） | ✅ | `hwap/actions.py`, `configs/action_chunk.schema.json` | 新增 `head.position: null` 表示只控制朝向（注视）；LLM 可写 `gaze: {yaw, pitch}`，由 harness 转成 `head` |
| §3.1 50 Hz 插值器 | ✅ | `CommandChunk.sample`, `HumanoidEnvironment.execute` | 500 Hz 物理 / 50 Hz 控制 |
| §4.4 注视控制（head pose = 相机位姿） | ✅ | `hwap/control.py` | 默认 Unitree 后端用腰 yaw/roll/pitch 转动躯干相机；SONIC 后端 head 位置可为空 |
| SONIC VR 3 点 + 运动学规划器 | 🟡 | `hwap/sonic.py` | 只有站立参考；行走、腰高规划器未接入（见 §3） |
| §4.3 Propose → imagine → critic → revise(K) → local search(N) → execute → compare | ✅ | `hwap/harness.py::run_episode` | 每步的想象帧、critic 输出、候选分数和想象与真实的差距都写入 `episode.json` 和 `imagination/` |
| 世界模型接口 | ✅/🟡 | `hwap/world_model.py` | `OracleWorldModel`（仿真想象、上界、可扰动采样以给出分歧度）✅；`RemoteVideoWorldModel`（学习式视频模型的 HTTP 协议，已定 pose 条件来源 sim/command）🟡，服务端后续接 Wan |
| `sim_check`（SONIC-in-sim 可行性和全身位姿） | ✅ | `hwap/world_model.py::sim_check` | 回滚后恢复状态，给出全身骨架用于 pose image |
| Critic | ✅ | `hwap/critic.py` | `VLMCritic`（只看想象帧和公开结果）；`PrivilegedCritic` 读取评估真值，只作带标签的参考 |
| LLM 后端 | ✅ | `hwap/llm.py` | Claude（官方 `anthropic` SDK，默认 `claude-opus-5-5`，结构化输出和拒答回退）；OpenAI 兼容端点和本机 Codex 桥接；每次调用记录哈希、用量和延迟 |
| 主动感知第 1 步（LLM 选择视角） | ✅ | `LLMPlanner`（`gaze` 字段）, `look_chunk` | |
| 主动感知第 3 步（信息增益） | 🟡 | `harness.py::rank_views`, 工具 `choose_view` | 以想象样本分歧度减去运动代价作为代理；第 2 步（ego 世界模型预览）依赖 Route A 模型 |
| §4.5 记忆 | ✅ | `hwap/memory.py` | Markdown 条目，五类（episode/world/body/world_model/perception），按任务、物体、标签检索 |
| §4.5 LLM-agent 技能 | ✅ | `hwap/skills.py`, `harness_seed/skills/` | SKILL.md、examples、scripts；常驻上下文的只有描述，命中后加载全文；辅助脚本在子进程中运行 |
| 版本化存储和回滚 | ✅ | `hwap/store.py` | 独立 git 仓库，隔离 git 身份；changelog 记录证据 |
| 测试闸门 | ✅ | `hwap/improvement.py::TestingGate`, `runner.make_sim_evaluator` | 回归集不变差且触发集变好才接受；`ablate=True` 即“无闸门”消融；安全相关改动可设为 `pending_human` |
| 中循环（反思） | ✅ | `harness.py::reflect` | 回合后反思，产生的改动经闸门处理 |
| 外循环（每 M 回合修订 guidance） | 🟡 | `Guidance`, `Change(kind="guidance_edit")` | 改动类型和闸门已有；按 M 回合触发的调度未写 |
| 附录 A 工具 | ✅ | `harness.py::HarnessTools` | `get_observation / sim_check / imagine / execute / run_policy / memory_* / skill_* / run_regression / choose_view`，带 JSON schema |
| 任务注册表和划分 | ✅ | `hwap/tasks.py` | seen / new_layout / zero_shot 已有任务，compositional 尚空；含 3 个需要主动感知的桌面任务 |
| §7.1 自建主动感知场景（柜子、多房间等） | 🟡 | `hwap/tasks.py` | 现在只有桌面变体（目标在初始视野外、被遮挡）；柜子、抽屉、多房间场景未建 |
| 基线和消融通过配置切换 | ✅ | `configs/harness/*.json`, `python -m hwap harness` | direct LLM、WAP 直接移植、H-WAP 完整版、学习式世界模型、无 API 的冒烟测试 |
| 世界模型数据导出（Route A/B） | ✅ | `hwap/conditioning.py`, `export_world_model_dataset`, `python -m hwap export-wm` | 片段级帧，achieved/command 两种 pose image，27 维动作向量和掩码，逐帧相机参数，可显式指定划分，可选 mp4 |
| 世界模型数据采集 | ✅ | `scripts/collect_data.py --mode wm` | 双臂、手指开合、注视和小速度的随机指令 |
| 指标 | ✅ | `episode.json`, `harness_summary.json` | 成功率、跌倒/碰撞、LLM 调用和 token、世界模型和 critic 调用、sim_check 次数、无效 chunk、注视 chunk、想象与真实的差距、闸门决策 |
| 精确回放和来源记录 | ✅ | `hwap/recording.py` | harness 回合同样可精确回放（想象不改变执行状态） |

### 注视与任务细节

- 默认后端的注视范围：站立时 yaw ±0.6、pitch −0.2…0.3（正值向下看）、roll ±0.08 rad；行走时 yaw ±0.4、pitch −0.1…0.2。
  超出范围的请求会被截断，每个 50 Hz 控制周期最多转动 0.02 rad。在这些范围内，站立和行走的所有角点组合都没有跌倒。
  `observation.capabilities.gaze` 提供这些范围和相机安装参数，`observation.robot.head` 提供测得的躯干朝向。
- 已知限制：行走策略训练时上身固定，所以前倾会让机器人以约 0.16 m/s/rad 的速度缓慢前移；前倾 0.3 rad 持续约 4 s 以上会撞桌。
  前倾的观察动作必须短。
- 任务注册表（`python -c "from hwap.tasks import task_splits; print(task_splits())"`）：

| 任务 | 划分 | 需要主动感知 |
|---|---|---|
| `reach`, `pick_place` | seen | 否（数值与原实现逐位一致） |
| `reach_wide` | new_layout | 否 |
| `reach_hidden_side` | seen | 是：初始 ego 视野外，转头后可见 |
| `reach_hidden_side_left` | zero_shot（换左手） | 是 |
| `reach_occluded` | seen | 是：桌边立柱遮挡，需前倾才能看到，伸手要绕开立柱 |

  `compositional` 划分还没有任务。`env.target_visibility()` 用视锥投影加 `mj_ray` 遮挡判断给出目标是否可见（仅用于评估记录）。

## 2. 新增入口

```bash
# 无 API 的完整循环冒烟测试（校准 planner + oracle 想象 + 特权 critic + 记忆/技能库）
python -m hwap harness configs/harness/smoke_oracle.json --output runs/harness_smoke

# 真实 LLM（需要 bash scripts/setup_env.sh --with-llm 和 ANTHROPIC_API_KEY 或 ant auth login）
python -m hwap harness configs/harness/direct_llm.json --output runs/direct_llm   # 基线：无世界模型
python -m hwap harness configs/harness/wap_port.json   --output runs/wap_port     # WAP 直接移植：无记忆/技能，固定注视
python -m hwap harness configs/harness/hwap_full.json  --output runs/hwap_full    # 完整版：想象 + VLM critic + 记忆/技能 + 反思 + 闸门

# 世界模型数据
python scripts/collect_data.py --mode wm --episodes 20 --output runs/wm_raw
python -m hwap export-wm runs/wm_raw datasets/wm_v1 --clip-frames 9 --stride 4 --video
```

消融项对应 `harness` 字段：`gaze: fixed`（无注视控制）、`use_memory` / `use_skills`、`gate: none`（无测试闸门）、
`refine_rounds` / `local_search`、`world_model.kind`（oracle / remote / none）、`world_model.pose_source`（sim / command 骨架来源）、
`critic.kind`（vlm / privileged / none），以及通过不同 `llm_config` 比较不同 LLM。

## 3. 仍然缺失、需要决策或外部资源的部分

1. **HumanoidArena / Isaac Lab 环境**。Proposal 的主基准是 Isaac Lab 下的 HumanoidArena（7 个任务，SR/AFR 指标），代码库目前只有 MuJoCo 桌面场景。
   需要决定：接 HumanoidArena 作为第二个环境后端（实现与 `HumanoidEnvironment` 相同的 `reset/observe/execute/snapshot/restore/render/metrics` 接口），
   还是论文主要用 MuJoCo 自建场景。若用 Isaac Lab，`snapshot/restore` 需要状态克隆，这是 oracle 想象的前提。
2. **完整 SONIC 控制链**。运动学规划器（速度、方向、骨盆高度 0.3–0.8 m）、行走和腰高均未接入；train 服务器上的 SONIC 权重不完整。
   目前 `waist_height` 和 SONIC 下的 `velocity` 会被拒绝；Unitree+IK 是可复现的参考后端。
3. **学习式世界模型服务端**。客户端协议已定（`RemoteVideoWorldModel` 文档字符串），Wan2.2-TI2V-5B 等模型的推理服务待训练后实现。
4. **人类 VR 遥操作通路**（PICO → command chunk 流的记录和回放）。这是人类上界基线，也是“免费配对数据”的来源，目前没有。
5. **策略基线**（ACT / Diffusion Policy / Flow Matching / pi0.5）的训练和评测接入，以及可选的 `run_policy` 工具，目前没有。
6. **更多主动感知场景**：带柜门/抽屉的 find-and-fetch、桌下/高架、搬运时的遮挡放置、深度判断、多房间搜索、长程整理。
   现有任务注册表和 MjSpec/XML 机制可以扩展，但需要建模铰接物体和对应评估器。
7. **可靠的抓取原语**。pick_place 有评估器，但还没有经过验证的抓取方式；手只有 0–1 开合度，没有 power/pinch 等命名抓握。
8. **外循环调度和持续学习曲线**。闸门和 guidance 编辑已有，“每 M 回合”的调度和 continual split 的批量运行脚本还没写。
9. **延迟隐藏**（当前 chunk 执行时并行规划下一个）。目前仿真在推理时暂停，这在离线实验中可以接受，上真机前需要实现。
10. **真机部分**（ICML）：安全限位、急停、真实相机和标定。按计划后续进行。
