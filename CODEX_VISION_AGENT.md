# 本机 Codex / ChatGPT 视觉 Agent

已固定模型为 `gpt-6.1-sol`，首轮测试使用 `low` reasoning。仿真和轨迹记录在 train；模型推理通过 Mac 上已登录 ChatGPT 的 Codex CLI 完成。

## 从本机启动

本机项目下已有标准库实现的启动器和模型桥接，运行前确认 `codex login status` 为 ChatGPT 登录：

```bash
cd /Users/Admin/projects/H-WAP
python3 scripts/test_local_codex_agent.py \
  --seeds 0 \
  --views ego workspace \
  --output runs/codex_my_test \
  --max-chunks 2 --budget-s 5
```

需要本机 `codex` 和 Python 3.9+，以及到 `ssh train` 的连通性。train 项目需已安装当前框架并能启用 EGL。每个实验批次使用新的输出目录。

启动器自动启动 `127.0.0.1:18561` 的私有推理桥接，建立 SSH 回环端口转发，运行 train 上的任务，随后关闭服务和转发。SSH 不复用旧的 ControlMaster 会话，以避免失效连接导致任务停滞。临时桥接 capability 用独立随机值生成，服务器文件权限为 0600，任务结束清理；它不是 OpenAI 登录凭据。

## 调用方式

使用文档支持的 `codex exec`，图像附件和 `--output-schema`。登录凭据留在本机，由 CLI 正常读取及刷新，不读取 `auth.json` 内容、不复制到 train、不直接访问未公开的 ChatGPT 后端接口。

每次请求使用临时会话、只读沙箱，关闭 shell、插件、应用、浏览器、其他 Agent 等工具。模型只接收观测，输出一个结构化动作片段；日志检查发现非推理工具活动会拒绝结果。

这不是通用的 OpenAI API 服务。桥接仅支持本项目需要的 chat-completions 子集、固定模型、内嵌 PNG 及固定动作 schema。`temperature` / `max_tokens` 等客户端字段不映射为 Codex CLI 参数；reasoning 在启动器中明确配置，推理有 180 秒超时。

## 观测和动作

发送：任务指令、机器人测得的腕部/基座状态、控制后端能力、相机内外参、所选 RGB 视角。任务反馈仅含执行时长、成功、失败。

不发送：目标/物体真值位置、goal_error、lifted、qpos/qvel、仿真快照或 Oracle 几何评分。请求采用字段白名单，模型输出中的目标坐标是其视觉估计，不是评测器提供的真值。

初测 profile 为固定站立的 reach：使用世界坐标、右掌位置目标、零身体速度；t=0 从当前右掌开始，1.5 秒到目标，2.5 秒保持。头、腰高、手部开合保持未设置。该 profile 不用于抓取或移动任务。双视角配置会提供额外视觉和几何信息；第一视角单独运行可以用 `--views ego` 做独立实验。

仿真在模型推理时暂停，动作返回后再执行 500 Hz 物理循环。这是离线闭环实验接口，模型调用与网络等待带来的延迟应与 2.5 秒的仿真动作时长分别报告。

## 产物与审计

- train：`runs/<output>/summary_direct.json`、每回合的 `episode.json`、轨迹和图像。
- `episode.json`：实际模型/提供方、每轮公开观测、图像 SHA256、模型动作、token 用量及桥接耗时。
- Mac：`runs/codex_bridge/<call-id>/`，含输入图像、公开提示、结构化响应、CLI 事件及调用报告；不记录桥接 Authorization 头。
- 回放只执行记录的动作，不再次调用模型：`python -m hwap replay <episode>`。

## 已验证

本机视觉探测已识别绿色 marker 和蓝色 cube。本机动作探测使用双视角，模型估计目标约 `[0.305, -0.260, 1.049]`，执行成功，末端误差 1.34 cm；推理约 31.8 秒，CLI 检查无工具调用。它只说明接口和小场景可工作，不代表完整任务基准的表现。

模型接入提交：`adba8dc`；HTTP 传输修正：`17d3dde`（分支 `codex/hwap-mujoco-framework`）。原有 24 项测试及 4 项新增接口/数据边界测试共 28 项通过。服务器端双视角独立种子测试已完成：seed 0 / 4 / 29 三个回合全部成功，误差 1.04–1.25 cm，每回合一次模型调用、零 Oracle 调用，全部轨迹精确回放。首组调用分别耗时 100.2 / 88.4 / 30.8 秒，其中长延迟来自 WebSocket 连接失败后重试和回退。验收报告位于 `runs/codex_6_1_sol_smoke/validation.json`。

已使用文档支持的独立 HTTP provider，仍由 CLI 使用本机 ChatGPT 登录，不设置 base_url 或登录 token。内置 provider 不被覆盖；新增 provider ID 为 `hwap_chatgpt_http`。默认 `--transport http`，必要时可用 `--transport auto` 恢复 CLI 自动传输。HTTP 配置的单图探测耗时 12.4 秒且无 WebSocket 重试；相同双视角机器人任务的 HTTP 复测成功，单次推理 18.85 秒（含 CLI 开销），末端误差 1.04 cm。该结果是单次复测，不是平均延迟承诺。记录在 `runs/codex_6_1_sol_http/`。

官方依据：
- [Codex 非交互模式与结构化输出](https://learn.chatgpt.com/docs/non-interactive-mode)
- [Codex ChatGPT 登录](https://learn.chatgpt.com/docs/auth)
- [GPT-6.1 Sol 模型](https://developers.openai.com/api/docs/models/gpt-6.1-sol)

- [自定义 provider 与传输配置](https://learn.chatgpt.com/docs/config-file/config-reference)
