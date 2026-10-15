# H-WAP 本地仿真环境

## 默认环境

推荐独立 Python 3.11，支持 3.10–3.12。运行 `bash scripts/setup_env.sh` 在当前项目创建/复用 `.venv`，安装：

| 依赖 | 固定版本 | 用途 |
|---|---|---|
| MuJoCo | 3.3.7 | G1 物理、模型与渲染 |
| NumPy | 2.0.2 | 数值计算及默认行走策略 |
| pytest | 8.4.2 | 验证现有控制与数据边界 |

默认不安装 PyTorch、CUDA、Isaac Lab 或大模型权重。项目自带的 `models/unitree_rl_gym/motion.pt` 由 NumPy loader 读取，不需要导入 PyTorch。

```bash
bash scripts/setup_env.sh --python /path/to/python3.11
source scripts/activate.sh
python scripts/check_env.py
python -m pytest tests -q
```

也可以手动安装：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

安装器默认复用已有环境。设置 `H_WAP_VENV=/another/path` 可使用其他环境目录；`H_WAP_PYTHON` 或 `--python` 指定创建环境的解释器。

## 渲染

- 无窗口 Linux：EGL，`MUJOCO_GL=egl`。
- Linux/macOS 桌面：GLFW，`MUJOCO_GL=glfw`。
- Linux 软件渲染备选：OSMesa，需安装系统 OSMesa 库。

图形后端在导入 MuJoCo 前选定。Ubuntu 缺少基础图形库时可安装：

```bash
sudo apt-get install libegl1 libgl1-mesa-dri libglfw3
# 仅在使用 OSMesa 时需要
sudo apt-get install libosmesa6
```

有 NVIDIA GPU 的机器需由系统提供正常驱动/EGL；安装器不修改驱动。只检查模型与物理而不渲染：

```bash
python scripts/check_env.py --skip-render
```

环境检查生成 `.setup/environment.json` 和三路 `.setup/render_<view>.png`。SONIC 权重、Torch、视频编码属于可选项，缺失时不会使默认环境检查失败。

## 桌面查看

```bash
python scripts/run_sim.py --viewer --gl glfw --demo reach
```

需要实际显示服务。macOS 的 passive viewer 必须使用 MuJoCo 提供的 `mjpython`：

```bash
.venv/bin/mjpython scripts/run_sim.py --viewer --gl glfw --demo reach
```

[MuJoCo 3.3.7 官方 Python 文档](https://mujoco.readthedocs.io/en/3.3.7/python.html) 说明了安装、passive viewer 和 macOS launcher 的要求。

## 可选依赖与权重

```bash
bash scripts/setup_env.sh --with-sonic
python scripts/download_sonic.py

bash scripts/setup_env.sh --with-torch
bash scripts/setup_env.sh --with-video
```

SONIC 使用 `configs/sonic_v1_1.manifest.json` 固定的 Hugging Face revision 和哈希。首次下载 encoder/decoder/config 总计约 200 MB；它不会自动成为默认控制后端。`--verify-only` 不发起下载，缺失/损坏时以非零状态退出。

视频模块优先使用 `H_WAP_FFMPEG` 指定的可执行文件，其次系统 ffmpeg，最后使用可选 `imageio-ffmpeg` 的打包程序。

## 复现边界

Makefile 是可选快捷入口，需要 GNU make；没有 make 时直接使用 README
中的 Bash / Python 命令即可。Ubuntu 可通过 `sudo apt-get install make` 安装。

每次实验保留 seed、环境版本、源码哈希、相机标定和动作。精确回放要求原始源码与模型相同。默认 direct/oracle smoke 是读取目标真值的校准实验；视觉模型实验需要用户显式配置。

旧的 Mac 环境链接和 train 服务器说明保留在 Git 历史 / `SERVER_ENVIRONMENT.md`，本机启动不依赖这些路径或 SSH 别名。
