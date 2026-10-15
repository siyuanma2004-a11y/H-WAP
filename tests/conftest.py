"""Select a usable renderer before test modules import MuJoCo."""
import os
import sys


os.environ.setdefault(
    "MUJOCO_GL",
    "glfw" if sys.platform in {"darwin", "win32"} or os.environ.get("DISPLAY") else "egl",
)
