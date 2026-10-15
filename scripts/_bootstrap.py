"""Shared local paths and render selection, before importing MuJoCo."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def configure_render(backend="auto", viewer=False):
    if viewer and sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError("Desktop viewer needs a display. Omit --viewer for headless rendering.")
    if backend == "auto":
        backend = "glfw" if viewer else os.environ.get("MUJOCO_GL")
        if not backend:
            backend = "glfw" if sys.platform == "darwin" or os.environ.get("DISPLAY") else "egl"
    if viewer and backend != "glfw":
        raise ValueError("The desktop viewer requires --gl glfw.")
    if backend not in {"egl", "osmesa", "glfw"}:
        raise ValueError("Choose egl, osmesa, or glfw.")
    os.environ["MUJOCO_GL"] = backend
    return backend


def new_run_path(prefix):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return ROOT / "runs" / f"{prefix}_{stamp}"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
