#!/usr/bin/env python3
"""Check dependencies, model compilation, stable control and RGB rendering."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

from _bootstrap import ROOT, configure_render, write_json


def probe(args):
    import mujoco
    from hwap.actions import CommandChunk, Keyframe, Pose
    from hwap.environment import HumanoidEnvironment
    from hwap.images import png_bytes

    with HumanoidEnvironment(views=("ego", "workspace", "side"), width=160, height=120) as env:
        env.reset(0)
        pose = Pose(tuple(env.sim.palm_position("right")))
        outcome = env.execute(CommandChunk("environment_hold", (
            Keyframe(0, right_wrist=pose), Keyframe(0.2, right_wrist=pose)), frame="world"))
        if outcome["failure"]:
            raise RuntimeError("Physics check failed: " + outcome["failure"])
        result = {"physics_dt": env.sim.model.opt.timestep, "control_dt": 0.02,
                  "actuators": env.sim.model.nu, "capabilities": env.observe(render=False)["capabilities"],
                  "pelvis_height_m": env.sim.pelvis_height, "rendered_images": {}}
        if not args.skip_render:
            for view, frame in env.render().items():
                if frame.shape != (120, 160, 3):
                    raise RuntimeError("Unexpected RGB shape")
                target = Path(args.output).parent / f"render_{view}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(png_bytes(frame))
                result["rendered_images"][view] = str(target.resolve())
        result["mujoco_version"] = mujoco.__version__
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gl", choices=("auto", "egl", "osmesa", "glfw"), default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / ".setup/environment.json")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    backend = configure_render(args.gl)
    if args.worker:
        print(json.dumps(probe(args)))
        return
    report = {"python": sys.version.split()[0], "executable": sys.executable,
              "render_backend": backend, "packages": {}, "status": "failed"}
    for name in ("mujoco", "numpy", "pytest", "torch", "onnxruntime", "imageio-ffmpeg"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    report["video_export_available"] = bool(shutil.which("ffmpeg") or report["packages"]["imageio-ffmpeg"])
    manifest = json.loads((ROOT / "configs/sonic_v1_1.manifest.json").read_text())
    report["optional_sonic_artifacts"] = {}
    for name, expected in manifest["files"].items():
        path = ROOT / "models/sonic_v1_1" / name
        valid = path.is_file() and path.stat().st_size == expected["size"]
        if valid:
            valid = hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
        report["optional_sonic_artifacts"][name] = "verified" if valid else "missing_or_invalid"
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--gl", backend,
               "--output", str(args.output)]
    if args.skip_render:
        command.append("--skip-render")
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
        if result.returncode:
            report["error"] = result.stderr.strip()[-3000:] or f"Probe exited {result.returncode}"
        else:
            report["simulation"] = json.loads(result.stdout)
            report["status"] = "ok"
    except (subprocess.TimeoutExpired, ValueError) as error:
        report["error"] = str(error)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        print("Run setup_env.sh first. Headless Linux needs EGL; use --gl osmesa if OSMesa is installed.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
