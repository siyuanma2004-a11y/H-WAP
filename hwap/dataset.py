"""Episode-level splits, legacy windows and clip exports for action-conditioned video models."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import numpy as np
from .conditioning import (ACTION_DIM, SLICES, KinematicsModel, achieved_skeleton_image, achieved_vectors,
    command_pose_image, command_pose_points, commanded_vectors, describe_layout, integrate_base_path)
from .images import png_bytes
from .recording import write_json


skeleton_image = achieved_skeleton_image  # backward-compatible name
SPLIT_NAME = re.compile(r"[A-Za-z0-9_-]+")


def load_splits(splits):
    """Explicit split map: JSON path or dict. Keys: "task:seed", "task" or "default"."""
    if splits is None:
        return None
    value = json.loads(Path(splits).read_text()) if isinstance(splits, (str, Path)) else dict(splits)
    if any(not isinstance(v, str) or not SPLIT_NAME.fullmatch(v) for v in value.values()):
        raise ValueError("split names must be simple identifiers")
    return value


def assign_split(group, splits=None):
    if splits:
        for key in (group, group.split(":")[0], "default"):
            if key in splits:
                return splits[key]
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%10
    return "train" if bucket<8 else "validation" if bucket==8 else "test"


def export_dataset(runs, output, horizon=4, splits=None):
    root, destination = Path(runs),Path(output)
    splits = load_splits(splits)
    destination.mkdir(parents=True,exist_ok=False)
    samples, episodes = [],[]
    for path in sorted(root.glob("*/episode.json")):
        record = json.loads(path.read_text())
        if record.get("run_error"):
            continue
        arrays = np.load(path.parent/"trajectory.npz",allow_pickle=False)
        if "robot_body_positions" not in arrays or "robot_body_ids" not in record:
            raise ValueError("episode predates robot-pose recording: "+str(path))
        group = f"{record['task']}:{record['seed']}"
        split = assign_split(group, splits)
        ep = destination/split/path.parent.name
        ep.mkdir(parents=True)
        frames = record["frames"]
        for frame_index,frame in enumerate(frames):
            for view,relative in frame["images"].items():
                folder = ep/view;folder.mkdir(exist_ok=True)
                shutil.copy2(path.parent/relative,folder/f"{frame_index:06d}_rgb.png")
                body = arrays["robot_body_positions"][frame["row"]]
                rgb = skeleton_image(body,record["robot_body_ids"],record["robot_parent_ids"],frame["cameras"][view])
                (folder/f"{frame_index:06d}_condition.png").write_bytes(png_bytes(rgb))
        for start in range(0,len(frames)-horizon,horizon):
            window_start, window_end = frames[start]["time"], frames[start+horizon]["time"]
            commands = [
                {"chunk": item["chunk"],
                 "chunk_start_relative_s": float(arrays["time"][item["start_row"]]-window_start)}
                for item in record["chunks"]
                if arrays["time"][item["start_row"]] < window_end
                and arrays["time"][item["end_row"]] > window_start
            ]
            for view in record["views"]:
                base = ep/view
                samples.append({"episode":path.parent.name,"group":group,"split":split,"view":view,
                    "history_frames":[str((base/f"{start:06d}_rgb.png").relative_to(destination))],
                    "history_conds":[str((base/f"{start:06d}_condition.png").relative_to(destination))],
                    "future_frames":[str((base/f"{i:06d}_rgb.png").relative_to(destination)) for i in range(start+1,start+horizon+1)],
                    "future_conds":[str((base/f"{i:06d}_condition.png").relative_to(destination)) for i in range(start+1,start+horizon+1)],
                    "command_chunks": commands,
                    "future_timestamps_s": [frames[i]["time"]-window_start for i in range(start+1,start+horizon+1)]})
        episodes.append({"episode":path.parent.name,"group":group,"split":split,"source":str(path.resolve())})
    if not samples:
        raise ValueError("no synchronized training windows found")
    summary = {"schema_version":1,"conditioning":"achieved_robot_skeleton_only","privileged_object_positions_in_inputs":False,
        "horizon":horizon,"episodes":episodes,"samples":samples,
        "counts":split_counts(samples),
        "command_conditioning_available": True, "readiness":"pilot_data_only"}
    write_json(destination/"manifest.json",summary)
    return {"episodes":len(episodes),"windows":len(samples),"counts":summary["counts"],"manifest":str(destination/"manifest.json")}


def split_counts(items):
    names = ["train","validation","test"]+sorted({i["split"] for i in items}-{"train","validation","test"})
    return {name:sum(i["split"]==name for i in items) for name in names}


def find_ffmpeg():
    import os
    ffmpeg = os.environ.get("H_WAP_FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as error:
            raise RuntimeError("Install video support: bash scripts/setup_env.sh --with-video") from error
    return ffmpeg


def write_clip_video(ffmpeg, paths, times, output, fps):
    """Timestamp-faithful mp4 (recorded frames are not exactly uniform)."""
    quoted = lambda p: str(Path(p).resolve()).replace("'","'\\''")
    lines = []
    for (a,ta),(_,tb) in zip(zip(paths,times),zip(paths[1:],times[1:])):
        lines += [f"file '{quoted(a)}'",f"duration {max(tb-ta,1e-6):.9f}"]
    lines.append(f"file '{quoted(paths[-1])}'")
    with tempfile.NamedTemporaryFile(mode="w",suffix=".txt") as stream:
        stream.write("\n".join(lines)+"\n");stream.flush()
        subprocess.run([ffmpeg,"-hide_banner","-loglevel","error","-n","-f","concat","-safe","0","-i",stream.name,
            "-t",str(times[-1]-times[0]+1/fps),"-vf",f"fps={fps},pad=ceil(iw/2)*2:ceil(ih/2)*2","-c:v","libx264",
            "-pix_fmt","yuv420p",str(output)],check=True)


def export_world_model_dataset(runs, output, clip_frames=9, stride=4, views=None, fps=10, include_failures=True,
                               splits=None, video=False, command_camera="recorded"):
    """Clip-level samples for action-conditioned video world models (Route A ego / Route B fixed views).

    Frame 0 of each clip is history; frames 1.. are futures. Per clip:
    actions.npz holds commanded/achieved vectors (conditioning.ACTION_LAYOUT),
    masks, timestamps and per-frame camera calibration. Condition images:
    ``achieved`` = measured robot skeleton (SONIC-in-sim style), ``command`` =
    commanded targets + integrated commanded base path, with shoulders taken
    from the history frame only. ``command_camera="history"`` renders command
    images with the history-frame camera instead of the recorded future camera.
    """
    if clip_frames < 2 or stride < 1 or fps <= 0 or command_camera not in {"recorded","history"}:
        raise ValueError("clip_frames >= 2, stride >= 1, fps > 0, command_camera recorded|history")
    root, destination = Path(runs),Path(output)
    splits = load_splits(splits)
    ffmpeg = find_ffmpeg() if video else None
    destination.mkdir(parents=True,exist_ok=False)
    kinematics, clips, episodes, skipped = None, [], [], []
    for path in sorted(root.glob("*/episode.json")):
        record = json.loads(path.read_text())
        metrics = record.get("metrics",{})
        if record.get("run_error") or (not include_failures and (metrics.get("failure") or not metrics.get("success"))):
            skipped.append({"episode":path.parent.name,"reason":"run_error" if record.get("run_error") else "failure"})
            continue
        selected = [v for v in (views or record["views"]) if v in record["views"]]
        frames = record["frames"]
        if not selected or len(frames) < clip_frames or not record.get("chunks"):
            skipped.append({"episode":path.parent.name,"reason":"no matching views, chunks or frames"})
            continue
        arrays = np.load(path.parent/"trajectory.npz",allow_pickle=False)
        if "robot_body_positions" not in arrays or "robot_body_ids" not in record:
            raise ValueError("episode predates robot-pose recording: "+str(path))
        kinematics = kinematics or KinematicsModel()
        rows = [f["row"] for f in frames]
        times = np.array([f["time"] for f in frames])
        commanded, command_mask = commanded_vectors(record, arrays, rows)
        achieved, achieved_mask = achieved_vectors(arrays, rows, kinematics)
        group = f"{record['task']}:{record['seed']}"
        split = assign_split(group, splits)
        ep = destination/split/path.parent.name
        for i,frame in enumerate(frames):
            for view in selected:
                folder = ep/view;folder.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path.parent/frame["images"][view],folder/f"{i:06d}_rgb.png")
                image = achieved_skeleton_image(arrays["robot_body_positions"][frame["row"]],record["robot_body_ids"],
                    record["robot_parent_ids"],frame["cameras"][view])
                (folder/f"{i:06d}_achieved.png").write_bytes(png_bytes(image))
        final_failure = metrics.get("failure")
        relative = lambda p: str(p.relative_to(destination))
        for start in range(0,len(frames)-clip_frames+1,stride):
            index = list(range(start,start+clip_frames))
            clip = ep/"clips"/f"{start:06d}"
            shoulders = kinematics.shoulders_relative(arrays["qpos"][rows[start]])
            velocity = np.where(command_mask[index][:,SLICES["base_velocity"]],commanded[index][:,SLICES["base_velocity"]],0)
            path_ = integrate_base_path(arrays["base_pose"][rows[start]],times[index],velocity)
            data = {"timestamps_s":times[index]-times[start],"absolute_time_s":times[index],"rows":np.array(rows)[index],
                    "commanded":commanded[index],"commanded_mask":command_mask[index],
                    "achieved":achieved[index],"achieved_mask":achieved_mask[index],"commanded_base_path":path_}
            entry = {"clip":relative(clip),"episode":path.parent.name,"group":group,"split":split,"task":record["task"],
                "seed":record["seed"],"views":selected,"actions":relative(clip/"actions.npz"),
                "timestamps_s":data["timestamps_s"].tolist(),"frames":{},"achieved_conditions":{},"command_conditions":{},
                "camera_static":{},"video":{},
                "outcome":{"episode_success":bool(metrics.get("success")),"episode_failure":final_failure,
                           "ends_in_failure":bool(final_failure) and index[-1]==len(frames)-1}}
            for view in selected:
                cams = [frames[i]["cameras"][view] for i in index]
                for key,name in (("position","position"),("rotation_world_from_camera","rotation"),("intrinsics","intrinsics")):
                    data[f"camera_{view}_{name}"] = np.array([c[key] for c in cams],dtype=float)
                entry["camera_static"][view] = bool(np.allclose(data[f"camera_{view}_position"],data[f"camera_{view}_position"][0],atol=1e-6)
                    and np.allclose(data[f"camera_{view}_rotation"],data[f"camera_{view}_rotation"][0],atol=1e-6))
                folder = clip/view;folder.mkdir(parents=True)
                commands = []
                for k,i in enumerate(index):
                    points = command_pose_points(commanded[i],command_mask[i],path_[k],shoulders,path_[:k+1])
                    camera = cams[0] if command_camera == "history" else cams[k]
                    target = folder/f"{k:03d}_command.png"
                    target.write_bytes(png_bytes(command_pose_image(points,camera)))
                    commands.append(relative(target))
                entry["frames"][view] = [relative(ep/view/f"{i:06d}_rgb.png") for i in index]
                entry["achieved_conditions"][view] = [relative(ep/view/f"{i:06d}_achieved.png") for i in index]
                entry["command_conditions"][view] = commands
                if ffmpeg:
                    target = clip/f"{view}.mp4"
                    write_clip_video(ffmpeg,[ep/view/f"{i:06d}_rgb.png" for i in index],times[index],target,fps)
                    entry["video"][view] = relative(target)
            np.savez_compressed(clip/"actions.npz",**data)
            clips.append(entry)
        episodes.append({"episode":path.parent.name,"group":group,"split":split,"source":str(path.resolve()),
                         "success":bool(metrics.get("success")),"failure":final_failure})
    if not clips:
        raise ValueError("no clips found; check views, clip_frames and run directories")
    with (destination/"clips.jsonl").open("w") as stream:
        for entry in clips:
            stream.write(json.dumps(entry,ensure_ascii=False)+"\n")
    summary = {"schema_version":1,"kind":"hwap_world_model_clips","action_dim":ACTION_DIM,"action_layout":describe_layout(),
        "action_frame":"world; base_velocity in heading frame [vx, vy, yaw_rate]","clip_frames":clip_frames,"stride":stride,
        "history_frames":1,"views":sorted({v for c in clips for v in c["views"]}),"video_fps":fps if video else None,
        "command_camera":command_camera,"include_failures":include_failures,"counts":split_counts(clips),
        "episode_counts":split_counts(episodes),"episodes":episodes,"skipped":skipped,"clips_index":"clips.jsonl",
        "privileged_object_positions_in_inputs":False,"readiness":"pilot_data_only",
        "notes":["Recorded frames are ~10 Hz but not exactly uniform at chunk boundaries; use timestamps_s.",
                 "heading_at_start chunks that begin inside a clip are anchored at the recorded base pose of their start.",
                 "Egocentric camera calibration at future frames is an outcome; camera_static marks moving views.",
                 "Achieved vectors are forward kinematics of recorded qpos with the current model source."]}
    write_json(destination/"manifest.json",summary)
    return {"episodes":len(episodes),"clips":len(clips),"counts":summary["counts"],"skipped":len(skipped),
            "manifest":str(destination/"manifest.json")}
