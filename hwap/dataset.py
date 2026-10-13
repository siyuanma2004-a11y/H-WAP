"""Episode-level split and robot-only skeleton conditioning for world models."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .images import png_bytes
from .recording import write_json


def skeleton_image(positions, ids, parents, camera):
    width, height = camera["width"], camera["height"]
    image = np.zeros((height,width,3),dtype=np.uint8)
    R = np.array(camera["rotation_world_from_camera"])
    local = (positions-np.array(camera["position"])) @ R
    depth = -local[:,2]
    f = camera["intrinsics"][0][0]
    projected = np.column_stack([width/2+f*local[:,0]/np.maximum(depth,1e-6),height/2-f*local[:,1]/np.maximum(depth,1e-6)])
    lookup = {body:i for i,body in enumerate(ids)}
    for i,parent in enumerate(parents):
        j = lookup.get(parent)
        if j is None or depth[i]<=0.02 or depth[j]<=0.02:
            continue
        if np.any(np.abs(projected[[i,j]])>max(width,height)*4):
            continue
        count = min(4*max(width,height),max(2,int(np.linalg.norm(projected[i]-projected[j]))+1))
        points = np.rint(np.linspace(projected[i],projected[j],count)).astype(int)
        valid = (points[:,0]>=0)&(points[:,0]<width)&(points[:,1]>=0)&(points[:,1]<height)
        image[points[valid,1],points[valid,0]] = [255,180,50]
    return image


def export_dataset(runs, output, horizon=4):
    root, destination = Path(runs),Path(output)
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
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%10
        split = "train" if bucket<8 else "validation" if bucket==8 else "test"
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
            for view in record["views"]:
                base = ep/view
                samples.append({"episode":path.parent.name,"group":group,"split":split,"view":view,
                    "history_frames":[str((base/f"{start:06d}_rgb.png").relative_to(destination))],
                    "history_conds":[str((base/f"{start:06d}_condition.png").relative_to(destination))],
                    "future_frames":[str((base/f"{i:06d}_rgb.png").relative_to(destination)) for i in range(start+1,start+horizon+1)],
                    "future_conds":[str((base/f"{i:06d}_condition.png").relative_to(destination)) for i in range(start+1,start+horizon+1)]})
        episodes.append({"episode":path.parent.name,"group":group,"split":split,"source":str(path.resolve())})
    if not samples:
        raise ValueError("no synchronized training windows found")
    summary = {"schema_version":1,"conditioning":"achieved_robot_skeleton_only","privileged_object_positions_in_inputs":False,
        "horizon":horizon,"episodes":episodes,"samples":samples,
        "counts":{split:sum(s["split"]==split for s in samples) for split in ("train","validation","test")},
        "readiness":"pilot_data_only"}
    write_json(destination/"manifest.json",summary)
    return {"episodes":len(episodes),"windows":len(samples),"counts":summary["counts"],"manifest":str(destination/"manifest.json")}
