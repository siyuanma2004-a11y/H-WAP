"""Explicit calibration baseline and a configurable multimodal model adapter."""
import base64
import hashlib
import json
import os
from pathlib import Path
import urllib.request
from .actions import CommandChunk
from .images import png_bytes


class ReplayAgent:
    name = "replay_calibration"

    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.calls, self.tokens = 0, 0

    def propose(self, observation, feedback=None):
        return next(self.chunks, None)


class VisionChatAgent:
    name = "vision_chat"

    def __init__(self, config):
        config = json.loads(Path(config).read_text())
        self.url = config["base_url"].rstrip("/")+"/chat/completions"
        if not self.url.startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
            raise ValueError("use HTTPS or a local inference endpoint")
        self.model = config["model"]
        self.provider = config.get("provider","openai_compatible")
        self.name = "codex_cli_chatgpt" if self.provider == "codex_cli_bridge" else "vision_chat"
        self.history = []
        self.key = os.environ.get(config.get("api_key_env", "H_WAP_API_KEY"))
        self.timeout = float(config.get("timeout_s", 60))
        self.calls, self.tokens = 0, 0
        self.system = (
            "Control a simulated Unitree G1. Return ONLY a JSON command chunk with schema_version=1, "
            "name, frame=heading_at_start, and 2-8 keyframes. Times start at 0 and end within 3 seconds. "
            "Pose fields left_wrist/right_wrist use position=[x,y,z] in metres and optional unit "
            "quaternion=[w,x,y,z]. Pose XY is relative to the base at chunk start; Z is above the floor. "
            "Initialize every used field in the first keyframe. velocity=[vx,vy,yaw_rate] is in the "
            "current robot heading frame; limits are [0.15,0.10,0.30]. Hands are closure amounts 0..1. "
            "The current baseline has no independent neck or waist-height control: omit head and "
            "waist_height. Obey observation.capabilities: use zero velocity if velocity is false. "
            "Use short conservative motions and the images plus measured robot state."
        )
        if config.get("prompt_profile") == "reach_stationary_world":
            self.system = (
                "Control a simulated Unitree G1 right palm to touch the green sphere marker. "
                "Estimate its 3D position ONLY from the attached camera images and public camera calibration; "
                "no hidden target coordinate is available. Return only a schema_version=1 command chunk, "
                "frame=world (all positions in global metres). Use 3 keyframes: t=0 exactly at the measured "
                "robot.right_wrist; t=1.5 at your estimated marker position; t=2.5 holding that same target. "
                "Each right_wrist pose has position=[x,y,z], quaternion=null. left_wrist=null. Keep velocity "
                "[0,0,0], both hands null, head null and waist_height null. Name the motion briefly. "
                "Keep wrist targets within 0.9 m in XY of the base and z between 0.55 and 1.55 m. "
                "The camera calibration gives world position, rotation_world_from_camera and intrinsics. "
                "Camera axes are X right, Y up, viewing direction -Z; image pixels start at top-left. "
                "Match the green sphere across views to estimate it, and use the current wrist state to "
                "start smoothly. On subsequent observations compare the actual hand with the green marker "
                "and refine the next target. The tabletop is a known static plane at world z=0.75 m, "
                "but the green reach marker floats above it. Do not treat the blue cube as the target. "
                "Do not call tools or read simulator files."
            )

    def propose(self, observation, feedback=None):
        public = {k:observation[k] for k in ("task","instruction","robot","capabilities","cameras") if k in observation}
        feedback = None if feedback is None else {k:feedback[k] for k in ("duration_s","success","failure") if k in feedback}
        content = [{"type":"text", "text":json.dumps({"observation":public,"feedback":feedback})}]
        image_hashes = {}
        for view, image in observation["images"].items():
            encoded = png_bytes(image)
            image_hashes[view] = hashlib.sha256(encoded).hexdigest()
            content.extend([{"type":"text","text":"Camera: "+view},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(encoded).decode()}}])
        body = json.dumps({"model":self.model,"messages":[{"role":"system","content":self.system},
            {"role":"user","content":content}],"temperature":0,"max_tokens":1800}).encode()
        headers = {"Content-Type":"application/json"}
        if self.key:
            headers["Authorization"] = "Bearer "+self.key
        self.calls += 1
        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        self.tokens += int(result.get("usage",{}).get("total_tokens",0))
        chunk = CommandChunk.parse(json.loads(result["choices"][0]["message"]["content"]))
        self.history.append({"observation":public,"feedback":feedback,"image_sha256":image_hashes,
            "chunk":chunk.to_dict(),"usage":result.get("usage",{}),"bridge":result.get("hwap_bridge")})
        return chunk
