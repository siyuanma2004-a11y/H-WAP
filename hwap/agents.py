"""Explicit calibration baseline and a configurable multimodal model adapter."""
import base64
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

    def propose(self, observation, feedback=None):
        public = {k:v for k,v in observation.items() if k != "images"}
        content = [{"type":"text", "text":json.dumps({"observation":public,"feedback":feedback})}]
        for view, image in observation["images"].items():
            content.extend([{"type":"text","text":"Camera: "+view},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(png_bytes(image)).decode()}}])
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
        return CommandChunk.parse(json.loads(result["choices"][0]["message"]["content"]))
