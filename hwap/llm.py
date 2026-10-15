"""Provider-neutral multimodal LLM calls with per-call audit records.

Content is a list of {"type":"text","text":...} and {"type":"image","rgb":array,
"label":view} blocks. Every call returns {"text","usage","latency_s","record"};
records hold prompt/image hashes, never credentials.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
from .images import png_bytes


def _encode(content):
    blocks, hashes = [], []
    for block in content:
        if block["type"] == "image":
            data = png_bytes(block["rgb"])
            hashes.append({"label": block.get("label"), "sha256": hashlib.sha256(data).hexdigest()})
            blocks.append({"type": "image", "label": block.get("label"), "png": data})
        elif block["type"] == "text":
            blocks.append(block)
        else:
            raise ValueError("unknown content block")
    return blocks, hashes


def parse_json(text):
    """Accept bare JSON or a single fenced JSON block."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


class LLMClient:
    provider = "abstract"

    def __init__(self, config):
        self.config = config
        self.model = config["model"]
        self.calls, self.tokens, self.log = 0, 0, []

    def complete(self, system, content, schema=None, purpose="chat"):
        blocks, hashes = _encode(content)
        started = time.monotonic()
        text, usage, extra = self._request(system, blocks, schema)
        latency = time.monotonic()-started
        self.calls += 1
        self.tokens += int(usage.get("total_tokens", usage.get("input_tokens", 0)+usage.get("output_tokens", 0)))
        record = {"purpose": purpose, "provider": self.provider, "model": self.model, "latency_s": latency,
                  "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
                  "text_sha256": hashlib.sha256("".join(b.get("text", "") for b in blocks).encode()).hexdigest(),
                  "images": hashes, "usage": usage, **extra}
        self.log.append(record)
        return {"text": text, "usage": usage, "latency_s": latency, "record": record}


class OpenAICompatibleClient(LLMClient):
    """Chat-completions endpoint, including the local Codex bridge."""

    def __init__(self, config):
        super().__init__(config)
        self.provider = config.get("provider", "openai_compatible")
        self.url = config["base_url"].rstrip("/")+"/chat/completions"
        if not self.url.startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
            raise ValueError("use HTTPS or a local inference endpoint")
        self.key = os.environ.get(config.get("api_key_env", "H_WAP_API_KEY"))
        self.timeout = float(config.get("timeout_s", 60))

    def _request(self, system, blocks, schema):
        content = []
        for block in blocks:
            if block["type"] == "text":
                content.append(block)
            else:
                content.extend([{"type": "text", "text": "Camera: "+str(block["label"])},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,"+base64.b64encode(block["png"]).decode()}}])
        body = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "temperature": 0, "max_tokens": int(self.config.get("max_tokens", 4000))}
        if schema is not None and self.config.get("json_schema", False):
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "hwap", "schema": schema}}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer "+self.key
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        return result["choices"][0]["message"]["content"], result.get("usage", {}), {"bridge": result.get("hwap_bridge")}


class AnthropicClient(LLMClient):
    """Claude Messages API through the official SDK (requirements-llm.txt)."""
    provider = "anthropic"

    def __init__(self, config):
        config = {"model": "claude-opus-5-5", **config}
        super().__init__(config)
        import anthropic
        self.client = anthropic.Anthropic(timeout=float(config.get("timeout_s", 600)))
        self.effort = config.get("effort", "medium")

    def _request(self, system, blocks, schema):
        content = []
        for block in blocks:
            if block["type"] == "text":
                content.append(block)
            else:
                content.extend([{"type": "text", "text": "Camera: "+str(block["label"])},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                     "data": base64.standard_b64encode(block["png"]).decode()}}])
        output_config = {"effort": self.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        response = self.client.beta.messages.create(
            model=self.model, max_tokens=int(self.config.get("max_tokens", 16000)), system=system,
            messages=[{"role": "user", "content": content}], output_config=output_config,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            raise RuntimeError(f"model refused the request (category={category})")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("model output hit max_tokens")
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        return text, usage, {"served_model": response.model, "request_id": response._request_id}


def client_from_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("provider") == "anthropic":
        return AnthropicClient(config)
    return OpenAICompatibleClient(config)
