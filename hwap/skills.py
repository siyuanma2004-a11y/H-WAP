"""LLM-agent skills: SKILL.md instructions, worked examples and optional helpers.

Only descriptions stay in context; the full skill is loaded when it matches.
Helper scripts are agent-authored code. ``run_script`` isolates them only by a
separate ``python -I`` process, a temporary working directory and a timeout;
it is NOT a security sandbox and must not be used with untrusted stores.
"""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from .memory import dump_front_matter, one_line, parse_front_matter, words

NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
FILENAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


def skill_name(name):
    if not isinstance(name, str) or len(name) > 64 or not NAME.fullmatch(name):
        raise ValueError("skill names are kebab-case, at most 64 characters")
    return name


def file_name(name, suffix=None):
    if not isinstance(name, str) or len(name) > 128 or not FILENAME.fullmatch(name) or ".." in name:
        raise ValueError("invalid skill file name: "+repr(name))
    if suffix and not name.endswith(suffix):
        raise ValueError(f"skill file must end with {suffix}")
    return name


class SkillLibrary:
    def __init__(self, store):
        self.store = store

    def _read(self, name):
        rel = f"skills/{skill_name(name)}/SKILL.md"
        if not self.store.exists(rel):
            raise KeyError(name)
        fields, body = parse_front_matter(self.store.read(rel))
        return {"name": name, "description": fields.get("description") or "", "body": body,
                "uses": int(fields.get("uses") or 0), "successes": int(fields.get("successes") or 0)}

    def _files(self, name, folder):
        path = self.store.root/"skills"/name/folder
        return [str(p.relative_to(self.store.root)) for p in sorted(path.iterdir()) if p.is_file()] if path.is_dir() else []

    def list(self):
        folder = self.store.root/"skills"
        names = sorted(p.name for p in folder.iterdir() if (p/"SKILL.md").is_file()) if folder.is_dir() else []
        return [{k: v for k, v in self._read(n).items() if k != "body"} for n in names]

    def describe(self):
        return "\n".join(f"- {s['name']}: {s['description']} (uses {s['uses']}, successes {s['successes']})"
                         for s in self.list())

    def match(self, task_text, limit=3):
        wanted = words(task_text)
        scored = [(len(wanted & words(s["name"].replace("-", " "), s["description"])), s) for s in self.list()]
        return [s for score, s in sorted(scored, key=lambda x: (-x[0], x[1]["name"])) if score > 0][:limit]

    def load(self, name):
        skill = self._read(name)
        return {k: skill[k] for k in ("name", "description", "body", "uses", "successes")} | {
            "examples": self._files(name, "examples"), "scripts": self._files(name, "scripts")}

    def write(self, name, description, body, examples=None, scripts=None):
        """Create or replace a skill. examples/scripts map file names to content."""
        name = skill_name(name)
        if description is None or len(one_line(description, "description")) > 300:
            raise ValueError("skill description must be one line of at most 300 characters")
        if not isinstance(body, str) or not body.strip() or len(body) > 20000:
            raise ValueError("skill body must be non-empty and at most 20000 characters")
        try:
            record = self._read(name)
        except KeyError:
            record = {"uses": 0, "successes": 0}
        for folder, files, suffix in (("examples", examples, None), ("scripts", scripts, ".py")):
            for filename, content in (files or {}).items():
                self.store.write(f"skills/{name}/{folder}/{file_name(filename, suffix)}", content)
        self._save(name, description, body, record["uses"], record["successes"])

    def _save(self, name, description, body, uses, successes):
        self.store.write(f"skills/{name}/SKILL.md", dump_front_matter(
            {"name": name, "description": description, "uses": str(uses), "successes": str(successes)}, body))

    def record_use(self, name, success):
        skill = self._read(name)
        self._save(name, skill["description"], skill["body"], skill["uses"]+1, skill["successes"]+int(bool(success)))

    def delete(self, name):
        self._read(name)
        self.store.remove(f"skills/{skill_name(name)}")

    def run_script(self, name, script, args, timeout=10):
        """Run a helper with JSON stdin; it must print keyframes as JSON."""
        source = self.store.path("skills", skill_name(name), "scripts", file_name(script, ".py"))
        if not source.is_file():
            raise KeyError(f"{name}/{script}")
        with tempfile.TemporaryDirectory(prefix="hwap_skill_") as work:
            copy = Path(work)/source.name
            shutil.copy2(source, copy)
            try:
                result = subprocess.run([sys.executable, "-I", str(copy)], cwd=work, input=json.dumps(args),
                                        text=True, capture_output=True, timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"skill script timed out after {timeout} s") from error
        if result.returncode:
            raise RuntimeError("skill script failed: "+result.stderr.strip()[-2000:])
        output = json.loads(result.stdout)
        keyframes = output.get("keyframes") if isinstance(output, dict) else output
        if not isinstance(keyframes, list) or not keyframes or not all(isinstance(k, dict) and "t" in k for k in keyframes):
            raise ValueError("skill script must return a non-empty keyframe list")
        from .actions import Keyframe
        for k in keyframes:
            Keyframe.parse(k)
        return keyframes
