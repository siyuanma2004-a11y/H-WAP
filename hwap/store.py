"""Git-versioned harness store for memory, skills and guidance.

Every accepted change is a commit whose message body carries its evidence as
JSON. ``changelog.jsonl`` (untracked) additionally records commits, rejected
or pending changes and rollbacks, so rejected edits leave content untouched.
Git runs with an isolated identity and ignores global/system configuration.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess

GIT = ["git", "-c", "user.name=hwap", "-c", "user.email=hwap@local", "-c", "commit.gpgsign=false",
       "-c", "init.defaultBranch=main", "-c", "core.hooksPath=/dev/null"]
CHANGELOG = "changelog.jsonl"


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def safe_relative(rel):
    """Return a normalized store-relative POSIX path; reject traversal."""
    if not isinstance(rel, str) or not rel or "\\" in rel or "\x00" in rel:
        raise ValueError("invalid store path")
    path = PurePosixPath(rel)
    if path.is_absolute() or any(p in ("", ".", "..") for p in path.parts) or path.parts[0] == ".git":
        raise ValueError("store path must stay inside the store: "+rel)
    if rel == CHANGELOG:
        raise ValueError("the changelog is append-only")
    return str(path)


class VersionedStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if not (self.root/".git").exists():
            self._git("init", "-q")
            ignore = self.root/".gitignore"
            lines = ignore.read_text().splitlines() if ignore.exists() else []
            if CHANGELOG not in lines:
                ignore.write_text("\n".join(lines+[CHANGELOG])+"\n")
            self.commit("initialize harness store", {"root": str(self.root)})

    @classmethod
    def from_seed(cls, root, seed_dir):
        root = Path(root)
        if not root.exists():
            shutil.copytree(seed_dir, root, ignore=shutil.ignore_patterns(".git", CHANGELOG, "__pycache__"))
            store = cls(root)
            store.append_log({"event": "seeded", "seed": str(Path(seed_dir).resolve()), "sha": store.head()})
            return store
        return cls(root)

    def _git(self, *args, stdin=None):
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
        result = subprocess.run(GIT+list(args), cwd=self.root, input=stdin, text=True, env=env,
                                capture_output=True)
        if result.returncode:
            raise RuntimeError(f"git {args[0]} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    def path(self, *parts):
        return self.root/safe_relative("/".join(parts))

    def read(self, rel):
        return self.path(rel).read_text()

    def exists(self, rel):
        return self.path(rel).exists()

    def write(self, rel, text):
        target = self.path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text, bytes):
            target.write_bytes(text)
        else:
            target.write_text(text)
        return target

    def remove(self, rel):
        target = self.path(rel)
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    def append_log(self, entry):
        with (self.root/CHANGELOG).open("a") as stream:
            stream.write(json.dumps({"time": now(), **entry}, ensure_ascii=False, allow_nan=False)+"\n")

    def commit(self, message, evidence=None):
        if not isinstance(message, str) or not message.strip() or "\n" in message:
            raise ValueError("commit message must be one non-empty line")
        evidence = evidence or {}
        body = message+"\n\n"+json.dumps({"evidence": evidence}, indent=2, ensure_ascii=False, allow_nan=False)
        self._git("add", "-A")
        self._git("commit", "-q", "--allow-empty", "-F", "-", stdin=body)
        sha = self.head()
        self.append_log({"event": "commit", "sha": sha, "message": message, "evidence": evidence})
        return sha

    def head(self):
        return self._git("rev-parse", "HEAD")

    def rollback(self, sha, reason=""):
        if not re.fullmatch(r"[0-9a-f]{7,40}", sha or ""):
            raise ValueError("rollback needs a commit sha")
        self._git("revert", "--no-edit", sha)
        new = self.head()
        self.append_log({"event": "rollback", "reverted": sha, "sha": new, "reason": reason})
        return new

    def snapshot(self, dest):
        """Full copy (including history) for isolated evaluation; dest must be new."""
        dest = Path(dest)
        if dest.exists():
            raise ValueError("snapshot destination already exists")
        shutil.copytree(self.root, dest)
        return dest

    def log(self, limit=20):
        path = self.root/CHANGELOG
        entries = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return entries[::-1][:limit]
