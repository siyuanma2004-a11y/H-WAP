"""Plain Markdown memory entries, retrieved by task, tags and keywords.

Writes only change files; the caller (normally the testing gate) commits them.
"""
from datetime import datetime, timezone
import hashlib
import re

KINDS = ("episode", "world", "body", "world_model", "perception")
TOKEN = re.compile(r"[A-Za-z0-9_.:-]+")
LIST_FIELDS = ("tags", "objects", "evidence")


def words(*texts):
    return {w for t in texts for w in re.findall(r"[a-z0-9]+", str(t or "").lower()) if len(w) > 1}


def tokens(values, name):
    values = [values] if isinstance(values, str) else list(values or ())
    for value in values:
        if not isinstance(value, str) or not TOKEN.fullmatch(value):
            raise ValueError(f"{name} entries must be simple tokens: {value!r}")
    return values


def one_line(value, name):
    if value is not None and (not isinstance(value, str) or "\n" in value or not value.strip()):
        raise ValueError(f"{name} must be a single non-empty line")
    return value


def dump_front_matter(fields, body):
    lines = ["---"]
    for key, value in fields.items():
        if isinstance(value, (list, tuple)):
            value = "["+", ".join(value)+"]"
        lines.append(f"{key}: {'' if value is None else value}")
    return "\n".join(lines+["---", body.strip(), ""])


def parse_front_matter(text):
    if not text.startswith("---\n") or "\n---\n" not in text[3:]:
        raise ValueError("missing front matter")
    head, body = text[4:].split("\n---\n", 1)
    fields = {}
    for line in head.splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        fields[key.strip()] = value if value != "" else None
    return fields, body.strip()


class MemoryBank:
    def __init__(self, store):
        self.store = store

    def write(self, kind, text, tags=(), task=None, evidence=None, objects=(), place=None):
        if kind not in KINDS:
            raise ValueError("memory kind must be one of "+", ".join(KINDS))
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("memory text must be non-empty and at most 4000 characters")
        created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        entry_id = f"{kind}-{created}-{hashlib.sha256(text.encode()).hexdigest()[:6]}"
        fields = {"id": entry_id, "kind": kind, "tags": tokens(tags, "tags"), "task": one_line(task, "task"),
                  "objects": tokens(objects, "objects"), "place": one_line(place, "place"), "created": created,
                  "evidence": tokens(evidence or (), "evidence")}
        self.store.write(f"memory/{kind}/{entry_id}.md", dump_front_matter(fields, text))
        return entry_id

    def entries(self, kinds=None):
        result = []
        for kind in kinds or KINDS:
            folder = self.store.root/"memory"/kind
            for path in sorted(folder.glob("*.md")) if folder.is_dir() else ():
                fields, body = parse_front_matter(path.read_text())
                for key in LIST_FIELDS:
                    fields[key] = fields.get(key) or []
                result.append({**fields, "text": body, "path": str(path.relative_to(self.store.root))})
        return result

    def search(self, query="", task=None, tags=(), kinds=None, limit=8):
        wanted, tags = words(query), set(tags)
        scored = []
        for entry in self.entries(kinds):
            score = len(wanted & words(entry["text"], *entry["tags"], *entry["objects"], entry["task"], entry["place"]))
            score += 2*len(tags & set(entry["tags"]))+(3 if task and entry["task"] == task else 0)
            if score > 0 or not (wanted or tags or task):
                scored.append({**entry, "score": score})
        # Deterministic: best score, then newest, then id.
        scored.sort(key=lambda e: (-e["score"], e["id"]))
        scored.sort(key=lambda e: e["created"], reverse=True)
        scored.sort(key=lambda e: -e["score"])
        return scored[:limit]

    def render(self, entries):
        return "\n".join(f"[{e['kind']}] {' '.join(e['text'].split())} ({e['id']})" for e in entries)

    def _find(self, entry_id):
        if not isinstance(entry_id, str) or not re.fullmatch(r"[a-z_]+-\d{8}T\d{12}Z-[0-9a-f]{6}", entry_id):
            raise ValueError("invalid memory id")
        kind = entry_id.rsplit("-", 2)[0]
        rel = f"memory/{kind}/{entry_id}.md"
        if kind not in KINDS or not self.store.exists(rel):
            raise KeyError(entry_id)
        return rel

    def get(self, entry_id):
        rel = self._find(entry_id)
        return next(e for e in self.entries([rel.split("/")[1]]) if e["path"] == rel)

    def delete(self, entry_id):
        self.store.remove(self._find(entry_id))

    def consolidate(self):
        """Remove exact duplicate texts within a kind, keeping the oldest entry."""
        seen, removed = set(), []
        for entry in sorted(self.entries(), key=lambda e: (e["created"], e["id"])):
            key = (entry["kind"], " ".join(entry["text"].split()).lower())
            if key in seen:
                self.delete(entry["id"])
                removed.append(entry["id"])
            seen.add(key)
        return removed
