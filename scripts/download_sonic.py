#!/usr/bin/env python3
"""Download the pinned SONIC ONNX artifacts and verify size and SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

from _bootstrap import ROOT, write_json


def verified(path, expected):
    if not path.is_file() or path.stat().st_size != expected["size"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/sonic_v1_1")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--repair", action="store_true", help="Replace existing files only after a verified download")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/sonic_v1_1.manifest.json").read_text())
    status = {}
    for name, expected in manifest["files"].items():
        target = args.output / name
        if verified(target, expected):
            status[name] = "verified"
            continue
        if args.verify_only:
            status[name] = "missing_or_invalid"
            continue
        if target.exists() and not args.repair:
            raise SystemExit(f"Invalid existing artifact: {target}. Pass --repair to replace it.")
        args.output.mkdir(parents=True, exist_ok=True)
        url = f"{manifest['repository']}/resolve/{manifest['revision']}/{manifest['variant']}/{name}"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=args.output, suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                with urllib.request.urlopen(url, timeout=60) as response:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
            if not verified(temporary, expected):
                raise RuntimeError("Downloaded artifact did not match the pinned manifest: " + name)
            temporary.replace(target)
            status[name] = "verified"
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    all_verified = all(value == "verified" for value in status.values())
    if all_verified and not args.verify_only:
        write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"revision": manifest["revision"], "variant": manifest["variant"],
                      "files": status, "ready": all_verified}, indent=2))
    if not all_verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
