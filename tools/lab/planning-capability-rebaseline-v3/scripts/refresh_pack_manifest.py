#!/usr/bin/env python3
"""Refresh PACK_MANIFEST.json hashes for the committed migration pack."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=Path(__file__).resolve().parents[1])
    ns = ap.parse_args()
    pack = ns.pack.resolve()
    manifest_path = pack / "PACK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = []
    for path in sorted(pack.rglob("*")):
        if not path.is_file() or path == manifest_path or "__pycache__" in path.parts:
            continue
        data = path.read_bytes()
        files.append({
            "path": path.relative_to(pack).as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    manifest["files"] = files
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest_files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
