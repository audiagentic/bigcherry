#!/usr/bin/env python3
"""Freeze an approved manifest bundle into the tracked migration pack.

The command is deliberately unusable while validation is non-zero. It creates
a content-addressed snapshot so approved review state survives ignored working
directories before any plan IDs are allocated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


FILES = [
    "SOURCE_LOCK.json", "PLAN_INVENTORY.csv", "LIFECYCLE_NORMALIZATION.csv",
    "SEMANTIC_REVIEW.csv", "DISPOSITIONS.csv", "NAMESPACES.csv", "SUCCESSORS.csv",
    "LINEAGE.csv", "DEPENDENCY_CANDIDATES.tsv", "DEPENDENCY_REMAP.tsv",
    "PLAN_REFERENCES.tsv", "REFERENCE_DECISIONS.tsv",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--destination", type=Path, required=True)
    ap.add_argument("--pack", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ns = ap.parse_args()
    work = ns.work.resolve()
    destination = ns.destination.resolve()
    validator = ns.pack.resolve() / "scripts" / "validate_manifests.py"
    validation = subprocess.run(
        [sys.executable, str(validator), "--pack", str(ns.pack.resolve()), "--work", str(work), "--repo", str(ns.repo.resolve()), "--phase", "review"],
        check=False,
    )
    if validation.returncode:
        raise SystemExit("review validation failed; snapshot not frozen")
    missing = [name for name in FILES if not (work / name).exists()]
    if missing:
        raise SystemExit(f"missing review manifests: {missing}")
    destination.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name in FILES:
        src = work / name
        dst = destination / name
        shutil.copyfile(src, dst)
        hashes[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
    (destination / "SNAPSHOT_HASHES.json").write_text(json.dumps({"files": hashes}, indent=2) + "\n", encoding="utf-8")
    print(f"snapshot={destination}")
    print(f"files={len(FILES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
