"""Fetch and checksum the pinned public 27B calibration Hessians.

IQR-V4 offline artifact preparation only. No GPU or source-checkpoint writes.
Upstream source: StillDeadcode/qwen3.8-27b-quant-challenge,
revision 2dcf0061cb683152742b904e2965ce7cd92b2da6.
Hessians released under CC BY 4.0 by the dataset publisher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REV = "2dcf0061cb683152742b904e2965ce7cd92b2da6"
REPOSITORY = "StillDeadcode/qwen3.8-27b-quant-challenge"
BASE_URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REV}/"
FILE = re.compile(r"hessians/calib/(?:lm_head[.]input|model[.]language_model[.]layers[.][0-9]+[.][A-Za-z0-9_.]+)[.]safetensors")
GiB = 1024**3


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest_entries(tree: list[dict]) -> list[dict]:
    entries = []
    paths = set()
    for item in tree:
        name = item["path"]
        if not FILE.fullmatch(name):
            raise ValueError(f"unexpected upstream path {name!r}")
        expected = item.get("lfs", {}).get("oid")
        if not expected or not re.fullmatch("[0-9a-f]{64}", expected):
            raise ValueError(f"no trustworthy SHA-256 for {name}")
        if name in paths:
            raise ValueError(f"duplicate file {name}")
        paths.add(name)
        entries.append({"path": name, "bytes": item["size"], "sha256": expected})
    if len(entries) != 257:
        raise ValueError(f"expected 257 Hessians; found {len(entries)}")
    return sorted(entries, key=lambda e: e["path"])


def download(entry: dict, output: Path) -> dict:
    dest = output / Path(entry["path"]).name
    if dest.exists() and dest.stat().st_size == entry["bytes"] and digest(dest) == entry["sha256"]:
        return {"path": entry["path"], "status": "verified-existing"}
    part = dest.with_suffix(".safetensors.part")
    # A prior interrupted partial file is resumable with a verified final SHA.
    result = subprocess.run(
        ["curl", "-fL", "--retry", "4", "--retry-delay", "2",
         "--connect-timeout", "25", "--max-time", "1200",
         "--continue-at", "-", "--silent", "--show-error",
         BASE_URL + entry["path"], "-o", str(part)],
        text=True, capture_output=True, timeout=1250, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"{entry['path']}: curl exit {result.returncode}: {result.stderr[-350:]}")
    if part.stat().st_size != entry["bytes"] or digest(part) != entry["sha256"]:
        raise ValueError(f"{entry['path']}: size or SHA-256 mismatch")
    part.replace(dest)
    return {"path": entry["path"], "status": "downloaded-verified"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error("workers must be 1..6")
    entries = manifest_entries(json.loads(args.tree.read_text()))
    required = sum(e["bytes"] for e in entries)
    args.output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output).free < required + 15 * GiB:
        raise RuntimeError("insufficient disk headroom for Hessians plus 15 GiB safety reserve")
    results = {}
    lock = threading.Lock()
    progress = args.output.parent / "upstream-hessian-download-progress.json"

    def record(entry: dict) -> None:
        with lock:
            results[entry["path"]] = entry["status"]
            progress.write_text(json.dumps({
                "repository": REPOSITORY, "revision": REV,
                "file_count": len(entries), "completed_verified": len(results),
                "statuses": results,
            }, indent=2) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, e, args.output): e for e in entries}
        try:
            for future in as_completed(futures):
                record(future.result())
        except Exception:
            for future in futures:
                future.cancel()
            raise
    print(f"Verified {len(results)}/257 Hessians; {required / GiB:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
