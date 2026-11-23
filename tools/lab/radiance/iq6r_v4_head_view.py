"""Construct a read-only classified checkpoint view with the original DFlash2 source head.

The vocabulary-classified checkpoint may omit lm_head.weight from its index
while the original shard still physically contains it. DFlash2's draft_head
source is that exact tensor. This utility re-exposes it in a separate index
and symlinks existing files; it never rewrites safetensors or input files.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path


def _index(path: Path) -> dict:
    data = json.loads(path.read_text())
    weights = data.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"invalid weight map: {path}")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in weights.items()):
        raise ValueError(f"malformed weight map: {path}")
    return data


def _has_tensor(shard: Path, name: str) -> bool:
    """Read only the safetensors header, never weight payloads."""
    with shard.open("rb") as handle:
        hdr = handle.read(8)
        if len(hdr) != 8:
            raise ValueError(f"invalid safetensors header in {shard.name}")
        length = struct.unpack("<Q", hdr)[0]
        if not (0 < length <= 64 * 1024 * 1024):
            raise ValueError(f"implausible safetensors header in {shard.name}")
        data = json.loads(handle.read(length))
    return isinstance(data, dict) and name in data


def expose_target_head(vhead: Path, original: Path, output: Path, work_root: Path) -> dict:
    vhead = vhead.resolve()
    original = original.resolve()
    work_root = work_root.resolve()
    raw_output = output.absolute()
    if raw_output.is_symlink():
        raise ValueError("the view must be a new directory under the work root")
    output = raw_output.resolve(strict=False)
    resume_root = work_root / "iq6r-v4-resume"
    if not output.is_relative_to(resume_root) or output == resume_root:
        raise ValueError("the view must be a new directory under the work root's resume area")
    if output.is_relative_to(vhead) or output.is_relative_to(original):
        raise ValueError("the view must not modify a source checkpoint")
    base = _index(vhead / "model.safetensors.index.json")
    orig = _index(original / "model.safetensors.index.json")
    head_name = "lm_head.weight"
    if head_name in base["weight_map"]:
        raise ValueError("source head already appears in the classified index")
    shard_name = orig["weight_map"].get(head_name)
    if not shard_name or Path(shard_name).name != shard_name:
        raise ValueError("original checkpoint lacks a safe source-head shard")
    shard = vhead / shard_name
    if not shard.is_file() or not shard.samefile(original / shard_name):
        raise ValueError("the classified view must retain the identical original head shard")
    if not _has_tensor(shard, head_name):
        raise ValueError("the head shard does not physically contain the target tensor")
    expected = {**base["weight_map"], head_name: shard_name}
    files = [p for p in vhead.iterdir() if p.name != "model.safetensors.index.json" and p.is_file()]
    if not files:
        raise ValueError("classified checkpoint has no source files")
    output.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = output / src.name
        if dst.is_symlink():
            if dst.resolve() != src.resolve():
                raise ValueError(f"existing view link changed: {src.name}")
        elif dst.exists():
            raise ValueError(f"view path occupied by non-link: {src.name}")
        else:
            dst.symlink_to(src)
    target = output / "model.safetensors.index.json"
    if target.exists():
        if _index(target)["weight_map"] != expected:
            raise ValueError("existing view index differs; refusing overwrite")
    else:
        base["weight_map"] = expected
        tmp = target.with_suffix(".json.tmp")
        with tmp.open("x") as handle:
            json.dump(base, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp.replace(target)
    return {"head_shard": shard_name, "mapped_weights": len(expected), "linked_files": len(files)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vhead", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work_root = os.environ.get("BC_WORK_ROOT")
    if not work_root:
        parser.error("BC_WORK_ROOT must designate the experimental work root")
    print(json.dumps(expose_target_head(args.vhead, args.original, args.output, Path(work_root)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
