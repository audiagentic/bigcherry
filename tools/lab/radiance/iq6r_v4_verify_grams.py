"""Card-free sample audit of all converted RADGRAM2 Grams against source Hessians.

This checks selected upper and lower triangle cells directly by random access
without loading multi-GiB matrices. It does not prove numerical Hessian quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ.get("BC_WORK_ROOT", ".")).joinpath("iq6r-v4-qualification/pydeps")))
from iq6r_v4_fetch_hessians import manifest_entries
from iq6r_v4_prepare_grams import HEADER, MAGIC, ROWS, check_and_map

F32 = struct.Struct("<f")


def inspect_one(source: Path, binary: Path, expected_sha: str) -> dict:
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha:
        raise ValueError(f"source digest mismatch: {source.name}")
    with source.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        info = json.loads(handle.read(length))
    k = int(info["__metadata__"]["k"])
    prefix = 8 + length
    if not binary.is_file() or binary.stat().st_size != HEADER.size + 4*k*k:
        raise ValueError(f"truncated RADGRAM2: {binary.name}")
    with binary.open("rb") as gram:
        if gram.read(HEADER.size) != HEADER.pack(MAGIC, k, 0, ROWS, 0, 0):
            raise ValueError(f"RADGRAM2 metadata mismatch: {binary.name}")
    points = sorted({0, 1, 2, k // 2, k - 3, k - 2, k - 1})
    with source.open("rb") as src, binary.open("rb") as dest:
        for i in points:
            for j in points:
                lo, hi = sorted((i, j))
                packed_pos = lo * k - (lo * (lo - 1)) // 2 + hi - lo
                src.seek(prefix + packed_pos * 4)
                dest.seek(HEADER.size + (i * k + j) * 4)
                if src.read(4) != dest.read(4):
                    raise ValueError(f"Gram data mismatch in {binary.name} at ({i},{j})")
    return {"k": k, "cells": len(points) ** 2}


def audit(tree: Path, source: Path, output: Path, recipe: Path) -> dict:
    entries = manifest_entries(json.loads(tree.read_text()))
    mapping = check_and_map(entries, recipe)
    byfile = {}
    for stem, entry in mapping.items():
        byfile.setdefault(entry["path"], []).append(stem)
    cells = 0
    for entry in entries:
        stems = sorted(byfile[entry["path"]])
        binary = output / f"gram.r0.{stems[0]}.bin"
        check = inspect_one(source / Path(entry["path"]).name,
                            binary, entry["sha256"])
        cells += check["cells"]
        for other in stems[1:]:
            alias = output / f"gram.r0.{other}.bin"
            if not alias.is_file() or not alias.samefile(binary):
                raise ValueError(f"missing/shared calibration alias: {other}")
    return {"status": "SAMPLED_GRAMS_VERIFIED", "sources": len(entries),
            "converter_stems": len(mapping), "cells_compared": cells,
            "numerical_quality_validated": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    for key in ("tree", "source", "output", "recipe"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = audit(args.tree, args.source, args.output, args.recipe)
    value = json.dumps(result, indent=2) + "\n"
    if args.receipt:
        args.receipt.write_text(value)
    print(value, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
