"""Convert pinned upper-triangle safetensors Hessians to RADGRAM2 for IQ6R-V4.

The pinned libquant GPTQ reader ONLY reads gram.r<rank>.<tensor-stem>.bin
with a 32-byte RADGRAM2 header and a full K*K row-major F32 payload.
Each model calibration site maps to one or more converter tensor stems.
Do not run concurrently with converter; completed output is immutable.

Usage:
  PYTHONPATH=$BC_WORK_ROOT/iq6r-v4-qualification/pydeps \
    python3 tools/lab/radiance/iq6r_v4_prepare_grams.py --input ... --output ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<8sIIQII")
MAGIC = b"RADGRAM2"
ROWS = 10_259_331
LAYER = re.compile(r"model[.]language_model[.]layers[.](\d+)[.](.*)[.]safetensors$")
KIND = {
    "input": ("ssm_inz",),
    "linear_attn.out_proj.input": ("ssm_out",),
    "self_attn.o_proj.input": ("attn_output",),
    "post_attention.input": ("ffn_gate_up",),
    "mlp.down_proj.input": ("ffn_down",),
}
HEAD_STEM = "output"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def names_for_site(name: str) -> tuple[str, ...]:
    if name == "lm_head.input.safetensors":
        return (HEAD_STEM,)
    match = LAYER.fullmatch(name)
    if not match:
        raise ValueError(f"unrecognized Hessian site: {name}")
    layer = int(match.group(1))
    if not 0 <= layer < 64:
        raise ValueError(f"invalid layer index: {layer}")
    site = match.group(2)
    if site == "input" and layer % 4 == 3:
        ops = ("attn_qg", "attn_k", "attn_v")
    elif site == "input":
        ops = KIND[site]
    else:
        ops = KIND[site]
    return tuple(f"blk.{layer}.{op}" for op in ops)


def check_and_map(entries: list[dict], recipe: Path) -> dict[str, dict]:
    actual = {}
    for entry in entries:
        for stem in names_for_site(Path(entry["path"]).name):
            if stem in actual:
                raise ValueError(f"duplicate calibration stem: {stem}")
            actual[stem] = entry
    requested = set()
    for line in recipe.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[1] != "gptq":
            continue
        token = parts[0]
        if not token.endswith(".weight"):
            raise ValueError(f"unexpected recipe name: {token}")
        stem = token.removesuffix(".weight")
        if stem.startswith("output.k"):
            stem = HEAD_STEM
        requested.add(stem)
    missing = sorted(requested - actual.keys())
    unused = sorted(actual.keys() - requested)
    if missing or unused:
        raise ValueError(f"calibration coverage: missing={missing} unused={unused}")
    return actual


def safe_header(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as stream:
        nraw = stream.read(8)
        if len(nraw) != 8:
            raise ValueError(f"short safetensors header: {path}")
        n = struct.unpack("<Q", nraw)[0]
        if not 0 < n < 100_000:
            raise ValueError(f"bad header length: {path}")
        data = json.loads(stream.read(n))
    meta = data["__metadata__"]
    tensor = data["hessian_upper"]
    k = int(meta["k"])
    rows = int(meta["rows"])
    expected = k * (k + 1) // 2
    if (rows != ROWS or tensor.get("dtype") != "F32"
            or tensor.get("shape") != [expected]
            or tensor.get("data_offsets") != [0, expected * 4]
            or path.stat().st_size != n + 8 + expected * 4):
        raise ValueError(f"unexpected Hessian geometry: {path}")
    return k, rows, n + 8


def convert_one(source: Path, dest: Path, aliases: tuple[str, ...]) -> None:
    k, rows, offset = safe_header(source)
    part = dest.with_suffix(".bin.incomplete")
    if part.exists() or dest.exists():
        raise FileExistsError(f"refusing overwrite of an existing RADGRAM file: {dest}")
    expected_bytes = HEADER.size + k * k * 4
    with part.open("xb") as out:
        out.write(HEADER.pack(MAGIC, k, 0, rows, 0, 0))
        out.truncate(expected_bytes)
    try:
        upper = np.memmap(source, dtype="<f4", mode="r", offset=offset,
                          shape=(k * (k + 1) // 2,))
        square = np.memmap(part, dtype="<f4", mode="r+", offset=HEADER.size,
                           shape=(k, k))
        # Write each full output row contiguously; avoid strided mmap writes.
        prior = np.arange(k, dtype=np.int64)
        bases = prior * k - prior * (prior - 1) // 2
        index = 0
        for i in range(k):
            if i:
                square[i, :i] = upper[bases[:i] + i - prior[:i]]
            n = k - i
            square[i, i:] = upper[index:index + n]
            index += n
        square.flush()
        del square, upper
        if part.stat().st_size != expected_bytes:
            raise ValueError(f"unexpected RADGRAM2 byte length: {part}")
        part.rename(dest)
        for alias in aliases:
            other = dest.with_name(f"gram.r0.{alias}.bin")
            if other.exists():
                raise FileExistsError(f"existing calibration alias: {other}")
            os.link(dest, other)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tree", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--recipe", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    from iq6r_v4_fetch_hessians import manifest_entries
    from iq6r_v4_validate_hessians import audit

    entries = manifest_entries(json.loads(args.tree.read_text()))
    audit(args.source, Path(os.environ["BC_MODEL_ROOT"]) / "qwen3.8-27b/Qwen3.8-27B", ROWS)
    mapping = check_and_map(entries, args.recipe)
    byfile: dict[str, list[str]] = {}
    for stem, entry in mapping.items():
        byfile.setdefault(entry["path"], []).append(stem)
    if len(byfile) != 257 or len(mapping) != 293 - 4:
        # four head rule stems collapse to one; other projections all distinct
        raise ValueError(f"unexpected site/stem coverage: {len(byfile)}, {len(mapping)}")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "adapter-progress.json"
    done = 0
    for entry in entries:
        stems = sorted(byfile[entry["path"]])
        source = args.source / Path(entry["path"]).name
        if sha(source) != entry["sha256"]:
            raise ValueError(f"input checksum mismatch: {source}")
        primary = args.output / f"gram.r0.{stems[0]}.bin"
        aliases = tuple(stems[1:])
        if primary.exists():
            k, rows, _ = safe_header(source)
            if primary.stat().st_size != HEADER.size + k * k * 4:
                raise ValueError(f"invalid partial output: {primary}")
            with primary.open("rb") as stream:
                header = stream.read(HEADER.size)
            if header != HEADER.pack(MAGIC, k, 0, rows, 0, 0):
                raise ValueError(f"RADGRAM2 header mismatch: {primary}")
            for alias in aliases:
                other = args.output / f"gram.r0.{alias}.bin"
                if not other.exists():
                    os.link(primary, other)
                elif not os.path.samefile(primary, other):
                    raise ValueError(f"alias not same calibration file: {other}")
        else:
            convert_one(source, primary, aliases)
        done += 1
        checkpoint.write_text(json.dumps({"completed": done, "total": len(entries),
                                          "last": entry["path"]}, indent=2) + "\n")
    print(f"RADGRAM2 calibration complete: {done} source sites, {len(mapping)} stems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
