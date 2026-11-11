"""VTP13 card-free original-BF16 Qwen3.8-27B GDN Q|K|V/conv repack.

The original GDN in_proj_qkv and depthwise conv tensors are grouped
as Q(all key heads)|K(all key heads)|V(all value heads). Each physical
rank needs its own Q|K|V in local head order; the original ranges are
non-contiguous. This reconstructs the complete original BF16 payload
byte-for-byte from the three packed rank files, with no GPU or quant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from datetime import UTC, datetime
from pathlib import Path

import xtx27_checkpoint_audit as audit
import xtx27_vtp13_ownership as ownership

ROOT = Path(__file__).resolve().parents[3]
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp13-gdn-pack"
ROLES = ("xtx0", "xtx1", "r9700")
OPS = {
    "in_proj_qkv": ([10240, 5120], 5120),
    "conv1d": ([10240, 1, 4], 4),
}
SCHEMA = "vtp13-gdn-qkv-conv-bf16-rank-pack-v1"


def layout() -> dict:
    k_counts = ownership.GDN_K_HEADS
    v_counts = ownership.GDN_V_HEADS
    assert k_counts == (4, 4, 8) and v_counts == (12, 12, 24)
    result = {}
    key_offset, value_offset = 0, 0
    for role, nk, nv in zip(ROLES, k_counts, v_counts, strict=True):
        q = (key_offset * 128, nk * 128)
        k = (2048 + key_offset * 128, nk * 128)
        v = (4096 + value_offset * 128, nv * 128)
        result[role] = [q, k, v]
        key_offset += nk
        value_offset += nv
    spans = sorted(seg for ranks in result.values() for seg in ranks)
    cursor = 0
    for start, width in spans:
        if start != cursor or width <= 0 or width % 128:
            raise ValueError("GDN source channel coverage gap, overlap or illegal tile")
        cursor += width
    if cursor != 10240:
        raise ValueError("GDN source channels incompletely covered")
    return result


def sources(checkpoint: Path, layer: int) -> dict:
    if type(layer) is not int or not 0 <= layer < 64 or layer % 4 == 3:
        raise ValueError("layer must be a GDN layer in 0..63")
    root = checkpoint.expanduser().resolve(strict=True)
    cfg, cfg_hash = audit.load_metadata(root / "config.json")
    ownership.expected_config(cfg)
    index, index_hash = audit.load_metadata(root / "model.safetensors.index.json")
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise TypeError("missing checkpoint weight map")
    entries = {}
    cache = {}
    for op, (shape, row_width) in OPS.items():
        name = f"model.language_model.layers.{layer}.linear_attn.{op}.weight"
        filename = index["weight_map"].get(name)
        if not audit.safe_filename(filename):
            raise ValueError(f"missing/unsafe original BF16 source: {name}")
        file = (root / filename).resolve(strict=True)
        if file.parent != root:
            raise ValueError("escaping safetensors filename")
        if file not in cache:
            meta = audit.shard_metadata(file)
            with file.open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                hdr = json.loads(stream.read(length), object_pairs_hook=audit.unique_keys)
            cache[file] = (meta, hdr, 8 + length)
        meta, hdr, base = cache[file]
        if meta["shapes"].get(name) != shape:
            raise ValueError(f"changed original BF16 source shape: {name}")
        tensor = hdr.get(name)
        if tensor is None or tensor["dtype"] != "BF16" or tensor["shape"] != shape:
            raise ValueError(f"bad BF16 source header: {name}")
        lo, hi = tensor["data_offsets"]
        if hi - lo != row_width * 10240 * 2:
            raise ValueError(f"unexpected tensor extent: {name}")
        entries[op] = {
            "source": name, "path": str(file), "file_basename": filename,
            "offset": base + lo, "bytes": hi - lo,
            "shape": shape, "row_width": row_width,
            "header_sha256": meta["header_sha256"],
        }
    return {
        "schema": SCHEMA, "layer": layer, "config_sha256": cfg_hash,
        "index_sha256": index_hash, "layout": layout(), "sources": entries,
    }


def sha256_bytes(path: Path, *, start: int = 0, size: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        file.seek(start)
        remaining = size
        while remaining is None or remaining:
            n = 4 * 1024 * 1024 if remaining is None else min(remaining, 4 * 1024 * 1024)
            chunk = file.read(n)
            if not chunk:
                if remaining:
                    raise ValueError("truncated payload while hashing")
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def pack(descriptor: dict, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to replace GDN pack receipt: {out}")
    if out == ROOT or ROOT in out.parents:
        raise ValueError("GDN pack output must be outside Git checkout")
    out.mkdir(parents=True, exist_ok=False)
    receipt = {
        **descriptor, "status": "INCOMPLETE",
        "created_utc": datetime.now(UTC).isoformat(),
        "writer_sha256": sha256_bytes(Path(__file__)),
        "outputs": {},
        "proof": "Original source recovered exactly from rank Q|K|V repack, no quantisation",
    }
    record = out / "summary.json"
    record.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        for op, desc in descriptor["sources"].items():
            source = Path(desc["path"])
            width = desc["row_width"]
            rowbytes = width * 2
            ranks = descriptor["layout"]
            reconstructed_segments = []
            payloads = {}
            for role in ROLES:
                target = out / f"{op}-{role}.bf16"
                with target.open("xb") as sink, source.open("rb") as stream:
                    for start, count in ranks[role]:
                        stream.seek(desc["offset"] + start * rowbytes)
                        remain = count * rowbytes
                        position = sink.tell()
                        while remain:
                            raw = stream.read(min(remain, 4 * 1024 * 1024))
                            if not raw:
                                raise ValueError(f"truncated original source {op}")
                            sink.write(raw)
                            remain -= len(raw)
                        reconstructed_segments.append((start, count, target, position))
                payloads[role] = {
                    "path": str(target), "bytes": target.stat().st_size,
                    "sha256": sha256_bytes(target),
                    "packed_segments": descriptor["layout"][role],
                }
            if len(reconstructed_segments) != 9:
                raise ValueError("expected exactly 9 rank/group segments")
            source_hash = sha256_bytes(source, start=desc["offset"], size=desc["bytes"])
            reconstructed = hashlib.sha256()
            cursor = 0
            for start, count, target, pos in sorted(reconstructed_segments):
                if start != cursor:
                    raise ValueError("reconstructed original has a source gap/overlap")
                size = count * rowbytes
                with target.open("rb") as data:
                    data.seek(pos)
                    remain = size
                    while remain:
                        block = data.read(min(4 * 1024 * 1024, remain))
                        if not block:
                            raise ValueError("packed rank file truncated")
                        reconstructed.update(block)
                        remain -= len(block)
                cursor += count
            if cursor != 10240 or reconstructed.hexdigest() != source_hash:
                raise ValueError("original BF16 Q|K|V payload reconstruction mismatch")
            receipt["outputs"][op] = {
                "source_payload_sha256": source_hash,
                "reassembled_source_sha256": reconstructed.hexdigest(),
                "complete_byte_equivalence": True,
                "ranks": payloads,
            }
            record.write_text(json.dumps(receipt, indent=2) + "\n")
        receipt["status"] = "EXACT_ORIGINAL_BF16_GDN_QKV_CONV_PACK_PASS_NOT_RUNTIME"
    except Exception as exc:
        receipt["status"] = "FAILED_INCOMPLETE"
        receipt["error"] = str(exc)
        record.write_text(json.dumps(receipt, indent=2) + "\n")
        raise
    record.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--phase", choices=("plan", "pack"), default="plan")
    p.add_argument("--run", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    if a.phase == "pack" and not a.run:
        p.error("packing original model payload requires explicit --run")
    descriptor = sources(a.checkpoint, a.layer)
    if a.phase == "plan":
        print(json.dumps({**descriptor, "status": "HEADER_ONLY_NO_GPU"}, indent=2))
        return 0
    if a.output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = SCRATCH / f"layer{a.layer}-{stamp}"
    else:
        path = a.output.expanduser().resolve()
    result = pack(descriptor, path)
    print(json.dumps({"status": result["status"], "receipt": str(path / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
