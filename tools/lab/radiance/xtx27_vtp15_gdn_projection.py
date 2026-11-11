"""VTP15: pack original BF16 GDN input projections for asymmetric physical TP3.

This is the next actual-model integration dependency. It extracts and checks
one layer's in_proj_qkv/a/b/z projection weights plus replicated
input_layernorm from the ORIGINAL safetensors checkpoint. It reuses the
previous byte-verified QKV rank packs and produces exactly owned rank-local
a/b/z weights. It also emits a deterministic BF16 hidden stream and the
model's real weighted pre-RMSNorm result.

No GPU or model serving; no quantisation; no full-model quality claim.
Original shapes: hidden=5120, QKV=10240, A/B=48, Z=6144.
Output order per physical rank: Q|K|V, A, B, Z. CPU/GPU projection
kernels consume this package at their next execution gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from datetime import UTC, datetime
from pathlib import Path

import xtx27_checkpoint_audit as audit
import xtx27_vtp13_gdn_pack as gdn
import xtx27_vtp13_ownership as own
import xtx27_vtp14_gdn_exact as exact

ROOT = Path(__file__).resolve().parents[3]
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp15-gdn-projection"
SCHEMA = "vtp15-qwen38-gdn-original-bf16-projection-package-v1"
HIDDEN = 5120
SPECS = {
    "a": ("linear_attn.in_proj_a.weight", [48, HIDDEN]),
    "b": ("linear_attn.in_proj_b.weight", [48, HIDDEN]),
    "z": ("linear_attn.in_proj_z.weight", [6144, HIDDEN]),
    "norm": ("input_layernorm.weight", [HIDDEN]),
}


def ranges() -> dict:
    widths = own.GDN_V_HEADS
    cursor = 0
    out = {}
    for role, n in zip(gdn.ROLES, widths, strict=True):
        out[role] = {
            "a": (cursor, n),
            "b": (cursor, n),
            "z": (cursor * 128, n * 128),
        }
        cursor += n
    if cursor != 48:
        raise ValueError("invalid GDN value head coverage")
    for role_name in ("a", "b", "z"):
        parts = sorted(layout[role_name] for layout in out.values())
        offset = 0
        for begin, count in parts:
            if begin != offset or count <= 0:
                raise ValueError(f"{role_name}: gap/overlap")
            offset += count
        if offset != SPECS[role_name][1][0]:
            raise ValueError(f"{role_name}: incomplete ownership")
    return out


def _source(root: Path, weights: dict, layer: int, key: str,
            spec: tuple[str, list[int]], cache: dict) -> dict:
    suffix, shape = spec
    name = f"model.language_model.layers.{layer}.{suffix}"
    filename = weights.get(name)
    if not audit.safe_filename(filename):
        raise ValueError(f"invalid or absent source {name}")
    file = (root / filename).resolve(strict=True)
    if file.parent != root:
        raise ValueError("checkpoint filename escaped root")
    if file not in cache:
        metadata = audit.shard_metadata(file)
        with file.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_size), object_pairs_hook=audit.unique_keys)
        cache[file] = metadata, header, 8 + header_size
    meta, header, start_offset = cache[file]
    entry = header.get(name)
    if (not isinstance(entry, dict) or entry.get("dtype") != "BF16"
            or entry.get("shape") != shape or meta["shapes"].get(name) != shape):
        raise ValueError(f"unexpected original BF16 shape/dtype: {name}")
    first, last = entry["data_offsets"]
    nbytes = math.prod(shape) * 2
    if last - first != nbytes:
        raise ValueError(f"invalid selected BF16 tensor extent: {name}")
    return {
        "name": name, "path": str(file), "file_basename": filename,
        "offset": start_offset + first, "bytes": nbytes,
        "shape": shape, "header_sha256": meta["header_sha256"],
    }


def sources(checkpoint: Path, layer: int, pack_receipt: Path) -> tuple[dict, dict]:
    if type(layer) is not int or layer < 0 or layer >= 64 or layer % 4 == 3:
        raise ValueError("expected Qwen3.8 GDN layer 0..63")
    root = checkpoint.expanduser().resolve(strict=True)
    config, config_sha = audit.load_metadata(root / "config.json")
    own.expected_config(config)
    index, index_sha = audit.load_metadata(root / "model.safetensors.index.json")
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise TypeError("missing original checkpoint weight map")
    receipt = json.loads(pack_receipt.read_text())
    if (receipt.get("status") != "EXACT_ORIGINAL_BF16_GDN_QKV_CONV_PACK_PASS_NOT_RUNTIME"
            or receipt.get("layer") != layer or receipt.get("config_sha256") != config_sha
            or receipt.get("index_sha256") != index_sha):
        raise ValueError("prior original BF16 QKV pack receipt is incompatible")
    origin = gdn.sources(root, layer)
    if origin["sources"] != receipt["sources"]:
        raise ValueError("prior original BF16 pack metadata does not match checkpoint")
    cache = {}
    descriptors = {
        key: _source(root, index["weight_map"], layer, key, spec, cache)
        for key, spec in SPECS.items()
    }
    return {
        "schema": SCHEMA, "layer": layer,
        "checkpoint": str(root), "config_sha256": config_sha,
        "index_sha256": index_sha, "source_descriptors": descriptors,
        "qkv_source": origin["sources"]["in_proj_qkv"],
        "qkv_pack_receipt": str(pack_receipt.resolve(strict=True)),
        "ownership": ranges(),
    }, receipt


def sha(path: Path, offset: int = 0, count: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(offset)
        left = count
        while left is None or left:
            buf = f.read(4 * 1024 * 1024 if left is None else min(left, 4 * 1024 * 1024))
            if not buf:
                if left:
                    raise ValueError("truncated source during payload hash")
                break
            h.update(buf)
            if left is not None:
                left -= len(buf)
    return h.hexdigest()


def raw_tensor(source: dict) -> bytes:
    path = Path(source["path"])
    with path.open("rb") as stream:
        stream.seek(source["offset"])
        data = stream.read(source["bytes"])
    if len(data) != source["bytes"]:
        raise ValueError("selected original BF16 source truncated")
    return data


def initial_hidden(t: int, i: int) -> float:
    return ((t * 11 + i * 7) % 29 - 14) / 64.0


def norm_input(norm_raw: bytes, rows: int) -> tuple[bytes, bytes]:
    if len(norm_raw) != HIDDEN * 2 or not 1 <= rows <= 8:
        raise ValueError("wrong original norm BF16 shape or rows")
    weights = exact.bf16_values(norm_raw)
    hidden, normal = bytearray(), bytearray()
    for row in range(rows):
        values = [initial_hidden(row, i) for i in range(HIDDEN)]
        ss = sum(x * x for x in values)
        scale = 1.0 / math.sqrt(ss / HIDDEN + 1e-6)
        for i, x in enumerate(values):
            # Qwen3.5 RMSNorm has weight_add=1.0 (pinned qwen35_bf16).
            for dest, value in ((hidden, x), (normal, x * scale * (1 + weights[i]))):
                v = exact.round_bf16(value)
                bits = struct.unpack("<I", struct.pack("<f", v))[0] >> 16
                dest.extend(struct.pack("<H", bits))
    return bytes(hidden), bytes(normal)


def write(checkpoint: Path, layer: int, pack_receipt: Path,
          rows: int, destination: Path) -> dict:
    if not 1 <= rows <= 8:
        raise ValueError("rows must be within 1..8")
    root = ROOT.resolve()
    dest = destination.expanduser().resolve()
    if dest.exists() or dest == root or root in dest.parents:
        raise ValueError("pack output must be a new directory outside Git")
    desc, prior = sources(checkpoint, layer, pack_receipt)
    dest.mkdir(parents=True, exist_ok=False)
    result = {
        **desc, "status": "INCOMPLETE", "rows": rows,
        "created_utc": datetime.now(UTC).isoformat(),
        "writer_sha256": sha(Path(__file__)), "rank_packs": {},
        "source_sha256": {}, "inputs": {},
        "limitations": "real BF16 weight sharding and real original pre-RMSNorm;"
                       " not actual projection GEMM or model serving",
    }
    receipt_path = dest / "summary.json"
    receipt_path.write_text(json.dumps(result, indent=2) + "\n")
    try:
        for name, origin in desc["source_descriptors"].items():
            raw = raw_tensor(origin)
            result["source_sha256"][name] = hashlib.sha256(raw).hexdigest()
            if name == "norm":
                raw_hidden, raw_normed = norm_input(raw, rows)
                for title, values in (("hidden_input.bf16", raw_hidden),
                                      ("normed_input.bf16", raw_normed)):
                    with (dest / title).open("xb") as sink:
                        sink.write(values)
                    result["inputs"][title] = {
                        "path": str(dest / title), "sha256": hashlib.sha256(values).hexdigest(),
                        "bytes": len(values), "shape": [rows, HIDDEN],
                    }
                continue
            recon = bytearray(origin["bytes"])
            covered = bytearray(origin["shape"][0])
            for role in gdn.ROLES:
                begin, count = desc["ownership"][role][name]
                start, stop = begin * HIDDEN * 2, (begin + count) * HIDDEN * 2
                part = raw[start:stop]
                output = dest / f"{name}-{role}.bf16"
                with output.open("xb") as sink:
                    sink.write(part)
                result["rank_packs"].setdefault(role, {})[name] = {
                    "file": str(output), "sha256": hashlib.sha256(part).hexdigest(),
                    "bytes": len(part), "shape": [count, HIDDEN],
                    "source_rows": [begin, begin + count],
                }
                if any(covered[begin:begin + count]):
                    raise ValueError(f"{name}: overlapping output rows")
                covered[begin:begin + count] = b"\x01" * count
                recon[start:stop] = part
            if any(x != 1 for x in covered) or recon != raw:
                raise ValueError(f"{name}: sharded BF16 tensor failed exact reconstruction")
        for role in gdn.ROLES:
            item = prior["outputs"]["in_proj_qkv"]["ranks"][role]
            source = Path(item["path"])
            if sha(source) != item["sha256"]:
                raise ValueError(f"{role} previous QKV pack has changed")
            channels = sum(count for _, count in desc["ownership"][role].values())  # not QKV count
            del channels
            result["rank_packs"][role]["qkv"] = {
                "file": str(source), "sha256": item["sha256"],
                "bytes": item["bytes"], "shape": [
                    sum(n for _, n in gdn.layout()[role]), HIDDEN,
                ],
                "source_segments": gdn.layout()[role],
            }
        result["status"] = "EXACT_ORIGINAL_BF16_GDN_INPUT_PROJECTION_PACK_PASS_NOT_GPU"
    except Exception as exc:
        result.update({"status": "FAILED_INCOMPLETE", "error": str(exc)})
        receipt_path.write_text(json.dumps(result, indent=2) + "\n")
        raise
    receipt_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--qkv-pack-receipt", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--phase", choices=("plan", "pack"), default="plan")
    p.add_argument("--run", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    if a.phase == "pack" and not a.run:
        p.error("packing original model weights needs --run")
    if not 1 <= a.rows <= 8:
        p.error("rows must be in 1..8")
    try:
        if a.phase == "plan":
            desc, _ = sources(a.checkpoint, a.layer, a.qkv_pack_receipt)
            print(json.dumps({**desc, "status": "HEADER_ONLY_NO_GPU", "rows": a.rows}, indent=2))
            return 0
        dest = (a.output if a.output else SCRATCH / datetime.now(UTC).strftime(
            f"layer{a.layer}-%Y%m%dT%H%M%S%fZ"))
        receipt = write(a.checkpoint, a.layer, a.qkv_pack_receipt, a.rows, dest)
        print(json.dumps({"status": receipt["status"],
                          "receipt": str(dest.resolve() / "summary.json")}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
