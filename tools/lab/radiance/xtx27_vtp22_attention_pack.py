"""VTP22: byte-exact original-BF16 full-attention source pack for physical TP3.

Qwen3.8-27B has 16 full attention layers at layer index 3 mod 4.
Each full-attention Q head owns an interleaved [query256, gate256]
pair of BF16 source rows, so the physical Q/Gate row partitions
are 6/6/12 * 512 = 3072/3072/6144, NOT 6/6/12 * 256.
K and V have four 256-wide heads, split 1/1/2. The output weight
matrix [5120,6144] splits by INPUT columns 1536/1536/3072.
Q and K 256-wide norm vectors are identical replicated sources.

The extractor reads the real checkpoint only, validates the source
safetensors geometry and cryptographically attests each selected
payload and output shard. It reconstructs the full matrices byte-for-
byte from the three rank files, including the strided O projection.

One synthetic BF16 hidden input is normalized by the ORIGINAL
layer-specific BF16 weighted RMSNorm, then shared across ranks.
No GPU, flash attention, RoPE computation, KV-cache, decoder,
checkpoint inference or model quality claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

import xtx27_checkpoint_audit as audit
import xtx27_vtp13_ownership as ownership
import xtx27_vtp15_gdn_projection as projection

ROOT = projection.ROOT
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp22-original-full-attention"
SCHEMA = "vtp22-qwen38-original-BF16-full-attention-physical-TP3-weight-pack-v1"
ROLES = ownership.ROLES
HIDDEN = 5120
HEAD = 256
QUERIES = (6, 6, 12)
KVS = (1, 1, 2)
SPECS = {
    "q": ("self_attn.q_proj.weight", [12288, HIDDEN]),
    "k": ("self_attn.k_proj.weight", [1024, HIDDEN]),
    "v": ("self_attn.v_proj.weight", [1024, HIDDEN]),
    "o": ("self_attn.o_proj.weight", [HIDDEN, 6144]),
    "q_norm": ("self_attn.q_norm.weight", [HEAD]),
    "k_norm": ("self_attn.k_norm.weight", [HEAD]),
    "input_norm": ("input_layernorm.weight", [HIDDEN]),
}


def ownership_map() -> dict:
    if len(ROLES) != 3 or sum(QUERIES) != 24 or sum(KVS) != 4:
        raise ValueError("full-attention physical rank head ownership mismatch")
    q_begin = kv_begin = o_begin = 0
    parts = {}
    for role, q_heads, kv_heads in zip(ROLES, QUERIES, KVS, strict=True):
        q = (q_begin, q_heads * HEAD * 2)  # Q + interleaved gate per head
        kv = (kv_begin, kv_heads * HEAD)
        o = (o_begin, q_heads * HEAD)
        if q_heads != kv_heads * 6:
            raise ValueError("each original KV head must own exactly six query heads")
        parts[role] = {
            "q": q, "k": kv, "v": kv, "o": o,
            "query_heads": q_heads, "kv_heads": kv_heads,
            "query_gate_interleaved": True,
        }
        q_begin += q[1]
        kv_begin += kv[1]
        o_begin += o[1]
    if (q_begin, kv_begin, o_begin) != (12288, 1024, 6144):
        raise ValueError("full attention rank coverage mismatch")
    return parts


def sources(checkpoint: Path, layer: int) -> dict:
    if type(layer) is not int or not 0 <= layer < 64 or layer % 4 != 3:
        raise ValueError("full attention layer must be 3,7,...,63")
    root = checkpoint.expanduser().resolve(strict=True)
    config, config_hash = audit.load_metadata(root / "config.json")
    cfg = ownership.expected_config(config)
    if cfg["layer_types"][layer] != "full_attention":
        raise ValueError("selected layer is not full attention")
    index, index_hash = audit.load_metadata(root / "model.safetensors.index.json")
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise TypeError("original checkpoint safetensors index invalid")
    cache = {}
    tensors = {
        key: projection._source(root, index["weight_map"], layer, key, spec, cache)
        for key, spec in SPECS.items()
    }
    return {
        "layer": layer, "checkpoint": str(root), "config_sha256": config_hash,
        "index_sha256": index_hash, "sources": tensors, "ownership": ownership_map(),
    }


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rows_shards(data: bytes, row_width: int, counts: tuple[int, ...]) -> tuple[bytes, ...]:
    if (type(row_width) is not int or row_width < 1
            or any(type(x) is not int or x < 1 for x in counts)
            or sum(counts) * row_width * 2 != len(data)):
        raise ValueError("original BF16 row shard geometry invalid")
    start = 0
    result = []
    for count in counts:
        stop = start + count * row_width * 2
        result.append(data[start:stop])
        start = stop
    if b"".join(result) != data:
        raise ValueError("original row-sharded source failed reconstruction")
    return tuple(result)


def col_shards(data: bytes, rows: int, width: int,
               counts: tuple[int, ...]) -> tuple[bytes, ...]:
    if (type(rows) is not int or rows < 1 or type(width) is not int or width < 1
            or any(type(x) is not int or x < 1 for x in counts)
            or sum(counts) != width or len(data) != rows * width * 2):
        raise ValueError("original BF16 column shard geometry invalid")
    output = [bytearray(rows * count * 2) for count in counts]
    reconstructed = bytearray(len(data))
    col = 0
    for idx, count in enumerate(counts):
        for row in range(rows):
            src = (row * width + col) * 2
            dst = row * count * 2
            span = data[src:src + count * 2]
            output[idx][dst:dst + count * 2] = span
            reconstructed[src:src + count * 2] = span
        col += count
    if bytes(reconstructed) != data:
        raise ValueError("original column-sharded source failed reconstruction")
    return tuple(bytes(x) for x in output)


def _output_record(file: Path, data: bytes, shape: list[int], *, source: dict | None = None) -> dict:
    if file.exists() or math.prod(shape) * 2 != len(data):
        raise ValueError(f"refusing wrong-size or existing output shard: {file}")
    file.write_bytes(data)
    output = {
        "file": str(file), "shape": shape, "bytes": len(data),
        "sha256": sha(data),
    }
    if source is not None:
        output["source"] = source
    return output


def pack(checkpoint: Path, layer: int, rows: int, destination: Path) -> dict:
    if type(rows) is not int or not 1 <= rows <= 8:
        raise ValueError("original BF16 attention reference limited to M1..M8")
    source = sources(checkpoint, layer)
    dest = destination.expanduser().resolve()
    if dest.exists() or dest == ROOT or ROOT in dest.parents:
        raise ValueError("immutable fresh output directory outside Git required")
    dest.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": SCHEMA, "status": "INCOMPLETE", "layer": layer,
        "rows": rows, "created_utc": datetime.now(UTC).isoformat(),
        "writer_sha256": projection.sha(Path(__file__)),
        **source,
        "original_tensor_sha256": {}, "rank_weights": {r: {} for r in ROLES},
        "replicated_norms": {}, "hidden_input": {},
        "limitations": [
            "actual original BF16 checkpoint payloads but synthetic hidden input",
            "interleaved full-attention output gate is not executed",
            "no RoPE, causal softmax or KV cache yet",
            "not a GPU kernel, native engine or full 27B inference",
        ],
    }
    receipt = dest / "summary.json"
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    try:
        weights = {}
        for key, descriptor in source["sources"].items():
            raw = projection.raw_tensor(descriptor)
            if len(raw) != descriptor["bytes"]:
                raise ValueError(f"changed source tensor extent: {key}")
            record["original_tensor_sha256"][key] = sha(raw)
            weights[key] = raw
        parts = source["ownership"]
        q_counts = tuple(parts[r]["q"][1] for r in ROLES)
        kv_counts = tuple(parts[r]["k"][1] for r in ROLES)
        o_counts = tuple(parts[r]["o"][1] for r in ROLES)
        by_op = {
            "q": rows_shards(weights["q"], HIDDEN, q_counts),
            "k": rows_shards(weights["k"], HIDDEN, kv_counts),
            "v": rows_shards(weights["v"], HIDDEN, kv_counts),
            "o": col_shards(weights["o"], HIDDEN, 6144, o_counts),
        }
        for op, chunks in by_op.items():
            for role, chunk in zip(ROLES, chunks, strict=True):
                nrows, ncols = (
                    (HIDDEN, parts[role]["o"][1])
                    if op == "o" else (parts[role][op][1], HIDDEN)
                )
                file = dest / f"{op}-{role}.bf16"
                record["rank_weights"][role][op] = _output_record(
                    file, chunk, [nrows, ncols],
                    source={"row_or_col": "input_columns" if op == "o" else "output_rows",
                            "offset": parts[role][op][0],
                            "length": parts[role][op][1]},
                )
        # Norms are physically replicated, not sliced into three separate heads.
        for op in ("q_norm", "k_norm", "input_norm"):
            file = dest / f"{op}.bf16"
            record["replicated_norms"][op] = _output_record(
                file, weights[op], source["sources"][op]["shape"]
            )
        hidden, normal = projection.norm_input(weights["input_norm"], rows)
        for name, blob in (("hidden_input", hidden), ("normed_input", normal)):
            record["hidden_input"][name] = _output_record(
                dest / f"{name}.bf16", blob, [rows, HIDDEN],
            )
        record["status"] = "ORIGINAL_BF16_FULL_ATTENTION_TP3_BYTE_EXACT_PACK_PASS_NO_GPU"
    except Exception as err:
        record.update({"status": "FAILED_INCOMPLETE", "error": str(err)})
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        raise
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    return record


def validate(receipt: Path) -> dict:
    record = json.loads(receipt.read_text())
    if (record.get("schema") != SCHEMA
            or record.get("status") != "ORIGINAL_BF16_FULL_ATTENTION_TP3_BYTE_EXACT_PACK_PASS_NO_GPU"
            or record.get("writer_sha256") != projection.sha(Path(__file__))
            or record.get("ownership") != json.loads(json.dumps(ownership_map()))
            or type(record.get("rows")) is not int or not 1 <= record["rows"] <= 8):
        raise ValueError("original BF16 full attention pack provenance invalid")
    if record["layer"] % 4 != 3:
        raise ValueError("not an original full-attention layer")
    for role in ROLES:
        for op in ("q", "k", "v", "o"):
            item = record["rank_weights"][role][op]
            path = Path(item["file"])
            if (path.stat().st_size != item["bytes"] or sha(path.read_bytes()) != item["sha256"]
                    or math.prod(item["shape"]) * 2 != item["bytes"]):
                raise ValueError(f"changed original rank weight: {role}.{op}")
    for key in ("q_norm", "k_norm", "input_norm"):
        item = record["replicated_norms"][key]
        path = Path(item["file"])
        if path.stat().st_size != item["bytes"] or sha(path.read_bytes()) != item["sha256"]:
            raise ValueError(f"changed original replicated norm {key}")
    for key in ("hidden_input", "normed_input"):
        item = record["hidden_input"][key]
        path = Path(item["file"])
        if (item["shape"] != [record["rows"], HIDDEN]
                or path.stat().st_size != item["bytes"]
                or sha(path.read_bytes()) != item["sha256"]):
            raise ValueError(f"changed synthetic incoming hidden: {key}")
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--phase", choices=("plan", "pack"), default="plan")
    p.add_argument("--run", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    if a.phase == "pack" and not a.run:
        p.error("reading and packaging original checkpoint payload requires explicit --run")
    try:
        src = sources(a.checkpoint, a.layer)
        if a.phase == "plan":
            print(json.dumps({
                "schema": SCHEMA, "phase": "card_free",
                "layer": src["layer"], "sources": src["sources"],
                "ownership": src["ownership"], "rows": a.rows,
                "real_weight_payloads_read": False, "GPU_required": False,
            }, indent=2))
            return 0
        dest = a.output or SCRATCH / datetime.now(UTC).strftime(
            f"layer{a.layer}-m{a.rows}-%Y%m%dT%H%M%S%fZ"
        )
        result = pack(a.checkpoint, a.layer, a.rows, dest)
        print(json.dumps({
            "status": result["status"],
            "receipt": str(dest.resolve() / "summary.json"),
            "layer": result["layer"], "rows": result["rows"],
            "original_tensor_sha256": result["original_tensor_sha256"],
        }, indent=2))
        return 0
    except (ValueError, OSError, TypeError, KeyError) as e:
        p.error(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
