"""VTP16: exact original-BF16 GDN output-column shards and residual ingress.

The original Qwen3.8-27B GDN out_proj has shape [5120,6144].
Three physical ranks own 12/12/24 consecutive 128-wide value heads,
so out_proj is split along INPUT COLUMNS [1536,1536,3072], not
OUTPUT ROWS. Every rank computes a partial 5120-vector; the host
collective combines those before adding the original residual.

Build a card-free, source-hashed BF16 package from the real layer-0
checkpoint weights and the independent VTP15 GDN-core output oracle.
The GDN-core FP32 values are rounded once to BF16 for the output GEMV.
The residual is the original deterministic hidden stream. This does
NOT execute a GPU, a full model layer, or native Radiance all-reduce.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from datetime import UTC, datetime
from pathlib import Path

import xtx27_vtp14_gdn_exact as exact
import xtx27_vtp15_gdn_projection as projection
import xtx27_vtp15_projection_ref as ref

ROOT = projection.ROOT
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp16-gdn-output"
SCHEMA = "vtp16-qwen38-original-bf16-gdn-output-column-package-v1"
ROLES = ("xtx0", "xtx1", "r9700")
HEADS = (12, 12, 24)
HIDDEN = 5120
K = 6144
SPLITS = (1536, 1536, 3072)


def split_columns() -> dict[str, tuple[int, int]]:
    result = {}
    off = 0
    for role, heads, width in zip(ROLES, HEADS, SPLITS, strict=True):
        if width != heads * 128 or width <= 0:
            raise ValueError("output input-channel ownership differs from value heads")
        result[role] = (off, width)
        off += width
    if off != K:
        raise ValueError("GDN output matrix column partition is incomplete")
    return result


def validate_inputs(weights_receipt: Path, core_receipt: Path) -> tuple[dict, dict, dict]:
    w = ref.validate(weights_receipt)
    core = json.loads(core_receipt.read_text())
    if (core.get("schema") != "vtp15-qwen38-27b-original-bf16-GDN-core-CPU-global-shard-v1"
            or core.get("status") != "ORIGINAL_BF16_QWEN_GDN_CORE_CPU_GLOBAL_SHARD_PASS_NOT_GPU"
            or core.get("weight_pack_sha256") != ref.sha(weights_receipt)
            or core.get("rows") != w["rows"] or core.get("layer") != w["layer"]):
        raise ValueError("original BF16 GDN core provenance mismatch")
    if core.get("writer_sha256") != ref.sha(Path(__file__).with_name("xtx27_vtp15_gdn_bridge.py")):
        raise ValueError("GDN core CPU reference writer differs from receipt")
    if not 1 <= w["rows"] <= 8:
        raise ValueError("bounded to M1-M8")
    for role, heads in zip(ROLES, HEADS, strict=True):
        item = core["rank_outputs"][role]
        path = Path(item["output"])
        if (item["shape"] != [heads, w["rows"], 128]
                or path.stat().st_size != heads * w["rows"] * 128 * 4
                or ref.sha(path) != item["sha256"]):
            raise ValueError(f"{role}: changed original GDN core output")
    hidden = w["inputs"]["hidden_input.bf16"]
    path = Path(hidden["path"])
    if (path.stat().st_size != w["rows"] * HIDDEN * 2
            or ref.sha(path) != hidden["sha256"]):
        raise ValueError("original BF16 residual input changed")
    root = Path(w["checkpoint"])
    index, index_sha = projection.audit.load_metadata(root / "model.safetensors.index.json")
    if index_sha != w["index_sha256"]:
        raise ValueError("checkpoint source index does not match earlier original pack")
    descriptor = projection._source(
        root, index["weight_map"], w["layer"], "out",
        ("linear_attn.out_proj.weight", [HIDDEN, K]), {},
    )
    return w, core, descriptor


def round_to_bf16(value: float) -> bytes:
    rounded = exact.round_bf16(value)
    bits = struct.unpack("<I", struct.pack("<f", rounded))[0] >> 16
    return struct.pack("<H", bits)


def pack(weights_receipt: Path, core_receipt: Path, output: Path) -> dict:
    w, core, descriptor = validate_inputs(weights_receipt, core_receipt)
    dest = output.expanduser().resolve()
    if dest.exists() or dest == ROOT or ROOT in dest.parents:
        raise ValueError("new out-of-repository receipt directory required")
    dest.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": SCHEMA, "status": "INCOMPLETE",
        "created_utc": datetime.now(UTC).isoformat(),
        "writer_sha256": ref.sha(Path(__file__)),
        "layer": w["layer"], "rows": w["rows"],
        "weight_input_receipt": str(weights_receipt.resolve()),
        "weight_input_sha256": ref.sha(weights_receipt),
        "core_receipt": str(core_receipt.resolve()),
        "core_sha256": ref.sha(core_receipt),
        "output_source": descriptor, "original_weights_sha256": None,
        "rank_weights": {}, "rank_inputs": {}, "global_input": None,
        "residual": None,
        "limitations": [
            "card-free only; no three-GPU output projection run",
            "GDN core FP32 output is converted to BF16 once for output GEMV",
            "synthetic first-layer incoming hidden, not a real sequence",
            "no native quantisation, engine dispatch or model quality claim",
        ],
    }
    path = dest / "summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    try:
        original = projection.raw_tensor(descriptor)
        if len(original) != HIDDEN * K * 2:
            raise ValueError("original out_proj payload length mismatch")
        result["original_weights_sha256"] = hashlib.sha256(original).hexdigest()
        original_file = dest / "out-original.bf16"
        original_file.write_bytes(original)
        result["original_weights_file"] = str(original_file)
        reconstruction = bytearray(len(original))
        parts = split_columns()
        for role, (offset, width) in parts.items():
            partial = bytearray(HIDDEN * width * 2)
            for row in range(HIDDEN):
                whole_start = (row * K + offset) * 2
                partial_start = row * width * 2
                chunk = original[whole_start:whole_start + width * 2]
                partial[partial_start:partial_start + width * 2] = chunk
                reconstruction[whole_start:whole_start + width * 2] = chunk
            shard = dest / f"out-{role}.bf16"
            shard.write_bytes(partial)
            result["rank_weights"][role] = {
                "file": str(shard), "sha256": hashlib.sha256(partial).hexdigest(),
                "bytes": len(partial), "shape": [HIDDEN, width],
                "source_columns": [offset, width],
            }
        if bytes(reconstruction) != original:
            raise ValueError("column-sharded original output weights did not reconstruct")
        global_rows = [bytearray(K * 2) for _ in range(w["rows"])]
        off = 0
        for role, heads in zip(ROLES, HEADS, strict=True):
            record = core["rank_outputs"][role]
            headmajor = struct.unpack(
                f"<{heads * w['rows'] * 128}f", Path(record["output"]).read_bytes()
            )
            local_rows = [bytearray(heads * 128 * 2) for _ in range(w["rows"])]
            for h in range(heads):
                for token in range(w["rows"]):
                    for d in range(128):
                        encoded = round_to_bf16(headmajor[(h * w["rows"] + token) * 128 + d])
                        start = (h * 128 + d) * 2
                        local_rows[token][start:start + 2] = encoded
                        glob = (off + h * 128 + d) * 2
                        global_rows[token][glob:glob + 2] = encoded
            local = b"".join(local_rows)
            fp = dest / f"gdn-{role}.bf16"
            fp.write_bytes(local)
            result["rank_inputs"][role] = {
                "file": str(fp), "sha256": hashlib.sha256(local).hexdigest(),
                "bytes": len(local), "shape": [w["rows"], heads * 128],
                "heads": heads,
            }
            off += heads * 128
        if off != K:
            raise ValueError("input head ownership incomplete")
        global_data = b"".join(global_rows)
        glob = dest / "gdn-original.bf16"
        glob.write_bytes(global_data)
        result["global_input"] = {
            "file": str(glob), "sha256": hashlib.sha256(global_data).hexdigest(),
            "bytes": len(global_data), "shape": [w["rows"], K],
        }
        hidden = w["inputs"]["hidden_input.bf16"]
        result["residual"] = {
            "file": hidden["path"], "sha256": hidden["sha256"],
            "bytes": hidden["bytes"], "shape": [w["rows"], HIDDEN],
        }
        result["status"] = "ORIGINAL_BF16_GDN_OUTPUT_COL_SHARDS_AND_RESIDUAL_INPUT_PASS_NO_GPU"
    except Exception as exc:
        result.update({"status": "FAILED_INCOMPLETE", "error": str(exc)})
        path.write_text(json.dumps(result, indent=2) + "\n")
        raise
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-receipt", required=True, type=Path)
    parser.add_argument("--core-receipt", required=True, type=Path)
    parser.add_argument("--phase", choices=("plan", "pack"), default="plan")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args(argv)
    if a.phase == "pack" and not a.run:
        parser.error("original checkpoint output needs explicit --run")
    try:
        w, core, src = validate_inputs(a.weights_receipt, a.core_receipt)
        if a.phase == "plan":
            print(json.dumps({
                "status": "ORIGINAL_BF16_OUTPUT_HEADER_VALIDATED_CARD_FREE",
                "output_shape": [HIDDEN, K],
                "source": src, "rows": w["rows"], "layer": w["layer"],
                "column_ownership": split_columns(),
                "core_schema": core["schema"],
                "gpu_required": False,
            }, indent=2))
            return 0
        dest = a.output if a.output else SCRATCH / datetime.now(UTC).strftime(
            f"layer{w['layer']}-%Y%m%dT%H%M%S%fZ"
        )
        record = pack(a.weights_receipt, a.core_receipt, dest)
        print(json.dumps({
            "status": record["status"], "receipt": str(dest.resolve() / "summary.json"),
            "original_weight_sha256": record["original_weights_sha256"],
        }, indent=2))
        return 0
    except (ValueError, TypeError, KeyError, OSError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
