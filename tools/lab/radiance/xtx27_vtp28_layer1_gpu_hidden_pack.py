"""VTP28: feed real layer-0 GPU output into original BF16 layer-1 GDN.

Consumes source-hashed VTP27 complete original BF16 GDN+FFN
physical-TP3 GPU layer-0 receipt and its ACTUAL GPU-produced final
hidden state (M1, 5120 BF16). Read original checkpoint layer-1
in_proj_qkv/a/b/z and input_layernorm BF16 weights, with exact
physical 4/4/8 key and 12/12/24 value head ownership.
Compute layer-1 input RMSNorm using the actual layer-0 GPU hidden;
not a deterministic synthetic activation and not a CPU-oracle
layer-0 replacement.

No GPU executed in this packer. No layer-1 inference yet.
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

import xtx27_vtp13_gdn_pack as gdn
import xtx27_vtp14_gdn_exact as exact
import xtx27_vtp15_gdn_projection as projection
import xtx27_vtp24_ffn_pack as ffn
import xtx27_vtp27_gpu_layer0_bridge as bridge

ROOT = projection.ROOT
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp28-original-layer1-gdn-gpu-input"
SCHEMA = "vtp28-qwen38-original-BF16-GPU-layer0-to-GDN-layer1-input-v1"
LAYER = 1


def validate_bridge(receipt: Path) -> dict:
    info = json.loads(receipt.read_text())
    cases = info.get("cases")
    if (info.get("schema") != bridge.SCHEMA
            or info.get("status") != "VTP27_ACTUAL_GPU_GDN_TO_FFN_LAYER0_HANDOFF_PASS_NOT_MODEL"
            or info.get("checkpoint_verified_complete_layers") != 1
            or not isinstance(cases, list) or len(cases) != 2
            or not all(row.get("rc") == 0 and row.get("oracle_pass") is True
                       for row in cases)
            or info.get("full_layer0_after_FFN", {}).get("tolerance_pass") is not True
            or info.get("original_model_sources", {}).get("rows") != 1):
        raise ValueError("VTP28 requires a completed original-BF16 GPU layer-0 bridge")
    path = Path(info["layer1_input_GPU_result_file"]).resolve(strict=True)
    raw = path.read_bytes()
    if (len(raw) != projection.HIDDEN * 2
            or hashlib.sha256(raw).hexdigest()
            != info["full_layer0_after_FFN"]["GPU_result_sha256"]):
        raise ValueError("VTP28 actual GPU-produced layer-0 hidden state changed")
    ancestor = info["original_model_sources"]
    ffn_record = ffn.validate(Path(ancestor["ffn"]["packet"]))
    if (info.get("source_digest") != bridge.micro.source_digest(bridge.SOURCES)
            or info.get("driver_sha256") != projection.sha(Path(bridge.__file__))
            or ancestor["ffn"]["packet_sha256"]
            != projection.sha(Path(ancestor["ffn"]["packet"]))):
        raise ValueError("VTP28 bridge source/FFN receipts changed")
    original = Path(ffn_record["checkpoint"]).resolve(strict=True)
    original_input = json.loads(Path(ancestor["gdn"]["weight_receipt"]).read_text())
    if original != Path(original_input["checkpoint"]).resolve(strict=True):
        raise ValueError("original GDN and FFN model source checkpoints differ")
    return {
        "parent_receipt": str(receipt.resolve(strict=True)),
        "parent_receipt_sha256": projection.sha(receipt),
        "hidden_file": str(path),
        "hidden_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint": str(original),
        "rows": 1, "layer0_reference_exact": False,
        "actual_GPU_layer0_output": True,
    }


def norm_gpu_hidden(hidden: bytes, norm: bytes) -> bytes:
    if len(hidden) != projection.HIDDEN * 2 or len(norm) != projection.HIDDEN * 2:
        raise ValueError("original layer1 input/norm BF16 extent mismatch")
    x = exact.bf16_values(hidden)
    gain = exact.bf16_values(norm)
    if not all(math.isfinite(v) for v in (*x, *gain)):
        raise ValueError("original layer1 hidden/norm contains nonfinite BF16")
    inv = 1.0 / math.sqrt(sum(v * v for v in x) / projection.HIDDEN + 1e-6)
    return b"".join(struct.pack(
        "<H", struct.unpack("<I", struct.pack(
            "<f", exact.round_bf16(v * inv * (1.0 + gain[i]))
        ))[0] >> 16,
    ) for i, v in enumerate(x))


def sources(checkpoint: Path, pack_receipt: Path, bridge_receipt: Path) -> tuple[dict, dict, dict]:
    parent = validate_bridge(bridge_receipt)
    desc, qkv = projection.sources(checkpoint, LAYER, pack_receipt)
    if (Path(desc["checkpoint"]) != Path(parent["checkpoint"])
            or desc["layer"] != LAYER or qkv["layer"] != LAYER):
        raise ValueError("VTP28 layer1 BF16 weights from a different checkpoint")
    for role in gdn.ROLES:
        for kind in ("in_proj_qkv", "conv1d"):
            item = qkv["outputs"][kind]["ranks"][role]
            path = Path(item["path"])
            if (path.stat().st_size != item["bytes"] or projection.sha(path) != item["sha256"]):
                raise ValueError("VTP28 original QKV/conv checkpoint shard changed")
    return desc, qkv, parent


def pack(checkpoint: Path, qkv_receipt: Path, parent_receipt: Path,
         destination: Path) -> dict:
    desc, qkv, parent = sources(checkpoint, qkv_receipt, parent_receipt)
    dest = destination.expanduser().resolve()
    if dest.exists() or dest == ROOT or ROOT in dest.parents:
        raise ValueError("VTP28 original layer1 output must be immutable outside Git")
    dest.mkdir(parents=True, exist_ok=False)
    record = {
        **desc,
        "schema": SCHEMA, "status": "INCOMPLETE",
        "created_utc": datetime.now(UTC).isoformat(),
        "writer_sha256": projection.sha(Path(__file__)),
        "upstream_GPU_layer0": parent,
        "rank_packs": {}, "source_sha256": {}, "inputs": {},
        "original_hidden_is_actual_GPU_layer0": True,
        "layer1_not_yet_executed": True,
        "full_model": False,
    }
    receipt = dest / "summary.json"
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    try:
        hidden = Path(parent["hidden_file"]).read_bytes()
        for name, origin in desc["source_descriptors"].items():
            raw = projection.raw_tensor(origin)
            record["source_sha256"][name] = hashlib.sha256(raw).hexdigest()
            if name == "norm":
                normalized = norm_gpu_hidden(hidden, raw)
                for title, data in (
                    ("hidden_input.bf16", hidden),
                    ("normed_input.bf16", normalized),
                ):
                    record["inputs"][title] = {
                        "file": str(dest / title),
                           "bytes": len(data), "shape": [1, projection.HIDDEN],
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                    (dest / title).write_bytes(data)
                continue
            # Real BF16 projection weights, output-row shards as in VTP15.
            rows = []
            for role in gdn.ROLES:
                begin, count = desc["ownership"][role][name]
                data = raw[begin * projection.HIDDEN * 2:
                           (begin + count) * projection.HIDDEN * 2]
                path = dest / f"{name}-{role}.bf16"
                path.write_bytes(data)
                record["rank_packs"].setdefault(role, {})[name] = {
                    "file": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data), "shape": [count, projection.HIDDEN],
                    "source_rows": [begin, begin + count],
                }
                rows.append((begin, data))
            if b"".join(data for _, data in sorted(rows)) != raw:
                raise ValueError("original BF16 layer1 projection lost source rows")
        for role in gdn.ROLES:
            item = qkv["outputs"]["in_proj_qkv"]["ranks"][role]
            record["rank_packs"][role]["qkv"] = {
                "file": item["path"], "sha256": item["sha256"],
                "bytes": item["bytes"],
                "shape": [sum(n for _, n in gdn.layout()[role]), projection.HIDDEN],
                "packed_segments": gdn.layout()[role],
            }
        record["status"] = "VTP28_ACTUAL_GPU_LAYER0_BF16_TO_ORIGINAL_GDN_LAYER1_PACK_NO_GPU"
    except Exception as exc:
        record.update({"status": "FAILED_INCOMPLETE", "error": str(exc)})
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        raise
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    return record



def validate(receipt: Path) -> dict:
    info = json.loads(receipt.read_text())
    if (info.get("schema") != SCHEMA
            or info.get("status") != "VTP28_ACTUAL_GPU_LAYER0_BF16_TO_ORIGINAL_GDN_LAYER1_PACK_NO_GPU"
            or info.get("writer_sha256") != projection.sha(Path(__file__))
            or info.get("layer") != LAYER
            or info.get("original_hidden_is_actual_GPU_layer0") is not True):
        raise ValueError("VTP28 original layer1 GPU hidden package not accepted")
    parent = validate_bridge(Path(info["upstream_GPU_layer0"]["parent_receipt"]))
    if (parent["parent_receipt_sha256"]
            != info["upstream_GPU_layer0"]["parent_receipt_sha256"]
            or parent["hidden_sha256"]
            != info["upstream_GPU_layer0"]["hidden_sha256"]
            or parent["checkpoint"] != info["checkpoint"]):
        raise ValueError("VTP28 actual GPU layer0 parent lineage changed")
    desc, qkv, _ = sources(
        Path(info["checkpoint"]), Path(info["qkv_pack_receipt"]),
        Path(info["upstream_GPU_layer0"]["parent_receipt"]),
    )
    if desc["source_descriptors"] != info["source_descriptors"]:
        raise ValueError("VTP28 original layer1 checkpoint descriptors changed")
    for name, entry in desc["source_descriptors"].items():
        actual = projection.sha(
            Path(entry["path"]), entry["offset"], entry["bytes"]
        )
        if actual != info["source_sha256"][name]:
            raise ValueError(f"VTP28 original checkpoint source changed: {name}")
    for role in gdn.ROLES:
        for name in ("a", "b", "z", "qkv"):
            item = info["rank_packs"][role][name]
            source = Path(item["file"])
            if (not source.is_file() or source.stat().st_size != item["bytes"]
                    or projection.sha(source) != item["sha256"]
                    or item["shape"][1] != projection.HIDDEN
                    or math.prod(item["shape"]) * 2 != item["bytes"]):
                raise ValueError(f"VTP28 original BF16 layer1 weight changed: {role}.{name}")
            if name == "qkv":
                expected = qkv["outputs"]["in_proj_qkv"]["ranks"][role]
                if item["sha256"] != expected["sha256"]:
                    raise ValueError("VTP28 layer1 original QKV source changed")
    for key in ("hidden_input.bf16", "normed_input.bf16"):
        item = info["inputs"][key]
        path = Path(item["file"])
        if (item["shape"] != [1, projection.HIDDEN]
                or path.stat().st_size != 10240
                or projection.sha(path) != item["sha256"]):
            raise ValueError("VTP28 layer1 incoming GPU activation changed")
    if info["inputs"]["hidden_input.bf16"]["sha256"] != parent["hidden_sha256"]:
        raise ValueError("VTP28 original GPU layer0 hidden not used at layer1 boundary")
    return info


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--qkv-pack-receipt", type=Path, required=True)
    p.add_argument("--bridge-receipt", type=Path, required=True)
    p.add_argument("--phase", choices=("plan", "pack"), default="plan")
    p.add_argument("--run", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    if a.phase == "pack" and not a.run:
        p.error("VTP28 reading original model weights requires explicit --run")
    try:
        desc, _, parent = sources(a.checkpoint, a.qkv_pack_receipt, a.bridge_receipt)
        if a.phase == "plan":
            print(json.dumps({
                "schema": SCHEMA, "status": "ORIGINAL_SOURCE_CHECKED_NO_GPU",
                "upstream_actual_GPU_layer0": parent,
                "layer1_original_sources": desc, "full_model": False,
            }, indent=2))
            return 0
        dest = a.output or SCRATCH / datetime.now(UTC).strftime(
            "layer1-m1-%Y%m%dT%H%M%S%fZ"
        )
        result = pack(a.checkpoint, a.qkv_pack_receipt, a.bridge_receipt, dest)
        print(json.dumps({
            "status": result["status"],
            "receipt": str(dest.resolve() / "summary.json"),
            "actual_GPU_layer0_hidden": parent["hidden_sha256"],
        }, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as e:
        p.error(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
