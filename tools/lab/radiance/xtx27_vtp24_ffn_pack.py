"""VTP24: package original BF16 FFN shards and post-attention norm for TP3.

Unlike VTP13's synthetic/quantized FFN microbench, this reads real
Qwen3.8-27B BF16 gate/up/down projection payloads and the original
post_attention_layernorm.weight. Row ownership is 5120/4864/7424,
and down_proj is split along its INPUT columns, byte-reconstructing
the complete 17408-wide source.

For layer 0 only, consume a previously validated original-weight
GDN CPU output/residual as the real FFN *input*, then apply layer-0
post_attention_layernorm with Qwen3.5 +1 weighted RMSNorm.
This is not yet an actual GPU-to-GPU/residual-layer handoff.
The output is a CPU-provenance input for the next 3-GPU FFN gate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
from datetime import UTC, datetime
from pathlib import Path

import xtx27_checkpoint_audit as audit
import xtx27_vtp13_ownership as ownership
import xtx27_vtp14_gdn_exact as exact
import xtx27_vtp15_gdn_projection as projection
import xtx27_vtp16_gdn_output_oracle as gdn_oracle
import xtx27_vtp22_attention_pack as attn_pack

ROOT = projection.ROOT
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp24-original-ffn"
SCHEMA = "vtp24-qwen38-original-bf16-ffn-three-physical-rank-package-v1"
WIDTHS = ownership.FFN_WIDTHS
ROLES = ownership.ROLES
HIDDEN = 5120
INTERMEDIATE = 17408
SPECS = {
    "gate": ("mlp.gate_proj.weight", [INTERMEDIATE, HIDDEN]),
    "up": ("mlp.up_proj.weight", [INTERMEDIATE, HIDDEN]),
    "down": ("mlp.down_proj.weight", [HIDDEN, INTERMEDIATE]),
    "post_norm": ("post_attention_layernorm.weight", [HIDDEN]),
}


def source_info(checkpoint: Path, layer: int) -> dict:
    if type(layer) is not int or not 0 <= layer <= 63:
        raise ValueError("FFN layer must be in 0..63")
    root = checkpoint.expanduser().resolve(strict=True)
    config, config_sha = audit.load_metadata(root / "config.json")
    ownership.expected_config(config)
    index, index_sha = audit.load_metadata(root / "model.safetensors.index.json")
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise TypeError("checkpoint index missing weight_map")
    cache = {}
    sources = {
        key: projection._source(root, index["weight_map"], layer, key, spec, cache)
        for key, spec in SPECS.items()
    }
    return {
        "layer": layer, "config_sha256": config_sha, "index_sha256": index_sha,
        "checkpoint": str(root), "sources": sources,
        "ffn_widths": list(WIDTHS),
    }


def validate_gdn_reference(path: Path) -> dict:
    ref = json.loads(path.read_text())
    if (ref.get("schema") != gdn_oracle.SCHEMA
            or ref.get("status") != "ORIGINAL_BF16_GDN_OUTPUT_PARTIAL_F32_REDUCTION_RESIDUAL_CPU_PASS"
            or ref.get("rows") not in (1, 4)
            or ref.get("layer") != 0
            or ref.get("writer_sha256") != projection.sha(Path(gdn_oracle.__file__))):
        raise ValueError("upstream connected original BF16 GDN reference is invalid")
    # Validate original checkpoint and source provenance, not just the
    # local residual file. This transitively checks real GDN input/out
    # weights, original coefficient/data and the per-rank core references.
    packet = gdn_oracle.validate(Path(ref["original_packet"]))
    if (projection.sha(Path(ref["original_packet"])) != ref["original_packet_sha256"]
            or packet["rows"] != ref["rows"]):
        raise ValueError("GDN parent receipt changed or wrong row count")
    item = ref["outputs"]["ordered-three-rank-residual.bf16"]
    hidden = Path(item["file"])
    if (item["shape"] != [ref["rows"], HIDDEN]
            or item["bytes"] != ref["rows"] * HIDDEN * 2
            or hidden.stat().st_size != item["bytes"]
            or projection.sha(hidden) != item["sha256"]):
        raise ValueError("actual GDN post-residual input changed")
    return {
        "receipt": str(path.resolve(strict=True)),
        "receipt_sha256": projection.sha(path),
        "gdn_packet": ref["original_packet"],
        "rows": ref["rows"],
        "hidden_file": str(hidden),
        "hidden_sha256": item["sha256"],
        "upstream_is_cpu_reference": True,
    }


def weighted_norm(hidden: bytes, norm: bytes, rows: int) -> bytes:
    if len(hidden) != rows * HIDDEN * 2 or len(norm) != HIDDEN * 2:
        raise ValueError("original FFN norm shape mismatch")
    norms = exact.bf16_values(norm)
    source = exact.bf16_values(hidden)
    output = bytearray()
    for row in range(rows):
        x = source[row * HIDDEN:(row + 1) * HIDDEN]
        ss = sum(float(t) * t for t in x)
        inv = 1.0 / math.sqrt(ss / HIDDEN + 1e-6)
        for d in range(HIDDEN):
            out = exact.round_bf16(x[d] * inv * (1.0 + norms[d]))
            output.extend(struct.pack("<f", out)[2:4])
    return bytes(output)


def pack(checkpoint: Path, layer: int, upstream: Path,
         destination: Path) -> dict:
    src = source_info(checkpoint, layer)
    if layer != 0:
        raise ValueError("until cross-layer forwarding exists, only layer 0 FFN is executable")
    parent = validate_gdn_reference(upstream)
    dest = destination.expanduser().resolve()
    if dest.exists() or dest == ROOT or ROOT in dest.parents:
        raise ValueError("original FFN pack needs a new directory outside Git")
    dest.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": SCHEMA, "status": "INCOMPLETE",
        "created_utc": datetime.now(UTC).isoformat(),
        "writer_sha256": projection.sha(Path(__file__)),
        **src, "input": parent,
        "original_tensor_sha256": {}, "rank_weights": {r: {} for r in ROLES},
        "norm": {}, "bf16_input": {},
        "model_inference": False, "gdn_output_is_cpu_reference": True,
        "notes": "Pre/post residual is computed from original BF16 GDN CPU oracle. "
                 "Not a GPU GDN->FFN handoff or model text generation.",
    }
    file = dest / "summary.json"
    file.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        originals = {}
        for name, descriptor in src["sources"].items():
            raw = projection.raw_tensor(descriptor)
            originals[name] = raw
            receipt["original_tensor_sha256"][name] = projection.sha(
                Path(descriptor["path"]), descriptor["offset"], descriptor["bytes"]
            )
        gates = attn_pack.rows_shards(originals["gate"], HIDDEN, WIDTHS)
        ups = attn_pack.rows_shards(originals["up"], HIDDEN, WIDTHS)
        downs = attn_pack.col_shards(originals["down"], HIDDEN, INTERMEDIATE, WIDTHS)
        for role, count, chunks in zip(
            ROLES, WIDTHS, zip(gates, ups, downs, strict=True), strict=True
        ):
            for name, chunk in zip(("gate", "up", "down"), chunks, strict=True):
                shape = [HIDDEN, count] if name == "down" else [count, HIDDEN]
                receipt["rank_weights"][role][name] = attn_pack._output_record(
                    dest / f"{name}-{role}.bf16", chunk, shape
                )
        receipt["norm"] = attn_pack._output_record(
            dest / "post_attention_layernorm.bf16",
            originals["post_norm"], [HIDDEN],
        )
        hidden = Path(parent["hidden_file"]).read_bytes()
        normed = weighted_norm(hidden, originals["post_norm"], parent["rows"])
        for name, data in (("input", hidden), ("normed", normed)):
            receipt["bf16_input"][name] = attn_pack._output_record(
                dest / f"{name}.bf16", data, [parent["rows"], HIDDEN]
            )
        receipt["rows"] = parent["rows"]
        receipt["status"] = "ORIGINAL_BF16_FFN_LAYER0_PACK_MATCHED_GDN_RESIDUAL_CPU_NOT_GPU"
    except Exception as exc:
        receipt.update({"status": "FAILED_INCOMPLETE", "error": str(exc)})
        file.write_text(json.dumps(receipt, indent=2) + "\n")
        raise
    file.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def validate(receipt: Path) -> dict:
    info = json.loads(receipt.read_text())
    if (info.get("schema") != SCHEMA
            or info.get("status") != "ORIGINAL_BF16_FFN_LAYER0_PACK_MATCHED_GDN_RESIDUAL_CPU_NOT_GPU"
            or info.get("writer_sha256") != projection.sha(Path(__file__))
            or info.get("layer") != 0 or info.get("ffn_widths") != list(WIDTHS)
            or type(info.get("rows")) is not int or info["rows"] not in (1, 4)):
        raise ValueError("VTP24 original FFN source/provenance mismatch")
    ancestor = validate_gdn_reference(Path(info["input"]["receipt"]))
    if (ancestor["receipt_sha256"] != info["input"]["receipt_sha256"]
            or ancestor["hidden_sha256"] != info["input"]["hidden_sha256"]):
        raise ValueError("VTP24 input ancestry no longer valid")
    for role, width in zip(ROLES, WIDTHS, strict=True):
        for name in ("gate", "up", "down"):
            entry = info["rank_weights"][role][name]
            expected = [HIDDEN, width] if name == "down" else [width, HIDDEN]
            target = Path(entry["file"])
            if (entry["shape"] != expected or target.stat().st_size != width * HIDDEN * 2
                    or projection.sha(target) != entry["sha256"]):
                raise ValueError(f"VTP24 original BF16 FFN {role}.{name} corrupted")
    for entry in (info["norm"], *info["bf16_input"].values()):
        target = Path(entry["file"])
        if target.stat().st_size != entry["bytes"] or projection.sha(target) != entry["sha256"]:
            raise ValueError("VTP24 FFN normalized input or norm corrupted")
    return info


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--gdn-reference", type=Path)
    p.add_argument("--phase", choices=("plan", "pack"), default="plan")
    p.add_argument("--run", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    if a.phase == "pack" and not a.run:
        p.error("original checkpoint FFN pack requires explicit --run")
    if a.phase == "pack" and a.gdn_reference is None:
        p.error("layer-0 FFN needs a source-validated GDN output receipt")
    try:
        plan = source_info(a.checkpoint, a.layer)
        if a.phase == "plan":
            print(json.dumps({
                "schema": SCHEMA, "status": "CARD_FREE_FFN_ORIGINAL_SOURCE_PLAN",
                "source": plan, "layer0_gdn_reference": (
                    validate_gdn_reference(a.gdn_reference)
                    if a.gdn_reference else None
                ), "gpu_required": False, "full_model": False,
            }, indent=2))
            return 0
        target = a.output or SCRATCH / datetime.now(UTC).strftime(
            f"layer{a.layer}-%Y%m%dT%H%M%S%fZ"
        )
        result = pack(a.checkpoint, a.layer, a.gdn_reference, target)
        print(json.dumps({
            "status": result["status"],
            "receipt": str(target.resolve() / "summary.json"),
            "layer": a.layer, "rows": result["rows"],
        }, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as e:
        p.error(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
