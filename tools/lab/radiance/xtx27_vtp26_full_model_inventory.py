"""VTP26: 64-layer Qwen3.8-27B physical-TP3 model execution inventory.

Header-only, card-free model-wide proof for the original BF16 source:
exact 48 GDN + 16 gated full-attention + 64 5120/4864/7424
SwiGLU FFN sharded tensors, exact per-layer replicated input/
post-attention Q/K/GDN norms, 48-layer resident 128x128 GDN
per-value-head state and three-row 4-tap conv history, 16-layer
per-physical KV cache bytes for requested active streams/context.
Embedding and language head are explicitly HOST-owned for the
first correctness path (not a promise of good throughput);
vision and speculative MTP are not part of text-only decode.
Missing tensors, changed source geometry or gaps/overlaps fail closed.

Important: numbers here are LOWER BOUNDS, not live VRAM admission.
They omit HIP scratch, graphs, weight conversion, quant scales,
allocator reserve/fragmentation, model execution buffers and
pinned host staging. Do not use them as a launch authorization.
No GPU accesses or model generation are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import xtx27_vtp13_ownership as own

SCHEMA = "vtp26-original-qwen38-27b-64layer-physical-tp3-lower-bound-v1"
HIDDEN = 5120
KVDIM = 256
GDN_DIM = 128
CONV_ROWS = 3
BF16 = 2
F32 = 4
FULL = 16
GDN = 48
LAYERS = 64
EDGE = {
    "model.language_model.embed_tokens.weight": [248320, HIDDEN],
    "lm_head.weight": [248320, HIDDEN],
    "model.language_model.norm.weight": [HIDDEN],
}


def digest(document: dict) -> str:
    return hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def tensor_bytes(shapes: dict[str, list[int]], name: str, required: list[int]) -> int:
    if shapes.get(name) != required:
        raise ValueError(f"original checkpoint model-wide tensor missing/changed: {name}")
    return math.prod(required) * BF16


def physical_ledger(config: dict, shapes: dict, meta: dict,
                    context: int, streams: int) -> dict:
    if (type(context) is not int or not 1 <= context <= 262144
            or type(streams) is not int or not 1 <= streams <= 32):
        raise ValueError("bounded positive streams/context required")
    cfg = own.expected_config(config)
    if context > cfg["max_position_embeddings"]:
        raise ValueError("requested context exceeds original native model limit")
    candidate = own.make_map(config, shapes)
    if (candidate["layers"] != LAYERS or candidate["full_attention_layers"] != FULL
            or candidate["gdn_layers"] != GDN or len(candidate["tensors"]) != 640
            or tuple(candidate["roles"]) != own.ROLES):
        raise ValueError("physical ownership does not match expected 64-layer original model")
    roles: dict[str, dict] = {
        name: {
            "sharded_weight_bytes": 0, "replicated_norm_bytes": 0,
            "gdn_recurrent_state_bytes": 0, "gdn_conv_history_bytes": 0,
            "kv_cache_bytes": 0, "layers": [],
        } for name in own.ROLES
    }
    total_source_sharded = 0
    for name, tensor in candidate["tensors"].items():
        axis = tensor["split_axis"]
        shape = tensor["shape"]
        if not 0 <= axis < len(shape) or any(x < 1 for x in shape):
            raise ValueError("invalid source BF16 tensor shape")
        source_bytes = tensor_bytes(shapes, name, shape)
        total_source_sharded += source_bytes
        partition_total = 0
        for role in own.ROLES:
            pieces = tensor["roles"][role]
            width = sum(piece["length"] for piece in pieces)
            if any(piece["offset"] < 0 or piece["length"] <= 0 for piece in pieces):
                raise ValueError("invalid original physical tensor partition")
            part = source_bytes // shape[axis] * width
            partition_total += part
            roles[role]["sharded_weight_bytes"] += part
        if partition_total != source_bytes:
            raise ValueError(f"three-rank tensor does not sum to full source: {name}")
    if total_source_sharded <= 0:
        raise ValueError("source BF16 model weights absent")
    layers = []
    discovered = set(candidate["tensors"])
    replicated = set()
    for index in range(LAYERS):
        prefix = f"model.language_model.layers.{index}."
        is_full = index % 4 == 3
        norms = {
            "input_layernorm": [HIDDEN],
            "post_attention_layernorm": [HIDDEN],
        }
        if is_full:
            norms.update({
                "self_attn.q_norm": [KVDIM],
                "self_attn.k_norm": [KVDIM],
            })
        else:
            norms["linear_attn.norm"] = [GDN_DIM]
        norm_bytes = 0
        for name, shape in norms.items():
            key = prefix + name + ".weight"
            norm_bytes += tensor_bytes(shapes, key, shape)
            replicated.add(key)
        row = {
            "layer": index, "type": "full_attention" if is_full else "gdn",
            "norm_replication_bytes_per_physical_rank": norm_bytes,
            "weight_shard_bytes": {},
            "persistent_state_bytes": {},
        }
        for role_idx, role in enumerate(own.ROLES):
            relevant = [
                (name, item) for name, item in candidate["tensors"].items()
                if name.startswith(prefix)
            ]
            shard_bytes = 0
            for name, item in relevant:
                tensor = item["shape"]
                axis = item["split_axis"]
                shard_bytes += (math.prod(tensor) * BF16 // tensor[axis]) * sum(
                    part["length"] for part in item["roles"][role]
                )
            role_info = roles[role]
            role_info["replicated_norm_bytes"] += norm_bytes
            if is_full:
                kv = (streams * context * own.FULL_KV_HEADS[role_idx]
                      * KVDIM * 2 * BF16)
                role_info["kv_cache_bytes"] += kv
                persistent = kv
            else:
                values = own.GDN_V_HEADS[role_idx]
                channels = (2 * own.GDN_K_HEADS[role_idx] + values) * GDN_DIM
                recurrent = streams * values * GDN_DIM * GDN_DIM * F32
                conv = streams * CONV_ROWS * channels * BF16
                role_info["gdn_recurrent_state_bytes"] += recurrent
                role_info["gdn_conv_history_bytes"] += conv
                persistent = recurrent + conv
            row["weight_shard_bytes"][role] = shard_bytes
            row["persistent_state_bytes"][role] = persistent
            role_info["layers"].append({
                "index": index, "type": row["type"], "weights": shard_bytes,
                "replicated_norms": norm_bytes, "persistent_state": persistent,
            })
        layers.append(row)
    host_text_edges = sum(tensor_bytes(shapes, n, shape) for n, shape in EDGE.items())
    excluded = set(shapes) - discovered - replicated - set(EDGE)
    # Only separate visual-encoder and speculative MTP sources may
    # be excluded from the first TEXT-ONLY, speculation-OFF model.
    if any(not (n.startswith(("model.visual.", "mtp.")))
           for n in excluded):
        missing = sorted(n for n in excluded if not (
            n.startswith(("model.visual.", "mtp."))
        ))
        raise ValueError(f"model-wide unowned original text tensor(s): {missing[:8]}")
    totals = {}
    for role in own.ROLES:
        x = roles[role]
        statics = x["sharded_weight_bytes"] + x["replicated_norm_bytes"]
        state = (x["gdn_recurrent_state_bytes"]
                 + x["gdn_conv_history_bytes"] + x["kv_cache_bytes"])
        totals[role] = {
            **{k: v for k, v in x.items() if k != "layers"},
            "model_weight_plus_replicated_norm_bytes": statics,
            "minimum_live_state_bytes": state,
            "minimum_GPU_bytes_before_scratch_and_reserve": statics + state,
        }
    return {
        "schema": SCHEMA, "status": "64L_SHAPE_VERIFIED_MINIMUM_ONLY_NOT_EXECUTABLE",
        "checkpoint_metadata": meta,
        "original_bf16_weight_provenance": "HEADERS_ONLY_NOT_FULL_PAYLOAD_HASHED",
        "ownership_digest": digest(candidate["tensors"]),
        "layer_schedule": [row["type"] for row in layers],
        "layers": layers, "count": LAYERS,
        "gdn_layers": GDN, "full_attention_layers": FULL,
        "sharded_source_tensors": len(candidate["tensors"]),
        "total_sharded_source_bf16_bytes": total_source_sharded,
        "excluded_text_tensors": [],
        "excluded_optional_tensor_count": len(excluded),
        "host_embedding_head_final_norm_bytes": host_text_edges,
        "host_edges_mode": "CPU_HOST_TEXT_ONLY_CORRECTNESS_BASELINE_NOT_PERFORMANCE",
        "streams": streams, "context_per_stream": context,
        "total_retained_tokens": streams * context,
        "per_gpu_minimum": totals,
        "gpu_admission_approved": False,
        "full_model_execution_supported": False,
        "unmeasured_nonzero_resources": [
            "live_available_VRAM", "allocator_reserve", "HIP_scratch",
            "graph_buffers", "collective_staging",
            "persistent_weight_conversion", "fragmentation_margin",
            "native_operator_dispatch", "host_pinned_staging",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--streams", type=int, default=4)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    if args.output and args.output.exists():
        p.error("refusing to overwrite model-wide original-BF16 report")
    config, shapes, meta = own.load_original_checkpoint(args.checkpoint)
    try:
        result = physical_ledger(config, shapes, meta, args.context, args.streams)
    except (ValueError, KeyError, TypeError) as err:
        p.error(str(err))
    if args.output:
        checkout = Path(__file__).resolve().parents[3]
        out = args.output.resolve()
        if out == checkout or checkout in out.parents:
            p.error("lower-bound output must be outside Git checkout")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(json.dumps({
            "schema": SCHEMA, "status": result["status"],
            "receipt": str(out), "minimum_per_GPU_bytes": {
                role: record["minimum_GPU_bytes_before_scratch_and_reserve"]
                for role, record in result["per_gpu_minimum"].items()
            }, "host_text_edge_bytes": result["host_embedding_head_final_norm_bytes"],
            "not_admitted_or_executable": True,
        }, indent=2))
    else:
        print(json.dumps({
            k: v for k, v in result.items() if k != "layers"
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
