"""Card-free, capacity-first dense-27B TP2 + full-layer-stage KV estimator.

Reads an explicit model config. Attention KV bytes are exact logical values for
the declared geometry; VRAM fit and stage weights are illustrations only.
No claim is made that the mixed-architecture stage graph is implemented.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

GIB = 1024 ** 3
SUPPORTED = frozenset({"full_attention", "linear_attention"})


def model_shape(config: dict) -> dict:
    """Require explicit layer types; do not infer future Qwen model geometry."""
    model = config.get("text_config", config)
    if not isinstance(model, dict):
        raise TypeError("text_config must be a mapping")
    layers = model.get("layer_types")
    if not isinstance(layers, list) or not layers or not set(layers) <= SUPPORTED:
        raise ValueError("exact full/linear attention layer_types are required")
    if model.get("num_hidden_layers") != len(layers):
        raise ValueError("num_hidden_layers must match layer_types")
    fields = ("num_key_value_heads", "head_dim", "max_position_embeddings")
    for field in fields:
        val = model.get(field)
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            raise ValueError(f"{field} must be a positive integer")
    return {
        "layers": layers,
        "kv_heads": model["num_key_value_heads"],
        "head_dim": model["head_dim"],
        "trained_context": model["max_position_embeddings"],
        "attention_indices": [i for i, kind in enumerate(layers) if kind == "full_attention"],
    }


def kv_bytes_per_token(*, full_layers: int, heads: int, head_dim: int,
                       dtype_bytes: int, tp: int) -> int:
    """K and V per token per stage rank. No assumed KV replication."""
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in
           (heads, head_dim, dtype_bytes, tp)) or full_layers < 0 or heads % tp:
        raise ValueError("positive integer geometry and head-shardable TP required")
    return 2 * full_layers * (heads // tp) * head_dim * dtype_bytes


def capacity_row(*, name: str, stage_layers: int, full_layers: int, tokens: int,
                 tp: int, per_gpu_vram_gib: float, estimated_weights_gib: float,
                 reserve_gib: float, shape: dict, dtype_bytes: int) -> dict:
    bytes_per_token = kv_bytes_per_token(
        full_layers=full_layers, heads=shape["kv_heads"],
        head_dim=shape["head_dim"], dtype_bytes=dtype_bytes, tp=tp,
    )
    logical_kv_gib = tokens * bytes_per_token / GIB
    budget_gib = per_gpu_vram_gib - estimated_weights_gib - reserve_gib
    max_tokens = (
        max(0, int(budget_gib * GIB // bytes_per_token))
        if bytes_per_token else None
    )
    return {
        "stage": name, "layers": stage_layers, "full_attention_layers": full_layers,
        "tensor_parallel": tp, "card_vram_gib": per_gpu_vram_gib,
        "logical_fp8_attention_kv_gib_per_card": round(logical_kv_gib, 3)
        if dtype_bytes == 1 else None,
        "logical_attention_kv_gib_per_card": round(logical_kv_gib, 3),
        "modelled_weights_gib_per_card": round(estimated_weights_gib, 3),
        "reserved_gib_per_card": reserve_gib,
        "estimated_free_after_kv_gib_per_card": round(budget_gib - logical_kv_gib, 3),
        "illustrative_max_context_from_attention_kv_tokens": max_tokens,
        "not_a_runtime_fit_result": True,
    }


def scenario(shape: dict, *, cut: int, tokens: int, dtype_bytes: int = 1,
             xtx_full_rank_weights_gib: float = 7.51,
             reserve_gib: float = 2.0) -> dict:
    """Cut after layer cut-1; TP2 XTX prefix, TP1 R9700 suffix."""
    layers = shape["layers"]
    if isinstance(cut, bool) or not isinstance(cut, int) or not 0 < cut <= len(layers):
        raise ValueError("cut must be 1..num_hidden_layers")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
        raise ValueError("tokens must be a positive integer")
    if dtype_bytes not in (1, 2) or not all(
        math.isfinite(x) and x >= 0 for x in (xtx_full_rank_weights_gib, reserve_gib)
    ):
        raise ValueError("invalid KV dtype bytes, weights or reserve")
    total = len(layers)
    pref = sum(k == "full_attention" for k in layers[:cut])
    suffix = sum(k == "full_attention" for k in layers[cut:])
    xtx = capacity_row(
        name="XTX TP2 prefix", stage_layers=cut, full_layers=pref, tp=2,
        per_gpu_vram_gib=24.0,
        estimated_weights_gib=xtx_full_rank_weights_gib * cut / total,
        reserve_gib=reserve_gib, shape=shape, tokens=tokens, dtype_bytes=dtype_bytes,
    )
    stages = [xtx]
    if suffix or cut != total:
        r9700 = capacity_row(
            name="R9700 TP1 suffix", stage_layers=total-cut, full_layers=suffix, tp=1,
            per_gpu_vram_gib=32.0,
            estimated_weights_gib=2 * xtx_full_rank_weights_gib * (total-cut) / total,
            reserve_gib=reserve_gib, shape=shape, tokens=tokens, dtype_bytes=dtype_bytes,
        )
        stages.append(r9700)
    capacity = [
        s["illustrative_max_context_from_attention_kv_tokens"] for s in stages
        if s["illustrative_max_context_from_attention_kv_tokens"] is not None
    ]
    return {
        "cut_after_layer": cut - 1,
        "token_context": tokens,
        "exceeds_model_native_context": tokens > shape["trained_context"],
        "stages": stages,
        "all_card_logical_kv_gib": round(
            sum(s["logical_attention_kv_gib_per_card"] * s["tensor_parallel"]
                for s in stages), 3),
        "illustrative_system_context_limit_tokens": min(capacity) if capacity else None,
        "illustrative_memory_margin_nonnegative": all(
            s["estimated_free_after_kv_gib_per_card"] >= 0 for s in stages
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--context", type=int, default=1048576)
    parser.add_argument("--cut", type=int, action="append", default=[])
    parser.add_argument("--kv-bytes", type=int, choices=(1, 2), default=1)
    parser.add_argument("--reference-xtx-rank-weights-gib", type=float, default=7.51)
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    parser.add_argument("--output", type=Path, help="optional scratch JSON path")
    args = parser.parse_args(argv)
    try:
        shape = model_shape(json.loads(args.config.read_text(encoding="utf-8")))
        cuts = args.cut or [32, 40, 48, len(shape["layers"])]
        if len(set(cuts)) != len(cuts):
            raise ValueError("duplicate cut")
        scenarios = [
            scenario(shape, cut=cut, tokens=args.context, dtype_bytes=args.kv_bytes,
                     xtx_full_rank_weights_gib=args.reference_xtx_rank_weights_gib,
                     reserve_gib=args.reserve_gib)
            for cut in cuts
        ]
    except (ValueError, OSError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    output = {
        "schema": 1, "config_path": str(args.config.resolve()),
        "model": {"layers": len(shape["layers"]),
                  "full_attention_layers": len(shape["attention_indices"]),
                  "full_attention_indices": shape["attention_indices"],
                  "trained_context": shape["trained_context"],
                  "kv_heads": shape["kv_heads"], "head_dim": shape["head_dim"]},
        "dtype_bytes_per_kv_element": args.kv_bytes,
        "measured_reference_full_tp2_weight_gib_per_xtx": args.reference_xtx_rank_weights_gib,
        "hypothetical_reserve_gib_per_card": args.reserve_gib,
        "scenarios": scenarios,
        "limitations": [
            "Only classical full-attention K+V scales with context in this calculation.",
            ("GDN state, GDN checkpoint slots, graph/runtime buffers, embedding, output head, "
             "allocator padding, vision encoder, multi-session state and host KV tier are excluded."),
            ("Stage weight estimates scale one existing full TP2 rank's resident weights by "
             "layer fraction; they are NOT actual per-stage memory plans."),
            "The R9700 full-layer stage and activation handoff are NOT implemented in Radiance.",
            "Do not interpret a positive memory margin as evidence of runtime fit or model quality.",
            "Contexts above trained_context are speculative unless supported by the future model.",
        ],
    }
    rendered = json.dumps(output, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
