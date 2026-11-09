"""Card-free heterogeneous weight-variant *planner*, not a Radiance runtime.

From measured, comparable per-layer variant costs, select at model admission
the lowest-cost format per layer that fits physical VRAM AFTER resident KV and
runtime reserve. Never convert weights, initialize GPUs, or migrate live state.
A selection remains UNQUALIFIED until end-to-end quality, placement, and serving
have passed. Synthetic fixtures are for tests, not performance claims.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if __package__:
    from . import xtx27_quant_plan as qp
else:
    import xtx27_quant_plan as qp

FORMAT_NAMES = frozenset(("w4", "mxfp4", "fp8"))
STAGES = ("xtx", "r9700")
MAX_DP_STATES = 100_000
MIB = 1024 * 1024


def require_float(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")  # noqa: TRY004
    n = float(value)
    if not math.isfinite(n) or n < 0 or (positive and n == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return n


def kv_residency_mib(config: dict, *, cut: int, tokens: int, tp: int = 2) -> dict[str, float]:
    layers, context = qp.shape_from_config(config)
    if not 1 <= cut < layers or tokens <= 0 or tokens > context:
        raise ValueError("invalid cut/context or context exceeds checkpoint native limit")
    cfg = config["text_config"]
    heads, dim = cfg.get("num_key_value_heads"), cfg.get("head_dim")
    if (isinstance(heads, bool) or not isinstance(heads, int) or heads <= 0 or heads % tp
            or isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0):
        raise ValueError("invalid KV dimensions or nonshardable KV heads")
    types = cfg["layer_types"]
    prefix = types[:cut].count("full_attention")
    suffix = types[cut:].count("full_attention")
    return {
        "xtx": 2 * prefix * (heads // tp) * dim * tokens / MIB,
        "r9700": 2 * suffix * heads * dim * tokens / MIB,
    }


def choose_stage(layers: list[int], records: list[dict], stage: str,
                 max_weights_mib: float, objective: str) -> dict:
    """Exact memory-bounded DP on conservative ceil(MiB) per weight candidate."""
    budget = math.floor(require_float(max_weights_mib, f"{stage} weight budget"))
    # state maps sum ceiling MiB -> (objective time, chosen format names)
    states: dict[int, tuple[float, tuple[str, ...]]] = {0: (0.0, ())}
    detail: dict[tuple[int, str], dict] = {}
    for idx in layers:
        opts = records[idx][stage]
        next_states: dict[int, tuple[float, tuple[str, ...]]] = {}
        for opt in opts:
            name = opt["format"]
            detail[(idx, name)] = opt
            cost = require_float(opt[f"{objective}_us"], "objective time", positive=True)
            memory = math.ceil(require_float(opt["weights_mib_per_card"],
                                              "weight memory", positive=True))
            for used, (score, formats) in states.items():
                total = used + memory
                if total > budget:
                    continue
                candidate = (score + cost, formats + (name,))
                previous = next_states.get(total)
                if previous is None or candidate < previous:
                    next_states[total] = candidate
        if not next_states:
            raise ValueError(f"{stage}: no valid weight configuration at layer {idx}")
        if len(next_states) > MAX_DP_STATES:
            raise ValueError(f"{stage}: DP state budget exceeded")
        # Preserve only strictly faster states as memory increases.
        best = math.inf
        states = {}
        for mem, val in sorted(next_states.items()):
            if val[0] < best:
                states[mem] = val
                best = val[0]
    mem, (time, formats) = min(states.items(), key=lambda pair: (pair[1][0], pair[0], pair[1][1]))
    chosen = [
        {"layer": idx, "format": fmt, "measured_weight_mib_per_card":
         detail[(idx, fmt)]["weights_mib_per_card"],
         f"measured_{objective}_us": detail[(idx, fmt)][f"{objective}_us"]}
        for idx, fmt in zip(layers, formats, strict=True)
    ]
    return {
        "stage": stage, "chosen": chosen,
        "conservative_weights_mib": mem,
        "measured_objective_us_sum": round(time, 3),
        "weight_budget_mib": budget,
        "remaining_mib_after_rounded_weights": round(max_weights_mib - mem, 3),
    }


def select(config: dict, metrics: dict, *, cut: int, tokens: int,
           objective: str = "prefill") -> dict:
    layers, context = qp.shape_from_config(config)
    if objective not in ("prefill", "decode"):
        raise ValueError("objective must be prefill or decode")
    if metrics.get("data_kind") not in ("synthetic", "measured"):
        raise ValueError("explicit synthetic/measured data_kind required")
    if metrics["data_kind"] == "measured" and not metrics.get("evidence_id"):
        raise ValueError("measured metrics require an evidence ID")
    digest = qp.sha256(json.dumps(config, sort_keys=True).encode())
    if metrics.get("source_config_sha256") != digest:
        raise ValueError("metric source config does not match checkpoint")
    if metrics.get("native_context") != context:
        raise ValueError("metric native context does not match checkpoint")
    kl = kv_residency_mib(config, cut=cut, tokens=tokens)
    budgets = metrics.get("budgets")
    records = metrics.get("layers")
    if not isinstance(budgets, dict) or not isinstance(records, list) or len(records) != layers:
        raise ValueError("explicit per-stage budgets and all layer records required")
    for stage in STAGES:
        if stage not in budgets or not isinstance(budgets[stage], dict):
            raise ValueError(f"missing {stage} budget")
        for field in ("vram_mib", "runtime_reserve_mib"):
            require_float(budgets[stage].get(field), f"{stage} {field}", positive=field == "vram_mib")
    for idx, record in enumerate(records):
        if not isinstance(record, dict) or record.get("layer") != idx:
            raise ValueError("layer metrics must be ordered and complete")
        for stage in STAGES:
            choices = record.get(stage)
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"layer {idx}: missing {stage} choices")
            found: set[str] = set()
            for choice in choices:
                if not isinstance(choice, dict) or choice.get("format") not in FORMAT_NAMES:
                    raise ValueError(f"layer {idx}: invalid {stage} format")
                name = choice["format"]
                if name in found:
                    raise ValueError(f"layer {idx}: repeated {stage} format")
                found.add(name)
                for field in ("prefill_us", "decode_us", "weights_mib_per_card"):
                    require_float(choice.get(field), f"{stage}/{idx}/{name}/{field}", positive=True)
    selected = []
    for stage, indices in (("xtx", list(range(cut))), ("r9700", list(range(cut, layers)))):
        vram = float(budgets[stage]["vram_mib"])
        reserve = float(budgets[stage]["runtime_reserve_mib"])
        kv = kl[stage]
        available = vram - reserve - kv
        if available <= 0:
            raise ValueError(f"{stage}: no weight space after KV and runtime reserve")
        result = choose_stage(indices, records, stage, available, objective)
        result.update({
            "vram_mib": vram,
            "runtime_reserve_mib": reserve,
            "logical_fp8_kv_mib_per_card": round(kv, 3),
            "kv_dtype": "fp8",
            "tp": 2 if stage == "xtx" else 1,
            "gpu": "gfx1100" if stage == "xtx" else "gfx1201",
        })
        selected.append(result)
    return {
        "schema": "xtx27.quant.selection.v1",
        "status": "PREVIEW_ONLY_NOT_END_TO_END_QUALIFIED",
        "data_kind": metrics["data_kind"],
        "evidence_id": metrics.get("evidence_id"),
        "source_config_canonical_sha256": digest,
        "objective": objective,
        "cut": cut,
        "context_tokens": tokens,
        "native_context_tokens": context,
        "stages": selected,
        "limitations": [
            "weights/compute estimates come from supplied comparable per-layer candidate metrics",
            "per-layer error does not imply end-to-end model quality; KL is still required",
            "runtime reserve must include all measured activation, graph, GDN and overhead buffers",
            "the R9700 full-layer stage and per-GPU variant selection are NOT implemented",
            "shared TP2 ranks use the same chosen quantization per layer",
            "one stored variant per loaded layer; no runtime weight conversion or migration",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--cut", type=int, default=40)
    parser.add_argument("--tokens", type=int, default=131072)
    parser.add_argument("--objective", choices=("prefill", "decode"), default="prefill")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text())
        metrics = json.loads(args.metrics.read_text())
        result = select(config, metrics, cut=args.cut, tokens=args.tokens,
                        objective=args.objective)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
