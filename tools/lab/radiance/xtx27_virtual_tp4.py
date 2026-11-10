"""VTP02 card-free virtual TP4 envelope; projections only, never model admission.

Four logical ranks are placed on three physical cards: [XTX0], [XTX1],
[R9700 rank2+rank3]. The projected weight split and the (1,1,2) physical
KV-head split are deliberately different. No GPU modules are imported.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation

GIB = 1 << 30
ATTENTION_LAYERS = 16
KV_HEADS = 4
HEAD_DIM = 256
NATIVE_CONTEXT_PER_REQUEST = 262144
VRAM_GIB = (Decimal(24), Decimal(24), Decimal(32))
BANDWIDTH_GBS = (Decimal(960), Decimal(960), Decimal(640))
CANDIDATES = (
    ("capacity", ("0.30", "0.30", "0.20", "0.20")),
    ("intermediate", ("0.35", "0.35", "0.15", "0.15")),
    ("bandwidth", ("0.375", "0.375", "0.125", "0.125")),
)
KV_BPT = {
    "fp8": Decimal(2) * ATTENTION_LAYERS * KV_HEADS * HEAD_DIM,
    "fp8_v4": Decimal(2) * ATTENTION_LAYERS * KV_HEADS * HEAD_DIM * Decimal(25) / Decimal(32),
    "k6_v4": Decimal(2) * ATTENTION_LAYERS * KV_HEADS * HEAD_DIM * Decimal(22) / Decimal(32),
}
PHYSICAL_KV_HEADS = (1, 1, 2)


def parse_ratios(raw: Sequence[str]) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    if len(raw) != 4:
        raise ValueError("exactly four logical weights required")
    try:
        values = tuple(Decimal(x) for x in raw)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("invalid logical shard fraction") from exc
    if not all(x.is_finite() and x > 0 for x in values):
        raise ValueError("fractions must be finite and positive")
    if sum(values) != Decimal(1):
        raise ValueError("fractions must add to exactly one")
    if values[2] != values[3]:
        raise ValueError("R9700's two logical shards must be equal in this VTP02 candidate")
    return values  # type: ignore[return-value]


def tile_counts(fractions: Sequence[Decimal], width: int = 17408, tile: int = 128) -> list[int]:
    """Hamilton rounding for a single shardable FFN dimension, not head geometry."""
    if width <= 0 or tile <= 0 or width % tile:
        raise ValueError("the FFN width must be a positive multiple of the tile")
    total = width // tile
    ideal = [total * f for f in fractions]
    result = [int(v) for v in ideal]
    left = total - sum(result)
    for i in sorted(range(len(result)), key=lambda i: (-(ideal[i] - result[i]), i))[:left]:
        result[i] += 1
    if sum(result) != total or any(x <= 0 for x in result):
        raise ValueError("cannot produce non-empty legal tiles")
    return result


def calculate(
    fractions: Sequence[Decimal], *, tokens: int, kv: str,
    model_shardable_gib: Decimal, overhead_gib: Sequence[Decimal],
    num_streams: int,
) -> dict:
    if tokens < 1 or num_streams < 1:
        raise ValueError("tokens/streams must be positive")
    if kv not in KV_BPT:
        raise ValueError("unknown KV format")
    if len(overhead_gib) != 3 or any(not v.is_finite() or v < 0 for v in overhead_gib):
        raise ValueError("three non-negative physical overhead amounts required")
    if not model_shardable_gib.is_finite() or model_shardable_gib <= 0:
        raise ValueError("shardable weight total must be positive")
    if sum(fractions) != Decimal(1):
        raise ValueError("ratios must sum to one")
    physical_shards = (fractions[0], fractions[1], fractions[2] + fractions[3])
    logical_kv_total_gib = KV_BPT[kv] * tokens / GIB
    per_card = []
    for i in range(3):
        weights = model_shardable_gib * physical_shards[i]
        attn_kv = logical_kv_total_gib * PHYSICAL_KV_HEADS[i] / KV_HEADS
        projected = weights + attn_kv + overhead_gib[i]
        per_card.append({
            "gpu": ("xtx0", "xtx1", "r9700")[i],
            "weight_fraction": str(physical_shards[i]),
            "projected_shardable_weights_gib": float(weights),
            "logical_attention_kv_gib": float(attn_kv),
            "assumed_nonshardable_plus_runtime_gib": float(overhead_gib[i]),
            "illustrative_total_gib": float(projected),
            "illustrative_remaining_gib": float(VRAM_GIB[i] - projected),
            "illustrative_within_vram": projected <= VRAM_GIB[i],
        })
    # Decoding bandwidth-only time is proportional to shard bytes / HBM BW.
    # This is NOT a prediction of end-to-end latency or matrix throughput.
    times = [physical_shards[i] / BANDWIDTH_GBS[i] for i in range(3)]
    tiles = tile_counts(fractions)
    return {
        "label": "CARD_FREE_PROJECTION_NOT_RUNTIME_ADMISSION",
        "model": "Qwen3.8-27B geometry proxy; not future Qwen4 evidence",
        "tokens_aggregate_across_streams": tokens,
        "num_streams": num_streams,
        "single_request_over_native_262144": tokens > NATIVE_CONTEXT_PER_REQUEST,
        "native_possible_even_split_per_stream": tokens % num_streams == 0 and tokens // num_streams <= NATIVE_CONTEXT_PER_REQUEST,
        "kv_format": kv,
        "kv_format_hardware_qualified": False,
        "logical_ratios": [str(v) for v in fractions],
        "physical_kv_heads": list(PHYSICAL_KV_HEADS),
        "ffn_128_tile_counts": tiles,
        "ffn_tile_rounded_fractions": [str(Decimal(x) / sum(tiles)) for x in tiles],
        "attention_and_gdn_may_require_separate_shard_mapping": True,
        "cards": per_card,
        "bandwidth_only_relative_slowest_to_xtx0": float(max(times) / times[0]),
        "excludes": ["true tensor census", "allocator/page rounding", "GDN/checkpoints not in supplied overhead",
                     "synchronisation", "collectives/PCIe", "kernel/quality validation"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true",
                        help="use explicit synthetic 30 GiB weights and 2/2/3 GiB overhead")
    parser.add_argument("--model-shardable-weight-gib", type=str, default=None)
    parser.add_argument("--overhead-gib", nargs=3, metavar=("XTX0", "XTX1", "R9700"))
    parser.add_argument("--tokens", type=int, default=1048576,
                        help="aggregate logical resident tokens across ALL streams (default 1048576)")
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--kv-format", choices=tuple(KV_BPT), default="fp8")
    parser.add_argument("--candidate", choices=[n for n, _ in CANDIDATES] + ["all"],
                        default="all")
    args = parser.parse_args(argv)
    if args.demo:
        if args.model_shardable_weight_gib is not None or args.overhead_gib is not None:
            parser.error("--demo cannot be combined with weight or overhead inputs")
        weights = Decimal(30)
        overhead = (Decimal(2), Decimal(2), Decimal(3))
    elif args.model_shardable_weight_gib is None or args.overhead_gib is None:
        parser.error("provide both --model-shardable-weight-gib and --overhead-gib, or --demo")
    else:
        try:
            weights = Decimal(args.model_shardable_weight_gib)
            overhead = tuple(Decimal(x) for x in args.overhead_gib)
        except (InvalidOperation, ValueError) as exc:
            parser.error(f"invalid decimal input: {exc}")
    results = []
    for name, raw in CANDIDATES:
        if args.candidate in ("all", name):
            try:
                entry = calculate(
                    parse_ratios(raw), tokens=args.tokens, kv=args.kv_format,
                    model_shardable_gib=weights, overhead_gib=overhead, num_streams=args.streams,
                )
            except ValueError as exc:
                parser.error(str(exc))
            results.append({"candidate": name, **entry})
    print(json.dumps({"type": "VTP02_CARD_FREE_EXPERIMENT",
                      "input_class": "synthetic" if args.demo else "user_assumed_not_measured",
                      "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
