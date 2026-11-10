"""VTP08: card-free W8A8 FFN + FP8-KV physical VRAM impossibility bounds.

Reads only native RAD2 metadata via rad-info -v. Confirms all 128
64-layer fused FFN tensors use i8*bf16[1x128]/fwht128, then derives
EXACT code+scale bytes for shardable FFN tensors from 27B geometry.
Assumes ONLY four KV heads/16 full-attention layers/FP8 K and V,
distributed 1/1/2 across XTX0/XTX1/R9700. All other weights, GDN,
runtime, graph, allocator, vision, draft, page alignment and replicated
metadata are excluded. Thus any VRAM deficit is a definitive NO-GO
under these assumptions; positive headroom is NEVER admission.
No GPU, checkpoint reads, model conversion or allocation.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from xtx27_virtual_tp4 import (
    ATTENTION_LAYERS,
    CANDIDATES,
    GIB,
    HEAD_DIM,
    KV_HEADS,
    NATIVE_CONTEXT_PER_REQUEST,
    PHYSICAL_KV_HEADS,
    VRAM_GIB,
    parse_ratios,
    tile_counts,
)

GPUS = ("xtx0", "xtx1", "r9700")
LAYERS = 64
HIDDEN = 5120
INTERMEDIATE = 17408
SCALE_BYTES_PER_GROUP = 2
GROUP_WIDTH = 128
CODE_BYTES_PER_WEIGHT = 1
FFN_ROW_RE = re.compile(
    r"^\s*blk\.(\d+)\.ffn_(gate_up|down)\.weight\s+"
    r"(\S+)\s+(\d+)x(\d+)\s+",
    re.MULTILINE,
)
ENCODING = "i8*bf16[1x128]/fwht128"
REQUIRED_META = {
    "n_layers": LAYERS,
    "n_embd": HIDDEN,
    "n_ff": INTERMEDIATE,
    "n_head_kv": KV_HEADS,
    "head_dim": HEAD_DIM,
    "n_ctx_train": NATIVE_CONTEXT_PER_REQUEST,
}


def parse_rad_info(output: str) -> dict:
    if not re.search(r"(?m)^\s*format\s+RAD2,", output):
        raise ValueError("not native RAD2 container metadata")
    if not re.search(r"(?m)^\s*source quant\s+w8a8_fwht128\s*$", output):
        raise ValueError("not the native W8A8/fwht128 format")
    for field, expected in REQUIRED_META.items():
        matches = re.findall(r"(?m)^\s*" + re.escape(field) + r"\s+(\d+)\s*$", output)
        if not matches or any(int(v) != expected for v in matches):
            raise ValueError(f"unexpected/missing native model field {field}")
    got: dict[tuple[int, str], tuple[str, int, int]] = {}
    for layer, kind, encoding, n, k in FFN_ROW_RE.findall(output):
        key = (int(layer), kind)
        if key in got:
            raise ValueError(f"duplicate FFN projection: {key}")
        got[key] = (encoding, int(n), int(k))
    if len(got) != LAYERS * 2:
        raise ValueError(f"expected 128 fused FFN projections, found {len(got)}")
    for layer in range(LAYERS):
        for kind in ("gate_up", "down"):
            expected = (
                ENCODING,
                2 * INTERMEDIATE if kind == "gate_up" else HIDDEN,
                HIDDEN if kind == "gate_up" else INTERMEDIATE,
            )
            if got.get((layer, kind)) != expected:
                raise ValueError(f"unsupported encoding or FFN dimensions: {layer}/{kind}")
    return {
        "confirmed_fused_ffn_projections": len(got),
        "ffn_encoder": ENCODING,
        "layers": LAYERS,
        "intermediate": INTERMEDIATE,
        "hidden": HIDDEN,
    }


def encoded_ffn_bytes(width: int) -> int:
    """For width along FFN intermediate: gate/up 2W*H and down H*W."""
    if not 0 < width <= INTERMEDIATE or width % 128:
        raise ValueError("illegal quant FFN width for group128")
    elements = 3 * LAYERS * HIDDEN * width
    assert elements % GROUP_WIDTH == 0
    return elements * CODE_BYTES_PER_WEIGHT + elements // GROUP_WIDTH * SCALE_BYTES_PER_GROUP


def vram_floor(
    *,
    candidate: str,
    streams: int,
    per_stream_tokens: int,
) -> dict:
    known = {name: ratios for name, ratios in CANDIDATES}
    if candidate not in known:
        raise ValueError("unknown shard candidate")
    if streams < 1 or per_stream_tokens < 1:
        raise ValueError("positive streams and tokens required")
    if per_stream_tokens > NATIVE_CONTEXT_PER_REQUEST:
        raise ValueError("per-stream context exceeds source model's native 262144")
    fractions = parse_ratios(known[candidate])
    tiles = tile_counts(fractions, INTERMEDIATE, 256)
    logical_widths = [n * 256 for n in tiles]
    widths = (logical_widths[0], logical_widths[1], logical_widths[2] + logical_widths[3])
    if sum(widths) != INTERMEDIATE:
        raise ValueError("sharded widths do not cover original FFN")
    total_tokens = streams * per_stream_tokens
    # Key+value, exactly one byte/element, for *full attention* layers.
    kv_per_head_bytes = total_tokens * 2 * ATTENTION_LAYERS * HEAD_DIM
    cards = []
    for card, width, heads, cap in zip(GPUS, widths, PHYSICAL_KV_HEADS, VRAM_GIB):
        ffn = encoded_ffn_bytes(width)
        kv = kv_per_head_bytes * heads
        floor = ffn + kv
        vram_bytes = int(cap * GIB)
        cards.append({
            "gpu": card,
            "ffn_intermediate_width": width,
            "kv_heads": heads,
            "ffn_codes_and_scales_bytes": ffn,
            "fp8_attention_kv_bytes": kv,
            "hard_ffn_plus_kv_floor_bytes": floor,
            "vram_bytes": vram_bytes,
            "headroom_before_all_nonffn_and_runtime_bytes": vram_bytes - floor,
            "physically_impossible_even_without_other_weights": floor > vram_bytes,
        })
    return {
        "kind": "VTP08_MINIMUM_FFN_KV_FLOOR_NOT_ADMISSION_NOT_THROUGHPUT",
        "source": "native RAD2 W8A8/fWHT128 + Qwen3.8-27B attention/KV schema",
        "candidate": candidate,
        "logical_ffn_tile_counts": tiles,
        "logical_ffn_widths": logical_widths,
        "streams": streams,
        "per_stream_tokens": per_stream_tokens,
        "aggregate_tokens": total_tokens,
        "fp8_kv_bytes_per_aggregate_token": 2 * ATTENTION_LAYERS * KV_HEADS * HEAD_DIM,
        "total_ffn_codes_and_scales_bytes": sum(c["ffn_codes_and_scales_bytes"] for c in cards),
        "cards": cards,
        "definite_no_go": any(c["physically_impossible_even_without_other_weights"] for c in cards),
        "positive_floor_headroom_is_not_admission": True,
        "missing": [
            "all non-FFN tensor placement/replication; attention and GDN",
            "drafter and vision", "all scratch and graph arenas",
            "quant/KV layout padding and allocator granularity",
            "three-physical-card communication, CPU orchestration and physical KV residency checks",
        ],
    }


def tp2_xtx_control_floor(streams: int, per_stream_tokens: int) -> dict:
    """Ideal equal FFN + 2/2 KV-head TP2 on two 24-GiB XTXs.

    No other weights included; positive headroom cannot imply admission.
    """
    if streams < 1 or per_stream_tokens < 1 or per_stream_tokens > NATIVE_CONTEXT_PER_REQUEST:
        raise ValueError("invalid stream count or native context")
    width = INTERMEDIATE // 2
    ffn = encoded_ffn_bytes(width)
    kv = streams * per_stream_tokens * 2 * ATTENTION_LAYERS * HEAD_DIM * 2
    cap = int(VRAM_GIB[0] * GIB)
    return {
        "kind": "HYPOTHETICAL_EQUAL_TP2_XTX_FFN_FP8_KV_FLOOR_NOT_ADMISSION",
        "physical_gpus": 2,
        "ffn_width_each": width,
        "kv_heads_each": 2,
        "ffn_codes_and_scales_bytes_each": ffn,
        "kv_bytes_each": kv,
        "hard_floor_bytes_each": ffn + kv,
        "vram_bytes_each": cap,
        "headroom_before_nonffn_runtime_each": cap - ffn - kv,
        "definite_no_go": ffn + kv > cap,
        "positive_floor_headroom_is_not_admission": True,
    }


def evaluate(text: str, *, cases: list[tuple[str, int, int]]) -> dict:
    source = parse_rad_info(text)
    bounds = [
        vram_floor(candidate=c, streams=streams, per_stream_tokens=tokens)
        for c, streams, tokens in cases
    ]
    return {
        "schema": "VTP08_RAD2_W8A8_MODEL_SHAPED_VRAM_LOWER_BOUND_V1",
        "source_verified": source,
        "cases": bounds,
        "hypothetical_equal_tp2_xtx_control": [
            tp2_xtx_control_floor(streams, tokens)
            for _, streams, tokens in cases
        ],
        "numerical_oracle": "NOT_TESTED",
        "serving_admission": "NOT_TESTED",
        "gpu_used": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rad-info", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate", choices=[c for c, _ in CANDIDATES], default="capacity")
    parser.add_argument("--streams", type=int, choices=(1, 4, 6, 8), default=4)
    parser.add_argument("--tokens-per-stream", type=int, default=262144)
    parser.add_argument("--receipt", type=Path, help="new result path, refuses overwrite")
    args = parser.parse_args(argv)
    if args.receipt and args.receipt.exists():
        raise FileExistsError(f"refusing existing result: {args.receipt}")
    if not args.model.is_file() or args.model.is_symlink() or args.model.suffix != ".rad":
        parser.error("existing native regular RAD2 model required")
    if not args.rad_info.is_file():
        parser.error("rad-info executable required")
    result = subprocess.run(
        [str(args.rad_info), "-v", str(args.model.resolve())],
        capture_output=True, text=True, check=True, timeout=30,
    )
    report = evaluate(result.stdout, cases=[
        (args.candidate, args.streams, args.tokens_per_stream),
    ])
    report["model_path"] = str(args.model.resolve())
    report["model_file_bytes"] = args.model.stat().st_size
    report["rad_info_executable"] = str(args.rad_info.resolve())
    # Full-file SHA-256 is performed separately under VTP06 and must be
    # checked by the caller; this is only a metadata-shaped lower bound.
    report["full_container_hash_in_this_tool"] = "NOT_COMPUTED"
    output = json.dumps(report, indent=2) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x", encoding="utf-8") as file:
            file.write(output)
        print(f"receipt={args.receipt} definite_no_go={report['cases'][0]['definite_no_go']}")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
