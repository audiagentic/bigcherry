#!/usr/bin/env python3
"""Verify every upstream device translation unit has exactly one build family."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--exclude", type=Path, help="exclude.txt: units libr3 leaves out (owned by an overlay or not built yet)")
    args = p.parse_args()
    text = (args.upstream / "CMakeLists.txt").read_text()
    units = re.findall(r'^  "(r4d_[A-Za-z0-9_]+)\|', text, re.M)
    assert units, "No upstream translation units found"
    counts = Counter(units)
    assert all(n == 1 for n in counts.values()), "Duplicate upstream units"
    if args.exclude:
        left_out = {line.split()[0] for line in args.exclude.read_text().splitlines() if line.startswith("r4d_")}
        assert left_out <= set(counts), f"exclude.txt names units upstream does not have: {sorted(left_out - set(counts))}"
        for unit in left_out:
            del counts[unit]   # an excluded unit has no libr3 family: its owner is the overlay (or it is not built yet)
    rows = args.manifest.read_text().splitlines()
    assert rows[0] == "group\tunit", "Invalid group manifest header"
    pairs = [line.split("\t") for line in rows[1:]]
    assert all(len(row) == 2 for row in pairs), "Invalid manifest row"
    grouped = Counter(unit for _, unit in pairs)
    assert grouped == counts, (
        f"Missing: {sorted(counts.keys() - grouped.keys())}; "
        f"Extra: {sorted(grouped.keys() - counts.keys())}; "
        f"Duplicate: {sorted(u for u, n in grouped.items() if n > 1)}"
    )
    groups = {family for family, _ in pairs}
    expected = {
        "core", "gemm_quant", "gemm_dense", "attention",
        "moe", "speculation", "multimodal", "collectives",
    }
    assert groups == expected, f"Unexpected/missing groups: {groups ^ expected}"
    for unit, family in {
        "r4d_gemm_w8a8_tiled": "gemm_quant",
        "r4d_gemm_i8a8": "gemm_quant",
        "r4d_logits_gemm_i8": "gemm_quant",
        "r4d_gram_accum": "gemm_quant",
        "r4d_fused_quant": "core",
        "r4d_ds41_ggml": "gemm_quant",
        "r4d_ds41_ggml_mr": "gemm_quant",
        "r4d_ds41_ggml_mr64": "gemm_quant",
        "r4d_ggml_wmma": "gemm_quant",
        "r4d_ds41_sparse": "attention",
        "r4d_ds41_walk": "speculation",
        "r4d_ds41_elem": "core",
        "r4d_gemm_bf16x12_nt_m64": "gemm_dense",
        "r4d_posemb": "multimodal",
        "r4d_situ": "moe",
        "r4d_gemm_bf16_nt_m16": "gemm_dense",
        "r4d_dflash_select_bf16": "speculation",
        "r4d_attn_paged_h256_gqa6": "attention",
        "r4d_ar_entry": "collectives",
        "r4d_moe": "moe",
        "r4d_ltx": "multimodal",
    }.items():
        assert [g for g, u in pairs if u == unit] == [family], f"Wrong group: {unit}"
    print(f"r3 groups OK: {len(units)} unique upstream units in {len(groups)} groups")


if __name__ == "__main__":
    main()
