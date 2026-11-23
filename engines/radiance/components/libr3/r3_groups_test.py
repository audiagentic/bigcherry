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
    args = p.parse_args()
    text = (args.upstream / "CMakeLists.txt").read_text()
    units = re.findall(r'^  "(r4d_[A-Za-z0-9_]+)\|', text, re.M)
    assert units, "No upstream translation units found"
    counts = Counter(units)
    assert all(n == 1 for n in counts.values()), "Duplicate upstream units"
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
        "r4d_gemm_iqr_p3": "gemm_quant",
        "r4d_gemm_fp8a8": "gemm_quant",
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
