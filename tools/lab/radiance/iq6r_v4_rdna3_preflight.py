"""Offline gfx1100 IQR code-image and exported-hook admission preflight.

The code and runtime ABI are inspected without loading the plugin or accessing a
GPU. A pass does NOT establish that plugin rows are live/selectable or that
IQR arithmetic works. Device selftests and model bring-up remain mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

REQUIRED = (
    "r4d_rad_gemm_iqr_m64",
    "r4d_rad_gemm_iqr_m64_scratch",
    "r4d_rad_gemm_iqr_tiled",
    "r4d_rad_gemm_iqr_tiled_scratch",
    "r4d_rad_gemm_iqr_gated",
    "r4d_rad_gemm_iqr_gated_scratch",
    "r4d_rad_layout_iqr",
    "r4d_rad_relayout_iqr",
    "r4d_rad_unrelayout_iqr",
    "r4d_rad_embed_lookup_iqr",
    "r4d_rad_layout_iqr_embed",
    "r4d_rad_relayout_iqr_embed",
    "r4d_rad_logits_gemm_iqr",
    "r4d_rad_layout_iqr_logits",
    "r4d_rad_relayout_iqr_logits",
    "r4d_rad_unrelayout_iqr_logits",
    "r4d_rad_layout_iqr_vocab",
    "r4d_rad_relayout_iqr_vocab",
    "r4d_rad_unrelayout_iqr_vocab",
    "r4d_rad_logit_rerank_iqr",
)
REQUIRED_ROWS = (
    "embed_lookup_iqr",
    "logits_gemm_vocab_iqr",
    "logits_gemm_iqr",
    "logit_rerank_iqr",
    "gemm_iqr_nt_m64",
    "gemm_iqr_gated",
    "gemm_iqr_tiled",
)


def symbols(table: str) -> dict[str, int]:
    found: dict[str, int] = {}
    for line in table.splitlines():
        match = re.fullmatch(r"\s*([0-9a-fA-F]+)\s+[A-Za-z]\s+(\S+)\s*", line)
        if match:
            found[match.group(2)] = int(match.group(1), 16)
    return found


def inspect(nm_text: str, offload_text: str, rows_text: str, excluded_text: str) -> dict:
    table = symbols(nm_text)
    if "r3_stub_begin" not in table or "r3_stub_end" not in table:
        raise ValueError("libr3 stub bounds are missing")
    a, b = table["r3_stub_begin"], table["r3_stub_end"]
    if a > b:
        raise ValueError("malformed stub bounds")
    missing = [key for key in REQUIRED if key not in table]
    if missing:
        raise ValueError(f"missing exported IQR hooks: {missing}")
    stubbed = [key for key in REQUIRED if a <= table[key] < b]
    if stubbed:
        raise ValueError(f"stubbed exported IQR hooks: {stubbed}")
    images = set(re.findall(r"hipv4-amdgcn-amd-amdhsa--(gfx\d+)", offload_text))
    if "gfx1100" not in images:
        raise ValueError(f"gfx1100 HIP device image missing: {sorted(images)}")
    # The compiled plugin and the pinned source are independent inputs.
    missing_rows = [name for name in REQUIRED_ROWS
                    if not re.search(r'\{\s*"' + re.escape(name) + r'"\s*,', rows_text)]
    if missing_rows:
        raise ValueError(f"required upstream IQR rows missing: {missing_rows}")
    omitted = [line.split()[0] for line in excluded_text.splitlines()
               if line.startswith("r4d_")]
    if "r4d_gemm_iqr" in omitted:
        raise ValueError("IQR GEMM unit excluded from RDNA3 build")
    return {
        "status": "CARD_FREE_SOURCE_AND_BINARY_CONTRACT_PASS_ONLY",
        "device_images": sorted(images),
        "exported_iqr_hooks": len(REQUIRED),
        "required_upstream_iqr_rows": len(REQUIRED_ROWS),
        "stub_range": [hex(a), hex(b)],
        "numeric_tests_passed": False,
        "model_load_passed": False,
        "tp3_passed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--exclude", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if not args.plugin.is_file():
        parser.error("compiled plugin does not exist")
    nm = subprocess.run(["nm", "-D", str(args.plugin)], capture_output=True,
                        text=True, check=True, timeout=120).stdout
    offload = subprocess.run(["llvm-objdump", "--offloading", str(args.plugin)],
                             capture_output=True, text=True, check=True, timeout=180).stdout
    report = inspect(nm, offload, args.rows.read_text(), args.exclude.read_text())
    h = hashlib.sha256()
    with args.plugin.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    report["plugin"] = str(args.plugin)
    report["plugin_sha256"] = h.hexdigest()
    report["source_row_file"] = str(args.rows)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
