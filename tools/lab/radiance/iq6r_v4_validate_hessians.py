"""Validate all 257 public IQ6R-V4 RADGRAM2 Hessians before conversion.

Full file SHA-256 verification is separately performed by iq6r_v4_fetch_hessians.py.
This checks every header, tensor shape, layer/site coverage and corpus row count.
"""
from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path


def inspect_header(path: Path) -> dict:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors: {path}")
        length = struct.unpack("<Q", prefix)[0]
        if length > 1_000_000:
            raise ValueError(f"unreasonably long header: {path}")
        header = json.loads(handle.read(length))
    metadata = header.get("__metadata__", {})
    tensor = header.get("hessian_upper", {})
    if set(header) != {"__metadata__", "hessian_upper"}:
        raise ValueError(f"unexpected keys in {path}")
    k = int(metadata["k"])
    rows = int(metadata["rows"])
    elements = k * (k + 1) // 2
    if (
        k <= 0
        or rows <= 0
        or tensor.get("dtype") != "F32"
        or tensor.get("shape") != [elements]
        or tensor.get("data_offsets") != [0, elements * 4]
        or metadata.get("layout") != "upper triangle, row-major, packed"
        or path.stat().st_size != 8 + length + elements * 4
    ):
        raise ValueError(f"invalid RADGRAM2 tensor or metadata: {path}")
    return {"k": k, "rows": rows, "linears": metadata.get("linears", "")}


def expected_names(config: dict) -> set[str]:
    types = config["text_config"]["layer_types"]
    if len(types) != 64:
        raise ValueError("expected 64 layer type declarations")
    names = {"lm_head.input.safetensors"}
    for i, kind in enumerate(types):
        prefix = f"model.language_model.layers.{i}."
        names.update({
            prefix + "input.safetensors",
            prefix + "post_attention.input.safetensors",
            prefix + "mlp.down_proj.input.safetensors",
        })
        tap = {
            "full_attention": "self_attn.o_proj.input.safetensors",
            "linear_attention": "linear_attn.out_proj.input.safetensors",
        }[kind]
        names.add(prefix + tap)
    return names


def audit(directory: Path, source: Path, nrows: int) -> dict:
    cfg = json.loads((source / "config.json").read_text())
    expected = expected_names(cfg)
    files = {path.name: path for path in directory.glob("*.safetensors")}
    missing = sorted(expected - files.keys())
    extra = sorted(files.keys() - expected)
    if missing or extra:
        raise ValueError(f"Hessian set missing={missing[:8]} extra={extra[:8]}")
    dimensions = Counter()
    for name in sorted(expected):
        result = inspect_header(files[name])
        if result["rows"] != nrows:
            raise ValueError(f"{name}: row count {result['rows']} != {nrows}")
        if not result["linears"]:
            raise ValueError(f"{name}: missing linears")
        dimensions[result["k"]] += 1
    return {
        "status": "HESSIAN_HEADERS_AND_COVERAGE_PASS_NOT_NUMERICAL_APPROVAL",
        "files": len(expected),
        "rows_each": nrows,
        "dimensions": dict(sorted(dimensions.items())),
        "sha256_checked_by_fetch_step": True,
        "numerical_quality_not_tested": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = audit(args.directory, args.source, args.rows)
    contents = json.dumps(receipt, indent=2) + "\n"
    if args.output:
        args.output.write_text(contents)
    print(contents, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
