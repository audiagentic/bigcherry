"""Read-only model-shape and existing weight-file inventory for dense27 experiments.

The output is a model facts record, NOT a GPU-memory allocation report.
No full model tensors are read; only config JSON and file metadata are accessed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def inspect_config(config: dict, *, contexts: list[int], kv_bytes: int) -> dict:
    if not isinstance(config, dict):
        raise TypeError("model config must be an object")
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise TypeError("text_config must be an object")
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or not layer_types:
        raise ValueError("model must declare exact layer_types: no inferred pattern")
    n_layers = text.get("num_hidden_layers")
    if n_layers != len(layer_types):
        raise ValueError("num_hidden_layers differs from layer_types length")
    counts = Counter(layer_types)
    allowed = {"full_attention", "linear_attention"}
    if not set(counts) <= allowed:
        raise ValueError(f"unknown layer types: {sorted(set(counts) - allowed)}")
    dims = ("hidden_size", "num_key_value_heads", "head_dim")
    for name in dims:
        value = text.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(kv_bytes, bool) or not isinstance(kv_bytes, int) or kv_bytes <= 0:
        raise ValueError("kv_bytes must be a positive integer")
    for ctx in contexts:
        if isinstance(ctx, bool) or not isinstance(ctx, int) or ctx <= 0:
            raise ValueError("contexts must be positive integers")
    hidden = text["hidden_size"]
    kv_heads = text["num_key_value_heads"]
    head_dim = text["head_dim"]
    full = counts["full_attention"]
    return {
        "model_type": config.get("model_type"),
        "text_architecture": text.get("model_type"),
        "n_layers": n_layers,
        "layer_types": layer_types,
        "counts": dict(sorted(counts.items())),
        "hidden_size": hidden,
        "num_attention_heads": text.get("num_attention_heads"),
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "tp2_even_kv_heads": (kv_heads % 2 == 0),
        "tp3_even_kv_heads": (kv_heads % 3 == 0),
        "classical_kv": {
            str(ctx): {
                "dtype_bytes_per_element": kv_bytes,
                "logical_full_attention_bytes": 2 * ctx * full * kv_heads * head_dim * kv_bytes,
                "per_full_attention_layer_bytes": 2 * ctx * kv_heads * head_dim * kv_bytes,
                "per_rank_if_tp2_head_sharded_bytes": (
                    ctx * full * kv_heads * head_dim * kv_bytes
                    if kv_heads % 2 == 0 else None
                ),
            }
            for ctx in contexts
        },
        "stage_activation_payload_bytes": {
            "decode_one_row_bf16": hidden * 2,
            "prefill_512_rows_bf16": hidden * 2 * 512,
            "prefill_2048_rows_bf16": hidden * 2 * 2048,
        },
        "excluded_from_kv_estimate": [
            "linear attention / GDN recurrent states",
            "sparse attention/indexer buffers",
            "MTP state and speculative verification",
            "KV block padding, allocator, graph and duplicated sessions",
        ],
    }


def inspect_files(entries: list[str]) -> list[dict]:
    records = []
    labels = set()
    for item in entries:
        label, separator, filename = item.partition("=")
        if not separator or not label or not filename:
            raise ValueError("each weight-file must be LABEL=/existing/file")
        if label in labels:
            raise ValueError(f"duplicate label {label}")
        labels.add(label)
        path = Path(filename)
        if not path.is_file():
            raise ValueError(f"missing weight file for {label}")
        stat = path.stat()
        if stat.st_size <= 0:
            raise ValueError(f"empty weight file for {label}")
        records.append({"label": label, "filename": path.name,
                        "file_bytes": stat.st_size, "file_gib": round(stat.st_size / (1024 ** 3), 3)})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weight-file", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--context", type=int, action="append", default=[])
    parser.add_argument("--kv-bytes", type=int, choices=[1, 2, 4], default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        info = inspect_config(json.loads(args.config.read_text(encoding="utf-8")),
                              contexts=args.context or [8192, 32768, 65536], kv_bytes=args.kv_bytes)
        info["weights"] = inspect_files(args.weight_file)
        info["limitations"] = [
            "File sizes are not measured VRAM resident tensor allocations.",
            "Weight file provenance, model conversion compatibility, GPU loading and quality are unverified.",
            "No allocation, GPU process, tensor-parallel or layer-split test was performed.",
        ]
    except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    output = json.dumps(info, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
