"""VTP06 card-free BF16 checkpoint index + safetensors metadata audit.

No GPU, quantization, full-payload reads, or download. Reads at most
MAX_HEADER_BYTES from each shard (plus config/index), checks tensor
sizes, nonoverlapping contiguous offsets, all shard/index references,
and the 27B dense-FFN source geometry. Receipt is deliberately NOT
a full-data cryptographic fingerprint or numerical-quality approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "xtx27.checkpoint.metadata.v1"
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_META_BYTES = 16 * 1024 * 1024
MAX_SHARDS = 256
MAX_TENSORS = 100_000
DTYPE_BYTES = {"BF16": 2}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256_stream(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_metadata(path: Path) -> tuple[dict, str]:
    if path.stat().st_size > MAX_META_BYTES:
        raise ValueError(f"metadata file too large: {path.name}")
    raw = path.read_bytes()
    return json.loads(raw, object_pairs_hook=unique_keys), sha(raw)


def safe_filename(filename: str) -> bool:
    return (
        isinstance(filename, str)
        and filename.endswith(".safetensors")
        and Path(filename).name == filename
        and filename not in (".", "..")
        and "/" not in filename and "\\" not in filename
    )


def shard_metadata(path: Path) -> dict:
    size = path.stat().st_size
    if size < 10:
        raise ValueError(f"{path.name}: shard too small")
    with path.open("rb") as file:
        length_raw = file.read(8)
        header_length = struct.unpack("<Q", length_raw)[0]
        if not 2 <= header_length <= MAX_HEADER_BYTES:
            raise ValueError(f"{path.name}: invalid header length {header_length}")
        if 8 + header_length > size:
            raise ValueError(f"{path.name}: truncated header")
        header_raw = file.read(header_length)
    if len(header_raw) != header_length:
        raise ValueError(f"{path.name}: truncated header")
    header = json.loads(header_raw, object_pairs_hook=unique_keys)
    if not isinstance(header, dict) or not header:
        raise ValueError(f"{path.name}: invalid header object")
    if "__metadata__" in header and not isinstance(header.pop("__metadata__"), dict):
        raise ValueError(f"{path.name}: malformed __metadata__")
    if len(header) > MAX_TENSORS:
        raise ValueError(f"{path.name}: too many tensors")
    cursor = 0
    dtype_counts: Counter[str] = Counter()
    shapes: dict[str, list[int]] = {}
    data_bytes = 0
    for name, value in sorted(
        header.items(), key=lambda item: item[1].get("data_offsets", [-1])[0]
        if isinstance(item[1], dict) else -1
    ):
        if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"{path.name}: {name}: malformed tensor descriptor")
        dtype, shape, offsets = value["dtype"], value["shape"], value["data_offsets"]
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"{path.name}: {name}: expected original BF16, got {dtype}")
        if not isinstance(shape, list) or len(shape) > 8 or any(
            type(d) is not int or d < 0 for d in shape
        ):
            raise ValueError(f"{path.name}: {name}: invalid tensor dimensions")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(
            type(d) is not int or d < 0 for d in offsets
        ):
            raise ValueError(f"{path.name}: {name}: invalid tensor data offsets")
        start, stop = offsets
        if start != cursor or stop < start:
            raise ValueError(f"{path.name}: {name}: noncontiguous or overlapping tensor")
        tensor_bytes = math.prod(shape) * DTYPE_BYTES[dtype]
        if tensor_bytes != stop - start:
            raise ValueError(f"{path.name}: {name}: dtype/shape/byte-count mismatch")
        cursor = stop
        data_bytes += tensor_bytes
        dtype_counts[dtype] += 1
        shapes[name] = shape
    if cursor != size - 8 - header_length:
        raise ValueError(f"{path.name}: final tensor data end != payload size")
    return {
        "bytes": size,
        "header_sha256": sha(length_raw + header_raw),
        "header_bytes": 8 + header_length,
        "payload_bytes": data_bytes,
        "tensors": len(shapes),
        "dtypes": dict(sorted(dtype_counts.items())),
        "shapes": shapes,
    }


def tensor_group(name: str) -> str:
    if name.startswith("model.visual."):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    if name in ("lm_head.weight", "model.language_model.embed_tokens.weight"):
        return "embedding_head"
    if name.startswith("model.language_model.layers.") and ".mlp." in name:
        return "text_ffn"
    if name.startswith("model.language_model."):
        return "text_non_ffn"
    return "other"


def check_dense_ffn(config: dict, shapes: dict[str, list[int]]) -> dict:
    if config.get("model_type") != "qwen3_5" or not isinstance(config.get("text_config"), dict):
        raise ValueError("not a Qwen3.8 dense text checkpoint")
    cfg = config["text_config"]
    layers = cfg.get("num_hidden_layers")
    hidden = cfg.get("hidden_size")
    intermediate = cfg.get("intermediate_size")
    context = cfg.get("max_position_embeddings")
    types = cfg.get("layer_types")
    if (layers != 64 or hidden != 5120 or intermediate != 17408 or
            type(context) is not int or context < 8192 or
            not isinstance(types, list) or len(types) != layers or
            types.count("full_attention") != 16 or
            types.count("linear_attention") != 48):
        raise ValueError("unexpected Qwen3.8-27B config geometry")
    for i in range(layers):
        prefix = f"model.language_model.layers.{i}.mlp."
        for kind, wanted in (
            ("gate_proj", [intermediate, hidden]),
            ("up_proj", [intermediate, hidden]),
            ("down_proj", [hidden, intermediate]),
        ):
            name = prefix + kind + ".weight"
            if shapes.get(name) != wanted:
                raise ValueError(f"missing/invalid FFN tensor geometry: {name}")
    return {
        "layers": layers, "hidden_size": hidden,
        "intermediate_size": intermediate,
        "native_context_limit": context,
        "full_attention_layers": types.count("full_attention"),
        "ffn_projection_tensors": 3 * layers,
    }


def audit(checkpoint: Path, *, fingerprint_payload: bool = False) -> dict:
    checkpoint = checkpoint.expanduser().resolve()
    config_path = checkpoint / "config.json"
    index_path = checkpoint / "model.safetensors.index.json"
    config, config_hash = load_metadata(config_path)
    index, index_hash = load_metadata(index_path)
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError("missing safetensors index weight_map")  # noqa: TRY004
    weight_map = index["weight_map"]
    if not 1 <= len(weight_map) <= MAX_TENSORS:
        raise ValueError("unexpected number of indexed tensors")
    if not all(safe_filename(s) for s in weight_map.values()):
        raise ValueError("invalid or escaping shard filename in index")
    groups: dict[str, set[str]] = defaultdict(set)
    for name, filename in weight_map.items():
        if not isinstance(name, str) or not name:
            raise ValueError("malformed tensor name in index")
        groups[filename].add(name)
    if not 1 <= len(groups) <= MAX_SHARDS:
        raise ValueError("unexpected number of shards")
    shards: dict[str, dict] = {}
    all_shapes: dict[str, list[int]] = {}
    total_bytes = 0
    for filename in sorted(groups):
        path = checkpoint / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or symlinked shard: {filename}")
        item = shard_metadata(path)
        if item["shapes"].keys() != groups[filename]:
            raise ValueError(f"{filename}: index and header tensor sets differ")
        if fingerprint_payload:
            item["full_shard_sha256"] = file_sha256_stream(path)
        all_shapes.update(item["shapes"])
        total_bytes += item["payload_bytes"]
        shards[filename] = {key: value for key, value in item.items() if key != "shapes"}
    if all_shapes.keys() != weight_map.keys():
        raise ValueError("missing or extra tensor names")
    expected = index.get("metadata", {}).get("total_size")
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        raise ValueError("index total_size missing")  # noqa: TRY004
    if not math.isfinite(expected) or total_bytes != expected:
        raise ValueError("indexed total tensor payload mismatch")
    model = check_dense_ffn(config, all_shapes)
    census: dict[str, dict[str, int]] = {}
    for name, shape in all_shapes.items():
        group = census.setdefault(tensor_group(name), {"tensors": 0, "bf16_bytes": 0})
        group["tensors"] += 1
        group["bf16_bytes"] += math.prod(shape) * 2
    if sum(row["bf16_bytes"] for row in census.values()) != total_bytes:
        raise ValueError("per-source-category census does not match indexed payload")
    source_files: dict[str, str] = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "generation_config.json", "vocab.json", "merges.txt"):
        path = checkpoint / name
        if path.is_file() and not path.is_symlink():
            if path.stat().st_size > 32 * 1024 * 1024:
                raise ValueError(f"oversized tokenizer/source metadata: {name}")
            source_files[name] = file_sha256_stream(path)
        elif name in ("tokenizer.json", "tokenizer_config.json"):
            raise ValueError(f"missing required tokenizer metadata: {name}")
    return {
        "schema": SCHEMA,
        "status": "STRUCTURALLY_VALID_HEADER_ONLY_PAYLOAD_NOT_HASHED",
        "checkpoint": str(checkpoint),
        "config_sha256": config_hash,
        "index_sha256": index_hash,
        "source_file_sha256": source_files,
        "source_tensors": len(all_shapes),
        "shard_count": len(shards),
        "total_tensor_payload_bytes": total_bytes,
        "shards": shards,
        "model": model,
        "tensor_census": dict(sorted(census.items())),
        "payload_integrity": (
            "FULL_SHARD_SHA256_FINGERPRINTED_NOT_EXTERNALLY_VERIFIED"
            if fingerprint_payload else "NOT_VERIFIED_FULL_SHARD_SHA256"
        ),
        "numerical_quality": "NOT_TESTED",
        "producer_conversion": "NOT_TESTED",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, help="new output JSON file; refuses overwrite")
    parser.add_argument("--fingerprint-payload", action="store_true",
                        help="stream every shard for SHA-256; heavier I/O, no GPU")
    args = parser.parse_args(argv)
    if args.receipt and args.receipt.exists():
        raise FileExistsError(f"refusing existing receipt before model I/O: {args.receipt}")
    result = audit(args.checkpoint, fingerprint_payload=args.fingerprint_payload)
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x", encoding="utf-8") as file:
            file.write(output)
        print(f"receipt={args.receipt} status={result['status']}")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
