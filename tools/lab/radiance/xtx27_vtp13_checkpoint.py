"""VTP13 metadata-verified BF16 checkpoint FFN source descriptors.

Card-free. Uses the original safetensors index + validated headers and
returns exact byte ranges for one Qwen3.8-27B FFN layer, whose three projections
will be independently quantized to W8 per output row by the HIP proof.
Hashing selected *payloads* is opt-in and distinct from header provenance.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import xtx27_checkpoint_audit as audit
import xtx27_vtp13_ownership as ownership

OPS = {
    "gate": ("gate_proj", [17408, 5120]),
    "up": ("up_proj", [17408, 5120]),
    "down": ("down_proj", [5120, 17408]),
}
MAX_LAYER = 63


def tensor_sha256(path: Path, offset: int, size: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = size
        while remaining:
            block = stream.read(min(4 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"{path.name}: truncated selected tensor payload")
            h.update(block)
            remaining -= len(block)
    return h.hexdigest()


def descriptors(checkpoint: Path, layer: int = 0, *, hash_payload: bool = False) -> dict:
    if type(layer) is not int or not 0 <= layer <= MAX_LAYER:
        raise ValueError("layer must be an integer in 0..63")
    root = checkpoint.expanduser().resolve(strict=True)
    config, config_hash = audit.load_metadata(root / "config.json")
    ownership.expected_config(config)
    index, index_hash = audit.load_metadata(root / "model.safetensors.index.json")
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise TypeError("missing checkpoint tensor index")
    weights = index["weight_map"]
    tensors = {}
    cache: dict[Path, tuple[dict, dict, int]] = {}
    for role, (op, shape) in OPS.items():
        name = f"model.language_model.layers.{layer}.mlp.{op}.weight"
        filename = weights.get(name)
        if not audit.safe_filename(filename):
            raise ValueError(f"missing/unsafe safetensors filename for {name}")
        file = (root / filename).resolve(strict=True)
        if file.parent != root:
            raise ValueError("escaping checkpoint path not permitted")
        if file not in cache:
            # Validates all tensor sizes, contiguous offsets, BF16 type and
            # full file length, without touching the tensor payload bytes.
            info = audit.shard_metadata(file)
            with file.open("rb") as stream:
                raw_length = stream.read(8)
                length = struct.unpack("<Q", raw_length)[0]
                header = json.loads(stream.read(length), object_pairs_hook=audit.unique_keys)
            cache[file] = info, header, 8 + length
        info, header, base = cache[file]
        if info["shapes"].get(name) != shape:
            raise ValueError(f"changed original BF16 tensor shape: {name}")
        d = header.get(name)
        if not isinstance(d, dict) or d.get("dtype") != "BF16" or d.get("shape") != shape:
            raise ValueError(f"changed source tensor: {name}")
        begin, end = d["data_offsets"]
        nbytes = shape[0] * shape[1] * 2
        if end - begin != nbytes:
            raise ValueError(f"incomplete tensor byte view: {name}")
        start = base + begin
        entry = {
            "name": name, "shape": shape, "file": str(file),
            "file_basename": filename,
            "offset": start, "bytes": nbytes,
            "header_sha256": info["header_sha256"],
        }
        if hash_payload:
            entry["selected_payload_sha256"] = tensor_sha256(file, start, nbytes)
        tensors[role] = entry
    return {
        "schema": "vtp13-checkpoint-bf16-ffn-sources-v1",
        "checkpoint": str(root),
        "layer": layer,
        "config_sha256": config_hash,
        "index_sha256": index_hash,
        "payloads_fingerprinted": hash_payload,
        "tensors": tensors,
        "limitations": "Three original BF16 FFN weight tensors only;"
                       " fixed synthetic BF16 input; not actual model inference.",
    }


def hip_args(receipt: dict) -> list[str]:
    args = []
    for role in ("gate", "up", "down"):
        tensor = receipt["tensors"][role]
        args.extend((f"--{role}-file", tensor["file"],
                     f"--{role}-byte", str(tensor["offset"])))
    return args
