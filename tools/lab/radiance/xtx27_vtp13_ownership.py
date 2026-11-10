"""VTP13 Qwen3.8-27B asymmetric physical-TP3 tensor-ownership proof.

Reads only original BF16 safetensors *headers*, config and index via the
existing VTP06 metadata verifier; no weight payloads, GPU, conversion, or
runtime placement. This is a structural feasibility result, not engine support.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import xtx27_checkpoint_audit as checkpoint

ROLES = ("xtx0", "xtx1", "r9700")
GDN_K_HEADS = (4, 4, 8)
GDN_V_HEADS = (12, 12, 24)
FULL_Q_HEADS = (6, 6, 12)
FULL_KV_HEADS = (1, 1, 2)
FFN_WIDTHS = (5120, 4864, 7424)
LAYER = "model.language_model.layers."
SCHEMA = "vtp13.qwen38_27b.physical_tp3_tensor_ownership.v1"


def head_offsets(counts: tuple[int, int, int], width: int, start: int = 0) -> list[list[tuple[int, int]]]:
    if len(counts) != 3 or width <= 0 or any(type(x) is not int or x <= 0 for x in counts):
        raise ValueError("invalid physical head counts")
    result = []
    offset = start
    for value in counts:
        extent = value * width
        result.append([(offset, extent)])
        offset += extent
    return result


def validate_parts(shape: list[int], axis: int, parts: list[list[tuple[int, int]]]) -> None:
    if (not shape or not 0 <= axis < len(shape) or len(parts) != 3
            or any(not segments for segments in parts)):
        raise ValueError("missing or malformed physical shards")
    segments = [seg for device in parts for seg in device]
    if any(type(a) is not int or type(b) is not int or a < 0 or b <= 0
           for a, b in segments):
        raise ValueError("invalid shard offset or extent")
    cursor = 0
    for start, width in sorted(segments):
        if start != cursor:
            raise ValueError("gap/overlap in three-physical shard ownership")
        cursor += width
    if cursor != shape[axis]:
        raise ValueError(f"incomplete shard coverage: {cursor} != {shape[axis]}")


def expected_config(config: dict) -> dict:
    if config.get("model_type") != "qwen3_5":
        raise ValueError("not the original Qwen3.8-27B model")
    cfg = config.get("text_config")
    if not isinstance(cfg, dict):
        raise TypeError("missing text_config")
    expected = {
        "num_hidden_layers": 64, "hidden_size": 5120,
        "intermediate_size": 17408, "num_attention_heads": 24,
        "num_key_value_heads": 4, "head_dim": 256,
        "linear_num_key_heads": 16, "linear_num_value_heads": 48,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "attn_output_gate": True,
        "full_attention_interval": 4, "mtp_num_hidden_layers": 1,
    }
    for key, value in expected.items():
        if type(cfg.get(key)) is not type(value) or cfg[key] != value:
            raise ValueError(f"unexpected checkpoint configuration: {key}")
    layer_types = cfg.get("layer_types")
    if (not isinstance(layer_types, list) or len(layer_types) != 64
            or any(t != ("full_attention" if i % 4 == 3 else "linear_attention")
                   for i, t in enumerate(layer_types))):
        raise ValueError("layer-type order is not expected 48-GDN / 16-full")
    return cfg


def make_map(config: dict, shapes: dict[str, list[int]],
             ffn_widths: tuple[int, int, int] = FFN_WIDTHS) -> dict:
    cfg = expected_config(config)
    if (len(ffn_widths) != 3 or any(type(x) is not int or x <= 0 or x % 128
                                    for x in ffn_widths)
            or sum(ffn_widths) != cfg["intermediate_size"]):
        raise ValueError("FFN split is incomplete or not 128-tile aligned")
    ffn = head_offsets(ffn_widths, 1)
    qh = head_offsets(FULL_Q_HEADS, cfg["head_dim"] * 2)
    kv = head_offsets(FULL_KV_HEADS, cfg["head_dim"])
    q_out = head_offsets(FULL_Q_HEADS, cfg["head_dim"])
    gk = head_offsets(GDN_K_HEADS, cfg["linear_key_head_dim"])
    gv = head_offsets(GDN_V_HEADS, cfg["linear_value_head_dim"])
    by_operator: dict[str, dict] = {}
    by_tensor: dict[str, dict] = {}

    def add(name: str, shape: list[int], axis: int, offsets: list[list[tuple[int, int]]],
            *, semantics: str) -> None:
        if shapes.get(name) != shape:
            raise ValueError(f"missing or changed original BF16 shape: {name}")
        validate_parts(shape, axis, offsets)
        if name in by_tensor:
            raise ValueError("duplicate source projection")
        by_tensor[name] = {
            "shape": shape, "split_axis": axis, "meaning": semantics,
            "roles": {role: [{"offset": s, "length": n} for s, n in offsets[i]]
                      for i, role in enumerate(ROLES)},
            "noncontiguous_source_requires_pack": any(len(x) > 1 for x in offsets),
        }
        by_operator[name] = {"segments_per_physical_rank": [len(x) for x in offsets]}

    for i in range(64):
        prefix = f"{LAYER}{i}."
        for n in ("gate_proj", "up_proj"):
            add(prefix + f"mlp.{n}.weight", [17408, 5120], 0, ffn,
                semantics="independent FFN intermediate output rows")
        add(prefix + "mlp.down_proj.weight", [5120, 17408], 1, ffn,
            semantics="FFN intermediate input columns, partial output reduction")
        if i % 4 == 3:
            attn = prefix + "self_attn."
            add(attn + "q_proj.weight", [12288, 5120], 0, qh,
                semantics="interleaved (query, output gate) pair per 256-wide Q head")
            for n in ("k_proj", "v_proj"):
                add(attn + n + ".weight", [1024, 5120], 0, kv,
                    semantics="full attention KV head output rows")
            add(attn + "o_proj.weight", [5120, 6144], 1, q_out,
                semantics="full attention Q head input columns, partial output reduction")
            for n in ("q_norm", "k_norm"):
                name = attn + n + ".weight"
                if shapes.get(name) != [256]:
                    raise ValueError(f"missing or changed replicated norm: {name}")
        else:
            root = prefix + "linear_attn."
            # In the BF16 checkpoint, in_proj_qkv has [Q(2048)|K(2048)|V(6144)]
            # rows. A physical rank must pull THREE disjoint slices, then pack
            # its Q|K|V projection in local head order for GDN's fused in GEMM.
            packed = []
            for rank in range(3):
                packed.append([
                    gk[rank][0],
                    (gk[rank][0][0] + 2048, gk[rank][0][1]),
                    (gv[rank][0][0] + 4096, gv[rank][0][1]),
                ])
            add(root + "in_proj_qkv.weight", [10240, 5120], 0, packed,
                semantics="three source-disjoint Q|K|V parts; local repack REQUIRED")
            for n in ("in_proj_a.weight", "in_proj_b.weight"):
                add(root + n, [48, 5120], 0, head_offsets(GDN_V_HEADS, 1),
                    semantics="one decay/beta scalar per value head")
            add(root + "in_proj_z.weight", [6144, 5120], 0, gv,
                semantics="GDN value head output-gate groups")
            add(root + "conv1d.weight", [10240, 1, 4], 0, packed,
                semantics="matching depthwise Q|K|V channel conv; local repack REQUIRED")
            for n in ("A_log", "dt_bias"):
                add(root + n, [48], 0, head_offsets(GDN_V_HEADS, 1),
                    semantics="value head recurrent coefficient; converted to F32 by engine")
            add(root + "out_proj.weight", [5120, 6144], 1, gv,
                semantics="value head input columns, partial output reduction")
            if shapes.get(root + "norm.weight") != [128]:
                raise ValueError("GDN shared 128-wide norm must be replicated")
    for name, wanted in (
        ("model.language_model.embed_tokens.weight", [248320, 5120]),
        ("lm_head.weight", [248320, 5120]),
    ):
        if shapes.get(name) != wanted:
            raise ValueError(f"missing or changed edge tensor: {name}")
    return {
        "schema": SCHEMA,
        "status": "STRUCTURAL_OWNERSHIP_PROVEN_NOT_ENGINE_SUPPORTED",
        "roles": list(ROLES),
        "physical_ffn_intermediate_widths": list(ffn_widths),
        "full_attention_q_heads": list(FULL_Q_HEADS),
        "full_attention_kv_heads": list(FULL_KV_HEADS),
        "gdn_key_heads": list(GDN_K_HEADS),
        "gdn_value_heads": list(GDN_V_HEADS),
        "layers": 64, "full_attention_layers": 16, "gdn_layers": 48,
        "sharded_source_tensors": len(by_tensor),
        "tensors": by_tensor,
        "noncontiguous_source_tensors": [
            name for name, info in by_tensor.items()
            if info["noncontiguous_source_requires_pack"]
        ],
        "not_proven": [
            "producer packing and per-arch native INT8/FP8 kernel ABI",
            "GDN fused Q/K/V and conv layout, head state and chunk replay",
            "per-rank numerical correctness, BF16 quality, and quant compatibility",
            "the replicated attention/GDN/FFN norms and embedding/vocab edge placement",
            "vision and MTP/drafter per-device placement and admission",
            "three-GPU runtime, collective concurrency, graph and cancellation",
            "real VRAM/KV/scratch fit or higher serving throughput",
            "full safetensors payload integrity (header-only source check)",
        ],
    }


def load_original_checkpoint(root: Path) -> tuple[dict, dict, dict]:
    config, chash = checkpoint.load_metadata(root / "config.json")
    index, ihash = checkpoint.load_metadata(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index weight_map missing")
    if not all(checkpoint.safe_filename(fn) for fn in weight_map.values()):
        raise ValueError("malformed/escaping safetensors file path")
    shapes = {}
    headers = {}
    for name in sorted(set(weight_map.values())):
        hdr = checkpoint.shard_metadata(root / name)
        headers[name] = hdr["header_sha256"]
        for key, shape in hdr["shapes"].items():
            if key in shapes or weight_map.get(key) != name:
                raise ValueError("index/header inconsistency")
            shapes[key] = shape
    if shapes.keys() != weight_map.keys():
        raise ValueError("index/header tensor inventory mismatch")
    return config, shapes, {
        "config_sha256": chash, "index_sha256": ihash,
        "header_sha256": headers, "source_tensor_count": len(shapes),
        "payload_integrity": "NOT_VERIFIED_FULL_SHARD_SHA256",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.output and args.output.exists():
        p.error("refusing to overwrite existing receipt")
    config, shapes, meta = load_original_checkpoint(args.checkpoint)
    result = make_map(config, shapes)
    # A digest of headers and config establishes metadata provenance, not weight integrity.
    result["checkpoint_provenance"] = meta
    if args.output:
        checkout = Path(__file__).resolve().parents[3]
        if args.output.resolve().is_relative_to(checkout):
            p.error("output must be outside the checkout")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(json.dumps({"status": result["status"], "receipt": str(args.output),
                          "tensor_count": len(result["tensors"]),
                          "noncontiguous": len(result["noncontiguous_source_tensors"])}))
    else:
        print(json.dumps({k: v for k, v in result.items() if k != "tensors"}, indent=2))


if __name__ == "__main__":
    main()
