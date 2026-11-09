"""Card-free Qwen3.8-27B per-layer quantization recipe and placement *proposal*.

No runtime layer staging or on-the-fly requantization is implemented here.
Default invocation reads local BF16 metadata and prints a plan; --write
writes recipe/manifest to scratch. It never executes rad-convert or touches GPUs.
The recipe uses encodings that Radiance's LinearFP8 declares today:
- W4 asymmetric (u4 + u8 zero + f16 group/128) -> W4A16/W4A8I8 kernel paths
- MXFP4 (FP4 E2M1 + E8M0 scale/group32) -> MXFP4A8 paths
- FP8 E4M3 (block 128x128) -> block-FP8 paths
These are *weight formats*, not proof of native IU4 WMMA execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MXFP4 = "rtn  codes=fp4_e2m1 group=32 scale=e8m0"
W4 = "rtn  codes=u4 zero=u8 group=128 scale=f16 rule=search"
FP8 = "rtn  codes=fp8_e4m3 block=128x128 scale=bf16"
LINEARS = ("ffn_gate_up", "ffn_down", "attn_qg", "attn_k", "attn_v",
           "attn_output", "ssm_inz", "ssm_out")
FFN = ("ffn_gate_up", "ffn_down")
SCHEMA = "xtx27.quant.plan.v1"
REPO_ROOT = Path(__file__).resolve().parents[3]


def shape_from_config(config: dict) -> tuple[int, int]:
    """Validate the dense source proxy and the full-layer stage geometry."""
    if config.get("model_type") != "qwen3_5":
        raise ValueError("only Qwen3.8 dense qwen3_5 checkpoint is supported")
    cfg = config.get("text_config")
    if not isinstance(cfg, dict):
        raise ValueError("missing text_config")  # noqa: TRY004
    layers = cfg.get("num_hidden_layers")
    types = cfg.get("layer_types")
    context = cfg.get("max_position_embeddings")
    if (isinstance(layers, bool) or not isinstance(layers, int) or layers != 64
            or not isinstance(types, list) or len(types) != layers
            or any(x not in ("full_attention", "linear_attention") for x in types)
            or types.count("full_attention") != 16
            or isinstance(context, bool) or not isinstance(context, int)
            or context <= 0):
        raise ValueError("unexpected 27B layer arrangement or context")
    return layers, context


def validate_settings(profile: str, cut: int, preserve: int,
                      suffix_fp8: bool, layers: int) -> None:
    if profile not in ("baseline", "mixed", "selected"):
        raise ValueError("unknown profile")
    if not 1 <= cut < layers:
        raise ValueError("cut must leave nonempty XTX and R9700 stages")
    if not 0 <= preserve < cut:
        raise ValueError("preserve-prefix must be below the layer cut")
    if profile == "baseline" and suffix_fp8:
        raise ValueError("suffix-fp8 requires mixed profile")


def selected_formats(selection: dict, config: dict, cut: int) -> dict[int, str]:
    """Validate provenance-bound measured choices, not runtime staging."""
    if (selection.get("schema") != "xtx27.quant.selection.v1"
            or selection.get("data_kind") != "measured"
            or not selection.get("evidence_id")
            or selection.get("status") != "PREVIEW_ONLY_NOT_END_TO_END_QUALIFIED"
            or selection.get("cut") != cut
            or selection.get("native_context_tokens") !=
            config["text_config"]["max_position_embeddings"]):
        raise ValueError("selection must be measured, consistent and explicit")
    canonical = sha256(json.dumps(config, sort_keys=True).encode())
    if selection.get("source_config_canonical_sha256") != canonical:
        raise ValueError("selection belongs to another BF16 checkpoint config")
    stages = selection.get("stages")
    if not isinstance(stages, list) or len(stages) != 2:
        raise ValueError("selection must have two stage groups")
    choice: dict[int, str] = {}
    for stage, begin, end, given in zip(
        ("xtx", "r9700"), (0, cut), (cut, 64), stages, strict=True
    ):
        if not isinstance(given, dict) or given.get("stage") != stage:
            raise ValueError("selection stages not ordered by device group")
        rows = given.get("chosen")
        if not isinstance(rows, list) or len(rows) != end - begin:
            raise ValueError("selection stage incomplete")
        for idx, row in enumerate(rows, start=begin):
            if (not isinstance(row, dict) or row.get("layer") != idx
                    or row.get("format") not in ("mxfp4", "w4", "fp8")):
                raise ValueError("selected layer missing or invalid")
            choice[idx] = row["format"]
    return choice


def build_recipe(layers: int, profile: str, cut: int, preserve: int,
                 suffix_fp8: bool, *, selection: dict[int, str] | None = None) -> tuple[str, dict]:
    validate_settings(profile, cut, preserve, suffix_fp8, layers)
    if profile == "selected":
        if (selection is None or set(selection) != set(range(layers))
                or any(fmt not in ("mxfp4", "w4", "fp8") for fmt in selection.values())):
            raise ValueError("selected profile needs all explicit per-layer formats")
        if suffix_fp8:
            raise ValueError("suffix-fp8 conflicts with selected profile")
    elif selection is not None:
        raise ValueError("selection allowed only with selected profile")
    lines = [
        "# EXPERIMENT ONLY: source is original BF16, no vendor changes.",
        "# First-match rules: selected W4/FP8 before MXFP4 fallback.",
        "# W4 is a compatible stored encoding, not confirmed native-IU4 WMMA.",
        "# --set quantization_config.quant_method=fp8 --set quantization_config.fmt=e4m3",
    ]
    choices: dict[str, str] = {}
    w4_indices: list[int] = []
    fp8_indices: list[int] = []
    if profile == "mixed":
        w4_indices = list(range(preserve, cut))
        if suffix_fp8:
            fp8_indices = list(range(cut, layers))
    elif profile == "selected" and selection is not None:
        w4_indices = [i for i in range(layers) if selection[i] == "w4"]
        fp8_indices = [i for i in range(layers) if selection[i] == "fp8"]
    for idx in w4_indices:
        for name in FFN:
            lines.append(f"blk.{idx}.{name}.weight  {W4}")
            choices[f"blk.{idx}.{name}.weight"] = "w4"
    for idx in fp8_indices:
        for name in FFN:
            lines.append(f"blk.{idx}.{name}.weight  {FP8}")
            choices[f"blk.{idx}.{name}.weight"] = "fp8"
    for name in LINEARS:
        lines.append(f"blk.*.{name}.weight  {MXFP4}")
    lines.append(f"output.weight  {FP8}")
    lines.append("")
    return "\n".join(lines), {
        "w4_ffn_layer_indices": w4_indices,
        "fp8_ffn_layer_indices": fp8_indices,
        "mx_fallback_linears": list(LINEARS),
        "ffn_overrides": choices,
        "head": "fp8_e4m3*bf16[128x128]",
        "unmatched_weights": "keep original checkpoint precision",
    }


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_plan(checkpoint: Path, output_dir: Path, *, profile: str = "mixed",
              cut: int = 40, preserve: int = 8, suffix_fp8: bool = False,
              convert_bin: str = "rad-convert", home: Path | None = None,
              selected: dict | None = None) -> tuple[dict, str]:
    if not checkpoint.is_dir():
        raise ValueError("BF16 checkpoint directory missing")
    if (output_dir.resolve().is_relative_to(REPO_ROOT)
            or output_dir.resolve().is_relative_to(checkpoint.resolve())):
        raise ValueError("output directory must be outside checkout and source checkpoint")
    cfg_path = checkpoint / "config.json"
    idx_path = checkpoint / "model.safetensors.index.json"
    if not cfg_path.is_file() or not idx_path.is_file():
        raise ValueError("BF16 config.json and safetensors index required")
    if not list(checkpoint.glob("*.safetensors")):
        raise ValueError("BF16 safetensors shards missing")
    index_data = json.loads(idx_path.read_text())
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("safetensors index must have a nonempty weight map")
    shards = set(weight_map.values())
    if (any(not isinstance(item, str) or "/" in item for item in shards)
            or any(not (checkpoint / shard).is_file() for shard in shards)):
        raise ValueError("BF16 safetensors shards referenced by index are missing")
    data = cfg_path.read_bytes()
    layers, context = shape_from_config(json.loads(data))
    selected_map = selected_formats(selected, json.loads(data), cut) if profile == "selected" and selected is not None else None
    recipe, choices = build_recipe(layers, profile, cut, preserve, suffix_fp8, selection=selected_map)
    selection_digest = sha256(json.dumps(selected, sort_keys=True).encode()) if selected is not None else None
    tag = f"{profile}-c{cut}-p{preserve}" + ("-sfp8" if suffix_fp8 else "")
    if selection_digest is not None:
        tag += f"-e{selection_digest[:10]}"
    stem = f"xtx27-{tag}"
    recipe_path = output_dir / f"{stem}.recipe"
    model_path = output_dir / f"{stem}.rad"
    command = [
        convert_bin, str(checkpoint), "-o", str(model_path),
        "--recipe", str(recipe_path),
        "--set", "quantization_config.quant_method=fp8",
        "--set", "quantization_config.fmt=e4m3",
        "--max-spec", "0", "--plan-only", "-v",
    ]
    if home is not None:
        command.extend(["--home", str(home)])
    manifest = {
        "schema": SCHEMA,
        "status": "UNVERIFIED_CARD_FREE_RECIPE",
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": sha256(data),
        "checkpoint_index_sha256": sha256(idx_path.read_bytes()),
        "model_native_context": context,
        "profile": profile,
        "target": {
            "xtx_gfx1100_tp2_layers": [0, cut],
            "r9700_gfx1201_tp1_layers": [cut, layers],
            "runtime_full_layer_staging_implemented": False,
            "runtime_gpu_quant_switching_implemented": False,
        },
        "quantization": choices,
        "recipe_sha256": sha256(recipe.encode()),
        "recipe_path": str(recipe_path),
        "candidate_model_path": str(model_path),
        "next_command_requires_exclusive_gpu_lease_if_using_gpu_kernels": command,
        "gates": [
            "run rad-convert --plan-only against production kernel plugins",
            "prove producer activation planes and all op layouts resolve",
            "convert a matched BF16->MXFP4 control and mixed candidate",
            "rad-kbench selftests, numerical agreement and end-to-end KL",
            "matched TTFT/decode with actual memory and resident KV accounting",
            "explicit stage execution design before GPU-dependent placement",
        ],
    }
    if selected is not None:
        manifest["selection_evidence_id"] = selected.get("evidence_id")
        manifest["selection_sha256"] = selection_digest
    return manifest, recipe


def write_plan(output_dir: Path, manifest: dict, recipe: str) -> None:
    rp = Path(manifest["recipe_path"])
    mp = rp.with_suffix(".json")
    documents = {
        rp: recipe,
        mp: json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    }
    # Fail closed rather than silently overwriting an experiment's provenance.
    for path, content in documents.items():
        if path.exists() and path.read_text() != content:
            raise FileExistsError(f"refusing to overwrite differing experiment: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, content in documents.items():
        path.write_text(content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", choices=("baseline", "mixed", "selected"), default="mixed")
    parser.add_argument("--selection-file", type=Path, help="measured selector JSON, selected profile only")
    parser.add_argument("--cut", type=int, default=40)
    parser.add_argument("--preserve-prefix", type=int, default=8)
    parser.add_argument("--suffix-fp8", action="store_true")
    parser.add_argument("--convert-bin", default="rad-convert")
    parser.add_argument("--home", type=Path)
    parser.add_argument("--write", action="store_true", help="write recipe and manifest; default is read-only")
    args = parser.parse_args(argv)
    try:
        if (args.selection_file is None) != (args.profile != "selected"):
            raise ValueError("selected profile requires --selection-file; others forbid it")
        selection = json.loads(args.selection_file.read_text()) if args.selection_file else None
        manifest, recipe = make_plan(
            args.checkpoint.resolve(), args.output_dir.resolve(),
            profile=args.profile, cut=args.cut, preserve=args.preserve_prefix,
            suffix_fp8=args.suffix_fp8, convert_bin=args.convert_bin,
            home=args.home.resolve() if args.home else None, selected=selection,
        )
        if args.write:
            write_plan(args.output_dir.resolve(), manifest, recipe)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
