"""Card-free Qwen3.8-27B IQ6R-V4 source/recipe/calibration inventory.

This is NOT rad-convert --plan-only or a kernel/layout qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def expected_weights() -> set[str]:
    names = set()
    for i in range(64):
        operators = (
            ["attn_qg", "attn_k", "attn_v", "attn_output"]
            if i % 4 == 3 else ["ssm_inz", "ssm_out"]
        )
        operators += ["ffn_gate_up", "ffn_down"]
        names.update(f"blk.{i}.{op}.weight" for op in operators)
    names.add("token_embd.weight")
    names.update(f"output.k{k}.weight" for k in (4, 5, 6, 7))
    return names


def inspect(source: Path, recipe: Path, vhead: Path, calibration: Path | None) -> dict:
    config = json.loads((source / "config.json").read_text())
    architecture = config.get("text_config", config)
    num_layers = architecture.get("num_hidden_layers")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    declared = set(index["weight_map"].values())
    shard_files = sorted(source.glob("model-*-of-*.safetensors"))
    actual = {f.name for f in shard_files}
    entries = {}
    for lineno, line in enumerate(recipe.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pieces = line.split()
        if len(pieces) < 3 or pieces[0] in entries:
            raise ValueError(f"invalid or repeated recipe rule on line {lineno}")
        entries[pieces[0]] = {"quantizer": pieces[1], "options": pieces[2:]}
    expected = expected_weights()
    missing = sorted(expected - entries.keys())
    unexpected = sorted(entries.keys() - expected)
    bad_trunk = [
        name for name, rule in entries.items()
        if name.startswith("blk.") and (
            rule["quantizer"] != "gptq"
            or not any(x.startswith("codes=u") for x in rule["options"])
            or "transform=fwht128" not in rule["options"]
            or "calib=$CALIB" not in rule["options"]
        )
    ]
    calib_layers: list[int] = []
    calib_file_count = 0
    if calibration is not None and calibration.exists():
        for file in calibration.rglob("*.safetensors"):
            match = re.search(r"layers\.(\d+)\.", file.name)
            if match:
                calib_layers.append(int(match.group(1)))
            calib_file_count += 1
    covered = sorted(set(calib_layers))
    errors = []
    if num_layers != 64:
        errors.append(f"expected 64 layers, got {num_layers}")
    if len(actual) != 18 or declared != actual:
        errors.append("checkpoint shard count or index coverage mismatch")
    if missing or unexpected or bad_trunk:
        errors.append("IQ6R recipe rule coverage or trunk encoding mismatch")
    if not vhead.is_file():
        errors.append("vocabulary-head classification file is absent")
    return {
        "status": "offline_inventory_failed" if errors else "offline_inventory_passed",
        "not_a_converter_or_gpu_qualification": True,
        "source": {"layers": num_layers, "shards": len(actual),
                   "index_shards": len(declared),
                   "source_has_tokenizer": (source / "tokenizer.json").is_file(),
                   "source_has_chat_template": (source / "chat_template.jinja").is_file()},
        "recipe": {"sha256": sha256(recipe), "rules": len(entries),
                   "code_widths": dict(sorted(Counter(
                       next((o.removeprefix("codes=") for o in rule["options"]
                             if o.startswith("codes=")), "none")
                       for rule in entries.values()).items())),
                   "missing": missing, "unexpected": unexpected,
                   "invalid_trunk": bad_trunk},
        "vhead": {"present": vhead.is_file(), "sha256": sha256(vhead) if vhead.is_file() else None},
        "calibration": {"path_supplied": calibration is not None,
                        "safetensors_files": calib_file_count,
                        "observed_layers": covered,
                        "observed_all_64_layer_indices": covered == list(range(64)),
                        "hessian_set_complete_and_validated": False,
                        "representativeness_not_established": True},
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--vhead", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.source, args.recipe, args.vhead, args.calibration)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0 if report["status"] == "offline_inventory_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
