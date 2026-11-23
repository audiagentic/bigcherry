"""Create cumulative, runnable IQ6R-V4 recipes using the pinned W8A8 trunk.

The model's qwen35_w8a8 architecture requires rotated-int8 or IQR trellis
trunk weights AND the four output.k4..k7 vocabulary-head trellis classes.
Each intermediate stage keeps head classes, selected IQ6R trunk GPTQ, then
fallback rotated-int8 wildcard rules for all not-yet-converted trunk weights.
The first matching rule wins, so explicit GPTQ must precede wildcards.
The FINAL stage is byte-identical to the full pinned IQ6R recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PINNED_RECIPE_SHA256 = "0c41614302d448ba61c39f31eb6da1ee4970fcd4fa1a6011ec479d51b64e597f"
PINNED_FALLBACK_SHA256 = "09725afb09d1816380c4128332ebf0eda8488e6018f9248843514a95d826d9ed"
EXPECTED_FALLBACK = {
    "blk.*.attn_qg.weight", "blk.*.attn_k.weight",
    "blk.*.attn_v.weight", "blk.*.attn_output.weight",
    "blk.*.ssm_inz.weight", "blk.*.ssm_out.weight",
    "blk.*.ffn_gate_up.weight", "blk.*.ffn_down.weight",
}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def recipes(source: bytes, fallback: bytes, *, chunk: int) -> list[bytes]:
    if not 1 <= chunk <= 96:
        raise ValueError("chunk size must be 1-96 trunk tensors per stage")
    if sha256(source) != PINNED_RECIPE_SHA256:
        raise ValueError("IQ6R-V4 recipe does not match source pin")
    if sha256(fallback) != PINNED_FALLBACK_SHA256:
        raise ValueError("W8A8 fallback recipe does not match source pin")
    lines = source.decode("utf-8").splitlines(keepends=True)
    rules = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    gptq = [line for line in rules if line.split()[1] == "gptq"]
    other = [line for line in rules if line.split()[1] != "gptq"]
    head = [line for line in gptq if line.startswith("output.k")]
    trunk = [line for line in gptq if line not in head]
    expected_head = {f"output.k{k}.weight" for k in range(4, 8)}
    if (len(gptq) != 292 or len(trunk) != 288 or len(head) != 4
            or {line.split()[0] for line in head} != expected_head
            or len(other) != 1
            or not other[0].startswith("token_embd.weight rtn ")):
        raise ValueError("unexpected quantizer recipe schema or rule count")

    fallback_rules = [
        line for line in fallback.decode("utf-8").splitlines(keepends=True)
        if line.strip() and not line.lstrip().startswith("#")
        and line.split()[0] in EXPECTED_FALLBACK
    ]
    if (len(fallback_rules) != 8
            or {line.split()[0] for line in fallback_rules} != EXPECTED_FALLBACK
            or any("rtn  codes=i8 group=128 scale=bf16 transform=fwht128" not in line
                   for line in fallback_rules)):
        raise ValueError("unexpected pinned rotated-INT8 W8A8 fallback")
    count = 1 + (len(trunk) + chunk - 1) // chunk
    outputs = []
    for stage in range(1, count + 1):
        if stage == count:
            outputs.append(source)
            continue
        selected = set(head + trunk[:(stage - 1) * chunk])
        prefix = "".join(
            line for line in lines
            if line.lstrip().startswith("#") or not line.strip()
            or line in selected or line in other
        )
        # Explicit per-tensor GPTQ rules precede wildcard fallback; first
        # match wins and already-GPTQ tensors remain eligible for --reuse.
        partial = prefix + "\n# Interim W8A8 trunk compatibility only\n" + "".join(fallback_rules)
        outputs.append(partial.encode("utf-8"))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=20)
    args = parser.parse_args()
    stages = recipes(args.source.read_bytes(), args.fallback.read_bytes(),
                     chunk=args.chunk)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, source in enumerate(stages, 1):
        path = args.output / f"stage-{i:02d}.recipe"
        if path.exists() and path.read_bytes() != source:
            raise ValueError(f"refusing overwrite of modified stage recipe: {path}")
        if not path.exists():
            path.write_bytes(source)
        manifest.append({
            "stage": i,
            "gptq_cumulative": min(4 + (i - 1) * args.chunk, 292),
            "sha256": sha256(source),
            "recipe": str(path),
            "intermediate_unqualified": i < len(stages),
        })
    receipt = args.output / "stages.json"
    data = {"full_recipe_sha256": PINNED_RECIPE_SHA256,
            "fallback_recipe_sha256": PINNED_FALLBACK_SHA256,
            "chunk": args.chunk, "stage_count": len(stages),
            "head_first": True, "fallback_rotated_i8": True,
            "stages": manifest, "validated_for_execution": False}
    receipt.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"stage_count": len(stages), "stage_trunk_rules": args.chunk,
                      "first_stage_gptq": 4, "final_sha256": sha256(stages[-1])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
