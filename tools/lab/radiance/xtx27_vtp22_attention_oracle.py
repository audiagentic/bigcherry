"""VTP22: original BF16 Qwen full-attention source-to-residual CPU oracle.

Consumes the actual original-checkpoint layer-3 Q/K/V/O rank packs.
A separate compiled C++ reference computes all per-rank BF16 GEMVs,
head-local zero-centred RMSNorm, text partial RoPE, causal grouped
softmax, per-query output gates, BF16 output-projection column
partials, XTX0/XTX1/R9700 ordered F32 reduction and BF16 residual.
Emits source-hashed stage outputs for subsequent genuine GPU tests.

No GPU, no KV-cache continuation, no full-model execution. Incoming
hidden is synthetic rather than the preceding model layer's output.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import xtx27_vtp15_gdn_projection as projection
import xtx27_vtp22_attention_pack as pack

SOURCE = Path(__file__).with_name("xtx27_vtp22_attention_ref.cpp")
SCHEMA = "vtp22-qwen38-original-bf16-full-attention-cpu-reference-v1"
ROOT = pack.ROOT
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp22-original-full-attention-cpu"


def validate(packet: Path) -> dict:
    manifest = pack.validate(packet)
    if manifest["layer"] != 3:
        raise ValueError("first original full-attention GPU integration requires layer 3")
    for name, descriptor in manifest["sources"].items():
        actual = projection.sha(
            Path(descriptor["path"]), descriptor["offset"], descriptor["bytes"]
        )
        if actual != manifest["original_tensor_sha256"][name]:
            raise ValueError(f"original BF16 checkpoint payload changed: {name}")
    return manifest


def compile_reference(scratch: Path) -> Path:
    digest = projection.sha(SOURCE)
    directory = scratch / "build" / digest[:16]
    executable = directory / "original_full_attention_cpu"
    if executable.exists():
        sha = (directory / "binary.sha256").read_text().strip()
        if projection.sha(executable) != sha:
            raise ValueError("original full-attention CPU reference binary changed")
        return executable
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "g++", "-O3", "-std=c++17", "-Wall", "-Wextra",
        str(SOURCE), "-o", str(executable),
    ], check=True)
    (directory / "binary.sha256").write_text(projection.sha(executable) + "\n")
    return executable


def run(packet: Path, destination: Path, scratch: Path) -> dict:
    verified = validate(packet)
    out = destination.expanduser().resolve()
    if out.exists() or out == ROOT or ROOT in out.parents:
        raise ValueError("fresh CPU receipt directory outside Git required")
    executable = compile_reference(scratch.resolve())
    out.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": SCHEMA, "status": "INCOMPLETE",
        "created_utc": datetime.now(UTC).isoformat(),
        "writer_sha256": projection.sha(Path(__file__)),
        "cpu_source_sha256": projection.sha(SOURCE),
        "cpu_binary_sha256": projection.sha(executable),
        "original_packet": str(packet.resolve()),
        "original_packet_sha256": projection.sha(packet),
        "layer": verified["layer"], "rows": verified["rows"],
        "outputs": {}, "not_gpu": True, "not_model_serving": True,
        "text_only_mrope": True, "text_rope_partial_dim": 64,
        "softmax_causal": True, "gated_output_projection": True,
    }
    file = out / "summary.json"
    file.write_text(json.dumps(record, indent=2) + "\n")
    try:
        p = subprocess.run([
            str(executable), "--rows", str(verified["rows"]),
            "--pack-dir", str(packet.resolve().parent),
            "--out-dir", str(out),
        ], capture_output=True, text=True, check=True)
        if "status=ORIGINAL_BF16_FULL_ATTENTION_CPU_LAYER3_REFERENCE_PASS" not in p.stdout:
            raise ValueError("full-attention CPU oracle did not declare PASS")
        record["oracle_stdout"] = p.stdout.strip()
        for role, qh, kvh in zip(pack.ROLES, pack.QUERIES, pack.KVS, strict=True):
            for name, width, bytes_per in (
                ("q-gate", qh * 512, 2),
                ("k-raw", kvh * 256, 2),
                ("v", kvh * 256, 2),
                ("q-rope", qh * 512, 2),
                ("k-rope", kvh * 256, 2),
                ("attended-gated", qh * 256, 2),
                ("partial", 5120, 4),
            ):
                target = out / f"{name}-{role}.{'f32' if bytes_per == 4 else 'bf16'}"
                extent = verified["rows"] * width * bytes_per
                if target.stat().st_size != extent:
                    raise ValueError(f"{role}: wrong original full-attention output: {name}")
                record["outputs"][f"{name}-{role}"] = {
                    "file": str(target), "shape": [verified["rows"], width],
                    "bytes": extent, "sha256": projection.sha(target),
                }
        for name, width, bytes_per in (
            ("original-attention-reduced.f32", 5120, 4),
            ("original-attention-post-residual.bf16", 5120, 2),
        ):
            target = out / name
            extent = verified["rows"] * width * bytes_per
            if target.stat().st_size != extent:
                raise ValueError(f"wrong whole-matrix output size: {name}")
            record["outputs"][name] = {
                "file": str(target), "shape": [verified["rows"], width],
                "bytes": extent, "sha256": projection.sha(target),
            }
        record["status"] = "ORIGINAL_BF16_FULL_ATTENTION_CAUSAL_GATED_CPU_ORACLE_PASS_NOT_GPU"
    except Exception as exc:
        record.update({"status": "FAILED_INCOMPLETE", "error": str(exc)})
        file.write_text(json.dumps(record, indent=2) + "\n")
        raise
    file.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-receipt", type=Path, required=True)
    p.add_argument("--phase", choices=("plan", "cpu"), default="plan")
    p.add_argument("--run", action="store_true")
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    if args.phase == "cpu" and not args.run:
        p.error("original checkpoint CPU numerical work needs explicit --run")
    try:
        record = validate(args.packet_receipt)
        if args.phase == "plan":
            print(json.dumps({
                "schema": SCHEMA, "status": "VALIDATED_ORIGINAL_BF16_ATTENTION_PACK",
                "layer": record["layer"], "rows": record["rows"],
                "source_sha256": projection.sha(SOURCE),
                "input_pack_sha256": projection.sha(args.packet_receipt),
                "gpu_required": False,
                "not_model_generation": True,
            }, indent=2))
            return 0
        target = args.output or args.scratch / datetime.now(UTC).strftime(
            f"layer{record['layer']}-m{record['rows']}-%Y%m%dT%H%M%S%fZ"
        )
        result = run(args.packet_receipt, target, args.scratch)
        print(json.dumps({
            "status": result["status"],
            "receipt": str(target.resolve() / "summary.json"),
            "oracle": result["oracle_stdout"],
        }, indent=2))
        return 0
    except (KeyError, ValueError, OSError, TypeError, subprocess.CalledProcessError) as err:
        p.error(str(err))


if __name__ == "__main__":
    raise SystemExit(main())
