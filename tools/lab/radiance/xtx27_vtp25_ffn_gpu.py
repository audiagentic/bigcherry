"""VTP25: original BF16 Qwen FFN (layer 0) physical-TP3 GPU oracle.

Verifies the VTP24 source-hashed checkpoint BF16 gate/up/down shards,
the real post-attention RMSNorm and original GDN CPU reference residual.
Executes one connected original BF16 gate/up/SwiGLU/down chain per GPU,
with one F32 partial per physical rank, pinned async D2H, fixed-order
host F32 combine, BF16 residual, and three-device broadcast.
Checks each stage against independent original-weight VTP24 CPU oracle.

This is NOT full-model generation: only layer-0 FFN, and GDN output
fed to this FFN is still a saved original-weight CPU reference,
not a live VTP20 GPU-to-FFN resident handoff. Experimental opt-in.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import xtx27_vtp24_ffn_oracle as oracle
import xtx27_vtp24_ffn_pack as pack
import xtx27_vtp_micro as micro

ROOT = pack.ROOT
SOURCE = Path(__file__).with_name("xtx27_vtp25_ffn_original.hip")
SOURCES = {"original_ffn": SOURCE}
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp25-original-ffn-gpu"
SCHEMA = "vtp25-qwen38-original-bf16-layer0-ffn-3physical-gpu-v1"
PURPOSE = "VTP25: real Qwen3.8 original BF16 FFN layer0 on physical TP3 GPUs"
RECEIPT_TYPE = "VTP25_ORIGINAL_BF16_LAYER0_FFN_THREE_GPU_CPU_ORACLE"


def validate(packet_path: Path, cpu_path: Path) -> dict:
    source = oracle.validate(packet_path)
    cpu = json.loads(cpu_path.read_text())
    if (cpu.get("schema") != oracle.SCHEMA
            or cpu.get("status") != "ORIGINAL_BF16_LAYER0_FFN_RESIDUAL_CPU_ORACLE_PASS_NOT_GPU"
            or cpu.get("writer_sha256") != pack.projection.sha(Path(oracle.__file__))
            or cpu.get("source_sha256") != pack.projection.sha(oracle.SOURCE)
            or cpu.get("original_checkpoint_packet_sha256") != pack.projection.sha(packet_path)
            or source["layer"] != 0 or cpu.get("layer") != 0
            or cpu.get("rows") != source["rows"]):
        raise ValueError("original BF16 FFN source/CPU receipt mismatch")
    for key, info in cpu["outputs"].items():
        source_file = Path(info["file"])
        if (not source_file.is_file()
                or source_file.stat().st_size != info["bytes"]
                or info["shape"][0] * info["shape"][1] *
                (4 if key.startswith("partial") or key.endswith(".f32") else 2)
                != info["bytes"]
                or pack.projection.sha(source_file) != info["sha256"]):
            raise ValueError(f"original FFN CPU reference modified: {key}")
    for role in pack.ROLES:
        for kind in ("gate", "up", "mid", "partial"):
            if f"{kind}-{role}" not in cpu["outputs"]:
                raise ValueError(f"original FFN CPU oracle missing {role}/{kind}")
    for kind in ("original-ffn-reduced.f32", "original-layer0-post-ffn.bf16"):
        if kind not in cpu["outputs"]:
            raise ValueError(f"original FFN full output missing: {kind}")
    return {
        "rows": source["rows"], "layer": 0,
        "packet": str(packet_path.resolve()),
        "packet_sha256": pack.projection.sha(packet_path),
        "cpu_oracle": str(cpu_path.resolve()),
        "cpu_oracle_sha256": pack.projection.sha(cpu_path),
        "pack_directory": str(packet_path.resolve().parent),
        "cpu_directory": str(cpu_path.resolve().parent),
        "original_weights": True,
        "ffn_widths": list(pack.WIDTHS),
        "upstream_GDN_is_cpu_reference": True,
        "not_full_model": True,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("plan", "compile", "gpu"), default="plan")
    p.add_argument("--run", action="store_true")
    p.add_argument("--_under-lease", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--packet-receipt", type=Path)
    p.add_argument("--cpu-receipt", type=Path)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    p.add_argument("--hipcc")
    a = p.parse_args(argv)
    if a.phase != "plan" and not a.run:
        p.error("compilation/GPU execution requires explicit --run")
    if a._under_lease and os.environ.get("VTP04_LEASED") != "1":
        p.error("internal VTP25 GPU execution requires validated GPU lease wrapper")
    if not 1 <= a.iterations <= 30:
        p.error("bounded FFN iterations must be in 1..30")
    if a.phase == "gpu" and not (a.packet_receipt and a.cpu_receipt):
        p.error("original checkpoint packet and independent CPU oracle are mandatory")
    try:
        selected = (validate(a.packet_receipt, a.cpu_receipt)
                    if a.packet_receipt and a.cpu_receipt else None)
    except (ValueError, TypeError, OSError, KeyError, json.JSONDecodeError) as e:
        p.error(str(e))
    digest = micro.source_digest(SOURCES)
    evidence = {
        "schema": SCHEMA, "source_digest": digest,
        "source_sha256": pack.projection.sha(SOURCE),
        "driver_sha256": pack.projection.sha(Path(__file__)),
        "pack_reader_sha256": pack.projection.sha(Path(pack.__file__)),
        "cpu_oracle_source_sha256": pack.projection.sha(oracle.SOURCE),
        "cpu_oracle_driver_sha256": pack.projection.sha(Path(oracle.__file__)),
        "input": selected, "full_model": False,
        "model_generation": False, "native_radiance": False,
    }
    if a.phase == "plan":
        print(json.dumps({**evidence, "phase": "card_free"}, indent=2))
        return 0
    compiler = micro.compiler_path(a.hipcc)
    if compiler is None:
        p.error("HIP compiler not found")
    scratch = a.scratch.expanduser().resolve()
    if scratch == ROOT or ROOT in scratch.parents:
        p.error("scratch cannot be in Git")
    binaries = micro.build_binaries(compiler, scratch, digest, sources=SOURCES)
    if a.phase == "compile":
        print(json.dumps({
            **evidence, "status": "ORIGINAL_BF16_FFN_GPU_MULTIARCH_COMPILED_CARD_FREE",
            "binary": str(binaries["original_ffn"]),
        }, indent=2))
        return 0
    if not a._under_lease:
        child = [
            sys.executable, str(Path(__file__).resolve()),
            "--phase", "gpu", "--run", "--_under-lease",
            "--packet-receipt", str(a.packet_receipt),
            "--cpu-receipt", str(a.cpu_receipt),
            "--iterations", str(a.iterations),
            "--scratch", str(scratch), "--hipcc", str(compiler),
        ]
        return subprocess.run([
            "agent-lease", "acquire", "gpu", "--ttl", "30m",
            "--purpose", PURPOSE, "--", "env", "VTP04_LEASED=1", *child,
        ], check=False).returncode
    case = {
        "kind": "original_ffn",
        "label": f"original-BF16-Qwen-layer0-FFN-three-GPU-m{selected['rows']}",
        "args": [
            "--rows", str(selected["rows"]), "--iterations", str(a.iterations),
            "--pack-dir", selected["pack_directory"],
            "--cpu-dir", selected["cpu_directory"],
        ],
        "data_definition": (
            "real original Qwen3.8-27B BF16 layer0 post_attention_layernorm"
            " (from CPU-original GDN output residual), original"
            " 5120/4864/7424 FFN gate+up+SwiGLU+down BF16 GEMV,"
            " concurrent XTX0/XTX1/R9700 output, pinned D2H"
            " F32 ordered host sum, BF16 residual, H2D broadcast;"
            " independent original BF16 CPU source/checkpoint oracle;"
            " single-layer only, not model generation"
        ),
    }
    run_dir = micro.run_leased(
        scratch=scratch, binaries=binaries, cases=[case], digest=digest,
        purpose=PURPOSE, receipt_type=RECEIPT_TYPE,
    )
    receipt = run_dir / "summary.json"
    data = json.loads(receipt.read_text())
    data.update({**evidence, "GPU_FFN_layer0": True})
    receipt.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({
        "status": "VTP25_ORIGINAL_BF16_FFN_NUMERICAL_GPU_PASS_NOT_MODEL",
        "receipt": str(receipt),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
