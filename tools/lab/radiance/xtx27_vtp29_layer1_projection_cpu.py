"""VTP29: original BF16 GDN layer-1 CPU projections from actual GPU layer0.

Consumes VTP28 source-hashed original checkpoint GDN QKV/a/b/z
per-physical-rank BF16 weights and the actual GPU-computed layer0
hidden AFTER its complete GDN+FFN residual. The incoming layer-1
RMSNorm is the original checkpoint input_layernorm.

Reuses the independent VTP15 C++ double-accumulation GEMV
oracle, without pretending that layer-1 computation runs on
GPU or that the remaining 63 layers execute. Produces
source-hashed CPU stage tensors for a future layer1 GDN GPU
numeric oracle with real upstream hidden rather than synthetic.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import xtx27_vtp15_projection_ref as projection_ref
import xtx27_vtp28_layer1_gpu_hidden_pack as pack

ROOT = pack.ROOT
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp29-gpu-layer0-to-original-layer1-projections"
SCHEMA = "vtp29-qwen38-original-BF16-layer1-QKV-abz-from-real-GPU-layer0-v1"
OPS = ("qkv", "a", "b", "z")


def run(packet: Path, destination: Path, scratch: Path) -> dict:
    source = pack.validate(packet)
    target = destination.expanduser().resolve()
    if target.exists() or target == ROOT or ROOT in target.parents:
        raise ValueError("original layer1 QKV CPU stage requires fresh immutable directory")
    exe = projection_ref.compile_reference(scratch.expanduser().resolve())
    target.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": SCHEMA, "status": "INCOMPLETE",
        "created_utc": datetime.now(UTC).isoformat(),
        "writer_sha256": pack.projection.sha(Path(__file__)),
        "original_BF16_layer1_source_sha256": pack.projection.sha(packet),
        "source": str(packet.resolve()), "cpu_source_sha256": pack.projection.sha(projection_ref.SOURCE),
        "cpu_binary_sha256": pack.projection.sha(exe),
        "gpu_computed_actual_layer0_hidden": True,
        "rows": 1, "layer": 1,
        "rank_results": {}, "not_GPU_layer1": True,
        "not_full_model": True,
    }
    output = target / "summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    try:
        x = source["inputs"]["normed_input.bf16"]["file"]
        for role in pack.gdn.ROLES:
            local = {}
            for op in OPS:
                weight = source["rank_packs"][role][op]
                b = target / f"{op}-{role}.bf16"
                f = target / f"{op}-{role}.f32"
                width = weight["shape"][0]
                run = subprocess.run([
                    str(exe),
                    "--m", "1", "--n", str(width), "--k", "5120",
                    "--input", str(x), "--weights", weight["file"],
                    "--out-bf16", str(b), "--out-f32", str(f),
                ], check=True, capture_output=True, text=True)
                if b.stat().st_size != width * 2 or f.stat().st_size != width * 4:
                    raise ValueError("original GPU-fed layer1 projected stage has invalid extent")
                local[op] = {
                    "bf16": str(b), "bf16_sha256": pack.projection.sha(b),
                    "fp32": str(f), "fp32_sha256": pack.projection.sha(f),
                    "shape": [1, width], "oracle_stdout": run.stdout.strip(),
                }
            result["rank_results"][role] = local
            output.write_text(json.dumps(result, indent=2) + "\n")
        result["status"] = "ORIGINAL_BF16_REAL_GPU_LAYER0_TO_LAYER1_GDN_PROJECTION_CPU_PASS"
    except Exception as exc:
        result.update({"status": "FAILED_INCOMPLETE", "error": str(exc)})
        output.write_text(json.dumps(result, indent=2) + "\n")
        raise
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-receipt", type=Path, required=True)
    p.add_argument("--phase", choices=("plan", "cpu"), default="plan")
    p.add_argument("--run", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    a = p.parse_args(argv)
    if a.phase == "cpu" and not a.run:
        p.error("original GPU-fed layer1 CPU reference requires explicit --run")
    try:
        data = pack.validate(a.packet_receipt)
        if a.phase == "plan":
            print(json.dumps({
                "schema": SCHEMA, "status": "ORIGINAL_SOURCE_VERIFIED_NO_GPU",
                "source_receipt_sha256": pack.projection.sha(a.packet_receipt),
                "layer": data["layer"], "rows": 1,
                "real_gpu_layer0_input": True, "not_full_model": True,
            }, indent=2))
            return 0
        output = a.output or a.scratch / datetime.now(UTC).strftime(
            "layer1-m1-%Y%m%dT%H%M%S%fZ"
        )
        result = run(a.packet_receipt, output, a.scratch)
        print(json.dumps({
            "status": result["status"], "receipt": str(output.resolve() / "summary.json"),
            "original_BF16_input_from_GPU_layer0": True,
        }, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
