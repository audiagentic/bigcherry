"""VTP20: one on-GPU original BF16 GDN full-block chain (3 physical GPUs).

The original checkpoint BF16 QKV/a/b/z projections run on each card
then immediately execute original conv -> learned delta recurrence and
gated norm -> GPU transpose/BF16 -> original out_proj column shard.
Pinned host D2H gathers three FP32 partials in XTX0/XTX1/R9700
order, adds the exact synthetic BF16 residual, and broadcasts to
each card. Independent CPU proofs validate every stage. This is a
one-layer original-weight microkernel proof, NOT 64-layer model
serving, native quant GEMM, context caching or real text generation.

Default is a GPU-free, fail-closed plan. Build and hardware require
explicit --run, GPU additionally requires exclusive agent-lease.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import xtx27_vtp15_gdn_integrated_gpu as core_gpu
import xtx27_vtp16_gdn_output_gpu as output_gpu
import xtx27_vtp_micro as micro

ROOT = micro.ROOT
SOURCE = Path(__file__).with_name("xtx27_vtp20_gdn_full_block.hip")
SOURCES = {"gdn_full_block": SOURCE}
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp20-full-gdn-block"
PURPOSE = "VTP20: original BF16 Qwen GDN GPU integrated full block TP3"
RECEIPT_TYPE = "VTP20_ORIGINAL_BF16_QWEN_GDN_COMPLETE_BLOCK_GPU_CHAIN_TP3"


def validate(
    weights: Path, projected: Path, core: Path, learned: Path,
    output_packet: Path, output_reference: Path,
) -> dict:
    first = core_gpu.validate(weights, projected, core, learned)
    last = output_gpu.validate(output_packet, output_reference)
    packet = json.loads(output_packet.read_text())
    if (first["rows"] != last["rows"]
            or packet["weight_input_sha256"] != core_gpu.ref.sha(weights)
            or packet["core_sha256"] != core_gpu.ref.sha(core)
            or packet["layer"] != last["layer"]
            or packet["layer"] != 0):
        raise ValueError("VTP20 requires one matching original checkpoint GDN layer 0")
    if (first["core_receipt_sha256"] != packet["core_sha256"]
            or first["weight_receipt_sha256"] != packet["weight_input_sha256"]):
        raise ValueError("original BF16 checkpoint core/source provenance does not agree")
    return {
        **first,
        "original_output_packet": last["packet_file"],
        "original_output_packet_sha256": last["packet_sha256"],
        "original_output_reference": last["reference_file"],
        "original_output_reference_sha256": last["reference_sha256"],
        "original_output_pack_directory": last["packet_directory"],
        "original_output_reference_directory": last["reference_directory"],
        "residual_file": last["residual_file"],
        "output_column_shards": [1536, 1536, 3072],
        "synthetic_incoming_hidden": True,
        "original_BF16_GDN_projection_conv_recurrence_out_proj": True,
        "full_model": False,
        "long_context_state": False,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("plan", "compile", "gpu"), default="plan")
    p.add_argument("--run", action="store_true")
    p.add_argument("--_under-lease", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--weights-receipt", type=Path)
    p.add_argument("--projection-receipt", type=Path)
    p.add_argument("--core-receipt", type=Path)
    p.add_argument("--learned-receipt", type=Path)
    p.add_argument("--output-packet", type=Path)
    p.add_argument("--output-reference", type=Path)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    p.add_argument("--hipcc")
    args = p.parse_args(argv)
    if args.phase != "plan" and not args.run:
        p.error("HIP compilation and GPU execution require explicit --run")
    if args._under_lease and os.environ.get("VTP04_LEASED") != "1":
        p.error("internal GPU runner requires agent-lease environment")
    if not 1 <= args.iterations <= 30:
        p.error("iterations must be in 1..30")
    sources = (
        args.weights_receipt, args.projection_receipt, args.core_receipt,
        args.learned_receipt, args.output_packet, args.output_reference,
    )
    if args.phase == "gpu" and any(x is None for x in sources):
        p.error("GPU needs original matched GDN projection/core/output receipts")
    try:
        validated = validate(*sources) if all(x is not None for x in sources) else None
    except (ValueError, KeyError, OSError, TypeError, json.JSONDecodeError) as e:
        p.error(str(e))
    digest = micro.source_digest(SOURCES)
    hashes = {
        "full_gpu_source_sha256": core_gpu.ref.sha(SOURCE),
        "gpu_driver_sha256": core_gpu.ref.sha(Path(__file__)),
        "input_gpu_driver_sha256": core_gpu.ref.sha(Path(core_gpu.__file__)),
        "output_gpu_driver_sha256": core_gpu.ref.sha(Path(output_gpu.__file__)),
        "shared_driver_sha256": core_gpu.ref.sha(Path(micro.__file__)),
    }
    if args.phase == "plan":
        print(json.dumps({
            "schema": "vtp20-original-BF16-Qwen-GDN-entire-block-physical-TP3-v1",
            "phase": "card_free",
            "source_digest": digest, "source_hashes": hashes,
            "original_reference_sources": validated,
            "not_full_model": True,
            "not_native_radiance": True,
        }, indent=2))
        return 0
    compiler = micro.compiler_path(args.hipcc)
    if compiler is None:
        p.error("HIP compiler unavailable")
    scratch = args.scratch.resolve()
    if scratch == ROOT or ROOT in scratch.parents:
        p.error("scratch output must be outside Git")
    binary = micro.build_binaries(compiler, scratch, digest, sources=SOURCES)
    if args.phase == "compile":
        print(json.dumps({
            "status": "original_BF16_GDN_entire_block_multiarch_compiled_no_GPU",
            "source_digest": digest, "binary": str(binary["gdn_full_block"]),
        }, indent=2))
        return 0
    if not args._under_lease:
        forwarded = [
            sys.executable, str(Path(__file__).resolve()),
            "--phase", "gpu", "--run", "--_under-lease",
            "--iterations", str(args.iterations),
            "--scratch", str(scratch),
            "--hipcc", str(compiler),
        ]
        for name, value in zip((
            "--weights-receipt", "--projection-receipt", "--core-receipt",
            "--learned-receipt", "--output-packet", "--output-reference",
        ), sources, strict=True):
            forwarded.extend([name, str(value)])
        return subprocess.run([
            "agent-lease", "acquire", "gpu", "--ttl", "30m",
            "--purpose", PURPOSE, "--", "env", "VTP04_LEASED=1", *forwarded,
        ], check=False).returncode
    case = {
        "kind": "gdn_full_block",
        "label": f"original-BF16-complete-GDN-three-GPU-m{validated['rows']}",
        "args": [
            "--rows", str(validated["rows"]),
            "--iterations", str(args.iterations),
            "--conv-dir", validated["qkv_directory"],
            "--pack-dir", validated["pack_directory"],
            "--reference-dir", validated["reference_directory"],
            "--core-dir", validated["core_directory"],
            "--coefficients", validated["coefficients"],
            "--output-pack-dir", validated["original_output_pack_directory"],
            "--output-reference-dir", validated["original_output_reference_directory"],
            "--residual", validated["residual_file"],
        ],
        "data_definition": (
            "ONE connected original BF16 Qwen GDN layer-0 GPU chain;"
            " same three physical XTX0/XTX1/R9700 HIP streams;"
            " checkpoint QKV/A/B/Z -> conv+delta+norm -> GPU head-major"
            " transpose BF16 -> original out_proj -> pinned D2H host"
            " ordered F32 sum -> BF16 residual -> H2D broadcast;"
            " independent CPU projection/GDN/output-oracle receipts;"
            " synthetic incoming hidden, not 64-layer model serving"
        ),
    }
    out = micro.run_leased(
        scratch=scratch, binaries=binary, cases=[case], digest=digest,
        purpose=PURPOSE, receipt_type=RECEIPT_TYPE,
    )
    receipt_file = out / "summary.json"
    receipt = json.loads(receipt_file.read_text())
    receipt.update({
        "schema": "vtp20-original-BF16-Qwen-GDN-entire-block-physical-TP3-v1",
        "original_reference_sources": validated,
        "code_hashes": hashes,
        "single_gpu_dependency_chain_to_output": True,
        "native_radiance_model": False,
        "long_context_state": False,
        "full_model_generation": False,
    })
    receipt_file.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "status": "VTP20_ORIGINAL_BF16_COMPLETE_GDN_GPU_BLOCK_NUMERICAL_PASS_NOT_MODEL",
        "receipt": str(receipt_file),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
