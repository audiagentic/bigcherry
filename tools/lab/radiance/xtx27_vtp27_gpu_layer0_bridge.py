"""VTP27: source-verified real GPU GDN -> FFN BF16 layer-0 handoff.

Executes the already independently accepted VTP20 full BF16 Qwen
GDN and VTP25 full BF16 FFN as separate subprocesses, but under
ONE exclusive GPU lease with an actual checkpoint-backed BF16
residual file produced by GDN GPU and imported by FFN GPU.

For M1 the independently checked VTP20 GPU residual was bit-exact
against the original-BF16 CPU reference. The VTP27 FFN importer
REFUSES any GPU residual differing even one BF16 bit from the
source-hashed post-GDN checkpoint-derived reference; it does
not pretend to accept M4, whose prior VTP20 result had 109
BF16 differences. This *is not* resident GPU-to-GPU transport:
the state is written through a file and each stage relaunches
the three HIP contexts. It proves the first genuine model
layer boundary without fabricating a 64-layer engine.

Requires explicit --run, fresh exclusive agent-lease and
no KFD processes from other agents. No production changes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import xtx27_vtp20_gdn_full_block_gpu as gdn
import xtx27_vtp24_ffn_pack as pack
import xtx27_vtp25_ffn_gpu as ffn
import xtx27_vtp_micro as micro

SOURCE_GDN = Path(__file__).with_name("xtx27_vtp27_gdn_export.hip")
SOURCE_FFN = Path(__file__).with_name("xtx27_vtp27_ffn_import.hip")
SOURCES = {"gdn_export": SOURCE_GDN, "ffn_import": SOURCE_FFN}
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", micro.ROOT.parent)) / "xtx27-prefill" / "vtp27-layer0-gpu-gdn-ffn-bridge"
SCHEMA = "vtp27-qwen38-original-BF16-layer0-gpu-GDN-to-FFN-bridge-v1"
PURPOSE = "VTP27: original BF16 GPU GDN-to-FFN physical TP3 first layer"
RECEIPT_TYPE = "VTP27_ORIGINAL_BF16_GDN_TO_FFN_GPU_SEQUENTIAL_HANDOFF"


def checked_sources(
    weights: Path, projected: Path, core: Path, learned: Path,
    output_packet: Path, output_reference: Path,
    ffn_packet: Path, ffn_reference: Path,
) -> dict:
    first = gdn.validate(weights, projected, core, learned,
                         output_packet, output_reference)
    last = ffn.validate(ffn_packet, ffn_reference)
    packet_data = pack.validate(ffn_packet)
    if first["rows"] != 1 or last["rows"] != 1:
        raise ValueError("VTP27 GPU-to-GPU layer bridge is limited to BF16-exact M1")
    if (first["original_output_reference_sha256"]
            != packet_data["input"]["receipt_sha256"]
            or Path(first["original_output_packet"]).resolve()
            != Path(packet_data["input"]["gdn_packet"]).resolve()
            or first["original_output_packet_sha256"]
            != pack.projection.sha(Path(packet_data["input"]["gdn_packet"]))
            or packet_data["layer"] != 0):
        raise ValueError("VTP27 actual GDN output and FFN sources have incompatible ancestry")
    return {
        "rows": 1, "gdn": first, "ffn": last,
        "original_BF16_GDN_FFN_same_checkpoint": True,
        "upstream_hidden_is_synthetic": True,
        "separate_gpu_process_for_each_model_operator": True,
        "live_resident_hidden_between_operators": False,
        "stage_plan_layers": 1,
        "full_model": False,
    }


def exact_bf16(expected: Path, actual: Path) -> dict:
    first = expected.read_bytes()
    second = actual.read_bytes()
    if len(first) != 5120 * 2 or len(second) != len(first):
        raise ValueError("VTP27 GDN->FFN model BF16 handoff size invalid")
    if first != second:
        differences = sum(a != b for a, b in zip(
            struct.unpack("<5120H", first), struct.unpack("<5120H", second), strict=True
        ))
        raise ValueError(
            f"VTP27 GPU GDN output did not exactly match FFN input: {differences} BF16 words"
        )
    return {
        "size_bytes": len(first),
        "exact_values": 5120,
        "expected_sha256": pack.projection.sha(expected),
        "GPU_export_sha256": pack.projection.sha(actual),
        "exact": True,
    }


def bf16_reference_difference(expected: Path, actual: Path) -> dict:
    a = expected.read_bytes()
    b = actual.read_bytes()
    if len(a) != 5120 * 2 or len(b) != len(a):
        raise ValueError("VTP27 full-layer GPU FFN output extent invalid")
    expected_vals = struct.unpack("<5120H", a)
    actual_vals = struct.unpack("<5120H", b)
    differences = 0
    max_error = 0.0
    for x, y in zip(expected_vals, actual_vals, strict=True):
        differences += x != y
        bx = struct.unpack("<f", struct.pack("<I", x << 16))[0]
        by = struct.unpack("<f", struct.pack("<I", y << 16))[0]
        error = abs(bx - by)
        if not math.isfinite(by) or error > 0.2 + 0.04 * abs(bx):
            raise ValueError("VTP27 GPU post-layer0 hidden exceeds original FFN reference")
        max_error = max(max_error, error)
    return {
        "size_bytes": len(b), "values": 5120,
        "exact_count": 5120 - differences,
        "nonexact_count": differences,
        "max_abs_difference": max_error,
        "original_CPU_reference_sha256": pack.projection.sha(expected),
        "GPU_result_sha256": pack.projection.sha(actual),
        "tolerance_pass": True,
        "bitwise_exact": differences == 0,
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
    p.add_argument("--ffn-packet", type=Path)
    p.add_argument("--ffn-reference", type=Path)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    p.add_argument("--hipcc")
    a = p.parse_args(argv)
    if a.phase != "plan" and not a.run:
        p.error("VTP27 compile/GPU stages require explicit --run")
    if a._under_lease and os.environ.get("VTP04_LEASED") != "1":
        p.error("internal VTP27 GPU runner requires a real agent-lease wrapper")
    if not 1 <= a.iterations <= 20:
        p.error("VTP27 iteration count must be 1..20")
    names = (
        "--weights-receipt", "--projection-receipt", "--core-receipt",
        "--learned-receipt", "--output-packet", "--output-reference",
        "--ffn-packet", "--ffn-reference",
    )
    originals = (
        a.weights_receipt, a.projection_receipt, a.core_receipt, a.learned_receipt,
        a.output_packet, a.output_reference, a.ffn_packet, a.ffn_reference,
    )
    if a.phase == "gpu" and any(s is None for s in originals):
        p.error("all eight original checkpoint and CPU reference receipts are mandatory")
    try:
        checked = (checked_sources(*originals)
                   if all(s is not None for s in originals) else None)
    except (ValueError, TypeError, OSError, KeyError) as e:
        p.error(str(e))
    digest = micro.source_digest(SOURCES)
    source_hashes = {
        name: pack.projection.sha(file) for name, file in SOURCES.items()
    }
    evidence = {
        "schema": SCHEMA, "source_digest": digest,
        "source_hashes": source_hashes,
        "driver_sha256": pack.projection.sha(Path(__file__)),
        "GDN_driver_sha256": pack.projection.sha(Path(gdn.__file__)),
        "FFN_driver_sha256": pack.projection.sha(Path(ffn.__file__)),
        "original_model_sources": checked,
        "still_not_full_model": True,
        "only_one_layer": True,
        "not_resident_or_native": True,
    }
    if a.phase == "plan":
        print(json.dumps({**evidence, "phase": "card_free"}, indent=2))
        return 0
    compiler = micro.compiler_path(a.hipcc)
    if compiler is None:
        p.error("HIP compiler not available")
    scratch = a.scratch.resolve()
    if scratch == micro.ROOT or micro.ROOT in scratch.parents:
        p.error("VTP27 artifacts must be written outside Git")
    binaries = micro.build_binaries(compiler, scratch, digest, sources=SOURCES)
    if a.phase == "compile":
        print(json.dumps({
            **evidence, "status": "VTP27_ORIGINAL_BF16_BRIDGE_MULTIARCH_COMPILED",
            "binaries": {key: str(binary) for key, binary in binaries.items()},
        }, indent=2))
        return 0
    if not a._under_lease:
        child = [
            sys.executable, str(Path(__file__).resolve()), "--phase", "gpu",
            "--run", "--_under-lease", "--iterations", str(a.iterations),
            "--scratch", str(scratch), "--hipcc", str(compiler),
        ]
        for name, src in zip(names, originals, strict=True):
            child.extend([name, str(src)])
        return subprocess.run([
            "agent-lease", "acquire", "gpu", "--ttl", "30m",
            "--purpose", PURPOSE, "--", "env", "VTP04_LEASED=1", *child,
        ], check=False).returncode
    handoff = scratch / "handoffs" / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + digest[:10]
    )
    handoff.mkdir(parents=True, exist_ok=False)
    gdn_path = handoff / "gpu-after-GDN-residual.bf16"
    ffn_path = handoff / "gpu-after-layer0-FFN.bf16"
    first = checked["gdn"]
    last = checked["ffn"]
    cases = [
        {
            "kind": "gdn_export", "label": "original-GDN-GPU-export-M1",
            "args": [
                "--rows", "1", "--iterations", str(a.iterations),
                "--conv-dir", first["qkv_directory"],
                "--pack-dir", first["pack_directory"],
                "--reference-dir", first["reference_directory"],
                "--core-dir", first["core_directory"],
                "--coefficients", first["coefficients"],
                "--output-pack-dir", first["original_output_pack_directory"],
                "--output-reference-dir", first["original_output_reference_directory"],
                "--residual", first["residual_file"],
                "--export-bf16", str(gdn_path),
            ],
            "data_definition": (
                "original checkpoint GDN kernel and out projection on 3 GPUs;"
                " source-verified original BF16 post-GDN output;"
                " export only after CPU numerical tolerance pass"
            ),
        },
        {
            "kind": "ffn_import", "label": "original-FFN-GPU-import-M1",
            "args": [
                "--rows", "1", "--iterations", str(a.iterations),
                "--pack-dir", last["pack_directory"],
                "--cpu-dir", last["cpu_directory"],
                "--import-bf16", str(gdn_path),
                "--export-bf16", str(ffn_path),
            ],
            "data_definition": (
                "original BF16 GDN GPU output imported and checked bit-exact"
                " against original GDN CPU baseline; original post-attention"
                " norm, gate/up/SwiGLU/down BF16 on physical TP3 GPUs,"
                " ordered pinned host F32 reduction, layer0 residual export;"
                " not one resident process, only first model layer"
            ),
        },
    ]
    run_dir = micro.run_leased(
        scratch=scratch, binaries=binaries, cases=cases, digest=digest,
        purpose=PURPOSE, receipt_type=RECEIPT_TYPE,
    )
    receipt = run_dir / "summary.json"
    try:
        gdn_verification = exact_bf16(
            Path(last["pack_directory"]) / "input.bf16", gdn_path
        )
        next_ref = Path(last["cpu_directory"]) / "original-layer0-post-ffn.bf16"
        ffn_verification = bf16_reference_difference(next_ref, ffn_path)
    except (OSError, TypeError, ValueError) as exc:
        incomplete = json.loads(receipt.read_text())
        incomplete.update({
            **evidence, "status": "VTP27_FAILED_INCOMPLETE_LAYER0_BOUNDARY",
            "boundary_error": str(exc),
        })
        receipt.write_text(json.dumps(incomplete, indent=2) + "\n")
        raise
    info = json.loads(receipt.read_text())
    info.update({
        **evidence, "status": "VTP27_ACTUAL_GPU_GDN_TO_FFN_LAYER0_HANDOFF_PASS_NOT_MODEL",
        "GDN_to_FFN_handoff": gdn_verification,
        "full_layer0_after_FFN": ffn_verification,
        "GPU_handoff_file": str(gdn_path),
        "layer1_input_GPU_result_file": str(ffn_path),
        "checkpoint_verified_complete_layers": 1,
        "zero_text_tokens_generated": True,
    })
    receipt.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({
        "status": info["status"],
        "receipt": str(receipt),
        "GDN_FFN_BF16_exact": gdn_verification,
        "layer0_post_ffn": ffn_verification,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
