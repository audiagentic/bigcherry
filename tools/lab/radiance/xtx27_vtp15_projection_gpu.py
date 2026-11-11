"""VTP15: original BF16 GDN QKV/A/B/Z GEMVs on 3 GPUs, opt-in.

The same BF16 normalized activation feeds original rank-owned checkpoint
weights on both XTXs and the R9700. All 12 independent GPU GEMM outputs
are numerically checked against a double-accumulation CPU oracle.
This proves real projection computation, NOT a complete GDN layer or a
production-speed native Radiance operator. No GPU peer access.

Reuses VTP04's exclusive agent-lease, empty KFD inventory, build
digests, immutable per-case results and failure cleanup.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import xtx27_vtp15_projection_ref as oracle
import xtx27_vtp_micro as micro

ROOT = micro.ROOT
SOURCE = Path(__file__).with_name("xtx27_vtp15_project.hip")
SOURCES = {"original_gdn_proj": SOURCE}
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp15-gdn-gpu"
LEASE_PURPOSE = "VTP15: original BF16 Qwen GDN projections TP3 numerical proof"
RECEIPT_TYPE = "VTP15_ORIGINAL_BF16_GDN_QKV_A_B_Z_GPU_GEMM_THREE_PHYSICAL"


def validate(weight_path: Path, reference_path: Path) -> dict:
    weights = oracle.validate(weight_path)
    reference = json.loads(reference_path.read_text())
    if (reference.get("schema") != "vtp15-original-bf16-GDN-projection-CPU-oracle-v1"
            or reference.get("status") != "ORIGINAL_BF16_QWEN_GDN_INPUT_PROJECTIONS_CPU_PASS_NOT_GPU"):
        raise ValueError("missing completed original BF16 CPU projection oracle")
    if (reference.get("rows") != weights["rows"]
            or reference.get("weight_pack_sha256") != oracle.sha(weight_path)):
        raise ValueError("CPU projection oracle input package changed")
    if (reference.get("oracle_cpp_sha256") != oracle.sha(oracle.SOURCE)
            or reference.get("writer_sha256") != oracle.sha(Path(oracle.__file__))):
        raise ValueError("CPU oracle source changed since producing expected output")
    for role in oracle.ROLES:
        for op in oracle.OPS:
            item = reference["rank_results"][role][op]
            shape = [weights["rows"], weights["rank_packs"][role][op]["shape"][0]]
            if item["shape"] != shape:
                raise ValueError(f"{role} {op}: changed reference shape")
            for suffix, bytes_per_elem in (("bf16", 2), ("fp32", 4)):
                path = Path(item[suffix])
                if (path.stat().st_size != shape[0] * shape[1] * bytes_per_elem
                        or oracle.sha(path) != item[suffix + "_sha256"]):
                    raise ValueError(f"{role} {op}: source/reference bytes changed")
    qkv_receipt = Path(weights["qkv_pack_receipt"])
    qkv = json.loads(qkv_receipt.read_text())
    if qkv["status"] != "EXACT_ORIGINAL_BF16_GDN_QKV_CONV_PACK_PASS_NOT_RUNTIME":
        raise ValueError("not the original BF16 QKV pack")
    return {
        "rows": weights["rows"],
        "weight_receipt": str(weight_path.resolve()),
        "weight_receipt_sha256": oracle.sha(weight_path),
        "reference_receipt": str(reference_path.resolve()),
        "reference_receipt_sha256": oracle.sha(reference_path),
        "pack_directory": str(weight_path.resolve().parent),
        "qkv_directory": str(qkv_receipt.resolve().parent),
        "reference_directory": str(reference_path.resolve().parent),
        "input_layernorm_sha256": weights["source_sha256"]["norm"],
        "real_checkpoint_payload": True,
        "synthetic_hidden": True,
        "cpu_double_accumulation_oracle": True,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("plan", "compile", "gpu"), default="plan")
    p.add_argument("--run", action="store_true")
    p.add_argument("--_under-lease", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--weights-receipt", type=Path)
    p.add_argument("--reference-receipt", type=Path)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    p.add_argument("--hipcc")
    a = p.parse_args(argv)
    if a.phase != "plan" and not a.run:
        p.error("compile/GPU requires --run")
    if a._under_lease and os.environ.get("VTP04_LEASED") != "1":
        p.error("internal GPU child requires exclusive agent-lease")
    if not 1 <= a.iterations <= 30:
        p.error("iterations must be 1..30")
    if a.phase == "gpu" and (not a.weights_receipt or not a.reference_receipt):
        p.error("GPU requires original BF16 pack and independent CPU oracle receipts")
    try:
        matched = (validate(a.weights_receipt, a.reference_receipt)
                   if a.weights_receipt and a.reference_receipt else None)
    except (ValueError, FileNotFoundError, KeyError, TypeError, OSError) as exc:
        p.error(str(exc))
    digest = micro.source_digest(SOURCES)
    hashes = {
        "source_sha256": oracle.sha(SOURCE),
        "gpu_driver_sha256": oracle.sha(Path(__file__)),
        "cpu_oracle_sha256": oracle.sha(oracle.SOURCE),
        "cpu_driver_sha256": oracle.sha(Path(oracle.__file__)),
        "shared_driver_sha256": oracle.sha(Path(micro.__file__)),
    }
    if a.phase == "plan":
        print(json.dumps({
            "schema": "vtp15-original-bf16-GDN-physical-tp3-GEMV-v1",
            "phase": "card_free",
            "source_digest": digest,
            "source_hashes": hashes,
            "source_receipts": matched,
            "not_native_optimized_gemm": True,
            "not_full_model": True,
        }, indent=2))
        return 0
    hipcc = micro.compiler_path(a.hipcc)
    if hipcc is None:
        p.error("HIP compiler missing")
    scratch = a.scratch.resolve()
    if scratch == ROOT or ROOT in scratch.parents:
        p.error("output must be outside Git checkout")
    binaries = micro.build_binaries(hipcc, scratch, digest, sources=SOURCES)
    if a.phase == "compile":
        print(json.dumps({
            "status": "card_free_compiled",
            "source_digest": digest,
            "binary": str(binaries["original_gdn_proj"]),
        }, indent=2))
        return 0
    if not a._under_lease:
        child = [
            sys.executable, str(Path(__file__).resolve()), "--phase", "gpu",
            "--run", "--_under-lease",
            "--weights-receipt", str(a.weights_receipt),
            "--reference-receipt", str(a.reference_receipt),
            "--iterations", str(a.iterations), "--scratch", str(scratch),
            "--hipcc", str(hipcc),
        ]
        return subprocess.run([
            "agent-lease", "acquire", "gpu", "--ttl", "30m",
            "--purpose", LEASE_PURPOSE, "--", "env", "VTP04_LEASED=1", *child,
        ], check=False).returncode
    case = {
        "kind": "original_gdn_proj",
        "label": f"original-BF16-GDN-all-input-projections-m{matched['rows']}",
        "args": [
            "--rows", str(matched["rows"]), "--iterations", str(a.iterations),
            "--pack-dir", matched["pack_directory"],
            "--qkv-dir", matched["qkv_directory"],
            "--reference-dir", matched["reference_directory"],
        ],
        "data_definition": "three physical GPUs concurrently perform actual"
                           " checkpoint BF16 QKV/A/B/Z projection GEMVs;"
                           " independent double accumulation CPU oracle, FP32"
                           " and BF16 checks, same weighted RMSNorm input;"
                           " not recurrence fusion or full model serving",
    }
    out = micro.run_leased(
        scratch=scratch, binaries=binaries, cases=[case], digest=digest,
        purpose=LEASE_PURPOSE, receipt_type=RECEIPT_TYPE,
    )
    rpath = out / "summary.json"
    record = json.loads(rpath.read_text())
    record.update({
        "schema": "vtp15-original-bf16-GDN-physical-tp3-GEMV-v1",
        "sources": matched, "source_hashes": hashes,
        "real_checkpoint_payload": True, "not_native_optimized_gemm": True,
        "not_full_model": True, "recurrence_attached": False,
    })
    rpath.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": "original_BF16_three_GPU_projection_oracle_pass",
                      "receipt": str(rpath)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
