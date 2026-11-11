"""VTP15: real BF16 Qwen GDN input GEMMs -> learned core, one GPU chain.

This opt-in checkpoint operator experiment launches the ORIGINAL layer-0
BF16 QKV/a/b/z projection GEMMs, convolution, Q/K normalization, delta
recurrence and gated RMSNorm concurrently on two gfx1100 XTX GPUs and one
gfx1201 R9700 GPU. Validates every projection versus independent
double-accumulation CPU reference and each resulting recurrent output
versus independently computed global/sharded original BF16 Qwen core.

The input hidden stream is synthetic, though input RMSNorm uses original
checkpoint weights. No output projection, layer residual, KV admission,
prefill-KKT, speculative rollback or full model generation is claimed.
The simple GEMV kernel is not native optimized Radiance dispatch.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import xtx27_vtp15_projection_gpu as projection
import xtx27_vtp15_projection_ref as ref
import xtx27_vtp_micro as micro

ROOT = micro.ROOT
SOURCE = Path(__file__).with_name("xtx27_vtp15_gdn_integrated.hip")
SOURCES = {"gdn_integrated": SOURCE}
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp15-integrated"
PURPOSE = "VTP15: original BF16 GDN projection-to-learned-core three-GPU chain"
RECEIPT_TYPE = "VTP15_ORIGINAL_BF16_GDN_PROJECTION_TO_RECURRENCE_GPU_CHAIN"


def validate(weights_path: Path, refs_path: Path, core_path: Path,
             learned_path: Path) -> dict:
    projections = projection.validate(weights_path, refs_path)
    core = json.loads(core_path.read_text())
    learned = json.loads(learned_path.read_text())
    if (core.get("schema") != "vtp15-qwen38-27b-original-bf16-GDN-core-CPU-global-shard-v1"
            or core.get("status") != "ORIGINAL_BF16_QWEN_GDN_CORE_CPU_GLOBAL_SHARD_PASS_NOT_GPU"):
        raise ValueError("need completed original checkpoint GDN core CPU oracle")
    if (core.get("weight_pack_sha256") != ref.sha(weights_path)
            or core.get("projected_input_ref_sha256") != ref.sha(refs_path)
            or core.get("rows") != projections["rows"]):
        raise ValueError("the original BF16 core/reference input receipts diverged")
    weights = ref.validate(weights_path)
    prior = Path(weights["qkv_pack_receipt"])
    if core.get("conv_pack_sha256") != ref.sha(prior):
        raise ValueError("GDN conv pack differs from core oracle")
    if (learned.get("status") != "CARD_FREE_QWEN_GDN_MATH_AND_SHARD_CONTINUITY_PASS_NOT_GPU"
            or learned.get("layer") != core.get("layer")
            or learned.get("parameter_hashes") != core.get("parameter_hashes")):
        raise ValueError("original learned Qwen GDN coefficients differ from reference")
    if not isinstance(learned.get("reference_coefficients"), dict):
        raise TypeError("original learned parameter packet missing")
    coeff = learned["reference_coefficients"]
    if ref.sha(Path(coeff["file"])) != coeff["sha256"] or Path(coeff["file"]).stat().st_size != 224 * 4:
        raise ValueError("original BF16 learned coefficient source changed")
    for role, heads in zip(("xtx0", "xtx1", "r9700"), (12, 12, 24), strict=True):
        item = core["rank_outputs"][role]
        output = Path(item["output"])
        if (item["shape"] != [heads, projections["rows"], 128]
                or output.stat().st_size != item["bytes"]
                or output.stat().st_size != heads * projections["rows"] * 128 * 4
                or ref.sha(output) != item["sha256"]):
            raise ValueError(f"{role} independent original GDN core output changed")
    if core.get("writer_sha256") != ref.sha(Path(__file__).with_name("xtx27_vtp15_gdn_bridge.py")):
        raise ValueError("independent original GDN core reference writer changed")
    return {
        **projections,
        "core_receipt": str(core_path.resolve()),
        "core_receipt_sha256": ref.sha(core_path),
        "core_directory": str(core_path.resolve().parent),
        "learned_receipt": str(learned_path.resolve()),
        "learned_receipt_sha256": ref.sha(learned_path),
        "coefficients": str(Path(coeff["file"]).resolve()),
        "real_QKV_A_B_Z_BF16_weights": True,
        "real_GDN_conv_decay_norm_BF16": True,
        "input_hidden_is_synthetic": True,
        "complete_model_layer": False,
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
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    p.add_argument("--hipcc")
    a = p.parse_args(argv)
    if a.phase != "plan" and not a.run:
        p.error("compile/GPU needs explicit --run")
    if a._under_lease and os.environ.get("VTP04_LEASED") != "1":
        p.error("GPU child requires an exclusive agent-lease")
    if not 1 <= a.iterations <= 30:
        p.error("iterations must be 1..30")
    receipts = (a.weights_receipt, a.projection_receipt, a.core_receipt, a.learned_receipt)
    if a.phase == "gpu" and any(x is None for x in receipts):
        p.error("GPU needs all four matching, immutable reference receipts")
    try:
        matched = validate(*receipts) if all(x is not None for x in receipts) else None
    except (KeyError, ValueError, TypeError, FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        p.error(str(exc))
    digest = micro.source_digest(SOURCES)
    hashes = {
        "source_sha256": ref.sha(SOURCE),
        "gpu_driver_sha256": ref.sha(Path(__file__)),
        "shared_driver_sha256": ref.sha(Path(micro.__file__)),
        "core_oracle_writer_sha256": ref.sha(Path(__file__).with_name("xtx27_vtp15_gdn_bridge.py")),
    }
    if a.phase == "plan":
        print(json.dumps({
            "schema": "vtp15-integrated-original-BF16-GDN-core-GPU-chain-v1",
            "phase": "card_free", "source_digest": digest,
            "hashes": hashes, "matched_sources": matched,
            "not_full_model": True,
            "not_optimized_gemm": True,
        }, indent=2))
        return 0
    hipcc = micro.compiler_path(a.hipcc)
    if hipcc is None:
        p.error("HIP compiler not available")
    scratch = a.scratch.resolve()
    if scratch == ROOT or ROOT in scratch.parents:
        p.error("scratch must be outside Git")
    binary = micro.build_binaries(hipcc, scratch, digest, sources=SOURCES)
    if a.phase == "compile":
        print(json.dumps({"status": "card_free_compiled",
                          "source_digest": digest,
                          "binary": str(binary["gdn_integrated"])}, indent=2))
        return 0
    if not a._under_lease:
        command = [
            sys.executable, str(Path(__file__).resolve()), "--phase", "gpu",
            "--run", "--_under-lease",
            "--weights-receipt", str(a.weights_receipt),
            "--projection-receipt", str(a.projection_receipt),
            "--core-receipt", str(a.core_receipt),
            "--learned-receipt", str(a.learned_receipt),
            "--iterations", str(a.iterations), "--scratch", str(scratch),
            "--hipcc", str(hipcc),
        ]
        return subprocess.run([
            "agent-lease", "acquire", "gpu", "--ttl", "30m",
            "--purpose", PURPOSE, "--", "env", "VTP04_LEASED=1", *command,
        ], check=False).returncode
    case = {
        "kind": "gdn_integrated",
        "label": f"original-BF16-GDN-projections-to-core-m{matched['rows']}",
        "args": [
            "--rows", str(matched["rows"]), "--iterations", str(a.iterations),
            "--conv-dir", matched["qkv_directory"],
            "--pack-dir", matched["pack_directory"],
            "--reference-dir", matched["reference_directory"],
            "--core-dir", matched["core_directory"],
            "--coefficients", matched["coefficients"],
        ],
        "data_definition": "real original BF16 QKV/a/b/z GEMMs -> original"
                           " convolution -> learned Qwen gated-delta recurrent core;"
                           " 3 physical streams concurrent, per-rank independent"
                           " BF16 double-accumulation GEMM + global/rank GDN"
                           " oracle; no output projection or model inference",
    }
    out = micro.run_leased(
        scratch=scratch, binaries=binary, cases=[case], digest=digest,
        purpose=PURPOSE, receipt_type=RECEIPT_TYPE,
    )
    path = out / "summary.json"
    r = json.loads(path.read_text())
    r.update({
        "schema": "vtp15-integrated-original-BF16-GDN-core-GPU-chain-v1",
        "sources": matched, "code_hashes": hashes,
        "full_model_layer": False, "native_optimized_gemm": False,
        "real_checkpoint_weights": True,
    })
    path.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({
        "status": "integrated_GDN_projections_recurrence_3GPU_oracle_pass",
        "receipt": str(path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
