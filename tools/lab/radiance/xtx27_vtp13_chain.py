"""VTP13 opt-in model-shaped INT8 FFN -> real 3-GPU host collective.

The FFN dimensions are original Qwen3.8-27B; weights/activations here
are deterministic exactly-BF16-representable SYNTHETIC sources. This proves
simultaneous INT8 computation, SwiGLU, shared ABI, no-P2P transfer, and an
independent numerical oracle. It is NOT model checkpoint execution or an
optimized/native Radiance GEMM performance result. Defaults are card-free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import xtx27_vtp_micro as micro

ROOT = micro.ROOT
SOURCE = Path(__file__).with_name("xtx27_vtp13_ffn.hip")
SOURCES = {"ffn_chain": SOURCE}
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp13-ffn-chain"
LEASE_PURPOSE = "VTP13: concurrent 27B-shaped synthetic BF16 INT8 FFN TP3"
RECEIPT_TYPE = "VTP13_CONCURRENT_INT8_FFN_AND_PHYSICAL_TP3_COLLECTIVE"


def cases(rows: tuple[int, ...], iterations: int, reverse: bool = False) -> list[dict]:
    if not 1 <= iterations <= 30 or any(v > 64 for v in rows):
        raise ValueError("VTP13 bounded at 1..30 iterations and 1..64 rows")
    return [{
        "kind": "ffn_chain",
        "label": f"ffn-int8-synthetic-bf16-{m}rows" + ("-reverse" if reverse else ""),
        "args": ["--rows", str(m), "--iterations", str(iterations),
                 *(["--reverse"] if reverse else [])],
        "data_definition": "full asymmetric 5120/4864/7424 INT8 gate/up/SwiGLU/down,"
                           " all 3 physical GPUs in flight before barrier,"
                           " 3x D2H + CPU F32 sum + 3x H2D;"
                           " independent synthetic BF16-source CPU oracle",
    } for m in rows]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("plan", "compile", "gpu"), default="plan")
    p.add_argument("--run", action="store_true", help="mandatory for compilation or GPU")
    p.add_argument("--_under-lease", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--rows", default="1,4", help="cheap default; optional 8,64")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--reverse-order", action="store_true")
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    p.add_argument("--hipcc")
    a = p.parse_args(argv)
    if a.phase != "plan" and not a.run:
        p.error("--run required for compile and GPU")
    if a._under_lease and os.environ.get("VTP04_LEASED") != "1":
        p.error("internal VTP13 GPU subprocess requires lease wrapper")
    try:
        rows = micro.csv_positive(a.rows, upper=64, name="rows")
        work = cases(rows, a.iterations, a.reverse_order)
    except ValueError as exc:
        p.error(str(exc))
    digest = micro.source_digest(SOURCES)
    runner_hashes = {
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_runner_sha256": hashlib.sha256(Path(micro.__file__).read_bytes()).hexdigest(),
    }
    if a.phase == "plan":
        print(json.dumps({
            "schema": "vtp13-concurrent-ffn-proof-v1",
            "phase": "card_free",
            "precision": "int8-int8-int32-f32, uniform three-card",
            "checkpoint_payload": False,
            "bf16_source": "exactly-representable synthetic BF16 values",
            "not_native_radiance_gemm": True,
            "not_a_model_or_serving_benchmark": True,
            "note": "GPU compute stage and collective are genuine, but synthetic weights"
                    " and simple scalar-warp kernels cannot establish serving speed.",
            "source_digest": digest,
            "runner_hashes": runner_hashes,
            "cases": work,
        }, indent=2))
        return 0
    hipcc = micro.compiler_path(a.hipcc)
    if hipcc is None:
        p.error("HIP compiler not found")
    scratch = a.scratch.resolve()
    if scratch == ROOT or ROOT in scratch.parents:
        p.error("scratch must be outside git checkout")
    binaries = micro.build_binaries(hipcc, scratch, digest, sources=SOURCES)
    if a.phase == "compile":
        print(json.dumps({"status": "card_free_compiled",
                          "binary": str(binaries["ffn_chain"]),
                          "source_digest": digest, "runner_hashes": runner_hashes}, indent=2))
        return 0
    if not a._under_lease:
        child = [sys.executable, str(Path(__file__).resolve()), "--phase", "gpu",
                 "--run", "--_under-lease", "--rows", a.rows,
                 "--iterations", str(a.iterations), "--scratch", str(scratch),
                 "--hipcc", str(hipcc),
                 *(["--reverse-order"] if a.reverse_order else [])]
        cmd = ["agent-lease", "acquire", "gpu", "--ttl", "30m",
               "--purpose", LEASE_PURPOSE, "--",
               "env", "VTP04_LEASED=1", *child]
        return subprocess.run(cmd, check=False).returncode
    out = micro.run_leased(
        scratch=scratch, binaries=binaries, cases=work, digest=digest,
        purpose=LEASE_PURPOSE, receipt_type=RECEIPT_TYPE,
    )
    receipt_path = out / "summary.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update({
        "schema": "vtp13-concurrent-ffn-proof-v1",
        "runner_hashes": runner_hashes,
        "checkpoint_payload": False,
        "bf16_source": "synthetic_bf16_exact",
        "not_native_radiance_gemm": True,
        "not_a_model_or_serving_benchmark": True,
    })
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "completed_numerical_gpu_proof_not_model_serving",
                      "receipt": str(receipt_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
