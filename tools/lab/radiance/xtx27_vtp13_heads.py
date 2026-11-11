"""VTP13 head-local R9700 2-KV-head attention and 8K/24V GDN proof.

An opt-in, leased micro-experiment reusing VTP04 source digests, builds,
strict GPU admission and immutable receipts. It verifies head-local
math with independent CPU references, not checkpoint projections or model
quality. GDN uses a simplified recurrent delta update: no conv/A_log/dt,
so it is NOT the complete Qwen GDN operator.
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
SOURCE = Path(__file__).with_name("xtx27_vtp13_heads.hip")
SOURCES = {"head_local": SOURCE}
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp13-head-ops"
LEASE_PURPOSE = "VTP13: head-local R9700 attention/GDN physical TP3"
RECEIPT_TYPE = "VTP13_HEAD_LOCAL_GQA_AND_SIMPLIFIED_GDN_NUMERICAL_PROOF"


def cases(operators: tuple[str, ...], rows: tuple[int, ...],
          iterations: int, context: int, reverse: bool = False) -> list[dict]:
    if not 1 <= iterations <= 30 or not 1 <= context <= 2048:
        raise ValueError("iterations 1..30 and context 1..2048 required")
    if any(op not in ("attention", "gdn") for op in operators):
        raise ValueError("operator must be attention or gdn")
    if any(m < 1 or m > 64 for m in rows):
        raise ValueError("rows must be within 1..64")
    return [{
        "kind": "head_local",
        "label": f"{op}-{m}rows-{context}context" + ("-reverse" if reverse else ""),
        "args": [
            "--rows", str(m), "--context", str(context),
            "--iterations", str(iterations),
            *(["--gdn"] if op == "gdn" else []),
            *(["--reverse"] if reverse else []),
        ],
        "data_definition": (
            "Original 24Q/4KV 256-dimensional full-attention GQA;"
            " R9700 12Q/2KV in one launch; independent CPU softmax oracle"
            if op == "attention" else
            "Head-local 16K/48V GDN-shaped persistent 128x128 state;"
            " R9700 8K/24V in one launch; synthetic simplified recurrent update"
            " without convolution, A_log or dt_bias; independent CPU oracle"
        ),
    } for op in operators for m in rows]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("plan", "compile", "gpu"), default="plan")
    p.add_argument("--run", action="store_true")
    p.add_argument("--_under-lease", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--operator", choices=("attention", "gdn", "both"), default="both")
    p.add_argument("--rows", default="1,4")
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--reverse-order", action="store_true")
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    p.add_argument("--hipcc")
    a = p.parse_args(argv)
    if a.phase != "plan" and not a.run:
        p.error("compile/GPU requires explicit --run")
    if a._under_lease and os.environ.get("VTP04_LEASED") != "1":
        p.error("internal GPU phase requires agent-lease wrapper")
    try:
        rows = micro.csv_positive(a.rows, upper=64, name="rows")
        ops = ("attention", "gdn") if a.operator == "both" else (a.operator,)
        work = cases(ops, rows, a.iterations, a.context, a.reverse_order)
    except ValueError as exc:
        p.error(str(exc))
    digest = micro.source_digest(SOURCES)
    hashes = {
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_driver_sha256": hashlib.sha256(Path(micro.__file__).read_bytes()).hexdigest(),
    }
    if a.phase == "plan":
        print(json.dumps({
            "phase": "card_free",
            "schema": "vtp13-head-local-proof-v1",
            "not_full_model": True,
            "checkpoint_payloads": False,
            "gdn_is_simplified": True,
            "source_digest": digest, "driver_hashes": hashes,
            "cases": work,
        }, indent=2))
        return 0
    hipcc = micro.compiler_path(a.hipcc)
    if hipcc is None:
        p.error("HIP compiler not found")
    scratch = a.scratch.resolve()
    if scratch == ROOT or ROOT in scratch.parents:
        p.error("scratch cannot be in Git checkout")
    binaries = micro.build_binaries(hipcc, scratch, digest, sources=SOURCES)
    if a.phase == "compile":
        print(json.dumps({"status": "card_free_compiled",
                          "source_digest": digest,
                          "binary": str(binaries["head_local"])}, indent=2))
        return 0
    if not a._under_lease:
        child = [
            sys.executable, str(Path(__file__).resolve()), "--phase", "gpu",
            "--run", "--_under-lease", "--rows", a.rows,
            "--context", str(a.context), "--operator", a.operator,
            "--iterations", str(a.iterations), "--scratch", str(scratch),
            "--hipcc", str(hipcc),
            *(["--reverse-order"] if a.reverse_order else []),
        ]
        cmd = [
            "agent-lease", "acquire", "gpu", "--ttl", "30m",
            "--purpose", LEASE_PURPOSE, "--", "env", "VTP04_LEASED=1", *child,
        ]
        return subprocess.run(cmd, check=False).returncode
    out = micro.run_leased(
        scratch=scratch, binaries=binaries, cases=work, digest=digest,
        purpose=LEASE_PURPOSE, receipt_type=RECEIPT_TYPE,
    )
    receipt = out / "summary.json"
    contents = json.loads(receipt.read_text())
    contents.update({
        "schema": "vtp13-head-local-proof-v1",
        "driver_hashes": hashes,
        "checkpoint_payloads": False,
        "gdn_is_simplified": True,
        "not_full_model": True,
    })
    receipt.write_text(json.dumps(contents, indent=2) + "\n")
    print(json.dumps({"status": "completed_head_local_micro_not_model_qualified",
                      "receipt": str(receipt)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
