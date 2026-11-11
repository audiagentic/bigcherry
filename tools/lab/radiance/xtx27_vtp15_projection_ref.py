"""VTP15 original BF16 rank-local projection reference, entirely card-free.

Consumes an immutable VTP15 projection weight package and computes
real checkpoint QKV/a/b/z GEMVs with a source-hashed standalone C++
double-accumulation oracle. Emits both BF16-rounded projection tensors
and pre-round FP32 outputs. The GPU projection path checks these outputs,
then the complete GDN path can consume the same preprojected values.
This is one layer's ingress, not yet the model/engine execution path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import xtx27_vtp15_gdn_projection as projection

ROOT = projection.ROOT
SOURCE = Path(__file__).with_name("xtx27_vtp15_project_ref.cpp")
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp15-gdn-ref"
ROLES = ("xtx0", "xtx1", "r9700")
OPS = ("qkv", "a", "b", "z")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for buf in iter(lambda: src.read(4 * 1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def validate(weight_receipt: Path) -> dict:
    r = json.loads(weight_receipt.read_text())
    if r.get("schema") != projection.SCHEMA or r.get("status") != (
        "EXACT_ORIGINAL_BF16_GDN_INPUT_PROJECTION_PACK_PASS_NOT_GPU"
    ):
        raise ValueError("not a completed VTP15 original BF16 projection pack")
    if r.get("writer_sha256") != sha(projection.Path(projection.__file__)):
        raise ValueError("VTP15 projection pack writer has changed since source extraction")
    rows = r.get("rows")
    if type(rows) is not int or not 1 <= rows <= 8:
        raise ValueError("invalid BF16 input rows")
    input_file = r["inputs"]["normed_input.bf16"]
    src = Path(input_file["path"])
    if src.stat().st_size != rows * 5120 * 2 or sha(src) != input_file["sha256"]:
        raise ValueError("the original pre-RMSNorm BF16 input changed")
    for role in ROLES:
        for op in OPS:
            entry = r["rank_packs"][role][op]
            p = Path(entry["file"])
            shape = entry["shape"]
            if (not isinstance(shape, list) or len(shape) != 2 or shape[1] != 5120
                    or shape[0] <= 0 or p.stat().st_size != shape[0] * 5120 * 2
                    or sha(p) != entry["sha256"]):
                raise ValueError(f"{role}.{op}: changed original BF16 packed weights")
    return r


def compile_reference(scratch: Path) -> Path:
    digest = sha(SOURCE)
    build = scratch / "build" / digest[:16]
    exe = build / "vtp15_project_ref"
    if exe.exists():
        if sha(exe) != (build / "binary.sha256").read_text().strip():
            raise ValueError("cached CPU projection binary changed")
        return exe
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(["g++", "-O3", "-std=c++17", "-Wall", "-Wextra",
                    str(SOURCE), "-o", str(exe)], check=True)
    (build / "binary.sha256").write_text(sha(exe) + "\n")
    return exe


def run(weight_receipt: Path, output: Path, scratch: Path) -> dict:
    weight = validate(weight_receipt)
    out = output.expanduser().resolve()
    root = ROOT.resolve()
    if out.exists() or out == root or root in out.parents:
        raise ValueError("receipt must be a fresh directory outside Git")
    binfile = compile_reference(scratch.resolve())
    out.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "vtp15-original-bf16-GDN-projection-CPU-oracle-v1",
        "status": "INCOMPLETE",
        "created_utc": datetime.now(UTC).isoformat(),
        "weight_pack_receipt": str(weight_receipt.resolve()),
        "weight_pack_sha256": sha(weight_receipt),
        "writer_sha256": sha(Path(__file__)),
        "oracle_cpp_sha256": sha(SOURCE),
        "oracle_binary_sha256": sha(binfile),
        "rows": weight["rows"], "rank_results": {},
        "limitations": ["No GPU", "Not an engine GEMM", "Not real model generation"],
    }
    receipt_file = out / "summary.json"
    receipt_file.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        for role in ROLES:
            ops = {}
            for op in OPS:
                w = weight["rank_packs"][role][op]
                bf16 = out / f"{op}-{role}.bf16"
                fp32 = out / f"{op}-{role}.f32"
                cmd = [
                    str(binfile), "--m", str(weight["rows"]),
                    "--n", str(w["shape"][0]), "--k", "5120",
                    "--input", weight["inputs"]["normed_input.bf16"]["path"],
                    "--weights", w["file"],
                    "--out-bf16", str(bf16), "--out-f32", str(fp32),
                ]
                completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
                ops[op] = {
                    "bf16": str(bf16), "fp32": str(fp32),
                    "bf16_sha256": sha(bf16), "fp32_sha256": sha(fp32),
                    "bf16_bytes": bf16.stat().st_size, "fp32_bytes": fp32.stat().st_size,
                    "shape": [weight["rows"], w["shape"][0]],
                    "stdout": completed.stdout.strip(),
                }
            receipt["rank_results"][role] = ops
            receipt_file.write_text(json.dumps(receipt, indent=2) + "\n")
        receipt["status"] = "ORIGINAL_BF16_QWEN_GDN_INPUT_PROJECTIONS_CPU_PASS_NOT_GPU"
    except Exception as exc:
        receipt["status"] = "FAILED_INCOMPLETE"
        receipt["error"] = str(exc)
        receipt_file.write_text(json.dumps(receipt, indent=2) + "\n")
        raise
    receipt_file.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights-receipt", required=True, type=Path)
    p.add_argument("--phase", choices=("plan", "run"), default="plan")
    p.add_argument("--run", action="store_true")
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    if a.phase == "run" and not a.run:
        p.error("processing checkpoint weights requires explicit --run")
    try:
        w = validate(a.weights_receipt)
        if a.phase == "plan":
            print(json.dumps({
                "status": "VALIDATED_ORIGINAL_BF16_WEIGHT_PACKAGE_NO_GPU",
                "rows": w["rows"], "source_sha256": sha(SOURCE),
                "weight_pack_sha256": sha(a.weights_receipt),
                "one_worker_per_rank": True, "roles": list(ROLES),
                "projection_ops": list(OPS),
            }, indent=2))
            return 0
        out = a.output or (a.scratch / datetime.now(UTC).strftime("layer0-%Y%m%dT%H%M%S%fZ"))
        record = run(a.weights_receipt, out, a.scratch)
        print(json.dumps({"status": record["status"], "receipt": str(out / "summary.json")}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as e:
        p.error(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
