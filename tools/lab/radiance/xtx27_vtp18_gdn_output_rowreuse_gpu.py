"""VTP18: opt-in batched-row weight-reuse GEMV plus pinned-async original BF16 GDN output.

The original layer-0 output projection remains column-sharded 1536/1536/3072
on two gfx1100 and one gfx1201 GPUs. Each concurrently computes one
partial [M,5120] F32 vector from the original BF16 weights and the
separate independently generated VTP15 GDN-core input. Three GPU D2H
transfers, an explicitly ordered host F32 reduction, BF16 residual add,
and H2D broadcast use portable pinned buffers and batched asynchronous\nD2H/H2D transfer enqueues (as distinct from pageable baseline).\nThe result is verified against the independent unsharded C++
CPU oracle. This is a separate output-stage proof: it is NOT yet joined
to the VTP15 integrated GDN on-device execution or the engine.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import xtx27_vtp16_gdn_output_oracle as oracle
import xtx27_vtp_micro as micro

SOURCE = Path(__file__).with_name("xtx27_vtp18_gdn_output_rowreuse.hip")
SOURCES = {"gdn_output_rowreuse": SOURCE}
ROOT = micro.ROOT
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp18-gdn-output-rowreuse"
LEASE_PURPOSE = "VTP18: original BF16 GDN row-reuse plus pinned-async TP3 A/B"
RECEIPT_TYPE = "VTP18_ORIGINAL_BF16_GDN_OUT_PROJ_ROW_REUSE_PINNED_ASYNC_TP3"


def validate(packet_path: Path, reference_path: Path) -> dict:
    packet = oracle.validate(packet_path)
    ref = json.loads(reference_path.read_text())
    if (ref.get("schema") != oracle.SCHEMA
            or ref.get("status") != "ORIGINAL_BF16_GDN_OUTPUT_PARTIAL_F32_REDUCTION_RESIDUAL_CPU_PASS"
            or ref.get("original_packet_sha256") != oracle.sha(packet_path)
            or ref.get("writer_sha256") != oracle.sha(Path(oracle.__file__))
            or ref.get("cpu_oracle_sha256") != oracle.sha(oracle.SOURCE)
            or ref.get("rows") != packet["rows"]
            or ref.get("layer") != packet["layer"]):
        raise ValueError("original output projection CPU oracle provenance disagrees")
    for name, item in ref["outputs"].items():
        ext = name.rsplit(".", 1)[-1]
        width = 2 if ext == "bf16" else 4
        actual = Path(item["file"])
        if (actual.stat().st_size != packet["rows"] * 5120 * width
                or item["shape"] != [packet["rows"], 5120]
                or item["bytes"] != actual.stat().st_size
                or oracle.sha(actual) != item["sha256"]):
            raise ValueError(f"original BF16 output control file {name} changed")
    return {
        "packet_file": str(packet_path.resolve()),
        "packet_sha256": oracle.sha(packet_path),
        "packet_directory": str(packet_path.resolve().parent),
        "reference_file": str(reference_path.resolve()),
        "reference_sha256": oracle.sha(reference_path),
        "reference_directory": str(reference_path.resolve().parent),
        "residual_file": packet["residual"]["file"],
        "rows": packet["rows"],
        "layer": packet["layer"],
        "not_model_serving": True,
        "not_optimized_gemm": True,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("plan", "compile", "gpu"), default="plan")
    p.add_argument("--run", action="store_true")
    p.add_argument("--_under-lease", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--packet-receipt", type=Path)
    p.add_argument("--reference-receipt", type=Path)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--scratch", type=Path, default=SCRATCH)
    p.add_argument("--hipcc")
    a = p.parse_args(argv)
    if a.phase != "plan" and not a.run:
        p.error("compile/GPU requires explicit --run")
    if a._under_lease and os.environ.get("VTP04_LEASED") != "1":
        p.error("internal child requires exclusive agent-lease wrapper")
    if not 1 <= a.iterations <= 30:
        p.error("iterations must be bounded 1..30")
    if a.phase == "gpu" and not (a.packet_receipt and a.reference_receipt):
        p.error("original output packet and CPU-oracle receipts are required for GPU")
    try:
        entry = (validate(a.packet_receipt, a.reference_receipt)
                 if a.packet_receipt and a.reference_receipt else None)
    except (ValueError, FileNotFoundError, KeyError, TypeError, OSError) as e:
        p.error(str(e))
    digest = micro.source_digest(SOURCES)
    hashes = {
        "source_sha256": oracle.sha(SOURCE),
        "driver_sha256": oracle.sha(Path(__file__)),
        "reference_cpp_sha256": oracle.sha(oracle.SOURCE),
        "reference_writer_sha256": oracle.sha(Path(oracle.__file__)),
        "shared_driver_sha256": oracle.sha(Path(micro.__file__)),
    }
    if a.phase == "plan":
        print(json.dumps({
            "schema": "vtp18-original-BF16-GDN-output-row-reuse-pinned-async-3GPU-v1",
            "phase": "card_free", "source_digest": digest,
            "code_hashes": hashes, "sources": entry,
            "not_integrated_with_gdn_core": True,
            "not_full_model": True,
        }, indent=2))
        return 0
    compiler = micro.compiler_path(a.hipcc)
    if compiler is None:
        p.error("missing HIP compiler")
    scratch = a.scratch.resolve()
    if scratch == ROOT or ROOT in scratch.parents:
        p.error("scratch output must be outside Git")
    binaries = micro.build_binaries(compiler, scratch, digest, sources=SOURCES)
    if a.phase == "compile":
        print(json.dumps({
            "status": "multiarch_compiled_card_free",
            "source_digest": digest,
            "binary": str(binaries["gdn_output_rowreuse"]),
        }, indent=2))
        return 0
    if not a._under_lease:
        child = [
            sys.executable, str(Path(__file__).resolve()), "--phase", "gpu",
            "--run", "--_under-lease", "--packet-receipt", str(a.packet_receipt),
            "--reference-receipt", str(a.reference_receipt),
            "--iterations", str(a.iterations), "--scratch", str(scratch),
            "--hipcc", str(compiler),
        ]
        return subprocess.run([
            "agent-lease", "acquire", "gpu", "--ttl", "30m",
            "--purpose", LEASE_PURPOSE, "--", "env", "VTP04_LEASED=1", *child,
        ], check=False).returncode
    case = {
        "kind": "gdn_output_rowreuse",
        "label": f"original-BF16-GDN-output-row-reuse-pinned-m{entry['rows']}",
        "args": [
            "--rows", str(entry["rows"]), "--iterations", str(a.iterations),
            "--pack-dir", entry["packet_directory"],
            "--reference-dir", entry["reference_directory"],
            "--residual", entry["residual_file"],
        ],
        "data_definition": "actual original BF16 5120x6144 column projection"
                           " across 3 physical GPUs using 1536/1536/3072 channels;"
                           " D2H F32 ordered host reduce, BF16 residual add and"
                           " H2D broadcast; independent unsharded CPU oracle;"
                           " M2/M4/M8 reused BF16 weight loads per wave across input rows; pinned async D2H/H2D, no P2P;"
                           " NOT full model",
    }
    out = micro.run_leased(
        scratch=scratch, binaries=binaries, cases=[case], digest=digest,
        purpose=LEASE_PURPOSE, receipt_type=RECEIPT_TYPE,
    )
    receipt_path = out / "summary.json"
    record = json.loads(receipt_path.read_text())
    record.update({
        "schema": "vtp18-original-BF16-GDN-output-row-reuse-pinned-async-3GPU-v1",
        "sources": entry, "code_hashes": hashes,
        "full_model": False, "integrated_gdn_chain": False,
        "native_optimized_dispatch": False, "p2p_required": False,
    })
    receipt_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({
        "status": "original_BF16_output_row_reuse_three_GPU_oracle_PASS",
        "receipt": str(receipt_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
