"""VTP23: real original BF16 full-attention layer-3 physical-TP3 GPU gate.

Uses the exact original Qwen3.8-27B checkpoint Q/G,K,V,O BF16 weights
in 6/6/12 Q and 1/1/2 KV head partitions across XTX0/XTX1/R9700.
GPU computes model-shaped QKV, zero-centered per-head RMSNorm,
text-only 64/256 partial RoPE, causal grouped-query attention, gated
attended output, out_proj column shards, pinned-host 3-rank reduction,
BF16 residual and three-device broadcast in one connected sequence.

Must match the independently compiled, source-hashed VTP22 CPU
reference at each stage. No full model inference, resident KV cache,
production mixed quantization or native Radiance dispatch.
Defaults GPU free, mandatory --run + exclusive GPU lease on execution.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import xtx27_vtp22_attention_oracle as oracle
import xtx27_vtp22_attention_pack as pack
import xtx27_vtp_micro as micro

SOURCE = Path(__file__).with_name("xtx27_vtp23_attention_original.hip")
SOURCES = {"original_attention": SOURCE}
ROOT = micro.ROOT
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp23-original-attention-gpu"
SCHEMA = "vtp23-qwen38-original-BF16-full-attention-3physical-gpu-v1"
PURPOSE = "VTP23: original BF16 Qwen full-attention layer-3 three-physical-GPU oracle"
RECEIPT_TYPE = "VTP23_ORIGINAL_BF16_FULL_ATTENTION_CONNECTED_GPU_TP3"


def validate(packet_path: Path, cpu_path: Path) -> dict:
    source = oracle.validate(packet_path)
    cpu = json.loads(cpu_path.read_text())
    if (cpu.get("schema") != oracle.SCHEMA
            or cpu.get("status") != "ORIGINAL_BF16_FULL_ATTENTION_CAUSAL_GATED_CPU_ORACLE_PASS_NOT_GPU"
            or cpu.get("writer_sha256") != pack.projection.sha(Path(oracle.__file__))
            or cpu.get("cpu_source_sha256") != pack.projection.sha(oracle.SOURCE)
            or cpu.get("original_packet_sha256") != pack.projection.sha(packet_path)
            or cpu.get("rows") != source["rows"]
            or cpu.get("layer") != source["layer"]
            or source["layer"] != 3):
        raise ValueError("original full-attention CPU reference is incompatible")
    for name, ref in cpu["outputs"].items():
        f = Path(ref["file"])
        if (ref["bytes"] != f.stat().st_size
                or ref["bytes"] != ref["shape"][0] * ref["shape"][1] *
                (4 if name.startswith("partial-") or name.endswith(".f32") else 2)
                or pack.projection.sha(f) != ref["sha256"]):
            raise ValueError(f"modified or truncated original BF16 oracle tensor: {name}")
    for role in pack.ROLES:
        for kind in ("q-gate", "k-raw", "v", "q-rope", "k-rope",
                     "attended-gated", "partial"):
            if f"{kind}-{role}" not in cpu["outputs"]:
                raise ValueError(f"missing original checkpoint head-local oracle: {role}.{kind}")
    for kind in ("original-attention-reduced.f32", "original-attention-post-residual.bf16"):
        if kind not in cpu["outputs"]:
            raise ValueError(f"missing original checkpoint final oracle: {kind}")
    return {
        "layer": 3,
        "rows": source["rows"],
        "checkpoint_packet": str(packet_path.resolve()),
        "checkpoint_packet_sha256": pack.projection.sha(packet_path),
        "cpu_reference": str(cpu_path.resolve()),
        "cpu_reference_sha256": pack.projection.sha(cpu_path),
        "pack_directory": str(packet_path.resolve().parent),
        "reference_directory": str(cpu_path.resolve().parent),
        "original_BF16_payloads_verified": True,
        "mrope_position_mode": "text-only-sequential",
        "full_attention_gate": "per-head sigmoid",
        "query_heads": [6, 6, 12],
        "kv_heads": [1, 1, 2],
        "output_column_width": [1536, 1536, 3072],
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
        p.error("HIP build and GPU benchmark require explicit --run")
    if a._under_lease and os.environ.get("VTP04_LEASED") != "1":
        p.error("internal leased runner requires owned agent-lease wrapper")
    if not 1 <= a.iterations <= 30:
        p.error("iterations outside bounded 1..30")
    if a.phase == "gpu" and not (a.packet_receipt and a.cpu_receipt):
        p.error("original BF16 checkpoint and independent CPU receipts required")
    try:
        checked = (validate(a.packet_receipt, a.cpu_receipt)
                   if a.packet_receipt and a.cpu_receipt else None)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as e:
        p.error(str(e))
    digest = micro.source_digest(SOURCES)
    metadata = {
        "schema": SCHEMA, "source_digest": digest,
        "runner_sha256": pack.projection.sha(Path(__file__)),
        "source_sha256": pack.projection.sha(SOURCE),
        "checkpoint_reader_sha256": pack.projection.sha(Path(pack.__file__)),
        "cpu_writer_sha256": pack.projection.sha(Path(oracle.__file__)),
        "cpu_cpp_sha256": pack.projection.sha(oracle.SOURCE),
        "source_receipts": checked,
        "full_model_generation": False,
        "cache_continuation": False,
        "native_radiance": False,
        "quality_qualified": False,
    }
    if a.phase == "plan":
        print(json.dumps({**metadata, "phase": "card_free"}, indent=2))
        return 0
    compiler = micro.compiler_path(a.hipcc)
    if compiler is None:
        p.error("HIP compiler missing")
    scratch = a.scratch.resolve()
    if scratch == ROOT or ROOT in scratch.parents:
        p.error("scratch output must be outside Git checkout")
    binaries = micro.build_binaries(compiler, scratch, digest, sources=SOURCES)
    if a.phase == "compile":
        print(json.dumps({
            **metadata, "status": "three-physical-GPU-multiarch-compiled-no-GPU",
            "binary": str(binaries["original_attention"]),
        }, indent=2))
        return 0
    if not a._under_lease:
        child = [
            sys.executable, str(Path(__file__).resolve()), "--phase", "gpu",
            "--run", "--_under-lease",
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
        "kind": "original_attention",
        "label": f"original-BF16-full-attention-layer3-m{checked['rows']}",
        "args": [
            "--rows", str(checked["rows"]), "--iterations", str(a.iterations),
            "--pack-dir", checked["pack_directory"],
            "--reference-dir", checked["reference_directory"],
        ],
        "data_definition": (
            "ORIGINAL checkpoint BF16 full-attention layer3;"
            " one context per physical XTX0/XTX1/R9700 GPU;"
            " 6/6/12 query-gate heads, 1/1/2 KV heads, 64-dim text RoPE,"
            " zero-centred Q/K norms and causal softmax, sigmoid output gate,"
            " 1536/1536/3072 original O-weight columns, pinned D2H F32"
            " ordered host sum BF16 residual and H2D broadcast;"
            " original BF16 checkpoint from CPU-sha256 control, not model"
        ),
    }
    run_dir = micro.run_leased(
        scratch=scratch, binaries=binaries, cases=[case], digest=digest,
        purpose=PURPOSE, receipt_type=RECEIPT_TYPE,
    )
    receipt = run_dir / "summary.json"
    state = json.loads(receipt.read_text())
    state.update({**metadata, "gpu_full_attention_stage": True})
    receipt.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({
        "status": "ORIGINAL_BF16_FULL_ATTENTION_TP3_GPU_LAYER_ORACLE_PASS_NOT_MODEL",
        "receipt": str(receipt),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
