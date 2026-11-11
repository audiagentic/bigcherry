"""VTP21: original BF16 GDN decode-state continuation across token chunks.

Card-free. Consume matching original-checkpoint QKV/a/b/z BF16 input
projections, original 4-tap depthwise convolution, original A_log/dt_bias
and RMS gain, plus independent monolithic four-token CPU oracle.

For every physical TP3 rank and every value head, replay the same four
tokens in contiguous chunks 1+3, 2+2, 3+1, 1+1+1+1, preserving the
three incoming BF16 convolution rows and explicit [key_dim,value_dim]
recurrent state across boundaries. At each boundary serialise and
restore state to canonical little-endian F64; compare against independent
unpartitioned reference outputs. Check a deliberately reset state
diverges, demonstrating that the handoff is necessary.

This is an exact CPU contract for a future resident GPU state ABI.
It is NOT the native FP32 GPU continuation, prefill KKT,
speculative rollback, real model text generation or 262K context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from datetime import UTC, datetime
from pathlib import Path

import xtx27_vtp14_gdn_exact as exact
import xtx27_vtp15_gdn_bridge as bridge
import xtx27_vtp15_gdn_integrated_gpu as integrated
import xtx27_vtp15_projection_ref as projection

ROOT = exact.ROOT
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "vtp21-gdn-state-continuation"
SCHEMA = "vtp21-original-bf16-qwen-gdn-conv-recurrence-chunk-continuation-v1"
ROLE_SPECS = (("xtx0", 4, 12), ("xtx1", 4, 12), ("r9700", 8, 24))
PARTITIONS = ((4,), (1, 3), (2, 2), (3, 1), (1, 1, 1, 1))
F64_STATE_LEN = 128 * 128 * 8


def validate(weights: Path, projected: Path, core: Path, learned: Path) -> dict:
    record = integrated.validate(weights, projected, core, learned)
    if record["rows"] != 4:
        raise ValueError("continuation proof requires original BF16 four-token reference")
    return record


def cached_conv(
    source_rows: list[list[float]], weights: list[float],
    partition: tuple[int, ...],
) -> tuple[list[list[float]], list[str]]:
    """Causal width-4 conv from a bounded 3-row BF16 projection ring.

    This function cannot access the original full source outside
    its current chunk and the restored history.
    """
    rows = len(source_rows)
    if (sum(partition) != rows or min(partition) < 1
            or len({len(row) for row in source_rows}) != 1
            or len(weights) != len(source_rows[0]) * 4):
        raise ValueError("chunk-convolution geometry invalid")
    channels = len(source_rows[0])
    history: list[list[float]] = []
    out: list[list[float]] = []
    snapshots = []
    offset = 0
    for count in partition:
        segment = source_rows[offset:offset + count]
        for row in segment:
            window = [*history, row]
            base = len(window) - 1
            conv = []
            for c in range(channels):
                acc = 0.0
                for tap in range(4):
                    at = base + tap - 3
                    if at >= 0:
                        acc += window[at][c] * weights[c * 4 + tap]
                conv.append(exact.round_bf16(exact.silu(acc)))
            out.append(conv)
            # Ring contains at most 3 *original BF16* projected rows.
            history.append(row)
            history = history[-3:]
        offset += count
        packed = struct.pack(f"<{len(history) * channels}f",
                             *(x for entry in history for x in entry))
        snapshots.append(hashlib.sha256(packed).hexdigest())
        # Actually restore the convolution history, not just keep
        # the references to previous full input rows.
        restored = struct.unpack(f"<{len(history) * channels}f", packed)
        history = [list(restored[i * channels:(i + 1) * channels])
                   for i in range(len(history))]
    if offset != rows:
        raise ValueError("chunk-sequence input not fully processed")
    return out, snapshots


def f64_checkpoint(state: list[list[float]], *, tamper: bool = False) -> tuple[list[list[float]], str]:
    if len(state) != 128 or any(len(row) != 128 for row in state):
        raise ValueError("GDN key x value state shape mismatch")
    values = [value for row in state for value in row]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite recurrent state cannot be checkpointed")
    blob = struct.pack("<16384d", *values)
    if len(blob) != F64_STATE_LEN:
        raise ValueError("wrong GDN checkpoint byte extent")
    if tamper:
        blob = blob[:-8]
    if len(blob) != F64_STATE_LEN:
        raise ValueError("corrupted GDN checkpoint byte extent")
    restored = struct.unpack("<16384d", blob)
    return [list(restored[d * 128:(d + 1) * 128]) for d in range(128)], hashlib.sha256(blob).hexdigest()


def replay_rank(
    role: str, nk: int, nv: int, projections: dict,
    conv_weights: list[float], params: dict,
    core_outputs: list[float], partition: tuple[int, ...],
) -> dict:
    rows = 4
    qkv = bridge._projection_values(projections, role, "qkv", rows, (2 * nk + nv) * 128)
    a = bridge._projection_values(projections, role, "a", rows, nv)
    b = bridge._projection_values(projections, role, "b", rows, nv)
    z = bridge._projection_values(projections, role, "z", rows, nv * 128)
    conv, conv_hash = cached_conv(qkv, conv_weights, partition)
    # Independent baseline recomputes each token using all source rows
    # and the original checkpoint conv tensor; the chunk path only
    # uses its rolling history.
    max_conv_error = 0.0
    for t in range(rows):
        for c in range(len(qkv[t])):
            expected = exact._bf16_conv(
                conv_weights, c, t,
                synthetic=lambda tt, cc: qkv[tt][cc],
            )
            max_conv_error = max(max_conv_error, abs(conv[t][c] - expected))
    if max_conv_error != 0.0:
        raise ValueError(f"{role}: conv continuity differs from unchunked original BF16")
    max_output_error = 0.0
    final_hashes = []
    max_state_difference_if_reset = 0.0
    for h in range(nv):
        key = h // 3
        q, k, v = exact.head_local_slice(conv, nk, nv, key, h)
        state = [[0.0] * 128 for _ in range(128)]
        offset = 0
        for count in partition:
            for t in range(offset, offset + count):
                global_h = {"xtx0": 0, "xtx1": 12, "r9700": 24}[role] + h
                result = exact.step_local(
                    state, q[t], k[t], v[t],
                    a[t][h], b[t][h], z[t][h * 128:(h + 1) * 128],
                    params["A_log"][global_h], params["dt_bias"][global_h],
                    params["norm"],
                )
                if t == rows - 1:
                    # Prove stale/reset cached state changes the *actual
                    # output*, not merely that the prior state was nonzero.
                    cold_result = exact.step_local(
                        [[0.0] * 128 for _ in range(128)],
                        q[t], k[t], v[t],
                        a[t][h], b[t][h], z[t][h * 128:(h + 1) * 128],
                        params["A_log"][global_h], params["dt_bias"][global_h],
                        params["norm"],
                    )
                    max_state_difference_if_reset = max(
                        max_state_difference_if_reset,
                        max(abs(a - b) for a, b in zip(
                            result, cold_result, strict=True
                        )),
                    )
                ref = core_outputs[(h * rows + t) * 128:(h * rows + t + 1) * 128]
                max_output_error = max(max_output_error, max(
                    abs(x - y) for x, y in zip(result, ref, strict=True)
                ))
            offset += count
            state, digest = f64_checkpoint(state)
        final_hashes.append(digest)
        # The reset negative-control output was evaluated on the same
        # fourth-token input above, versus the restored state.
    if max_output_error > 0.000002:
        raise ValueError(f"{role}: streamed GDN differs from monolithic CPU oracle")
    if max_state_difference_if_reset < 1e-10:
        raise ValueError(f"{role}: recurrent state unexpectedly zero after continuation")
    return {
        "role": role, "rows": rows, "keys": nk, "values": nv,
        "partition": list(partition),
        "conv_history_sha256": conv_hash,
        "final_state_sha256": hashlib.sha256(
            "".join(final_hashes).encode("ascii")
        ).hexdigest(),
        "max_conv_abs": max_conv_error,
        "max_original_cpu_oracle_output_abs": max_output_error,
        "last_token_reset_output_difference_max_abs": max_state_difference_if_reset,
    }


def verify(weights: Path, projected: Path, core: Path, learned: Path) -> dict:
    validated = validate(weights, projected, core, learned)
    refs = json.loads(projected.read_text())
    oracle = json.loads(core.read_text())
    w = projection.validate(weights)
    pack = json.loads(Path(w["qkv_pack_receipt"]).read_text())
    if pack["status"] != "EXACT_ORIGINAL_BF16_GDN_QKV_CONV_PACK_PASS_NOT_RUNTIME":
        raise ValueError("need original checkpoint BF16 conv source pack")
    params, hashes = exact.original_parameters(Path(w["checkpoint"]), w["layer"])
    if hashes != oracle["parameter_hashes"]:
        raise ValueError("original checkpoint decay/normalisation parameters changed")
    conv_rank_weights = {}
    oracle_outputs = {}
    for role, _, _ in ROLE_SPECS:
        entry = pack["outputs"]["conv1d"]["ranks"][role]
        raw = Path(entry["path"]).read_bytes()
        if (hashlib.sha256(raw).hexdigest() != entry["sha256"]
                or len(raw) != entry["bytes"]):
            raise ValueError(f"{role}: original BF16 conv weights mismatch")
        conv_rank_weights[role] = exact.bf16_values(raw)
        source = oracle["rank_outputs"][role]
        blob = Path(source["output"]).read_bytes()
        if (hashlib.sha256(blob).hexdigest() != source["sha256"]
                or len(blob) != source["bytes"]):
            raise ValueError(f"{role}: monolithic original CPU oracle output changed")
        oracle_outputs[role] = struct.unpack(f"<{len(blob) // 4}f", blob)
    data = []
    for partition in PARTITIONS:
        for role, nk, nv in ROLE_SPECS:
            data.append(replay_rank(
                role, nk, nv, refs, conv_rank_weights[role],
                params, oracle_outputs[role], partition,
            ))
    by_role = {}
    for role, _, _ in ROLE_SPECS:
        entries = [entry for entry in data if entry["role"] == role]
        baselines = entries[0]["final_state_sha256"]
        if any(e["final_state_sha256"] != baselines for e in entries):
            raise ValueError(f"{role}: serialized recurrent state depends on partitioning")
        conv_final = entries[0]["conv_history_sha256"][-1]
        if any(e["conv_history_sha256"][-1] != conv_final for e in entries):
            raise ValueError(f"{role}: final convolution history depends on partitioning")
        if any(e["last_token_reset_output_difference_max_abs"] < 1e-10
               for e in entries):
            raise ValueError(f"{role}: reset-state negative control did not diverge")
        by_role[role] = {
            "final_state_sha256": baselines,
            "final_conv_history_sha256": entries[0]["conv_history_sha256"][-1],
            "max_gdn_output_abs": max(e["max_original_cpu_oracle_output_abs"] for e in entries),
        }
    return {
        "schema": SCHEMA,
        "status": "VTP21_ORIGINAL_BF16_GDN_PARTITION_CONTINUATION_CPU_PASS_NOT_GPU",
        "rows": 4, "layer": 0,
        "input_sources": {
            "projection_receipt_sha256": projection.sha(projected),
            "core_receipt_sha256": projection.sha(core),
            "weight_pack_receipt_sha256": projection.sha(weights),
            "learned_receipt_sha256": projection.sha(learned),
            "source_checkpoint": str(Path(w["checkpoint"]).resolve()),
            "validated_input": validated,
        },
        "partitions": [list(p) for p in PARTITIONS],
        "rank_results": by_role, "per_partition_checks": data,
        "guarantees": [
            "original BF16 checkpoint projection+conv+learned GDN CPU",
            "three original TP3 rank head allocations 4/4/8 and 12/12/24",
            "exact BF16 causal three-row convolution history boundary",
            "serialised F64 CPU recurrence state [key_dim,value_dim] per head",
            "matches original M4 monolithic independent CPU oracle",
        ],
        "limitations": [
            "synthetic incoming hidden activations",
            "GPU FP32 recurrent checkpoint ABI and native precision remain open",
            "no full-model generation, attention, KKT prefill or long-context",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("plan", "verify"), default="plan")
    p.add_argument("--run", action="store_true")
    p.add_argument("--weights-receipt", type=Path)
    p.add_argument("--projection-receipt", type=Path)
    p.add_argument("--core-receipt", type=Path)
    p.add_argument("--learned-receipt", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    if a.phase == "verify" and not a.run:
        p.error("original BF16 checkpoint continuation verification needs explicit --run")
    receipt_paths = (a.weights_receipt, a.projection_receipt, a.core_receipt,
                     a.learned_receipt)
    if a.phase == "plan":
        print(json.dumps({"schema": SCHEMA, "phase": "card_free",
                          "partitions": [list(x) for x in PARTITIONS],
                          "physical_rank_ownership": ROLE_SPECS,
                          "gpu_required": False, "model_serving": False}, indent=2))
        return 0
    if any(p is None for p in receipt_paths):
        p.error("four validated original BF16 receipts required")
    dest = a.output.resolve() if a.output else SCRATCH / datetime.now(UTC).strftime(
        "layer0-%Y%m%dT%H%M%S%fZ"
    )
    if dest == ROOT or ROOT in dest.parents or dest.exists():
        p.error("must choose new outside-Git receipt directory")
    dest.mkdir(parents=True, exist_ok=False)
    record = dest / "summary.json"
    result = {"schema": SCHEMA, "status": "INCOMPLETE",
              "writer_sha256": projection.sha(Path(__file__))}
    record.write_text(json.dumps(result, indent=2) + "\n")
    try:
        result.update(verify(*receipt_paths))
    except Exception as exc:
        result.update({"status": "FAILED_INCOMPLETE", "error": str(exc)})
        record.write_text(json.dumps(result, indent=2) + "\n")
        raise
    record.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "receipt": str(record),
                      "rank_results": result["rank_results"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
