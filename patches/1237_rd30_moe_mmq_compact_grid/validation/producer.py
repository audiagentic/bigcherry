"""PRBE25 (RD30/1237): patch-local validation producer.

RD30-MOE-MMQ-COMPACT-GRID is a performance contract with a bit_identical
correctness obligation, scoped to gfx1100.

- correctness (``bit_identical``): test-backend-ops on the exact 256-expert
  MUL_MAT_ID shapes (Q4_K and Q8_0, 32 routed tokens so MMQ is taken) must
  produce byte-identical backend output, routing and reference digests on
  control and subject for every seed. Both arms carry the deterministic
  test-backend-ops evidence patches; they differ only by 1237.
  ``backend_reference`` (each arm within its own NMSE threshold) is recorded
  as a diagnostic.
- activation: llama-bench prefill with BIGCHERRY_PATCH_TRACE=1 must emit the
  compact-grid marker on the subject and not on the control.
- performance: standard-scaffold llama-bench pair (control = baseline,
  subject = baseline + 1237), 10 paired rounds each: pp512 on the MoE model
  is the positive lane, tg128 on the same model is the control lane (MoE
  decode never takes the compact path).
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

from bigcherry.patch import producer_support as support
from bigcherry.patch import validation_producer as vp  # type: ignore[import-not-found]

_CONTRACT_ID = "RD30-MOE-MMQ-COMPACT-GRID"
_ARCHITECTURE = "gfx1100"
_SUBJECT_PATCH = "1237_rd30_moe_mmq_compact_grid"
_EVIDENCE_PATCHES = (
    "1222_hi67_deterministic_test_backend_ops_seed",
    "1223_hi67_machine_readable_correctness_metrics",
    "1236_hi105_deterministic_mul_mat_id_ids",
)
_SEEDS = (1, 2, 3)
# 32 routed tokens is deliberately > MMVQ_MAX_BATCH_SIZE (8), so MMQ runs.
_SHAPES = (("q4_k-moe-prefill32", "Q4_K"), ("q8_0-moe-prefill32", "Q8_0"))
_MARKER_REGEX = r"BIGCHERRY_PATCH_HIT patch=1237_rd30 path=moe_mmq_compact_grid"
_MIN_PAIRED_ROUNDS = support.contract_paired_rounds(_CONTRACT_ID)

_CORRECTNESS_ARTIFACT = "rd30-correctness.json"
_PERFORMANCE_ARTIFACT = "rd30-performance.json"
_SUBJECT_TRACE_ARTIFACT = "rd30-subject-trace.log"
_CONTROL_TRACE_ARTIFACT = "rd30-control-trace.log"


def _fail(message: str) -> vp.ValidationProducerError:
    return vp.ValidationProducerError(f"rd30: {message}")


def _enum_id(names: Mapping[int, str], wanted: str, kind: str) -> int:
    for value, name in names.items():
        if name.upper() == wanted.upper():
            return int(value)
    raise _fail(f"{kind} enum {wanted!r} is absent from the materialized source")


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _scaffold_bench(ctx: vp.ProducerContext, role: str) -> Path:
    binary = ctx.validation_binaries.get(role, {}).get("llama-bench")
    if not isinstance(binary, Path) or not binary.is_file():
        raise _fail(f"standard scaffold {role} llama-bench binary is missing")
    return binary


def _lane_effect(outcome, *, workload: str, metric: str, role: str):
    execution = importlib.import_module("bigcherry.experiment.execution")
    if set(outcome.runs) != {workload}:
        raise _fail(f"{role} lane must produce exactly one {workload} lane; got {sorted(outcome.runs)!r}")
    run = outcome.runs[workload]
    if dict(run.stats).get("paired_rounds") != _MIN_PAIRED_ROUNDS:
        raise _fail(f"{role} lane has {run.stats.get('paired_rounds')!r} paired rounds; expected {_MIN_PAIRED_ROUNDS}")
    return execution.lane_effect_from_run(role, metric, run), run


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    experiment_contract = importlib.import_module("bigcherry.experiment.contract")
    experiment_execution = importlib.import_module("bigcherry.experiment.execution")
    ActivationEvidence = importlib.import_module("bigcherry.patch.activation").ActivationEvidence
    correctness_evidence = importlib.import_module("bigcherry.tuning.correctness_evidence")
    signature_mapping = importlib.import_module("bigcherry.tuning.signature_mapping")

    if ctx.fat_targets.targets != (_ARCHITECTURE,):
        raise _fail(f"{_CONTRACT_ID} is scoped to gfx1100 exactly; got {ctx.fat_targets.targets!r}")
    if ctx.model is None:
        raise _fail("the MoE contract model (--model) is required")
    model = ctx.model

    devices = [d for d in ctx.runtime.device_contexts(device_map=ctx.device_map) if d.architecture == _ARCHITECTURE]
    if len(devices) != 1:
        raise _fail(f"--device-map must select exactly one gfx1100 device; got {len(devices)}")
    device = devices[0]

    # ---- correctness: bit-identical 256-expert MUL_MAT_ID ----
    pair = ctx.runtime.build_pair(
        targets=(_ARCHITECTURE,),
        primary_target="test-backend-ops",
        common_extra_patches=_EVIDENCE_PATCHES,
        baseline_source="bigcherry",
        require_parity=True,
    )
    op_names = signature_mapping.load_ggml_op_names(pair.control_source)
    type_names = signature_mapping.load_ggml_type_names(pair.control_source)
    op_mul_mat_id = _enum_id(op_names, "MUL_MAT_ID", "ggml_op")
    type_f32 = _enum_id(type_names, "F32", "ggml_type")

    scratch_dir = ctx.workdir / "scratch" / "rd30-correctness"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    shapes: list[dict[str, object]] = []
    for name, weight_type in _SHAPES:
        signature = {
            "op": op_mul_mat_id,
            "flags": 0x0F,
            "ne0": [2048, 256, 256, 1],
            "ne1": [2048, 1, 32, 1],
            "ned": [256, 8, 32, 1],
            "n_expert": 256,
            "n_expert_used": 8,
            "src0_type": _enum_id(type_names, weight_type, "ggml_type"),
            "src1_type": type_f32,
            "dst_type": type_f32,
        }
        line, target_tensor, digest_tensor = signature_mapping.signature_to_mul_mat_id_test_file_line(
            signature, vendor_root=pair.control_source
        )
        test_file = scratch_dir / f"{name}.txt"
        test_file.write_text(line + "\n", encoding="utf-8")
        shapes.append(
            {
                "name": name,
                "weight_type": weight_type,
                "signature": signature,
                "test_file": test_file,
                "target_tensor": target_tensor,
                "digest_tensor": digest_tensor,
            }
        )

    device_env = dict(device.env_overrides)
    rows: list[dict[str, object]] = []
    for shape in shapes:
        for seed in _SEEDS:
            arms = {
                role: correctness_evidence.collect_native_seed_evidence(
                    binary,
                    test_file=shape["test_file"],
                    target_tensor=shape["target_tensor"],
                    digest_tensor=shape["digest_tensor"],
                    seed=seed,
                    env=device_env,
                    runner=subprocess.run,
                )
                for role, binary in (("control", pair.control_bin), ("subject", pair.subject_bin))
            }
            c, s = arms["control"], arms["subject"]

            def _within(e) -> bool:
                return (
                    e.native_execution_status == "ok"
                    and math.isfinite(e.e_n_nmse)
                    and math.isfinite(e.threshold_t)
                    and e.e_n_nmse <= e.threshold_t
                )

            bit_identical = (
                c.native_execution_status == "ok"
                and s.native_execution_status == "ok"
                and c.reference_digest == s.reference_digest
                and c.reference_output_digest is not None
                and c.reference_output_digest == s.reference_output_digest
                and c.native_output_digest is not None
                and c.native_output_digest == s.native_output_digest
                and c.output_nels is not None
                and c.output_nels == s.output_nels
            )
            rows.append(
                {
                    "shape": shape["name"],
                    "seed": seed,
                    "control_status": c.native_execution_status,
                    "subject_status": s.native_execution_status,
                    "ids_equal": c.reference_digest == s.reference_digest,
                    "reference_equal": c.reference_output_digest == s.reference_output_digest,
                    "output_equal": c.native_output_digest == s.native_output_digest,
                    "control_output_digest": c.native_output_digest,
                    "subject_output_digest": s.native_output_digest,
                    "control_nmse": _finite_or_none(c.e_n_nmse),
                    "subject_nmse": _finite_or_none(s.e_n_nmse),
                    "backend_reference_ok": _within(c) and _within(s),
                    "bit_identical": bit_identical,
                }
            )

    exact_failure = next((r for r in rows if not r["bit_identical"]), None)
    bit_identical_result = experiment_contract.CorrectnessResult(
        check="bit_identical",
        passed=exact_failure is None,
        detail=(
            f"{len(rows)} 256-expert MUL_MAT_ID (shape,seed) pairs produced byte-identical HIP output"
            if exact_failure is None
            else f"exact-output mismatch: {exact_failure!r}"
        ),
    )
    backend_failure = next((r for r in rows if not r["backend_reference_ok"]), None)
    backend_reference_result = experiment_contract.CorrectnessResult(
        check="backend_reference",
        passed=backend_failure is None,
        detail=(
            f"{len(rows)} rows within each arm's backend-reference threshold"
            if backend_failure is None
            else f"backend-reference failure: {backend_failure!r}"
        ),
    )
    ctx.runtime.write_artifact(
        name=_CORRECTNESS_ARTIFACT,
        payload={
            "schema_version": 1,
            "contract_id": _CONTRACT_ID,
            "check": "bit_identical",
            "passed": bit_identical_result.passed,
            "architecture": _ARCHITECTURE,
            "evidence_patches": list(_EVIDENCE_PATCHES),
            "seeds": list(_SEEDS),
            "shapes": [
                {k: v for k, v in shape.items() if k != "test_file"} for shape in shapes
            ],
            "build_identities": {r: dict(i) for r, i in pair.validation_build_identities.items()},
            "rows": rows,
        },
    )

    # ---- activation + performance on the standard scaffold pair ----
    bench_control = _scaffold_bench(ctx, "control")
    bench_subject = _scaffold_bench(ctx, "subject")
    subject_trace = ctx.runtime.run_trace_probe(
        binary=bench_subject, model=model, device=device, bench_prompt=512, bench_gen=0,
        log_context="rd30-trigger-subject",
    )
    control_trace = ctx.runtime.run_trace_probe(
        binary=bench_control, model=model, device=device, bench_prompt=512, bench_gen=0,
        log_context="rd30-trigger-control",
    )
    marker = re.compile(_MARKER_REGEX)
    subject_hit = marker.search(subject_trace) is not None
    control_hit = marker.search(control_trace) is not None
    trigger_hit = subject_hit and not control_hit
    activation = ActivationEvidence(
        status="executed" if trigger_hit else ("unobservable" if subject_hit else "not_executed"),
        mechanism="trace_marker",
        detail=f"marker {_MARKER_REGEX!r} subject_hit={subject_hit} control_hit={control_hit}",
    )
    subject_trace_ref = ctx.runtime.write_text_artifact(name=_SUBJECT_TRACE_ARTIFACT, text=subject_trace)
    control_trace_ref = ctx.runtime.write_text_artifact(name=_CONTROL_TRACE_ARTIFACT, text=control_trace)

    positive_outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=bench_control, subject_binary=bench_subject, model=model,
        workloads=("prefill",), pairs=_MIN_PAIRED_ROUNDS, log_context="rd30-positive-prefill", device=device,
    )
    control_outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=bench_control, subject_binary=bench_subject, model=model,
        workloads=("decode",), pairs=_MIN_PAIRED_ROUNDS, log_context="rd30-control-decode", device=device,
    )
    positive_effect, positive_run = _lane_effect(positive_outcome, workload="prefill", metric="pp512", role="positive")
    control_effect, control_run = _lane_effect(control_outcome, workload="decode", metric="tg128", role="control")

    performance_ref = ctx.runtime.write_artifact(
        name=_PERFORMANCE_ARTIFACT,
        payload={
            "passed": True,
            "metrics": support.performance_metrics(positive_effect, control_effect),
            "schema_version": 1,
            "contract_id": _CONTRACT_ID,
            "architecture": _ARCHITECTURE,
            "model": str(model),
            "build_identities": {r: dict(i) for r, i in ctx.validation_build_identities.items()},
            "positive": {"metric": "pp512", "effect": dataclasses.asdict(positive_effect),
                         "runs": list(positive_run.runs), "stats": dict(positive_run.stats)},
            "control": {"metric": "tg128", "effect": dataclasses.asdict(control_effect),
                        "runs": list(control_run.runs), "stats": dict(control_run.stats)},
            "trigger": {"subject_hit": subject_hit, "control_hit": control_hit},
        },
    )
    trigger_evidence = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id="rd30-prefill-subject", role="positive", positive_hit=trigger_hit
    )

    return vp.ProducerResult(
        correctness={
            "disposition": "passed" if bit_identical_result.passed else "failed",
            "mechanism": "rd30-mul-mat-id-bit-identical",
            "detail": bit_identical_result.detail,
        },
        validation_build_identities=ctx.validation_build_identities,
        activation_evidence=activation,
        performance_evidence={"artifact": {"path": performance_ref.path, "sha256": performance_ref.sha256}},
        trace_evidence={
            "positive": {"artifact": {"path": subject_trace_ref.path, "sha256": subject_trace_ref.sha256}},
            "negative": {"artifact": {"path": control_trace_ref.path, "sha256": control_trace_ref.sha256}},
        },
        check_results=(),
        lane_effects=(),
        contract_correctness_results=(bit_identical_result, backend_reference_result),
        promotion_lane_effects={_CONTRACT_ID: (positive_effect, control_effect)},
        promotion_target_metric={_CONTRACT_ID: "pp512"},
        promotion_trigger_evidence={_CONTRACT_ID: (trigger_evidence,)},
        emitted_artifacts=frozenset(
            {_CORRECTNESS_ARTIFACT, _PERFORMANCE_ARTIFACT, _SUBJECT_TRACE_ARTIFACT, _CONTROL_TRACE_ARTIFACT}
        ),
    )
