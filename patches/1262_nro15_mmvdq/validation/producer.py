"""PNRO15 (NRO15/1262): patch-local validation producer.

NRO15-MMVDQ-KQUANT-DECODE on gfx1100/gfx1201, where mmvdq is opt-in: both
arms run with GGML_CUDA_DQ_MMV=1 and GGML_CUDA_DQ_Q6K=1 (a no-op on the
control, which lacks the code), so the comparison is "mmvdq on" vs mmvq.

- correctness (``backend_reference``): test-backend-ops MUL_MAT for
  type_a in {q4_K, q5_K, q6_K} x F32 must pass the CPU-reference tolerance on
  both arms (float dot vs Q8_1 dot is not bit-identical by design).
- activation: the subject test-backend-ops output carries the mmvdq marker;
  the control none.
- performance: standard scaffold llama-bench pair, tg128 decode on the Q6_K
  contract model (positive) and pp512 prefill (control; ne11 > 1 never takes
  mmvdq), 10 paired rounds each.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from bigcherry.experiment import contract as experiment_contract
from bigcherry.experiment import execution as experiment_execution
from bigcherry.patch import producer_support as support
from bigcherry.patch import validation_producer as vp
from bigcherry.patch.activation import ActivationEvidence

_LABEL = "nro15"
_ARCHITECTURES = ("gfx1100", "gfx1201")
_CONTRACT_ID = "NRO15-MMVDQ-KQUANT-DECODE"
_MODEL_REF = "tierA-qwen4b-q6k"
_DQ_ENV = {"GGML_CUDA_DQ_MMV": "1", "GGML_CUDA_DQ_Q6K": "1"}
_MARKER = re.compile(r"BIGCHERRY_PATCH_HIT patch=1262_nro15 path=mmvdq")
_TBO_ARGS = ("-o", "MUL_MAT", "-p", "type_a=q[456]_K,type_b=f32", "-b", "ROCm0")
_ROUNDS = 10

_CORRECTNESS_ARTIFACT = "nro15-correctness.json"
_PERFORMANCE_ARTIFACT = "nro15-performance.json"
_SUBJECT_TRACE_ARTIFACT = "nro15-subject-trace.log"
_CONTROL_TRACE_ARTIFACT = "nro15-control-trace.log"


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    architecture = support.single_architecture(ctx, _ARCHITECTURES, label=_LABEL)
    if ctx.model is None:
        raise vp.ValidationProducerError(f"{_LABEL}: the Q6_K contract model (--model) is required")
    identity = support.model_identity(ctx.model, model_id=_MODEL_REF, label=_LABEL)
    device = support.select_device(ctx, architecture, label=_LABEL)

    pair = ctx.runtime.build_pair(
        targets=_ARCHITECTURES, primary_target="test-backend-ops",
        baseline_source="bigcherry", require_parity=True,
    )
    env = support.device_env(ctx, device, {**_DQ_ENV, "BIGCHERRY_PATCH_TRACE": "1"})
    arms = {
        role: support.run_backend_ops(binary, _TBO_ARGS, env, label=_LABEL)
        for role, binary in (("control", pair.control_bin), ("subject", pair.subject_bin))
    }
    passed = all(rc == 0 and total > 0 and ok == total for _, rc, ok, total in arms.values())
    detail = (
        f"K-quant MUL_MAT test-backend-ops with mmvdq enabled: control {arms['control'][2]}/{arms['control'][3]}, "
        f"subject {arms['subject'][2]}/{arms['subject'][3]} within CPU-reference tolerance"
    )
    backend_reference = experiment_contract.CorrectnessResult(check="backend_reference", passed=passed, detail=detail)
    subject_hit = _MARKER.search(arms["subject"][0]) is not None
    control_hit = _MARKER.search(arms["control"][0]) is not None
    trigger_hit = subject_hit and not control_hit
    activation = ActivationEvidence(
        status="executed" if trigger_hit else ("unobservable" if subject_hit else "not_executed"),
        mechanism="trace_marker",
        detail=f"marker {_MARKER.pattern!r} subject_hit={subject_hit} control_hit={control_hit}",
    )
    subject_trace_ref = ctx.runtime.write_text_artifact(name=_SUBJECT_TRACE_ARTIFACT, text=arms["subject"][0])
    control_trace_ref = ctx.runtime.write_text_artifact(name=_CONTROL_TRACE_ARTIFACT, text=arms["control"][0])
    ctx.runtime.write_artifact(
        name=_CORRECTNESS_ARTIFACT,
        payload={
            "schema_version": 1,
            "contract_id": _CONTRACT_ID,
            "check": "backend_reference",
            "passed": passed,
            "detail": detail,
            "env": _DQ_ENV,
            "arms": {role: {"returncode": rc, "passed": ok, "total": total} for role, (_, rc, ok, total) in arms.items()},
            "build_identities": {r: dict(i) for r, i in pair.validation_build_identities.items()},
        },
    )

    benches = {role: ctx.validation_binaries.get(role, {}).get("llama-bench") for role in ("control", "subject")}
    if not all(isinstance(b, Path) and b.is_file() for b in benches.values()):
        raise vp.ValidationProducerError(f"{_LABEL}: standard scaffold llama-bench pair is missing")
    lanes = {}
    for role, workload, metric in (("positive", "decode", "tg128"), ("control", "prefill", "pp512")):
        outcome = ctx.runtime.run_paired_llama_benchmark(
            control_binary=benches["control"], subject_binary=benches["subject"], model=ctx.model,
            workloads=(workload,), pairs=_ROUNDS, log_context=f"nro15-{role}", device=device,
            env_overrides=_DQ_ENV,
        )
        lanes[role] = support.lane_effect(
            outcome, workload=workload, metric=metric, role=role, rounds=_ROUNDS, label=_LABEL
        )
    (positive_effect, positive_run), (control_effect, control_run) = lanes["positive"], lanes["control"]
    performance_ref = ctx.runtime.write_artifact(
        name=_PERFORMANCE_ARTIFACT,
        payload={
            "passed": True,
            "metrics": support.performance_metrics(positive_effect, control_effect),
            "schema_version": 1,
            "contract_id": _CONTRACT_ID,
            "architecture": architecture,
            "env": _DQ_ENV,
            "model_identity": identity,
            "build_identities": {r: dict(i) for r, i in ctx.validation_build_identities.items()},
            "positive": {"metric": "tg128", "effect": dataclasses.asdict(positive_effect),
                         "runs": list(positive_run.runs), "stats": dict(positive_run.stats)},
            "control": {"metric": "pp512", "effect": dataclasses.asdict(control_effect),
                        "runs": list(control_run.runs), "stats": dict(control_run.stats)},
        },
    )
    trigger_evidence = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id="nro15-test-backend-ops-subject", role="positive", positive_hit=trigger_hit
    )
    return vp.ProducerResult(
        correctness={"disposition": "passed" if passed else "failed", "mechanism": "nro15-kquant-backend-reference",
                     "detail": detail},
        validation_build_identities=pair.validation_build_identities,
        activation_evidence=activation,
        performance_evidence={"artifact": {"path": performance_ref.path, "sha256": performance_ref.sha256}},
        trace_evidence={
            "positive": {"artifact": {"path": subject_trace_ref.path, "sha256": subject_trace_ref.sha256}},
            "negative": {"artifact": {"path": control_trace_ref.path, "sha256": control_trace_ref.sha256}},
        },
        check_results=(),
        lane_effects=(),
        contract_correctness_results=(backend_reference,),
        promotion_lane_effects={_CONTRACT_ID: (positive_effect, control_effect)},
        promotion_target_metric={_CONTRACT_ID: "tg128"},
        promotion_trigger_evidence={_CONTRACT_ID: (trigger_evidence,)},
        emitted_artifacts=frozenset(
            {_CORRECTNESS_ARTIFACT, _PERFORMANCE_ARTIFACT, _SUBJECT_TRACE_ARTIFACT, _CONTROL_TRACE_ARTIFACT}
        ),
    )
