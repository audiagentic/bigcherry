"""RDNA4-MMQ-Q6K-CODEGEN (1006): patch-local validation producer.

Fresh, isolated Q6_K-only re-measurement (single-GPU gfx1201). The rejected
combined package (1000_rdna4_mmq_q2k_q6k_fix) measured Q6_K at 1.365x
[CI95 1.362x-1.367x] on exact-shape backend-ops (m=4096, n=512, k=14336) --
a real gain, but that evidence does not transfer to this narrower
single-edit composition identity (project re-promotion doctrine: a changed
composition requires fresh evidence).

- correctness (``backend_reference``): test-backend-ops MUL_MAT type_a=q6_k
  cases must pass the CPU-reference tolerance on BOTH arms. Filters by type
  only and accepts the full built-in shape sweep test-backend-ops generates
  for that type (mirrors 1241/RD33's own precedent) -- test-backend-ops has
  no CLI knob to isolate a single m/n/k case, so exact-shape reproduction of
  the original evidence is not attempted here.
- activation: 1006's own Q6_K MMA float-promotion edit lives inside a
  __device__ kernel body (no host I/O possible there), so the marker
  instead instruments the HOST dispatch switch (ggml_cuda_mul_mat_q_switch_
  type's case GGML_TYPE_Q6_K:, mmq.cu) -- the same dispatch site 1267/RD07
  already instruments the same way for the same reason. Subject must hit
  it under BIGCHERRY_PATCH_TRACE=1 during the correctness run; control must
  not (the marker line itself doesn't exist in an unpatched build).
- performance (positive): paired llama-bench prefill on tierA-qwen4b-q6k (a
  real Q6_K-quantized model -- prefill's larger batch is closer to the
  original n=512 evidence shape than single-token decode), gfx1201, 10
  paired rounds.
- controls: paired llama-bench decode on the same model, same device --
  ordinary single-token use, proving no regression outside the prefill path.
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

_LABEL = "mmq-q6k"
_ARCHITECTURE = "gfx1201"
_CONTRACT_ID = "RDNA4-MMQ-Q6K-CODEGEN"
_MODEL_REF = "tierA-qwen4b-q6k"
_MARKER = re.compile(
    r"BIGCHERRY_PATCH_HIT patch=1006_rdna4_mmq_q6k_codegen_fix path=q6k_mmq_dispatch"
)
_ROUNDS = support.contract_paired_rounds(_CONTRACT_ID)

_CORRECTNESS_ARTIFACT = "mmq-q6k-correctness.json"
_PERFORMANCE_ARTIFACT = "mmq-q6k-performance.json"
_SUBJECT_TRACE_ARTIFACT = "mmq-q6k-subject-trace.log"
_CONTROL_TRACE_ARTIFACT = "mmq-q6k-control-trace.log"


def _fail(message: str) -> vp.ValidationProducerError:
    return vp.ValidationProducerError(f"{_LABEL}: {message}")


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    if ctx.fat_targets.targets != (_ARCHITECTURE,):
        raise _fail(f"{_CONTRACT_ID} is scoped to gfx1201; got {ctx.fat_targets.targets!r}")
    if ctx.model is None:
        raise _fail("the contract model (--model) is required")
    identity = support.model_identity(ctx.model, model_id=_MODEL_REF, label=_LABEL)
    device = support.select_device(ctx, _ARCHITECTURE, label=_LABEL)

    # ---- correctness + activation: test-backend-ops Q6_K MUL_MAT ----
    pair = ctx.runtime.build_pair(
        targets=(_ARCHITECTURE,), primary_target="test-backend-ops",
        baseline_source="bigcherry", require_parity=True,
    )
    tbo_env = support.device_env(ctx, device, {"BIGCHERRY_PATCH_TRACE": "1"})
    tbo_args = ("-o", "MUL_MAT", "-p", "type_a=q6_k")
    arms = {role: support.run_backend_ops(binary, tbo_args, tbo_env, label=_LABEL)
            for role, binary in (("control", pair.control_bin), ("subject", pair.subject_bin))}
    arm_ok = {role: rc == 0 and total > 0 and passed == total for role, (_, rc, passed, total) in arms.items()}
    correctness_passed = all(arm_ok.values())
    detail = (
        f"Q6_K MUL_MAT test-backend-ops: control {arms['control'][2]}/{arms['control'][3]}, "
        f"subject {arms['subject'][2]}/{arms['subject'][3]} within CPU-reference tolerance"
    )
    backend_reference = experiment_contract.CorrectnessResult(
        check="backend_reference", passed=correctness_passed, detail=detail
    )
    subject_hit = _MARKER.search(arms["subject"][0]) is not None
    control_hit = _MARKER.search(arms["control"][0]) is not None
    trigger_hit = subject_hit and not control_hit
    activation = ActivationEvidence(
        status="executed" if trigger_hit else ("unobservable" if subject_hit else "not_executed"),
        mechanism="trace_marker",
        detail=f"marker subject_hit={subject_hit} control_hit={control_hit}",
    )
    subject_trace_ref = ctx.runtime.write_text_artifact(name=_SUBJECT_TRACE_ARTIFACT, text=arms["subject"][0])
    control_trace_ref = ctx.runtime.write_text_artifact(name=_CONTROL_TRACE_ARTIFACT, text=arms["control"][0])
    ctx.runtime.write_artifact(
        name=_CORRECTNESS_ARTIFACT,
        payload={
            "schema_version": 1, "contract_id": _CONTRACT_ID, "check": "backend_reference",
            "passed": correctness_passed, "detail": detail,
            "arms": {role: {"returncode": rc, "passed": p, "total": t}
                     for role, (_, rc, p, t) in arms.items()},
            "subject_marker_hit": subject_hit, "control_marker_hit": control_hit,
            "build_identities": {r: dict(i) for r, i in pair.validation_build_identities.items()},
        },
    )

    # ---- performance: paired prefill (positive) + decode (controls) ----
    benches = {role: ctx.validation_binaries.get(role, {}).get("llama-bench") for role in ("control", "subject")}
    if not all(isinstance(b, Path) and b.is_file() for b in benches.values()):
        raise _fail("standard scaffold llama-bench pair is missing")

    combined_outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=benches["control"], subject_binary=benches["subject"], model=ctx.model,
        workloads=("prefill", "decode"), pairs=_ROUNDS, log_context="mmq-q6k", device=device,
    )
    positive_effect, positive_run = support.lane_effect(
        combined_outcome, workload="prefill", metric="pp512", role="positive", rounds=_ROUNDS, label=_LABEL
    )
    control_effect, control_run = support.lane_effect(
        combined_outcome, workload="decode", metric="tg128", role="control", rounds=_ROUNDS, label=_LABEL
    )
    performance_ref = ctx.runtime.write_artifact(
        name=_PERFORMANCE_ARTIFACT,
        payload={
            "passed": True,
            "metrics": support.performance_metrics(positive_effect, control_effect),
            "schema_version": 1,
            "contract_id": _CONTRACT_ID,
            "model_identity": identity,
            "build_identities": {r: dict(i) for r, i in ctx.validation_build_identities.items()},
            "positive": {"metric": "pp512", "effect": dataclasses.asdict(positive_effect),
                         "runs": list(positive_run.runs), "stats": dict(positive_run.stats)},
            "control": {"metric": "tg128", "effect": dataclasses.asdict(control_effect),
                        "runs": list(control_run.runs), "stats": dict(control_run.stats)},
        },
    )
    trigger_evidence = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id="mmq-q6k-backend-ops-subject", role="positive", positive_hit=trigger_hit
    )

    return vp.ProducerResult(
        correctness={"disposition": "passed" if correctness_passed else "failed",
                     "mechanism": "mmq-q6k-mul-mat-backend-reference", "detail": detail},
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
        promotion_target_metric={_CONTRACT_ID: "pp512"},
        promotion_trigger_evidence={_CONTRACT_ID: (trigger_evidence,)},
        emitted_artifacts=frozenset(
            {_CORRECTNESS_ARTIFACT, _PERFORMANCE_ARTIFACT, _SUBJECT_TRACE_ARTIFACT, _CONTROL_TRACE_ARTIFACT}
        ),
    )
