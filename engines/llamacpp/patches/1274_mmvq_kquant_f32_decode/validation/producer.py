"""1274: patch-local validation producer.

RD74-MMVQ-Q6_K-F32-DECODE, gfx1100 only; control = baseline (incl. 1241),
subject = baseline + 1274.

- correctness (``backend_reference``): test-backend-ops MUL_MAT cases with
  type_a=q6_K must pass CPU-reference tolerance on BOTH arms.
- activation: subject under BIGCHERRY_PATCH_TRACE=1 must log the q6_k marker
  for ncols_dst=1 only; ncols 2..8 forbidden; control logs none.
- performance (positive): paired plain decode on tierB-qwen9b-q6k (one card).
- controls: paired plain decode on tierL-qwen27b-q8 (-sm tensor, both cards),
  which has no Q6_K weights and cannot take 1274.

Run with --device-map gfx1100=0,1 and HIP_VISIBLE_DEVICES=0,1.
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

_LABEL = "rd74"
_ARCHITECTURE = "gfx1100"
_CONTRACT_ID = "RD74-MMVQ-Q6_K-F32-DECODE"
_MODEL_REF = "tierB-qwen9b-q6k"
_CONTROL_MODEL_REF = "tierL-qwen27b-q8"
_MARKER = re.compile(r"BIGCHERRY_PATCH_HIT patch=1274_kquant_f32 path=f32_decode type=q6_k ncols=(\d+)")
_REQUIRED_NCOLS = frozenset({1})
_FORBIDDEN_NCOLS = frozenset(range(2, 9))
_ROUNDS = support.contract_paired_rounds(_CONTRACT_ID)

_CORRECTNESS_ARTIFACT = "rd74-correctness.json"
_PERFORMANCE_ARTIFACT = "rd74-performance.json"
_SUBJECT_TRACE_ARTIFACT = "rd74-subject-trace.log"
_CONTROL_TRACE_ARTIFACT = "rd74-control-trace.log"


def _fail(message: str) -> vp.ValidationProducerError:
    return vp.ValidationProducerError(f"{_LABEL}: {message}")


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    if ctx.fat_targets.targets != (_ARCHITECTURE,):
        raise _fail(f"{_CONTRACT_ID} is scoped to gfx1100; got {ctx.fat_targets.targets!r}")
    if ctx.model is None:
        raise _fail("the positive decode model (--model) is required")
    identity = support.model_identity(ctx.model, model_id=_MODEL_REF, label=_LABEL)
    control_model_raw = ctx.inputs.get("control_model")
    if not control_model_raw:
        raise _fail("--producer-input control_model=<tierL Q8_0 gguf> is required")
    control_model = Path(control_model_raw)
    control_identity = support.model_identity(control_model, model_id=_CONTROL_MODEL_REF, label=_LABEL)

    experiment_execution.require_device_visibility(
        context=f"{ctx.patch_id}: RD74 preflight", exact_count=2, env=ctx.build_env
    )
    devices = [d for d in ctx.runtime.device_contexts(device_map=ctx.device_map) if d.architecture == _ARCHITECTURE]
    if not devices:
        raise _fail("--device-map must map gfx1100 devices")
    device = devices[0]

    # Correctness + activation: width-1 Q6_K path must execute; widths 2..8 must not.
    pair = ctx.runtime.build_pair(
        targets=(_ARCHITECTURE,), primary_target="test-backend-ops",
        baseline_source="bigcherry", require_parity=True,
    )
    tbo_env = support.device_env(ctx, device, {"BIGCHERRY_PATCH_TRACE": "1"})
    tbo_args = ("-o", "MUL_MAT", "-p", "type_a=q6_K,type_b=f32", "-b", "ROCm0")
    arms = {role: support.run_backend_ops(binary, tbo_args, tbo_env, label=_LABEL)
            for role, binary in (("control", pair.control_bin), ("subject", pair.subject_bin))}
    arm_ok = {role: rc == 0 and total > 0 and passed == total for role, (_, rc, passed, total) in arms.items()}
    subject_ncols = {int(n) for n in _MARKER.findall(arms["subject"][0])}
    control_ncols = {int(n) for n in _MARKER.findall(arms["control"][0])}
    correctness_passed = all(arm_ok.values())
    detail = (
        f"Q6_K MUL_MAT test-backend-ops: control {arms['control'][2]}/{arms['control'][3]}, "
        f"subject {arms['subject'][2]}/{arms['subject'][3]} within CPU-reference tolerance"
    )
    backend_reference = experiment_contract.CorrectnessResult(
        check="backend_reference", passed=correctness_passed, detail=detail
    )
    activation_ok = (
        _REQUIRED_NCOLS <= subject_ncols
        and not (_FORBIDDEN_NCOLS & subject_ncols)
        and not control_ncols
    )
    vp.require_activation_ok(
        activation_ok,
        label="RD74",
        detail=(
            f"subject ncols hit={sorted(subject_ncols)} (require [1], forbid 2..8), "
            f"control hits={sorted(control_ncols)}"
        ),
    )
    activation = ActivationEvidence(
        status="executed" if activation_ok else "not_executed",
        mechanism="trace_marker",
        detail=(
            f"subject ncols hit={sorted(subject_ncols)} (require [1], forbid 2..8), "
            f"control hits={sorted(control_ncols)}"
        ),
    )
    subject_trace_ref = ctx.runtime.write_text_artifact(name=_SUBJECT_TRACE_ARTIFACT, text=arms["subject"][0])
    control_trace_ref = ctx.runtime.write_text_artifact(name=_CONTROL_TRACE_ARTIFACT, text=arms["control"][0])
    ctx.runtime.write_artifact(
        name=_CORRECTNESS_ARTIFACT,
        payload={
            "schema_version": 1,
            "contract_id": _CONTRACT_ID,
            "check": "backend_reference",
            "passed": correctness_passed,
            "detail": detail,
            "arms": {role: {"returncode": rc, "passed": p, "total": t}
                     for role, (_, rc, p, t) in arms.items()},
            "subject_marker_ncols": sorted(subject_ncols),
            "control_marker_ncols": sorted(control_ncols),
            "build_identities": {r: dict(i) for r, i in pair.validation_build_identities.items()},
        },
    )

    # Positive: plain Q6_K decode on one card. Control: 27B Q8_0 (no Q6_K weights) on both cards, -sm tensor.
    benches = {role: ctx.validation_binaries.get(role, {}).get("llama-bench") for role in ("control", "subject")}
    if not all(isinstance(b, Path) and b.is_file() for b in benches.values()):
        raise _fail("standard scaffold llama-bench pair is missing")
    positive_outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=benches["control"], subject_binary=benches["subject"], model=ctx.model,
        workloads=("decode",), pairs=_ROUNDS, log_context="rd74-positive", device=device,
    )
    positive_effect, positive_run = support.lane_effect(
        positive_outcome, workload="decode", metric="tg128", role="positive", rounds=_ROUNDS, label=_LABEL
    )
    control_outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=benches["control"], subject_binary=benches["subject"], model=control_model,
        workloads=("decode",), pairs=_ROUNDS, log_context="rd74-control", device=None,
        env_unset=("ROCR_VISIBLE_DEVICES",), runtime_args=("-sm", "tensor"),
    )
    control_effect, control_run = support.lane_effect(
        control_outcome, workload="decode", metric="tg128", role="control", rounds=_ROUNDS, label=_LABEL
    )

    performance_ref = ctx.runtime.write_artifact(
        name=_PERFORMANCE_ARTIFACT,
        payload={
            "passed": True,
            "metrics": support.performance_metrics(positive_effect, control_effect),
            "schema_version": 1,
            "contract_id": _CONTRACT_ID,
            "positive_model_identity": identity,
            "control_model_identity": control_identity,
            "build_identities": {r: dict(i) for r, i in ctx.validation_build_identities.items()},
            "positive": {"metric": "tg128", "effect": dataclasses.asdict(positive_effect),
                         "runs": list(positive_run.runs), "stats": dict(positive_run.stats)},
            "control": {"metric": "tg128", "effect": dataclasses.asdict(control_effect),
                        "runs": list(control_run.runs), "stats": dict(control_run.stats)},
            "mtp_no_regression_requirement": (
                "MTP verify widths must not activate 1274; draft acceptance/output must remain control-parity"
            ),
        },
    )
    trigger_evidence = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id="rd74-test-backend-ops-subject", role="positive", positive_hit=activation_ok
    )

    return vp.ProducerResult(
        correctness={
            "disposition": "passed" if correctness_passed else "failed",
            "mechanism": "rd74-q6_k-mul-mat-backend-reference",
            "detail": detail,
        },
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
