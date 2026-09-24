"""PRBE14 (RD17/1207): patch-local validation producer.

RD17-MOE-TOPK-DOWN-FOLD folds the MoE top-k routing-weight MUL into the
down-projection MMVQ epilogue. Control = baseline, subject = baseline + 1207,
both from the standard scaffold (llama-server + llama-bench).

- correctness (``backend_reference``): full-vocabulary llama-server logprobs
  for a fixed temperature-0 request on the MoE model agree within 5e-4 with
  identical generated tokens (the epilogue scale can change FP rounding).
- activation: the subject server log carries the 1207 marker; control none.
- performance (positive): paired llama-bench tg128 on the MoE model
  (single-token MoE decode is where the fold applies), 10 rounds.
- controls: paired llama-bench tg128 on a dense model with no MoE
  (``control_model``), 10 rounds.

An informal 6-round dual-gfx1100 run measured -1.32% (GPT
req_7add510830424329); this producer is the formal per-architecture
measurement that a lifecycle decision can rest on.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from bigcherry.experiment import contract as experiment_contract
from bigcherry.experiment import execution as experiment_execution
from bigcherry.experiment import full_vocab
from bigcherry.patch import producer_support as support
from bigcherry.patch import validation_producer as vp
from bigcherry.patch.activation import ActivationEvidence

_LABEL = "rd17"
_CONTRACT_ARCHITECTURES = ("gfx1100", "gfx1201", "gfx1030")
_CONTRACT_ID = "RD17-MOE-TOPK-DOWN-FOLD"
_MODEL_REF = "tierM-gptoss20b-q6k"
_CONTROL_MODEL_REF = "tierA-qwen4b-q6k"
_MARKER_REGEX = r"BIGCHERRY_PATCH_HIT patch=1207_rd17 path=moe_topk_down_fold"
_TOLERANCE = 0.0005
_N_PREDICT = 64
_PROMPT = "Explain in one concise sentence how a router picks experts for each token."
_ROUNDS = 10

_ARTIFACT_NAME = "rd17-backend-reference.json"
_PERFORMANCE_ARTIFACT_NAME = "rd17-performance.json"
_SUBJECT_TRACE_ARTIFACT_NAME = "rd17-subject-server.log"
_CONTROL_TRACE_ARTIFACT_NAME = "rd17-control-server.log"


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    architecture = support.single_architecture(ctx, _CONTRACT_ARCHITECTURES, label=_LABEL)
    if ctx.model is None:
        raise vp.ValidationProducerError(f"{_LABEL}: the MoE model (--model) is required")
    model = ctx.model
    identity = support.model_identity(model, model_id=_MODEL_REF, label=_LABEL)
    control_model_raw = ctx.inputs.get("control_model")
    if not control_model_raw:
        raise vp.ValidationProducerError(f"{_LABEL}: --producer-input control_model=<dense gguf> is required")
    control_model = Path(control_model_raw)
    control_identity = support.model_identity(control_model, model_id=_CONTROL_MODEL_REF, label=_LABEL)
    device = support.select_device(ctx, architecture, label=_LABEL)

    binaries = {
        role: {target: ctx.validation_binaries.get(role, {}).get(target) for target in ("llama-server", "llama-bench")}
        for role in ("control", "subject")
    }
    if not all(isinstance(b, Path) and b.is_file() for arm in binaries.values() for b in arm.values()):
        raise vp.ValidationProducerError(f"{_LABEL}: standard scaffold llama-server/llama-bench pair is missing")

    logs = ctx.workdir / "logs"
    control_log = logs / "rd17-control-server.log"
    subject_log = logs / "rd17-subject-server.log"

    def _factory(role, log_path):
        return support.server_session_factory(
            ctx, device=device, architecture=architecture, binary=binaries[role]["llama-server"],
            model=model, log_path=log_path, env={"BIGCHERRY_PATCH_TRACE": "1"},
        )

    try:
        comparison = full_vocab.compare_servers(
            control_session=_factory("control", control_log),
            subject_session=_factory("subject", subject_log),
            prompt=_PROMPT,
            n_predict=_N_PREDICT,
            tolerance=_TOLERANCE,
            scratch_dir=ctx.workdir / "scratch" / "rd17",
        )
    except full_vocab.FullVocabError as exc:
        raise vp.ValidationProducerError(f"{_LABEL} backend_reference: {exc}") from exc
    backend_reference = experiment_contract.CorrectnessResult(
        check="backend_reference", passed=comparison.passed, detail=comparison.detail
    )

    subject_text = subject_log.read_text(encoding="utf-8", errors="replace")
    control_text = control_log.read_text(encoding="utf-8", errors="replace")
    marker = re.compile(_MARKER_REGEX)
    subject_hit = marker.search(subject_text) is not None
    control_hit = marker.search(control_text) is not None
    trigger_hit = subject_hit and not control_hit
    activation = ActivationEvidence(
        status="executed" if trigger_hit else ("unobservable" if subject_hit else "not_executed"),
        mechanism="trace_marker",
        detail=f"marker {_MARKER_REGEX!r} subject_hit={subject_hit} control_hit={control_hit}",
    )
    subject_trace_ref = ctx.runtime.write_text_artifact(name=_SUBJECT_TRACE_ARTIFACT_NAME, text=subject_text)
    control_trace_ref = ctx.runtime.write_text_artifact(name=_CONTROL_TRACE_ARTIFACT_NAME, text=control_text)

    lanes = {}
    for role, lane_model in (("positive", model), ("control", control_model)):
        outcome = ctx.runtime.run_paired_llama_benchmark(
            control_binary=binaries["control"]["llama-bench"], subject_binary=binaries["subject"]["llama-bench"],
            model=lane_model, workloads=("decode",), pairs=_ROUNDS, log_context=f"rd17-{role}", device=device,
        )
        lanes[role] = support.lane_effect(
            outcome, workload="decode", metric="tg128", role=role, rounds=_ROUNDS, label=_LABEL
        )
    (positive_effect, positive_run), (control_effect, control_run) = lanes["positive"], lanes["control"]

    performance_ref = ctx.runtime.write_artifact(
        name=_PERFORMANCE_ARTIFACT_NAME,
        payload={
            "passed": True,
            "metrics": support.performance_metrics(positive_effect, control_effect),
            "schema_version": 1,
            "contract_id": _CONTRACT_ID,
            "architecture": architecture,
            "positive_model_identity": identity,
            "control_model_identity": control_identity,
            "build_identities": {r: dict(i) for r, i in ctx.validation_build_identities.items()},
            "positive": {"metric": "tg128", "effect": dataclasses.asdict(positive_effect),
                         "runs": list(positive_run.runs), "stats": dict(positive_run.stats)},
            "control": {"metric": "tg128", "effect": dataclasses.asdict(control_effect),
                        "runs": list(control_run.runs), "stats": dict(control_run.stats)},
            "trigger": {"subject_hit": subject_hit, "control_hit": control_hit},
        },
    )
    ctx.runtime.write_artifact(
        name=_ARTIFACT_NAME,
        payload={
            "schema_version": 1,
            "check": "backend_reference",
            "passed": comparison.passed,
            "detail": comparison.detail,
            "model_identity": identity,
            "comparison": comparison.document(),
        },
    )
    trigger_evidence = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id="rd17-server-subject", role="positive", positive_hit=trigger_hit
    )

    return vp.ProducerResult(
        correctness={
            "disposition": "passed" if comparison.passed else "failed",
            "mechanism": "rd17-full-vocab-backend-reference",
            "detail": comparison.detail,
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
        contract_correctness_results=(backend_reference,),
        promotion_lane_effects={_CONTRACT_ID: (positive_effect, control_effect)},
        promotion_target_metric={_CONTRACT_ID: "tg128"},
        promotion_trigger_evidence={_CONTRACT_ID: (trigger_evidence,)},
        emitted_artifacts=frozenset(
            {_ARTIFACT_NAME, _PERFORMANCE_ARTIFACT_NAME, _SUBJECT_TRACE_ARTIFACT_NAME, _CONTROL_TRACE_ARTIFACT_NAME}
        ),
    )
