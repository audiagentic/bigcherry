"""PRBE41 (1263): patch-local validation producer.

PRBE41-SSM-CONV-CHANNELS-MAJOR: control = baseline, subject = baseline + 1263
(standard scaffold llama-server + llama-bench pair).

- correctness (``backend_reference``): full-vocabulary llama-server logprobs
  on the hybrid GDN model meet full_vocab.NEAR_LOSSLESS with identical tokens (the
  graph now feeds SSM_CONV channels-major instead of a transposed copy).
- activation: the subject server log carries the channels-major marker.
- performance: pp512 prefill on the hybrid GDN model (positive; the removed
  transpose scales with ubatch), tg128 decode on gpt-oss-20B (control; no
  SSM layers), 10 paired rounds each.
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

_LABEL = "prbe41"
_CONTRACT_ARCHITECTURES = ("gfx1100", "gfx1201", "gfx1030")
_CONTRACT_ID = "PRBE41-SSM-CONV-CHANNELS-MAJOR"
_MODEL_REF = "tierA-qwen4b-q6k"
_CONTROL_MODEL_REF = "tierM-gptoss20b-q6k"
_MARKER_REGEX = r"BIGCHERRY_PATCH_HIT patch=1263_prbe41 path=ssm_conv_channels_major"
_N_PREDICT = 64
_PROMPT = " ".join(["A gated delta network mixes a short causal convolution with a recurrent state."] * 8)
_ROUNDS = 10

_ARTIFACT_NAME = "prbe41-backend-reference.json"
_PERFORMANCE_ARTIFACT_NAME = "prbe41-performance.json"
_SUBJECT_TRACE_ARTIFACT_NAME = "prbe41-subject-server.log"
_CONTROL_TRACE_ARTIFACT_NAME = "prbe41-control-server.log"


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
    control_log = logs / "prbe41-control-server.log"
    subject_log = logs / "prbe41-subject-server.log"

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
            criterion=full_vocab.NEAR_LOSSLESS,
            scratch_dir=ctx.workdir / "scratch" / "prbe41",
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
    subject_trace_ref = ctx.runtime.write_text_artifact(name=_SUBJECT_TRACE_ARTIFACT_NAME, text=support.compact_log(subject_text))
    control_trace_ref = ctx.runtime.write_text_artifact(name=_CONTROL_TRACE_ARTIFACT_NAME, text=support.compact_log(control_text))

    lanes = {}
    for role, lane_model, workload, metric in (("positive", model, "prefill", "pp512"),
                                               ("control", control_model, "decode", "tg128")):
        outcome = ctx.runtime.run_paired_llama_benchmark(
            control_binary=binaries["control"]["llama-bench"], subject_binary=binaries["subject"]["llama-bench"],
            model=lane_model, workloads=(workload,), pairs=_ROUNDS, log_context=f"prbe41-{role}", device=device,
        )
        lanes[role] = support.lane_effect(
            outcome, workload=workload, metric=metric, role=role, rounds=_ROUNDS, label=_LABEL
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
            "positive": {"metric": "pp512", "effect": dataclasses.asdict(positive_effect),
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
        lane_id="prbe41-server-subject", role="positive", positive_hit=trigger_hit
    )

    return vp.ProducerResult(
        correctness={
            "disposition": "passed" if comparison.passed else "failed",
            "mechanism": "prbe41-full-vocab-backend-reference",
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
        promotion_target_metric={_CONTRACT_ID: "pp512"},
        promotion_trigger_evidence={_CONTRACT_ID: (trigger_evidence,)},
        emitted_artifacts=frozenset(
            {_ARTIFACT_NAME, _PERFORMANCE_ARTIFACT_NAME, _SUBJECT_TRACE_ARTIFACT_NAME, _CONTROL_TRACE_ARTIFACT_NAME}
        ),
    )
