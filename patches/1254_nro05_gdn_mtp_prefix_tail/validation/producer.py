"""PNRO05 (NRO05/1254): patch-local validation producer.

NRO05-GDN-MTP-PREFIX on gfx1100/gfx1201. 1254 requires 1253, so the run uses
``--common-patches 1253_nro04_gfx1100_bf16_chunked_gdn``: control = baseline
+ 1253, subject = + 1254 (standard scaffold pair).

- correctness (``backend_reference``): full-vocabulary llama-server logprobs
  with MTP enabled (draft-n-max 4) on a long prompt (> K + 64 tokens, so the
  prefix path engages) agree within 5e-4 and generate the same tokens.
- activation: the subject server log carries the MTP-prefix marker.
- performance (positive): paired MTP speculative decode on llama-server
  (prompts from --producer-corpus), metric mtp_wall_tps, 10 pairs.
- controls: paired llama-bench tg128 (no MTP, the prefix path never runs).
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

_LABEL = "nro05"
_ARCHITECTURES = ("gfx1100", "gfx1201")
_CONTRACT_ID = "NRO05-GDN-MTP-PREFIX"
_MODEL_REF = "tierM-qwen35b-a3b-moe-mtp"
_MARKER_REGEX = r"BIGCHERRY_PATCH_HIT patch=1254_nro05 path=gdn_mtp_prefix_bf16"
_MTP_ARGS = ("-ngl", "99", "-c", "4096", "--parallel", "1", "--spec-type", "draft-mtp", "--spec-draft-n-max", "4")
_MTP_LANE_ARGS = ("--parallel", "1", "--metrics", "-ngl", "99", "--fit", "off",
                  "--spec-type", "draft-mtp", "--spec-draft-n-max", "4")
_TOLERANCE = 0.0005
_N_PREDICT = 32
# Long enough that n_tokens > K + 64 for the prefill batch.
_PROMPT = " ".join(["The recurrent state of a gated delta network carries information across tokens."] * 24)
_ROUNDS = 10

_ARTIFACT_NAME = "nro05-backend-reference.json"
_PERFORMANCE_ARTIFACT_NAME = "nro05-performance.json"
_SUBJECT_TRACE_ARTIFACT_NAME = "nro05-subject-server.log"
_CONTROL_TRACE_ARTIFACT_NAME = "nro05-control-server.log"


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    architecture = support.single_architecture(ctx, _ARCHITECTURES, label=_LABEL)
    if ctx.model is None:
        raise vp.ValidationProducerError(f"{_LABEL}: the MTP contract model (--model) is required")
    model = ctx.model
    identity = support.model_identity(model, model_id=_MODEL_REF, label=_LABEL)
    device = support.select_device(ctx, architecture, label=_LABEL)
    binaries = {
        role: {target: ctx.validation_binaries.get(role, {}).get(target) for target in ("llama-server", "llama-bench")}
        for role in ("control", "subject")
    }
    if not all(isinstance(b, Path) and b.is_file() for arm in binaries.values() for b in arm.values()):
        raise vp.ValidationProducerError(f"{_LABEL}: standard scaffold llama-server/llama-bench pair is missing")

    logs = ctx.workdir / "logs"
    control_log = logs / "nro05-control-server.log"
    subject_log = logs / "nro05-subject-server.log"

    def _factory(role, log_path):
        return support.server_session_factory(
            ctx, device=device, architecture=architecture, binary=binaries[role]["llama-server"], model=model,
            log_path=log_path, env={"BIGCHERRY_PATCH_TRACE": "1"}, server_args=_MTP_ARGS,
        )

    try:
        comparison = full_vocab.compare_servers(
            control_session=_factory("control", control_log),
            subject_session=_factory("subject", subject_log),
            prompt=_PROMPT, n_predict=_N_PREDICT, tolerance=_TOLERANCE,
            scratch_dir=ctx.workdir / "scratch" / "nro05",
        )
    except full_vocab.FullVocabError as exc:
        raise vp.ValidationProducerError(f"{_LABEL} backend_reference: {exc}") from exc
    backend_reference = experiment_contract.CorrectnessResult(
        check="backend_reference", passed=comparison.passed, detail=f"MTP draft-n-max 4: {comparison.detail}")

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
    subject_trace_ref = ctx.runtime.write_text_artifact(
        name=_SUBJECT_TRACE_ARTIFACT_NAME, text=support.compact_log(subject_text))
    control_trace_ref = ctx.runtime.write_text_artifact(
        name=_CONTROL_TRACE_ARTIFACT_NAME, text=support.compact_log(control_text))

    positive_effect, records, _ = support.mtp_server_lane(
        ctx,
        control_binary=binaries["control"]["llama-server"],
        subject_binary=binaries["subject"]["llama-server"],
        expected=device.execution_identity,
        env=dict(device.env_overrides),
        label=_LABEL,
        server_args=_MTP_LANE_ARGS,
        measured_pairs=_ROUNDS,
    )
    control_outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=binaries["control"]["llama-bench"], subject_binary=binaries["subject"]["llama-bench"],
        model=model, workloads=("decode",), pairs=_ROUNDS, log_context="nro05-control", device=device,
    )
    control_effect, control_run = support.lane_effect(
        control_outcome, workload="decode", metric="tg128", role="control", rounds=_ROUNDS, label=_LABEL)

    performance_ref = ctx.runtime.write_artifact(
        name=_PERFORMANCE_ARTIFACT_NAME,
        payload={
            "passed": True,
            "metrics": support.performance_metrics(positive_effect, control_effect),
            "schema_version": 1,
            "contract_id": _CONTRACT_ID,
            "architecture": architecture,
            "model_identity": identity,
            "build_identities": {r: dict(i) for r, i in ctx.validation_build_identities.items()},
            "positive": {"metric": "mtp_wall_tps", "effect": dataclasses.asdict(positive_effect), "requests": records},
            "control": {"metric": "tg128", "effect": dataclasses.asdict(control_effect),
                        "runs": list(control_run.runs), "stats": dict(control_run.stats)},
        },
    )
    ctx.runtime.write_artifact(
        name=_ARTIFACT_NAME,
        payload={"schema_version": 1, "check": "backend_reference", "passed": comparison.passed,
                 "detail": backend_reference.detail, "model_identity": identity, "comparison": comparison.document()},
    )
    trigger_evidence = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id="nro05-server-subject", role="positive", positive_hit=trigger_hit)
    return vp.ProducerResult(
        correctness={"disposition": "passed" if comparison.passed else "failed",
                     "mechanism": "nro05-mtp-full-vocab-backend-reference", "detail": backend_reference.detail},
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
        promotion_target_metric={_CONTRACT_ID: "mtp_wall_tps"},
        promotion_trigger_evidence={_CONTRACT_ID: (trigger_evidence,)},
        emitted_artifacts=frozenset(
            {_ARTIFACT_NAME, _PERFORMANCE_ARTIFACT_NAME, _SUBJECT_TRACE_ARTIFACT_NAME, _CONTROL_TRACE_ARTIFACT_NAME}),
    )
