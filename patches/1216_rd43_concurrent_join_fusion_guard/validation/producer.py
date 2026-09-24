"""PRBE35 (RD43/1216): patch-local validation producer.

RD43-CONCURRENT-JOIN-FUSION-GUARD is a correctness contract: 1216 keeps
op-fusion from absorbing the join node of 1215's concurrent shared-expert
region. Both arms carry 1215 (its hard prerequisite) and differ only by 1216:

  control = bigcherry + 1215
  subject = bigcherry + 1215 + 1216

Everything runs with GGML_CUDA_GRAPH_OPT=1, the only mode in which 1215
launches concurrent regions and the guard can engage.

- correctness (``backend_reference``): a fixed temperature-0 request on
  llama-server gives the same generated tokens and full-vocabulary logprobs
  within 5e-4 on both arms (fusion placement may change FP rounding, so bit
  identity is not required).
- activation: the subject server log carries the 1216 marker (the fusion
  horizon was capped at a join node); the control log cannot.
- controls: paired decode llama-bench, 10 rounds, both arms under
  GGML_CUDA_GRAPH_OPT=1 (the contract allows at most a 1% regression).

What this does NOT prove: that the control would have aborted. Earlier real
runs show the 1215-only control completes on this model, so the evidence is
"guard engaged, harmless, free", not a reproduction of the hazard.

The standard scaffold is skipped: its control is the plain baseline, which
cannot carry the subject's prerequisite. Correctness and activation run on
this producer's llama-server pair; the controls lane on its llama-bench pair
(same sources and composition, different target), whose identities are the
record's build identities; the server identities are in the correctness
artifact.
"""

from __future__ import annotations

import dataclasses
import re

from bigcherry.experiment import contract as experiment_contract
from bigcherry.experiment import execution as experiment_execution
from bigcherry.experiment import full_vocab
from bigcherry.patch import producer_support as support
from bigcherry.patch import source as psi
from bigcherry.patch import validation_producer as vp
from bigcherry.patch.activation import ActivationEvidence

_LABEL = "rd43"
_CONTRACT_ARCHITECTURES = ("gfx1100", "gfx1201", "gfx1030")
_CONTRACT_ID = "RD43-CONCURRENT-JOIN-FUSION-GUARD"
_PREREQUISITE = "1215_rd394041_amd_stream_moe_overlap"
_MODEL_REF = "tierM-qwen35b-a3b-moe-mtp"
_GRAPH_OPT_ENV = {"GGML_CUDA_GRAPH_OPT": "1"}
_MARKER_REGEX = r"BIGCHERRY_PATCH_HIT patch=1216_rd43 path=join_fusion_cap"
_TOLERANCE = 0.0005
_N_PREDICT = 64
_PROMPT = (
    "Explain in one concise sentence why a shared expert can run beside the "
    "routed experts without changing the layer's output."
)
_ROUNDS = 10

_ARTIFACT_NAME = "rd43-backend-reference.json"
_PERFORMANCE_ARTIFACT_NAME = "rd43-controls.json"
_SUBJECT_TRACE_ARTIFACT_NAME = "rd43-subject-server.log"
_CONTROL_TRACE_ARTIFACT_NAME = "rd43-control-server.log"


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    architecture = support.single_architecture(ctx, _CONTRACT_ARCHITECTURES, label=_LABEL)
    if ctx.model is None:
        raise vp.ValidationProducerError(f"{_LABEL}: a real model (--model) is required")
    model = ctx.model
    identity = support.model_identity(model, model_id=_MODEL_REF, label=_LABEL)
    device = support.select_device(ctx, architecture, label=_LABEL)

    pairs = {
        target: ctx.runtime.build_pair(
            targets=_CONTRACT_ARCHITECTURES,
            primary_target=target,
            common_extra_patches=(_PREREQUISITE,),
            baseline_source="bigcherry",
            require_parity=True,
        )
        for target in ("llama-server", "llama-bench")
    }
    server_pair, bench_pair = pairs["llama-server"], pairs["llama-bench"]

    logs = ctx.workdir / "logs"
    control_log = logs / "rd43-control-server.log"
    subject_log = logs / "rd43-subject-server.log"
    server_env = {**_GRAPH_OPT_ENV, "BIGCHERRY_PATCH_TRACE": "1"}

    def _factory(binary, log_path):
        return support.server_session_factory(
            ctx, device=device, architecture=architecture, binary=binary, model=model,
            log_path=log_path, env=server_env,
        )

    try:
        comparison = full_vocab.compare_servers(
            control_session=_factory(server_pair.control_bin, control_log),
            subject_session=_factory(server_pair.subject_bin, subject_log),
            prompt=_PROMPT,
            n_predict=_N_PREDICT,
            tolerance=_TOLERANCE,
            scratch_dir=ctx.workdir / "scratch" / "rd43",
        )
    except full_vocab.FullVocabError as exc:
        raise vp.ValidationProducerError(f"{_LABEL} backend_reference: {exc}") from exc
    backend_reference = experiment_contract.CorrectnessResult(
        check="backend_reference", passed=comparison.passed,
        detail=f"GGML_CUDA_GRAPH_OPT=1: {comparison.detail}",
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

    outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=bench_pair.control_bin, subject_binary=bench_pair.subject_bin, model=model,
        workloads=("decode",), pairs=_ROUNDS, log_context="rd43-controls", device=device,
        env_overrides=_GRAPH_OPT_ENV,
    )
    control_effect, decode_run = support.lane_effect(
        outcome, workload="decode", metric="tg128", role="control", rounds=_ROUNDS, label=_LABEL
    )

    controls_ref = ctx.runtime.write_artifact(
        name=_PERFORMANCE_ARTIFACT_NAME,
        payload={
            "passed": True,
            "metrics": support.performance_metrics(control_effect),
            "schema_version": 1,
            "contract_id": _CONTRACT_ID,
            "architecture": architecture,
            "model_identity": identity,
            "env": _GRAPH_OPT_ENV,
            "build_identities": {r: dict(i) for r, i in bench_pair.validation_build_identities.items()},
            "control": {"metric": "tg128", "effect": dataclasses.asdict(control_effect),
                        "runs": list(decode_run.runs), "stats": dict(decode_run.stats)},
        },
    )
    ctx.runtime.write_artifact(
        name=_ARTIFACT_NAME,
        payload={
            "schema_version": 1,
            "check": "backend_reference",
            "passed": comparison.passed,
            "detail": backend_reference.detail,
            "model_identity": identity,
            "env": server_env,
            "comparison": comparison.document(),
            "subject_source_tree": psi.git_worktree_tree(server_pair.subject_source),
            "control_source_tree": psi.git_worktree_tree(server_pair.control_source),
            "server_build_identities": {r: dict(i) for r, i in server_pair.validation_build_identities.items()},
        },
    )
    trigger_evidence = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id="rd43-server-subject", role="positive", positive_hit=trigger_hit
    )

    return vp.ProducerResult(
        correctness={
            "disposition": "passed" if comparison.passed else "failed",
            "mechanism": "rd43-full-vocab-backend-reference",
            "detail": backend_reference.detail,
        },
        validation_build_identities=bench_pair.validation_build_identities,
        activation_evidence=activation,
        performance_evidence={"artifact": {"path": controls_ref.path, "sha256": controls_ref.sha256}},
        trace_evidence={
            "positive": {"artifact": {"path": subject_trace_ref.path, "sha256": subject_trace_ref.sha256}},
            "negative": {"artifact": {"path": control_trace_ref.path, "sha256": control_trace_ref.sha256}},
        },
        check_results=(),
        lane_effects=(),
        contract_correctness_results=(backend_reference,),
        promotion_lane_effects={_CONTRACT_ID: (control_effect,)},
        promotion_target_metric={_CONTRACT_ID: "tg128"},
        promotion_trigger_evidence={_CONTRACT_ID: (trigger_evidence,)},
        emitted_artifacts=frozenset(
            {_ARTIFACT_NAME, _PERFORMANCE_ARTIFACT_NAME, _SUBJECT_TRACE_ARTIFACT_NAME, _CONTROL_TRACE_ARTIFACT_NAME}
        ),
    )
