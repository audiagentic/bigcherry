"""PNRO03 (NRO03/1252): patch-local validation producer.

NRO03-ALLREDUCE-P2P, dual gfx1100 only; control = validated BC, subject =
validated BC + 1252 (standard scaffold pair). P2P is opt-in, so every server
arm runs with GGML_CUDA_AR_P2P=1 (the control build ignores it).

- correctness (``bit_identical``): a fixed temperature-0 request on a
  dual-gfx1100 ``-sm tensor`` llama-server WITHOUT speculative decoding (so
  every step returns a full-vocabulary row) must give byte-identical
  logprobs and tokens on both arms: P2P only changes how the same bytes
  travel between the cards.
- activation: the subject server log carries the 1252 marker (the startup
  probe passed and the P2P path executed); the control log cannot.
- performance (positive): paired MTP speculative decode on the production
  model across both gfx1100 (-sm tensor, draft-n-max 4), metric
  mtp_wall_tps, 10 measured pairs, draft acceptance recorded.
- controls: paired llama-bench tg128 on a dense model on ONE gfx1100 (a
  single GPU never runs the AllReduce), 10 rounds.

Run with --device-map gfx1100=0,1 and HIP_VISIBLE_DEVICES=0,1 (queue VIS=0,1).
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from bigcherry.experiment import contract as experiment_contract
from bigcherry.experiment import execution as experiment_execution
from bigcherry.experiment import full_vocab
from bigcherry.experiment.attestation import ExecutionIdentity
from bigcherry.experiment.server_execution import AttestedServerSession
from bigcherry.patch import producer_support as support
from bigcherry.patch import validation_producer as vp
from bigcherry.patch.activation import ActivationEvidence

_LABEL = "nro03"
_ARCHITECTURE = "gfx1100"
_CONTRACT_ID = "NRO03-ALLREDUCE-P2P"
_MODEL_REF = "tierL-qwen27b-q8"
_CONTROL_MODEL_REF = "tierA-qwen4b-q6k"
_MARKER = re.compile(r"BIGCHERRY_PATCH_HIT patch=1252_nro03 path=allreduce_p2p_source_push")
_P2P_ENV = {"GGML_CUDA_AR_P2P": "1"}
_TENSOR_ARGS = ("-ngl", "99", "-c", "4096", "--parallel", "1", "-sm", "tensor", "--fit", "off")
_N_PREDICT = 32
_PROMPT = "Explain in two sentences how a tensor-parallel all-reduce combines partial sums across GPUs."
_ROUNDS = support.contract_paired_rounds(_CONTRACT_ID)

_CORRECTNESS_ARTIFACT = "nro03-correctness.json"
_PERFORMANCE_ARTIFACT = "nro03-performance.json"
_SUBJECT_TRACE_ARTIFACT = "nro03-subject-server.log"
_CONTROL_TRACE_ARTIFACT = "nro03-control-server.log"


def _fail(message: str) -> vp.ValidationProducerError:
    return vp.ValidationProducerError(f"{_LABEL}: {message}")


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    if ctx.fat_targets.targets != (_ARCHITECTURE,):
        raise _fail(f"{_CONTRACT_ID} is scoped to gfx1100; got {ctx.fat_targets.targets!r}")
    if ctx.model is None:
        raise _fail("the dual-GPU contract model (--model) is required")
    identity = support.model_identity(ctx.model, model_id=_MODEL_REF, label=_LABEL)
    control_model_raw = ctx.inputs.get("control_model")
    if not control_model_raw:
        raise _fail("--producer-input control_model=<dense gguf> is required")
    control_model = Path(control_model_raw)
    control_identity = support.model_identity(control_model, model_id=_CONTROL_MODEL_REF, label=_LABEL)

    visibility = experiment_execution.require_device_visibility(
        context=f"{ctx.patch_id}: NRO03 preflight", exact_count=2, env=ctx.build_env
    )
    devices = [d for d in ctx.runtime.device_contexts(device_map=ctx.device_map) if d.architecture == _ARCHITECTURE]
    if not devices:
        raise _fail("--device-map must map gfx1100 devices")
    device = devices[0]

    servers = {role: ctx.validation_binaries.get(role, {}).get("llama-server") for role in ("control", "subject")}
    benches = {role: ctx.validation_binaries.get(role, {}).get("llama-bench") for role in ("control", "subject")}
    if not all(isinstance(b, Path) and b.is_file() for b in (*servers.values(), *benches.values())):
        raise _fail("standard scaffold llama-server/llama-bench pair is missing")

    pair_env = {"HIP_VISIBLE_DEVICES": ",".join(str(d) for d in visibility.device_ids)}
    server_env = {**pair_env, **_P2P_ENV, "BIGCHERRY_PATCH_TRACE": "1"}
    expected = ExecutionIdentity(backend="rocm", architectures=(_ARCHITECTURE, _ARCHITECTURE))

    # ---- correctness + activation: bit-identical full-vocab on -sm tensor ----
    logs = ctx.workdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    preflights = support.tensor_split_preflights(
        servers, model=ctx.model, server_args=_TENSOR_ARGS, env=server_env, workdir=logs, label=_LABEL
    )
    server_logs = {role: logs / f"nro03-{role}-server.log" for role in servers}

    def _factory(role: str):
        def _open():
            return AttestedServerSession(
                binary=servers[role], model=ctx.model, expected=expected, extra_args=_TENSOR_ARGS,
                log_path=server_logs[role], env_overrides=server_env, env_unset=("ROCR_VISIBLE_DEVICES",),
                tensor_split_preflight=preflights[role],
            )
        return _open

    try:
        comparison = full_vocab.compare_servers(
            control_session=_factory("control"), subject_session=_factory("subject"),
            prompt=_PROMPT, n_predict=_N_PREDICT, criterion=full_vocab.BIT_IDENTICAL,
            scratch_dir=ctx.workdir / "scratch" / "nro03",
        )
    except full_vocab.FullVocabError as exc:
        raise _fail(f"bit_identical: {exc}") from exc
    bit_identical = experiment_contract.CorrectnessResult(
        check="bit_identical", passed=comparison.passed, detail=f"dual-gfx1100 -sm tensor, P2P on: {comparison.detail}"
    )

    subject_text = server_logs["subject"].read_text(encoding="utf-8", errors="replace")
    control_text = server_logs["control"].read_text(encoding="utf-8", errors="replace")
    subject_hit = _MARKER.search(subject_text) is not None
    control_hit = _MARKER.search(control_text) is not None
    trigger_hit = subject_hit and not control_hit
    activation = ActivationEvidence(
        status="executed" if trigger_hit else ("unobservable" if subject_hit else "not_executed"),
        mechanism="trace_marker",
        detail=(f"marker {_MARKER.pattern!r} subject_hit={subject_hit} control_hit={control_hit}"
                + ("" if subject_hit else " (P2P probe may have failed: see the subject server log)")),
    )
    subject_trace_ref = ctx.runtime.write_text_artifact(name=_SUBJECT_TRACE_ARTIFACT, text=support.compact_log(subject_text))
    control_trace_ref = ctx.runtime.write_text_artifact(name=_CONTROL_TRACE_ARTIFACT, text=support.compact_log(control_text))
    ctx.runtime.write_artifact(
        name=_CORRECTNESS_ARTIFACT,
        payload={"schema_version": 1, "contract_id": _CONTRACT_ID, "check": "bit_identical",
                 "passed": comparison.passed, "detail": bit_identical.detail, "model_identity": identity,
                 "comparison": comparison.document()},
    )

    # ---- performance: MTP decode on both gfx1100 with P2P requested ----
    positive_effect, records, combined_logs = support.mtp_server_lane(
        ctx,
        control_binary=servers["control"],
        subject_binary=servers["subject"],
        expected=expected,
        env={**pair_env, **_P2P_ENV},
        label=_LABEL,
        measured_pairs=_ROUNDS,
    )
    control_outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=benches["control"], subject_binary=benches["subject"], model=control_model,
        workloads=("decode",), pairs=_ROUNDS, log_context="nro03-control", device=device,
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
            "positive": {"metric": "mtp_wall_tps", "effect": dataclasses.asdict(positive_effect),
                         "draft_acceptance": {arm: [r.get("draft_acceptance") for r in rows]
                                              for arm, rows in records.items()},
                         "requests": records},
            "control": {"metric": "tg128", "effect": dataclasses.asdict(control_effect),
                        "runs": list(control_run.runs), "stats": dict(control_run.stats)},
            "server_logs": {arm: str(path) for arm, path in combined_logs.items()},
        },
    )
    trigger_evidence = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id="nro03-server-subject", role="positive", positive_hit=trigger_hit
    )
    return vp.ProducerResult(
        correctness={"disposition": "passed" if comparison.passed else "failed",
                     "mechanism": "nro03-dual-gfx1100-full-vocab-bit-identical", "detail": bit_identical.detail},
        validation_build_identities=ctx.validation_build_identities,
        activation_evidence=activation,
        performance_evidence={"artifact": {"path": performance_ref.path, "sha256": performance_ref.sha256}},
        trace_evidence={
            "positive": {"artifact": {"path": subject_trace_ref.path, "sha256": subject_trace_ref.sha256}},
            "negative": {"artifact": {"path": control_trace_ref.path, "sha256": control_trace_ref.sha256}},
        },
        check_results=(),
        lane_effects=(),
        contract_correctness_results=(bit_identical,),
        promotion_lane_effects={_CONTRACT_ID: (positive_effect, control_effect)},
        promotion_target_metric={_CONTRACT_ID: "mtp_wall_tps"},
        promotion_trigger_evidence={_CONTRACT_ID: (trigger_evidence,)},
        emitted_artifacts=frozenset(
            {_CORRECTNESS_ARTIFACT, _PERFORMANCE_ARTIFACT, _SUBJECT_TRACE_ARTIFACT, _CONTROL_TRACE_ARTIFACT}),
    )
