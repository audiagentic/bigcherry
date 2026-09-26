"""PNRO05 (NRO05/1254): patch-local validation producer.

NRO05-GDN-MTP-PREFIX on gfx1100/gfx1201. 1254 requires 1253, so the run uses
``--common-patches 1253_nro04_gfx1100_bf16_chunked_gdn``: control = baseline
+ 1253, subject = + 1254 (standard scaffold pair).

- correctness (``backend_reference``), both required:
  1. test-backend-ops GATED_DELTA_NET (1253's cases plus 1254's S_v 128,
     K > 1, n > K + 64 prefix/tail cases and the n = K + 64 boundary) within
     the CPU-reference tolerance on both arms (control = baseline + 1253);
  2. greedy (temperature 0) generation with MTP on (draft-n-max 4) over a
     prompt long enough to take the prefix path yields the same token ids on
     both arms. A per-step full-vocabulary comparison is not possible under
     MTP: llama-server returns probabilities only for main-sampler tokens,
     not for accepted draft tokens.
- activation: the subject's test-backend-ops output AND its server log carry
  the MTP-prefix marker; the control's carry none.
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
_N_PREDICT = 64
_COMMON = ("1253_nro04_gfx1100_bf16_chunked_gdn",)
_TBO_ARGS = ("-o", "GATED_DELTA_NET", "-b", "ROCm0")
# Long enough that n_tokens > K + 64 for the prefill batch.
_PROMPT = " ".join(["The recurrent state of a gated delta network carries information across tokens."] * 24)
_ROUNDS = support.contract_paired_rounds(_CONTRACT_ID)

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

    marker = re.compile(_MARKER_REGEX)

    # ---- 1. kernel reference: test-backend-ops on both arms ----
    pair = ctx.runtime.build_pair(
        targets=_ARCHITECTURES, primary_target="test-backend-ops", common_extra_patches=_COMMON,
        baseline_source="bigcherry", require_parity=True,
    )
    tbo_env = support.device_env(ctx, device, {"BIGCHERRY_PATCH_TRACE": "1"})
    tbo = {role: support.run_backend_ops(binary, _TBO_ARGS, tbo_env, label=_LABEL)
           for role, binary in (("control", pair.control_bin), ("subject", pair.subject_bin))}
    tbo_ok = all(rc == 0 and total > 0 and ok == total for _, rc, ok, total in tbo.values())

    # ---- 2. greedy MTP token identity on llama-server ----
    logs = ctx.workdir / "logs"
    server_logs = {"control": logs / "nro05-control-server.log", "subject": logs / "nro05-subject-server.log"}
    tokens: dict[str, list[int]] = {}
    for role in ("control", "subject"):
        factory = support.server_session_factory(
            ctx, device=device, architecture=architecture, binary=binaries[role]["llama-server"], model=model,
            log_path=server_logs[role], env={"BIGCHERRY_PATCH_TRACE": "1"}, server_args=_MTP_ARGS,
        )
        with factory() as session:
            reply = session.post_json("/completion", {
                "prompt": _PROMPT, "n_predict": _N_PREDICT, "temperature": 0.0, "top_k": 1, "seed": 42,
                "cache_prompt": False, "ignore_eos": True, "return_tokens": True,
            })
        ids = reply.get("tokens")
        if not isinstance(ids, list) or len(ids) != _N_PREDICT:
            raise vp.ValidationProducerError(f"{_LABEL}: {role} server returned {len(ids) if isinstance(ids, list) else ids!r} tokens")
        tokens[role] = ids
    first_diff = next((i for i, (a, b) in enumerate(zip(tokens["control"], tokens["subject"])) if a != b), None)
    greedy_ok = first_diff is None

    passed = tbo_ok and greedy_ok
    detail = (
        f"GATED_DELTA_NET test-backend-ops control {tbo['control'][2]}/{tbo['control'][3]}, "
        f"subject {tbo['subject'][2]}/{tbo['subject'][3]}; greedy MTP (draft-n-max 4) {_N_PREDICT} tokens "
        + ("identical" if greedy_ok else f"diverge at step {first_diff}")
    )
    backend_reference = experiment_contract.CorrectnessResult(check="backend_reference", passed=passed, detail=detail)

    subject_text = server_logs["subject"].read_text(encoding="utf-8", errors="replace")
    control_text = server_logs["control"].read_text(encoding="utf-8", errors="replace")
    subject_hit = marker.search(subject_text) is not None and marker.search(tbo["subject"][0]) is not None
    control_hit = marker.search(control_text) is not None or marker.search(tbo["control"][0]) is not None
    trigger_hit = subject_hit and not control_hit
    activation = ActivationEvidence(
        status="executed" if trigger_hit else ("unobservable" if subject_hit else "not_executed"),
        mechanism="trace_marker",
        detail=f"marker {_MARKER_REGEX!r} subject(tbo+server)={subject_hit} control(any)={control_hit}",
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
        requests_per_start=support.contract_measurement(_CONTRACT_ID).server_requests_per_start,
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
        payload={"schema_version": 1, "check": "backend_reference", "passed": passed,
                 "detail": backend_reference.detail, "model_identity": identity,
                 "test_backend_ops": {role: {"returncode": rc, "passed": ok, "total": total}
                                      for role, (_, rc, ok, total) in tbo.items()},
                 "greedy_mtp": {"n_predict": _N_PREDICT, "first_divergence": first_diff, "tokens": tokens},
                 "build_identities": {r: dict(i) for r, i in pair.validation_build_identities.items()}},
    )
    trigger_evidence = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id="nro05-server-subject", role="positive", positive_hit=trigger_hit)
    return vp.ProducerResult(
        correctness={"disposition": "passed" if passed else "failed",
                     "mechanism": "nro05-tbo-and-greedy-mtp-identity", "detail": backend_reference.detail},
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
