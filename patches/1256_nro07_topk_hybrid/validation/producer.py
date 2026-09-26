"""PNRO06 (NRO07/1256): patch-local validation producer.

NRO07-TOPK-HYBRID. The fork's small-row route is compiled only for
HIP >= 7.15, so runs use the ROCm 10.x toolchain (HIP 7.16); both arms are
built with it. Control = baseline, subject = baseline + 1256 (standard
scaffold).

- correctness (``backend_reference``): test-backend-ops TOP_K (upstream
  cases: k==1, small rows, 1024-boundary, radix-sized rows, ties) within the
  reference check on both arms.
- activation: the subject test-backend-ops output carries at least one
  1256 route marker (small / parallel_radix); the control none.
- performance (series 2): MoE routing does NOT run TOP_K -- it uses the fused
  ``topk_moe`` kernel (PVPS10 profile, 2026-09-26) -- so the positive lane is
  llama-server MTP decode with ``--backend-sampling``: the top-k sampler runs
  ``ggml_top_k`` over the full ~151k-logit vocabulary every token (pre-flight
  2026-09-26: served by 1256's ``topk_small`` route). Metric: client wall-clock tokens/s. Control: dense
  llama-bench tg128 (sampling-free, so TOP_K-free).
- activation (series 2): the subject's backend-sampling server log carries
  a 1256 route marker (any route); the control's none.
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

_LABEL = "nro07"
_PATCH_TAG = "1256_nro07"
_ARCHITECTURES = ("gfx1100", "gfx1201")
_CONTRACT_ID = "NRO07-TOPK-HYBRID"
_MODEL_REF = "tierM-qwen35b-a3b-moe-mtp"
_CONTROL_MODEL_REF = "tierA-qwen4b-q6k"
_MARKER = re.compile(rf"BIGCHERRY_PATCH_HIT patch={_PATCH_TAG} path=topk_(\w+)")
_TBO_ARGS = ("-o", "TOP_K", "-b", "ROCm0")
# Single-GPU MTP server with backend sampling on: the request's top_k=20 then
# runs as ggml_top_k on the device over the full vocabulary.
_SERVER_ARGS = ("--parallel", "1", "--metrics", "-ngl", "99", "--fit", "off",
                "--spec-type", "draft-mtp", "--spec-draft-n-max", "4", "--backend-sampling")
_ROUNDS = support.contract_paired_rounds(_CONTRACT_ID)

_CORRECTNESS_ARTIFACT = f"{_LABEL}-correctness.json"
_PERFORMANCE_ARTIFACT = f"{_LABEL}-performance.json"
_SUBJECT_TRACE_ARTIFACT = f"{_LABEL}-subject-trace.log"
_CONTROL_TRACE_ARTIFACT = f"{_LABEL}-control-trace.log"


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    architecture = support.single_architecture(ctx, _ARCHITECTURES, label=_LABEL)
    if ctx.model is None:
        raise vp.ValidationProducerError(f"{_LABEL}: the MoE contract model (--model) is required")
    identity = support.model_identity(ctx.model, model_id=_MODEL_REF, label=_LABEL)
    control_model_raw = ctx.inputs.get("control_model")
    if not control_model_raw:
        raise vp.ValidationProducerError(f"{_LABEL}: --producer-input control_model=<dense gguf> is required")
    control_model = Path(control_model_raw)
    control_identity = support.model_identity(control_model, model_id=_CONTROL_MODEL_REF, label=_LABEL)
    device = support.select_device(ctx, architecture, label=_LABEL)

    pair = ctx.runtime.build_pair(
        targets=_ARCHITECTURES, primary_target="test-backend-ops",
        baseline_source="bigcherry", require_parity=True,
    )
    env = support.device_env(ctx, device, {"BIGCHERRY_PATCH_TRACE": "1"})
    arms = {
        role: support.run_backend_ops(binary, _TBO_ARGS, env, label=_LABEL)
        for role, binary in (("control", pair.control_bin), ("subject", pair.subject_bin))
    }
    passed = all(rc == 0 and total > 0 and ok == total for _, rc, ok, total in arms.values())
    detail = (
        f"TOP_K test-backend-ops: control {arms['control'][2]}/{arms['control'][3]}, "
        f"subject {arms['subject'][2]}/{arms['subject'][3]} passed"
    )
    backend_reference = experiment_contract.CorrectnessResult(check="backend_reference", passed=passed, detail=detail)
    ctx.runtime.write_artifact(
        name=_CORRECTNESS_ARTIFACT,
        payload={
            "schema_version": 1, "contract_id": _CONTRACT_ID, "check": "backend_reference",
            "passed": passed, "detail": detail,
            "subject_tbo_routes": sorted(set(_MARKER.findall(arms["subject"][0]))),
            "arms": {role: {"returncode": rc, "passed": ok, "total": total} for role, (_, rc, ok, total) in arms.items()},
            "build_identities": {r: dict(i) for r, i in pair.validation_build_identities.items()},
        },
    )

    binaries = {
        role: {tool: ctx.validation_binaries.get(role, {}).get(tool) for tool in ("llama-bench", "llama-server")}
        for role in ("control", "subject")
    }
    if not all(isinstance(b, Path) and b.is_file() for tools in binaries.values() for b in tools.values()):
        raise vp.ValidationProducerError(f"{_LABEL}: standard scaffold llama-server/llama-bench pair is missing")
    positive_effect, records, server_logs = support.mtp_server_lane(
        ctx,
        control_binary=binaries["control"]["llama-server"],
        subject_binary=binaries["subject"]["llama-server"],
        # The server attestation names the device only by PCI locator.
        expected=dataclasses.replace(device.execution_identity, locators=(device.locator,))
        if device.locator is not None else device.execution_identity,
        env={**dict(device.env_overrides), "BIGCHERRY_PATCH_TRACE": "1"},
        label=_LABEL,
        server_args=_SERVER_ARGS,
        measured_pairs=_ROUNDS,
        requests_per_start=support.contract_measurement(_CONTRACT_ID).server_requests_per_start,
    )
    control_outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=binaries["control"]["llama-bench"], subject_binary=binaries["subject"]["llama-bench"],
        model=control_model, workloads=("decode",), pairs=_ROUNDS, log_context=f"{_LABEL}-control", device=device,
    )
    control_effect, control_run = support.lane_effect(
        control_outcome, workload="decode", metric="tg128", role="control", rounds=_ROUNDS, label=_LABEL)

    server_text = {arm: path.read_text(encoding="utf-8", errors="replace") for arm, path in server_logs.items()}
    subject_routes = sorted(set(_MARKER.findall(arms["subject"][0] + server_text["subject"])))
    control_routes = sorted(set(_MARKER.findall(arms["control"][0] + server_text["control"])))
    served_routes = sorted(set(_MARKER.findall(server_text["subject"])))
    trigger_hit = bool(served_routes) and not control_routes
    activation = ActivationEvidence(
        status="executed" if trigger_hit else "not_executed",
        mechanism="trace_marker",
        detail=(f"subject routes={subject_routes} (backend-sampling server routes={served_routes}) "
                f"control routes={control_routes}"),
    )
    subject_trace_ref = ctx.runtime.write_text_artifact(
        name=_SUBJECT_TRACE_ARTIFACT, text=support.compact_log(arms["subject"][0] + server_text["subject"]))
    control_trace_ref = ctx.runtime.write_text_artifact(
        name=_CONTROL_TRACE_ARTIFACT, text=support.compact_log(arms["control"][0] + server_text["control"]))
    performance_ref = ctx.runtime.write_artifact(
        name=_PERFORMANCE_ARTIFACT,
        payload={
            "passed": True,
            "metrics": support.performance_metrics(positive_effect, control_effect),
            "schema_version": 1, "contract_id": _CONTRACT_ID, "architecture": architecture,
            "positive_model_identity": identity, "control_model_identity": control_identity,
            "build_identities": {r: dict(i) for r, i in ctx.validation_build_identities.items()},
            "positive": {"metric": "mtp_wall_tps", "effect": dataclasses.asdict(positive_effect),
                         "requests": records},
            "control": {"metric": "tg128", "effect": dataclasses.asdict(control_effect),
                        "runs": list(control_run.runs), "stats": dict(control_run.stats)},
        },
    )
    trigger_evidence = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id=f"{_LABEL}-backend-sampling-server-subject", role="positive", positive_hit=trigger_hit)
    return vp.ProducerResult(
        correctness={"disposition": "passed" if passed else "failed",
                     "mechanism": f"{_LABEL}-topk-backend-reference", "detail": detail},
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
        promotion_target_metric={_CONTRACT_ID: "mtp_wall_tps"},
        promotion_trigger_evidence={_CONTRACT_ID: (trigger_evidence,)},
        emitted_artifacts=frozenset(
            {_CORRECTNESS_ARTIFACT, _PERFORMANCE_ARTIFACT, _SUBJECT_TRACE_ARTIFACT, _CONTROL_TRACE_ARTIFACT}),
    )
