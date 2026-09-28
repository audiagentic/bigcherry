"""PNRO10 (NRO10/1261): patch-local validation producer.

PNRO10-SPEC-CTX-OTHER-DEVICES, dual gfx1100 only; control = validated BC,
subject = validated BC + 1261 (standard scaffold pair).

1261 adds ctx_other's model devices to the draft/MTP context's scheduler
backend list. common_speculative_init_result() (common/speculative.cpp)
always sets ``cparams.ctx_other = ctx_tgt`` whenever a draft context is
created -- there is no separate opt-in flag; the real-world trigger is an
ordinary dual-GPU ``-sm tensor`` MTP deployment (the shape 1252/1241/1269
already exercise), where the draft/MTP context can otherwise lack a
scheduler backend for a tensor the target context owns on another device.

- correctness (``greedy_parity``): a fixed temperature-0/top_k=1/seeded MTP
  completion on the dual-gfx1100 ``-sm tensor`` server must return the
  identical token sequence on control and subject -- 1261 changes only
  whether the scheduler can find a backend for the shared ctx_other
  tensors, never what tokens are ultimately emitted.
- No activation check: the fix is unconditional, with no marker/env toggle
  to probe.
- performance (positive): paired MTP speculative decode, workload
  ``mtp_verify`` per the contract (mtp_server_lane's real metric key is
  ``mtp_wall_tps``, same as every other MTP-lane producer -- the
  contract's workload label and the lane-effect metric key are distinct
  names), tierA-qwen4b-q6k across both gfx1100 (-sm tensor, draft-n-max
  4), 10 measured pairs.
- controls: paired llama-bench tg128 on the SAME model (tierA-qwen4b-q6k;
  the contract uses one model for both positive and controls), ONE gfx1100
  (a single GPU never needs ctx_other's cross-device backend), 10 rounds.

Run with --device-map gfx1100=0,1 and HIP_VISIBLE_DEVICES=0,1 (queue VIS=0,1).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from bigcherry.experiment import contract as experiment_contract
from bigcherry.experiment import execution as experiment_execution
from bigcherry.experiment.attestation import ExecutionIdentity
from bigcherry.experiment.server_execution import AttestedServerSession
from bigcherry.patch import producer_support as support
from bigcherry.patch import validation_producer as vp

_LABEL = "nro10"
_ARCHITECTURE = "gfx1100"
_CONTRACT_ID = "PNRO10-SPEC-CTX-OTHER-DEVICES"
_MODEL_REF = "tierA-qwen4b-q6k"
_TENSOR_MTP_ARGS = (
    "-ngl", "99", "-c", "4096", "--parallel", "1", "-sm", "tensor", "--fit", "off",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
)
_N_PREDICT = 64
_PROMPT = " ".join(["Verify each speculative draft token against the target model before accepting it."] * 20)
_ROUNDS = support.contract_paired_rounds(_CONTRACT_ID)

_CORRECTNESS_ARTIFACT = "nro10-correctness.json"
_PERFORMANCE_ARTIFACT = "nro10-performance.json"


def _fail(message: str) -> vp.ValidationProducerError:
    return vp.ValidationProducerError(f"{_LABEL}: {message}")


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    if ctx.fat_targets.targets != (_ARCHITECTURE,):
        raise _fail(f"{_CONTRACT_ID} is scoped to gfx1100; got {ctx.fat_targets.targets!r}")
    if ctx.model is None:
        raise _fail("the dual-GPU contract model (--model) is required")
    identity = support.model_identity(ctx.model, model_id=_MODEL_REF, label=_LABEL)

    visibility = experiment_execution.require_device_visibility(
        context=f"{ctx.patch_id}: PNRO10 preflight", exact_count=2, env=ctx.build_env
    )
    devices = [d for d in ctx.runtime.device_contexts(device_map=ctx.device_map) if d.architecture == _ARCHITECTURE]
    if len(devices) != 2:
        raise _fail(f"--device-map must map exactly two {_ARCHITECTURE} devices for the dual-GPU tensor-split lane; got {len(devices)}")
    if any(d.locator is None for d in devices):
        raise _fail(
            "RSA01: server-lane attestation requires real per-device PCI locators (the legacy "
            "log-parsing ROCm attestor does not see llama-server's runtime output at all -- see "
            "docs/planning/active/run-validation-attestation/RSA01.md); --device-map entries "
            "without a locator cannot be attested for a server session"
        )
    device = devices[0]

    servers = {role: ctx.validation_binaries.get(role, {}).get("llama-server") for role in ("control", "subject")}
    benches = {role: ctx.validation_binaries.get(role, {}).get("llama-bench") for role in ("control", "subject")}
    if not all(isinstance(b, Path) and b.is_file() for b in (*servers.values(), *benches.values())):
        raise _fail("standard scaffold llama-server/llama-bench pair is missing")

    pair_env = {"HIP_VISIBLE_DEVICES": ",".join(str(d) for d in visibility.device_ids)}
    # RSA01: llama-server does not print the legacy ggml_cuda_init/gfx log lines the plain
    # log-parsing ROCm attestor (parse_rocm_attestation) depends on -- that attestor is only
    # meaningful for llama-bench-style measured processes. A server session must instead be
    # identified via real per-device PCI locators (the same architecture_by_locator mechanism
    # server_session_factory already uses for the single-device case), extended to both devices
    # of this tensor-split pair.
    expected = ExecutionIdentity(
        backend="rocm", architectures=(_ARCHITECTURE, _ARCHITECTURE),
        locators=tuple(d.locator for d in devices),
    )
    by_locator = {d.locator: _ARCHITECTURE for d in devices}

    # ---- correctness: greedy MTP token identity on the dual-gfx1100 -sm tensor server ----
    logs = ctx.workdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # RSA01: under -sm tensor, layers are assigned to a single virtual
    # "Meta()" scheduler device, never per-GPU -- the server's own log can
    # never attest per-device identity for a tensor-split session on its
    # own. tensor_split_preflights() runs a once-per-arm untimed RCCL
    # preflight probe (the only diagnostic delta is NCCL_DEBUG) and binds
    # that attestation to every timed session of the same binary/model/
    # args/devices via tensor_split_preflight= below (the same pattern
    # 1252/nro03 uses).
    preflights = support.tensor_split_preflights(
        servers, model=ctx.model, server_args=_TENSOR_MTP_ARGS, env=pair_env, workdir=logs, label=_LABEL
    )
    server_logs = {role: logs / f"nro10-{role}-server.log" for role in servers}
    tokens: dict[str, list[int]] = {}
    for role in ("control", "subject"):
        session = AttestedServerSession(
            binary=servers[role], model=ctx.model, expected=expected, extra_args=_TENSOR_MTP_ARGS,
            log_path=server_logs[role], env_overrides=pair_env, env_unset=("ROCR_VISIBLE_DEVICES",),
            architecture_by_locator=by_locator, tensor_split_preflight=preflights[role],
        )
        with session:
            reply = session.post_json("/completion", {
                "prompt": _PROMPT,
                "n_predict": _N_PREDICT,
                "temperature": 0.0,
                "top_k": 1,
                "seed": 42,
                "cache_prompt": False,
                "ignore_eos": True,
                "return_tokens": True,
            })
        ids = reply.get("tokens")
        if not isinstance(ids, list) or len(ids) != _N_PREDICT:
            raise _fail(f"{role} returned invalid token vector")
        tokens[role] = ids

    first_diff = next((i for i, (a, b) in enumerate(zip(tokens["control"], tokens["subject"])) if a != b), None)
    passed = first_diff is None
    detail = f"greedy MTP {_N_PREDICT} tokens " + ("identical" if passed else f"diverge at step {first_diff}")
    correctness = experiment_contract.CorrectnessResult(check="greedy_parity", passed=passed, detail=detail)
    ctx.runtime.write_artifact(
        name=_CORRECTNESS_ARTIFACT,
        payload={
            "schema_version": 1, "contract_id": _CONTRACT_ID, "check": "greedy_parity",
            "passed": passed, "detail": detail, "model_identity": identity,
            "first_divergence": first_diff, "tokens": tokens,
        },
    )

    # ---- performance: MTP verify on both gfx1100; controls: tg128 on one gfx1100 ----
    positive_effect, records, _ = support.mtp_server_lane(
        ctx,
        control_binary=servers["control"],
        subject_binary=servers["subject"],
        expected=expected,
        env=pair_env,
        label=_LABEL,
        measured_pairs=_ROUNDS,
        requests_per_start=support.contract_measurement(_CONTRACT_ID).server_requests_per_start,
    )
    control_outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=benches["control"], subject_binary=benches["subject"], model=ctx.model,
        workloads=("decode",), pairs=_ROUNDS, log_context="nro10-control", device=device,
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
            "model_identity": identity,
            "build_identities": {r: dict(i) for r, i in ctx.validation_build_identities.items()},
            "positive": {"metric": "mtp_wall_tps", "effect": dataclasses.asdict(positive_effect),
                         "draft_acceptance": {arm: [r.get("draft_acceptance") for r in rows]
                                              for arm, rows in records.items()},
                         "requests": records},
            "control": {"metric": "tg128", "effect": dataclasses.asdict(control_effect),
                        "runs": list(control_run.runs), "stats": dict(control_run.stats)},
        },
    )

    return vp.ProducerResult(
        correctness={"disposition": "passed" if passed else "failed",
                     "mechanism": "nro10-dual-gfx1100-greedy-mtp-token-identity", "detail": detail},
        validation_build_identities=ctx.validation_build_identities,
        activation_evidence=None,
        performance_evidence={"artifact": {"path": performance_ref.path, "sha256": performance_ref.sha256}},
        trace_evidence=None,
        check_results=(),
        lane_effects=(),
        contract_correctness_results=(correctness,),
        promotion_lane_effects={_CONTRACT_ID: (positive_effect, control_effect)},
        promotion_target_metric={_CONTRACT_ID: "mtp_wall_tps"},
        # No activation marker exists for this fix (unconditional whenever a
        # draft context is created), so there is no positive trigger to
        # probe -- record honestly as not-hit rather than fabricate one,
        # same pattern RD26 (1210) uses for its own marker-less contract.
        promotion_trigger_evidence={_CONTRACT_ID: (experiment_execution.trigger_evidence_from_marker_probe(
            lane_id="nro10-server", role="positive", positive_hit=False),)},
        emitted_artifacts=frozenset({_CORRECTNESS_ARTIFACT, _PERFORMANCE_ARTIFACT}),
    )
