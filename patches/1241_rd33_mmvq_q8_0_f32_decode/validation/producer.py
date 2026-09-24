"""PRBE26 (RD33/1241): patch-local validation producer.

RD33-MMVQ-Q8_0-F32-DECODE, gfx1100 only; control = baseline, subject =
baseline + 1241 (standard scaffold pair for the server/bench lanes, an own
test-backend-ops pair for correctness).

- correctness (``backend_reference``): test-backend-ops MUL_MAT cases with
  type_a=q8_0 (upstream covers n=1..9, k=256 and n=1/8, k=4096) must all pass
  the CPU-reference tolerance on BOTH arms. Output is not bit-identical by
  design (F32 activations instead of Q8_1), so tolerance is the contract.
- activation: the subject test-backend-ops run under BIGCHERRY_PATCH_TRACE=1
  must log the marker for every ncols_dst 1..8; the control none.
- performance (positive): paired MTP speculative decode on llama-server,
  tierL-qwen27b-q8 across both gfx1100 cards (-sm tensor, draft-n-max 4, so
  verification batches of 5), metric mtp_wall_tps, 10 measured pairs.
  Per-request draft acceptance is recorded (work equivalence).
- controls: paired llama-bench tg128 on tierA-qwen4b-q6k (Q6_K never takes
  the path), one gfx1100, 10 rounds.

Run with --device-map gfx1100=0,1 and HIP_VISIBLE_DEVICES=0,1.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
from pathlib import Path

from bigcherry.experiment import contract as experiment_contract
from bigcherry.experiment import execution as experiment_execution
from bigcherry.experiment.attestation import ExecutionIdentity
from bigcherry.patch import producer_support as support
from bigcherry.patch import validation_producer as vp
from bigcherry.patch.activation import ActivationEvidence

_LABEL = "rd33"
_ARCHITECTURE = "gfx1100"
_CONTRACT_ID = "RD33-MMVQ-Q8_0-F32-DECODE"
_MODEL_REF = "tierL-qwen27b-q8"
_CONTROL_MODEL_REF = "tierA-qwen4b-q6k"
_MARKER = re.compile(r"BIGCHERRY_PATCH_HIT patch=1241_rd33 path=q8_0_f32_decode ncols=(\d+)")
_REQUIRED_NCOLS = frozenset(range(1, 9))
_PASSED = re.compile(r"(\d+)/(\d+) tests passed")
_ROUNDS = 10

_CORRECTNESS_ARTIFACT = "rd33-correctness.json"
_PERFORMANCE_ARTIFACT = "rd33-performance.json"
_SUBJECT_TRACE_ARTIFACT = "rd33-subject-trace.log"
_CONTROL_TRACE_ARTIFACT = "rd33-control-trace.log"


def _fail(message: str) -> vp.ValidationProducerError:
    return vp.ValidationProducerError(f"{_LABEL}: {message}")


def _run_backend_ops(binary: Path, env: dict[str, str]) -> tuple[str, int, int, int]:
    """(combined output, returncode, passed, total) for the Q8_0 MUL_MAT cases."""
    completed = subprocess.run(
        [str(binary), "-o", "MUL_MAT", "-p", "type_a=q8_0,type_b=f32", "-b", "ROCm0"],
        capture_output=True, text=True, env=env, check=False, timeout=3600,
    )
    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    counts = _PASSED.findall(text)
    if not counts:
        raise _fail(f"test-backend-ops printed no pass summary (exit {completed.returncode})")
    passed, total = (int(v) for v in counts[-1])
    return text, completed.returncode, passed, total


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    if ctx.fat_targets.targets != (_ARCHITECTURE,):
        raise _fail(f"{_CONTRACT_ID} is scoped to gfx1100; got {ctx.fat_targets.targets!r}")
    if ctx.model is None:
        raise _fail("the MTP contract model (--model) is required")
    identity = support.model_identity(ctx.model, model_id=_MODEL_REF, label=_LABEL)
    control_model_raw = ctx.inputs.get("control_model")
    if not control_model_raw:
        raise _fail("--producer-input control_model=<Q6_K dense gguf> is required")
    control_model = Path(control_model_raw)
    control_identity = support.model_identity(control_model, model_id=_CONTROL_MODEL_REF, label=_LABEL)

    visibility = experiment_execution.require_device_visibility(
        context=f"{ctx.patch_id}: RD33 preflight", exact_count=2, env=ctx.build_env
    )
    devices = [d for d in ctx.runtime.device_contexts(device_map=ctx.device_map) if d.architecture == _ARCHITECTURE]
    if not devices:
        raise _fail("--device-map must map gfx1100 devices")
    device = devices[0]

    # ---- correctness + activation: test-backend-ops Q8_0 MUL_MAT ----
    pair = ctx.runtime.build_pair(
        targets=(_ARCHITECTURE,), primary_target="test-backend-ops",
        baseline_source="bigcherry", require_parity=True,
    )
    tbo_env = dict(ctx.build_env)
    tbo_env.update(dict(device.env_overrides))
    for key in device.env_unset:
        tbo_env.pop(key, None)
    tbo_env["BIGCHERRY_PATCH_TRACE"] = "1"
    arms = {role: _run_backend_ops(binary, tbo_env)
            for role, binary in (("control", pair.control_bin), ("subject", pair.subject_bin))}
    arm_ok = {role: rc == 0 and total > 0 and passed == total for role, (_, rc, passed, total) in arms.items()}
    subject_ncols = {int(n) for n in _MARKER.findall(arms["subject"][0])}
    control_ncols = {int(n) for n in _MARKER.findall(arms["control"][0])}
    correctness_passed = all(arm_ok.values())
    detail = (
        f"Q8_0 MUL_MAT test-backend-ops: control {arms['control'][2]}/{arms['control'][3]}, "
        f"subject {arms['subject'][2]}/{arms['subject'][3]} within CPU-reference tolerance"
    )
    backend_reference = experiment_contract.CorrectnessResult(
        check="backend_reference", passed=correctness_passed, detail=detail
    )
    activation_ok = _REQUIRED_NCOLS <= subject_ncols and not control_ncols
    activation = ActivationEvidence(
        status="executed" if activation_ok else "not_executed",
        mechanism="trace_marker",
        detail=f"subject ncols hit={sorted(subject_ncols)} (need 1..8), control hits={sorted(control_ncols)}",
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

    # ---- performance: MTP server lane (dual gfx1100) + dense control ----
    servers = {role: ctx.validation_binaries.get(role, {}).get("llama-server") for role in ("control", "subject")}
    benches = {role: ctx.validation_binaries.get(role, {}).get("llama-bench") for role in ("control", "subject")}
    if not all(isinstance(b, Path) and b.is_file() for b in (*servers.values(), *benches.values())):
        raise _fail("standard scaffold llama-server/llama-bench pair is missing")
    positive_effect, records, combined_logs = support.mtp_server_lane(
        ctx,
        control_binary=servers["control"],
        subject_binary=servers["subject"],
        expected=ExecutionIdentity(backend="rocm", architectures=(_ARCHITECTURE, _ARCHITECTURE)),
        env={"HIP_VISIBLE_DEVICES": ",".join(str(d) for d in visibility.device_ids)},
        label=_LABEL,
        measured_pairs=_ROUNDS,
    )
    control_outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=benches["control"], subject_binary=benches["subject"], model=control_model,
        workloads=("decode",), pairs=_ROUNDS, log_context="rd33-control", device=device,
    )
    control_effect, control_run = support.lane_effect(
        control_outcome, workload="decode", metric="tg128", role="control", rounds=_ROUNDS, label=_LABEL
    )

    def _acceptance(rows: list[dict]) -> list[object]:
        return [row.get("draft_acceptance", row.get("spec_acceptance")) for row in rows]

    performance_ref = ctx.runtime.write_artifact(
        name=_PERFORMANCE_ARTIFACT,
        payload={
            "passed": True,
            "schema_version": 1,
            "contract_id": _CONTRACT_ID,
            "positive_model_identity": identity,
            "control_model_identity": control_identity,
            "build_identities": {r: dict(i) for r, i in ctx.validation_build_identities.items()},
            "positive": {"metric": "mtp_wall_tps", "effect": dataclasses.asdict(positive_effect),
                         "draft_acceptance": {arm: _acceptance(rows) for arm, rows in records.items()},
                         "requests": records},
            "control": {"metric": "tg128", "effect": dataclasses.asdict(control_effect),
                        "runs": list(control_run.runs), "stats": dict(control_run.stats)},
            "server_logs": {arm: str(path) for arm, path in combined_logs.items()},
        },
    )
    trigger_evidence = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id="rd33-test-backend-ops-subject", role="positive", positive_hit=activation_ok
    )

    return vp.ProducerResult(
        correctness={
            "disposition": "passed" if correctness_passed else "failed",
            "mechanism": "rd33-q8_0-mul-mat-backend-reference",
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
        promotion_target_metric={_CONTRACT_ID: "mtp_wall_tps"},
        promotion_trigger_evidence={_CONTRACT_ID: (trigger_evidence,)},
        emitted_artifacts=frozenset(
            {_CORRECTNESS_ARTIFACT, _PERFORMANCE_ARTIFACT, _SUBJECT_TRACE_ARTIFACT, _CONTROL_TRACE_ARTIFACT}
        ),
    )
