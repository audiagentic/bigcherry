"""PA36 migration #4 (dev-gpt-agent req_82fbbafe52c0472d): patch-local
validation producer for 1234 (RD58-PIN-STATE-BUFFER-MULTIGPU-RESTORE).

Mechanically migrated off validation_campaign.py's
run_rd58_state_restore_evidence() -- that function (and the
--run-rd58-state-restore CLI path) is DELETED from shared code in the
same change (no compatibility layer, per the project's migrate-up
doctrine).

The producer owns ONLY the measurement: the paired test-save-load-state
build pair (singleton gfx1100, built once via ``ctx.runtime.build_pair()``),
the real state-restore execution under the SAME GGML_CUDA_REGISTER_HOST=1
env and ``-sm tensor`` topology (the ambient dual-GPU visibility is
preserved -- RD58's contract requires 2+ real GPUs, this producer never
restricts to one device the way RD04/RD08's single-GPU producers do),
the raw per-arm activation logs, the raw correctness artifact, the raw
controls artifact (performance.json -- the repeated state-restore
execution record), and the decode llama-bench promotion lane (reusing
the scaffold parity llama-bench pair).

Three real, independent claims, all from the SAME repeated runs:
- correctness: the subject binary's test-save-load-state exit code
  (0 iff every one of its 5 internal tests, including "Test 4: seq
  copy (host)", actually passed -- that internal baseline-vs-
  restored-continuation comparison IS the real correctness proof;
  this function does not re-implement or second-guess it).
- activation: subject-hit/control-miss on the real
  "pinned state buffer (... bytes) for restore" registration-success
  marker RD58's own diagnostic emits -- NOT the generic tune-binary/
  fusion-disabled negative control, which is meaningless here.
- controls: a real repeated control/subject execution record proving
  no crash/regression across ``repetitions`` real restore cycles --
  no invented latency threshold (this contract carries none).

The decode llama-bench is the promotion lane (the contract's control
workload). The contract is control-only (both acceptance gain fields
None), so the promotion lane is the control lane (decode), not a
positive lane. The producer returns the decode lane effects as
``promotion_lane_effects`` and the decode metric as
``promotion_target_metric``; the dispatcher owns
``aggregate_contract_effects()`` + ``evaluate_promotion_gate()`` (the
producer never computes a gate itself).
"""

from __future__ import annotations

import re
import subprocess

from bigcherry.patch import validation_producer as vp

# One contract architecture per run: the operator names it via
# --amdgpu-targets (singleton gfx1100, NOT fat-3).
_CONTRACT_ARCHITECTURES: tuple[str, ...] = ("gfx1100",)

# The real state-restore marker RD58's own diagnostic emits.
_MARKER_PATTERN = re.compile(r"pinned state buffer \(\d+ bytes\) for restore")

# The number of real restore cycles (the historical default).
_REPETITIONS: int = 3


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    from bigcherry.experiment import contract as experiment_contract
    from bigcherry.patch import activation as patch_activation

    if (
        len(ctx.fat_targets.targets) != 1
        or ctx.fat_targets.targets[0] not in _CONTRACT_ARCHITECTURES
    ):
        raise vp.ValidationProducerError(
            "RD58 state-restore requires exactly one contract architecture "
            "per run: gfx1100; "
            f"got targets={ctx.fat_targets.targets!r}"
        )
    architecture = ctx.fat_targets.targets[0]

    model = ctx.model
    if model is None:
        raise vp.ValidationProducerError(
            "RD58 state-restore requires a model (--model); got None"
        )

    # GPT round 1 MAJOR #4: repeat the visibility check producer-side
    # (the generic pre-scaffold enforcement remains the primary early
    # guard; this is the producer's own provenance record).
    from bigcherry.experiment import execution as _exec
    visibility = _exec.require_device_visibility(
        context=f"{ctx.patch_id}: RD58 state-restore",
        minimum_count=2,
    )
    hardware_doc = visibility.document()

    # The one sanctioned pair authority: control = baseline; subject =
    # baseline + focal (1234). Singleton gfx1100 (NOT fat-3), parity
    # asserted (the state-restore comparison depends on build parity).
    pair = ctx.runtime.build_pair(
        targets=_CONTRACT_ARCHITECTURES,
        primary_target="test-save-load-state",
        baseline_source="bigcherry",
        require_parity=True,
    )
    control_binary = pair.control_bin
    subject_binary = pair.subject_bin

    # GPT direction (req_82fbbafe52c0472d Q3): do NOT use
    # device_contexts(). Use the ambient dual-GPU visibility (the
    # pre-scaffold preflight -- Change 1 -- already validated
    # require_device_visibility(minimum_count=2)). Strip
    # ROCR_VISIBLE_DEVICES (to avoid the double-filtering bug) and
    # BIGCHERRY_* (to avoid stale overrides), set
    # GGML_CUDA_REGISTER_HOST=1.
    # GPT round 2 MAJOR: use the canonical sanitize_environment()
    # (mode="stock") instead of hand-copying its key list -- the
    # hand-copied version missed GGML_HIP_AUTOTUNE_MODE and
    # over-stripped NCCL_* (only NCCL_DEBUG* should be stripped).
    from bigcherry.campaign.benchmark import sanitize_environment
    env = sanitize_environment(dict(ctx.build_env), mode="stock")
    for key in list(env):
        if key.startswith("BIGCHERRY_"):
            env.pop(key, None)
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env["GGML_CUDA_REGISTER_HOST"] = "1"

    def _command(binary: object) -> list[str]:
        return [str(binary), "-m", str(model), "-sm", "tensor", "-ngl", "99"]

    def _run(binary: object, role: str, index: int) -> dict[str, object]:
        command = _command(binary)
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, env=env
        )
        combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
        return {
            "role": role,
            "index": index,
            "command": command,
            "returncode": completed.returncode,
            "marker_hit": _MARKER_PATTERN.search(combined) is not None,
            "stdout": completed.stdout or "",
            "stderr": completed.stderr or "",
        }

    subject_runs = [_run(subject_binary, "subject", i) for i in range(_REPETITIONS)]
    control_runs = [_run(control_binary, "control", i) for i in range(_REPETITIONS)]

    correctness_passed = all(r["returncode"] == 0 for r in subject_runs)
    subject_hit = any(r["marker_hit"] for r in subject_runs)
    control_hit = any(r["marker_hit"] for r in control_runs)
    # GPT round 3: the controls artifact claims "no crash/regression
    # across repeated control/subject execution" -- it must fail if
    # EITHER arm fails, not just the control arm (a subject that only
    # crashed on repeated runs would otherwise be reported as a controls
    # pass).
    controls_passed = all(r["returncode"] == 0 for r in control_runs) and all(
        r["returncode"] == 0 for r in subject_runs
    )

    def _strip_output(runs: list[dict[str, object]]) -> list[dict[str, object]]:
        # Persist returncode/marker_hit/command in full; truncate raw
        # stdout/stderr to a bounded tail per run so the artifact stays
        # a reasonable size across `repetitions` real process launches.
        return [
            {
                **r,
                "stdout": str(r["stdout"])[-2000:],
                "stderr": str(r["stderr"])[-2000:],
            }
            for r in runs
        ]

    # --- Artifact 1: rd58-correctness.json ---
    correctness_doc = {
        "ops": ["STATE_RESTORE_SEQ_CP_HOST"],
        "passed": correctness_passed,
        "campaign_id": ctx.campaign_id,
        "architecture": architecture,
        "hardware": hardware_doc,
        "validation_build_identities": {
            "control": dict(pair.validation_build_identities["control"]),
            "subject": dict(pair.validation_build_identities["subject"]),
        },
        "subject_runs": _strip_output(subject_runs),
    }
    correctness_artifact_ref = ctx.runtime.write_artifact(
        name="rd58-correctness.json",
        payload=correctness_doc,
    )

    # --- Artifact 2: rd58-trigger-subject.log ---
    subject_log_ref = ctx.runtime.write_text_artifact(
        name="rd58-trigger-subject.log",
        text="\n---\n".join(
            f"{r['role']}-{r['index']}:\n{r['stdout']}\n{r['stderr']}"
            for r in subject_runs
        ),
    )

    # --- Artifact 3: rd58-trigger-control.log ---
    control_log_ref = ctx.runtime.write_text_artifact(
        name="rd58-trigger-control.log",
        text="\n---\n".join(
            f"{r['role']}-{r['index']}:\n{r['stdout']}\n{r['stderr']}"
            for r in control_runs
        ),
    )

    # --- Artifact 4: rd58-trigger.json ---
    trigger_doc = {
        "marker_regex": _MARKER_PATTERN.pattern,
        "subject_hit": subject_hit,
        "control_hit": control_hit,
        "subject_runs": _strip_output(subject_runs),
        "control_runs": _strip_output(control_runs),
    }
    trigger_artifact_ref = ctx.runtime.write_artifact(
        name="rd58-trigger.json",
        payload=trigger_doc,
    )

    # --- Artifact 5: performance.json (the controls artifact) ---
    controls_doc = {
        "campaign_id": ctx.campaign_id,
        "passed": controls_passed,
        "repetitions": _REPETITIONS,
        "hardware": hardware_doc,
        "metrics": {
            "control_pass_count": sum(1 for r in control_runs if r["returncode"] == 0),
            "subject_pass_count": sum(1 for r in subject_runs if r["returncode"] == 0),
        },
        "validation_build_identities": {
            "control": dict(pair.validation_build_identities["control"]),
            "subject": dict(pair.validation_build_identities["subject"]),
        },
    }
    controls_artifact_ref = ctx.runtime.write_artifact(
        name="performance.json",
        payload=controls_doc,
    )

    # --- Decode llama-bench (the promotion lane) ---
    # The paired benchmark REUSES the scaffold parity llama-bench pair
    # (design ruling: one standard scaffold, no second build pair).
    bench_control = ctx.validation_binaries.get("control", {}).get("llama-bench")
    bench_subject = ctx.validation_binaries.get("subject", {}).get("llama-bench")
    if bench_control is None or bench_subject is None:
        raise vp.ValidationProducerError(
            "RD58 benchmark requires the standard scaffold's llama-bench "
            "binaries (standard_campaign='run' populates "
            "ProducerContext.validation_binaries); got "
            f"{ {k: sorted(v) for k, v in ctx.validation_binaries.items()}!r}"
        )

    # GPT direction (req_82fbbafe52c0472d Q3): do NOT use device_contexts().
    # Use device=None (preserve the explicit ambient HIP_VISIBLE_DEVICES),
    # set GGML_CUDA_REGISTER_HOST=1, strip ROCR_VISIBLE_DEVICES.
    outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=bench_control,
        subject_binary=bench_subject,
        model=model,
        workloads=("decode",),
        pairs=3,
        log_context="rd58",
        device=None,
        env_overrides={"GGML_CUDA_REGISTER_HOST": "1"},
        env_unset=("ROCR_VISIBLE_DEVICES",),
        runtime_args=("-sm", "tensor"),
    )
    decode_run = outcome.runs["decode"]
    decode_stats = dict(decode_run.stats)
    decode_geometric = decode_stats.get("geometric_effect_pct")
    if not isinstance(decode_geometric, (int, float)):
        raise vp.ValidationProducerError(
            "RD58 benchmark: the decode lane did not produce a valid "
            f"geometric_effect_pct; got {decode_geometric!r} "
            f"(stats={decode_stats!r})"
        )
    decode_ci_low = decode_stats.get("ci95_low_pct")
    decode_ci_high = decode_stats.get("ci95_high_pct")
    decode_paired_rounds = decode_stats.get("paired_rounds")
    decode_pair_ratios = decode_stats.get("pair_ratios") or ()

    # The contract is control-only (both acceptance gain fields None), so
    # the promotion lane is the control lane (decode), not a positive
    # lane. The producer returns the decode lane effects as
    # promotion_lane_effects and the decode metric as
    # promotion_target_metric; the dispatcher owns
    # aggregate_contract_effects() + evaluate_promotion_gate() (the
    # producer never computes a gate itself).
    contract_id = "RD58-PIN-STATE-BUFFER-MULTIGPU-RESTORE"
    decode_lane_effect = experiment_contract.LaneEffect(
        role="control",
        metric="tg128",
        geometric_effect_pct=float(decode_geometric),
        ci95_low_pct=float(decode_ci_low)
        if isinstance(decode_ci_low, (int, float))
        else None,
        ci95_high_pct=float(decode_ci_high)
        if isinstance(decode_ci_high, (int, float))
        else None,
        paired_rounds=int(decode_paired_rounds)
        if isinstance(decode_paired_rounds, (int, float))
        else None,
        pair_ratios=tuple(float(r) for r in decode_pair_ratios),
    )

    # The state_restore_integrity correctness result (the contract's
    # required check).
    state_restore_result = experiment_contract.CorrectnessResult(
        check="state_restore_integrity",
        passed=correctness_passed,
        detail=(
            f"RD58 state-restore: subject binary's test-save-load-state "
            f"exit code 0 across all {_REPETITIONS} real restore cycles "
            f"(including 'Test 4: seq copy (host)')"
            if correctness_passed
            else (
                f"RD58 state-restore: subject binary's test-save-load-state "
                f"failed on at least one of {_REPETITIONS} real restore cycles"
            )
        ),
    )

    # The activation evidence (subject-hit/control-miss on the real
    # "pinned state buffer (... bytes) for restore" marker).
    activation_ok = subject_hit and not control_hit
    activation_evidence = patch_activation.ActivationEvidence(
        status="executed" if activation_ok else "not_executed",
        mechanism="rd58-trigger-marker",
        detail=(
            f"RD58 activation: subject hit the 'pinned state buffer' marker "
            f"on at least one of {_REPETITIONS} real restore cycles; "
            f"control emitted no marker"
            if activation_ok
            else (
                f"RD58 activation: subject_hit={subject_hit} "
                f"control_hit={control_hit} (expected subject_hit=True, "
                f"control_hit=False)"
            )
        ),
    )

    # The trigger evidence for the promotion gate (the dispatcher owns
    # evaluate_trigger_proof()). The target code path is the "pinned
    # state buffer (... bytes) for restore" marker (the state-restore
    # path). candidate_launches is 1 if the subject hit the marker
    # (the real positive evidence), 0 otherwise -- NOT the test
    # invocation count (GPT round 1 BLOCKER #2).
    trigger_evidence = experiment_contract.TriggerEvidence(
        role="positive",
        lane_id="rd58-subject",
        candidate_launches=1 if subject_hit else 0,
        expected_route_selected=1 if subject_hit else 0,
    )

    # Semantic evidence for the shared binder: exactly
    # {disposition, mechanism, detail} -- no identity fields.
    return vp.ProducerResult(
        correctness={
            "disposition": "passed" if correctness_passed else "failed",
            "mechanism": "rd58-state-restore-integrity",
            "detail": state_restore_result.detail,
        },
        validation_build_identities=pair.validation_build_identities,
        activation_evidence=activation_evidence,
        performance_evidence={
            "artifact": {
                "path": controls_artifact_ref.path,
                "sha256": controls_artifact_ref.sha256,
            }
        },
        # The producer owns ONLY the per-arm artifact refs -- the plan
        # owns the marker semantics (the shared binder injects the
        # marker_regex from validation.toml's trace-marker check).
        trace_evidence={
            "positive": {
                "artifact": {
                    "path": subject_log_ref.path,
                    "sha256": subject_log_ref.sha256,
                },
            },
            "negative": {
                "artifact": {
                    "path": control_log_ref.path,
                    "sha256": control_log_ref.sha256,
                },
            },
        },
        check_results=(),
        lane_effects=(),
        contract_correctness_results=(state_restore_result,),
        promotion_lane_effects={
            contract_id: (decode_lane_effect,),
        },
        promotion_target_metric={
            contract_id: "tg128",
        },
        promotion_trigger_evidence={
            contract_id: (trigger_evidence,),
        },
        emitted_artifacts=frozenset(
            {
                correctness_artifact_ref.name,
                subject_log_ref.name,
                control_log_ref.name,
                trigger_artifact_ref.name,
                controls_artifact_ref.name,
            }
        ),
    )
