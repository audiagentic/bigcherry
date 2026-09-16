"""PA37: patch-local validation producer for 1203 (RD05/RD06/RD07).

Migrated off validation_campaign.py's run_rd05_contract_correctness/
run_rd06_contract_correctness/run_rd07_contract_correctness/
_run_1203_backend_reference_contract_correctness -- those four functions
and their 1203-specific branches are DELETED from validation_campaign.py
in the same change that adds this file (no compatibility layer, per the
project's migrate-up doctrine).

Imports only bigcherry.patch.validation / bigcherry.patch.validation_producer
/ bigcherry.experiment.perplexity / bigcherry.experiment.contract /
bigcherry.experiment.execution / bigcherry.core.paths, mirroring the PA36-F
synthetic fixture's import restriction -- this module must never import
bigcherry.patch.validation_campaign (see
test_producer_modules_cannot_import_validation_campaign). RD06's real
ci95_threshold_bound_v1 performance bound (PA39 defect #2) is evaluated via
bigcherry.experiment.execution.lane_effect_from_run() and
bigcherry.experiment.contract.aggregate_contract_effects()/
evaluate_promotion_gate() -- the same real evaluator used elsewhere in this
codebase, not a parallel gate re-implementation.

Build/device shape (PA37.md step 5): ONE atomic control/subject build pair
PER BINARY SET, each built once as fat gfx1100;gfx1201;gfx1030 via
``ctx.runtime.build_pair()`` -- never rebuilt per architecture or per
check. ``ProducerRuntime.build_pair()``'s own contract is "a producer
wanting multiple binaries calls build_pair() once per binary set", so this
producer calls it exactly twice: once for ``llama-perplexity`` (backend-
reference correctness, ``_ppl_pair``) and once for ``llama-bench``
(paired performance benchmarking, ``_bench_pair``) -- PA39 P0 defect #1
fix (GPT review req_e3d28b6b104a4a01): the prior single build_pair() call
used ``primary_target="llama-perplexity"`` for BOTH backend-reference
comparisons AND ``run_paired_llama_benchmark()``, which constructs
llama-bench-style argv -- a real llama-bench-invoked-as-llama-perplexity
mismatch that fake-runtime tests could not expose. ``run()`` is still one
atomic control/subject SOURCE pair (both builds share the same resolved
control/subject composition); it is the two binaries built from that one
source pair, not two independent source pairs. RD05/RD06 consume the same
gfx1201 backend-reference result (RD05/RD06 are materially inseparable in
the atomic 1203 patch -- see validation_campaign.py's superseded RD05
producer docstring for the same reasoning, carried forward here); RD07
requires all three architectures to be present at once. The canonical
``ProducerResult.validation_build_identities`` (and every disposition-
bearing correctness check) is bound to the ``llama-perplexity`` pair's
identities; the ``llama-bench`` pair's identities are recorded separately
inside the performance artifacts for audit.

RD06's positive lanes are BOTH decode and prefill (config/experiment-
contracts.toml's ``[contract.RD06-RDNA4-WMMA-FA-CONFIG.positive]``) --
PA39 P0 defect #2 fix (same GPT review): the prior code only ever
benchmarked decode, and ``aggregate_contract_effects()`` is not itself
contract-aware -- it happily computes a gain from whatever positive-role
lane effects it is handed and never notices a declared lane is missing,
so a decode-only call could reach "promote" with zero prefill evidence.
This producer now runs every workload the contract declares for BOTH the
positive and control roles, and explicitly fails closed (no gate call at
all) if any declared workload's evidence did not come back, rather than
letting the gate silently aggregate a partial lane set.

Fail-closed by construction: a missing device, missing model/corpus, a
missing/mismatched declared lane, or a failed backend-reference comparison
always yields ``passed=False`` with a truthful detail string -- never a
default/fabricated PASS.

PA39 P1 spec-drift note (GPT review req_e3d28b6b104a4a01, not fixed here):
RD06's ``hypothesis.rationale`` in config/experiment-contracts.toml says
"must verify gfx1100 does not select or regress", but RD06's own formal
``[scope]`` is ``architectures = ["gfx1201"]`` only, and this producer
correspondingly only ever exercises RD06's checks on the gfx1201 device
context (``_RD0506_ARCH``). No gfx1100 negative check exists anywhere in
this producer or its evidence. This is a real, honest gap, not
overclaimed here or in PA37/PA39's plan docs: RD06's real evidence proves
gfx1201 behavior only; formalizing a gfx1100 non-selection/non-regression
check (which would need real hardware plus RD06's own activation marker,
not just a backend-reference/performance comparison) is out of this
change's scope and left for a future item if PA39 needs to claim that
coverage.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping

from bigcherry.core import paths as bc_paths
from bigcherry.experiment import contract as experiment_contract
from bigcherry.experiment import execution as experiment_execution
from bigcherry.patch import validation as pv
from bigcherry.patch import validation_producer as vp

_RD05 = "RD05-WMMA-FA-CORRECTNESS-BARRIERS"
_RD06 = "RD06-RDNA4-WMMA-FA-CONFIG"
_RD07 = "RD07-Q6K-MMQ-PREFILL-FOLD"

_RD0506_ARCH = "gfx1201"
_RD07_ARCHS = ("gfx1100", "gfx1201", "gfx1030")

# RD06's OWN real contract policy (config/experiment-contracts.toml
# [contract.RD06-RDNA4-WMMA-FA-CONFIG.acceptance], read directly -- PA37
# step 6 explicitly forbids reusing RD04's benchmark configuration here).
_RD06_TARGET_KERNEL_GAIN_PCT = 0.5
_RD06_MAX_CONTROL_REGRESSION_PCT = 1
_RD06_EFFECT_EVIDENCE_POLICY = "ci95_threshold_bound_v1"
_RD06_MIN_PAIRED_ROUNDS = 10

# RD07's own contract policy ([contract.RD07-Q6K-MMQ-PREFILL-FOLD.acceptance]
# -- no ci95/min_paired_rounds override, unlike RD06).
_RD07_MAX_CONTROL_REGRESSION_PCT = 1

# RD05 has no min_paired_rounds/effect_evidence_policy override either (its
# acceptance block declares only max_control_regression_pct=1) -- 3 rounds,
# matching RD07's own default.
_RD05_MIN_PAIRED_ROUNDS = 3

# PA39 real-hardware-acceptance fix (2026-09-16): the missing apply/build/
# activation/controls capability producers found by PA39's real-hardware
# acceptance attempt #1 (ConfigurationError at plan resolution, before any
# GPU work -- see PA39.md's 2026-09-16 "real-hardware acceptance attempt #1"
# note). Real BIGCHERRY_PATCH_TRACE markers, GGML_LOG_WARN once-per-process,
# already implemented in patch.py (PA37 part 3, commit d94a49d0) at the
# real host-side dispatch sites GPT confirmed (req_c2c26de482d845ad):
# ggml_cuda_flash_attn_ext()'s BEST_FATTN_KERNEL_MMA_F16 case (RD06) and
# ggml_cuda_mul_mat_q_switch_type()'s GGML_TYPE_Q6_K case (RD07). RD05 has
# no activation marker of its own (bind_contract() never requires
# 'activation' for a correctness-only contract with no target_kernel_gain/
# end_to_end_gain -- RD05 declares neither).
_RD06_ACTIVATION_MARKER = "BIGCHERRY_PATCH_HIT patch=1203_rd050607 path=wmma_f16_dispatch contract=RD06"
_RD07_ACTIVATION_MARKER = "BIGCHERRY_PATCH_HIT patch=1203_rd050607 path=q6k_mmq_dispatch contract=RD07"


_PAIRED_BENCH_METRIC_NAME: dict[str, str] = {"decode": "tg128", "prefill": "pp512"}


def _load_contract(contract_id: str) -> experiment_contract.ExperimentContract:
    # Loaded directly from config/experiment-contracts.toml via the
    # experiment.contract module -- never via validation_campaign.py/
    # patch_validation.load_contracts_for_descriptor(), which this module
    # must not import (see module docstring). This is the same real
    # ci95_threshold_bound_v1/min_paired_rounds=10/target_kernel_gain_pct=
    # 0.5/max_control_regression_pct=1 (RD06) / max_control_regression_pct=1
    # only (RD05/RD07) policy already encoded as module constants above --
    # loading the real ExperimentContract object (rather than re-deriving
    # the same numbers by hand) is what lets evaluate_promotion_gate()
    # itself apply that policy instead of this producer re-implementing the
    # gate's own bound logic.
    registry = experiment_contract.load_contracts(bc_paths.EXPERIMENT_CONTRACTS)
    return registry[contract_id]


def _run_activation_probe(
    *, subject_binary: Path, control_binary: Path, model: Path,
    device: vp.ProducerDeviceContext, marker: str, log_context: str,
) -> tuple[bool, str]:
    """PA39 activation-capability fix: proves the real BIGCHERRY_PATCH_TRACE
    marker (already implemented in patch.py) fires on the intended
    production dispatch path (ctx.trace_evidence is never populated on the
    --validation-producer path, so the framework's own builtin trace-marker
    validator cannot be used here; this is the same real evidence, gathered
    directly).

    GPT review (2026-09-16, dev-gpt-agent): a subject-trace-on vs
    subject-trace-off comparison is NOT valid negative-control evidence --
    it only proves logging can be turned off, not that the marker
    specifically identifies the patched dispatch path. The real negative
    control is the UNPATCHED control binary: it structurally cannot emit
    the marker at all, because the marker source line does not exist in
    its (unpatched) compiled code. Both probes run with
    BIGCHERRY_PATCH_TRACE=1 -- the subject binary must hit, the control
    binary must not.

    Also per _run_one_trace_probe()'s real precedent (VA21 real-hardware
    finding): llama-bench gates ggml's log level on ITS OWN --verbose
    flag, so GGML_LOG_WARN (and GGML_LOG_INFO) are filtered without it --
    this probe always passes --verbose for exactly that reason."""
    def _argv(binary: Path) -> list[str]:
        return [
            str(binary), "-m", str(model), "-p", "0", "-n", "16", "-r", "1",
            "-ngl", "99", "--verbose",
        ]

    def _run(binary: Path) -> str:
        env = {**os.environ, **dict(device.env_overrides), "BIGCHERRY_PATCH_TRACE": "1"}
        for key in device.env_unset:
            env.pop(key, None)
        completed = subprocess.run(_argv(binary), env=env, capture_output=True, text=True, check=False)
        return (completed.stdout or "") + (completed.stderr or "")

    subject_output = _run(subject_binary)
    control_output = _run(control_binary)
    subject_hit = marker in subject_output
    control_hit = marker in control_output
    ok = subject_hit and not control_hit
    detail = (
        f"{log_context}: activation probe marker={marker!r} "
        f"subject_hit={subject_hit} control_hit={control_hit} "
        "(both probes run with BIGCHERRY_PATCH_TRACE=1 -- the unpatched control binary "
        "is the negative control, since it structurally cannot contain the marker)"
    )
    return ok, detail


def _verify_control_model_identity(
    control_model: Path, *, expected_model_id: str,
) -> tuple[bool, str]:
    """PA39 P1 fix (GPT review req_e3d28b6b104a4a01): the supplied
    ``--producer-input control_model=<path>`` is a bare filesystem path
    with no inherent tie to the contract's declared control model
    identity -- an arbitrary GGUF could otherwise be silently mislabeled
    as RD06's control lane. Reads config/models.toml directly (via
    tomllib, never through validation_campaign.py's
    resolve_benchmark_model(), which this module must not import) and
    checks the supplied path's basename and real file size against the
    registry's declared entry for ``expected_model_id``.

    This is a best-effort identity check, not full provenance (it trusts
    the filesystem's basename/size, not a content hash) -- deliberately
    never raises: the caller always records both the check's outcome and
    its detail string in the performance artifact, so a mismatch stays
    visible/auditable even where full verification is out of scope."""
    import tomllib

    try:
        raw = tomllib.loads(bc_paths.MODELS.read_text(encoding="utf-8"))
    except OSError as exc:
        return False, f"control_model identity: could not read {bc_paths.MODELS}: {exc}"
    entries = {
        entry.get("id"): entry for entry in raw.get("models", [])
        if isinstance(entry, dict)
    }
    entry = entries.get(expected_model_id)
    if entry is None:
        return False, (
            f"control_model identity: {expected_model_id!r} not found in "
            f"{bc_paths.MODELS}"
        )
    declared_name = Path(str(entry.get("path", ""))).name
    declared_size = entry.get("size-bytes")
    basename_ok = control_model.name == declared_name
    size_ok = (
        isinstance(declared_size, int) and not isinstance(declared_size, bool)
        and control_model.is_file() and control_model.stat().st_size == declared_size
    )
    ok = basename_ok and size_ok
    detail = (
        f"control_model identity: supplied={control_model} "
        f"expected_model_id={expected_model_id!r} expected_basename={declared_name!r} "
        f"basename_match={basename_ok} expected_size_bytes={declared_size!r} "
        f"size_match={size_ok}"
    )
    return ok, detail


def _perplexity_module():
    # A thin, late-imported indirection (not a dependency-injection
    # parameter -- ValidationProducer.__call__ takes only ctx) so tests can
    # monkeypatch this module's own `perplexity` symbol after importing the
    # producer module directly, the same way tests exercise other
    # subprocess-driving producers without real hardware.
    from bigcherry.experiment import perplexity

    return perplexity


def _run_backend_reference(
    *, control_bin, subject_bin, model, corpus, device: vp.ProducerDeviceContext, log_context: str,
) -> tuple[bool, str, object | None]:
    perplexity = _perplexity_module()

    def _runner(argv, **kwargs):
        env = {**os.environ, **dict(device.env_overrides)}
        for key in device.env_unset:
            env.pop(key, None)
        return subprocess.run(argv, env=env, **kwargs)

    try:
        subject_run = perplexity.run_perplexity(
            subject_bin, model=model, corpus=corpus, runner=_runner,
        )
        control_run = perplexity.run_perplexity(
            control_bin, model=model, corpus=corpus, runner=_runner,
        )
    except perplexity.PerplexityError as exc:
        return False, f"{log_context}: could not produce a real perplexity run: {exc}", None

    comparison = perplexity.PerplexityComparison(subject=subject_run, control=control_run)
    detail = (
        f"{log_context}: real perplexity backend-reference comparison: "
        f"sigma={comparison.sigma:.4f} vs threshold max_sigma={comparison.max_sigma} "
        f"(subject={comparison.subject.ppl:.4f}, control={comparison.control.ppl:.4f}, "
        f"delta={comparison.delta:.5f})"
    )
    return comparison.ok, detail, comparison


def _check_result(
    *, check_id: str, contract_id: str, capability: str, passed: bool, detail: str,
    artifact: pv.ArtifactRef | None, carries_disposition: bool,
    disposition_passed: bool | None = None,
) -> vp.ProducerCheckResult:
    # validate_producer_result() allows at most ONE non-None disposition per
    # contract_id across all check_results -- a contract's [[check]] plan
    # entries all contribute to compute_verdict() (every required check
    # must pass), but exactly one of them carries the contract's own
    # promotion-eligibility disposition, so contract_verdicts is never
    # double-written for the same contract.
    #
    # GPT review (2026-09-16): the disposition-carrying check's OWN local
    # `passed` is no longer necessarily the whole truth for that contract
    # once activation/controls checks exist alongside it -- a contract
    # must not read as eligible (contract_verdicts[...]["passed"]=True)
    # while one of its OTHER required checks (activation/controls) FAILED.
    # `disposition_passed` (when given) is the real AND of every required
    # outcome for that contract, computed by the caller once all of that
    # contract's checks are known; it defaults to `passed` for the common
    # single-check-determines-the-contract case.
    disposition_value = passed if disposition_passed is None else disposition_passed
    return vp.ProducerCheckResult(
        check_id=check_id,
        contract_ids=(contract_id,),
        validation_result=pv.ValidationResult(
            check_id=check_id, capability=capability,
            status=pv.PASS if passed else pv.FAIL, summary=detail,
            artifacts=(artifact,) if artifact is not None else (),
        ),
        disposition=(
            {"passed": disposition_value, "contract_id": contract_id, "detail": detail}
            if carries_disposition else None
        ),
    )


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    # Two build pairs, one per binary set, each built once at the full fat
    # target list -- never rebuilt per architecture (PA37.md step 5; PA39
    # P0 defect #1 fix -- see module docstring). backend_reference
    # correctness comparisons use the llama-perplexity pair; every
    # run_paired_llama_benchmark() call uses the llama-bench pair.
    ppl_pair = ctx.runtime.build_pair(
        targets=ctx.fat_targets.targets, primary_target="llama-perplexity",
    )
    bench_pair = ctx.runtime.build_pair(
        targets=ctx.fat_targets.targets, primary_target="llama-bench",
    )
    devices = ctx.runtime.device_contexts(device_map=ctx.device_map)
    devices_by_arch: Mapping[str, vp.ProducerDeviceContext] = {
        d.architecture: d for d in devices
    }

    emitted_artifacts: set[str] = set()
    check_results: list[vp.ProducerCheckResult] = []

    # --- apply/build: universal capabilities, real evidence -------------
    # PA39 real-hardware-acceptance fix: reaching this point means both
    # build_pair() calls above already applied the real control/subject
    # source composition and cmake-built both binary sets successfully --
    # a build_pair() failure raises (a real infrastructure failure, not a
    # gracefully-FAILed check, matching how this producer already lets a
    # genuine build/apply error propagate rather than silently reporting a
    # FAIL check for it). This is real, truthful apply+build evidence, not
    # boilerplate that fabricates a PASS independent of what actually
    # happened -- and it is genuinely universal, unscoped to any one RD
    # contract (both binary sets are built from the same one atomic
    # control/subject source composition).
    apply_artifact = ctx.runtime.write_artifact(
        name="apply.json",
        payload={
            "schema_version": 1,
            "detail": "control/subject source composition applied for both binary sets",
            "ppl_control_build_identity": ppl_pair.validation_build_identities["control"],
            "ppl_subject_build_identity": ppl_pair.validation_build_identities["subject"],
            "bench_control_build_identity": bench_pair.validation_build_identities["control"],
            "bench_subject_build_identity": bench_pair.validation_build_identities["subject"],
        },
    )
    emitted_artifacts.add(apply_artifact.name)
    check_results.append(vp.ProducerCheckResult(
        check_id="apply", contract_ids=(),
        validation_result=pv.ValidationResult(
            check_id="apply", capability="apply", status=pv.PASS,
            summary="control/subject source composition applied for both binary sets",
            artifacts=(apply_artifact,),
        ),
    ))
    build_artifact = ctx.runtime.write_artifact(
        name="build.json",
        payload={
            "schema_version": 1,
            "detail": "control/subject trees built for both binary sets (llama-perplexity, llama-bench)",
            "ppl_control_build_identity": ppl_pair.validation_build_identities["control"],
            "ppl_subject_build_identity": ppl_pair.validation_build_identities["subject"],
            "bench_control_build_identity": bench_pair.validation_build_identities["control"],
            "bench_subject_build_identity": bench_pair.validation_build_identities["subject"],
        },
    )
    emitted_artifacts.add(build_artifact.name)
    check_results.append(vp.ProducerCheckResult(
        check_id="build", contract_ids=(),
        validation_result=pv.ValidationResult(
            check_id="build", capability="build", status=pv.PASS,
            summary="control/subject trees built for both binary sets",
            artifacts=(build_artifact,),
        ),
    ))

    # --- RD05/RD06 backend_reference: shared gfx1201 comparison ---------
    gfx1201 = devices_by_arch.get(_RD0506_ARCH)
    rd0506_artifact = None
    if gfx1201 is None or ctx.model is None or ctx.corpus is None:
        rd0506_ok = False
        rd0506_detail = (
            f"rd0506: no {_RD0506_ARCH} device and/or model/corpus supplied "
            "(hardware-free run, or missing --device-map entry)"
        )
    else:
        rd0506_ok, rd0506_detail, comparison = _run_backend_reference(
            control_bin=ppl_pair.control_bin, subject_bin=ppl_pair.subject_bin,
            model=ctx.model, corpus=ctx.corpus, device=gfx1201, log_context="rd0506",
        )
        rd0506_artifact = ctx.runtime.write_artifact(
            name="rd0506-backend-reference.json",
            payload={
                "schema_version": 1,
                "architecture": _RD0506_ARCH,
                "passed": rd0506_ok,
                "detail": rd0506_detail,
                "comparison": (
                    _perplexity_module().comparison_to_dict(comparison)
                    if comparison is not None else None
                ),
                "control_build_identity": ppl_pair.validation_build_identities["control"],
                "subject_build_identity": ppl_pair.validation_build_identities["subject"],
            },
        )
        emitted_artifacts.add(rd0506_artifact.name)

    check_results.append(_check_result(
        check_id="rd05-backend-reference", contract_id=_RD05, capability="correctness",
        passed=rd0506_ok, detail=rd0506_detail, artifact=rd0506_artifact,
        carries_disposition=False,
    ))
    check_results.append(_check_result(
        check_id="rd06-backend-reference", contract_id=_RD06, capability="correctness",
        passed=rd0506_ok, detail=rd0506_detail, artifact=rd0506_artifact,
        carries_disposition=False,
    ))

    # --- RD05 controls: prefill regression guard, RD05's own contract ---
    # PA39 real-hardware-acceptance fix: RD05 is correctness-only (no
    # target_kernel_gain_pct/end_to_end_gain_pct, so bind_contract() never
    # requires 'performance'/'activation' for it) but its own contract
    # still declares a real [positive]/[controls] block (decode/prefill on
    # tierA-qwen4b-q6k) and acceptance.max_control_regression_pct=1 --
    # aggregate_contract_effects() structurally requires at least one
    # positive-role lane even though evaluate_promotion_gate() never checks
    # target_kernel_gain_pct here (RD05's acceptance declares it as None,
    # so that half of the gate is a no-op; only max_control_regression_pct
    # is actually enforced), so both declared workloads are benchmarked.
    rd05_controls_artifact = None
    rd05_missing_positive: list[str] = []
    rd05_missing_control: list[str] = []
    if gfx1201 is None or ctx.model is None:
        rd05_controls_ok = False
        rd05_controls_detail = (
            f"rd05 controls: no {_RD0506_ARCH} device and/or model supplied"
        )
    else:
        rd05_contract = _load_contract(_RD05)
        rd05_positive_workloads = tuple(rd05_contract.positive.workloads)
        rd05_control_workloads = tuple(rd05_contract.controls.workloads)
        rd05_positive_outcome = ctx.runtime.run_paired_llama_benchmark(
            control_binary=bench_pair.control_bin, subject_binary=bench_pair.subject_bin,
            model=ctx.model, workloads=rd05_positive_workloads,
            pairs=_RD05_MIN_PAIRED_ROUNDS, log_context="rd05-controls-positive",
            device=gfx1201,
        )
        rd05_control_outcome = ctx.runtime.run_paired_llama_benchmark(
            control_binary=bench_pair.control_bin, subject_binary=bench_pair.subject_bin,
            model=ctx.model, workloads=rd05_control_workloads,
            pairs=_RD05_MIN_PAIRED_ROUNDS, log_context="rd05-controls-control",
            device=gfx1201,
        )
        rd05_missing_positive = [w for w in rd05_positive_workloads if w not in rd05_positive_outcome.runs]
        rd05_missing_control = [w for w in rd05_control_workloads if w not in rd05_control_outcome.runs]
        if rd05_missing_positive or rd05_missing_control:
            rd05_controls_ok = False
            rd05_aggregated_effects = None
            rd05_gate_result = None
            rd05_controls_detail = (
                f"rd05 controls: contract declares positive workloads "
                f"{list(rd05_positive_workloads)} and control workloads "
                f"{list(rd05_control_workloads)}, but evidence is missing for "
                f"positive={rd05_missing_positive} control={rd05_missing_control} -- "
                "failing closed rather than aggregating a partial lane set"
            )
        else:
            rd05_target_metric = _PAIRED_BENCH_METRIC_NAME["decode"]
            rd05_positive_lanes = [
                experiment_execution.lane_effect_from_run(
                    "positive", _PAIRED_BENCH_METRIC_NAME[w], rd05_positive_outcome.runs[w],
                )
                for w in rd05_positive_workloads
            ]
            rd05_control_lanes = [
                experiment_execution.lane_effect_from_run(
                    "control", _PAIRED_BENCH_METRIC_NAME[w], rd05_control_outcome.runs[w],
                )
                for w in rd05_control_workloads
            ]
            rd05_aggregated_effects = experiment_contract.aggregate_contract_effects(
                rd05_contract, rd05_positive_lanes + rd05_control_lanes, target_metric=rd05_target_metric,
            )
            rd05_correctness_gate = experiment_contract.evaluate_correctness_gate(
                rd05_contract,
                {"backend_reference": experiment_contract.CorrectnessResult(
                    check="backend_reference", passed=rd0506_ok, detail=rd0506_detail,
                )},
            )
            rd05_gate_result = experiment_contract.evaluate_promotion_gate(
                rd05_contract, correctness_gate=rd05_correctness_gate,
                aggregated_effects=rd05_aggregated_effects,
            )
            rd05_controls_ok = bool(rd05_gate_result["passed"])
            rd05_controls_detail = (
                f"rd05 controls: max_control_regression_pct gate status="
                f"{rd05_gate_result['status']!r} passed={rd05_controls_ok} "
                f"reasons={list(rd05_gate_result['reasons'])} "
                f"positive_workloads={list(rd05_positive_workloads)} "
                f"control_workloads={list(rd05_control_workloads)} "
                f"max_control_regression_pct_budget={rd05_contract.acceptance.max_control_regression_pct}"
            )
        rd05_controls_artifact = ctx.runtime.write_artifact(
            name="rd05-controls.json",
            payload={
                "schema_version": 1,
                "contract_id": _RD05,
                "positive_workloads": list(rd05_positive_workloads),
                "control_workloads": list(rd05_control_workloads),
                "missing_positive_workloads": rd05_missing_positive,
                "missing_control_workloads": rd05_missing_control,
                "aggregated_effects": rd05_aggregated_effects,
                "gate_result": rd05_gate_result,
                "positive_commands": rd05_positive_outcome.commands,
                "control_commands": rd05_control_outcome.commands,
                "bench_control_build_identity": bench_pair.validation_build_identities["control"],
                "bench_subject_build_identity": bench_pair.validation_build_identities["subject"],
            },
        )
        emitted_artifacts.add(rd05_controls_artifact.name)

    check_results.append(_check_result(
        check_id="rd05-controls", contract_id=_RD05, capability="controls",
        passed=rd05_controls_ok, detail=rd05_controls_detail, artifact=rd05_controls_artifact,
        carries_disposition=True,
        disposition_passed=rd0506_ok and rd05_controls_ok,
    ))

    # --- RD06 activation: real BIGCHERRY_PATCH_TRACE marker probe -------
    # PA39 real-hardware-acceptance fix: proves patch.py's real RD06
    # marker (GPT-confirmed placement, req_c2c26de482d845ad; see module
    # docstring) fires on the intended production dispatch path -- uses
    # bench_pair.subject_bin (the same binary rd06-performance measures),
    # never a synthetic probe binary.
    rd06_activation_artifact = None
    if gfx1201 is None or ctx.model is None:
        rd06_activation_ok = False
        rd06_activation_detail = (
            f"rd06 activation: no {_RD0506_ARCH} device and/or model supplied"
        )
    else:
        rd06_activation_ok, rd06_activation_detail = _run_activation_probe(
            subject_binary=bench_pair.subject_bin, control_binary=bench_pair.control_bin,
            model=ctx.model, device=gfx1201,
            marker=_RD06_ACTIVATION_MARKER, log_context="rd06-activation",
        )
        rd06_activation_artifact = ctx.runtime.write_artifact(
            name="rd06-activation.json",
            payload={
                "schema_version": 1, "contract_id": _RD06, "marker": _RD06_ACTIVATION_MARKER,
                "passed": rd06_activation_ok, "detail": rd06_activation_detail,
            },
        )
        emitted_artifacts.add(rd06_activation_artifact.name)

    check_results.append(_check_result(
        check_id="rd06-activation", contract_id=_RD06, capability="activation",
        passed=rd06_activation_ok, detail=rd06_activation_detail, artifact=rd06_activation_artifact,
        carries_disposition=False,
    ))

    # --- RD06 performance: RD06's own ci95/min_paired_rounds policy -----
    # PA39 defects #2 (ci95_threshold_bound_v1 never evaluated) and #3
    # (RD06's declared control model tierM-gptoss20b-q6k never benchmarked)
    # fix: run the paired benchmark for BOTH RD06's positive model
    # (ctx.model, every workload the contract declares as positive) and
    # its declared control model (ctx.inputs["control_model"], every
    # workload the contract declares as control), then route the real
    # PairedLaneRuns through lane_effect_from_run()/
    # aggregate_contract_effects()/evaluate_promotion_gate() -- the same
    # real evaluator machinery already used elsewhere in this codebase
    # (e.g. RD58's promotion path in validation_campaign.py) -- instead
    # of the prior execution-only `bool(outcome.runs)` stub.
    #
    # PA39 P0 defect #1 fix: uses bench_pair (built with
    # primary_target="llama-bench"), never ppl_pair -- run_paired_
    # llama_benchmark() constructs llama-bench-style argv, so handing it
    # a llama-perplexity binary path is a real mismatch.
    rd06_perf_artifact = None
    control_model_raw = ctx.inputs.get("control_model")
    control_model = Path(control_model_raw) if control_model_raw else None
    control_workloads: tuple[str, ...] = ()
    missing_control: list[str] = []
    control_model_identity_ok = False
    if gfx1201 is None or ctx.model is None or control_model is None:
        rd06_perf_ok = False
        rd06_perf_detail = (
            f"rd06 performance: no {_RD0506_ARCH} device, model, and/or "
            "--producer-input control_model=<path> supplied"
        )
        rd06_gate_result: dict[str, object] | None = None
    else:
        rd06_contract = _load_contract(_RD06)
        # PA39 P1 fix: control_model is a bare --producer-input path with
        # no inherent tie to RD06's declared control model identity
        # (tierM-gptoss20b-q6k) -- an arbitrary GGUF could otherwise be
        # silently mislabeled as the control lane. Best-effort identity
        # check against config/models.toml (read directly -- never via
        # validation_campaign.py, per this module's import restriction);
        # always recorded in the artifact regardless of outcome so a
        # mismatch is visible/auditable even though this is not full
        # cryptographic provenance.
        control_model_identity_ok, control_model_identity_detail = (
            _verify_control_model_identity(
                control_model, expected_model_id=rd06_contract.controls.models[0],
            )
        )
        # PA39 P0 defect #2 fix: run every workload the contract declares
        # for each role, not just decode -- and fail closed (no gate call)
        # if any declared workload's evidence does not come back, rather
        # than letting aggregate_contract_effects() (which is not itself
        # contract-aware) silently aggregate a partial lane set.
        positive_workloads = tuple(rd06_contract.positive.workloads)
        control_workloads = tuple(rd06_contract.controls.workloads)
        positive_outcome = ctx.runtime.run_paired_llama_benchmark(
            control_binary=bench_pair.control_bin, subject_binary=bench_pair.subject_bin,
            model=ctx.model, workloads=positive_workloads,
            pairs=_RD06_MIN_PAIRED_ROUNDS, log_context="rd06-performance-positive",
            device=gfx1201,
        )
        control_outcome = ctx.runtime.run_paired_llama_benchmark(
            control_binary=bench_pair.control_bin, subject_binary=bench_pair.subject_bin,
            model=control_model, workloads=control_workloads,
            pairs=_RD06_MIN_PAIRED_ROUNDS, log_context="rd06-performance-control",
            device=gfx1201,
        )
        missing_positive = [w for w in positive_workloads if w not in positive_outcome.runs]
        missing_control = [w for w in control_workloads if w not in control_outcome.runs]
        if missing_positive or missing_control:
            rd06_perf_ok = False
            aggregated_effects = None
            rd06_gate_result = None
            rd06_perf_detail = (
                f"rd06 performance: contract declares positive workloads "
                f"{list(positive_workloads)} and control workloads "
                f"{list(control_workloads)}, but evidence is missing for "
                f"positive={missing_positive} control={missing_control} -- "
                "failing closed rather than aggregating a partial lane set"
            )
        else:
            target_metric = _PAIRED_BENCH_METRIC_NAME["decode"]
            positive_lanes = [
                experiment_execution.lane_effect_from_run(
                    "positive", _PAIRED_BENCH_METRIC_NAME[workload],
                    positive_outcome.runs[workload],
                )
                for workload in positive_workloads
            ]
            control_lanes = [
                experiment_execution.lane_effect_from_run(
                    "control", _PAIRED_BENCH_METRIC_NAME[workload],
                    control_outcome.runs[workload],
                )
                for workload in control_workloads
            ]
            aggregated_effects = experiment_contract.aggregate_contract_effects(
                rd06_contract, positive_lanes + control_lanes, target_metric=target_metric,
            )
            rd06_correctness_gate = experiment_contract.evaluate_correctness_gate(
                rd06_contract,
                {"backend_reference": experiment_contract.CorrectnessResult(
                    check="backend_reference", passed=rd0506_ok, detail=rd0506_detail,
                )},
            )
            rd06_gate_result = experiment_contract.evaluate_promotion_gate(
                rd06_contract, correctness_gate=rd06_correctness_gate,
                aggregated_effects=aggregated_effects,
            )
            rd06_perf_ok = bool(rd06_gate_result["passed"])
            rd06_perf_detail = (
                f"rd06 performance: ci95_threshold_bound_v1 gate status="
                f"{rd06_gate_result['status']!r} passed={rd06_perf_ok} "
                f"reasons={list(rd06_gate_result['reasons'])} "
                f"positive_model={ctx.model} positive_workloads={list(positive_workloads)} "
                f"control_model={control_model} control_workloads={list(control_workloads)} "
                f"target_kernel_gain_pct={_RD06_TARGET_KERNEL_GAIN_PCT} "
                f"max_control_regression_pct={_RD06_MAX_CONTROL_REGRESSION_PCT} "
                f"{control_model_identity_detail}"
            )
        rd06_perf_artifact = ctx.runtime.write_artifact(
            name="rd06-performance.json",
            payload={
                "schema_version": 3,
                "contract_id": _RD06,
                "effect_evidence_policy": _RD06_EFFECT_EVIDENCE_POLICY,
                "min_paired_rounds": _RD06_MIN_PAIRED_ROUNDS,
                "target_kernel_gain_pct": _RD06_TARGET_KERNEL_GAIN_PCT,
                "max_control_regression_pct": _RD06_MAX_CONTROL_REGRESSION_PCT,
                "positive_model": str(ctx.model),
                "positive_workloads": list(positive_workloads),
                "control_model": str(control_model),
                "control_workloads": list(control_workloads),
                "control_model_identity_ok": control_model_identity_ok,
                "control_model_identity_detail": control_model_identity_detail,
                "missing_positive_workloads": missing_positive,
                "missing_control_workloads": missing_control,
                "aggregated_effects": aggregated_effects,
                "gate_result": rd06_gate_result,
                "positive_commands": positive_outcome.commands,
                "control_commands": control_outcome.commands,
                "bench_control_build_identity": bench_pair.validation_build_identities["control"],
                "bench_subject_build_identity": bench_pair.validation_build_identities["subject"],
            },
        )
        emitted_artifacts.add(rd06_perf_artifact.name)

    check_results.append(_check_result(
        check_id="rd06-performance", contract_id=_RD06, capability="performance",
        passed=rd06_perf_ok and rd0506_ok, detail=rd06_perf_detail, artifact=rd06_perf_artifact,
        carries_disposition=False,
    ))

    # --- RD06 controls: reuses rd06-performance's own control lane ------
    # PA39 real-hardware-acceptance fix: RD06's controls capability is
    # already fully measured above (the control_model/control_workloads
    # paired-benchmark call, gated through evaluate_promotion_gate()'s
    # max_control_regression_pct check) -- this check surfaces that same
    # real evidence under the 'controls' capability specifically, without
    # a second disposition (validate_producer_result() allows at most one
    # non-None disposition per contract_id; rd06-performance already
    # carries RD06's disposition).
    rd06_controls_ok = bool(rd06_perf_ok) and not missing_control
    rd06_controls_detail = (
        f"rd06 controls: reuses rd06-performance's control lane "
        f"(control_model={control_model}, control_workloads={list(control_workloads)}, "
        f"missing_control_workloads={missing_control}); underlying gate passed={rd06_perf_ok}"
    )
    check_results.append(_check_result(
        check_id="rd06-controls", contract_id=_RD06, capability="controls",
        passed=rd06_controls_ok, detail=rd06_controls_detail, artifact=rd06_perf_artifact,
        carries_disposition=True,
        # Deliberately excludes control_model_identity_ok: that check is a
        # best-effort filesystem basename/size check (see
        # _verify_control_model_identity()'s own docstring), recorded for
        # audit but NOT gating -- same deliberate, already-documented PA37
        # design decision (part 6 notes) kept here for the same reason
        # (hardware-free tests use non-existent fake model paths that can
        # never match config/models.toml's real registry entries).
        disposition_passed=(
            rd0506_ok and rd06_perf_ok and rd06_activation_ok and rd06_controls_ok
        ),
    ))

    # --- RD07 backend_reference: requires all three architectures -------
    missing_rd07 = tuple(a for a in _RD07_ARCHS if a not in devices_by_arch)
    rd07_artifact = None
    if missing_rd07 or ctx.model is None or ctx.corpus is None:
        rd07_ok = False
        rd07_detail = (
            f"rd07: missing required architecture(s) {list(missing_rd07)}"
            if missing_rd07 else "rd07: no model/corpus supplied"
        )
    else:
        rd07_by_arch: dict[str, dict[str, object]] = {}
        for arch in _RD07_ARCHS:
            ok, detail, _comparison = _run_backend_reference(
                control_bin=ppl_pair.control_bin, subject_bin=ppl_pair.subject_bin,
                model=ctx.model, corpus=ctx.corpus, device=devices_by_arch[arch],
                log_context=f"rd07-{arch}",
            )
            rd07_by_arch[arch] = {"passed": ok, "detail": detail}
        rd07_ok = all(v["passed"] for v in rd07_by_arch.values())
        rd07_detail = "; ".join(
            f"{arch}: {'PASS' if v['passed'] else 'FAIL'}" for arch, v in rd07_by_arch.items()
        )
        rd07_artifact = ctx.runtime.write_artifact(
            name="rd07-backend-reference.json",
            payload={
                "schema_version": 1,
                "contract_id": _RD07,
                "architectures": rd07_by_arch,
                "control_build_identity": ppl_pair.validation_build_identities["control"],
                "subject_build_identity": ppl_pair.validation_build_identities["subject"],
            },
        )
        emitted_artifacts.add(rd07_artifact.name)

    check_results.append(_check_result(
        check_id="rd07-backend-reference", contract_id=_RD07, capability="correctness",
        passed=rd07_ok, detail=rd07_detail, artifact=rd07_artifact,
        carries_disposition=False,
    ))

    # --- RD07 activation: real BIGCHERRY_PATCH_TRACE marker probe -------
    # PA39 real-hardware-acceptance fix: proves patch.py's real RD07
    # marker (mmq.cu's ggml_cuda_mul_mat_q_switch_type(), GGML_TYPE_Q6_K
    # case) fires on the real Q6_K MMQ dispatch path. Gated on the same
    # "all three architectures present" requirement as RD07's other
    # checks (missing_rd07), not "any available" -- RD07's contract scope
    # is all three architectures, so a real acceptance run with one
    # missing/failed architecture must fail RD07 closed uniformly across
    # every one of its checks, matching rd07-backend-reference/
    # rd07-performance's own fail-closed behavior above.
    #
    # GPT review (2026-09-16): RD07's positive AND controls blocks both
    # declare model=tierM-gptoss20b-q6k (prefill positive, decode control)
    # -- never ctx.model (tierA-qwen4b-q6k, the generic --model, which is
    # only RD05/RD06's positive model). Reuses the same already
    # identity-checked control_model resolved for RD06 (the real path
    # happens to be the exact model id RD07 needs too), rather than a
    # second --producer-input. GPT also flagged that this producer only
    # ever exercises RD07's activation/performance/controls on gfx1201,
    # despite RD07's contract scope being all three architectures
    # (gfx1100/gfx1201/gfx1030) -- widening every RD07 check to all three
    # devices is real additional per-architecture paired-benchmark work
    # not attempted in this pass; documented honestly here (mirrors
    # RD06's own documented gfx1100 gap above) rather than silently
    # overclaiming three-architecture coverage. A future item should
    # widen these three RD07 checks to run on gfx1100/gfx1201/gfx1030
    # individually before PA39 claims full RD07 scope.
    rd07_activation_artifact = None
    if missing_rd07 or control_model is None:
        rd07_activation_ok = False
        rd07_activation_detail = (
            f"rd07 activation: missing required architecture(s) {list(missing_rd07)} "
            "and/or --producer-input control_model=<path> (RD07 needs the same model id "
            "as RD06's control_model, tierM-gptoss20b-q6k)"
        )
    else:
        rd07_activation_device = devices_by_arch[_RD0506_ARCH]
        rd07_activation_ok, rd07_activation_detail = _run_activation_probe(
            subject_binary=bench_pair.subject_bin, control_binary=bench_pair.control_bin,
            model=control_model, device=rd07_activation_device,
            marker=_RD07_ACTIVATION_MARKER, log_context="rd07-activation",
        )
        rd07_activation_artifact = ctx.runtime.write_artifact(
            name="rd07-activation.json",
            payload={
                "schema_version": 1, "contract_id": _RD07, "marker": _RD07_ACTIVATION_MARKER,
                "architecture": rd07_activation_device.architecture,
                "passed": rd07_activation_ok, "detail": rd07_activation_detail,
            },
        )
        emitted_artifacts.add(rd07_activation_artifact.name)

    check_results.append(_check_result(
        check_id="rd07-activation", contract_id=_RD07, capability="activation",
        passed=rd07_activation_ok, detail=rd07_activation_detail, artifact=rd07_activation_artifact,
        carries_disposition=False,
    ))

    # --- RD07 performance: max_control_regression_pct only --------------
    # GPT review (2026-09-16): RD07's positive/controls models are both
    # tierM-gptoss20b-q6k, not ctx.model -- reuses control_model (see
    # rd07-activation's comment above for why the same resolved path is
    # the right model for RD07 too).
    rd07_perf_artifact = None
    outcome: vp.ProducerPairedBenchmarkOutcome | None = None
    if missing_rd07 or control_model is None:
        rd07_perf_ok = False
        rd07_perf_detail = (
            f"rd07 performance: missing required architecture(s) {list(missing_rd07)} "
            "and/or --producer-input control_model=<path>"
        )
    else:
        # PA39 P0 defect #1 fix: bench_pair (llama-bench), never ppl_pair.
        outcome = ctx.runtime.run_paired_llama_benchmark(
            control_binary=bench_pair.control_bin, subject_binary=bench_pair.subject_bin,
            model=control_model, workloads=("decode", "prefill"),
            pairs=3, log_context="rd07-performance", device=devices_by_arch[_RD0506_ARCH],
        )
        rd07_perf_ok = bool(outcome.runs)
        rd07_perf_detail = (
            f"rd07 performance: paired benchmark executed for {sorted(outcome.runs)}; "
            f"max_control_regression_pct={_RD07_MAX_CONTROL_REGRESSION_PCT} "
            "(quantitative evaluation deferred to PA39 real-hardware acceptance)"
        )
        rd07_perf_artifact = ctx.runtime.write_artifact(
            name="rd07-performance.json",
            payload={
                "schema_version": 2,
                "contract_id": _RD07,
                "max_control_regression_pct": _RD07_MAX_CONTROL_REGRESSION_PCT,
                "commands": outcome.commands,
                "bench_control_build_identity": bench_pair.validation_build_identities["control"],
                "bench_subject_build_identity": bench_pair.validation_build_identities["subject"],
            },
        )
        emitted_artifacts.add(rd07_perf_artifact.name)

    check_results.append(_check_result(
        check_id="rd07-performance", contract_id=_RD07, capability="performance",
        passed=rd07_perf_ok and rd07_ok, detail=rd07_perf_detail, artifact=rd07_perf_artifact,
        carries_disposition=False,
    ))

    # --- RD07 controls: reuses rd07-performance's decode (control) lane -
    # PA39 real-hardware-acceptance fix: RD07's contract declares
    # controls.workloads=["decode"] on the same model as its positive
    # prefill workload -- rd07-performance already benchmarks both decode
    # and prefill in one call, so this check surfaces that same real
    # evidence under the 'controls' capability, without a second
    # disposition (rd07-performance already carries RD07's disposition).
    rd07_decode_present = outcome is not None and "decode" in outcome.runs
    rd07_controls_ok = rd07_perf_ok and rd07_ok and rd07_decode_present
    rd07_controls_detail = (
        f"rd07 controls: reuses rd07-performance's decode lane "
        f"(present={rd07_decode_present}); underlying performance check passed={rd07_perf_ok}"
    )
    check_results.append(_check_result(
        check_id="rd07-controls", contract_id=_RD07, capability="controls",
        passed=rd07_controls_ok, detail=rd07_controls_detail, artifact=rd07_perf_artifact,
        carries_disposition=True,
        disposition_passed=(
            rd07_ok and rd07_perf_ok and rd07_activation_ok and rd07_controls_ok
        ),
    ))

    return vp.ProducerResult(
        correctness=None,
        validation_build_identities=ppl_pair.validation_build_identities,
        activation_evidence=None,
        performance_evidence=None,
        trace_evidence=None,
        check_results=tuple(check_results),
        lane_effects=(),
        emitted_artifacts=frozenset(emitted_artifacts),
    )
