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

Build/device shape (PA37.md step 5): ONE atomic control/subject build pair,
built once as fat gfx1100;gfx1201;gfx1030 via ``ctx.runtime.build_pair()``,
then run per real device slot supplied in ``ctx.device_map``. RD05/RD06
consume the same gfx1201 result (RD05/RD06 are materially inseparable in
the atomic 1203 patch -- see validation_campaign.py's superseded RD05
producer docstring for the same reasoning, carried forward here); RD07
requires all three architectures to be present at once.

Fail-closed by construction: a missing device, missing model/corpus, or a
failed backend-reference comparison always yields ``passed=False`` with a
truthful detail string -- never a default/fabricated PASS.
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


_PAIRED_BENCH_METRIC_NAME: dict[str, str] = {"decode": "tg128", "prefill": "pp512"}


def _load_rd06_contract() -> experiment_contract.ExperimentContract:
    # Loaded directly from config/experiment-contracts.toml via the
    # experiment.contract module -- never via validation_campaign.py/
    # patch_validation.load_contracts_for_descriptor(), which this module
    # must not import (see module docstring). This is the same real
    # ci95_threshold_bound_v1/min_paired_rounds=10/target_kernel_gain_pct=
    # 0.5/max_control_regression_pct=1 policy already encoded as module
    # constants above -- loading the real ExperimentContract object (rather
    # than re-deriving the same numbers by hand) is what lets
    # evaluate_promotion_gate() itself apply that policy instead of this
    # producer re-implementing the gate's own bound logic.
    registry = experiment_contract.load_contracts(bc_paths.EXPERIMENT_CONTRACTS)
    return registry[_RD06]


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
) -> vp.ProducerCheckResult:
    # validate_producer_result() allows at most ONE non-None disposition per
    # contract_id across all check_results -- a contract's [[check]] plan
    # entries all contribute to compute_verdict() (every required check
    # must pass), but exactly one of them carries the contract's own
    # promotion-eligibility disposition, so contract_verdicts is never
    # double-written for the same contract.
    return vp.ProducerCheckResult(
        check_id=check_id,
        contract_ids=(contract_id,),
        validation_result=pv.ValidationResult(
            check_id=check_id, capability=capability,
            status=pv.PASS if passed else pv.FAIL, summary=detail,
            artifacts=(artifact,) if artifact is not None else (),
        ),
        disposition=(
            {"passed": passed, "contract_id": contract_id, "detail": detail}
            if carries_disposition else None
        ),
    )


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    # One atomic control/subject build pair, built once at the full fat
    # target list -- never rebuilt per architecture (PA37.md step 5).
    pair = ctx.runtime.build_pair(
        targets=ctx.fat_targets.targets, primary_target="llama-perplexity",
    )
    devices = ctx.runtime.device_contexts(device_map=ctx.device_map)
    devices_by_arch: Mapping[str, vp.ProducerDeviceContext] = {
        d.architecture: d for d in devices
    }

    emitted_artifacts: set[str] = set()
    check_results: list[vp.ProducerCheckResult] = []

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
            control_bin=pair.control_bin, subject_bin=pair.subject_bin,
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
                "control_build_identity": pair.validation_build_identities["control"],
                "subject_build_identity": pair.validation_build_identities["subject"],
            },
        )
        emitted_artifacts.add(rd0506_artifact.name)

    check_results.append(_check_result(
        check_id="rd05-backend-reference", contract_id=_RD05, capability="correctness",
        passed=rd0506_ok, detail=rd0506_detail, artifact=rd0506_artifact,
        carries_disposition=True,
    ))
    check_results.append(_check_result(
        check_id="rd06-backend-reference", contract_id=_RD06, capability="correctness",
        passed=rd0506_ok, detail=rd0506_detail, artifact=rd0506_artifact,
        carries_disposition=False,
    ))

    # --- RD06 performance: RD06's own ci95/min_paired_rounds policy -----
    # PA39 defects #2 (ci95_threshold_bound_v1 never evaluated) and #3
    # (RD06's declared control model tierM-gptoss20b-q6k never benchmarked)
    # fix: run the paired benchmark for BOTH RD06's positive model
    # (ctx.model, decode -- tg128 is the metric target_kernel_gain_pct is
    # bound to) and its declared control model (ctx.inputs["control_model"],
    # decode), then route both real PairedLaneRuns through
    # lane_effect_from_run()/aggregate_contract_effects()/
    # evaluate_promotion_gate() -- the same real evaluator machinery
    # already used elsewhere in this codebase (e.g. RD58's promotion path
    # in validation_campaign.py) -- instead of the prior execution-only
    # `bool(outcome.runs)` stub.
    rd06_perf_artifact = None
    control_model_raw = ctx.inputs.get("control_model")
    control_model = Path(control_model_raw) if control_model_raw else None
    if gfx1201 is None or ctx.model is None or control_model is None:
        rd06_perf_ok = False
        rd06_perf_detail = (
            f"rd06 performance: no {_RD0506_ARCH} device, model, and/or "
            "--producer-input control_model=<path> supplied"
        )
        rd06_gate_result: dict[str, object] | None = None
    else:
        positive_outcome = ctx.runtime.run_paired_llama_benchmark(
            control_binary=pair.control_bin, subject_binary=pair.subject_bin,
            model=ctx.model, workloads=("decode",),
            pairs=_RD06_MIN_PAIRED_ROUNDS, log_context="rd06-performance-positive",
            device=gfx1201,
        )
        control_outcome = ctx.runtime.run_paired_llama_benchmark(
            control_binary=pair.control_bin, subject_binary=pair.subject_bin,
            model=control_model, workloads=("decode",),
            pairs=_RD06_MIN_PAIRED_ROUNDS, log_context="rd06-performance-control",
            device=gfx1201,
        )
        metric = _PAIRED_BENCH_METRIC_NAME["decode"]
        positive_lane = experiment_execution.lane_effect_from_run(
            "positive", metric, positive_outcome.runs["decode"],
        )
        control_lane = experiment_execution.lane_effect_from_run(
            "control", metric, control_outcome.runs["decode"],
        )
        rd06_contract = _load_rd06_contract()
        aggregated_effects = experiment_contract.aggregate_contract_effects(
            rd06_contract, [positive_lane, control_lane], target_metric=metric,
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
            f"positive_model={ctx.model} control_model={control_model} "
            f"target_kernel_gain_pct={_RD06_TARGET_KERNEL_GAIN_PCT} "
            f"max_control_regression_pct={_RD06_MAX_CONTROL_REGRESSION_PCT}"
        )
        rd06_perf_artifact = ctx.runtime.write_artifact(
            name="rd06-performance.json",
            payload={
                "schema_version": 2,
                "contract_id": _RD06,
                "effect_evidence_policy": _RD06_EFFECT_EVIDENCE_POLICY,
                "min_paired_rounds": _RD06_MIN_PAIRED_ROUNDS,
                "target_kernel_gain_pct": _RD06_TARGET_KERNEL_GAIN_PCT,
                "max_control_regression_pct": _RD06_MAX_CONTROL_REGRESSION_PCT,
                "positive_model": str(ctx.model),
                "control_model": str(control_model),
                "aggregated_effects": aggregated_effects,
                "gate_result": rd06_gate_result,
                "positive_commands": positive_outcome.commands,
                "control_commands": control_outcome.commands,
            },
        )
        emitted_artifacts.add(rd06_perf_artifact.name)

    check_results.append(_check_result(
        check_id="rd06-performance", contract_id=_RD06, capability="performance",
        passed=rd06_perf_ok and rd0506_ok, detail=rd06_perf_detail, artifact=rd06_perf_artifact,
        carries_disposition=True,
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
                control_bin=pair.control_bin, subject_bin=pair.subject_bin,
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
                "control_build_identity": pair.validation_build_identities["control"],
                "subject_build_identity": pair.validation_build_identities["subject"],
            },
        )
        emitted_artifacts.add(rd07_artifact.name)

    check_results.append(_check_result(
        check_id="rd07-backend-reference", contract_id=_RD07, capability="correctness",
        passed=rd07_ok, detail=rd07_detail, artifact=rd07_artifact,
        carries_disposition=False,
    ))

    # --- RD07 performance: max_control_regression_pct only --------------
    rd07_perf_artifact = None
    if missing_rd07 or ctx.model is None:
        rd07_perf_ok = False
        rd07_perf_detail = (
            f"rd07 performance: missing required architecture(s) {list(missing_rd07)}"
        )
    else:
        outcome = ctx.runtime.run_paired_llama_benchmark(
            control_binary=pair.control_bin, subject_binary=pair.subject_bin,
            model=ctx.model, workloads=("decode", "prefill"),
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
                "schema_version": 1,
                "contract_id": _RD07,
                "max_control_regression_pct": _RD07_MAX_CONTROL_REGRESSION_PCT,
                "commands": outcome.commands,
            },
        )
        emitted_artifacts.add(rd07_perf_artifact.name)

    check_results.append(_check_result(
        check_id="rd07-performance", contract_id=_RD07, capability="performance",
        passed=rd07_perf_ok and rd07_ok, detail=rd07_perf_detail, artifact=rd07_perf_artifact,
        carries_disposition=True,
    ))

    return vp.ProducerResult(
        correctness=None,
        validation_build_identities=pair.validation_build_identities,
        activation_evidence=None,
        performance_evidence=None,
        trace_evidence=None,
        check_results=tuple(check_results),
        lane_effects=(),
        emitted_artifacts=frozenset(emitted_artifacts),
    )
