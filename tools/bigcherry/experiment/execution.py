"""VA14: the real per-contract-lane execution primitive that was missing.

GPT design review (session ses_5bbee8ce5c9a4265, req_2072dae840434295)
confirmed no real per-contract-lane executor exists yet -- the docstring
references in contract.py to campaign_planner.expand_contract()/
comparisons.run_comparison() describe intended architecture, not real
code. This module is the real thing, scoped to the smallest slice GPT
identified: a neutral paired control/subject lane runner reusing
campaign/benchmark.py's already-neutral statistics primitive
(block_bootstrap_effect()), NOT its ab-benchmark executor (which is
specifically native<->replay dispatch-mode comparison -- answers tune
effectiveness, not patch effectiveness).

Deliberately does NOT reuse tune-campaign's S6 stock/native/replay
build/measurement machinery: VA14's control/subject binaries must be
built with matching options apart from the patch itself (validation
build-parity), never a comparison against tune/replay artifacts that
carry unrelated autotune instrumentation (GGML_HIP_AUTOTUNE,
AUTOTUNE_RECORD, ROUTING_TRANSFORM) the control build never had -- that
would confound the measured effect with instrumentation overhead, not
just the patch.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

from . import attestation
from . import contract as experiment_contract
from ..campaign.benchmark import block_bootstrap_effect, extract_metrics

# VA14: canonical workload -> llama-bench metric name mapping. Contracts
# and validation.toml both intentionally do not bind a metric identifier
# (guide rule: no thresholds/metric-binding outside the Experiment
# Contract's own acceptance fields) -- something has to translate a
# workload TAG into what argument/regex actually measures it, and that
# translation lives here, on the execution side, not fabricated by
# silently reusing an unrelated tune-campaign result. Deliberately closed
# and minimal: extend only when a new workload is actually being
# validated, never speculatively for every WORKLOAD_TAGS value.
WORKLOAD_METRIC: dict[str, str] = {
    "decode": "tg128",
    "prefill": "pp512",
    # RD73/VA06: mtp_wall_tps is the CLIENT-measured, real request-to-
    # response wall-clock throughput from validation_campaign.py's
    # run_rd73_mtp_server_lane() (bench/server_completion.py's run_request()
    # wall_tps field) -- deliberately not the server's own self-reported
    # predicted_tps, which can exclude HTTP/queueing overhead. Registered
    # only once that adapter existed and was proven with real tests (GPT
    # scoping, session ses_1e0bd1ea53db4311).
    "mtp_verify": "mtp_wall_tps",
}


class DeviceVisibilityError(ValueError):
    """Raised by require_device_visibility() on any unsafe device-selector
    state. A ValueError subclass, not LaneExecutionError -- this fires
    before any process launch, not during/after one."""


@dataclass(frozen=True)
class DeviceVisibility:
    """A validated, ordered device-SELECTOR declaration -- the explicit
    HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES token pool a paired-benchmark
    cell was launched against. PVPS02 (2026-09-11, dev-gpt-agent design +
    self-critique, docs/planning/active/
    patching-validation-package-standard/PVPS02.md): extracted from
    run_rd58_state_restore_evidence()'s original inline HIP_VISIBLE_DEVICES/
    ROCR_VISIBLE_DEVICES guard so every paired-benchmark path can share one
    fail-closed contract instead of reimplementing it per patch.

    GPT review correction (req_e608313764834497, 2026-09-11): this proves
    LAUNCH INTENT (which selector tokens were passed to the child process),
    not physical-device identity -- the runtime's own ROCm attestation
    (parse_rocm_attestation()) never carries a device locator, so nothing
    downstream of this class can confirm which physical card a given
    selector token actually resolved to. Earlier wording here claimed more
    than that; corrected.

    device_ids is ORDERED and that order is load-bearing: an N-device cell
    consumes the first N ids, so e.g. "1,0" deliberately differs from "0,1"
    -- for a tensor-split cell this determines DECLARED SELECTOR ORDER
    (which token becomes tensor-split slot 0 vs 1), not a proven physical
    mapping."""

    device_ids: tuple[str, ...]
    hip_visible_devices: str
    rocr_visible_devices: str

    @property
    def gpu_count(self) -> int:
        return len(self.device_ids)

    def document(self) -> dict[str, object]:
        """The same observed_devices shape run_rd58_state_restore_evidence()
        already records today -- kept identical so a future caller can
        adopt this primitive without changing evidence schema."""
        return {
            "hip_visible_devices": list(self.device_ids),
            "rocr_visible_devices": list(self.device_ids),
            "gpu_count": self.gpu_count,
        }


def require_device_visibility(
    *,
    context: str,
    env: Mapping[str, str] | None = None,
    exact_count: int | None = None,
    minimum_count: int = 1,
) -> DeviceVisibility:
    """Fail closed before any hardware use unless HIP_VISIBLE_DEVICES and
    ROCR_VISIBLE_DEVICES are both explicitly set, consistent with each
    other, free of duplicates, and declare enough distinct selector tokens
    (not independently verified against physically-present hardware --
    see DeviceVisibility's own docstring for what this class can and
    cannot prove).

    PVPS02 step 1 (see DeviceVisibility's own docstring): a standalone,
    unit-tested extraction only. Nothing calls this yet -- RD58's own
    inline guard and RD73's lane functions are deliberately NOT wired to
    it in this change; wiring them is its own later, dedicated,
    regression-tested step (several existing tests depend on RD58's
    producer function staying callable WITHOUT selector validation, and
    RD73's hardware-free lane tests call functions without HIP/ROCR setup
    at all -- rewiring either now would break currently-passing tests for
    no functional gain yet).

    This intentionally goes stricter than RD58's original inline check:
    it also rejects a blank entry inside the list (e.g. "0,,1" or a
    trailing comma), which the original split-and-filter silently
    tolerated.
    """
    source = os.environ if env is None else env

    hip_raw = source.get("HIP_VISIBLE_DEVICES")
    rocr_raw = source.get("ROCR_VISIBLE_DEVICES")
    if not hip_raw or not rocr_raw:
        raise DeviceVisibilityError(
            f"{context}: HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES must "
            "both be explicitly set (an unset/ambient-default device list "
            "cannot be trusted for a real-hardware measurement)"
        )

    def _parse(name: str, raw: str) -> tuple[str, ...]:
        parts = raw.split(",")
        if any(not part.strip() for part in parts):
            raise DeviceVisibilityError(
                f"{context}: {name}={raw!r} contains a blank device id "
                "(e.g. a stray comma) -- every entry must be a non-blank device selector token"
            )
        return tuple(part.strip() for part in parts)

    hip_ids = _parse("HIP_VISIBLE_DEVICES", hip_raw)
    rocr_ids = _parse("ROCR_VISIBLE_DEVICES", rocr_raw)

    if hip_ids != rocr_ids:
        raise DeviceVisibilityError(
            f"{context}: HIP_VISIBLE_DEVICES ({hip_raw!r}) and "
            f"ROCR_VISIBLE_DEVICES ({rocr_raw!r}) must match exactly, "
            "including order"
        )
    if len(set(hip_ids)) != len(hip_ids):
        raise DeviceVisibilityError(
            f"{context}: HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES "
            f"({hip_raw!r}) contains duplicate device ids -- this does not "
            "declare distinct device selector tokens"
        )
    if len(hip_ids) < minimum_count:
        raise DeviceVisibilityError(
            f"{context}: requires {minimum_count}+ GPUs; "
            f"HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES expose only "
            f"{len(hip_ids)} ({hip_raw!r})"
        )
    if exact_count is not None and len(hip_ids) != exact_count:
        raise DeviceVisibilityError(
            f"{context}: requires exactly {exact_count} GPU(s); "
            f"HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES expose "
            f"{len(hip_ids)} ({hip_raw!r})"
        )

    return DeviceVisibility(
        device_ids=hip_ids, hip_visible_devices=hip_raw, rocr_visible_devices=rocr_raw,
    )


def metric_for_workload(workload: str) -> str:
    try:
        return WORKLOAD_METRIC[workload]
    except KeyError:
        raise experiment_contract.ExperimentContractError(
            f"no canonical metric mapping for workload {workload!r} -- add one to "
            "WORKLOAD_METRIC only once a real lane for it is being executed, never "
            "speculatively"
        ) from None


@dataclass(frozen=True)
class RunnerOutput:
    """A completed arm execution's structured result -- GPT round 2
    (req_240634997c1a4ee9): the runner must expose returncode so a failed
    benchmark that happens to print a parseable metric can never become
    valid LaneEffect evidence. Real callers construct this from
    subprocess.CompletedProcess; fake/injected runners in tests construct
    it directly."""
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined(self) -> str:
        return self.stdout + "\n" + self.stderr


def _default_runner(command: list[str]) -> RunnerOutput:
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False,
    )
    return RunnerOutput(
        returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr,
    )


Runner = Callable[[list[str]], RunnerOutput]


class LaneExecutionError(RuntimeError):
    """A control/subject arm execution failed (nonzero exit) -- fail
    closed, never salvage a parseable metric from a failed run."""


@dataclass(frozen=True)
class PairedLaneRun:
    """One real paired control/subject execution's raw result, kept for
    provenance -- callers may want to persist these logs alongside the
    computed LaneEffect."""
    runs: tuple[dict[str, object], ...]
    stats: dict[str, object]


def run_paired_lane(
    *, metric: str, control_command: list[str], subject_command: list[str],
    pattern: re.Pattern[str], pairs: int = 3, runner: Runner | None = None,
    lower_is_better: bool = False, seed: int = 0, resamples: int = 10_000,
    execution_identity: "attestation.ExecutionIdentity | None" = None,
) -> PairedLaneRun:
    """Run ``pairs`` alternating control/subject rounds of the SAME
    command shape (only the binary differs -- control vs. subject build),
    reusing block_bootstrap_effect() (already arm-name-neutral) for the
    paired geometric effect + deterministic bootstrap CI. ``runner`` is
    injectable so this is fully unit-testable without real hardware --
    defaults to a real subprocess.run. Alternates starting order each
    round (matching ab_benchmark.py's own precedent) so thermal/clock
    drift cannot quietly become a control/subject result. A nonzero
    return code from either arm raises LaneExecutionError immediately --
    never salvages a parseable metric from a failed run (GPT round 2,
    req_240634997c1a4ee9)."""
    if pairs < 1:
        raise ValueError("pairs must be >= 1")
    active_runner = runner or _default_runner
    runs: list[dict[str, object]] = []
    for pair in range(pairs):
        order = ("control", "subject") if pair % 2 == 0 else ("subject", "control")
        for mode in order:
            command = control_command if mode == "control" else subject_command
            result = active_runner(command)
            if result.returncode != 0:
                raise LaneExecutionError(
                    f"{mode} arm exited {result.returncode} (pair {pair}): "
                    f"command={command!r}; stderr={result.stderr[-500:]!r}"
                )
            # VA25: attest EVERY measured process, before its metrics are
            # accepted. Not a single preflight -- each arm is a separate
            # process, and one preflight cannot prove the ones that follow.
            # A ROCm init failure does not stop llama.cpp; it falls back to
            # CPU and still prints a well-formed table labelled "ROCm", so a
            # clean exit code and a parseable metric are not evidence that
            # anything ran on a GPU.
            if execution_identity is not None:
                attestation.require_execution_identity(
                    execution_identity,
                    attestation.parse_rocm_attestation(result.combined),
                    context=f"{mode} arm, pair {pair}",
                )
            metrics = extract_metrics(result.combined, {metric: pattern})
            runs.append({"pair": pair, "mode": mode, "metrics": metrics})
    stats = block_bootstrap_effect(
        runs, candidate="subject", reference="control", metric=metric,
        lower_is_better=lower_is_better, seed=seed, resamples=resamples,
    )
    return PairedLaneRun(runs=tuple(runs), stats=stats)


def lane_effect_from_run(role: str, metric: str, run: PairedLaneRun) -> experiment_contract.LaneEffect:
    """VA24: carry the interval and the paired-round count through.

    block_bootstrap_effect() already produces ci95_low_pct/ci95_high_pct
    alongside the point estimate; this function used to drop them, so the
    promotion gate could only ever see a point estimate. Under
    ci95_threshold_bound_v1 the gate needs the interval, and needs the round
    count because run_paired_lane() accepts pairs=1 -- whose bootstrap yields
    a degenerate interval that can look arbitrarily significant.

    Values are read straight from the producing report; nothing is derived or
    defaulted to a plausible number. A stats block lacking an interval yields
    None, which the gate treats as unevaluable ("invalid") rather than
    passing.
    """
    # VA24 P0 (dev-gpt-agent, req_d563bd481bcf4324): take paired_rounds ONLY
    # from the bootstrap's own stats. block_bootstrap_effect() derives it from
    # COMPLETE candidate/reference pairs that actually contained the metric
    # (benchmark.py: "paired_rounds": len(ratios)).
    #
    # An earlier revision fell back to counting distinct `pair` values in
    # run.runs when stats lacked the field. That was wrong: it counts pairs
    # that were incomplete or missing the metric, so it can OVERCOUNT the
    # usable evidence and let a lane clear a rounds floor it did not really
    # meet. A missing value must stay None, which an interval policy with a
    # floor then treats as unevaluable ("invalid") rather than sufficient.
    paired_rounds = run.stats.get("paired_rounds")
    if not isinstance(paired_rounds, int) or isinstance(paired_rounds, bool):
        paired_rounds = None
    return experiment_contract.LaneEffect(
        role=role, metric=metric, geometric_effect_pct=run.stats["geometric_effect_pct"],
        ci95_low_pct=run.stats.get("ci95_low_pct"),
        ci95_high_pct=run.stats.get("ci95_high_pct"),
        paired_rounds=paired_rounds,
        # RV99: carry the ratio vector through. block_bootstrap_effect()
        # returns it and LaneEffect has always had the field, but this
        # converter -- the only thing that turns a real paired run into a
        # LaneEffect -- silently dropped it, so every lane effect the campaign
        # produced arrived with pair_ratios=() and any session aggregation
        # over them counted ZERO sessions while looking perfectly healthy.
        # Caught on hardware: a governed session wrote lane_effects with
        # pairs=0, which would have made the session policy demand more
        # evidence for ever, one 13-minute run at a time.
        pair_ratios=tuple(run.stats.get("pair_ratios") or ()),
    )


def trigger_evidence_from_marker_probe(
    *, lane_id: str, role: str, positive_hit: bool,
) -> experiment_contract.TriggerEvidence:
    """VA14: build real TriggerEvidence from RD08-style trace-marker
    activation evidence (already produced by
    validation_campaign.py::run_trace_activation_probes()). A boolean
    marker observation is a coarse but real and honest signal -- record it
    as candidate_launches=1 (observed at least once) or 0 (never
    observed), not a fabricated precise count the marker mechanism does
    not actually provide."""
    return experiment_contract.TriggerEvidence(
        role=role, lane_id=lane_id, candidate_launches=1 if positive_hit else 0,
    )
