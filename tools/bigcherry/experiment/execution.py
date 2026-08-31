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

import re
import subprocess
from dataclasses import dataclass
from typing import Callable

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
}


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
            metrics = extract_metrics(result.combined, {metric: pattern})
            runs.append({"pair": pair, "mode": mode, "metrics": metrics})
    stats = block_bootstrap_effect(
        runs, candidate="subject", reference="control", metric=metric,
        lower_is_better=lower_is_better, seed=seed, resamples=resamples,
    )
    return PairedLaneRun(runs=tuple(runs), stats=stats)


def lane_effect_from_run(role: str, metric: str, run: PairedLaneRun) -> experiment_contract.LaneEffect:
    return experiment_contract.LaneEffect(
        role=role, metric=metric, geometric_effect_pct=run.stats["geometric_effect_pct"],
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
