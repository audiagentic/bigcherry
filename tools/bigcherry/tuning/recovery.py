"""HTR01: bounded recovery search for HI143 hard-fail campaigns.

HI143's behavioral gate is validated on real hardware (two independent
campaigns: one correctly promoted a safe cache, one correctly refused to
promote an unsafe one -- see HI143's plan-item notes). But it is
all-or-nothing: a single hard_fail anywhere in a promoted cache discards
the ENTIRE campaign's yield, including every other candidate's real,
already-measured speed win.

This module recovers as much of a failed campaign's yield as safely
possible using ONLY data the failed campaign already collected -- no new
GPU tuning/timing measurements -- before falling back to native for the
minimum necessary set of signatures.

Design adversarially negotiated with GPT (session ses_330ae3c055084f38,
2 rounds, 2026-08-29). GPT's key correction to the original bisection-only
proposal: HI141 already proved candidate-set safety is NON-MONOTONIC (the
guilty candidate alone fails a regression vector; a different, larger real
ensemble containing the SAME candidate+signature passed -- the
"candidate-mix masking" finding). A plain "binary search for the one
guilty candidate" is therefore not sound as the whole architecture -- it
is one strategy among several, plugged into a boundary that never lets a
strategy itself declare anything safe.

Plugin contract (GPT-specified, adopted verbatim):

    RecoveryStrategy.propose(state) -> list[AssignmentProposal]
        -- owns: group splitting, interaction handling, alternative
           ordering, future value-aware/op-shape heuristics. NEVER mutates
           caches or declares a proposal safe.

    AssignmentExecutor.evaluate(proposal) -> Observation
        -- owns: candidate eligibility/trust checks, cache construction
           (via the existing, already-validated replay.build() seed-
           override mechanism -- HI22), behavioral execution against
           cached native traces, final full-corpus validation.
        -- ONLY the executor (the oracle) can produce a behavioral PASS.

This is what lets v1's bounded delta-debugging and a later value-aware
best-first search share the same plumbing without redesign.

GPT's rejected alternative (explicitly considered and dismissed):
incremental candidate-by-candidate pre-admission testing before full-cache
promotion. Turns HI143's cheap common-path (optimistic full-cache gate
first) into O(N) on every successful campaign too, is ordering-dependent,
and can reject a candidate individually even where a later full ensemble
would have masked it safely (a false rejection, not just an inefficiency).
Retained instead: optimistic full-cache gate first -> bounded recovery
search only on actual failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from . import behavioral_gate as behavioral_gate_mod
from . import replay as replay_mod
from .server_runner import ServerError, ServerRunner

DEFAULT_MAX_RECOVERY_EVALUATIONS = 24
RESERVE_FINAL_VALIDATION = 1
# HTR01 (2026-08-30, adversarially reviewed with GPT, session
# ses_330ae3c055084f38): real-hardware validation found that every
# already-measured alternative in ranking_decisions lacks correctness
# evidence (normal campaigns only evidence the eventual winner), so
# recovery's "no new work" premise was true for TIMING but false for
# CORRECTNESS EVIDENCE. GPT's verdict: lazy, on-demand evidence generation
# ONLY for an alternative recovery is actually about to probe -- never
# eager pre-qualification of every ranked alternative on every campaign
# (taxes every successful campaign for a benefit that mostly never pays
# off) -- under its OWN separate, small budget, distinct from the
# behavioral-evaluation budget, so qualification work can never silently
# consume the whole recovery budget.
DEFAULT_MAX_NEW_CORRECTNESS_CANDIDATES = 16


class RecoveryError(RuntimeError):
    """Recovery search failed closed -- never a false-positive publish."""


# --------------------------------------------------------------------- state


@dataclass(frozen=True)
class SignatureAssignment:
    """One signature's current candidate choice in a full cache assignment.

    ``alternatives`` is the ordered (best-first by effective_us, i.e. least
    predicted performance loss) list of OTHER candidates real tune
    measurements exist for at this exact signature, taken directly from
    that signature's own ``ranking_decisions`` entry -- a free lookup, no
    retune. ``"native"`` is always implicitly the last, guaranteed-safe
    alternative even if not explicitly present in that list.
    """
    dispatch: str  # dispatch digest -- the seed-override key replay.build() expects
    signature: str  # signature digest -- required alongside dispatch by the override format
    current_candidate: str
    native_candidate: str  # the REAL native stable_name for this op family (e.g.
                            # "mmvq:native:v1") -- replay.build()'s seed-override
                            # mechanism requires a manifest-real candidate identity,
                            # not the literal string "native" (a real bug caught
                            # during this item's own first real-hardware validation
                            # run, 2026-08-29: "seed override 'native' for dispatch
                            # ... is not in the manifest").
    alternatives: tuple[str, ...]  # excludes current_candidate and native_candidate


@dataclass(frozen=True)
class RecoveryState:
    """Everything a RecoveryStrategy needs to propose the next move.
    Strategies read this; only strategy.record() (never the orchestrator
    directly -- see run_recovery) produces the next one.

    ``committed_overrides`` (v2 design, 2026-08-30, fixing a real bug found
    on the second real-hardware run -- see module docstring's "history"
    note) is the ONLY assignment ever treated as a safe working baseline.
    It starts empty (meaning: the campaign's original, all-winners
    assignment) and is updated EXCLUSIVELY by a strategy's own record()
    when it decides to accept something -- never by the orchestrator
    reacting to an individual diagnostic probe's PASS, since a diagnostic
    PASS during isolation is evidence about a SUBSET, not permission to
    adopt that subset's exact override values as the new baseline."""
    assignments: dict[str, SignatureAssignment]  # keyed by dispatch digest
    failing_vectors: tuple[behavioral_gate_mod.BehavioralVector, ...]
    dispatch_hits: frozenset[str]  # dispatch digests the failing vectors actually exercised
    evaluated: tuple["Observation", ...]  # prior probes this run, for dedup and history
    remaining_budget: int
    committed_overrides: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AssignmentProposal:
    """A candidate next assignment to probe. ``overrides`` maps dispatch
    digest -> candidate stable_name -- the COMPLETE, self-contained
    assignment for every dispatch this probe cares about (v2 design: a
    proposal is never "merged" with anything else by the orchestrator; it
    is evaluated exactly as given). ``probe_vectors`` lets a strategy ask
    for a cheap partial probe (failing vectors only) vs. the mandatory
    full-corpus check the executor always runs before treating a proposal
    as publishable."""
    label: str
    overrides: dict[str, str]
    probe_vectors: tuple[behavioral_gate_mod.BehavioralVector, ...] | None = None  # None = use state.failing_vectors


@dataclass(frozen=True)
class Observation:
    """What actually happened when ONE proposal was evaluated."""
    proposal: AssignmentProposal
    verdict: str  # "pass" (probe vectors clean) | "hard_fail" | "behavior_changed" |
                  # "ineligible" (proposal itself couldn't even be built/run -- e.g. a
                  # candidate lacking correctness evidence; NOT evidence about the
                  # signature's guilt either way, just wasted budget) | "unstable"
                  # (a probe that genuinely produced inconsistent repeat results --
                  # not yet detected by any code path here; reserved for future use)
    report: behavioral_gate_mod.BehavioralGateReport | None
    full_corpus_validated: bool = False


@dataclass(frozen=True)
class RecoveryAction:
    """v2 design (2026-08-30, GPT session ses_330ae3c055084f38, fixing a
    real pairing bug found on the second real-hardware validation run --
    see module docstring). A strategy proposes an ACTION, not a bag of
    independent proposals: one proposal for a single repair-phase trial,
    or exactly two for a bisection split whose siblings must be evaluated
    together, atomically, against the SAME parent baseline, before the
    strategy is ever asked to interpret either outcome. This is what makes
    the "both halves individually pass but their union fails -> genuine
    interaction group" case actually detectable, instead of silently
    falling through un-paired the way the v1 implementation did."""
    action_id: str
    proposals: tuple[AssignmentProposal, ...]


@dataclass(frozen=True)
class ActionObservation:
    """Both (or the one) Observation(s) for one RecoveryAction, delivered
    to strategy.record() together -- never one at a time."""
    action: RecoveryAction
    observations: tuple[Observation, ...]


class RecoveryStrategy(Protocol):
    """Strategy proposes; only the executor/oracle decides. A strategy
    must never be trusted to declare a proposal safe on its own, and the
    orchestrator (run_recovery) must never mutate the working baseline on
    its own initiative either -- only strategy.record() may update
    RecoveryState.committed_overrides."""

    def propose(self, state: RecoveryState) -> RecoveryAction | None:
        ...

    def record(self, state: RecoveryState, result: ActionObservation) -> RecoveryState:
        """Fold one ActionObservation (both sibling outcomes, for a split;
        the one outcome, for a repair trial) into a new RecoveryState."""
        ...


# --------------------------------------------------- v1: paired bisection


@dataclass
class BoundedPairedBisectionStrategy:
    """v1 RecoveryStrategy (renamed from BoundedDeltaDebugStrategy,
    2026-08-30, per GPT's explicit naming correction -- this provides
    logarithmic sparse-failure localization with explicit interaction
    detection and bounded fallback, NOT a general/causally-complete ddmin;
    overclaiming the latter is exactly what let the v1 pairing bug go
    unnoticed until real hardware caught it).

    Two phases:

    ISOLATE -- paired bisection over H = non-native dispatches actually hit
    by the failing vectors (``state.dispatch_hits & assignments``). Each
    split evaluates BOTH children atomically (as one RecoveryAction) against
    the SAME parent baseline (everything outside the current group forced
    native; within the group, only the child under test keeps its original
    candidate) -- GPT's exact outcome matrix:

        A FAIL, B PASS         -> recurse into A
        A PASS, B FAIL         -> recurse into B
        A FAIL, B FAIL         -> queue BOTH as independent failing groups
        A PASS, B PASS         -> parent group is an INTERACTION group;
                                   stop splitting it, blame neither half
        any unstable           -> abort recovery entirely
        any native probe ineligible -> structural error, abort (native
                                   should never lack correctness evidence)

    REPAIR -- once isolation is done, build the conservative baseline
    (every isolated/interaction-group dispatch forced native, probed as a
    single action) and, if it passes, walk each implicated signature's
    already-measured alternatives IN ORDER (best-first by effective_us),
    one at a time, against that baseline -- accept on PASS; on a genuine
    behavioral rejection (hard_fail/behavior_changed), try the next
    alternative for the SAME signature before giving up on it (GPT deep-
    dive review, round 2, 2026-08-30 -- corrected from an earlier version
    that tried only the single best alternative and then reported the
    entire remaining catalog as "exhausted", which was never actually
    tested). An "ineligible" verdict (the candidate could not even be
    behaviorally tested, e.g. failed correctness evidence) moves past that
    alternative WITHOUT counting it as behaviorally exhausted -- only a
    completed real comparison may ever be reported as tried-and-rejected."""

    _initial_hits: tuple[str, ...] | None = None
    _pending_groups: list[tuple[str, ...]] = field(default_factory=list)
    _isolated_groups: list[tuple[str, ...]] = field(default_factory=list)
    _interaction_groups: list[tuple[str, ...]] = field(default_factory=list)
    _phase: str = "isolate"  # "isolate" | "repair"
    _repair_queue: list[str] = field(default_factory=list)
    _repair_current_dispatch: str | None = None
    _repair_baseline_proposed: bool = False
    # GPT deep-dive review (2026-08-30, session ses_330ae3c055084f38): the
    # original repair phase tried ONLY alternatives[0] per signature, but
    # RetuneRecommendation.exhausted_candidates reported the ENTIRE
    # alternatives tuple as "exhausted" -- a real, misleading overclaim
    # ("5 real alternatives genuinely tried" was false; only 1 was). Now
    # walks alternatives in order (budget-bounded, same as before -- no
    # new GPU/timing measurement, still v1-scoped) and tracks exactly
    # which were actually tried per dispatch so reporting can be accurate.
    _repair_alt_index: int = 0
    tried_alternatives: dict[str, list[str]] = field(default_factory=dict)

    def propose(self, state: RecoveryState) -> RecoveryAction | None:
        if self._initial_hits is None:
            self._initial_hits = tuple(sorted(state.dispatch_hits & set(state.assignments)))
            self._pending_groups = [self._initial_hits] if self._initial_hits else []

        if self._phase == "isolate":
            while self._pending_groups and len(self._pending_groups[-1]) <= 1:
                self._isolated_groups.append(self._pending_groups.pop())
            if self._pending_groups:
                if len(self._pending_groups) * 2 > state.remaining_budget:
                    # Not enough budget left to even evaluate one more
                    # split's pair -- stop isolating and move straight to
                    # the conservative repair baseline with whatever is
                    # already isolated/pending (pending groups are folded
                    # into "isolated" wholesale, since we cannot afford to
                    # keep splitting them).
                    while self._pending_groups:
                        self._isolated_groups.append(self._pending_groups.pop())
                else:
                    group = self._pending_groups[-1]
                    mid = len(group) // 2
                    a_group, b_group = group[:mid], group[mid:]
                    return RecoveryAction(
                        action_id=f"split:{group[0][:8]}",
                        proposals=(
                            self._probe_proposal(state, "A", a_group),
                            self._probe_proposal(state, "B", b_group),
                        ),
                    )

            # Isolation complete (or budget-forced early stop) -- move to repair.
            self._phase = "repair"
            implicated = tuple(
                d for group in (self._isolated_groups + self._interaction_groups) for d in group
            )
            baseline = {d: state.assignments[d].native_candidate for d in implicated
                        if d in state.assignments}
            self._repair_queue = list(baseline)
            self._repair_baseline_proposed = True
            return RecoveryAction(
                action_id="repair-baseline",
                proposals=(AssignmentProposal(label="repair-baseline", overrides=baseline),),
            )

        # repair phase: one already-timed alternative at a time.
        while self._repair_current_dispatch is None and self._repair_queue:
            candidate_dispatch = self._repair_queue.pop(0)
            assignment = state.assignments.get(candidate_dispatch)
            if assignment is not None and assignment.alternatives:
                self._repair_current_dispatch = candidate_dispatch
            # else: no real alternative exists for this signature -- stays
            # native in committed_overrides, nothing more to try for it.

        if self._repair_current_dispatch is None:
            return None  # nothing left to try -- run_recovery does final validation

        assignment = state.assignments[self._repair_current_dispatch]
        if self._repair_alt_index >= len(assignment.alternatives):
            # Every alternative for this dispatch has now genuinely been
            # tried and rejected -- move to the next queued dispatch.
            self._repair_current_dispatch = None
            self._repair_alt_index = 0
            return self.propose(state)
        alt = assignment.alternatives[self._repair_alt_index]
        # GPT deep-dive review, round 2 (2026-08-30): do NOT record this as
        # "tried" here, optimistically, before the real verdict is known --
        # run_recovery's except RecoveryError -> "ineligible" fallback
        # means a STRUCTURAL failure (e.g. the cardinality guard itself, or
        # a server crash) raises the exact same exception type as a
        # genuine correctness-ineligibility, so recording here would let a
        # structural failure silently count as "behaviorally rejected" in
        # exhausted_candidates. Only record() may add to tried_alternatives,
        # and only for a completed real behavioral verdict.
        trial = dict(state.committed_overrides)
        trial[self._repair_current_dispatch] = alt
        return RecoveryAction(
            action_id=f"repair-try:{self._repair_current_dispatch[:8]}:{alt}",
            proposals=(AssignmentProposal(label=f"repair-try:{alt}", overrides=trial),),
        )

    def _probe_proposal(self, state: RecoveryState, label: str, active_group: tuple[str, ...]) -> AssignmentProposal:
        """A complete, self-contained assignment over ALL of H (not just
        the current group): members of ``active_group`` keep their real
        candidate; every OTHER dispatch in H is forced native -- GPT's
        exact invariant ("only members of G retain original candidate;
        every H-G member is forced native"). Both siblings of one split
        therefore always share the identical parent baseline outside
        their own group, fixing the v1 bug where a mutating
        ``current_overrides`` meant sibling B was not necessarily tested
        against the same baseline sibling A was."""
        overrides = {}
        for dispatch in self._initial_hits:
            assignment = state.assignments.get(dispatch)
            if assignment is None:
                continue
            overrides[dispatch] = (
                assignment.current_candidate if dispatch in active_group else assignment.native_candidate
            )
        return AssignmentProposal(
            label=f"probe-{label}:{','.join(d[:8] for d in active_group)}", overrides=overrides,
        )

    def record(self, state: RecoveryState, result: ActionObservation) -> RecoveryState:
        evaluated = state.evaluated + result.observations
        remaining = state.remaining_budget - len(result.observations)
        for obs in result.observations:
            if obs.verdict == "unstable":
                raise RecoveryError(
                    f"non-deterministic result probing {obs.proposal.label!r} -- "
                    f"aborting recovery, this is not a candidate-attributable failure"
                )

        if len(result.action.proposals) == 2:
            # An isolate-phase split -- both siblings' outcomes are known
            # NOW, together, for the first time (fixes the v1 pairing bug).
            obs_a, obs_b = result.observations
            if obs_a.verdict == "ineligible" or obs_b.verdict == "ineligible":
                raise RecoveryError(
                    "a native-forcing isolation probe was ineligible -- structural "
                    "error (native should never lack correctness evidence)"
                )
            group = self._pending_groups.pop()
            mid = len(group) // 2
            a_group, b_group = group[:mid], group[mid:]
            a_fail = obs_a.verdict in ("hard_fail", "behavior_changed")
            b_fail = obs_b.verdict in ("hard_fail", "behavior_changed")
            if a_fail and not b_fail:
                self._pending_groups.append(a_group)
            elif b_fail and not a_fail:
                self._pending_groups.append(b_group)
            elif a_fail and b_fail:
                self._pending_groups.append(a_group)
                self._pending_groups.append(b_group)
            else:  # both pass -- genuine interaction, stop splitting this group
                self._interaction_groups.append(group)
            return RecoveryState(
                assignments=state.assignments, failing_vectors=state.failing_vectors,
                dispatch_hits=state.dispatch_hits, evaluated=evaluated, remaining_budget=remaining,
                committed_overrides=state.committed_overrides,
            )

        # A single-proposal repair-phase action.
        observation = result.observations[0]
        committed = state.committed_overrides
        if result.action.action_id == "repair-baseline":
            if observation.verdict == "pass":
                committed = dict(result.action.proposals[0].overrides)
            # else: baseline itself failed -- leave committed_overrides as
            # it was (empty); run_recovery's own final circuit-breaker
            # fallback (force EVERY implicated dispatch to native) covers
            # this rare case.
        else:
            dispatch = self._repair_current_dispatch
            tried_alt = result.action.proposals[0].overrides[dispatch]
            if observation.verdict == "pass":
                committed = dict(committed)
                committed[dispatch] = tried_alt
                self._repair_current_dispatch = None
                self._repair_alt_index = 0
            elif observation.verdict in ("hard_fail", "behavior_changed"):
                # A REAL, completed behavioral rejection -- only this case
                # may count toward exhausted_candidates (GPT deep-dive
                # review, round 2: an "ineligible" verdict here would mean
                # the candidate was never actually behaviorally tested at
                # all, so it must never be reported as "tried and
                # rejected"). Try the NEXT alternative for the SAME
                # dispatch (if any remain) before giving up on it.
                self.tried_alternatives.setdefault(dispatch, []).append(tried_alt)
                self._repair_alt_index += 1
            else:
                # "ineligible" (or any other non-terminal verdict): this
                # specific alternative could not be tested at all (e.g. it
                # failed correctness evidence) -- move past it WITHOUT
                # recording it as behaviorally exhausted, since it never
                # actually ran a real comparison.
                self._repair_alt_index += 1
        return RecoveryState(
            assignments=state.assignments, failing_vectors=state.failing_vectors,
            dispatch_hits=state.dispatch_hits, evaluated=evaluated, remaining_budget=remaining,
            committed_overrides=committed,
        )


# ------------------------------------------------------------- executor


@dataclass
class AssignmentExecutor:
    """Owns everything a RecoveryStrategy is never trusted with: real cache
    construction, real server launches, real comparison against cached
    native traces, and the mandatory full-corpus check before any
    assignment is treated as publishable.

    Cache construction reuses replay.build()'s existing, HI22-validated
    seed-override mechanism (an explicit ``{dispatch: stable_name}``
    override file, manifest-bound and identity-checked) rather than
    hand-rolling new measurements-row mutation logic -- this is the same
    trusted path an operator's manual seed file already goes through.
    """
    binary_path: Path
    model_path: Path
    devices: str
    common_args: tuple[str, ...]
    measurements_path: Path
    manifest_path: Path
    ggml_h_path: Path
    workdir: Path
    # RV49 correctness-gate verification database (workdir/"tune.sqlite" in
    # the normal campaign path) -- required so replay.build() can verify
    # each winner (original or substituted alternative) actually has real
    # correctness evidence bound to this exact build+hardware identity,
    # matching _stage_replay_export's own require_winner_verification=True
    # behavior. Omitting this (a real bug caught during this item's own
    # first real-hardware validation run, 2026-08-29) makes replay.build()
    # fall back to demanding an explicit --dispatch-db and rejecting EVERY
    # winner, not just an actually-unevidenced one.
    dispatch_db: Path
    # HTR01 (2026-08-30, locked design, GPT session ses_330ae3c055084f38):
    # real-hardware validation found every already-measured alternative
    # lacks correctness evidence (normal campaigns only evidence the
    # eventual winner). correctness_binary_path is the PATCHED
    # test-backend-ops binary (patches 1222+1223 applied) -- a separate
    # artifact from binary_path (llama-server) -- needed to generate that
    # evidence lazily, on-demand, for exactly the alternative a proposal
    # is about to try.
    correctness_binary_path: Path
    vendor_root: Path
    campaign_run_id: str | None = None
    recovery_run_id: str | None = None
    correctness_seeds: tuple[int, ...] = (1, 2, 3)
    max_new_correctness_candidates: int = DEFAULT_MAX_NEW_CORRECTNESS_CANDIDATES
    native_trace_cache: dict[str, behavioral_gate_mod.BehavioralTrace] = field(default_factory=dict)
    _correctness_candidates_used: int = field(default=0, init=False, repr=False)
    _native_seed_caches: dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _rows_by_dispatch: dict | None = field(default=None, init=False, repr=False)
    _dispatch_db_conn: object | None = field(default=None, init=False, repr=False)

    def capture_native_traces(self, vectors: list[behavioral_gate_mod.BehavioralVector]) -> None:
        """Run every corpus vector against forced-native ONCE and cache the
        traces -- per GPT's explicit cost optimization, native behavior is
        invariant for a fixed behavioral identity (build+model+runtime),
        so every subsequent probe only needs a SECOND (candidate) server
        load, not two, cutting recovery-search cost roughly in half."""
        env = {"HIP_VISIBLE_DEVICES": self.devices, "GGML_HIP_DISPATCH_MODE": "native"}
        runner = ServerRunner(
            binary=self.binary_path, model=self.model_path, extra_args=self.common_args,
            env_overrides=env, log_path=self.workdir / "recovery-native.log",
        )
        try:
            with runner:
                for vector in vectors:
                    self.native_trace_cache[vector.name] = behavioral_gate_mod.run_vector(runner, vector)
        except (behavioral_gate_mod.BehavioralGateError, ServerError) as exc:
            raise RecoveryError(f"failed to capture native traces for recovery: {exc}") from exc

    def _write_seed_file(self, overrides: dict[str, str], manifest: dict) -> Path:
        seed_path = self.workdir / "recovery-seed.json"
        document = {
            "version": 1,
            "provenance": {
                "source_revision": manifest["source_revision"],
                "manifest_hash": manifest.get("manifest_hash"),
            },
            "overrides": overrides,
        }
        seed_path.write_text(json.dumps(document), encoding="utf-8")
        return seed_path

    def build_candidate_cache(self, overrides: dict[str, str]) -> Path:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        seed_path = self._write_seed_file(overrides, manifest)
        cache_bytes = replay_mod.build(
            self.measurements_path, self.manifest_path, self.ggml_h_path,
            dispatch_db=self.dispatch_db, seed_file=seed_path,
            require_winner_verification=True,
        )
        cache_path = self.workdir / "recovery-candidate.cache"
        cache_path.write_bytes(cache_bytes)
        return cache_path

    def _load_rows_by_dispatch(self) -> dict:
        if self._rows_by_dispatch is None:
            rows: dict[str, dict] = {}
            with self.measurements_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    dispatch = row.get("dispatch")
                    if dispatch:
                        rows[dispatch] = row
            self._rows_by_dispatch = rows
        return self._rows_by_dispatch

    def ensure_correctness_evidence(self, dispatch: str, candidate_name: str) -> None:
        """HTR01: lazily qualify ONE alternative candidate for ONE signature
        (dispatch digest) under its OWN separate, bounded budget --
        max_new_correctness_candidates, distinct from run_recovery's
        behavioral-evaluation budget, per GPT's explicit requirement that
        correctness-qualification work can never silently consume the
        whole recovery budget. Raises RecoveryError (never silently skips)
        when: the row/signature can't be resolved, evidence generation
        genuinely fails (the candidate really is numerically bad), or the
        correctness budget is exhausted."""
        import sqlite3

        from .. import hi80_generate_correctness_evidence as hi80
        from . import correctness_evidence as ce

        if self._dispatch_db_conn is None:
            self._dispatch_db_conn = sqlite3.connect(str(self.dispatch_db))
        conn = self._dispatch_db_conn
        rows = self._load_rows_by_dispatch()
        row = rows.get(dispatch)
        if row is None:
            raise RecoveryError(f"no measurements row found for dispatch {dispatch!r}")

        if self._correctness_candidates_used >= self.max_new_correctness_candidates:
            raise RecoveryError(
                f"correctness-qualification budget exhausted "
                f"({self.max_new_correctness_candidates} candidates) -- cannot "
                f"qualify {candidate_name!r} for dispatch {dispatch!r}"
            )

        native_cache = self._native_seed_caches.setdefault(dispatch, {})
        try:
            result = hi80.generate_for_candidate(
                conn, row, candidate_name=candidate_name,
                binary=self.correctness_binary_path, vendor_root=self.vendor_root,
                seeds=self.correctness_seeds, headroom_fraction=ce.DEFAULT_HEADROOM_FRACTION,
                contract_version=ce.CONTRACT_VERSION, tool_version="htr01-recovery",
                origin=ce.EvidenceOrigin(
                    reason="recovery_alternative", campaign_run_id=self.campaign_run_id,
                    recovery_run_id=self.recovery_run_id,
                ),
                native_seed_cache=native_cache,
            )
        except Exception as exc:  # noqa: BLE001 -- any generation failure means "not eligible"
            raise RecoveryError(
                f"correctness-evidence generation failed for {candidate_name!r} "
                f"(dispatch {dispatch!r}): {exc}"
            ) from exc

        if result.status == "generated":
            self._correctness_candidates_used += 1
        if not result.dispatchable:
            raise RecoveryError(
                f"{candidate_name!r} (dispatch {dispatch!r}) failed correctness "
                f"evidence -- not eligible for behavioral probing"
            )

    def evaluate(self, proposal: AssignmentProposal, *, full_corpus: list[behavioral_gate_mod.BehavioralVector]) -> Observation:
        probe_vectors = list(proposal.probe_vectors) if proposal.probe_vectors is not None else None
        rows = self._load_rows_by_dispatch()
        for dispatch, candidate_name in proposal.overrides.items():
            row = rows.get(dispatch)
            if row is not None and candidate_name == row.get("native"):
                continue  # native is the reference -- never needs its own evidence
            self.ensure_correctness_evidence(dispatch, candidate_name)
        try:
            cache_path = self.build_candidate_cache(proposal.overrides)
        except (KeyError, ValueError, SystemExit) as exc:
            raise RecoveryError(
                f"proposal {proposal.label!r} is not eligible (cache construction "
                f"failed -- likely a missing correctness-evidence or manifest "
                f"identity mismatch for one of its candidates): {exc}"
            ) from exc

        env = {
            "HIP_VISIBLE_DEVICES": self.devices, "GGML_HIP_DISPATCH_MODE": "replay",
            "GGML_HIP_DISPATCH_CACHE": str(cache_path),
        }
        runner = ServerRunner(
            binary=self.binary_path, model=self.model_path, extra_args=self.common_args,
            env_overrides=env, log_path=self.workdir / f"recovery-probe-{proposal.label.replace(':', '_')}.log",
        )
        # Real, severe bug found on real hardware (2026-08-30): this used
        # to be `vectors_to_run = probe_vectors if ... else
        # list(self.native_trace_cache.keys())` -- probe_vectors is a list
        # of REAL BehavioralVector OBJECTS (every real call supplies one:
        # run_recovery always passes proposal.probe_vectors or
        # failing_vectors), but the else-branch produced a list of NAME
        # STRINGS. The loop below assumed `name` was always a string
        # (`v.name == name`), so whenever probe_vectors was given --
        # i.e. EVERY real invocation -- that comparison was always False,
        # `vector` was always None, every iteration silently `continue`d,
        # and report.verdicts stayed EMPTY. hard_fail/needs_throughput_
        # adjudication are both `any(...)` over that empty list, so EVERY
        # probe throughout the ENTIRE recovery search vacuously "passed"
        # without ever running a single real behavioral comparison. A real
        # tg128 benchmark against a "successfully recovered" cache caught
        # this: it measured a genuine 27% regression the vacuous "pass"
        # never actually checked for. Normalize to names ONCE, up front,
        # so this class of type confusion cannot recur.
        vector_names = (
            [v.name for v in probe_vectors] if probe_vectors is not None
            else list(self.native_trace_cache.keys())
        )
        report = behavioral_gate_mod.BehavioralGateReport()
        try:
            with runner:
                for name in vector_names:
                    vector = next((v for v in full_corpus if v.name == name), None)
                    if vector is None:
                        raise RecoveryError(
                            f"proposal {proposal.label!r} requested vector {name!r} "
                            f"which is not present in full_corpus -- refusing to "
                            f"silently skip a requested comparison"
                        )
                    native_trace = self.native_trace_cache.get(name)
                    if native_trace is None:
                        raise RecoveryError(f"no cached native trace for vector {name!r}")
                    candidate_trace = behavioral_gate_mod.run_vector(runner, vector)
                    report.verdicts.append(
                        behavioral_gate_mod.compare_traces(name, native_trace, candidate_trace)
                    )
        except (behavioral_gate_mod.BehavioralGateError, ServerError) as exc:
            raise RecoveryError(f"probe for {proposal.label!r} failed to execute: {exc}") from exc

        # GPT deep-dive review (2026-08-30): nonzero is necessary but not
        # sufficient -- enforce EXACT cardinality/order against what was
        # actually requested, catching partial execution, duplicates, or
        # ordering mistakes, not just total silence.
        actual_names = tuple(v.vector_name for v in report.verdicts)
        if actual_names != tuple(vector_names):
            raise RecoveryError(
                f"proposal {proposal.label!r} produced verdicts for {actual_names!r} "
                f"but {tuple(vector_names)!r} were requested -- refusing to treat "
                f"partial/mismatched coverage as complete"
            )
        if report.hard_fail:
            verdict = "hard_fail"
        elif report.needs_throughput_adjudication:
            verdict = "behavior_changed"
        else:
            verdict = "pass"
        return Observation(proposal=proposal, verdict=verdict, report=report)

    def validate_full_corpus(self, overrides: dict[str, str], *, full_corpus: list[behavioral_gate_mod.BehavioralVector]) -> Observation:
        """The mandatory check before ANY assignment (partial or full) is
        treated as publishable -- always the complete corpus, never just
        the vectors that were previously failing, since interaction/
        masking makes local validation insufficient (HI143's own
        contract is defined at the complete-cache level)."""
        proposal = AssignmentProposal(label="final-validation", overrides=overrides,
                                       probe_vectors=tuple(full_corpus))
        observation = self.evaluate(proposal, full_corpus=full_corpus)
        return Observation(
            proposal=observation.proposal, verdict=observation.verdict,
            report=observation.report, full_corpus_validated=True,
        )


# ------------------------------------------------------------ orchestrator


@dataclass(frozen=True)
class RetuneRecommendation:
    """HTR04 (adversarial review, ses_330ae3c055084f38, 2026-08-29):
    recovery failure is NOT a retune trigger by itself -- GPT was explicit
    that this must be a RECOMMENDATION only, never an autonomous action.
    "HTR01 failure -> native fallback -> quantify loss -> if material:
    determine staleness or search-space-expansion need -> explicit retune
    recommendation" -- not "HTR01 failure -> retune". This record is the
    structured evidence an operator (or, much later, an explicit
    GPU-budget-governed automatic policy that does not yet exist) uses to
    decide, not a decision itself. No code path may act on this
    autonomously."""
    signature_dispatch: str
    reason: str  # "alternatives_exhausted" -- the only reason v1 can detect;
                 # "search_space_expansion_available" / "repeated_behavioral_implication" /
                 # "material_native_fallback_loss" require data (a candidate registry
                 # delta, BehavioralFailureWitness history, and an E2E loss estimate,
                 # respectively) this module does not yet have access to -- see HTR04.
    current_assignment: str  # what this signature was assigned to (native or a candidate)
    exhausted_candidates: tuple[str, ...]


@dataclass(frozen=True)
class RecoveryResult:
    published: bool
    final_overrides: dict[str, str]
    evaluations_used: int
    stop_reason: str
    cache_path: Path | None
    retune_recommendations: tuple[RetuneRecommendation, ...] = ()


def run_recovery(
    *, executor: AssignmentExecutor, strategy: RecoveryStrategy,
    initial_assignments: dict[str, SignatureAssignment],
    initial_report: behavioral_gate_mod.BehavioralGateReport,
    full_corpus: list[behavioral_gate_mod.BehavioralVector],
    dispatch_hits: frozenset[str],
    max_evaluations: int = DEFAULT_MAX_RECOVERY_EVALUATIONS,
) -> RecoveryResult:
    """Bounded recovery loop: propose -> evaluate -> record, until either a
    proposal's full-corpus validation passes, or the evaluation budget
    (GPT: cap real behavioral candidate-server evaluations, not abstract
    "iterations") is exhausted.

    Collects ALL failing vectors from the initial report up front (GPT's
    explicit requirement) rather than stopping at the first hard_fail, so
    recovery does not repair vector 1, reload, discover vector 2 broken
    too, and repair again.
    """
    failing_vectors = tuple(
        v for v in full_corpus
        if any(verdict.vector_name == v.name and verdict.verdict != "exact_pass"
               for verdict in initial_report.verdicts)
    )
    if not failing_vectors:
        raise RecoveryError("run_recovery called with no failing vectors -- nothing to recover from")

    executor.capture_native_traces(full_corpus)
    budget = max_evaluations - RESERVE_FINAL_VALIDATION
    state = RecoveryState(
        assignments=initial_assignments, failing_vectors=failing_vectors,
        dispatch_hits=dispatch_hits, evaluated=(), remaining_budget=budget,
        committed_overrides={},
    )

    # v2 orchestrator (2026-08-30, fixing the real pairing bug the second
    # real-hardware run found): evaluate an ENTIRE RecoveryAction (both
    # siblings of a split, atomically, against the identical parent
    # baseline) before the strategy is ever asked to interpret either
    # outcome. The orchestrator NEVER mutates a working baseline itself on
    # a diagnostic PASS -- only strategy.record() may update
    # state.committed_overrides, and only when it decides an outcome
    # actually warrants adopting it (the repair phase's accepted
    # alternatives; the accepted conservative baseline).
    while state.remaining_budget > 0:
        action = strategy.propose(state)
        if action is None:
            break
        if len(action.proposals) > state.remaining_budget:
            break  # cannot even start this action within the remaining budget
        observations = []
        for proposal in action.proposals:
            probe = AssignmentProposal(
                label=proposal.label, overrides=proposal.overrides,
                probe_vectors=proposal.probe_vectors or failing_vectors,
            )
            try:
                observations.append(executor.evaluate(probe, full_corpus=full_corpus))
            except RecoveryError:
                observations.append(Observation(proposal=probe, verdict="ineligible", report=None))
        state = strategy.record(state, ActionObservation(action=action, observations=tuple(observations)))

    best_overrides = state.committed_overrides
    if not best_overrides:
        # Nothing recovered (or isolation never got a chance to run a
        # repair phase at all) -- every implicated signature falls back
        # to its own real native candidate identity, per GPT's circuit-
        # breaker behavior. This still preserves every unrelated
        # already-clean winner untouched.
        best_overrides = {
            dispatch: initial_assignments[dispatch].native_candidate
            for dispatch in dispatch_hits & set(initial_assignments)
        }

    # HTR04: a signature that ended up forced to native despite having had
    # real measured alternatives is exactly the "alternatives exhausted"
    # recommendation trigger GPT specified -- record it as structured
    # evidence, never as an action. The other three reasons GPT specified
    # (search-space-expansion-available, repeated-behavioral-implication,
    # material-native-fallback-loss) need data this function does not have
    # (a candidate registry delta, BehavioralFailureWitness history, and an
    # E2E loss estimate) -- deferred to HTR04, not fabricated here.
    # GPT deep-dive review (2026-08-30): the original code reported the
    # signature's ENTIRE alternatives tuple as "exhausted" regardless of
    # how many were actually tried -- a real, misleading overclaim. Use
    # the strategy's own tried_alternatives bookkeeping (defensive
    # getattr: this is a real property of BoundedPairedBisectionStrategy,
    # not part of the RecoveryStrategy Protocol, so a different strategy
    # implementation without it degrades to an empty list rather than a
    # crash) and only emit a recommendation when EVERY real alternative
    # was genuinely tried and rejected -- not merely "some were available."
    strategy_tried = getattr(strategy, "tried_alternatives", {})
    recommendations = tuple(
        RetuneRecommendation(
            signature_dispatch=dispatch, reason="alternatives_exhausted",
            current_assignment="native",
            exhausted_candidates=tuple(strategy_tried.get(dispatch, [])),
        )
        for dispatch, candidate in best_overrides.items()
        if initial_assignments.get(dispatch) is not None
        and candidate == initial_assignments[dispatch].native_candidate
        and initial_assignments[dispatch].alternatives
        and len(strategy_tried.get(dispatch, [])) >= len(initial_assignments[dispatch].alternatives)
    )

    # GPT deep-dive review (2026-08-30): the old formula
    # (max_evaluations - state.remaining_budget + 1) double-counted the
    # final validation -- `budget` (max_evaluations - RESERVE_FINAL_
    # VALIDATION) is the correct baseline to diff against, since that is
    # what state.remaining_budget was actually initialized from; the "+1"
    # then correctly represents exactly the one mandatory final-validation
    # call below, which is never decremented from remaining_budget itself.
    evaluations_used = (budget - state.remaining_budget) + 1

    final_observation = executor.validate_full_corpus(best_overrides, full_corpus=full_corpus)
    if final_observation.verdict != "pass":
        return RecoveryResult(
            published=False, final_overrides=best_overrides,
            evaluations_used=evaluations_used,
            stop_reason="final full-corpus validation failed -- refusing to publish",
            cache_path=None, retune_recommendations=recommendations,
        )
    cache_path = executor.build_candidate_cache(best_overrides)
    return RecoveryResult(
        published=True, final_overrides=best_overrides,
        evaluations_used=evaluations_used,
        stop_reason="full-corpus validation passed",
        cache_path=cache_path, retune_recommendations=recommendations,
    )
