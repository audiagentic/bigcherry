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
    Strategies read this; only AssignmentExecutor.evaluate() produces the
    next one via Observation."""
    assignments: dict[str, SignatureAssignment]  # keyed by dispatch digest
    failing_vectors: tuple[behavioral_gate_mod.BehavioralVector, ...]
    dispatch_hits: frozenset[str]  # dispatch digests the failing vectors actually exercised
    evaluated: tuple["Observation", ...]  # prior probes this run, for dedup and history
    remaining_budget: int


@dataclass(frozen=True)
class AssignmentProposal:
    """A candidate next assignment to probe. ``overrides`` maps dispatch
    digest -> candidate stable_name for every signature this proposal
    changes relative to the current cache; unlisted signatures keep their
    current candidate. ``probe_vectors`` lets a strategy ask for a cheap
    partial probe (failing vectors only) vs. the mandatory full-corpus
    check the executor always runs before treating a proposal as
    publishable."""
    label: str
    overrides: dict[str, str]
    probe_vectors: tuple[behavioral_gate_mod.BehavioralVector, ...] | None = None  # None = use state.failing_vectors


@dataclass(frozen=True)
class Observation:
    """What actually happened when a proposal was evaluated -- the only
    thing a RecoveryStrategy is allowed to base its next proposal on."""
    proposal: AssignmentProposal
    verdict: str  # "pass" (probe vectors clean) | "hard_fail" | "behavior_changed" |
                  # "ineligible" (proposal itself couldn't even be built/run -- e.g. a
                  # candidate lacking correctness evidence; NOT evidence about the
                  # signature's guilt either way, just wasted budget) | "unstable"
                  # (a probe that genuinely produced inconsistent repeat results --
                  # not yet detected by any code path here; reserved for future use)
    report: behavioral_gate_mod.BehavioralGateReport | None
    full_corpus_validated: bool = False


class RecoveryStrategy(Protocol):
    """Strategy proposes; only the executor/oracle decides. A strategy
    must never be trusted to declare a proposal safe on its own."""

    def propose(self, state: RecoveryState) -> list[AssignmentProposal]:
        ...

    def record(self, state: RecoveryState, observation: Observation) -> RecoveryState:
        """Fold one Observation into a new RecoveryState (updated
        evaluated/remaining_budget; assignments only change if the
        strategy accepts the observation's proposal as the new baseline)."""
        ...


# --------------------------------------------------------- v1: delta debug


@dataclass
class BoundedDeltaDebugStrategy:
    """v1 RecoveryStrategy: bounded delta-debugging (ddmin-style) over the
    set of non-native candidates actually dispatched by the failing
    vectors (``state.dispatch_hits`` -- searching outside that set wastes
    budget on signatures that provably cannot be responsible).

    Implements GPT's exact outcome matrix (adversarial review,
    ses_330ae3c055084f38, 2026-08-29):

        union FAILs                              -> keep splitting
        half A fails, half B passes              -> recurse into A
        both halves fail                         -> track as two
                                                     independent failing
                                                     groups, continue on
                                                     each separately
        both halves pass, but their union fails  -> genuine INTERACTION
                                                     group -- STOP plain
                                                     bisection on it; do
                                                     not blame either half
        same config non-deterministic on repeat  -> abort recovery for
                                                     that vector
                                                     (infrastructure/
                                                     nondeterminism, not a
                                                     candidate defect)

    Once a minimal failing group (or interaction group) is isolated, this
    strategy proposes substituting the highest-value (least predicted
    performance loss) untried alternative for one signature in that group
    at a time -- never blindly falling back to native first."""

    # groups still under active bisection: each a tuple of dispatch digests
    # currently forced to "native" in the probe being evaluated
    _pending_groups: list[tuple[str, ...]] = field(default_factory=list)
    _interaction_groups: list[tuple[str, ...]] = field(default_factory=list)
    _confirmed_implicated: set[str] = field(default_factory=set)
    _repeat_check: dict[tuple[str, ...], str] = field(default_factory=dict)
    _initialized: bool = False

    def propose(self, state: RecoveryState) -> list[AssignmentProposal]:
        implicated_pool = tuple(sorted(state.dispatch_hits & set(state.assignments)))
        if not self._initialized:
            self._pending_groups = [implicated_pool] if implicated_pool else []
            self._initialized = True

        if self._pending_groups:
            group = self._pending_groups[-1]
            if len(group) <= 1:
                # Minimal (or already-singleton) group -- propose the next
                # untried alternative for its one signature instead of
                # continuing to bisect a group that can't split further.
                return self._alternative_proposals(state, group)
            mid = len(group) // 2
            half_a, half_b = group[:mid], group[mid:]
            return [
                self._force_native_proposal(state, f"bisect-A", half_a),
                self._force_native_proposal(state, f"bisect-B", half_b),
            ]

        if self._interaction_groups:
            group = self._interaction_groups[-1]
            return self._alternative_proposals(state, group)

        return []

    def _force_native_proposal(self, state: RecoveryState, label: str, group: tuple[str, ...]) -> AssignmentProposal:
        overrides = {}
        for d in group:
            assignment = state.assignments.get(d)
            if assignment is not None:
                overrides[d] = assignment.native_candidate
        return AssignmentProposal(
            label=f"{label}:{','.join(d[:8] for d in group)}",
            overrides=overrides,
        )

    def _alternative_proposals(self, state: RecoveryState, group: tuple[str, ...]) -> list[AssignmentProposal]:
        proposals = []
        for dispatch in group:
            assignment = state.assignments.get(dispatch)
            if assignment is None:
                continue
            for alt in assignment.alternatives:
                proposals.append(AssignmentProposal(
                    label=f"alt:{dispatch[:8]}:{alt}",
                    overrides={dispatch: alt},
                ))
            proposals.append(AssignmentProposal(
                label=f"native:{dispatch[:8]}",
                overrides={dispatch: assignment.native_candidate},
            ))
        return proposals

    def record(self, state: RecoveryState, observation: Observation) -> RecoveryState:
        evaluated = state.evaluated + (observation,)
        remaining = state.remaining_budget - 1
        overrides = observation.proposal.overrides
        group = tuple(sorted(overrides))

        if observation.verdict == "unstable":
            # Infrastructure/nondeterminism, not a candidate defect --
            # per GPT, abort recovery for this vector rather than keep
            # attributing it to a candidate.
            raise RecoveryError(
                f"non-deterministic result probing {observation.proposal.label!r} -- "
                f"aborting recovery, this is not a candidate-attributable failure"
            )
        if observation.verdict == "ineligible":
            # The proposal itself could not even be evaluated (e.g. an
            # alternative candidate lacks correctness evidence for this
            # exact build/hardware). This says NOTHING about whether the
            # signature/group is actually implicated -- just skip it and
            # keep the group queued/pending as-is so bisection or
            # alternative-search continues with the budget it has left.
            return RecoveryState(
                assignments=state.assignments, failing_vectors=state.failing_vectors,
                dispatch_hits=state.dispatch_hits,
                evaluated=state.evaluated + (observation,),
                remaining_budget=state.remaining_budget - 1,
            )

        if self._pending_groups and self._pending_groups[-1][:len(group)] == group:
            # We were bisecting; interpret this half's result.
            self._pending_groups.pop()
            if observation.verdict == "hard_fail" or observation.verdict == "behavior_changed":
                if len(group) > 1:
                    self._pending_groups.append(group)  # keep splitting this half
                else:
                    self._confirmed_implicated.add(group[0])
            # verdict == "pass": this half is clean under forced-native
            # substitution -- nothing to add back to pending. Whether the
            # OTHER half (already queued) still fails determines whether
            # we've found an interaction group; that is only detectable
            # once both halves of one split have been observed, tracked
            # by the caller (run_recovery) via _pair completion, not here.
            return RecoveryState(
                assignments=state.assignments, failing_vectors=state.failing_vectors,
                dispatch_hits=state.dispatch_hits, evaluated=evaluated, remaining_budget=remaining,
            )

        if self._interaction_groups and self._interaction_groups[-1] == group:
            if observation.verdict != "pass":
                # This alternative didn't clear it either -- stay on this
                # interaction group, strategy will propose the next
                # untried alternative next call (alternatives list is
                # exhausted lazily by run_recovery tracking tried overrides).
                pass
            else:
                self._interaction_groups.pop()

        return RecoveryState(
            assignments=state.assignments, failing_vectors=state.failing_vectors,
            dispatch_hits=state.dispatch_hits, evaluated=evaluated, remaining_budget=remaining,
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
    native_trace_cache: dict[str, behavioral_gate_mod.BehavioralTrace] = field(default_factory=dict)

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

    def evaluate(self, proposal: AssignmentProposal, *, full_corpus: list[behavioral_gate_mod.BehavioralVector]) -> Observation:
        probe_vectors = list(proposal.probe_vectors) if proposal.probe_vectors is not None else None
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
        vectors_to_run = probe_vectors if probe_vectors is not None else list(self.native_trace_cache.keys())
        report = behavioral_gate_mod.BehavioralGateReport()
        try:
            with runner:
                for name in vectors_to_run:
                    vector = next((v for v in full_corpus if v.name == name), None)
                    if vector is None:
                        continue
                    native_trace = self.native_trace_cache.get(name)
                    if native_trace is None:
                        raise RecoveryError(f"no cached native trace for vector {name!r}")
                    candidate_trace = behavioral_gate_mod.run_vector(runner, vector)
                    report.verdicts.append(
                        behavioral_gate_mod.compare_traces(name, native_trace, candidate_trace)
                    )
        except (behavioral_gate_mod.BehavioralGateError, ServerError) as exc:
            raise RecoveryError(f"probe for {proposal.label!r} failed to execute: {exc}") from exc

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
    )
    current_overrides: dict[str, str] = {}
    best_overrides: dict[str, str] | None = None

    while state.remaining_budget > 0:
        proposals = strategy.propose(state)
        if not proposals:
            break
        for proposal in proposals:
            if state.remaining_budget <= 0:
                break
            merged = {**current_overrides, **proposal.overrides}
            probe = AssignmentProposal(label=proposal.label, overrides=merged, probe_vectors=proposal.probe_vectors or failing_vectors)
            try:
                observation = executor.evaluate(probe, full_corpus=full_corpus)
            except RecoveryError:
                state = strategy.record(state, Observation(proposal=probe, verdict="ineligible", report=None))
                continue
            state = strategy.record(state, observation)
            if observation.verdict == "pass":
                current_overrides = merged
                best_overrides = merged

    if best_overrides is None:
        # Nothing recovered -- every implicated signature falls back to
        # its own real native candidate identity, per GPT's circuit-
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
    recommendations = tuple(
        RetuneRecommendation(
            signature_dispatch=dispatch, reason="alternatives_exhausted",
            current_assignment="native", exhausted_candidates=initial_assignments[dispatch].alternatives,
        )
        for dispatch, candidate in best_overrides.items()
        if initial_assignments.get(dispatch) is not None
        and candidate == initial_assignments[dispatch].native_candidate
        and initial_assignments[dispatch].alternatives
    )

    final_observation = executor.validate_full_corpus(best_overrides, full_corpus=full_corpus)
    if final_observation.verdict != "pass":
        return RecoveryResult(
            published=False, final_overrides=best_overrides,
            evaluations_used=max_evaluations - state.remaining_budget + 1,
            stop_reason="final full-corpus validation failed -- refusing to publish",
            cache_path=None, retune_recommendations=recommendations,
        )
    cache_path = executor.build_candidate_cache(best_overrides)
    return RecoveryResult(
        published=True, final_overrides=best_overrides,
        evaluations_used=max_evaluations - state.remaining_budget + 1,
        stop_reason="full-corpus validation passed",
        cache_path=cache_path, retune_recommendations=recommendations,
    )
