---
id: HTR01
order: 0
plan: hip-tune-recovery
state: pending
created-at: '2026-08-29T13:28:09.305813+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
---

# Bounded recovery search: pluggable RecoveryStrategy + AssignmentExecutor for HI143 hard-fail campaigns

## Description

HI143's behavioral gate is validated (two real hardware runs: safe cache promoted, unsafe cache correctly blocked) but is all-or-nothing -- a single hard_fail anywhere in a promoted cache discards the ENTIRE campaign's yield, including every other candidate's legitimate real speed win. A real fresh campaign (run-id hi141-proof-20260829-2231, 2026-08-29) demonstrated this: 41 provisional winners, one real regression, zero candidates shipped.

This item builds a bounded recovery search that runs on data the failed campaign already collected -- NO new GPU tuning/timing measurements -- to recover as much of a failed campaign's yield as safely possible before falling back to native for the minimum necessary set of signatures.

Design adversarially negotiated with GPT (session ses_330ae3c055084f38, 2 rounds, 2026-08-29). GPT's key correction to the original bisection-only proposal: HI141 already proved candidate-set safety is NON-MONOTONIC (the guilty candidate alone fails; a larger, different ensemble containing it can pass -- the 'candidate-mix masking' finding). Simple binary-search-for-the-guilty-candidate assumptions are therefore invalid on their own and must be one strategy among several, not the architecture.

## Steps

1. Define the RecoveryState / RecoveryStrategy.propose() / AssignmentExecutor.evaluate() contract (see code_samples) -- this is the pluggable boundary the user explicitly required (bisection now, other heuristics such as op-shape grouping or GPT's value-aware best-first repair search later, without redesigning gate/cache plumbing).
2. Implement v1's RecoveryStrategy as bounded delta-debugging (ddmin-style): split the promoted non-native set into groups, swap groups back to forced-native, re-run ONLY the failing corpus vectors (not the full corpus) against a CACHED native trace (see HTR02-adjacent optimization: capture native traces ONCE per corpus at the start of recovery, reuse for every probe -- avoids a second real model load on every single diagnostic probe). Handle GPT's explicit outcome matrix: half A fails/B passes -> recurse into A; both halves fail -> track as multiple independent failing groups; both halves pass but the union fails -> record as a genuine interaction group, do NOT falsely blame either half, switch to complement/group-level substitution rather than continuing plain bisection on it; identical config non-deterministic across repeats -> abort recovery for that vector (infrastructure/nondeterminism, not a candidate defect).
3. Implement AssignmentExecutor: for each proposal, verify candidate eligibility (real tune measurement + signature match + correctness evidence, generating it on-demand if the alternative was measured but never evidenced -- do NOT treat every raw ranking_decisions entry as automatically substitution-eligible), splice into a fresh provisional cache, run the targeted probe (failing vectors only, against cached native), and only after a proposed repaired assignment looks promising, run the FULL corpus (not just the failing vectors) against cached native as the promotion-eligibility check.
4. Alternative selection order: for each implicated signature, prefer the next-best-ranked already-measured candidate from that signature's own ranking_decisions (free lookup, no retune) over immediately falling back to native -- ordered by least predicted performance loss (effective_us), per GPT's 'value-aware' framing, not simply by tune-time rank position.
5. Circuit breaker: cap TOTAL behavioral candidate-server evaluations (GPT recommended v1 default: max_recovery_evaluations=24, reserve_final_validation=1), not abstract iteration count -- configurable, and the actual count + stop reason recorded in the campaign receipt. On cap: set all unresolved implicated/interaction signatures to native, preserve all unrelated already-clean winners, spend the reserved evaluation on exactly one FULL HI143 corpus validation of that resulting partial assignment. If that final validation passes, publish the partial cache (real, safe, if reduced, yield). If it fails, abort publication entirely (today's status quo) -- NEVER ship on the basis of a subset/individual-vector pass, only ever on a full-corpus PASS of the exact artifact being published.
6. Collect ALL failing corpus vectors from the initial gate run before starting recovery search, not just the first hard_fail encountered -- avoids repairing vector 1, reloading, discovering vector 2 broken too, repairing again (GPT's explicit additional requirement).
7. Treat HI143's full three-state verdict contract (hard_fail / exact_pass / behavior_changed-needs-throughput-adjudication) as the oracle throughout, not simplified token-equality -- behavior_changed still blocks recovery from declaring success (throughput adjudication remains unimplemented, tracked in HI143, not duplicated here).
8. Wire into workflow.py's _stage_replay_validate: on TuneCampaignError from a hard_fail, invoke recovery instead of immediately propagating, and only propagate if recovery itself exhausts its budget without a passing final validation.

## Detailed Solution & Technical Design

Plugin contract (GPT-specified, adopted verbatim):

RecoveryState:
  current full assignment
  eligible alternatives/rankings (from ranking_decisions)
  failing vectors
  dispatch-hit sets
  prior evaluated assignments + verdicts (avoid re-probing an identical assignment)
  remaining evaluation budget

RecoveryStrategy.propose(state) -> list[AssignmentProposal]
  -- owns: ddmin/group splitting, interaction handling, alternative ordering,
     future value-aware/op-shape heuristics. NEVER mutates caches or declares
     a proposal safe.

AssignmentExecutor.evaluate(proposal) -> Observation
  -- owns: candidate eligibility/trust checks, correctness evidence
     generation, cache construction, cached-native comparison, behavioral
     execution, final full-corpus validation, artifact/receipt persistence.
  -- ONLY the executor (the oracle) can produce a behavioral PASS verdict.

Critical boundary (GPT): strategy proposes, executor/oracle decides. This is
what lets v1's ddmin and a later best-first value-aware search share the same
plumbing without redesign.

GPT's rejected alternative (explicitly considered and dismissed): incremental
candidate-by-candidate pre-admission testing before full-cache promotion.
Rejected because it turns HI143's current O(1) cheap-path (optimistic
full-cache gate first) into O(N) on every successful campaign too, is
ordering-dependent, can reject a candidate individually even where a later
full ensemble would have masked it safely (a false rejection, not just an
inefficiency), and STILL cannot eliminate the need for final full-cache
validation. Retain: optimistic full-cache gate first (cheap on the common
path) -> bounded recovery search only on actual failure.

## Code Samples & Guidance



## Files

tools/bigcherry/tuning/workflow.py, tools/bigcherry/tuning/behavioral_gate.py, new: tools/bigcherry/tuning/recovery.py (or similar), tools/tests/tuning/

## Validation

Offline unit tests for the RecoveryStrategy contract (ddmin split/recurse/interaction-group outcomes per GPT's matrix) and AssignmentExecutor (eligibility checks, cache splicing, budget accounting, partial-cache circuit-breaker behavior) using fakes/mocks, matching HI143's own test discipline (behavioral_gate_mod.run_vector mocked, no real GPU needed for logic tests). Real-hardware validation: re-run a campaign that reproduces a real hard_fail (e.g. reusing hi141-proof-20260829-2231's exact conditions) and confirm recovery either (a) finds a passing alternative assignment and ships a reduced-but-real cache, or (b) correctly falls back to native for the minimum necessary signatures and ships everything else, or (c) correctly aborts publication entirely if recovery cannot converge within budget -- never a false-positive publish.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

See HTR02 for the durable BehavioralFailureWitness persistence layer this item's Observation records should feed (a separate, adjacent concern GPT was explicit should not be conflated with the recovery search itself). See HI143 for the existing gate this wires into and its still-open throughput-adjudication gap. See HI141 for the real regression and candidate-mix-masking evidence this whole design responds to.

IMPLEMENTED (2026-08-29): tools/bigcherry/tuning/recovery.py written per the design above (RecoveryState/AssignmentProposal/Observation/RecoveryStrategy protocol/BoundedDeltaDebugStrategy/AssignmentExecutor/run_recovery), wired into workflow.py's _stage_replay_validate (on hard_fail or needs_throughput_adjudication, attempt bounded recovery before raising; only re-raises if recovery itself is unavailable or exhausts its budget without a publishable assignment). Cache splicing reuses the existing, HI22-validated seed-override mechanism in replay.build() (a manifest-bound {dispatch: stable_name} override file) rather than hand-rolling new measurements-row mutation logic. 9 offline unit tests added (tools/tests/tuning/test_recovery.py) covering the ddmin outcome matrix and run_recovery's budget/circuit-breaker/all-failing-vectors-collected-upfront behavior with a fully mocked executor -- all passing, plus the full existing workflow/behavioral_gate/server_runner suite (52 tests) still passing after the wiring change.

HTR04 FOLLOW-UP LANDED ALONGSIDE (2026-08-29, per GPT's explicit instruction that this be exposed NOW, not deferred): run_recovery emits RetuneRecommendation records (reason='alternatives_exhausted' only, v1) whenever a signature ends up forced to native despite having had real measured alternatives -- pure structured evidence written to workdir/recovery-result.json, asserted in code comments to change nothing automatically. GPT's explicit, adopted principle: 'HTR01 failure != retune' -- see HTR04 for the full escalation design (three of its four reasons are explicitly NOT implemented yet, deferred pending real HTR01 usage history).

KNOWN LIMITATIONS, not yet closed (documented in code, not hidden):
1. Precise dispatch-hit scoping via GGML_HIP_DISPATCH_HIT_LOG cross-referencing is NOT wired -- recovery currently searches over every promoted non-native signature rather than only the ones the failing vector(s) actually exercised (workflow.py's _stage_replay_validate sets dispatch_hits=frozenset(assignments) as a placeholder). Correct but less efficient than the design's intent; a real follow-up, not a correctness bug.
2. Alternative candidate eligibility (does an alternative already have correctness evidence?) is discovered lazily when AssignmentExecutor tries to build a cache with it (via replay.build()'s own existing checks), not filtered in advance -- an ineligible alternative simply raises RecoveryError for that one proposal and the search continues, which is safe but means some evaluation-budget waste on doomed proposals.
3. NOT YET real-hardware validated -- offline logic tests only so far. Real-hardware validation (re-running a campaign that reproduces a genuine hard_fail, e.g. reusing hi141-proof-20260829-2231's exact conditions, and confirming recovery either ships a reduced-but-real cache or correctly falls back/aborts) remains open per this item's own validation section.

## Change Log

- 2026-08-29T13:28:09.305813+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260829_132841_planned-the-next-improvement-t_2822
- 2026-08-29T13:28:41.371872+00:00 (updated-by): Updated: section:ledger-events
- 2026-08-29T13:47:39.236165+00:00 (updated-by): Updated: section:notes
