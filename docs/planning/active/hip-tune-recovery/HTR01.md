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

REAL-HARDWARE VALIDATION COMPLETE (2026-08-30, run against the actual captured hi141-proof-20260829-2231 failure -- real binary, real manifest, real dispatch_db, real server loads, no simulation). Two real bugs found and fixed during this validation (both committed): (1) AssignmentExecutor never wired dispatch_db/require_winner_verification into replay.build(), causing it to reject every winner as unevidenced rather than checking real evidence; (2) the literal string 'native' is not a valid replay.build() seed-override target (it requires a real manifest candidate identity like 'mmvq:native:v1') -- SignatureAssignment now carries a real native_candidate field extracted from the signature's own ranking_decisions/candidates list. A related fragility was also fixed: any RecoveryError (including a merely-ineligible candidate) was being mapped to verdict='unstable', which the strategy treats as a hard abort for non-determinism -- split into a distinct 'ineligible' verdict that consumes budget and moves on without concluding anything about a signature's guilt.

With both fixes applied, the real run: reproduced the exact original hard_fail (native draft=[107,100] vs candidate draft=[145,90], identical to the original finding), ran a real bounded recovery search using only 4 of its 24-evaluation budget, and its final full-corpus validation genuinely passed -- confirmed via a real dispatch cache written to disk (recovery-candidate.cache). The mechanism (bisection, budget accounting, mandatory final full-corpus gate, safe circuit-breaker fallback) is proven sound and safe on real hardware.

IMPORTANT REAL FINDING (not a bug, a genuine scope gap): the search recovered ZERO real performance -- all 20 originally-promoted signatures ended up forced back to native, not just the one guilty signature. Root cause: every alternative candidate listed in each signature's own ranking_decisions (the 'free lookup, no retune needed' data source this item's whole design was built on) turned out to be correctness-evidence-INELIGIBLE -- normal campaigns only run hi80_generate_correctness_evidence against the WINNING candidate per signature, never the other ranked alternatives, so replay.build()'s RV49 gate correctly rejected every substitution attempt as unevidenced. The strategy exhausted all alternatives almost immediately (hence only 4 evaluations used, not the expected many-more for a real bisection+substitution search) and converged safely but conservatively to all-native fallback.

This means HTR01 as built today reliably prevents shipping a bad cache (the safety goal, fully achieved) but does NOT yet reliably recover real yield (the efficiency goal, NOT achieved in this real test) -- because the 'no new measurement needed' assumption for alternatives was wrong: alternatives need at minimum on-demand correctness-evidence generation (real but modest CPU/GPU test-backend-ops work, NOT a GPU timing retune) before they can actually be used. Sent to GPT for adversarial review of the fix (on-demand per-alternative correctness-evidence generation inside AssignmentExecutor, vs. pre-filtering the alternatives list to only already-evidenced candidates) -- response pending. This is real follow-up scope, not something to declare done; recorded here rather than silently shipped as a full success.

LAZY CORRECTNESS-EVIDENCE GAP CLOSED (2026-08-30): implemented the locked design GPT produced when asked to design this concretely (session ses_330ae3c055084f38, user explicit instruction to combine the efficiency fix with capturing its broader analytical value rather than treating them separately). Full implementation:

- schema 9 (sql/dispatch-db.sql + sql/migrations/0009_correctness_evidence_analytics.sql): correctness_evidence_origin table (reason: promotion_winner|recovery_alternative|manual_analysis) + four new nullable output-digest columns on correctness_evidence_seed, capturing HI83's existing backend1_digest/backend2_digest (previously parsed and discarded) for exact numerical-family clustering.
- correctness_evidence.py: NativeSeedEvidence/collect_native_seed_evidence/collect_candidate_seed_evidence split from collect_seed_evidence, enabling native-baseline reuse across multiple candidates for the same signature (first alternative pays native+candidate per seed, every later one pays candidate only) -- confirmed by a real test asserting exactly this (6 runs first candidate, 3 runs second).
- hi80_generate_correctness_evidence.py: new generate_for_candidate() primitive (explicit candidate_name + EvidenceOrigin + optional native_seed_cache) as the single authoritative evidence-generation implementation; generate_for_row() is now a thin wrapper preserving its exact original behavior.
- recovery.py: AssignmentExecutor.ensure_correctness_evidence() called from evaluate() before every cache-build attempt, lazily qualifying exactly the alternative about to be tried, under its own separate budget (max_new_correctness_candidates, default 16 per GPT's real-data-informed revision from an initial 12), skipping native (never needs its own evidence), failing closed (RecoveryError) on budget exhaustion or genuine correctness failure.

Also simplified _load_signature_assignments while fixing this: discovered row['native'] in real promoted.jsonl data already IS the real native candidate stable_name directly (confirmed against real hi141-proof-20260829-2231 campaign data) -- removed the earlier commit's unnecessarily complex regex-based derivation from ranking_decisions/candidates lists.

18 hi80 tests + 14 recovery tests (both including new coverage for this change) pass; full 2493-test suite otherwise unaffected. NOT yet re-validated on real hardware against the actual hi141-proof-20260829-2231 failure with this fix in place -- that is the natural next step to confirm the fix actually recovers real yield (not just that it's logically correct offline), tracked as open follow-up.

SECOND REAL-HARDWARE RUN FOUND A DEEPER BUG (2026-08-30): re-ran the exact same captured hi141-proof-20260829-2231 failure with the lazy correctness-evidence fix in place (migrated that campaign's tune.sqlite to schema 9 first). Result was BYTE-IDENTICAL to the pre-fix run: same 4 evaluations, same all-native fallback for all 20 signatures. This proves the correctness-evidence fix never actually got exercised -- the search never reached the alternative-proposal stage at all.

ROOT CAUSE (found by re-reading BoundedDeltaDebugStrategy, a real, structural bug distinct from the two already-fixed real-hardware bugs): propose() returns BOTH halves of a bisection split in one call, but record()'s pairing logic only correctly handles the FIRST half processed -- it pops the parent group unconditionally and pushes back only that first half's own outcome. When the SECOND half is evaluated and recorded, self._pending_groups[-1] no longer matches it (it's now the first half's sub-group), so the second half's outcome falls through un-paired. Consequence: the critical 'both halves individually pass but their union fails -> genuine interaction group' case (exactly what HI141's own candidate-mix-masking finding says is real) can never be detected, and in practice the search degenerates toward rapidly forcing large swaths of the cache to native without ever isolating a small enough group to try real alternatives on.

This exact gap was FLAGGED IN THE ORIGINAL CODE COMMENT ('that is only detectable once both halves of one split have been observed, tracked by the caller... not here') but the promised caller-side pairing logic was never actually implemented in run_recovery -- an honest miss, not a new discovery of an unknown problem.

Sent to GPT (session ses_330ae3c055084f38) for the concrete fix, explicitly asking: (1) correct pairing state-machine design (RecoveryState-level pending-pairs structure vs. orchestrator-evaluates-both-before-one-record() vs. something else), (2) whether this real case (evidently most/all of 20 signatures implicated together) means bisection was never going to be efficient here regardless of the bug, (3) given this is the SECOND real bug found in this exact strategy (after the native-name string bug), whether a simpler, more obviously-correct v1 strategy (e.g. linear one-at-a-time, no bisection) should replace bounded delta-debugging for now, deferring interaction-group handling to v2. Response pending.

STATUS: HTR01's correctness-evidence lazy-generation fix is implemented and unit-tested correctly in isolation, but REMAINS REAL-HARDWARE UNVALIDATED as an end-to-end fix, because the strategy bug it depends on reaching never lets it run. Do not consider HTR01 complete or the correctness-evidence fix proven until this pairing bug is fixed and a third real-hardware run actually exercises ensure_correctness_evidence() for at least one real alternative.

PAIRING BUG FIXED, v2 REDESIGN LANDED (2026-08-30): implemented GPT's locked fix (session ses_330ae3c055084f38) -- BoundedDeltaDebugStrategy renamed and rewritten as BoundedPairedBisectionStrategy. RecoveryStrategy protocol changed from propose()->list[AssignmentProposal] to propose()->RecoveryAction|None (bundling both sibling proposals of a split atomically) and record(state, ActionObservation) receiving BOTH outcomes together, never one at a time. Orchestrator (run_recovery) no longer mutates any working baseline itself on a diagnostic PASS -- RecoveryState now carries committed_overrides, updated EXCLUSIVELY by strategy.record(), only during an explicit two-phase isolate/repair design (isolate: paired bisection over H with GPT's exact outcome matrix, budget-aware early-stop-to-repair; repair: conservative all-native baseline probed once, then the single best already-measured alternative tried per implicated signature, one at a time).

19 tests (11 new/rewritten) now drive the REAL strategy through the REAL run_recovery orchestrator against a scripted ground-truth fake executor (not unit-level state pokes, which is exactly the testing style that let the original pairing bug slip through) -- covering GPT's full mandatory checklist: single-culprit isolation reaching the alternative stage, A-only/B-only/both-independently-fail/genuine-interaction outcomes, identical parent baseline between siblings, diagnostic-PASS-never-commits, alternative-PASS-commits-only-via-record, budget cannot afford an unaffordable split, unstable/ineligible-native abort paths. Full 2498-test suite otherwise unaffected (same 5 pre-existing unrelated environment failures).

STATUS: ready for a THIRD real-hardware run against the same captured hi141-proof-20260829-2231 failure. Acceptance criterion (GPT's own words): the reproducer must reach at least one real alt:* proposal, invoke generate_for_candidate() for real, retain non-native winners if a safe repair exists, and publish only after full-corpus PASS. Do not consider HTR01 validated until that third run actually demonstrates this end-to-end on real hardware -- two real-hardware runs have already found two real, distinct bugs that only real hardware (not offline tests alone) exposed, so a third real run is the actual bar, not optional polish.

THIRD REAL-HARDWARE RUN: FULL SUCCESS (2026-08-30). Two more small real bugs found and fixed en route (both from this same run's own failures, not new investigation): (1) my own validate_recovery.py script still referenced the renamed BoundedDeltaDebugStrategy class -- fixed, trivial; (2) DEFAULT_MAX_NEW_CORRECTNESS_CANDIDATES was still 12 in code despite HTR01's own notes already recording GPT's real-data-informed revision to 16 -- the revision was written down but never actually applied; a real run then genuinely exhausted the budget at 12, catching the miss. Fixed to 16.

With all three real-hardware-found bugs fixed (native-name string bug, atomic-pairing bug, budget-not-applied bug), a clean run against the exact captured hi141-proof-20260829-2231 failure produced:

published: True, stop_reason: full-corpus validation passed, evaluations_used: 25, real cache written to disk.

17 of 20 originally-promoted signatures were reassigned to a REAL, different, working alternative candidate (not bulk native fallback). Critically, the ORIGINALLY-GUILTY signature (cd3b5f5bd371..., the exact dispatch this entire investigation started from) was correctly reassigned from the bad mmvq:q8_0:w4:nw8:rpb1:sk0:v1 to a genuinely different, safe candidate -- mmvq:q8_0:w4:nw6:rpb1:sk0:v1 (nw6, not nw8) -- which then passed the same real HI141 regression vector. Only 3 of 20 signatures exhausted their alternatives and correctly fell back to native, each with a structured RetuneRecommendation (alternatives_exhausted) attached, exactly as HTR04 designed -- purely informational, nothing acted on automatically.

This satisfies GPT's own stated acceptance criterion in full: the reproducer reached real alt:* proposals, invoked generate_for_candidate() for real (confirmed via live test-backend-ops processes during the run), retained non-native winners (17/20, not 0/20 as in the two prior real runs), and published only after a genuine full-corpus PASS.

STATUS: HTR01 is now considered REAL-HARDWARE VALIDATED end-to-end -- both the safety property (never ships a regression -- proven across all runs) and the efficiency property (recovers real yield, not just safe-but-useless native fallback -- proven by this run) are demonstrated on real hardware against the real HI141 regression this whole chain of work (HI121/HI136/HI141/HI143/HTR01) was built to solve. Remaining open work is tracked separately in HTR02 (failure-witness persistence), HTR03 (corpus/applicability configurability), HTR04 (retune escalation, still deferred), and HTR05 (multiplicity-correction study, still deferred) -- none of which block considering HTR01 itself complete and proven.

CRITICAL RETRACTION (2026-08-30): the 'THIRD REAL-HARDWARE RUN: FULL SUCCESS' entry above is WRONG and is retracted. A real tg128 benchmark run against the 'successfully recovered' cache (user's own explicit follow-up request: 'Did we do benchmarks with tunes to see how we went') measured a genuine 27% throughput regression and draft-acceptance collapse (89/149 accepted, ~60%) comparable in severity to the ORIGINAL HI141 defect this whole chain of work exists to catch -- despite the recovery run reporting 'full-corpus validation passed'.

Root cause found by direct re-invocation of executor.validate_full_corpus() against the exact committed overrides (authoritatively reconstructed from the cache file on disk, ruling out a stale-file theory): the re-check returned verdict='pass' with report={'hard_fail': false, 'needs_throughput_adjudication': false, 'vectors': []} -- an EMPTY verdicts list. AssignmentExecutor.evaluate()'s vector-matching loop compared real BehavioralVector OBJECTS (what every actual caller supplies) against an implicit assumption of NAME STRINGS (`v.name == name`), which is never true when `name` is actually an object -- so `vector` was always None, every iteration silently `continue`d, and NO real behavioral comparison EVER ran, in ANY of the three prior 'successful' real-hardware validation runs. Every reported 'pass' throughout this entire investigation was vacuous (hard_fail/needs_throughput_adjudication are both `any()` over an empty list).

This was NOT caught by any offline test because every test in the suite (including the 19 tests added specifically to validate the pairing-bug fix) exercised the RecoveryStrategy via a fake/mocked executor that bypassed AssignmentExecutor.evaluate() entirely -- a real, structural blind spot in the test suite's own design, now fixed (AssignmentExecutorEvaluateRealVectorMatchingTests, calling the REAL evaluate() with only ServerRunner/run_vector mocked).

Fix applied and committed: normalize vectors_to_run to names once, up front; added two fail-closed guards (a requested vector missing from full_corpus now raises rather than silently skipping; zero verdicts from any probe now raises rather than vacuously passing) so an empty-comparison class of bug cannot silently recur.

STATUS: HTR01's real-hardware validation is VOID and must be REDONE from scratch with this fix in place. State reverted from completed back to pending. User's explicit closing requirements for this item (2026-08-30), to be satisfied before re-closing: (1) a full real perf test -- actual tg128/throughput benchmarking of the recovered cache, not just the pass/fail behavioral check; (2) full, complete validation of ALL recovered candidates (not just the single pinned regression vector) and analysis of the real results; (3) a full GPT deep-dive assessment of the actual logs and run data (not a design conversation -- GPT must be given the real artifacts to review); (4) all of the above (logs, cache, benchmark data, analysis) must be committed to the repo so GPT can actually see it. None of these are satisfied yet. Do not mark this item completed again until all four are genuinely done.

## Change Log

- 2026-08-29T13:28:09.305813+00:00 (created-by): Created by agent

## Ledger-events


- chg_20260829_132841_planned-the-next-improvement-t_2822
- 2026-08-29T13:28:41.371872+00:00 (updated-by): Updated: section:ledger-events
- 2026-08-29T13:47:39.236165+00:00 (updated-by): Updated: section:notes
- chg_20260829_134919_the-tuning-system-can-now-reco_8101
- 2026-08-29T13:49:19.485229+00:00 (updated-by): Updated: section:ledger-events
- 2026-08-29T14:02:43.243098+00:00 (updated-by): Updated: section:notes
- chg_20260829_144231_closed-the-gap-found-in-real-h_7575
- 2026-08-29T14:42:31.033442+00:00 (updated-by): Updated: section:ledger-events
- 2026-08-29T14:42:48.499020+00:00 (updated-by): Updated: section:notes
- 2026-08-29T20:46:47.061609+00:00 (updated-by): Updated: section:notes
- 2026-08-29T21:01:00.462677+00:00 (updated-by): Updated: section:notes
- 2026-08-29T21:20:32.479110+00:00 (updated-by): Updated: section:notes
- 2026-08-29T21:20:36.595649+00:00 (state-transition): State: pending → completed
- chg_20260829_212045_proved-on-real-hardware-that-t_7840
- 2026-08-29T21:20:45.936383+00:00 (updated-by): Updated: section:ledger-events
- 2026-08-29T22:52:31.972214+00:00 (updated-by): Updated: section:notes
- 2026-08-29T22:52:36.227787+00:00 (state-transition): State: completed → pending
- chg_20260829_225254_found-and-fixed-a-serious-bug_2334
- 2026-08-29T22:52:54.507711+00:00 (updated-by): Updated: section:ledger-events
