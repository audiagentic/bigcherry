# HTR01: vacuous-pass bug found by real benchmarking, honest re-validation

Real, raw evidence for a serious bug found in HTR01's recovery search, and
the genuine (not vacuous) real-hardware re-validation that followed the
fix. Recorded here so the full sequence -- false claim, real detection,
root cause, fix, honest result -- is auditable from raw data, not just
narrated in plan-item notes.

## Sequence of events

1. **Three "successful" real-hardware recovery runs were reported** against
   the captured HI141 regression (`hi141-proof-20260829-2231`). The third
   claimed 17 of 20 signatures recovered to real alternative candidates,
   full-corpus validation "passed", and was marked complete.

2. **User asked for a real speed benchmark** of the "recovered" cache
   (`Did we do benchmarks with tunes to see how we went`). Running the
   actual `tg128` throughput test the correctness gate was supposed to
   protect exposed a **27% regression** and a draft-acceptance collapse
   (`original_vacuous_benchmark_showing_27pct_regression.txt`: draft_n=149,
   accepted=89, ~74 tok/s vs native's 101 tok/s) -- comparable in severity
   to the *original* HI141 defect this whole mechanism exists to catch.

3. **Root cause, found by direct re-invocation of
   `AssignmentExecutor.validate_full_corpus()`** against the exact
   committed overrides, reconstructed authoritatively from the cache file
   on disk (`vacuous_pass_diagnostic.txt`): the re-check returned
   `verdict: pass` with `report: {"hard_fail": false,
   "needs_throughput_adjudication": false, "vectors": []}` -- an **empty**
   verdicts list. `AssignmentExecutor.evaluate()`'s vector-matching loop
   compared real `BehavioralVector` **objects** (what every real caller
   supplies) against an implicit assumption that `name` was always a
   **string** (`v.name == name`) -- never true when `name` is actually an
   object. `vector` was therefore always `None`, every loop iteration
   silently `continue`d, and `report.verdicts` stayed empty for **every
   single probe in all three prior runs**. `hard_fail` /
   `needs_throughput_adjudication` are both `any(...)` over that empty
   list, so every probe vacuously "passed" without ever running one real
   behavioral comparison.

4. **Fixed** (`tools/bigcherry/tuning/recovery.py`, commit `ed3f5eb`):
   normalized vector matching to names up front regardless of source;
   added two fail-closed guards (`original_vacuous_recovery_run.txt` is
   what the *buggy* code produced -- compare its structure to the honest
   run below). Added `AssignmentExecutorEvaluateRealVectorMatchingTests`,
   the first tests in the suite to call the real `evaluate()` rather than
   a fake executor that bypassed it entirely -- the structural reason this
   bug shipped three times undetected.

5. **Honest re-run** (`validate_recovery_honest.txt`): with the fix in
   place, recovery correctly isolated **only the actual guilty signature**
   (`cd3b5f5bd371...`) -- 19 of 20 originally-promoted signatures were
   "never touched" (correctly determined to have never been implicated at
   all, not silently reassigned). All 5 of the guilty signature's real
   alternatives were tried and genuinely failed real behavioral validation
   this time; it correctly fell back to native.

6. **Honest benchmark** (`benchmark_recovered_honest.txt`): the resulting
   cache measured `draft_n=107, draft_n_accepted=100` -- **bit-identical**
   to native's own draft trace -- and `102.29 tok/s` vs native's
   `101.46 tok/s` (`+0.82%`, parity within noise). A safe, small, honest
   result: the one truly guilty signature now costs nothing extra (it is
   genuinely running native code for that op), and the other 19 keep their
   real, already-safe tuned speed.

## Files

- `validate_recovery.py` -- the real-hardware validation driver script
  (runs `run_recovery()` directly against the captured campaign data).
- `benchmark_recovered_cache.py` -- direct server-timing tg128 benchmark
  driver (native vs a given `GGML_HIP_DISPATCH_CACHE`).
- `original_vacuous_recovery_run.txt` -- the buggy run's real captured
  stdout (compare against `validate_recovery_honest.txt`'s structure).
- `original_vacuous_benchmark_showing_27pct_regression.txt` -- the real
  benchmark numbers that exposed the bug.
- `vacuous_pass_diagnostic.txt` -- the direct `validate_full_corpus()`
  re-invocation proving the empty-verdicts root cause.
- `validate_recovery_honest.txt` -- the real, fixed re-run's output.
- `benchmark_recovered_honest.txt` -- the real, fixed cache's real speed.
- `recovery-candidate.cache` -- the actual binary dispatch cache the
  honest run produced (real artifact, not a description of one).

## Second correction: GPT deep-dive found a real overclaim in the "honest" run

The user's closing requirements for this item required a genuine GPT deep-dive
of the actual committed logs (not a design conversation). That review
(session `ses_330ae3c055084f38`) found the "honest" run above still
overclaimed: `RetuneRecommendation.exhausted_candidates` reported the
signature's entire 5-alternative catalog as "exhausted" even though the
repair phase's v1 implementation only ever tried `alternatives[0]` -- one
attempt, not five. It also found a real evaluations-count off-by-one and
recommended a stronger cardinality guard (exact requested-vs-actual vector
name/order match, not merely "nonzero verdicts").

Fixed (commit `791729a`): the repair phase now walks every real,
already-measured alternative in order (still v1-scoped -- no new GPU/timing
measurement) until one passes full-corpus validation or all are genuinely
exhausted, tracking exactly which were attempted
(`BoundedPairedBisectionStrategy.tried_alternatives`).

## Final, fully-accurate re-run

- `validate_recovery_final_accurate.txt` -- with the fix applied, all 5
  real alternatives for the guilty signature (`cd3b5f5bd371...`) were
  genuinely tried and genuinely rejected (`evaluations_used: 15`, matching
  8 isolation probes + 1 baseline + 5 real alternative trials + 1 final
  validation). Correctly reverted to native; 19 of 20 signatures untouched;
  `retune_recommendations` now accurately lists all 5 real attempts.
- `recovery-candidate-final.cache` -- the resulting cache artifact.
- `benchmark_recovered_final.txt` -- a fresh benchmark of this exact
  artifact: `draft_n=107, draft_n_accepted=100` (bit-identical to native)
  across all 5 reps, `102.11 tok/s` vs native's `101.46` (`+0.64%`,
  consistent with the earlier `+0.82%` run within measurement noise).

## What this does NOT cover yet

Only the single pinned HI141 regression vector was exercised (HTR03's
broader corpus work is still deferred). No other real-world prompt/
scenario has been checked against this recovered cache. This bundle is
evidence for the recovery MECHANISM's correctness after the fix, not a
claim that the resulting cache is validated against arbitrary production
traffic.
