# rad0034_prefill_concurrency

## Summary

VTP39 step 3 adds the generic engine option `--max-prefill-seqs N` and the matching optional
serving v2 tuning field `max_prefill_seqs`. A value of **0 (default)** imposes no limit; the
existing scheduler step planner, priority order, token budgets and KV allocation decisions retain
their previous control flow. Values above zero cap long-running prompt prefills in the RUNNING set.
This controls admission, not the compute slices of already-running sequences.

The evidence motivating the option is in
`docs/evidence/2026-11-14-tp3-long-context/README.md` and the follow-up
`docs/planning/active/run-dense-27b-virtual-tp/VTP39.md`. This is not a new default or a production
recommendation: the scheduler patch requires G4 requalification and hardware ABBA against the
full-cache client before the owner selects a value.

## Semantics

- A RUNNING sequence counts while its prompt has uncomputed tokens, including after a preemption
  when its KV is being recomputed. A preempted WAITING sequence is not RUNNING. Admission
  rechecks the cap on each attempt, in the existing priority-then-arrival queue order.
- `N=1` serialises the long-prefill admissions. If a candidate would exceed the cap,
  `admit_one` returns `kAdmitWait`: the request retains its queue position and the next
  candidate gets an admission attempt. A runner is not preempted because of this policy.
- A request is **exempt** if its *actual remaining* prompt can be computed and produce its first
  token in the admission step and the remaining work fits the normal shared-token step budget.
  A cached prefix may make a previously long request eligible. The checkpoint split, busy-chunk
  limit, current budget and KV allocation must still allow it to finish *in that step*; otherwise
  it counts as a long prefill. An exempt short prefill can execute beside N longer prefills.
- This patch limits **admission**, not KV reservation: it deliberately keeps the engine's
  existing per-chunk KV growth and whole-recompute admission tests. It does not reserve the
  entire future prompt up front. Whole-prompt reservation would be a separate KV allocator
  change, potentially reducing short-prompt bypass and cache reuse; its benefit must be
  evaluated separately rather than implied by this concurrency cap.
- A decoding runner uses no prefill slot, may continue normally, and retains its existing
  priority/arrival order in each step. `--max-num-seqs` and available card slots remain hard
  independent limits; the exemption never creates a slot or increases the token budget.
- `--prefill-interleave` still holds prefill rows **and admission** while decode-only steps
  run. The cap is evaluated when the hold ends; it does not end or shorten the hold.
- `--busy-prefill-chunk` still computes `behind_pf`, decode reserves, and `admit_waits`
  from pending work. Deferred admissions may leave a reserve unused; existing capped-chunk
  give-back remains in control. An admission must really finish under the busy cap to qualify
  for the short exception.
- `--prefill-span` still chooses wide steps only for a lone eligible prefill. The existing
  `admit_possible` predicate is intentionally conservative: an available sequence/card slot
  with any waiting request may keep the span disabled even if that wait is cap-blocked. This
  preserves the short-bypass opportunity and the span's single-prefill guarantee.
- An overdue preempted request still passes `preempted_can_prefill` (whole recompute plus
  headroom) and `preempt_wait_overdue`; while the cap is occupied, its eviction fallback does
  not force an unrelated decoder off for an admission that cannot proceed. Once a long runner
  completes, the normal fallback may reclaim or force a permissible decoder victim. With no
  running prefill, the cap cannot prevent an overdue request being considered, so it does not
  introduce a parked idle engine.
- Prefix hits are resolved before cap admission. A refused hit that carries a checkpoint slot
  is undone/held by the existing failure cleanup; retries must not restore a stale snapshot.

`radiance:prefilling_sequences` is a gauge of RUNNING prompts still in prefill/recompute.
`radiance:prefill_admissions_deferred_total` counts cap deferral decisions (not distinct
requests); `scheduler: --max-prefill-seqs` prints the resolved setting at startup.

## Outcome

**Untested (not promoted).** Host-only tests and the patch application check are required
development checks, not G4 evidence or hardware approval. No serving configuration sets this
option; production behaviour remains unlimited.

## Requalification and owner gate

The owner must produce a qualification record for the pinned source/patch-set identity and
pass G4 via `bigcherry patch-gates`, with the applicable compile/correctness evidence
(including the changed scheduler and serving contract). Separately measure the VTP39 full-cache
client at eight concurrent long prompts under `N=0`, `N=1`, `N=2`, with repeated ABBA
order and an identical build/input set. Record each request's first-token latency, mean TTFT,
preemptions, forced preemptions, KV occupancy, and short-request latency; retain command lines
and machine-readable run artifacts in managed evidence, then record the owner's default choice.
