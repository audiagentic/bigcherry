# rad0034_prefill_concurrency

## Summary

VTP39 step 3 adds the generic engine option `--max-prefill-seqs N` and the matching optional serving v2 tuning field
`max_prefill_seqs`. **0 (default)** imposes no limit: the scheduler's step planner, priority order, token budgets and KV
allocation keep their previous control flow. Above zero, at most N long prompt prefills are RUNNING at once; the rest
wait in priority-then-arrival order. This controls admission, not the compute slices of sequences already running.

Motivation: `docs/evidence/2026-11-14-tp3-long-context/README.md`; plan item VTP39.

## Outcome

Opt-in; no serving configuration sets it yet (the default is the owner's choice). Hardware results on
`qwen27b-tp3-int8-556-long-bf16` (eight ~100k-token prompts at once into a 624k-token cache, four ~450-token requests
behind them; two starts an arm; `docs/evidence/2026-11-15-vtp39-prefill-concurrency/README.md`):

| `--max-prefill-seqs` | mean first token, long prompts | first / last | short requests | preemptions (forced) |
|---|---|---|---|---|
| 0 | 434 s | 377 / 530 s | 321-338 s | 6 (4) |
| 1 | 208 s (-52%) | 50 / 368 s | 3.6-4.5 s | 0 (0) |
| 2 | 265 s (-39%) | 82 / 440 s | 121-123 s (before the short-prompt fix) | 0 (0) |

Card-free: `tests/sched_test.cpp` (195 cases, the eight `vtp39_*` ones among them) and the config parse test. The patch
state is `untested` in the recipe; promotion is a separate step.

## Detail

- A RUNNING sequence holds a cap slot while more than one shared step (`ChunkGeometry::shared_tok`) of its prompt is
  left, including a recompute after preemption. A preempted WAITING sequence is not RUNNING. Admission rechecks the cap
  on each attempt in the existing queue order; a refused candidate returns `kAdmitWait`, keeps its queue position, and
  the next candidate is tried. No runner is preempted because of the cap.
- A **short** prompt, whose remaining tokens (after any prefix hit) fit one shared step, is exempt: it is admitted
  beside N long prefills and holds no slot, even when a busy step leaves it only a share (an eighth of the budget) and
  it takes two or three steps to finish. The first version exempted only prompts finished in the admission step itself,
  so a ~450-token request beside a running 100k prefill (256-token share) waited for that whole prefill (about 45 s);
  `vtp39_short_prompt_over_one_share_still_bypasses` covers it.
- A long prefill stops holding a slot in its last shared step, so the next long prompt may start one step early.
- Admission, not KV reservation: the engine's per-chunk KV growth and whole-recompute admission tests are unchanged; no
  future prompt is reserved up front.
- Decoding runners use no slot and keep their priority/arrival order. `--max-num-seqs` and card slots stay hard limits;
  the exemption never creates a slot or adds budget.
- `--prefill-interleave` still holds prefill rows and admission while decode-only steps run; the cap is evaluated when
  the hold ends. `--busy-prefill-chunk` still computes its reserves from pending work; a deferred admission may leave a
  reserve unused, which the existing give-back returns. `--prefill-span` still spans only a lone eligible prefill.
- An overdue preempted request keeps `preempted_can_prefill` and `preempt_wait_overdue`; while every slot is held by a
  long runner, the eviction fallback does not force a decoder off for an admission that cannot proceed. With no running
  long prefill the cap cannot block it, so the engine is never parked idle.
- Prefix hits are resolved before the cap check; a refused hit's checkpoint slot is undone by the existing cleanup.
- Metrics: `radiance:prefilling_sequences` (RUNNING prompts holding a slot),
  `radiance:prefill_admissions_deferred_total` (deferral decisions, not distinct requests); the startup log prints the
  resolved setting.
