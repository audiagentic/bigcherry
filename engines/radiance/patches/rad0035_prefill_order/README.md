# rad0035_prefill_order

## Summary

VTP41 adds three generic engine options, each with a matching optional serving v2 tuning field, to the prefill policy
of rad0034:

- `--long-prefill-tokens N` (`long_prefill_tokens`): a prefill counts toward `--max-prefill-seqs` while more than N
  prompt tokens are left, and a prompt within N is exempt. **0 (default)** keeps rad0034's rule, one shared step.
- `--prefill-order arrival|remaining` (`prefill_order`): after priority, running prefills are walked and waiting prompts
  admitted in arrival order (**default**, as before) or by least prompt left (`remaining`).
- `--prefill-age-ms T` (`prefill_age_ms`): with `remaining`, a prompt waiting longer than T since arrival goes ahead of
  younger ones, in arrival order. **0 (default)** never ages.

Motivation: VTP39 (`docs/evidence/2026-11-15-vtp39-prefill-concurrency/README.md`); plan item VTP41.

## Outcome

Opt-in; no serving configuration sets them yet. Card-free: `tests/sched_test.cpp` (the five `vtp41_*` cases; with
three 4000-token prefills running, a 1000-token prompt gets its first token at step 5 under `remaining` against more
than 12 steps under `arrival`) and `tests/config_test.cpp`. Hardware A/B pending. The patch state is `untested`.

## Detail

- **Defaults are exact.** With all three at their defaults, or a threshold equal to one shared step, or an age under
  arrival order, step plans are identical to rad0034's (`vtp41_defaults_keep_step_plans`, random schedules, cap 0-2).
- **Order key.** Under `remaining`, `ahead_of` compares priority, then the prompt tokens left, then arrival. Decoders,
  preempted requests and aged prompts key 0, so among them arrival decides as before: a preempted request still
  re-enters at the front, and the forced-preemption fallback still finds a decoding victim only for a waiter ahead of
  it. Keys move as prompts are computed and time passes, so `running_` and `waiting_` are re-sorted at the start of
  each step.
- **Shares.** With `--busy-prefill-chunk` sharing (auto, as served), each prefill behind the first reserves one share
  (an eighth of the step) in arrival order, so several running prefills advance at about a share each. Under
  `remaining` the prefills behind the first keep one share between them, so the step goes to the prompt nearest its
  first token. Decode reserves and the admission share are unchanged. Chunks still end on the checkpoint grid.
- **Preemption.** On a KV refusal the walk preempts `running_.back()`; under `remaining` that is the lowest-priority
  prefill with most prompt left, rather than the latest arrival.
- **Threshold.** `long_prefill_tokens` replaces `shared_tok` in rad0034's three tests (the running count, same-step
  admissions and the exemption); the startup log prints the resolved value. It does not change step size.
- **Not here.** Decode share by time, and a cap derived from free KV, are separate (VTP41 notes).
