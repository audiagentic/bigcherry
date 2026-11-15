# rad0034_prefill_concurrency

**Status:** untested
**Plan item:** VTP39

Scope: generic Radiance scheduler admission cap for unfinished long prefills, off by default.

## Summary

`--max-prefill-seqs N` limits only long-prefill RUNNING admissions. A prompt whose
remainder fits one shared-budget step (including after a prefix-cache hit) bypasses the cap,
whatever share of a busy step it gets. Decoders, the interleave hold, the preemption whole-recompute gate and the other
resource limits retain their existing rules. One gauge and one cumulative counter make the
policy visible. Serving v2 exposes the optional tuning field; the `-long` qwen27b configs set it to 2.

## Outcome

Untested and not promoted; off by default. Hardware (two starts an arm, eight 100k prompts at
once): cap 1 lowers mean first-token time 434 -> 208 s with no preemptions, short requests
3.6-4.5 s; cap 2 (the `-long` config default) 434 -> 268 s, short requests 4.6-5.6 s; see README.md.

## Detail

See README.md.
