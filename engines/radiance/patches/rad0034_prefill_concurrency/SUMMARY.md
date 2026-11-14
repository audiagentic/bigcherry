# rad0034_prefill_concurrency

**Status:** untested
**Plan item:** VTP39

Scope: generic Radiance scheduler admission cap for unfinished long prefills, off by default.

## Summary

`--max-prefill-seqs N` limits only long-prefill RUNNING admissions. A prompt that can
complete in the admitting shared-budget step (including after a prefix-cache hit) bypasses
the cap. Decoders, the interleave hold, the preemption whole-recompute gate and the other
resource limits retain their existing rules. One gauge and one cumulative counter make the
policy visible. Serving v2 exposes the optional tuning field; no configuration enables it.

## Outcome

Untested and not promoted. Card-free tests do not replace G4 current-pin requalification,
the VTP39 full-cache hardware ABBA or the owner default decision.

## Detail

See README.md.
