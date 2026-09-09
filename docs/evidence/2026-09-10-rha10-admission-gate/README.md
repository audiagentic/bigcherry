# RHA10 admission-gate checkpoint

This checkpoint records the strongest existing replay activation evidence while
keeping the production admission gate fail-closed. It is derived from the
RHA08 `replay-diagnostic` arm, not from a diagnostics-off production timing arm.

The retained diagnostic server log reports 54 replay winners, 9 exact hits,
zero unavailable/rerun-required/incompatible/miss outcomes, and 1,988/1,988
dispatch coverage. The retained `rha08-replay-hit.jsonl` contains 57 hits,
including 12 non-native candidates. This proves that the cache can activate
and that tuned candidates were selected in a diagnostic run; it does not prove
production admission or work equivalence.

Remaining blockers are source/build equivalence across the stock/native/replay
timed arms and an explicit diagnostics-off `final_tuned_launches > 0` record
bound to the same cache, signature catalog, hardware and workload identities.
