# 1257_nro08_topk_wave32

## Summary

Makes 1256's HIP TOP_K kernels wave32-native (wave32 shuffles instead of LDS/barrier reductions) and drops the wave64 compile flag. Same routes and markers as 1256, tagged `patch=1257_nro08`.

## Outcome

Recorded state: `untested` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

Plan `NRO08`; state `untested`; requires `1256_nro07_topk_hybrid`.

This package stages the source follow-up's core wave32 idea as a fixture-testable device helper: reduce each 32-lane wave with shuffle-down, publish one winner per wave to shared memory, then let wave 0 reduce wave winners. It is not called by NRO07 yet.

The remaining source follow-up—two-half 64-bin radix scan and items-per-thread tuning—will be added only after NRO07's exact TOP_K semantics and this primitive's lane/tie behavior pass tests. This avoids mixing algorithm introduction, wave semantics and launch-coverage changes in one unreviewable draft.
