# 1261 PNRO10 ctx_other speculative scheduler devices

Untested. Adds `ctx_other`'s model devices to the draft/MTP context's scheduler backend list, deduplicated, so shared tensors between the draft and target contexts can be scheduled when the two contexts' device sets differ.

## Validation

Contract `PNRO10-SPEC-CTX-OTHER-DEVICES`. Execution target is Brutus, dual gfx1100 only. Require greedy token parity under MTP on the dual-gfx1100 `-sm tensor` server (no activation marker -- the fix is unconditional), and `ci95_threshold_bound_v1` with `max_control_regression_pct=1`, `min_paired_rounds=10`.
