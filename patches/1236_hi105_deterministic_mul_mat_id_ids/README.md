# 1236_hi105_deterministic_mul_mat_id_ids

Plan item HI105 (completed). See `SUMMARY.md` for the mechanism.

## Real hardware evidence (2026-08-24, Brutus, dual gfx1100)

Built as part of `experiment.hi105-correctness` (patches 1222+1223+1236) and
exercised directly against RD54's real K=256 MoE down-projection signature
from Qwen3.6-35B-A3B:

- `leaf_2`-as-ids-tensor assumption confirmed empirically, not just by
  extrapolation from the plain-`MUL_MAT` convention: `BIGCHERRY_REF_DIGEST`
  element counts matched `as`/`b`/`ids` exactly (134217728 / 2048 / 8).
- Real CPU-reference correctness run on both ROCm backends passed:
  `err=9.4e-15` / `9.3e-15`, both far under the `5e-4` threshold.
- Real `tune-promote` rerun: `promotion_status=promoted` for the K=256
  dispatch (up from `rejected_no_correctness_evidence` before this patch),
  statistics unchanged (`p_value=0.0`, 95% CI 60.68-60.88% isolated-kernel
  speedup).
- Broader same-day campaign (29 real signatures, Qwen3.6-35B-A3B, real
  record/tune/promote/replay cycle) found two more real MUL_MAT_ID
  candidates and promoted all three; a real 8-round interleaved replay-vs-
  native A/B showed the full-model production effect is statistically
  indistinguishable from noise (net +0.12% tg128) -- a genuine structural
  finding about MoE-routed matmul's share of total compute, not a flaw in
  this patch or its correctness evidence.

Full detail, including the broader campaign, is recorded in HI105's plan
item (`docs/planning/completed/hip-autotune/HI105.md`).

## Disposition

`state` stays `"untested"` -- this is a `kind = "diagnostic"` test-harness
patch (modifies `test-backend-ops.cpp`'s own initializer), not a production
dispatch patch, so it is not itself a promotion candidate. Its purpose --
unblocking correctness-gated promotion for MUL_MAT_ID candidates -- is
proven on real hardware above.
