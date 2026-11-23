# Stage 9 — effective helper dispatch weighting

This is a source-level reachability audit, not a GPU timing claim. It distinguishes standalone plugin-only GEMM dispatch from the normal helper route. No serving trace was executed.

## Actual gfx1030 helper policy

- The architecture declares `rows-kernel = true` for this helper and W4NL64 as its supported expert-weight form. The manifest combines required architecture rows with user policy using logical OR; the worker subsequently sets `rows_ = conf.rows`.
- A small-lane request uses the prequeued row class (normally eight rows) and, with the default GEMV enabled, runs its dedicated `lane_moe_gemv` family unless an optional `rows_small` setting selects dedicated rows instead.
- A request above the small-lane class enters the large lane. On gfx1030, `rowk = rows_ && !small_lane` is true. Both expert projections take dedicated `lane_moe_rows_w4_*` kernels. This `rowk` branch is checked **before** any GEMV or plugin GEMM fallback. Even a configured large-lane GEMV threshold does not supersede rows on this architecture.
- The `moe_gemm_q_kernel` in the libr2 plugin is reached only through the grouped-GEMM fallback when the above dedicated kernels are not selected. That is not the normal gfx1030 helper policy for either requested decode band.

| Request token rows | Lane policy on gfx1030 | Gate/up | Down | libr2 expert-GEMM invocations per request |
|---|---|---|---|---:|
| 1–8 | Prequeued small lane (normal eight-row class) | Dedicated GEMV; optional dedicated rows | Dedicated GEMV; optional dedicated rows | **0** |
| 9–32 | Large lane (unless an explicitly configured larger small class) | Dedicated accurate W4 rows | Dedicated accurate W4 rows | **0** |
| More than 32 | Large lane | Dedicated accurate W4 rows | Dedicated accurate W4 rows | **0** |

This mapping is per helper and applies when the architecture-required rows policy and normal W4NL64 weight form are effective. Explicit diagnostic configurations may choose other dispatch paths, but are not production evidence.

## What the standalone replay actually timed

The synthetic full-plugin harness exercises W4=1/2/3 forms for 512 experts and ten routes per token; it does not use the helper's inlined GEMV/rows dispatch. For the token-ordered gate/up geometries, the following rows are relevant to the *standalone plugin*:

| M | W4 standard | W4NL64 | W4NL32 | Production helper weight on libr2 wave-cap optimization |
|---:|---|---|---|---:|
| 1 | Dedicated one-row kernel | NTL4 / ES1 / FK256 | NTL4 / ES1 / FK256 | 0 |
| 4 | NTL4 / ES1 / FK128 | NTL4 / ES1 / FK256 | NTL4 / ES1 / FK256 | 0 |
| 8 | NTL4 / ES1 / FK128 | NTL4 / ES1 / FK256 | NTL4 / ES1 / FK256 | 0 |
| 16 | NTL4 / ES1 / FK128 | NTL4 / ES1 / FK128 | NTL4 / ES1 / FK128 | 0 |
| 32 | NTL4 / ES1 / FK128 | NTL4 / ES1 / FK128 | NTL4 / ES1 / FK128 | 0 |
| 64 | NTL4 / ES1 / FK128 | NTL4 / ES1 / FK128 | NTL4 / ES1 / FK128 | 0 |

All W4 forms of the sorted down projection in this 54-case replay use the separate down-decode kernel, not the modified `moe_gemm_q_kernel`.

**Decode weighting:** the helper's 1–8 and 9–32 row bands assign **zero actual requests** to any modified expert-GEMM instantiation under normal gfx1030 dispatch. The original motivation (helper pacer) remains valid as an end-to-end observation, but this particular plugin wave-cap change is not a route to improving that pacer. Future effort should focus on the helper's actually launched GEMV, rows, route, quantise and publish kernels, subject to new bitwise and performance gates.
