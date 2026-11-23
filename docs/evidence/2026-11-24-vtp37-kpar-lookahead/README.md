# VTP37: gfx1100 K-parallel look-ahead qualification

## Stage 1 — compile-time occupancy rule and resource gate: PASS

Only the gfx1100 native rewrite's K-parallel body is changed. The formula
counts the simultaneously live register arrays and the weight/token
look-ahead rings from their actual C++ fragment types (Stage8, i2v, f8v,
r3_i8), using 32-bit words. Predicted wave residency uses the gfx1100
wave32 register file (1,536 VGPR per SIMD). The predictor is deliberately
restricted to NT <= SB/128: with extra token tiles, the compiler's
overlapping inter-tile bq loads cause the resource records (SB128 NT2,
NT3/4) to deviate from this simple model. For all KW1/2/4 and both
chain modes, the consequent changed set is SB128 NT1 and SB256 NT1/2,
18 of the 48 instantiations. This is a constexpr rule over the body types
and SB/NT, not a lookup by model or N/K/M. All other instantiations compile
the stock look-ahead.

The previous acc-once WMMA path, scale order and SB256 multi-tile unroll
are untouched. The old R4D_WAVES(6) hint is retained: actual compiler
allocation for other shapes is not predictable from these live type sizes
well enough to derive a safe new waves-per-EU hint. The v7-style
gain from a fixed higher hint is therefore **not shipped**.

An isolated gfx1100 libr3 unit build completed (nice -n 19, -j1).
Extracted AMDGPU metadata from the built plugin and the 2020.8.3
baseline. Every kpar instantiation has zero new spills, zero increased
VGPR/private scratch and zero occupancy losses. All 278 non-kpar kernels
in the same unit retain identical resource metadata. Resource table
cells are VGPR/spill/private-bytes/theoretical waves per SIMD.

| KW | SB | NT | chain | 2020.8.3 VGPR/spill/private/waves | candidate VGPR/spill/private/waves |
|---:|---:|---:|---|---|---|
| 1 | 128 | 1 | ch | 143/0/0/10 | 117/0/0/13 |
| 1 | 128 | 1 | un | 140/0/0/10 | 114/0/0/13 |
| 1 | 128 | 2 | ch | 177/0/0/8 | 177/0/0/8 |
| 1 | 128 | 2 | un | 174/0/0/8 | 174/0/0/8 |
| 1 | 128 | 3 | ch | 240/2/12/6 | 240/2/12/6 |
| 1 | 128 | 3 | un | 240/0/0/6 | 240/0/0/6 |
| 1 | 128 | 4 | ch | 179/0/0/8 | 179/0/0/8 |
| 1 | 128 | 4 | un | 176/0/0/8 | 176/0/0/8 |
| 1 | 256 | 1 | ch | 212/0/0/7 | 181/0/0/8 |
| 1 | 256 | 1 | un | 209/0/0/7 | 178/0/0/8 |
| 1 | 256 | 2 | ch | 176/0/0/8 | 154/0/0/9 |
| 1 | 256 | 2 | un | 173/0/0/8 | 151/0/0/10 |
| 1 | 256 | 3 | ch | 213/0/0/7 | 213/0/0/7 |
| 1 | 256 | 3 | un | 210/0/0/7 | 210/0/0/7 |
| 1 | 256 | 4 | ch | 173/0/0/8 | 173/0/0/8 |
| 1 | 256 | 4 | un | 170/0/0/9 | 170/0/0/9 |
| 2 | 128 | 1 | ch | 144/0/0/10 | 118/0/0/13 |
| 2 | 128 | 1 | un | 141/0/0/10 | 115/0/0/13 |
| 2 | 128 | 2 | ch | 178/0/0/8 | 178/0/0/8 |
| 2 | 128 | 2 | un | 175/0/0/8 | 175/0/0/8 |
| 2 | 128 | 3 | ch | 240/3/16/6 | 240/3/16/6 |
| 2 | 128 | 3 | un | 240/1/8/6 | 240/1/8/6 |
| 2 | 128 | 4 | ch | 180/0/0/8 | 180/0/0/8 |
| 2 | 128 | 4 | un | 177/0/0/8 | 177/0/0/8 |
| 2 | 256 | 1 | ch | 213/0/0/7 | 182/0/0/8 |
| 2 | 256 | 1 | un | 210/0/0/7 | 179/0/0/8 |
| 2 | 256 | 2 | ch | 177/0/0/8 | 155/0/0/9 |
| 2 | 256 | 2 | un | 174/0/0/8 | 152/0/0/10 |
| 2 | 256 | 3 | ch | 215/0/0/7 | 215/0/0/7 |
| 2 | 256 | 3 | un | 212/0/0/7 | 212/0/0/7 |
| 2 | 256 | 4 | ch | 174/0/0/8 | 174/0/0/8 |
| 2 | 256 | 4 | un | 171/0/0/8 | 171/0/0/8 |
| 4 | 128 | 1 | ch | 144/0/0/10 | 118/0/0/13 |
| 4 | 128 | 1 | un | 141/0/0/10 | 115/0/0/13 |
| 4 | 128 | 2 | ch | 178/0/0/8 | 178/0/0/8 |
| 4 | 128 | 2 | un | 175/0/0/8 | 175/0/0/8 |
| 4 | 128 | 3 | ch | 240/3/16/6 | 240/3/16/6 |
| 4 | 128 | 3 | un | 240/1/8/6 | 240/1/8/6 |
| 4 | 128 | 4 | ch | 180/0/0/8 | 180/0/0/8 |
| 4 | 128 | 4 | un | 177/0/0/8 | 177/0/0/8 |
| 4 | 256 | 1 | ch | 213/0/0/7 | 182/0/0/8 |
| 4 | 256 | 1 | un | 210/0/0/7 | 179/0/0/8 |
| 4 | 256 | 2 | ch | 177/0/0/8 | 155/0/0/9 |
| 4 | 256 | 2 | un | 174/0/0/8 | 152/0/0/10 |
| 4 | 256 | 3 | ch | 215/0/0/7 | 215/0/0/7 |
| 4 | 256 | 3 | un | 212/0/0/7 | 212/0/0/7 |
| 4 | 256 | 4 | ch | 174/0/0/8 | 174/0/0/8 |
| 4 | 256 | 4 | un | 171/0/0/8 | 171/0/0/8 |

## Stage 2 — bit-identical BF16 output: PASS

Same-process 2020.8.3 baseline and new single-unit plugin comparisons,
identical deterministic input/scales, complete output BF16 bytes over
every M 1–64 for the six TP3 per-rank shapes, ragged N=5136 and
short-K KW2 cases K=512 and K=256. Each test acquired and released
its own queued GPU lease (12-minute TTL, 5-minute estimate).

| Row/dispatch | PASS | SKIP | FAIL |
|---|---:|---:|---:|
| iq/unchained | 576 | 0 | 0 |
| iq/chained | 576 | 0 | 0 |
| i8/unchained | 576 | 0 | 0 |
| i8/chained | 576 | 0 | 0 |

**Total 2,304 PASS, 0 SKIP, 0 FAIL.**
No numerical divergence. No serving, ms/step A/B or KLD.

