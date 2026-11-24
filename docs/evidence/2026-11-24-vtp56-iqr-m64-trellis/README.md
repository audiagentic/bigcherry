# IQR wide-tile trellis resource audit

## Stage 0 — scope and dispatch

The existing 16-row native-accumulation change is present. The queued native-fragment change is not part of this branch's base. The kernel under audit is the narrow tile for 17–64 tokens. Larger token counts dispatch a separate prefill tile: M=65–256 comparisons alone cannot qualify this rewrite. Computed-trellis forms with compatible row/BF16 scales and aligned outputs also use the 16-row route through M=64.

## Stage 1a — baseline metadata

The gfx1100 wide tile has 544 compiled specializations, all with register spills. There are 464 trellis specializations: 96 table-trellis (kind 5), 368 computed-trellis (kind 6). All 544 use 240–256 VGPR and six register-file waves per SIMD. The largest private allocation is 3,640 bytes per thread. The separate report records all 544 baseline VGPR/spilled-VGPR/private-byte/wave records.

## Stage 1b — native accumulation and resource gate

The integer WMMA accumulator stays in native layout for eight successive K fragments of each 128-K scale block, followed by one conversion before the existing float fold. The old per-WMMA conversion is eliminated. The initial M64 integer seed is zero and invariant under the layout permutation; no BIAS seed exists on this path, and the 16-row BIAS logic remains unchanged. Within a group the absolute signed 8-bit product-sum bound is 128 × 128 × 128 = 2,097,152, below signed 32-bit range, so regrouping lane permutations across integer additions cannot overflow.

The four-unit controlled compile produced exactly 3,169 matched kernel records. All **544/544** wide-tile specializations reduced spilled VGPRs, **438** became spill-free, and **106** still spill at most 136 private bytes per thread. The remaining spills are concentrated at NT4 (101 records), NT3 (four) and NT2 (one): the wide tile still holds token, weight and scale fragments live across the K group. Those are specialization-based live-range indicators, not a proof of the only cause. Among trellis specializations, 19 table-trellis and 80 computed-trellis records still spill.

Across the M64 records there are no new spill or private allocations, no increased SGPR counts and no lost register-file waves per SIMD. Exactly **2,625/2,625** non-M64 resources match an unchanged same-source/same-toolchain control, including VGPR, spills, private memory, LDS and workgroup sizes. The source file containing these measurements is unchanged outside the managed rewrite.

**Stage 1 resource gate passed.** Byte identity, full plugin build, IQR selftest, cold/warm replay and final no-regression qualification are still pending. No serving test, A/B, KLD, push or shipment has been performed.

## Stage 2 — remaining spill and lifetime experiments

Two additional loop-lifetime variants were built against the first 128-instance partition. Both were rejected before GPU testing:

- Rolling the inner WMMA K loop reduced spilling instances from 23 to five, but grew private memory on 69 forms and lost VGPR register-file waves on two. A representative computed-trellis four-token tile grew from 100 to 208 B/thread of private memory.
- Deferring activation and weight look-ahead grew spilling instances from 23 to 39, increased private allocation on 39 forms, and reduced register-file waves on 32.

Both variants left all other kernel resource records unchanged. The resource-safe accumulate-once rewrite remains selected, with its residual 106 spilling instances documented individually in the report. No hand-picked per-model guard or new constant was introduced.

**Stage 2 result:** further lifetime changes rejected by the no-regression gate; source unchanged. Numerical and timing validation of the selected rewrite remains required.

## Adjacent compiled paths — report only

The separate 16-row tile has 316 computed-trellis specializations averaging 104.101 lane-exchange and 14.684 WMMA instructions, versus 48 table-trellis specializations at 64 exchanges and eight WMMA instructions each. The computed-trellis row varies from 64/8 at one token tile to 208/32 at four.

Of the 49 computed-trellis MoE specializations, 13 spill up to 412 B/thread. The adjacent 16-row and MoE paths are unchanged by this patch. These are compiled code-object facts, not new timings.

## Exhaustive 6-bit and 7-bit wide-tile specialization coverage

All 80 B6 trellis wide-tile specializations were compiled and resource-audited: 16 table-trellis and 64 computed-trellis. Baseline spilling was 80/80; candidate spilling is 19/80, with maximum private memory falling from 3,408 to 120 B/thread. For B7, 80/80 forms were also compiled: spilling fell from 80 to 22, with maximum private memory falling from 3,640 to 136 B/thread. Both chain modes are covered where the form supports chaining; no per-shape constants or exclusions were introduced.

## Source-level correctness review (GPU gate pending)

The old WMMA wrapper translates each signed/signed 16-K integer product to compatibility lane order before elementwise i32 addition. The native accumulation form uses the same native operand permutation and signed WMMA flags, sums eight K16 products in native lanes, then translates once. As the translation is a lane permutation and each 128-K sum has magnitude at most 2,097,152, addition cannot overflow i32 and conversion commutes with accumulation exactly. The initial M64 sum is zero in either layout; 16-row bias initialization is untouched. Scale-fold order, split-K completion and chain wait/done are also unchanged.

The unchanged-control and candidate builds produced the same 197 occupancy-target warnings, and no additional SGPR use. **GPU byte identity and replay must still pass before acceptance.**
## Continuation step 1 — GPU byte identity

The merged 16-row native-fragment change and wide-tile native integer accumulation are both present. With the shipped plugin loaded as the control in the same process, 18 independent short GPU leases checked the complete BF16 output buffers for all M=17–64 across seven trunk shapes, including N=5376 on trellis forms. The nine forms (computed and table trellis B4/B6/B7, B4 table, B8 grid, B6 zero) each passed chained and unchained: **6,048 shape/M byte comparisons, zero differing elements**, with nonzero outputs. The unchanged 16-row BIAS and gated paths retain their separately required selftest checks. No serving experiment was run.

## Continuation step 2 — build and control regression checks

The complete plugin and IQR selftest executable built successfully. Queued GPU selftest runs of `gemm_iqr_nt_m64`, `gemm_iqr_tiled`, `gemm_iqr_gated` and `gemm_iqr` returned **2398 successful checks, zero failures**. The merged same-toolchain control retains the upstream 16-row optimization without the wide-tile rewrite. Against that control, all 544 wide-tile specializations show no increased spilling or private memory and no lost register-file waves; all 2,625 other matching records have identical resource metadata. There are 438 spill-free and 106 residual spilling wide-tile records.

The SGPR-specific control also matched all 2,625 unrelated symbols exactly, with no increase across any of the 544 wide-tile instances.

## Continuation step 3 — cold and prefetch-warm GPU replay: PASS

- Executed sixteen short individually queued GPU comparisons (four forms × two chain states × cold or 12-layer prefetch-warm). The same process alternated shipped/candidate measurements three times per shape/M and preserved all raw medians and min–max samples.
- Six trunk shapes plus the ragged N=5376 trellis shape (N=4112 on table/grid), all M=17,24,32,48,64. **560/560 requested group comparisons improved and zero groups regressed outside their observed three-repeat spread.**
- A broader exploratory recording also passed an 805-group no-out-of-spread check, including previously collected additional heights, but the explicit acceptance gate was the 560 requested groups.
- The original exploratory M65 table4 ragged prefill launch failed identically in stock and candidate, outside the requested path; scratch replay was corrected to only the required wide-tile heights. No source or plugin change.
- The full per-form, M, shape, chain, mode median [min–max] table is preserved in the evidence CSV. Times are microseconds and gains are relative to the shipped plugin.

| Form | Cold median gain | Warm median gain | Smallest gain in both |
|---|---:|---:|---:|
| mad6 | 81.9% | 81.9% | 62.7% |
| trellis6 | 79.2% | 79.3% | 60.6% |
| table4 | 31.7% | 32.6% | 18.0% |
| grid8 | 84.8% | 84.6% | 66.9% |

Example full-range timing, N=34816 K=5120 unchained:

| Form | M | Cold shipped → candidate (µs), med [min–max] | Warm shipped → candidate (µs), med [min–max] |
|---|---:|---|---|
| mad6 | 17 | 1958.1 [1943.7–2943.6] → 396.4 [350.4–437.6] | 1937.0 [1917.4–2959.7] → 396.8 [346.9–439.4] |
| mad6 | 24 | 1936.6 [1930.2–1996.3] → 351.1 [343.6–352.0] | 1918.1 [1914.5–1929.5] → 347.8 [346.3–348.2] |
| mad6 | 32 | 1921.3 [1902.2–1924.0] → 351.5 [351.4–352.1] | 1914.8 [1886.1–1921.0] → 349.1 [348.3–358.6] |
| mad6 | 48 | 4165.8 [4083.6–4235.7] → 420.9 [420.8–427.0] | 3943.3 [3943.0–4055.6] → 423.1 [403.2–425.9] |
| mad6 | 64 | 5428.6 [5357.0–5455.5] → 544.5 [536.4–590.8] | 5322.0 [5271.5–5425.6] → 564.0 [558.3–580.2] |
| trellis6 | 17 | 1987.4 [1947.6–3005.3] → 460.8 [399.9–511.8] | 1963.1 [1935.5–3012.5] → 455.3 [394.0–510.9] |
| trellis6 | 24 | 1956.9 [1951.8–1964.7] → 402.2 [401.3–403.1] | 1942.5 [1934.0–1957.4] → 399.3 [398.4–399.9] |
| trellis6 | 32 | 1950.7 [1934.8–1959.3] → 408.3 [407.4–408.8] | 1912.3 [1904.9–1921.7] → 399.9 [398.5–400.1] |
| trellis6 | 48 | 4044.9 [4008.2–4073.6] → 477.1 [473.6–491.0] | 3843.5 [3841.8–3858.0] → 471.9 [466.9–481.7] |
| trellis6 | 64 | 5422.0 [5391.4–5482.1] → 621.8 [616.8–646.5] | 5229.7 [5155.9–5426.5] → 561.3 [558.3–596.6] |
| table4 | 17 | 449.7 [405.1–846.0] → 322.8 [252.6–385.5] | 457.0 [403.5–843.9] → 324.6 [243.2–385.0] |
| table4 | 24 | 418.5 [382.9–420.5] → 265.5 [261.8–268.8] | 419.9 [384.1–419.9] → 261.6 [260.4–262.3] |
| table4 | 32 | 414.2 [411.9–422.3] → 263.8 [263.1–265.2] | 414.0 [411.6–422.6] → 257.9 [255.0–261.2] |
| table4 | 48 | 576.6 [576.1–612.5] → 384.1 [382.3–387.5] | 603.8 [579.5–605.0] → 378.6 [343.8–380.3] |
| table4 | 64 | 771.8 [720.0–817.1] → 467.7 [464.6–513.2] | 766.4 [736.8–810.3] → 458.9 [458.2–505.7] |
| grid8 | 17 | 1680.2 [1585.7–2749.3] → 297.0 [234.9–318.5] | 1665.3 [1554.2–2717.5] → 293.3 [233.8–318.9] |
| grid8 | 24 | 1591.2 [1590.4–1595.1] → 235.0 [234.4–235.6] | 1569.0 [1554.0–1574.2] → 233.6 [233.2–235.0] |
| grid8 | 32 | 1573.4 [1570.2–1590.3] → 235.1 [227.2–235.7] | 1569.7 [1567.3–1572.9] → 233.5 [229.0–233.8] |
| grid8 | 48 | 3267.5 [3213.4–3306.9] → 322.6 [311.0–330.3] | 3376.2 [3334.0–3379.6] → 310.9 [308.8–321.0] |
| grid8 | 64 | 4915.0 [4748.8–4964.9] → 382.3 [382.0–383.9] | 4717.8 [4691.1–4733.8] → 382.0 [374.1–397.7] |
