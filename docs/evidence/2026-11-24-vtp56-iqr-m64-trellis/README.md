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
