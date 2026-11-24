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
