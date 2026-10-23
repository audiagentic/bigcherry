# rad0008_hc_upmix_rt_sweep

**Status:** validated
**Plan item:** RAD17/RAD18

Promoted 2026-10-23: releases/evidence/rad0008_hc_upmix_rt_sweep-promotion.md.

## Outcome

Sibling investigation to RAD17 (FP8 down-product), targeting the shared BF16 up-mix kernel
instead. Per GPT dev-agent review (req_bigcherry_46bf68059a11), the diagnostic hipBLASLt
GEMM + separate epilogue approach (tools/lab/radiance/hc-upmix-epilogue-diag.hip, RAD17/RAD18)
measured 395us combined vs the fused kernel's 261us -- the separate epilogue is bandwidth-bound,
round-tripping the fp32 up-product through global memory that the fused kernel never materializes.
GPT's own decision tree rejected the GEMM-replacement angle and recommended sweeping
`hc_up_mix_wide_kernel`'s RT (rows per workgroup, selecting MW waves-per-block) instead, to test
whether occupancy/register pressure -- not grid fill, already ample at this shape -- is where any
remaining headroom is.

GPT's first proposed lever (sweeping CH, the output tile width) was checked against the real kernel
source (vendor/radiance/libr4d/r4d_hc_bf16.hip) and found infeasible as scoped: CH is not a software
parameter, it is gfx1201's native 16-wide WMMA N-dimension, hard-wired into `frag()`'s `c*16`
fragment offset and the epilogue's `lane & 15` column index. Reported this back to GPT
(req_bigcherry_0641281ad540), which then recommended the RT sweep implemented here.

RT=512 (GPT's originally-suggested "double") is a real, hardware-enforced dead end, not merely
untested: confirmed via an actual standalone compile of that kernel instantiation --
`hipcc` reported "local memory (78336) exceeds limit (65536) in void (anonymous
namespace)::hc_up_mix_wide_kernel<4, false, 512>(...)". The kernel's LDS use is
`sA[RT*kTAP] + sB[NT*kTAP]` shorts (kTAP=68, NT=64) = `RT*136 + 8704` bytes; at RT=512 that's 78336
bytes against gfx1201's 65536-byte per-workgroup LDS limit. RT=384 (60928 bytes) is the largest
legal value and replaces RT=512 as the sweep's upper point.

## What this patch does

Adds a runtime switch (`RAD_HC_UPMIX_RT`, values "128"/"256"/"384") at the PF2 up-mix launch call
site inside `hc_pf2_launch` (vendor/radiance/libr4d/r4d_hc_bf16.hip, ~line 3167 as of the 1.9.0
pin). `hc_up_mix_wide_kernel` itself is completely unchanged -- RT is already one of its template
parameters; this only changes which instantiation gets launched and with what grid. Unset or any
other value falls back to the exact original call (`RT=kPf2UpRT=256`), bit-identical to pre-patch.

Verified (this revision):

- `patch-lint`: clean (same pre-existing `max_span_lines` warning class as rad0007's own edits;
  no errors).
- Anchor match count against the real pristine vendor file: exactly 1, via a throwaway script
  (not checked in) performing the same regex substitution patch.py does, run directly against a
  scratch copy of the real vendor/radiance/libr4d/r4d_hc_bf16.hip on dev -- not the shared
  wt-accept checkout.
- Standalone `hipcc --offload-arch=gfx1201` compile of the patched file: clean, zero errors.
- `nm`/`c++filt` on the compiled object: confirms all three new instantiations
  (`hc_up_mix_wide_kernel<4, false, 128/256/384>`) are present as real compiled device code
  (`__device_stub__` symbols too), alongside the pre-existing `<4, false>`/`<4, true>` instantiations
  used elsewhere in the file.

## Anchor fix (first real build)

The first `bigcherry build --experiment hc-upmix-rt-sweep` stopped at patch application: the anchor began on the
`/* The up product and mix ... */` comment and the guard was the `RAD_HC_UPMIX_RT` string literal, but the patch
engine strips comments and string literals before matching, so the anchor matched 0 times. The earlier check used
a plain regex on the raw file and missed this. The anchor now starts at the `hipLaunchKernelGGL` call and the guard
is the `rad18_upmix_rt` variable; `apply_patch` on the pinned v1.9.0 file applies, and a second apply reports
`already-applied`.

## Hardware result (2026-10-22, gfx1201 R9700)

Built through the real pipeline (`bigcherry build --lane radiance-lab:stock:linux-radiance --experiment
hc-upmix-rt-sweep`, build plan `da22861d7c120bb8e078f39e86ca4037`) and measured with `r4d_selftest --perfhc <M> 50`
(PF2 `hc_read` whole-call time), three rounds per size with the RT order rotated each round. Every run passed the
selftest's byte-identical check of the PF2 output against the decode-tile kernels, RT=384 included (its last
workgroup covers 256 padded rows at M=2048).

| RT | M=2048 runs (us) | mean | M=4096 runs (us) | mean |
|---|---|---|---|---|
| 128 | 503.6 / 511.4 / 506.8 | 507.3 | 1250.6 / 1252.9 / 1249.9 | 1251.1 |
| 256 (production) | 515.0 / 513.2 / 517.0 | 515.1 | 1291.5 / 1293.7 / 1289.6 | 1291.6 |
| 384 | 549.4 / 560.7 / 554.1 | 554.7 | 1440.1 / 1454.6 / 1445.7 | 1446.8 |

RT=128 is faster than production at both sizes with complete separation: -1.5% of the whole call at M=2048
(about 8 us, roughly 3% of the up-mix kernel) and -3.1% at M=4096 (about 40 us). RT=384 is slower at both, in
line with the LDS-occupancy prediction (26/44/61 KB per workgroup at RT 128/256/384).

## Against the unpatched build, and across row counts

Six rotated rounds against the unpatched `radiance-lab` stock build (build plan `5e758b5e337a361e853b77e0f7beaf4c`),
all byte-identical: M=2048 stock 515.3 / patched with the original RT 516.5 / RT=128 504.7 us (-2.1%, RT=128 ahead
in all six paired rounds); M=4096 1293.1 / 1293.4 / 1252.3 us (-3.2%, complete separation). With the original RT
the patched build matches stock, so the switch itself costs nothing.

Rows sweep, stock vs RT=128 (two runs each, us): 512 169.7 vs 166.8; 768 218.5 vs 213.8; 1000 272.8 vs 268.2;
1024 276.9 vs 275.8; 1536 415.7 vs 407.4; 2047 523.3 vs 508.0; 2049 574.9 vs 552.5; 3000 833.3 vs 808.1;
6144 1702.5 vs 1649.7; 8192 2426.4 vs 2325.9. Never slower, byte-identical at every size (256 rows does not take
the PF2 form). RT=128 is therefore the patch's default; RAD_HC_UPMIX_RT=256 restores the original launch.

Evidence covers one shape (n=2560, hc=4, lr=320: Qwen3.8-Flash-Next), the only one r4d_selftest's PF2 bench runs.

## RDNA3 (7900 XTX, gfx1100) through libr3

Production Flash-Next runs `--tp 2` on the two XTX, where `hc_read` comes from libr3, which compiles this same libr4d
source for gfx1100. Measured with `r3_selftest <libr3.so> --perfhc <M> 30` on device 0, four rotated rounds, every
run byte-identical:

| arm | M=2048 PF2 (us) | mean | M=4096 PF2 (us) | mean |
|---|---|---|---|---|
| stock | 1319.0 / 1325.9 / 1324.9 / 1320.3 | 1322.5 | 2611.8 / 2625.2 / 2620.7 / 2618.9 | 2619.2 |
| rad0008, RT=256 | 1331.7 / 1330.2 / 1341.3 / 1343.6 | 1336.7 | 2622.8 / 2627.1 / 2631.8 / 2634.0 | 2628.9 |
| rad0008, RT=128 | 1240.9 / 1242.2 / 1249.1 / 1253.3 | 1246.4 | 2529.0 / 2541.2 / 2555.0 / 2560.8 | 2546.5 |

RT=128 is -5.8% (M=2048) and -2.8% (M=4096) against stock, complete separation. On this card stock PF2 is slower
than the wide form (about 1292 us at M=2048); RT=128 brings it below. Open question: the patched build at RT=256
runs the original launch yet measures about 1% slower than stock on gfx1100 (not seen on gfx1201).

## Not yet done

- Not wired into the actual `bigcherry patch-rebase-check` / registry flow on a fresh checkout of
  this exact repo state (verification above used a manual regex+compile check against dev's
  vendor tree directly, since dev's own `tools/bigcherry` checkout is on a much older commit
  than this repo's current main and its patch schema/tag vocabulary may have drifted).
- Superseded by the hardware result above: `r4d_selftest --perfhc` at `RAD_HC_UPMIX_RT=128/256/384`,
  comparing kernel latency (GPU-event, not wall clock) and, where obtainable, occupancy/register
  metrics, at the production PF2 shape (M=2048, hc=4, n=2560, lr=320), per GPT's decision criteria
  (>=5% improvement retains the winning RT; no improvement stops RT tuning; spills/occupancy
  collapse moves investigation to the `acc[MW*HC]` register allocation).
- No numerical correctness re-check needed beyond the existing byte-identical guarantee: this patch
  cannot change output bytes at any RT value (same kernel body, same math, different grid), only
  which already-existing template instantiation is launched.
