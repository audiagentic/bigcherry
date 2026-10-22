# rad0008_hc_upmix_rt_sweep

**Status:** untested
**Plan item:** RAD17/RAD18

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
  scratch copy of the real vendor/radiance/libr4d/r4d_hc_bf16.hip on Brutus -- not the shared
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

## Not yet done

- Not wired into the actual `bigcherry patch-rebase-check` / registry flow on a fresh checkout of
  this exact repo state (verification above used a manual regex+compile check against Brutus's
  vendor tree directly, since Brutus's own `tools/bigcherry` checkout is on a much older commit
  than this repo's current main and its patch schema/tag vocabulary may have drifted).
- Not yet benchmarked on real hardware: `r4d_selftest --perfhc` at `RAD_HC_UPMIX_RT=128/256/384`,
  comparing kernel latency (GPU-event, not wall clock) and, where obtainable, occupancy/register
  metrics, at the production PF2 shape (M=2048, hc=4, n=2560, lr=320), per GPT's decision criteria
  (>=5% improvement retains the winning RT; no improvement stops RT tuning; spills/occupancy
  collapse moves investigation to the `acc[MW*HC]` register allocation).
- No numerical correctness re-check needed beyond the existing byte-identical guarantee: this patch
  cannot change output bytes at any RT value (same kernel body, same math, different grid), only
  which already-existing template instantiation is launched.
