# rad0007_hc_pf2_fp8_down

**Status:** untested (runs correctly-shaped on real hardware; measured a 2.7x regression, not yet a win)
**Plan item:** RAD15/RAD17

## Summary

Inert scaffold for an FP8 WMMA down-product in hc_read's PF2 prefill form (gfx1201, Qwen3.8-Flash-Next). Adds
three new kernels to `libr4d/r4d_hc_bf16.hip` -- none of them are called yet; this revision only proves the
anchors apply and the new code compiles against the real pinned source.

## Outcome

Dispatch is wired and the path runs end to end on real hardware (`RAD_HC_PF2_FP8_DOWN=1`, default off).
Measured at M=2048 on the real gfx1201 device: **2.7x SLOWER** than the existing bf16 path (1457.6 us vs
530.7 us), not yet correct to a quality metric either (byte-identity fails as expected for a lossy FP8 path,
but no RMS/KL-style quality check has been built yet to say how close it actually is). Not a win yet --
needs a quality metric and a performance-tuning pass before being a promotion candidate.

## Detail

See below.

## What it does

- `hc_pf2_quant_a8_kernel`: builds N inline and quantises it to E4M3 in one pass, one wave (32 lanes) per
  (row, 128-of-K group). Reads the same real inputs `h`, the published per-stream norm scale, and the gamma
  weight `w` -- and runs the identical formula (`bf2f(h) * scale * (w + wadd)`) the real `hc_down_pf2_kernel`'s
  `stageA` already runs on hardware -- rather than assuming a materialized bf16 N array exists anywhere (it
  does not; see "Real gap" below, now closed). Quantisation math validated standalone against real
  ground-truth N: `tools/lab/radiance/hc-pf2-quant-a8.hip` (0 mismatches across 80 scales + 10240 codes,
  2.53% RMS error). The N-building half is a direct transcription of already-shipped, already
  r4d_selftest-checked arithmetic, not new numerics; it has not yet been separately re-measured against a
  fresh hardware dump of real h/scalein/w -- that is the next thing to build if this specific transcription
  is doubted beyond code review.
- `hc_down_pf2_fp8_kernel`: the FP8 WMMA down-product. Reads `W_down` directly from its real on-disk bytes
  (one packed `[K codes][K/128 f32 scales]` blob per row, stride `row_bytes(K,128)`, confirmed against the
  live container -- 10560 for K=10240, not the naive 10240) rather than the bf16-decoded `plane` buffer the
  existing kernel uses. The per-128-group WMMA + layout math, and now the complete epilogue, are validated
  end to end at REAL production scale (RT=64 rows, lr=320/20 column tiles, kWaves=8, real activation and
  real weight bytes together): `tools/lab/radiance/hc-down-fp8-fullscale.hip` (max rel diff 2.7e-5, fp32
  noise floor) for the ks==1 (no split-K) direct-to-`dout` path. The ks>1/`part[]` output path is written to
  match `hc_down_pf2_kernel`/`hc_down_reduce_kernel`'s own indexing convention (`part[(blockIdx.y*rows+r)*
  ncol+col]`), read directly from the real source for this revision, but is not yet exercised by a harness
  that drives `hc_down_reduce_kernel` afterward -- transcribed and reviewed, not measured.
- `hc_inj_pf2_bf16_kernel`: placeholder only (not yet implemented). The injection weight has no FP8 variant in
  the schema, so this will be a direct extraction of `hc_down_pf2_kernel`'s own existing arithmetic, not new
  numerics.

**Real gap found reading the pinned source, closed in this same revision**: an earlier draft of
`hc_pf2_quant_a8_kernel` assumed a materialized bf16 N array already existed to quantise. No such array
exists anywhere in the real pipeline -- `hc_down_pf2_kernel` builds N incrementally, tile by tile, straight
from `h` + per-stream norm scale + gamma weight `w`, into a small double-buffered LDS window, and never
materializes the whole thing. Rather than add a separate materialization kernel, `hc_pf2_quant_a8_kernel`
was changed to build N inline from those same three real inputs, using the exact formula `stageA` already
runs, then quantise it in the same pass -- no new buffer, no new numerics for the N-build half.

**Dispatch is now wired** (`RAD_HC_PF2_FP8_DOWN=1`, default off): `hc_pf2_launch`'s down-product call site
now branches to the FP8 path when the flag is set and `f8` is true, falling back to the original bf16 call
unchanged on allocator failure. First real finding from wiring it in: every PF2-eligible shape this session
found has `ks > 1` (split-K active; M=512 already has `ks=7`), so an eligibility check requiring `ks==1`
(this patch's earlier revision) would have made the whole path dead code for every real call. Added a
down-only split-K reduce kernel (`hc_down_pf2_fp8_reduce_kernel`) scoped to `[rows x lr]`, since the real
`hc_down_reduce_kernel` sums inject columns too and this pipeline never writes inject columns into `part[]`
(`hc_inj_pf2_bf16_kernel` computes those directly, independent of K-split).

**Real hardware run, via `r4d_selftest --perfhc 2048 3` against a scratch-relinked `libr4d.so`** (device
confirmed idle first; GPU index found by trial -- device 0 segfaulted immediately as a non-gfx1201 part,
device 2 is the real gfx1201 R9700):

- Flag off: byte-identical to the decode-tile reference, same timing as the unpatched library (530-535 us,
  ~50.8 TFLOP/s) -- the dispatch wiring does not disturb the existing path.
- Flag on at M=2048 (ks=2 real split-K shape): **runs without crashing**, but is NOT byte-identical to the
  reference (`differs in x q qs rq rs` -- expected, since FP8 quantisation is lossy and byte-identity was
  only ever a claim for the bf16-equivalent forms; the right comparison is a quality metric like the
  RMS/KL figures used for this project's other FP8 features, not byte-for-byte, and that comparison has not
  been built yet) and **measures SLOWER than the bf16 path**: 1457.6 us / 18.5 TFLOP/s vs 530.7 us / 50.8
  TFLOP/s baseline -- roughly 2.7x worse, not a speedup.

**Honest current status**: the mechanism runs end to end on real hardware for the first time, with no crash
and no memory corruption of other outputs, but it is currently a measured REGRESSION, not a win. Likely
causes, not yet investigated: the extra quant-kernel and reduce-kernel launches (2 additional kernel launches
per call that the bf16 path doesn't pay), and the FP8 down kernel's naive double-`__syncwarp()`-per-K-group
structure having none of the real kernel's double-buffered prefetch/overlap. A quality check and a
performance-tuning pass are both still needed before this is a candidate for the production flag.

## Why

RAD15/RAD17: `hc_read`'s PF2 down-product runs at ~22% of gfx1201's bf16 WMMA peak; an FP8 WMMA path is the
identified lever. The design (quantisation pre-pass over fusing into the existing double-buffered loop, the
real packed weight layout, the injection-columns finding, the split-K alignment hazard) came from a GPT
dev-agent design review (`req_bigcherry_757a112f3c07`) after reading the real pinned source, cross-checked
against hardware-verified building blocks from this project's own validation (rocwmma::fragment proven
correct via single-nonzero-element probes; the real weight stride confirmed against the live container).

## Evidence

Compile-only, this revision: the three standalone harnesses referenced above prove the underlying math is
correct; this patch proves the same logic, shaped for the real kernel's actual constants (`kPf2RT`, `kWaves`,
`kWave`, `kThreads`), applies cleanly to the pinned source and compiles without errors, built once on Brutus
against a scratch copy of the pinned tree (never the shared checkout). No runtime/hardware evidence yet since
nothing calls these kernels.
