# rad0007_hc_pf2_fp8_down

**Status:** untested (runs correctly-shaped on real hardware; a real P0 correctness bug found by GPT
dev-agent review and fixed; two real perf optimisations landed, 1457.6us -> 1282.5us at M=2048, still
~2.4x slower than the 530.7us baseline, not yet a win -- not a promotion candidate)
**Plan item:** RAD15/RAD17

## Summary

Inert scaffold for an FP8 WMMA down-product in hc_read's PF2 prefill form (gfx1201, Qwen3.8-Flash-Next). Adds
three new kernels to `libr4d/r4d_hc_bf16.hip` -- none of them are called yet; this revision only proves the
anchors apply and the new code compiles against the real pinned source.

## Outcome

Dispatch is wired and the path runs end to end on real hardware (`RAD_HC_PF2_FP8_DOWN=1`, default off). A GPT
dev-agent code review (`req_bigcherry_d67c3e42e319`) found a real P0 correctness bug before it went further:
the original, unconditional `if (ks > 1) { hc_down_reduce_kernel(...) }` call right after the down-product
launch was never gated on which path ran, so it re-ran on top of the FP8 path's own already-correct output --
reading uninitialised `part[]` for every inject column (the FP8 path never writes inject columns into
`part[]`; `hc_inj_pf2_bf16_kernel` computes those directly) and overwriting `inj` with garbage. Fixed by
gating the original call on `!fp8_down_ok`. Measured at M=2048 on the real gfx1201 device, after the P0 fix
and two real validated optimisation passes: **1282.5 us**, down from 1457.6 us (inject kernel's
redundant-read fix, the down kernel's LDS-round-trip elimination, then this P0 fix) -- still **~2.4x
SLOWER** than the existing bf16 path's 530.7 us. Not correct to a quality metric either yet (byte-identity
fails as expected for a lossy FP8 path -- `inj` now correctly shows as differing too, the expected result of
a genuinely lossy computation rather than the earlier bug's corruption; no RMS/KL-style quality check has
been built). Not a
win -- the down kernel's inner loop still lacks the real kernel's double-buffered prefetch/fused tiling, and
a quality metric is still needed before any promotion decision.

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

**Per-kernel timing breakdown** (hipEvent-instrumented throwaway diagnostic build, not part of this patch,
M=2048/ks=2): `quant=0.157ms down=0.688ms reduce=0.017ms inj=0.202ms total=1.065ms`. This pinpointed two real
findings rather than leaving the regression as a guess:

1. `hc_inj_pf2_bf16_kernel` was re-reading the entire `a8`/`ascale` activation once per `hc` column (grid
   `(rows, hc)`), for math where only `injw` actually depends on the column -- almost pure redundant memory
   traffic for a trivial amount of compute. Fixed: one wave per ROW (grid `(rows)`), covering all `hc`
   columns in a single pass, accruing `hc` partial dot products in per-lane registers. Reduced inj's share
   from 0.202ms to about 0.167ms (measured via the ~35us drop in the total, 1457.6us -> 1422.0us at the same
   shape) -- a real, measured improvement, but small next to the total.
2. `hc_down_pf2_fp8_kernel` itself was the dominant cost at 0.688ms -- **more than the ENTIRE 530.7us
   baseline PF2 call**. Root cause, found by reading this project's own production FP8 GEMM kernel
   (`r4d_gemm_fp8a8.hip`): `rocwmma::fragment`'s accumulator is opaque, so rescaling per 128-wide K group
   (unavoidable for block-scaled FP8 -- each group has its own scale) required `store_matrix_sync` through
   LDS plus a `__syncwarp()`, 80 times a call, purely to extract 8 register values per thread that were
   already sitting in registers. Fixed: `fragment::operator[]` gives direct per-thread register access.
   rocWMMA documents it as having "no guaranteed order or locality", so this was NOT assumed -- the mapping
   was empirically derived and cross-checked against `store_matrix_sync`'s own documented `mem_row_major`
   output on real hardware (`tools/lab/radiance/hc-down-fp8-regmap.hip`): a consistent bijection across all
   32 lanes, found to be the IDENTITY permutation against the kernel's existing `i=idx/16,j=idx%16,
   idx=lane+e*32` formula -- i.e. `fc[e]` already equals what `store_matrix_sync` would have placed at
   `tile[i][j]`. Replacing the LDS round-trip with direct `fc[t]` reads measured **1422.0us -> 1296.9us**
   at the same shape (flag off still byte-identical, unaffected).
   Still short of baseline, and the kernel still has none of the real `hc_down_pf2_kernel`'s double-buffered
   prefetch (`fetchA`/`stageA` overlapping the next K-slice's global loads with the current slice's WMMAs)
   or its single fused loop across all `NW` n-tiles and `MT` m-tiles -- that remains real follow-up
   engineering, now on top of a kernel with its one clearly-wrong inefficiency already removed.

**Honest current status**: the full pipeline runs correctly-shaped end to end on real hardware, with no
crash and no corruption of other outputs, and received one genuine, measured optimisation this session. It
is still a measured REGRESSION overall (currently slower than the baseline it is meant to replace), and the
diagnostic work above identified exactly why: the FP8 down kernel's inner loop, not launch overhead or an
easily-fixed inefficiency. Matching or beating the baseline needs a real kernel-performance-engineering pass
(double-buffered prefetch, fused tiling) comparable in scope to the hand-tuned original -- that is RAD17's
next real step, not a quick follow-up. A quality metric (replacing the byte-identity check, which fails as
expected for a lossy FP8 path) is also still needed before any promotion decision.

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

## GPT dev-agent review (req_bigcherry_d67c3e42e319)

Full code review against the pushed branch plus `r4d_gemm_fp8a8.hip`/`r4d_fp8_frag.h`/`r4d_fp8_layout.cpp`
for the frag64/raw-WMMA architecture question. **Verdict: NOT READY for promotion.** Findings:

- **P0 (fixed in this revision)**: the original unconditional `if (ks>1) hc_down_reduce_kernel(...)` call
  re-ran after the FP8 path's own down-only reduce, reading uninitialised `part[]` for inject columns and
  overwriting `inj`. See "Outcome" above.
- **P1 (fixed in this revision)**: added explicit `hc==4 && HN%128==0 && kc%128==0` guards to
  `fp8_down_eligible` -- the kernel's K-group bounds assumed `kc`'s 128-alignment without checking it.
- **P1 (not yet fixed)**: `g_fp8a_scratch` is a single process-global allocation with no stream-safety --
  concurrent invocations on different streams could race. Needs integration with the engine's real scratch
  allocation model, not a global static.
- **P2 (not yet addressed)**: no RMS/max-error numerical-equivalence evidence yet (down preactivation,
  injection gain, final x, output quantisation) -- required before any promotion decision, per this
  project's own lifecycle standards.
- **Confirms the frag64/raw-WMMA architecture is the correct long-term direction**, but flags real
  load-bearing details the naive port would get wrong: `gemm_fp8a8_kernel`'s production FP8 path scales
  per 128 output rows x 128 K, while `hc_read`'s W_down scales per OUTPUT ROW x 128 K (finer-grained) --
  reuse the fragment permutation (`r4d_fp8::frag_src` is directly reusable for this K=10240, N=320 shape
  with no modification) and raw-WMMA infrastructure, but adapt the scale lookup, and give `hc_read` a
  distinct layout tag (e.g. `r4d.hc.down.e4m3.frag64.g128`) rather than reusing `r4d.fp8.w.frag64` as if its
  encoding contract were identical. The BF16 fallback decoder needs to learn the new layout too, or a
  persistent-layout change breaks the flag-off path.
- **Highest-value intermediate win identified, not yet attempted**: the FP8 path still pays for
  `hc_pf2_prep_kernel`'s full BF16 W_down/injection plane construction even though it never reads the
  decoded `plane` buffer. Selecting the FP8 path before prep and skipping that unused construction (while
  keeping the norm-scale calculation and W_up decode prep still needs) is flagged as the single most
  important missed optimisation, ahead of the raw-WMMA rewrite.
- Full phased plan (correctness -> remove redundant prep work -> rewrite compute on current weight format
  -> frag64 weight layout -> acceptance) recorded in RAD17's plan notes; not yet executed beyond the P0/P1
  fixes in this revision.
