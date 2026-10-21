# rad0007_hc_pf2_fp8_down

**Status:** untested
**Plan item:** RAD15/RAD17

## Summary

Inert scaffold for an FP8 WMMA down-product in hc_read's PF2 prefill form (gfx1201, Qwen3.8-Flash-Next). Adds
three new kernels to `libr4d/r4d_hc_bf16.hip` -- none of them are called yet; this revision only proves the
anchors apply and the new code compiles against the real pinned source.

## Outcome

Not yet measurable: no dispatch path reaches the new kernels in this revision.

## Detail

See below.

## What it does

- `hc_pf2_quant_a8_kernel`: activation quantisation pre-pass, one wave (32 lanes) per (row, 128-of-K group).
  Validated standalone against real ground-truth N: `tools/lab/radiance/hc-pf2-quant-a8.hip` (0 mismatches
  across 80 scales + 10240 codes, 2.53% RMS error).
- `hc_down_pf2_fp8_kernel`: the FP8 WMMA down-product. Reads `W_down` directly from its real on-disk bytes
  (one packed `[K codes][K/128 f32 scales]` blob per row, stride `row_bytes(K,128)`, confirmed against the
  live container -- 10560 for K=10240, not the naive 10240) rather than the bf16-decoded `plane` buffer the
  existing kernel uses. The per-128-group WMMA + layout math is validated standalone, on real activation and
  real weight bytes together: `tools/lab/radiance/hc-down-fp8-real-layout.hip` (max rel diff 3.2e-6, fp32
  noise floor). The rescale-and-accumulate epilogue at the real RT=64/kWaves tiling is **not yet written** --
  marked with a TODO in the source; this is RAD17's next integration step, not a correctness question (the
  math it needs is already proven in the standalone harness at a smaller scale).
- `hc_inj_pf2_bf16_kernel`: placeholder only (not yet implemented). The injection weight has no FP8 variant in
  the schema, so this will be a direct extraction of `hc_down_pf2_kernel`'s own existing arithmetic, not new
  numerics.

No dispatch wiring: `hc_pf2_launch` is untouched by this revision, so these kernels compile but are never
called. `RAD_HC_PF2_FP8_DOWN` and the guard checks (including `kc % 128 == 0` for split-K alignment, per the
design review) are added in a later revision once the epilogue above is finished.

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
