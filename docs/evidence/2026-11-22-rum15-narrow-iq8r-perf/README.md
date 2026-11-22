# RUM15 — isolate the F32-scale narrow-GEMM slowdown

**2026-11-22.** Follow-up on the merged RUM15 precision-preserving diagnostic patch `rad0061` and its measured admission evidence. This note is a **rejected-optimization result**: no additional kernel code is selected or committed. Keep the CPU `libref` FP32 reference and already-verified native C06 exact QKV outputs intact.

## The measured issue

The installed stock BF16 scale-fold and the numerically corrected diagnostic FP32 scale-fold were compared on the same R9700 (gfx1201), same released Radiance ABI, same shipped `rad-kbench` Qwen3.8-27B-IQ8R fixture, and the existing `gemm_nt_q` op. A/B/A uses device timestamps, 750-ms warm-up, 10 repeats, one case per geometry, `--bench-flush 0`, and the arithmetic kernels' original selection. The registered test row is **`gemm_iq8r_nt_m16`** although the benchmark's operation filter is **`gemm_nt_q`**.

At **M=33, N=1024, K=5120, group=256**, the R9700 FP32 scale layout consistently costs approximately 25% more than BF16. Independent captures of the same kernel row:

| Experiment | BF16-stock first | FP32 candidate | BF16-stock repeat | FP32/stock (A/B/A) |
| --- | ---: | ---: | ---: | ---: |
| Original kernel dispatch | 5.71 µs | 7.15 µs | 5.74 µs | +24.89% |
| Existing native wave-count hook forced to MW=4 | 5.73 µs | 7.24 µs | 5.75 µs | +26.13% |

The original `narrow_mw(N,M)` chooses MW=8 for this shape. With the existing `kSplitTarget=128` policy, MW=8 yields `gx=8`, two K splits and **16 workgroups**; pinning MW=4 yields `gx=16`, still two K splits but **32 workgroups**, with the same overall 128 dispatched waves. `r4d_gemm_fp8a8_pin_mw` is a **pre-existing native test hook**, invoked only by a temporary local diagnostic interposer; the captured kernel MW logs explicitly confirm that all three benchmark passes used MW=4 for N=1024 and M=33.

**Conclusion:** doubling the number of workgroups without changing total waves does not recover the ~25% gap. Do **not** encode an N/M-specific MW dispatch branch based on this hypothesis.

## Compiled gfx1201 resource and instruction comparison

Inspected the **actual old-stock and F32-diagnostic** `r4d_gemm_fp8a8.hip.o` HIP offload bundles from the pinned compiler. Extracted the gfx1201 device ELFs, read `amdhsa.kernels` metadata, and disassembled the exact unchained template `gemm_fp8a8_kernel<Full=false, NT=3, MW=8, FK=256, GATED=false, I8=true, BIAS=false, SB=256>`.

| Codegen attribute | BF16 stock | FP32 staged |
| --- | ---: | ---: |
| VGPR count | **167** | **167** |
| SGPR count | 56 | 55 |
| Private-segment bytes | **0** | **0** |
| VGPR/SGPR spills | **0 / 0** | **0 / 0** |
| Fixed group LDS bytes | **13,124** | **13,380** |
| `ds_load_b128` instruction count | 16 | 32 |
| `global_load_b128` instruction count | 28 | 28 |
| Total static disassembled instructions | 5,881 | 5,733 |
| Kernel code byte count | 32,392 | 31,492 |

The F32 scale format adds 256 bytes of LDS per workgroup and doubles the count of static 128-bit LDS loads at this specialization. There is **no register spill and no larger VGPR requirement**, so neither register spilling nor the proposed grid-size workaround explains the gap.

These are static machine-code counts, **not measured hardware-cache misses or dynamic instruction totals**. Additional scale bytes and LDS traffic remain plausible causes; cache capacity and instruction scheduling require measurement before attribution.

## Test of a smaller scale-staging pathway

Built an **isolated scratch HIP object** that retains the exact same FP32 scale product but loads each group-256 weight scale **directly from global memory at the existing fold** instead of prefetching it through LDS. It reuses the *same* `libr4d/r4d_gemm_fp8a8.hip` kernel and complete native plugin ABI; only this one HIP object was rebuilt and substituted into an isolated diagnostic link. The exact stage and reference plugin, full IQ8R layout, other HIP objects, FP32 fold arithmetic, output BF16 and production installation were unchanged.

The existing native R9700 IQ8R selftest with the **higher-precision reference** passes **35/35 checks, exit 0**, including narrow, wider, group-256 activation producer and tiled paths.

Direct versus staged FP32, using another independent device-timed A/B/A on the same fixture:

| M=33, N=1024, K=5120 | F32 staged first | F32 direct | F32 staged repeat | Direct/staged |
| --- | ---: | ---: | ---: | ---: |
| Time | 7.16 µs | **7.20 µs** | 7.19 µs | **+0.35%** |

Across all **54 matching IQ8R rows**, the median direct/staged change is **0.00%**. Other isolated cells sometimes improve (for example M=64, N=12288), but this is a single pass and does not justify merging an additional scale-load path. The persistent narrow regression is **not fixed by bypassing LDS staging**. Retain the already-reviewed source and reject this experimental source change.

The scratch plugin, device ELF metadata, single-card selftest receipts and benchmark reports remain outside Git; this evidence records the numbers and test methodology without introducing any production helpers.

## Rotating-weight A/B/A (both architectures)

The same unmodified native benchmark was repeated with **--bench-flush 256** instead of 0: 256 MiB of rotated weight/input buffers, intended to simulate repeated streams rather than a single warm-cache copy. This is **not** measured actual offload/refill or LTX video generation. Both runs retained the same shipped Qwen IQ8R fixture, shape coverage, 750-ms warm-up, 10 device timing batches, 256-MiB case bound and stock → FP32 → stock control, under separate exclusive GPU leases. Each device produced **77 timed cases, including 54 matched IQ8R kernel rows**, and released the lease.

| Metric | R9700 gfx1201 | XTX gfx1100 |
| --- | ---: | ---: |
| Median FP32/stock change over 54 IQ8R rows | **+0.63%** | **+0.27%** |
| Median absolute A/A control drift | 0.08% | 0.51% |
| Median narrow GEMM at M ≤ 16 | +0.72% | +0.41% |
| Median narrow GEMM at 17 ≤ M ≤ 64 | +0.78% | −0.48% |
| Median tiled prefill at M ≥ 65 | −0.20% | +0.15% |

The R9700 repeatable small-N regression **persists with rotated weights**:

| R9700, N=1024, K=5120, group=256 | BF16 stock A1 | FP32 | BF16 stock A2 | FP32 delta |
| --- | ---: | ---: | ---: | ---: |
| M=1 | 10.99 µs | 11.95 µs | 10.99 µs | **+8.74%** |
| M=8 | 10.99 µs | 11.96 µs | 10.99 µs | **+8.83%** |
| M=33 | 11.27 µs | 13.24 µs | 11.27 µs | **+17.48%** |

On the XTX, other narrow IQ8R shapes also incur costs in this pass: **M=32, N=5120, K=6144** is 168.24/168.28 µs stock versus 181.34 µs FP32 (**+7.77%**, 0.02% stock drift); **M=18 at the same N/K** is 168.56/169.67 µs stock versus 181.78 µs FP32 (**+7.49%**, 0.66% drift). These are Qwen fixture geometry, **not confirmed LTX inference bottlenecks**.

The warm single-copy aggregate median changes were −0.59% (R9700) and −0.61% (XTX); rotating weights move medians slightly above zero. **Do not interpret the warm medians as throughput improvements for streamed inference**. The isolated R9700 and XTX slow paths are the production-admission concern, and the data support retaining the higher-precision format only as diagnostic pending realistic LTX serving measurements.

## Outstanding RUM15 acceptance work

1. Rotating-weight comparisons are now recorded above, but actual **LTX2.5** M=1/16/64 shapes, streaming loader tiers and model-quality impacts are unmeasured. Confirm the slowdown's applicability to served LTX before selecting another scale representation.
2. If the 25% narrow regression persists on served LTX shapes, evaluate the *original U8 per-group plus BF16 per-row* source-factor representation, multiplied **in FP32 at the fold**, in the existing layout/kernel family. Do not duplicate the loader, kernels or CPU math. It requires a real layout and memory-budget assessment, not a guessed constant or lowered reference precision.
3. Verify the changed layout tag, stored-plan/cache invalidation, actual per-rank VRAM placement, host/SSD refill and reload, TP1 captured activations, video/audio quality and 16/64-row C06 execution before production selection.

**Decision:** The FP32 precision correction stays diagnostic. Neither a wave-count special case nor bypassing the LDS scale stage fixes the narrow regression; further changes need measured evidence.
