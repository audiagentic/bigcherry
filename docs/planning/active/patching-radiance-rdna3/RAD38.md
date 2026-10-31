---
id: RAD38
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-31T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Mixed-precision KV cache for Flash-Next: K at 16 bits, V at 8 (`--kv-cache-dtype k16_v8`)

## Description

The owner's KV rule is 8 bits at the least, 16 preferred, and K is the half that matters most. Radiance 1.14.0 offers `bf16` (K and V 16-bit: 843,144 tokens of capacity on the R9700 in the Flash-Next layout, 0.8 sessions at 1M), `fp8` (both 8-bit: 1,453,872 tokens, 1.4 sessions) and the 4-bit-V forms `fp8_v4`/`k6_v4`, which are below the floor and are not used. Nothing offers K at 16 and V at 8. At 3 head-dim bytes a token and head against 4 for bf16 and 2 for fp8, the mix should hold about 1.1 M tokens (4/3 of bf16) with the K precision the owner prefers.

## Steps

1. Measure first (card-free where possible): quality of fp8 against bf16 KV at long context on Flash-Next, since the 8K KLD gate says nothing about KV rounding at 500K. A long fixed document scored against a bf16-KV reference, K-only and V-only perturbation (fp8 K with bf16 V, and the reverse, by the existing kernels' emulation in `libr4d` selftests) to confirm that K carries the error before building a new layout.
2. The layout: the paged-KV group declares a row per token and head. The 4-bit forms already change that row (`row16` 25 or 22 of 32 in `arch/qwen4exp_fp8` and `arch/qwen35_fp8`); a `k16_v8` row is 48 of 32 (K bf16 2*hd bytes, V E4M3 hd bytes, V scale as in `fp8`). New `RAD_KVQ_K16V8` in `core/rad_core.h` and `abi/rad_builder.h`, the flag in `core/config.cpp`, the group declaration in both arch plugins, the cache sizing and `rad_note`.
3. Kernels: `libr4d` attention reads the layout in `r4d_attn_decode_h256_gqa6.hip`, `r4d_attn_paged_gqa8.hip` (and the gqa12 forms), `r4d_model_bf16.hip`, `r4d_rows.cpp`; each needs a K-bf16 dot and a V-E4M3 dequant path, for decode, prefill and the MTP head (which shares the group). `--attn-prefill-fp8` must refuse the new layout or get a K-bf16 path.
4. Everything that sizes or copies a KV page: `core/kvstream` (`--kv-resident` ring and host mirror), the prefix cache host tier, checkpoint copies, the layout probe tables in `tests/abi_layout_*.h`.
5. As patches on the pinned v1.14.0 source (a numbered patch beside rad0002), not a fork: the engine stays upstream-pinned.

## Detailed Solution & Technical Design

The change is one new row format and the code that reads and sizes it. Because upstream libr4d owns the attention kernels, the work is a patch series: (a) config and enum; (b) group declaration and sizing in both arch plugins; (c) the two to four attention kernels with the new row (reuse the `fp8` V path and the `bf16` K path already in `r4d_attn_*`); (d) selftest rows against a CPU reference at head_dim 256, gqa 12 and 6, as the 4-bit forms do (`r4d_selftest.cpp`); (e) the layout tables. The probe mechanism ("the engine asks only for a layout probe() offered") means a build lacking the kernels refuses the flag instead of running a wrong layout.

## Files

`core/rad_core.h`, `abi/rad_builder.h`, `core/config.cpp`, `core/engine_bringup.cpp`, `arch/qwen4exp_fp8/qwen4exp_fp8.cpp`, `arch/qwen35_fp8/qwen35_fp8.cpp`, `libr4d/r4d_attn_decode_h256_gqa6.hip`, `libr4d/r4d_attn_paged_gqa8.hip`, `libr4d/r4d_model_bf16.hip`, `libr4d/r4d_rows.cpp`, `libr4d/r4d_selftest.cpp`, `core/kvstream/*`, `tests/abi_layout_*.h`; a rad0003 patch and its fixture; the launcher's `--kv-dtype`.

## Validation

Selftest rows for the new layout match the CPU reference; Flash-Next serves with `--kv-cache-dtype k16_v8`; capacity line reports about 1.1 M tokens; long-context quality (step 1's document) at or between bf16 and fp8; prefill and decode within a few percent of fp8 at 8K, 32K and 147K; the KLD gate at 8K unchanged.

## Effort & Risk

Large: engine kernels and two plugins on an upstream-pinned tree, re-done at every radiance bump. Risk: attention kernels at head_dim 256 are tuned for the byte layouts they have; a 3-byte-per-element row changes their memory access. Mitigation: step 1 first; if V at 8 bits costs nothing measurable against bf16, the plan is moot and bf16 is the answer where 0.84M tokens suffice. Depends on the baseline runs of 2026-10-31 (`rad23-base512`, `rad23-kvcap2`).
