# rad0037_bf16_fp8_kv

## Summary

VTP43 adds `--kv-cache-dtype bf16_fp8` (serving v2 `kv_cache_dtype = "bf16_fp8"`), a packed paged KV layout: per token
and head, `[0, 2*HD)` K as bf16, `[2*HD, 3*HD)` V as E4M3 (saturating at +-448, scale 1.0, descales absent = 1.0).
That is 3*HD bytes against 4*HD for bf16 (-25%) and 2*HD for fp8, so qwen27b's bf16-key `-long` capacity of about
624k tokens rises to about 830k (+33%) at keys' bf16 quality.

The cache is declared to the core as an E4M3 group of head_dim 3*HD/2 (extent 3*HD bytes a row), the way `fp8_v4` and
`k6_v4` are; only the layout kernels read it. `RAD_KVQ_B16F8 = 3` in the ABI; `bc_kv_quant_of()` (core/rad_core.h) is
the one name -> layout map used by bringup and the streaming check.

| piece | where |
|---|---|
| ABI enum, config parse/help | `abi/rad_builder.h`, `core/config.cpp`, `core/rad_core.h` |
| declare + probe (h256, gqa 6 only) | `arch/qwen35_fp8/qwen35_fp8.cpp` |
| bringup refusal + streaming warning | `core/engine_bringup.cpp`, `core/kvstream/bringup.cpp` |
| prefill / decode kernels (`O_KB16V8`) | `libr4d/r4d_attn_{prefill,decode,paged}_h256_gqa6.hip` |
| rows | `libr4d/r4d_rows.cpp`: `attn_prefill_h256_gqa6_b16f8`, `attn_decode_h256_gqa6_b16f8`, `kv_store_b16f8`, `qk_norm_rope_gate_kv_store_b16f8` |
| store kernels | `libr4d/r4d_model_bf16.hip`, `libr4d/r4d_qk_norm_rope_gate.hip` |
| tests | `tests/{arch,config,mem}_test.cpp`; `libr4d/r4d_selftest.cpp` rows |

## Outcome

Untested at the PR; the hardware run follows (evidence under `docs/evidence/2026-11-16-vtp43-bf16-fp8-kv/`).

## Detail

- The kernels reuse the bf16 cache's K fetch (two uint4 a thread, byte-addressed) and the fp8 cache's V fetch and widen
  to f16; no new numeric path. The e4m3 product legs and `O_KLATE` are off for this layout.
- libr3's native gfx1100 prefill takes the same V fetch (`VBOFF`); its rewrite anchors were updated.
- Refused: another head_dim/gqa (probe), `prefill_fp8` (no row), streaming (a warning, as for the other packed layouts),
  a `k_scale`/`v_scale` other than 1 (the layout has no scale).
