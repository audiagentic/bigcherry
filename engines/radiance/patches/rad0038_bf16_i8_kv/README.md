# rad0038_bf16_i8_kv

## Summary

VTP43 adds `--kv-cache-dtype bf16_i8` (serving v2 `kv_cache_dtype = "bf16_i8"`), a packed paged KV layout: per token and
head, `[0, 2*HD)` K as bf16, `[2*HD, 3*HD)` V as int8 (`q = clamp(rint(v / s), -127, 127)`), then HD/32 f16 scales, one
per 32 V dims, `s` the smallest f16 with `127 * s >= absmax` (floored at 2^-14 so every product is a normal f16; a zero
group stores s = 0). Row bytes `49 * HD / 16` (784 at HD 256) against 1024 for bf16 and 768 for `bf16_fp8`, about 1.31x
bf16's capacity. `RAD_KVQ_B16I8 = 4`; the group is one constant (`RAD_KVQ_B16I8_GROUP`, 32; 16 builds and passes the
same selftests at 800 bytes).

Why: `bf16_fp8` (rad0037) measured long-corpus KLD 0.00336 with two documents' p99.9 at 3.3 and 4.0; 79% of the mass
sat in 0.1% of positions, exact-copy tokens (long identifiers, numbers repeated from far earlier) whose probability the
E4M3 value's 3-bit mantissa flips. Keys are exact there, so the residual is the value's precision.

| piece | where |
|---|---|
| ABI enum + group constant, config, core map | `abi/rad_builder.h`, `core/config.cpp`, `core/rad_core.h` |
| declare + probe (h256, gqa 6 only) | `arch/qwen35_fp8/qwen35_fp8.cpp` |
| bringup refusal + streaming warning | `core/engine_bringup.cpp`, `core/kvstream/bringup.cpp` |
| prefill / decode kernels (`O_KB16I8`, `fetchVi8`/`storeVi8`) | `libr4d/r4d_attn_{prefill,decode,paged}_h256_gqa6.hip` |
| rows | `libr4d/r4d_rows.cpp`: `attn_prefill_h256_gqa6_b16i8`, `attn_decode_h256_gqa6_b16i8`, `kv_store_b16i8`, `qk_norm_rope_gate_kv_store_b16i8` |
| store kernels + shared quantiser | `libr4d/r4d_model_bf16.hip`, `libr4d/r4d_qk_norm_rope_gate.hip`, `libr4d/r4d_common.h` |
| tests | `tests/{arch,config,mem}_test.cpp`; `libr4d/r4d_selftest.cpp` rows |

## Outcome

Untested at the PR; the hardware run follows (evidence under `docs/evidence/2026-11-16-vtp43-bf16-fp8-kv/`).

## Detail

- V dequant is `(0x6400 | (code ^ 0x80)) - 1152` times the f16 scale, no divide; zero scale x any code is 0.
- Refused as for `bf16_fp8`: another head_dim/gqa, `prefill_fp8`, a `k_scale`/`v_scale` other than 1, streaming warns.
- libr3's native gfx1100 prefill reads the same LDS layout `storeVi8` writes; only the selftest rewrite gained the
  b16i8 split-row line.
