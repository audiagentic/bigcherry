# rad0041_kv_b16_flashnext

## Summary

VTP44 patch B. rad0040 made the packed KV layouts one table; this patch adds Flash-Next's geometry for `bf16_fp8` and `bf16_i8` as data plus kernel instantiations. It is opt-in (`--kv-cache-dtype bf16_fp8 | bf16_i8`); the default cache is untouched. Flash-Next is the second validation model for rad0037/38.

| piece | where |
|---|---|
| `RAD_KVQ_B16F8_ROW16`; `RAD_KV_GEOMETRIES(X)`: `X(id, head_dim, gqa)`; `rad_kvq_geometry_ok` | `abi/rad_builder.h` |
| `R4D_ATTN_B16`: prefill, decode and fused decode entry points for both layouts (row bytes from the ABI's row16) | `libr4d/r4d_attn_paged_gqa8.hip`, flag set in the gqa12 and gqa6_bs4 wrappers |
| declarations (one macro, four geometry/layout pairs) | `libr4d/r4d_plugin.h` |
| `R4D_B16_QSA_TUPLES`: constraints and rows for prefill, decode (causal or not) and `attn_paged_gate_quant`; `static_assert`s on the layout name and `RAD_KV_GEOMETRIES` | `libr4d/r4d_rows.cpp` |
| layout name, row size, geometry refusal, probe bits, note | `arch/qwen4exp_fp8/qwen4exp_fp8.cpp` |
| tests (qwen4exp now declares both layouts and its probe offers them) | `tests/arch_test.cpp` |

Not changed, on purpose: the Qwen3.5 plugin keeps its own `gqa 6` check (it serves one geometry; exposing it to the shared list would admit untested ones); libr3 needs no rewrite change (the gqa8 body includes the gqa6 templates whose rewrites rad0037/38 already carry).

## Detail

The pieces above; the evidence below.

## Evidence so far (`docs/evidence/2026-11-16-vtp44-generic-kv-layouts/`)

- `device-compile-b.md`: gfx1201 kernel resource usage (VGPR, spills, LDS) for every new instantiation. No errors or warnings. `bf16_i8` spills at most 22 VGPRs (the existing kernels reach 25); `bf16_fp8` single-wave decode spills 45 against at most 20 for the existing single-wave variants, so Flash-Next decode needs a speed A/B.
- gfx1100: the rewritten gqa12 and gqa6_bs4 units compile with libr3's compat layer, 0 errors, 10 `b16` host entry points and 36 kernels each; `r3_rewrite.py --check` passes.
- Host: `r4d_rows.cpp` compiles with the new tuples (12 rows by name); `arch_test` keeps its 2 pre-existing failures and the packed-KV case passes for both plugins; `config_test` passes.

## Outcome

Untested: the compile and host evidence above is in; the device and model evidence below is pending.

## Not yet verified

r4d_selftest on the R9700 against the rad0038 baseline (0 failures, 246 skipped); new selftest cases for the gqa12 and block-4 rows; 27B KLD equality; Flash-Next KLD, speed and identity on `flashnext-tp2xtx-r9700`.
