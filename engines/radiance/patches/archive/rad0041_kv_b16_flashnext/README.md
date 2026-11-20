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

### Promotion record

Tier: profile-evidence

Model: Qwen3.8-27B w8a8 (.rad), qwen27b-tp3-int8-556-long-bf16 (TP3 over two RX 7900 XTX and an R9700, --kv-cache-dtype bf16_i8 and bf16_fp8, up to 262,144-token context)
Model: Qwen3.8-Flash-Next hcbf16 (.rad), two RX 7900 XTX at TP2 with exact wire, experts tiered through the host pool, no helper, no speculation (--kv-cache-dtype bf16_i8 and bf16_fp8)
Mechanics: r4d_selftest on the R9700 (gfx1201) with rad0040, rad0041 and rad0042 applied: 0 failures, 248 skipped (baseline 246; the two extra are the gqa6 block-4 fused rows with no smoke case, as for fp8_v4 and k6_v4), every baseline ok line still passes and 274 new rows pass, including fused gqa12 byte-identity (docs/evidence/2026-11-16-vtp44-generic-kv-layouts/DEVICE-AB.md); libr3 and libr4d compile clean for gfx1100 and gfx1201 with no spill above the existing worst case for bf16_i8 (device-compile-b.md); arch_test, config and bringup tests, patch-lint and the docs catalog tests passed on the patch PRs (#1012, #1013); the libr3 gfx1100 device selftest was not run, the gqa12 and gqa6 rows ran on gfx1100 in Flash-Next serving on the two XTX (FLASHNEXT-AB.md).
Activation: `--kv-cache-dtype bf16_i8` or `bf16_fp8` selects the packed layout from the one layout table: the engine reports the layout and sizes pages from it (Flash-Next attention page 36.75 and 36.00 KiB against 48.00 KiB for bf16, FLASHNEXT-AB.md; 27B 818,896 and 835,952 tokens against 624,048 for bf16, VTP43.md); an unsupported (layout, head_dim, gqa) refuses at startup naming the tuple; the default caches are selected exactly as before.
Identity: the default caches are unchanged: against references the stock build recorded, the patched build scores Flash-Next bf16 6.4e-11 and fp8 3.3e-11 mean KLD with top-1 100% over 4,161 positions (IDENTITY.md), and 27B bf16 6.1e-11 and fp8 4.7e-11 on the short corpus (42,761 positions) and bf16 4.8e-11 on the long one (276,900), top-1 100% (EQUALITY-27B.md); at the tool's float floor, not bit-exact (two binaries); the packed caches differ from bf16 by design and are measured in the A/B line.
A/B: 27B against the stock build's bf16 reference (EQUALITY-27B.md, VTP43.md): long-corpus mean KLD bf16_i8 0.002685 and bf16_fp8 0.003228 against fp8 0.004947 (bar 0.0027 met by bf16_i8), short 0.001411 and 0.001922 against fp8 0.002235; capacity 1.31x and 1.34x bf16; decode at 1, 4 and 8 streams and TTFT within 3% of fp8 (VTP43.md). Flash-Next (FLASHNEXT-AB.md, single runs, two XTX, no helper, no speculation): decode and TTFT within about 1% of bf16 for bf16_i8, bf16_fp8 +6.7% on one-stream decode and 5.9% slower 48K TTFT; capacity 1.283x (bf16_i8) and 1.307x (bf16_fp8) of bf16 because the QSA block-key cache is not quantised; needle probe at 48K tokens 9 of 9 and 8 of 9 against 8 of 9 for bf16; Flash-Next KLD against bf16 sits inside the model's arithmetic-noise band (re-chunking the bf16 cache alone gives the same) so it cannot judge KV error (FLASHNEXT-KLD.md); the R9700 helper arm was not run.
