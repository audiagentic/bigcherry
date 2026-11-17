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

### Promotion record

Tier: profile-evidence

Model: Qwen3.8-27B w8a8 (.rad), qwen27b-tp3-int8-556-long-bf16 (TP3 over two RX 7900 XTX and an R9700, --kv-cache-dtype bf16_i8 and bf16_fp8, up to 262,144-token context)
Model: Qwen3.8-Flash-Next hcbf16 (.rad), two RX 7900 XTX at TP2 with exact wire, experts tiered through the host pool, no helper, no speculation (--kv-cache-dtype bf16_i8 and bf16_fp8)
Mechanics: r4d_selftest on the R9700 (gfx1201) with rad0040, rad0041 and rad0042 applied: 0 failures, 248 skipped (baseline 246; the two extra are the gqa6 block-4 fused rows with no smoke case, as for fp8_v4 and k6_v4), every baseline ok line still passes and 274 new rows pass, including fused gqa12 byte-identity (docs/evidence/2026-11-16-vtp44-generic-kv-layouts/DEVICE-AB.md); libr3 and libr4d compile clean for gfx1100 and gfx1201 with no spill above the existing worst case for bf16_i8 (device-compile-b.md); arch_test, config and bringup tests, patch-lint and the docs catalog tests passed on the patch PRs (#1012, #1013); the libr3 gfx1100 device selftest was not run, the gqa12 and gqa6 rows ran on gfx1100 in Flash-Next serving on the two XTX (FLASHNEXT-AB.md).
Activation: `--kv-cache-dtype bf16_i8` or `bf16_fp8` selects the packed layout from the one layout table: the engine reports the layout and sizes pages from it (Flash-Next attention page 36.75 and 36.00 KiB against 48.00 KiB for bf16, FLASHNEXT-AB.md; 27B 818,896 and 835,952 tokens against 624,048 for bf16, VTP43.md); an unsupported (layout, head_dim, gqa) refuses at startup naming the tuple; the default caches are selected exactly as before.
Identity: the default caches are unchanged: against references the stock build recorded, the patched build scores Flash-Next bf16 6.4e-11 and fp8 3.3e-11 mean KLD with top-1 100% over 4,161 positions (IDENTITY.md), and 27B bf16 6.1e-11 and fp8 4.7e-11 on the short corpus (42,761 positions) and bf16 4.8e-11 on the long one (276,900), top-1 100% (EQUALITY-27B.md); at the tool's float floor, not bit-exact (two binaries); the packed caches differ from bf16 by design and are measured in the A/B line.
A/B: 27B against the stock build's bf16 reference (EQUALITY-27B.md, VTP43.md): long-corpus mean KLD bf16_i8 0.002685 and bf16_fp8 0.003228 against fp8 0.004947 (bar 0.0027 met by bf16_i8), short 0.001411 and 0.001922 against fp8 0.002235; capacity 1.31x and 1.34x bf16; decode at 1, 4 and 8 streams and TTFT within 3% of fp8 (VTP43.md). Flash-Next (FLASHNEXT-AB.md, single runs, two XTX, no helper, no speculation): decode and TTFT within about 1% of bf16 for bf16_i8, bf16_fp8 +6.7% on one-stream decode and 5.9% slower 48K TTFT; capacity 1.283x (bf16_i8) and 1.307x (bf16_fp8) of bf16 because the QSA block-key cache is not quantised; needle probe at 48K tokens 9 of 9 and 8 of 9 against 8 of 9 for bf16; Flash-Next KLD against bf16 sits inside the model's arithmetic-noise band (re-chunking the bf16 cache alone gives the same) so it cannot judge KV error (FLASHNEXT-KLD.md); the R9700 helper arm was not run.
