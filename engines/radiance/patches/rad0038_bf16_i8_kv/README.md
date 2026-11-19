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

### Promotion record

Tier: profile-evidence

Model: Qwen3.8-27B w8a8 (.rad), qwen27b-tp3-int8-556-long-bf16 (TP3 over two RX 7900 XTX and an R9700, --kv-cache-dtype bf16_i8 and bf16_fp8, up to 262,144-token context)
Model: Qwen3.8-Flash-Next hcbf16 (.rad), two RX 7900 XTX at TP2 with exact wire, experts tiered through the host pool, no helper, no speculation (--kv-cache-dtype bf16_i8 and bf16_fp8)
Mechanics: r4d_selftest on the R9700 (gfx1201) with rad0040, rad0041 and rad0042 applied: 0 failures, 248 skipped (baseline 246; the two extra are the gqa6 block-4 fused rows with no smoke case, as for fp8_v4 and k6_v4), every baseline ok line still passes and 274 new rows pass, including fused gqa12 byte-identity (docs/evidence/2026-11-16-vtp44-generic-kv-layouts/DEVICE-AB.md); libr3 and libr4d compile clean for gfx1100 and gfx1201 with no spill above the existing worst case for bf16_i8 (device-compile-b.md); arch_test, config and bringup tests, patch-lint and the docs catalog tests passed on the patch PRs (#1012, #1013); the libr3 gfx1100 device selftest was not run, the gqa12 and gqa6 rows ran on gfx1100 in Flash-Next serving on the two XTX (FLASHNEXT-AB.md).
Activation: `--kv-cache-dtype bf16_i8` or `bf16_fp8` selects the packed layout from the one layout table: the engine reports the layout and sizes pages from it (Flash-Next attention page 36.75 and 36.00 KiB against 48.00 KiB for bf16, FLASHNEXT-AB.md; 27B 818,896 and 835,952 tokens against 624,048 for bf16, VTP43.md); an unsupported (layout, head_dim, gqa) refuses at startup naming the tuple; the default caches are selected exactly as before.
Identity: the default caches are unchanged: against references the stock build recorded, the patched build scores Flash-Next bf16 6.4e-11 and fp8 3.3e-11 mean KLD with top-1 100% over 4,161 positions (IDENTITY.md), and 27B bf16 6.1e-11 and fp8 4.7e-11 on the short corpus (42,761 positions) and bf16 4.8e-11 on the long one (276,900), top-1 100% (EQUALITY-27B.md); at the tool's float floor, not bit-exact (two binaries); the packed caches differ from bf16 by design and are measured in the A/B line.
A/B: 27B against the stock build's bf16 reference (EQUALITY-27B.md, VTP43.md): long-corpus mean KLD bf16_i8 0.002685 and bf16_fp8 0.003228 against fp8 0.004947 (bar 0.0027 met by bf16_i8), short 0.001411 and 0.001922 against fp8 0.002235; capacity 1.31x and 1.34x bf16; decode at 1, 4 and 8 streams and TTFT within 3% of fp8 (VTP43.md). Flash-Next (FLASHNEXT-AB.md, single runs, two XTX, no helper, no speculation): decode and TTFT within about 1% of bf16 for bf16_i8, bf16_fp8 +6.7% on one-stream decode and 5.9% slower 48K TTFT; capacity 1.283x (bf16_i8) and 1.307x (bf16_fp8) of bf16 because the QSA block-key cache is not quantised; needle probe at 48K tokens 9 of 9 and 8 of 9 against 8 of 9 for bf16; Flash-Next KLD against bf16 sits inside the model's arithmetic-noise band (re-chunking the bf16 cache alone gives the same) so it cannot judge KV error (FLASHNEXT-KLD.md); the R9700 helper arm was not run.

## Shipped configs and requalification (2026-11-19, ENG06)

`bf16_i8` is opt-in (`default-on = false` in the promotion evidence) and now has two named configs:
`qwen27b-tp3-int8-556-long-bf16i8` (818,896 tokens, 3.1 sessions at 262,144) and the non-default
`flashnext-4gpu-bf16i8` (4 sessions: at the production 8 sessions x 262,144 the engine card is 5.47 GB short at admission).
Requalified on current main (docs/evidence/2026-11-19-eng06-bf16i8/README.md): TP3 27B decode 27.6 ms/step (bf16 27.6, fp8
28.0), first token of a 100K prompt 42.5 s (bf16 43.6, fp8 45.1), needles at 48K and 100K 18 of 18 (Flash-Next shipped variant 17 of 18, the miss a refusal); Flash-Next 4gpu with the
helpers and the live policy, two launches: 256K prefill 2916 / 3051 tok/s, one-stream decode 25.8 ms/step (fp8 25.87). The
decode V-scale load is the subject of the separate ENG06 step 1 patch (one scale a token and group, broadcast across the
lanes that share it); it is not part of this package yet.

