# VTP44 patches A and B: device selftest on the R9700 (gfx1201)

`r4d_selftest` with rad0040, rad0041 and rad0042 applied (build `vtp43/hb-b`, plugin and executable hashes in the run's `receipt.txt`), pinned to
`HIP_VISIBLE_DEVICES=2` (the gfx1201 card), run under the `gpu` lease on 2026-11-16 (14:19 to 14:48 UTC), compared line by line with the corrected rad0038
baseline (`BASELINE-STATUS.md`).

| | rad0038 baseline | rad0040 + rad0041 + rad0042 |
|---|---|---|
| failures | 0 | **0** (exit 0) |
| skipped | 246 | 248 |
| `ok` lines | 7576 | 7850 |
| baseline `ok` lines no longer `ok` | | **0** |

- Every line that passed in the baseline passes now: patch A (the table refactor) and the rows' move onto generated tuples change no result on the device.
- 274 new `ok` lines, all for the new rows (`attn_prefill_h256_gqa12_{b16f8,b16i8}`, `attn_decode_h256_gqa12_*`, and the `gqa6_bs4` pairs): numeric cases with their negative controls (gqa12 decode at 1 to 5 rows, prefill 64 and 100 rows), the split-KV prefill (split against unsplit and against the oracle), the mixed step and the declare-time scratch.
- The fused gate+quantiser rows `attn_decode_gq_h256_gqa12_{b16f8,b16i8}` are byte-identical to the decode plus `gate_quant_fp8` pair at 3 and 24 rows.
- The two extra skips are `attn_decode_gq_h256_gqa6_bs4_{b16f8,b16i8}` ("no smoke case"): the same state as the existing `attn_decode_gq_h256_gqa6_bs4_{fp8v4,k6v4}` rows, whose fused form has a gqa12 case only. Not a regression; a gqa6 block-4 fused case is a follow-up if Flash-Next's four-rank layout is served.

## A characteristic to carry into the Flash-Next validation
bf16_fp8's split-versus-unsplit difference can reach 0.25 to 0.5 at long contexts (`attn_prefill_h256_gqa12_b16f8 ... 53/18000`: 0.25; the same shape at gqa 6, block 16 in the rad0038 baseline: 0.5), against 3e-05 for the fp8 row and 1e-03 for bf16_i8. The cases pass (their bound is relative to the output's scale and the oracle comparison passes), so this is a property of E4M3 V with bf16 K accumulation order, not a failure; Flash-Next's long-context KLD and identity check must watch for it.

## Not covered by this run
27B KLD equality for patch A, Flash-Next KLD, speed (the bf16_fp8 single-wave decode spills 45 VGPRs, `device-compile-b.md`) and identity on `flashnext-tp2xtx-r9700`; libr3 (gfx1100) device selftest.
