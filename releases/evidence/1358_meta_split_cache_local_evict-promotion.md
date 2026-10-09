# 1358_meta_split_cache_local_evict promotion evidence

Promotion evidence, profile-evidence tier (QFP18), pin b11474, 2026-10-09.

Model: Qwen3.8-Flash-Next UD-IQ4_XS
Model: Qwen3.8-27B Q8_0

Mechanics: offline tests for 1358 pass, patch-lint clean, composition with the production set clean (58/58).
Activation: BIGCHERRY_PATCH_HIT marker at 24K in the on arm only (run sce1-r1, order A B B A: 1 0 0 1).
Identity: greedy text identical in every run at 8K / 24K / 98K on Flash-Next, and at 10K / 32K on Qwen3.8-27B.
A/B: one binary (b-metamem-sce1 = production + 1358), on against BIGCHERRY_META_SPLIT_CACHE_EVICT=0, two rounds: prefill +3.1% at 98K, +3.0% at 24K, +2.8% at 8K.
No-regression: Qwen3.8-27B, two RX 7900 XTX, production flags (prod27b-ab.sh, run nr27-1358): prefill +1.0% at 10K and +1.1% at 32K, decode and draft acceptance equal, text identical.

## Flash-Next, production profile

2x RX 7900 XTX + R9700 tensor split, MTP drafter on the RX 6900 XT, ctx 245760, f16 KV, ub512. Chain c171,
`tools/lab/flash-next/queue-split-cache-evict.sh`, A = default (on), B = off.

| Depth | On (t/s) | Off (t/s) | Gain |
|---|---|---|---|
| 98K, both rounds | 1255.7 / 1259.4 / 1259.5 / 1263.5 | 1221.9 / 1216.0 / 1223.2 / 1223.1 | +3.1% |
| 24K, round 2 | 1300.7 / 1308.9 | 1270.7 / 1267.2 | +3.0% |
| 8K, round 2 | 1282.9 / 1248.6 | 1246.1 / 1217.6 | +2.8% |

At 98K every on-run is above every off-run (4 against 4). At 8K the narrowest gap is 2.5 t/s. Decode and draft
acceptance are unchanged.

A two-build ABBA before the patch was split out of 1328 (runs aux3off-r1/r2, production against production + 1328
with expert offload off) gave the same result: +2.8% / +3.2% / +2.9% at 8K / 24K / 98K, text identical.

## Qwen3.8-27B, two-build ABBA

A = production build b-main2, B = b-metamem-sce1, -sm tensor, built-in MTP, cpu-root AllReduce.

| Depth | Production (t/s) | With 1358 (t/s) |
|---|---|---|
| 10K | 1297.4 / 1300.2 | 1311.7 / 1310.6 |
| 32K | 1255.1 / 1253.1 | 1269.0 / 1267.1 |

Decode 72.5-74.8 t/s in both builds; acceptance 177/310 and 181/293 in every run.

## Limits

The 27B comparison is between two builds, not one binary with the off switch. Gemma 4 26B and Qwen3.6-35B runs were
queued but had not finished when this was written.

## Native llama.cpp baseline

No separate native build was needed for this mechanism: with `BIGCHERRY_META_SPLIT_CACHE_EVICT=0` the cache reacts to
a recycled tensor address exactly as native llama.cpp b11474 does (`split_state_cache.clear()`), on the same binary.
The off arm of the ABBA is therefore the native llama.cpp behaviour for the code this patch changes, the on arm is
BigCherry with the patch, and the rest of the build is the BigCherry production baseline in both arms.
