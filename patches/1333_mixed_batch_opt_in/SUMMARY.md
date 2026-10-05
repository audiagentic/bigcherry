# 1333_mixed_batch_opt_in

**Status:** untested
**Plan item:** QFP22

Kind: performance fix for an upstream regression, flag `BIGCHERRY_MIXED_BATCH` (default 0).

Upstream #29622 (0bb496dbd) builds a third, "mixed token/embd" input branch into every graph of every architecture
that is not on a short exclusion list: three more graph inputs per ubatch (one as large as the embedding input), a
second token-embedding lookup and a `set_rows`, and a mixed-capable batch allocator. The branch only runs for a
batch that mixes token ids and embedding rows, which no BigCherry workload sends.

The patch makes `llm_arch_supports_mixed_batch()` return false unless `BIGCHERRY_MIXED_BATCH=1`, which restores the
pre-#29622 graph and allocator. With the flag set, upstream behaviour is unchanged.

## Evidence

Bisect on Brutus (2x 7900 XTX + R9700 tensor split, 6900 XT MTP drafter), recipe deploy-v6-plus-chunk on every build,
each point compared with b11402 in the same session, greedy text and draft acceptance identical throughout:

| Upstream point | Flash-Next 24K MTP decode (ms/step) | 27B prefill 10K / 32K (t/s) |
|---|---|---|
| 0504396 (old pin) | 41.1 - 41.7 | 1285 - 1291 / 1251 |
| 0eb6d9a81 (#29940) | 41.2 | 1292 - 1296 / 1246 - 1250 |
| 2ca15f540 (#29612) | 41.1 | 1290 - 1291 / 1247 |
| 0bb496dbd (#29622) | 44.6 | pending |
| b11401 (a7fb71fab) | 42.5 | 1279 - 1280 / 1234 |
| b11402 (d89651a7b) | 42.3 - 43.5 | 1270 - 1282 / 1231 - 1236 |

- Which part of the added branch costs the time (input allocation and upload per GPU under the meta backend is the
  suspect) has not been profiled.
- Hardware confirmation of this patch: pending (b11402 + 1333 against b11402 and the old pin).
