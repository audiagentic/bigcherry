# 1333_mixed_batch_on_demand

**Status:** untested
**Plan item:** QFP22

Kind: performance fix for an upstream regression, no flag.

Upstream #29622 (0bb496dbd) builds a third, "mixed token/embd" input branch into every graph of every architecture
that is not on a short exclusion list: three more graph inputs per ubatch (one as large as the embedding input), a
dup, a second token-embedding lookup and a `set_rows`. The branch is only selected for a ubatch that mixes token ids
and embedding rows.

The patch adds `ubatch.is_mixed()` to the condition in `build_inp_embd`, so token-only and embedding-only ubatches
get the pre-#29622 graph and a mixed ubatch gets upstream's graph. Graph reuse already distinguishes the two
(`llm_graph_params` compares `is_mixed()`). The batch allocator is untouched and still accepts mixed batches.

Trade-off: the mixed graph is not part of the worst-case reserve any more, so the first mixed ubatch of a context
makes the scheduler reallocate once.

## Evidence

Bisect on Brutus (2x 7900 XTX + R9700 tensor split, 6900 XT MTP drafter), recipe deploy-v6-plus-chunk on every build,
each point compared with b11402 in the same session, greedy text and draft acceptance identical throughout:

| Upstream point | Flash-Next 24K MTP decode (ms/step) | 27B prefill 10K / 32K (t/s) |
|---|---|---|
| 0504396 (old pin) | 41.1 - 41.8 | 1285 - 1292 / 1251 |
| 0eb6d9a81 (#29940) | 41.2 | 1292 - 1296 / 1246 - 1250 |
| 2ca15f540 (#29612) | 41.1 | 1290 - 1291 / 1247 |
| 0bb496dbd (#29622) | 44.6 | 1267 - 1276 / 1228 - 1230 |
| b11401 (a7fb71fab) | 42.5 | 1279 - 1280 / 1234 |
| b11402 (d89651a7b) | 42.3 - 43.5 | 1255 - 1282 / 1227 - 1236 |

First form of this patch (whole feature behind `BIGCHERRY_MIXED_BATCH`, default off; build b-mixoff on b11402)
against the old-pin build in one session: Flash-Next 41.1 ms/step vs 41.8 / 41.4; 27B prefill 10K 1290.7 / 1291.1 vs
1291.9 t/s. That form was replaced by the on-demand condition before promotion.

- Hardware confirmation of the on-demand form: pending.
- Which part of the branch costs the time has not been profiled (for Flash-Next the token embedding table is on the
  CPU, so the second lookup adds a CPU-side node to every graph).
- No workload with real mixed batches has been run with this patch.
