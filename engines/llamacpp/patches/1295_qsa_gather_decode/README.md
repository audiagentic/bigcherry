# 1295_qsa_gather_decode

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11402 / d89651a7, 2026-10-08). Mechanism and the
2026-10-03 screening are in SUMMARY.md: for batches of at most 8 tokens and a cache above
`BIGCHERRY_QSA_GATHER_MIN` cells, Qwen4Exp's sparse attention gathers each token's selected cells and attends over
them instead of masking the whole cache. On by default; `BIGCHERRY_QSA_GATHER=0` restores the masked path.

It was left at "evaluated" for two reasons, both closed:
- its output was not repeatable from run to run. Cause: the top-k kernel wrote the selected cells in atomic arrival
  order, and the gathered path sums over them in that order. 1294 now writes the selection in ascending order
  (`BIGCHERRY_TOPK_ORDERED`, on by default); with it the gathered path gives one text per depth.
- it ran out of memory on the R9700 at the deepest context tier. With the per-device compute arena (1340) and the
  subset-mirrored indexer cache and masks (1341) it loads and runs at 196K depth in the 245760 context.

## Evidence

Flash-Next UD-IQ4_XS, production topology (2x RX 7900 XTX gfx1100 + R9700 gfx1201 tensor split, MTP drafter on the
RX 6900 XT), ctx 245760, f16 KV, ub512. ABBA on one binary with `tools/lab/flash-next/queue-env-ab.sh`.

- Repeatability (build `b-metamem-qo2`, gathered decode on both arms, A = ordered top-k / B =
  `BIGCHERRY_TOPK_ORDERED=0`, run `qo-gather`): one greedy md5 for the two A runs at 24K and at 98K; two different
  md5s for the two B runs at both depths. On production (masked path, run `qo-ab`) the ordering changes nothing:
  identical text at 8K and 98K, probes top-1 24/24 and TV 0.0000, decode 84.1 / 84.5, 84.2 t/s at 8K and
  57.9 / 57.5, 58.6 at 98K (second A run against the two B runs; the first A run of each pair was the cold load).
- Decode, A = masked / B = gathered, t/s (run `qo-gather-ab`, ordered top-k on both arms): 24K 72.3, 71.2 / 72.4,
  72.4 (equal); 98K 56.9, 57.6 / 65.4, 65.7 (+14%, complete separation, n = 2). Earlier the same day without the
  ordering (run `qg-ab`): 98K 58.4, 58.8 / 63.9, 66.2; on the low-VRAM layout (run `qg-lv-ab`) 196K 44.4, 44.1 /
  50.6, 49.2 (+13%). Prefill is unchanged in every run (the path takes batches of at most 8 tokens).
- Accuracy (run `qg-ab`, 24 probes at 24K against the CPU f32 reference): masked top-1 22/24, TV mean 0.0739;
  gathered top-1 22/24, TV mean 0.0793; gathered against masked top-1 24/24, TV mean 0.0555. The generated text
  differs from the masked path (another summation). Stated equivalence, not identity.
- Offline: package tests, patch-lint, production composition check.

## Native llama.cpp comparison

The masked path of arm A is native llama.cpp b11402's Qwen4Exp sparse attention on HIP (upstream's gathered kernel is
compiled out for HIP), so the ABBAs are native against gathered inside one BigCherry binary. No separate run against a
fully native binary was made for this patch.
