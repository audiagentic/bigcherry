# 1333_mixed_batch_on_demand

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11402 / d89651a7). Mechanism and the bisect are in
SUMMARY.md; this file records why the patch is promoted into `[patch-set.validated-enhancements]`.

## Evidence

Brutus, 2x 7900 XTX + R9700 tensor split, 6900 XT MTP drafter, 2026-10-05. Every build uses recipe
deploy-v6-plus-chunk; the patched builds add only this patch. Greedy text and draft acceptance are identical between
the arms of every comparison below.

- Activation: the bisect isolates upstream #29622 (fast at 2ca15f540, slow at 0bb496dbd), and removing its always-on
  mixed branch restores the old speed, so the mechanism is the one the patch targets. There is no runtime marker
  (the patch removes graph nodes; it adds no code path of its own).
- On-demand form against the previous pin's build (`b-mixauto` vs `b-chunk8`): Flash-Next 24K MTP decode 41.2 ms/step
  vs 41.8 (a second old-pin sample read 44.4 and is treated as a stray: that build measured 41.1 - 41.8 in six other
  runs the same day); 27B prefill 10K 1289.9 / 1290.6 vs 1291.8 / 1290.9 t/s; 32K 1246.1 / 1244.3 vs 1251.9 / 1248.4.
- Flag form (same graph for unmixed batches, build `b-mixoff`) against unpatched b11402 (`b-bump3-b11402`), complete
  separation where both arms have two clean samples: Flash-Next 41.2 vs 43.4 / 44.4 ms/step; 27B prefill 32K
  1246.5 / 1243.9 vs 1236.0 / 1234.1 t/s; 10K 1291.6 vs 1254.2 / 1278.5 (the other patched 10K sample, 1145.7, is a
  stray).
- On-demand form against the flag form (`b-mixauto` vs `b-mixoff`): no difference - Flash-Next 41.7 vs 41.8 / 41.1
  ms/step; 27B prefill 10K 1288.5 / 1290.3 vs 1288.4 / 1289.7; 32K 1244.9 / 1244.6 vs 1247.2 / 1244.7 t/s.
- Real mixed batches: upstream `test-llama-archs` (mixed batch vs the same input as token/embd/token batches, which
  alternates the two graph shapes) for qwen4exp, qwen35, qwen35moe and gemma3 on each 7900 XTX, the R9700, the CPU
  and the Meta tensor split: all OK on the patched build, error 1e-13 .. 3e-13 on the GPUs and Meta, same as the
  unpatched build (`tools/lab/flash-next/mixed-batch-test.sh`, `qfp22-mixed-batch-test`).
- Mechanics: offline patch test and patch-lint pass on pin b11402; composes with the build selection.

## Native llama.cpp comparison

Pending: native llama.cpp b11402 (source `llama-native`, no patches) against native + this patch alone
(experiment `native-plus-1333`), Qwen3.8-27B dual-XTX production config, `tools/lab/flash-next/queue-native-1333.sh`.
The comparisons above are BigCherry builds on both sides; the unpatched arm runs upstream's own code for this path.

## Not covered

- Mixed batches at real model size, at a tight VRAM limit, or their transition cost (the test models are tiny).
  The server does not send mixed batches yet (upstream lists mtmd support as a to-do).
- 27B prefill at 32K remains about 0.4% below the previous pin with this patch; that residual has another cause.
- The regression has not been reported upstream.

Runs: `/mnt/data/bigcherry-work/runs/` `mixoff-vs-old-*`, `mixoff-vs-new-*`, `b-mixauto-vs-b-chunk8-*`,
`b-mixauto-vs-b-mixoff-*`, bisect `bisect3..6-*`.
