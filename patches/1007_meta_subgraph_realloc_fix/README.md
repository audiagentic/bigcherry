# 1007_meta_subgraph_realloc_fix

Promotion record (QFP18 lightweight evidence-reuse tier, pin 0504396). Mechanism and root cause are in SUMMARY.md;
this file records why the patch is promoted into `[patch-set.upstream-fixes]`.

## Evidence

- Kind: correctness fix for an upstream meta backend bug (use of freed subgraph cgraphs), no flag, no performance
  claim. Upstream master still carries the same loop as of 2026-10-05; not reported upstream yet.
- Reproduction (without the fix): build b-chunk7, Qwen3.8 Flash-Next, 2x 7900 XTX + R9700 tensor split, no MTP,
  24.5K prompt, 1332 chunk 256 - SIGSEGV in `ggml_backend_meta_graph_compute` at `cgraph_ij->n_nodes = ...`,
  subgraph 96 of 97 on a 6810-node graph that followed a 7197-node graph (`qfp22-chunk7-diag2/gdb2`). 1326 off
  and scheduler realloc-debug runs excluded the async-input and allocator paths.
- Fix confirmed: build b-chunk8 (same recipe + 1007) completes the same case, rc=0, 39.1 t/s decode vs 38.4 t/s
  for chunk 0 (`qfp22-chunk8-nomtp`).
- No regression: b-chunk8 unchunked arms match the earlier baselines - 24K MTP 41.5 / 41.3 ms/step (b-chunk7
  42.9 / 41.3), 80K ub512 868.6 / 891.3 t/s prefill and 60.5 / 61.2 t/s decode (b-chunk7 871.8 / 892.1 and
  61.1 / 60.5). The edit only runs on the reallocation path (node or subgraph high-water mark raised).
- Mechanics: offline patch test (`tools/tests/patch/test_1007_meta_subgraph_realloc_fix.py`) and patch-lint pass on
  pin 0504396.

Not covered: the text difference between chunked and dense 1332 output is a separate open issue (QFP22) and is not
attributed to this bug.
