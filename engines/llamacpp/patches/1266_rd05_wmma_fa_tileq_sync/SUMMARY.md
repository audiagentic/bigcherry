# 1266_rd05_wmma_fa_tileq_sync

**Patch metadata state:** untested (unchanged). **Operational disposition:** do not apply to b11552 unchanged; correctness-only historical candidate, not promoted. **Owner:** PRBE110 (patch); PRBE02 (residual kbc-boundary decision).

This package extracted two synchronization edits from rejected 1203, plus a once-per-process WMMA dispatch marker. Four archived gfx1201 sessions passed full-vocabulary correctness and marker admission but did not improve throughput (target -0.0499%, CI95-low -0.1609%); they did not reproduce a stock race.

Merged llama.cpp [#30103](https://github.com/ggml-org/llama.cpp/pull/30103) changes the stock end-of-k00 guard in b11552 to `np > 1 || nbatch_combine != DKQ/2`. That **supersedes the first 1266 edit** and removes its old text anchor. The second inter-`process_tile` barrier is not upstream-absorbed, but no residual race is demonstrated. Do not silently reanchor the old package, double-insert barriers, inherit 1203 evidence, or queue a new benchmark. See PRBE02 and BCOP136 for the exact post-#30103 correctness gate.
