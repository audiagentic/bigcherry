# 1292_kpool_tail_truncate

**Status:** validated
**Plan item:** QFN01

Qwen4Exp indexer pool layout (`kpool_layout_update`) rebuilt every sequence from scratch after any edit.
MTP removes the rejected draft tail with `seq_rm [p, inf)` on almost every decode step, so each step walked
the whole cell map: O(n_ctx) host work per step. At ~80K cached context, `perf` on the server showed
`std::_Rb_tree_increment` and `vector::_M_assign_aux` (from that map) as the top libllama symbols while the
GPUs were idle ~75% of each step.

The patch truncates the layout at the stale position (cells before it are unchanged by the stale contract)
and keeps the pools that end before the cut; the append path re-reads the tail. Same layout as the full
rebuild. Sequences with shared cells still rebuild.

Validation: decode t/s at 80K cached context with MTP3, A/B against 1291 alone, greedy
output identical.

## Result (2026-10-03)

Speed: two ABBAs at ~80K cached context, MTP3 (flashnext-kpool-ab-1/2): 77.1/77.3 ms per step (1291 only)
vs 65.5/66.4 and 65.5/65.6 (with 1292), -14%; at 10K -2 to -3%. No effect without MTP (no tail edits).
Parity: greedy text without MTP byte-identical to the 1291+1294 baseline at 10K and 80K, all four arms
(flashnext-kpool-parity-2; 1294 makes long-context runs deterministic, which this check needed).
