# 1007_meta_subgraph_realloc_fix

**Status:** validated
**Plan item:** QFP22

Kind: upstream correctness fix (meta backend), no flag.

`ggml_backend_meta_graph_compute` resets its graph context when a graph raises the node or subgraph high-water mark.
The reset frees every per-subgraph cgraph, but only the current graph's `n_subgraphs` were re-created (with the
current graph's node capacity). A later graph with more subgraphs than that one, but not more than `max_subgraphs`,
skipped the reallocation and used freed cgraph pointers (SIGSEGV in `ggml_backend_meta_graph_compute`).

The fix re-creates `max_subgraphs` cgraphs with `max_nnodes` capacity, which is what the context is sized for.

## Evidence

- b-chunk7 (1332, chunk 256, no MTP, 24.5K prompt): segfault at `mov %eax,0x4(%r13)` = `cgraph_ij->n_nodes = ...`
  with subgraph 96 of 97 on a 6810-node dense 4-token graph, after reserve (7139/6814 nodes) and a 7197-node chunked
  prefill graph (`qfp22-chunk7-diag2/gdb2`). Same binary without chunking does not crash.
- Hardware confirmation: b-chunk8 (same recipe + 1007) runs the same case to completion, rc=0, 39.1 t/s decode vs
  38.4 t/s for chunk 0 (`qfp22-chunk8-nomtp`, 2026-10-05). Promotion record: README.md.
