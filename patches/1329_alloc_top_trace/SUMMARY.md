# 1329_alloc_top_trace

**Status:** untested
**Plan item:** QFP17

## What it does

Diagnostic only. With `BIGCHERRY_ALLOC_TOP=N`, `ggml_gallocr_reserve_n` logs for every reserved graph the N largest
non-view node tensors (MiB, op, name, shape, buffer id) and the summed bytes per op. Used to attribute the compute-buffer
growth from -ub 512 to 1024 (MoE intermediates vs attention / QSA indexer / mask buffers) before designing chunked
prefill scratch.
