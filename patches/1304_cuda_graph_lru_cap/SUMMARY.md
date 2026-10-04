# 1304_cuda_graph_lru_cap

**Status:** rejected
**Plan item:** QFP06

## What it does

`BIGCHERRY_CUDA_GRAPH_CAP=N` caps the per-backend-context CUDA/HIP graph cache at N entries: on a cache miss that
would exceed N, the least-recently-used graph (and its executable instance) is destroyed after a device
synchronise; never while the stream is capturing. Default 0 keeps upstream behaviour (10 s idle sweep only).
`BIGCHERRY_GRAPH_MEMLOG=1` logs the device free-memory delta of every graph instantiation (through 1302's
helper), so the VRAM held by cached instances is measured. Motivation: 193 live graph instances on the R9700 at
164K context (QFP06). Activation evidence: `BIGCHERRY_PATCH_HIT patch=1304_graph_lru` (logged at 1, 2, 4, 8, ...
evictions) and `patch=1304_graph_memlog`.

## Hardware result (2026-10-04 review)

Disproven (QFP06): a cap below the ~195-instance live split-graph working set causes recapture churn; cap-based graph-memory control rejected as a performance strategy.
