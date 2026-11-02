# 1304_cuda_graph_lru_cap

## Summary

`BIGCHERRY_CUDA_GRAPH_CAP=N` caps the per-backend-context CUDA/HIP graph cache at N entries: on a cache miss that would exceed N, the least-recently-used graph (and its executable instance) is destroyed after a device synchronise; never while the stream is capturing. Default 0 keeps upstream behaviour (10 s idle sweep only).

## Outcome

Recorded state: `rejected` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
