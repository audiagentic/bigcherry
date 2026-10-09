# 1302_cuda_graph_oom_evict

**Status:** validated
**Plan item:** QFN03

## What it does

Graph instantiation (`cudaGraphInstantiate` / `hipGraphInstantiate`) that fails with out-of-memory no longer
aborts the server. The backend context destroys every other cached graph (each holds an executable instance in
device memory), synchronises, clears the error and instantiates again; only a second failure aborts. Found at
Flash-Next 192K: a ~164K-token fill OOMs in hipGraphInstantiate on the R9700 after KV and compute buffers fill
it (production build, flashnext-gather-ab-2/d131072). Evicted graphs are re-captured on their next use.
Activation evidence: `BIGCHERRY_PATCH_HIT patch=1302_graph_oom_evict` with the evicted count.

## Hardware result (2026-10-04 review)

Part of production profile v2 (2026-10-03/04): needed to instantiate HIP graphs at 240K f16 KV without OOM; greedy identical in every v2/v3 screen.
