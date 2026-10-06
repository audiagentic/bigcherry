# 1339_meta_memory_report

**Status:** untested
**Plan item:** MSM01

Kind: diagnostic, flag `BIGCHERRY_META_MEM` (default 0).

Prints the real size of every tensor-split (meta) buffer on each device: the scheduler's compute arena, which is
allocated at the same size on every device, and the static buffers (weights, KV, indexer state) with the size of
each device's slices and the first tensor's name. The server otherwise logs one size per Meta buffer, which hides
what each card holds. Nothing is allocated differently.

## Evidence

- Offline mechanics test: `tools/tests/patch/test_1339_meta_memory_report.py`. Hardware: pending.
