# 1340_meta_per_device_arena

**Status:** untested
**Plan item:** MSM02

Kind: enhancement, flag `BIGCHERRY_META_PER_DEVICE_ARENA` (default 0).

When enabled, Meta compute buffers defer their per-simple-device storage and one ggml_gallocr per simple backend allocates the transformed device graph. Static weight/KV buffers and the flag-off common-arena path are unchanged. With `BIGCHERRY_META_MEM=1`, each physical arena reports as `BIGCHERRY_META_MEM arena dev=...`.

## Evidence

- Offline mechanics test: `tools/tests/patch/test_1340_meta_per_device_arena.py`.
- Compile, runtime fidelity and hardware memory measurements: pending.
