# 1340_meta_per_device_arena

**Status:** untested
**Plan item:** MSM02

Kind: enhancement, flag `BIGCHERRY_META_PER_DEVICE_ARENA` (default 0).

When enabled, Meta compute buffers defer their per-simple-device storage and one ggml_gallocr per simple backend allocates the transformed device graph. Static weight/KV buffers and the flag-off common-arena path are unchanged. With `BIGCHERRY_META_MEM=1`, each physical arena reports as `BIGCHERRY_META_MEM arena dev=...`.

## Evidence

- Offline mechanics test: `tools/tests/patch/test_1340_meta_per_device_arena.py` (includes the b11402 zero-head `FLASH_ATTN_EXT` allocator-size regression).
- Pre-fix hardware proved dev0/dev1 arenas shrink (285.3 -> 267.3 MiB at ctx 49152; 1020.9 -> 880.8 MiB at ctx 245760) but exposed a dev2 zero-slice load crash.
- Root cause fixed: zero-sized disabled transformed tensors are kept external to the simple gallocr, so backend alloc-size hooks are not called on zero-head attention ops.
- Final-fix compile, runtime fidelity and hardware memory rerun: pending.
