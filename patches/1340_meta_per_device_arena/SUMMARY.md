# 1340_meta_per_device_arena

**Status:** validated  
**Plan item:** MSM02

Kind: enhancement, on by default; `BIGCHERRY_META_PER_DEVICE_ARENA=0` restores the common-size arena.

With the flag enabled, the Meta tensor-split compute buffer no longer gives every simple device the scheduler's common physical arena size. `ggml_backend_sched_reserve()` materialises the scheduler's worst-case measure graph once, then 1340 translates that graph to each simple backend. Each device owns one grow-only physical gallocr arena; graph-shape gallocr plans keep allocation metadata only and bind into that shared arena during compute.

Compute-time growth of the physical arena is not a normal path: it emits `GGML_LOG_ERROR` (`arena_grew`) and is counted. A new layout inside the reserved arena is normal and quiet (`arena_replan` under `BIGCHERRY_META_MEM=1`).

With `BIGCHERRY_META_MEM=1`, load/reserve reports:

`BIGCHERRY_META_MEM arena dev=<n> buft=<name> reserved_mib=<MiB> plans=<count> replans=<count>`

Static weight/KV allocations are unchanged. With the flag off, the upstream common-size Meta compute-buffer path is unchanged. Patch 1341 can additionally make subset-inactive mirrored inputs zero-sized before this reserve translation, so devices reserve only for tensors they actually own.

## Offline evidence

- `tools/tests/patch/test_1340_meta_per_device_arena.py` applies against pinned b11402 source via `git show`, composes the active Meta/Qwen patch stack, checks idempotence/fail-closed anchors, verifies guards only match post-edit output, verifies reserve-only growth/shared-owner binding, and locks the `[experiment.meta-memory]` selection.
- `tools/tests/patch/test_1341_meta_subset_mirrored.py` is run in the same gate.
- Required pre-push gate: `PYTHONPATH=tools python -m bigcherry patch-rebase-check --source bigcherry --experiment meta-memory` must report zero failures.

## Hardware status

Pending owner build/runtime validation at ctx 245760 / ub512. Acceptance signal: final per-device `reserved_mib` is established during load and stays fixed through fill, `replans=0`, and output/fusion behaviour matches production.
