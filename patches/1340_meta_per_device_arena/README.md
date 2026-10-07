# 1340 — Meta per-device compute arena

Purpose: remove the tensor-split Meta backend's common compute-arena reservation when devices own different transformed graphs.

## Runtime contract

Enable with `BIGCHERRY_META_PER_DEVICE_ARENA=1`. Default is off.

At llama.cpp b11402 (`d89651a7b205`):

- `ggml_backend_sched_reserve()` owns the worst-case scheduler measure graph. 1340 calls a Meta reserve-only hook there, after logical split/reserve and before scheduler reset.
- `ggml_backend_meta_reserve_graph()` maps that same graph to every simple backend. Each simple backend owns one `arena_galloc` physical buffer. Per-shape `arena_plan_t` gallocr objects are bufferless allocation plans.
- `ggml_gallocr_reserve_grow()` preserves the maximum size reached by each physical vbuffer chunk across reserve shapes; it is not called in the normal compute path.
- `ggml_backend_meta_alloc_graph()` maps the current graph, validates the matching reserve-time plan, and binds it into the shared physical owner with `ggml_gallocr_alloc_graph_reuse_from()`.
- A post-load non-fit is an invariant violation, not an ordinary growth policy. It logs `BIGCHERRY_META_MEM arena_nonfit ...`, increments `arena_replans`, performs one fallback re-plan/grow, then binds.

The design is device-count and topology agnostic: no GPU index, model name, attention split, or backend name is encoded in the arena logic.

## Reporting

With `BIGCHERRY_META_MEM=1`:

```
BIGCHERRY_META_MEM arena dev=<n> buft=<name> reserved_mib=<MiB> plans=<count> replans=<count>
```

Expected normal result after load: `replans=0`; `reserved_mib` does not grow with context fill.

## Validation

Before every push:

```bash
PYTHONPATH=tools python -m unittest tools.tests.patch.test_1340_meta_per_device_arena tools.tests.patch.test_1341_meta_subset_mirrored
PYTHONPATH=tools python -m bigcherry patch-rebase-check --source bigcherry --experiment meta-memory
```

The offline test reads b11402 from the vendor Git object (`git show`), never from its patched working tree.
