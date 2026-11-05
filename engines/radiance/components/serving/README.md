# serving

## Summary

The `bc_serving` C++ library and `bc-serve` tool provide a strict typed
TOML loader, hardware detection, configuration validation and a canonical
serving manifest with provenance.

## Outcome

The strict TOML loader, `bc-serve` inspection and manifest generation,
the shared v1 reader, and the engine-side memory-plan measurement and
admission check are implemented. The serving component now includes an
owner-map verifier that checks the planner's owned expert sets and exact
device byte totals. Card-free tests cover the verifier. The helpers read their
whole configuration from the manifest (`r3_moe_worker --manifest FILE --role
helperN`, through `v1/helper_settings.h`; RSC06) and the launcher writes the
manifests they start from. The engine still starts from flags until the S4
cutover, so this is not yet a fully qualified v1 launch.


## Detail

CPU-only static library `bc_serving` and `bc-serve`, packaged at
`radiance_home/bin/bc-serve`. One C++ schema and strict TOML loader;
no Python schema, shell launcher parser, or process starting.

Build with `-DCMAKE_PREFIX_PATH=<installed radiance>` and
`-DRADIANCE_SRC=<radiance source>`; build `bc-serve bc_serving_test`
then run `ctest --output-on-failure`. Tests are CPU-only, including a
synthetic KFD/PCI/proc/sysfs tree and launcher-value parity.

Commands: `--detect [--json]`, `--schema`, `--validate <site> [profile]`,
`--explain [--json]`, `--write-manifest OUT` (the serving manifest v1; with `--plan-cache` and `--engine-binary`,
or with `--unmeasured` and neither: the measuring start's manifest, `plan.state` unmeasured, which only a measuring start
(`radiance --plan-memory`, `r3_moe_worker --measuring`) accepts; `--live-policy FILE` adds the live placement policy and
`--experiment FILE` a lab experiment, each a JSON object the reader checks member by member; `--measured-plan FILE
--helpers-resident-mib N --engine-resident-mib M` takes the plan a measuring start produced and the host memory the helpers and
the engine held in it, checks the plan against the configuration, stores both in the plan cache under this configuration's key
and writes the measured manifest from it; the host ledger counts the host pool and those two figures against MemAvailable as
read when the manifest is written, so the checkpoint slots follow what is really free at start, not at measurement; a cache
entry without the figures is a miss). The engine admits the plan the manifest carries; no other
command or flag hands it one (RSC08 S3 removed `--plan-key`, `--plan-check-args` and `--plan-store`).
Use `--site FILE`, `--profile FILE`, and repeated
`--set section.key=value` with the latter three.
`--sys-root DIR` overrides the filesystem root for fake-tree tests.
Site-only `[hard_caps.memory]` defines maximum MiB; caps may only
tighten the discovered/default baseline. Neither profiles nor CLI may
modify hard caps. Unknown keys fail closed with their dotted path,
source file and closest valid key.

Schema:
- `service`: objective, max_sessions, max_context, retention, quality
- `layout`: name, engine_card (PCI BDF), helper_cards (PCI BDFs)
- `memory`: host_pool_mib, host_kv_mib, tail_budget_mib,
  helper_budgets_mib (in helper order)
- `handoff`: shm, max_rows, small_rows, gemv_large_rows, moe_down_tile,
  skip_owned, heat, histogram, histogram_period_s, placement; these are
  the launcher's COMMON mailbox settings, not WorkerConfig fields
- `helper`: **exactly** the ten WorkerConfig fields: count, index,
  engine_ranks, cpu, power_save, queue_warm, combine, combine_min_rows,
  combine_threads, cpu_set_auto
- `engine`: model, radiance_home, port, kernels, placement,
  ngram_placement, host_pool_mib, prefix_cache_mib, tp, max_num_seqs,
  max_num_batched_tokens, checkpoint_interval, kv_cache_dtype,
  num_speculative_tokens, max_model_len, rope_scaling, prefix_resume

WorkerConfig's index is expanded per helper in `process_helpers`.
The `engine.model`, `engine.radiance_home`, `handoff.heat`, and
`handoff.histogram` fields have no portable defaults and are required.
Use `handoff.histogram = "off"` to disable recording. Values for heat
and histogram should be supplied as paths from the site/profile.
For example, for a read-only explanation:

```
bc-serve --explain --set engine.model=/models/model.rad \
  --set engine.radiance_home=/runtime \
  --set handoff.heat=/data/heat.hot-first \
  --set handoff.histogram=/data/heat.live
```

The JSON Schema `x-origin` annotations document every default source.
Interim memory baselines and named layout GPU budget baselines come from
the 2026-11-02 launcher and `engines/radiance/layouts.toml`; RSC03
will derive them from detected resources. A first-slice snapshot of the
named layouts is used to select cards from live KFD discovery rather
than reading any per-host configured GPU index list. Unsupported layouts
and unavailable GPUs fail closed; layout/catalog generalization is later
RSC05 work.

The one document processes load is the serving manifest v1 (`--write-manifest`; spec in
`docs/design/serving-manifest-v1/`), read by the shared strict reader. The resolver's own record (`--explain --json`,
the hash `--validate` prints) is not a file any process reads. The engine starts from the v1 manifest (rad0009, shared
`bc_serving_v1` library); helper-side manifest consumption and process supervision are still pending.
This component does **not** start processes.

Vendor: toml++ v3.4.0 root single header and MIT license (`third_party/`; the repository ignores every `vendor/` directory);
reuse radiance's existing nlohmann/json v3.12.0.
