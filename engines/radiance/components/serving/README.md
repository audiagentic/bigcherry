# serving (RSC01 slice 1)

CPU-only static library `bc_serving` and `bc-serve`, packaged at
`radiance_home/bin/bc-serve`. One C++ schema and strict TOML loader;
no Python schema, shell launcher parser, or process starting.

Build with `-DCMAKE_PREFIX_PATH=<installed radiance>` and
`-DRADIANCE_SRC=<radiance source>`; build `bc-serve bc_serving_test`
then run `ctest --output-on-failure`. Tests are CPU-only, including a
synthetic KFD/PCI/proc/sysfs tree and launcher-value parity.

Commands: `--detect [--json]`, `--schema`, `--validate <site> [profile]`,
`--explain [--json]`, `--dry-run --out <manifest.json>`.
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

The manifest uses sorted-key nlohmann JSON. Its `content_sha256` is
SHA256 of the canonical serialized manifest before the hash field is
added; the hash covers values, provenance, facts and process_helpers.
RSC06/RSC07 will wire the manifest into each process, replacing
environment variables and individual engine flags. RSC02 will add
supervision. This slice does **not** start processes.

Vendor: toml++ v3.4.0 root single header and MIT license (`vendor/`);
reuse radiance's existing nlohmann/json v3.12.0.
