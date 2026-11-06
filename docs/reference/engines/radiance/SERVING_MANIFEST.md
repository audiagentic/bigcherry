# Serving manifest: writing one, starting with one

Operator guide for the Radiance serving manifest (`bigcherry/v1`). The contract itself is
[the v1 spec](../../../design/serving-manifest-v1/spec.md); every field and its type is in the generated
[field table](../../../../engines/radiance/components/serving/generated/serving-fields.md). This page is how to use them.

**One configuration method.** A manifest is the whole configuration of an engine or helper start: what to serve, how, and
(for a serving start) the memory plan it must admit. Direct engine flags (`--tp`, `--kernels`, `--no-prefix-cache`, ...)
are being retired in favour of it (RSC11). Status: the launcher and the helpers already start from a manifest; the
campaign adapter, the lab job files and the engine's acceptance of direct flags move over in the steps listed under
[Migration status](#migration-status). New work uses a manifest and never adds a flag.

## Who does what

| Tool | Role |
|---|---|
| `bc-serve` | Resolves a configuration (host facts, site file, `--set`, layout) and writes the manifest. Starts no process. |
| `radiance --manifest M --role engine` | The engine reads its whole configuration from `M`. |
| `r3_moe_worker --manifest M --role helperN` | A helper reads its role from the same `M`. |
| `tools/lab/radiance/launch-flashnext-best.sh` | The lab launcher: asks `bc-serve` for the manifest of the configuration it launches, measures the plan if none is cached, starts engine and helpers from the one file. |

## Writing a manifest

```bash
bc-serve --write-manifest OUT --plan-cache DIR --engine-binary PATH \
  --site SITE.toml --set layout.name='"flashnext-2xtx"' --set service.max_sessions=6 ...
```

- `OUT` must not exist: a manifest is immutable, and an existing file is refused.
- The result is read back through the same strict reader the engine uses before the file stays.
- A serving manifest carries a **measured plan**. `bc-serve` takes it from the plan cache for this exact configuration; a
  miss exits **10**. To measure: write the unmeasured manifest, start the engine once with `--plan-memory FILE` (helpers
  up), then write the measured one from that plan:

  ```bash
  bc-serve --write-manifest M0 --unmeasured --site SITE.toml --set ...          # measuring start's manifest
  radiance --manifest M0 --role engine --plan-memory PLAN.json                   # writes the plan and exits
  bc-serve --write-manifest M --plan-cache DIR --engine-binary PATH --site SITE.toml --set ... \
    --measured-plan PLAN.json --helpers-resident-mib H --engine-resident-mib E
  ```

  Run the measuring start under the `gpu` lease. An unmeasured manifest is accepted only by a measuring start.
- Lab-only: `--experiment FILE` (a JSON object for `spec.experiment`: `profile_ops`, `kv_resident`, `debug_selection`,
  `deterministic`, placement experiments) writes the manifest with profile `lab`; `--live-policy FILE` sets the helper's
  live placement policy. A production manifest carries neither.

Inspect without writing: `bc-serve --explain [--json] --set ...` shows every resolved value with its origin
(`schema-default`, `site`, `profile`, `cli`, `derived`, `discovered`, `calibrated`).

## Layouts and engine-only manifests

A layout (`engines/radiance/layouts.toml`, card types in `engines/radiance/archs.toml`; both compiled into `bc-serve`, so there is
one copy) names which cards run the engine and which hold experts. `bc-serve` selects the cards by architecture from the host's
detected GPUs; it never takes device indices.

| Layout shape | Manifest |
|---|---|
| engine card + helper groups (the flashnext layouts) | `processes.helpers` per helper, `handoff`, `helper_common`, a planned owner map |
| engine cards only (`qwen27b-2xtx`: two RDNA3 ranks, `tp-wire = "wht6"`) | `processes.helpers` and the owner map empty; **no** `handoff` and **no** `helper_common` (contract amendment 12) |

For an engine-only layout `engine.tp` is the card count, `engine.kernels` is the engine architecture's plugin plus `libref` (the
mailbox plugin `libr3moe` only when helpers exist), `handoff.heat` and `handoff.histogram` are not needed, and the host-memory
measurement is the engine's alone (`--helpers-resident-mib 0`). A manifest with a `handoff` or `helper_common` and no helpers, or
helpers and no `handoff`, is refused by the reader.

An engine-only manifest never reads the model's expert tables, so a dense model (a container without routed experts, such as the
27B) gets one like any other; only a layout with helpers needs the experts, to plan which ones each helper owns.

## Starting with a manifest

```bash
radiance --manifest M --role engine            # serving: the manifest must be measured
radiance --manifest M --role engine --plan-memory PLAN.json   # measuring start (unmeasured manifest)
r3_moe_worker --manifest M --role helper0
```

With `--manifest` the engine accepts only `--role engine`, `--plan-memory FILE`, `-v` and `-vv`; any other flag is an
error naming the flag. `--radiance-home` and `--kernels` must be in the manifest (the environment variables never fill
them in). A serving start checks the live memory plan against the plan the manifest carries before any weights load; there
is no switch to skip it.

## Settings: `--set` key, manifest field, engine flag

`--set key=value` (strings are JSON, so `engine.tp_wire='"wht6"'`). A key marked *derived* cannot be set; it follows from
the layout or another key.

| `--set` key | Manifest field | Replaces engine flag |
|---|---|---|
| `engine.model` | `/spec/intent/model` | `--model` |
| `layout.name` | `/spec/intent/layout` | (layout selection) |
| `service.max_sessions` | `/spec/intent/service/max_sessions` | `--max-num-seqs` |
| `service.max_context` | `/spec/intent/service/max_context_tokens` | `--max-model-len` |
| `service.objective`, `service.retention`, `service.quality` | `/spec/intent/service/...` | policy, no flag |
| `engine.radiance_home` | `/spec/processes/engine/runtime/radiance_home` | `--radiance-home` |
| `engine.kernels` (*derived*) | `/spec/processes/engine/runtime/kernels` | `--kernels` |
| `engine.port` | `/spec/processes/engine/server/port` | `--port` |
| `engine.tp` (*derived*) | `/spec/processes/engine/scheduler/tp` | `--tp` (the layout's engine card count) |
| `engine.max_num_batched_tokens` | `/spec/processes/engine/scheduler/max_num_batched_tokens` | `--max-num-batched-tokens` |
| `engine.checkpoint_interval` | `/spec/processes/engine/scheduler/checkpoint_interval` | `--checkpoint-interval` |
| `engine.placement` | `/spec/processes/engine/execution/placement` | `--placement` |
| `engine.ngram_placement` | `/spec/processes/engine/execution/ngram_placement` | `--ngram-placement` |
| `engine.kv_cache_dtype` | `/spec/processes/engine/execution/kv_cache_dtype` | `--kv-cache-dtype` |
| `engine.num_speculative_tokens` | `/spec/processes/engine/execution/num_speculative_tokens` | `--num-speculative-tokens` |
| `engine.rope_scaling` | `/spec/processes/engine/execution/rope_scaling` | `--rope-scaling` |
| `engine.prefix_resume` | `/spec/processes/engine/execution/prefix_resume` | `--prefix-resume` |
| `engine.tp_wire` | `/spec/processes/engine/execution/tp_wire` | `--tp-wire` (`exact`, `wht6`) |
| `engine.no_prefix_cache` | `/spec/processes/engine/execution/no_prefix_cache` | `--no-prefix-cache` (value `"on"`) |
| `engine.gpu_headroom_mib` | `/spec/processes/engine/device/headroom_mib` | `--gpu-headroom-mib` |
| `memory.host_pool_mib` | `/spec/resources/host/pool_mib` | `--host-pool-mib` |
| `memory.host_kv_mib` | `/spec/resources/host/kv_mib` | `--prefix-cache-mib` |
| `memory.host_reserve_mib` | `/spec/resources/host/reserve_mib` | `--host-headroom-mib` |
| `handoff.*` (`shm`, `max_rows`, `small_rows`, `gemv_large_rows`, `skip_owned`, `moe_down_tile`, `histogram`, `histogram_period_s`, `placement`, `heat`) | `/spec/handoff/...` | helper-side, no engine flag |
| `helper.combine`, `helper.combine_min_rows`, `helper.combine_threads` | `/spec/processes/helper_common/combine/...` | helper-side, no engine flag |

`engine.tp_wire` follows the layout when it names one (`tp-wire` in `layouts.toml`; required for two or more engine cards) and can be overridden; `engine.no_prefix_cache` and `engine.gpu_headroom_mib` have no default: an unset one is absent from the
manifest and radiance keeps its own default. The four experiment flags have no `--set` key; they are fields of
`spec.experiment` and come in through `--experiment`:

| `spec.experiment.engine` field | Engine flag |
|---|---|
| `profile_ops` | `--profile-ops` |
| `kv_resident` | `--kv-resident` |
| `debug_selection` | `--debug-selection` |
| `deterministic` | `--deterministic` |

## Errors

A manifest that fails the strict reader is refused with a stable error code and the offending field
(`engines/radiance/components/serving` `errors.md`; the spec's error section). The launcher returns **96** for a failed
start canary, **97** for plan measurement/storage/setup failure, **98** for a held launch lock or an already running
server.

## Migration status

| Step | State |
|---|---|
| Engine and helpers read a manifest; the engine admits the plan it carries | done (RSC06, RSC08) |
| `--set` for `tp_wire`, `no_prefix_cache`, `gpu_headroom_mib` | done (RSC11 A1) |
| Resolver reads `layouts.toml` and `archs.toml`; engine-only and multi-card-engine layouts; kernels derived per architecture; contract amendment 12 | done (RSC11 A2) |
| Dense models (containers without routed experts) get engine-only manifests | done (RSC11 A2b) |
| Campaign arms get a manifest each; `BC_SERVING_MANIFEST` removed | pending (A3) |
| Lab job files and `layouts.toml` side channels moved or retired | pending (B) |
| The engine refuses direct config flags | pending (C) |
