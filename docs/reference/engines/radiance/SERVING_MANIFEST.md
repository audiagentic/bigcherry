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

With `--manifest` the engine accepts only `--role engine`, `--plan-memory FILE`, `-v`, `-vv` and the quality-run flags `--kld-record`, `--kld-corpus`, `--kld-ref` and `--kld-out` (a KL run scores the model's own rows and never serves, so the serving plan check is not armed for it); any other flag is an
error naming the flag. `--radiance-home` and `--kernels` must be in the manifest (the environment variables never fill
them in). A serving start checks the live memory plan against the plan the manifest carries before any weights load; there
is no switch to skip it.

The engine also applies the manifest's hand-off settings to its environment at start, before it initialises, so the manifest is the only place they are configured and they win over a value the launcher left behind: `R3_MOE_WORKER` (set when the manifest has helpers), `R3_MOE_WORKER_SHM` (`spec.handoff.transport.shm`), `R3_MOE_WORKER_MAX_ROWS` (`transport.max_rows`), `R3_MOE_WORKER_SKIP_OWNED` (`transport.skip_owned`), `R3_MOE_WORKER_HISTOGRAM` and `R3_MOE_WORKER_HISTOGRAM_PERIOD_S` (`telemetry.histogram_file`, `telemetry.period_s`). Kernel switches that were environment variables are manifest fields under `spec.processes.engine.kernels` (contract amendment 14): `attn_native` (`R3_ATTN_NATIVE`), `moe_native`, `moe_fp16`, `moe_ahalf` (`R3_MOE_*` in the engine process), `gfx12_moe_gemv` (`BC_GFX12_MOE_GEMV`), `all_reduce_wire4` (`R3_AR_WIRE4`), `dec11` (`R3_DEC11`), `gemv` (`R3_GEMV`) and `fuse_stage` (`R3_MOE_WORKER_FUSESTAGE`, amendment 16). `R3_MOE_WORKER_COMBINE` is set from `helper_common.combine.enabled` so the engine agrees with the helpers. Each is a resolver key (`kernels.dec11 = true`), and rad0009 sets the variable from the field before the engine initialises (generated from the registry's `env_setting`), so a value in the launcher's environment loses to the manifest and an absent field leaves the kernel's own default. The lab-only patch and diagnostic-build switches are optional members of `spec.experiment.engine` (amendment 17): `mtp_full_head` (`RAD_MTP_FULL_HEAD`), `hc_pf2_fp8_down` (`RAD_HC_PF2_FP8_DOWN`), `routing_evidence` (`RAD_ROUTING_EVIDENCE`), `freeze_promotion_after` (`RAD_DIAG_FREEZE_PROMOTION_AFTER`), `shadow` (`R3_MOE_WORKER_SHADOW`) and `mask_only` (`R3_MOE_WORKER_MASK_ONLY`), applied the same way and refused by the `production` profile.

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

## Engine options

Every engine option that is configuration has a manifest field (contract amendment 13; the full list is `docs/design/serving-manifest-v1/coverage-plan.md`
and the generated field reference). Beyond the capacity and placement keys above, the engine options sit in seven optional sections under
`spec.processes.engine`, set through `bc-serve --set section.name=value` or the same key in a profile TOML:

| Section | What | Examples |
|---|---|---|
| `sampling` (20) | default sampling for requests that set none | `sampling.temp`, `sampling.top_k`, `sampling.dry_multiplier` |
| `limits` (13) | HTTP identity, transport and request limits | `limits.host`, `limits.max_body_mib`, `limits.read_timeout`, `limits.no_cors` |
| `chat` (6) | chat template and reasoning | `chat.reasoning_format`, `chat.thinking_token_budget` |
| `lora` (5) | adapters directory and limits, control vector | `lora.lora_dir`, `lora.max_loras` |
| `vision` (12) | image and video input limits and placement | `vision.image_max_pixels`, `vision.video_fps` |
| `tuning` (22) | prefill, tensor-parallel, expert, draft and cache tuning | `tuning.prefill_span`, `tuning.p2p`, `tuning.draft_window` |
| `diagnostics` (2) | debug switches that change what the engine does | `diagnostics.debug_placement` |

Integers are integers; a decimal or free-text option is a string carrying the flag's value text (`sampling.temp="0.7"`); a flag with no value
is the string `"on"`. The engine checks ranges and value text itself, as for every manifest flag. An unset key leaves the engine's own default.
Still flags beside a manifest, because they are a kind of run: `--kld-*`, `--download-dir`, `--debug-graph`, `--lora` (repeatable), `--role`,
`--plan-memory`, `-v`, `-vv`.

## Campaign arms

A campaign arm that serves from a manifest describes its configuration under `serving`; the harness makes the manifest.

```json
"serving": {"layout": "flashnext-2xtx-combine",
            "set": {"service.max_sessions": 6, "service.max_context": 32768, "memory.host_pool_mib": 22528},
            "heat": "${BC_WORK_ROOT}/heat/flashnext-mixed.hot-first", "histogram": "/dev/null"}
```

`layout` names an entry of `engines/radiance/layouts.toml` (required); `set` holds `bc-serve --set key=value` settings (strings, numbers,
booleans); `heat` and `histogram` are the layout's route-heat inputs and are required exactly when the layout has helpers (`/dev/null`
for no recording). `calibration` (a calibration store, from which bc-serve derives the live policy), `live_policy` and `experiment` (files)
and `calibration_run` (true) are the matching `bc-serve --calibration`, `--live-policy`, `--experiment` and `--calibration-run`; they reach
every bc-serve call of the arm and the experiment file's contents are part of the plan-cache key. The model, the runtime home and the engine port come from the job and the build; environment variables in the
block are expanded. An arm with `serving` takes no `server_args` (they are settings) and needs `execution_evidence: "observe"` (a
manifest start has no attestation preflight).

Before the first cell the harness asks the arm's build for the measured manifest of that configuration (`bc-serve --write-manifest`,
plan cache `<work-root>/plan-cache`). On a cache miss it measures first, as the launcher does: the helpers come up on their cards
(`--measuring`), the engine runs to the end of its memory plan (`--plan-memory`) and exits, and `bc-serve ... --measured-plan` writes
the measured manifest and stores the plan. That measuring start uses the GPUs, so a campaign with a new configuration runs under the
`gpu` lease like any measurement. The manifests are kept in the output directory under `manifests/<arm>/`.

The engine is then started as `radiance --manifest M --role engine` (`[serve] manifest-args` in `engines/radiance/engine.toml`), the port
to drive comes from the manifest, and one `r3_moe_worker --manifest M --role helperN` starts per `spec.processes.helpers` entry, each
seeing the host card whose `locator` equals the helper's `gpu`. The engine still reads `R3_MOE_WORKER=1` from its environment to
enable the helper hook (RSC08 S5: not yet a manifest field), so a job with helpers sets it in `environment`.

A quality (KL) arm uses the same block. The harness starts the engine KL run as `radiance --manifest M --role engine --kld-ref DIR --kld-out FILE`
(or `--kld-record DIR --kld-corpus FILE` for the reference), and `quality.args` must be empty for such an arm. **The engine accepts those four
flags beside a manifest only once rad0009 is changed to allow them (a separate change that needs the patch requalified on hardware); until then
it refuses them, so a quality arm with a serving block cannot run.** A reference recorded in the same job
(`quality.record`) still starts from raw flags; a stored reference is reused as before.

`R3_MOE_WORKER_*` arm settings, `R3_MOE_WORKER_LAYOUT` and `BC_SERVING_MANIFEST` are gone. Jobs that used them and could not be migrated
are archived in `experiments/radiance/archive-helper-env-jobs/` and `experiments/radiance/archive-engine-only-jobs/`. The lab scripts
that start a helper by hand (`fn-helper.sh`, `w4-replay-hash.sh`, `replay-kernel-time.sh`, `helper-gemm-counters.sh`,
`helper-long-rows.sh`) take the manifest as an argument.

## Migration status

| Step | State |
|---|---|
| Engine and helpers read a manifest; the engine admits the plan it carries | done (RSC06, RSC08) |
| `--set` for `tp_wire`, `no_prefix_cache`, `gpu_headroom_mib` | done (RSC11 A1) |
| Resolver reads `layouts.toml` and `archs.toml`; engine-only and multi-card-engine layouts; kernels derived per architecture; contract amendment 12 | done (RSC11 A2) |
| Dense models (containers without routed experts) get engine-only manifests | done (RSC11 A2b) |
| Campaign arms describe a `serving` configuration; the harness makes each arm's measured manifest; `BC_SERVING_MANIFEST` and the helper environment settings removed | done in code (RSC11 A3/B); the measuring path is not yet run on hardware |
| Lab job files and `layouts.toml` side channels moved or retired | pending (B) |
| Every engine option that is configuration has a manifest field (80 more, contract amendment 13) | done (RSC12 slice 1) |
| The engine reads its environment settings from the manifest | pending (RSC12 slice 2) |
| The engine refuses direct config flags | pending (C) |
