# Radiance serving configs: one native YAML per deployment with an `x-bigcherry` section

> **Serving/configuration closure sweep (2026-11-25):** [Field ownership, measured caches and caller migration](radiance-serve-config-review-20261125.md) specifies the remaining S18 work against the current implementation. In particular, native server.port remains authored, run capture is evidence-only, and actual resolved power settings scope performance calibration.

> **Launch-transport decision superseded (2026-11-25):** [Native YAML launch and manifest retirement](radiance-native-yaml-launch.md) retains the one-file authoring and native-parser rules below, but replaces manifest-based execution with native --config, owner-only typed admission and evidence-only run capture. This is the target design, not a claim that the current binary already implements it.

Status: design for RUM04 (Radiance 2.2 bump), 2026-11-20. Owner decision: the bump carries every serving config, and
a config is one seamless file. Supersedes the "narrow x-bigcherry selector only" step in RUM04 and the interim layout
(native YAML beside slimmed RSC12 TOML) committed as d99323272..c04c69128 on node/radiance-latest-bump. Governing
reviews: [ASTRA-CONSOLIDATION](../reviews/radiance-upstream-2026-11-18/ASTRA-CONSOLIDATION.md) (Radiance owns engine
options; BigCherry owns cross-process settings, derived decisions and proofs) and
[MANIFEST-AND-EXTENSIONS](../reviews/radiance-upstream-2026-11-18/MANIFEST-AND-EXTENSIONS.md) (closed, versioned
`x-bigcherry` namespace). RSC12/RV4225 already gave RUM04 "upstream 2.x native YAML plus typed x-bigcherry intent".

## Decision

1. A serving config is one file, `engines/radiance/configs/<name>.yaml`. The file name is the config name
   (`bc-serve start <name>`). The RSC12 TOML catalogue (`flashnext.toml`, `qwen27b.toml`), the layout files
   (`engines/radiance/layouts/*.toml`) and their readers are deleted in the same change.
2. Everything Radiance has a native option for is authored in Radiance's own sections (`server`, `cards`, `memory`,
   `scheduler`, `defaults`, `models.<id>`), with upstream meanings, parsed only by Radiance's parser.
3. Everything BigCherry owns and Radiance does not (rank architectures, helper groups, handoff/helper process
   settings, our kernel switches, service policy, memory-plan overrides, variant provenance) is authored in one
   top-level `x-bigcherry` section. Radiance's parser skips `x-*` top-level keys, so a stock Radiance binary can run
   the same file natively (without helpers).
4. Nothing is authored twice. A value Radiance owns never appears in `x-bigcherry`; where BigCherry needs it
   (TP, model, cards, kernels) it reads Radiance's resolved value from the inspector and checks it against its own
   intent (rank count, rank archs), refusing with both sources named.
5. Machine facts and derived values are never authored: runtime home, BDFs, host pool / KV tier / checkpoint
   slots (the memory plan), helper indices. bc-serve supplies them into the sealed launch.

## File shape

```yaml
cards: [0, 1, 2]                    # Radiance: rank visibility order; managed start requires [0..tp-1]
server:
  host: 127.0.0.1                   # managed start: loopback only
  kernels: libr3,libr4d,libref      # must cover every rank arch below
models:
  qwen27b:                          # exactly one model for a managed start
    model: radiance/StillDeadcode/qwen3.8-27b-iq8r-dflash/qwen3.8-27b-iq8r-dflash.rad
    tp: 3
    tp-wire: exact
    max-model-len: 262144
    max-num-seqs: 8
    kv-cache-dtype: fp8
    num-speculative-tokens: 3
    ngram-placement: disk
    rope-scaling: none
    p2p: "off"
    tp-ownership: "kv=1,1,2;ffn=6656,6656,4096;lk=6,6,4;lv=18,18,12"
    pipelined-prefill: "on"
    pipelined-prefill-min-rows: 256
    tune-pin: "all_reduce:nblocks=4,nthreads=256,ts_pieces=4,ts_fuse=1,ts_once=1,ts_duplex=1"
x-bigcherry:
  schema: 1
  description: "Qwen3.8-27B IQ8R over XTX, XTX, R9700 (native TP3), 6,6,4 split, 262,144 tokens a session"
  variant-of: qwen27b-tp3-q8      # optional provenance, see Variants
  differs: [models.qwen27b.max-model-len]
  ranks: [{arch: gfx1100, count: 2}, {arch: gfx1201, count: 1}]
  memory: {host-pool-mib: 0, host-kv-mib: 0}   # explicit overrides of derived values; normally omitted
```

A helper config adds `helpers` (ordered groups: `arch`, `count` number or `all`, `placement` head|tail, optional
`budget-mib`, `share`) and the `handoff`/`helper` sub-tables. `server.port` is a native authored option, preserved by bc-serve (closure review); lab arms author a distinct endpoint if required.

## The `x-bigcherry` schema (v1)

Closed and typed: unknown keys, wrong types, duplicate keys, an unsupported `schema`, a `differs` path that does not
exist, and a model id that is not the file's one model are refusals naming the file and the YAML path.
Top level: `schema` (required, 1), `description` (required), `variant-of`, `differs`, `ranks` (required),
`helpers`, `service` {objective, quality, retention}, `memory` {host-pool-mib, host-kv-mib, host-reserve-mib,
tail-budget-mib, helper-budgets-mib, hard-caps, hard-floors}, `handoff` {heat, histogram, histogram-period-s,
moe-down-tile, skip-owned, gemv-large-rows, small-rows, max-rows, placement, shm}, `helper` {combine,
combine-min-rows, combine-threads, cpu, cpu-set-auto, power-save, queue-warm, count, engine-ranks, index},
`kernels` {all-reduce-wire4, attn-native, dec11, fuse-stage, gemv, gfx12-moe-gemv, moe-ahalf, moe-fp16,
moe-native}. Keys whose class is `derived` or `discovered` in the resolver schema are accepted only as explicit,
documented overrides (as today) and are named in the launch provenance as overrides. The generated schema
(`bc-serve --schema`) is regenerated from the same registry, now describing the YAML paths, not TOML tables.

## Parsing: one reader per half, same bytes

- Native half: only Radiance's parser (`fleet::parse` / `Reader::build_model`) via the engine's
  `--bc-inspect-native-config` report from the matched build, as on the branch today (including the `tuning` report
  added in 6edb46038). bc-serve never parses native keys itself.
- `x-bigcherry` half: Radiance's `fleet::parse` skips top-level `x-*` keys and keeps no tree, so the inspector
  (our component native_fleet_config.cpp, in the matched engine) parses the same sealed bytes a second time with the
  same library Radiance uses (rapidyaml, already linked into the engine) and returns the `x-bigcherry` subtree as
  JSON in its report. bc-serve decodes that JSON against the closed v1 schema. One YAML library, one sealed byte
  sequence, no YAML parser in bc-serve.
- Duplicate keys: Radiance refuses a repeated key only in the maps it walks, and it never walks `x-*`, so the
  inspector checks duplicates inside `x-bigcherry` itself and names the YAML path.
- Scalars follow the YAML 1.2 core schema in the JSON: plain `true`/`false`, integers and floats are typed, quoted
  values stay strings, empty and `null` become null. Words such as `on`/`off` are strings and must be quoted where a
  string is meant (as `p2p: "off"` is today). The closed decode never coerces: a string where a bool or number is
  wanted is a refusal naming the path.
- The v3 launch manifest seals the whole file's bytes and digest (as now); a change in either half changes the plan
  key and the stored-manifest identity. `x-bigcherry` changes are excluded from upstream reload comparison (RUM10)
  because managed hot reload stays refused.

## Precedence and experiments

Site file and profile keep their roles (machine facts, power, calibration); they never carry native engine options.
A managed (served) start refuses any `--set`. A lab arm that changes a setting authors its own YAML file (a copy of
the config with the change), passed with `--native-config`; the file is sealed into the launch like any config, so
the arm's exact inputs are recorded with no override channel. The lab harnesses (`flashnext-parity.sh`
`FLASHNEXT_PARITY_SETS`) migrate to writing such an arm file. A sealed override map applied before parse is not
part of this design; add it later only if a real caller cannot use an arm file. A setting worth keeping becomes its
own config file.

## Variants

Radiance YAML has no cross-file inheritance (anchors and `defaults` are per file), so each deployment is a full file.
To keep "what a variant changes" visible and checked, a variant declares `variant-of: BASE` and `differs: [paths]`.
A test (and `bigcherry configs check`) loads every pair and fails if the resolved native settings or `x-bigcherry`
values differ anywhere outside `differs`, or if a listed path does not actually differ. A fix to a base then fails
the check on every variant until it is applied there too: deliberate, visible, no second source.

## Mapping of every current key

Each RSC12 TOML key and resolver key moves exactly once. File-level TOML keys: `description` -> `x-bigcherry.
description`; `model` -> `models.<id>.model`; `ranks` -> `x-bigcherry.ranks`; `tp_wire` -> `models.<id>.tp-wire`;
`helpers` -> `x-bigcherry.helpers`; `base` -> `x-bigcherry.variant-of` + `differs`. Resolver keys (generated from
`bc-serve --schema`, 153 keys; destinations verified against `radiance --help` of the candidate build):

| resolver key (current TOML) | class | destination |
|---|---|---|
| `chat.chat_template_kwargs` | policy | native `models.<id>.chat-template-kwargs` |
| `chat.override_chat_template` | policy | native `models.<id>.override-chat-template` |
| `chat.reasoning_effort` | policy | native `models.<id>.reasoning-effort` |
| `chat.reasoning_effort_map` | policy | native `models.<id>.reasoning-effort-map` |
| `chat.reasoning_format` | policy | native `models.<id>.reasoning-format` |
| `chat.thinking_token_budget` | policy | native `models.<id>.thinking-token-budget` |
| `diagnostics.debug_accept_reference_kernels` | policy | native `models.<id>.debug-accept-reference-kernels` |
| `diagnostics.debug_placement` | policy | native `models.<id>.debug-placement` |
| `engine.checkpoint_interval` | derived | native `models.<id>.checkpoint-interval` |
| `engine.gpu_headroom_mib` | policy | native `models.<id>.gpu-headroom-mib` |
| `engine.host_headroom_mib` | derived | derived (memory plan); never authored |
| `engine.host_pool_mib` | derived | derived by the memory plan; native `host-pool-mib` is written into the sealed launch, never authored |
| `engine.kernels` | discovered | native `server.kernels`; must cover every rank arch in `x-bigcherry.ranks` (refusal otherwise) |
| `engine.kv_cache_dtype` | policy | native `models.<id>.kv-cache-dtype` |
| `engine.max_model_len` | derived | native `models.<id>.max-model-len` (authored once; replaces `service.max_context`) |
| `engine.max_num_batched_tokens` | derived | native `models.<id>.max-num-batched-tokens` |
| `engine.max_num_seqs` | derived | native `models.<id>.max-num-seqs` (authored once; replaces `service.max_sessions`) |
| `engine.model` | policy | native `models.<id>.model` (only here; x-bigcherry has no model) |
| `engine.ngram_placement` | policy | native `models.<id>.ngram-placement` |
| `engine.no_prefix_cache` | policy | native `models.<id>.no-prefix-cache` |
| `engine.num_speculative_tokens` | derived | native `models.<id>.num-speculative-tokens` (explicit; admission refuses -1) |
| `engine.placement` | policy | native `models.<id>.placement` |
| `engine.port` | policy | machine: `server.port` written by bc-serve from the site/run (never authored in a config file) |
| `engine.prefix_cache_mib` | derived | derived (memory plan); never authored |
| `engine.prefix_resume` | policy | native `models.<id>.prefix-resume` |
| `engine.radiance_home` | discovered | machine: discovered (installed bundle) or `--engine-binary` build home; never in a config file |
| `engine.rope_scaling` | policy | native `models.<id>.rope-scaling` |
| `engine.tp` | derived | native `models.<id>.tp`; must equal the sum of `x-bigcherry.ranks[].count` (refusal names both) |
| `engine.tp_wire` | policy | native `models.<id>.tp-wire` |
| `handoff.gemv_large_rows` | tuned | x-bigcherry `handoff.gemv-large-rows` |
| `handoff.heat` | policy | x-bigcherry `handoff.heat` |
| `handoff.histogram` | policy | x-bigcherry `handoff.histogram` |
| `handoff.histogram_period_s` | policy | x-bigcherry `handoff.histogram-period-s` |
| `handoff.max_rows` | derived | x-bigcherry `handoff.max-rows` |
| `handoff.moe_down_tile` | policy | x-bigcherry `handoff.moe-down-tile` |
| `handoff.placement` | derived | x-bigcherry `handoff.placement` |
| `handoff.shm` | policy | x-bigcherry `handoff.shm` |
| `handoff.skip_owned` | policy | x-bigcherry `handoff.skip-owned` |
| `handoff.small_rows` | tuned | x-bigcherry `handoff.small-rows` |
| `hard_caps.memory` |  | x-bigcherry `memory.hard-caps` (admission policy) |
| `hard_floors.memory` |  | x-bigcherry `memory.hard-floors` (admission policy) |
| `helper.combine` | policy | x-bigcherry `helper.combine` |
| `helper.combine_min_rows` | tuned | x-bigcherry `helper.combine-min-rows` |
| `helper.combine_threads` | policy | x-bigcherry `helper.combine-threads` |
| `helper.count` | discovered | x-bigcherry `helper.count` |
| `helper.cpu` | policy | x-bigcherry `helper.cpu` |
| `helper.cpu_set_auto` | policy | x-bigcherry `helper.cpu-set-auto` |
| `helper.engine_ranks` | derived | x-bigcherry `helper.engine-ranks` |
| `helper.index` | derived | x-bigcherry `helper.index` |
| `helper.power_save` | policy | x-bigcherry `helper.power-save` |
| `helper.queue_warm` | policy | x-bigcherry `helper.queue-warm` |
| `kernels.all_reduce_wire4` | policy | x-bigcherry `kernels.all-reduce-wire4` |
| `kernels.attn_native` | policy | x-bigcherry `kernels.attn-native` |
| `kernels.dec11` | policy | x-bigcherry `kernels.dec11` |
| `kernels.fuse_stage` | policy | x-bigcherry `kernels.fuse-stage` |
| `kernels.gemv` | policy | x-bigcherry `kernels.gemv` |
| `kernels.gfx12_moe_gemv` | policy | x-bigcherry `kernels.gfx12-moe-gemv` |
| `kernels.moe_ahalf` | policy | x-bigcherry `kernels.moe-ahalf` |
| `kernels.moe_fp16` | policy | x-bigcherry `kernels.moe-fp16` |
| `kernels.moe_native` | policy | x-bigcherry `kernels.moe-native` |
| `layout.engine_cards` | discovered | machine: discovered BDFs in rank order from `x-bigcherry.ranks`; native `cards: [0..tp-1]` must match |
| `layout.helper_cards` | discovered | machine: discovered from `x-bigcherry.helpers` |
| `layout.name` | policy | deleted: the file name is the config name |
| `layout.rank_devices` | discovered | machine: discovered; native `rank-devices`, if given, must match |
| `limits.api_key` | policy | native `server.api-key-env` (a reference, never the secret) |
| `limits.host` | policy | native `server.host`; managed start refuses anything but 127.0.0.1 |
| `limits.http_threads` | policy | native `server.http-threads` (check the native level) |
| `limits.keep_alive_timeout` | policy | native `server.keep-alive-timeout` (check the native level) |
| `limits.max_body_mib` | policy | native `server.max-body-mib` (check the native level) |
| `limits.max_queued_requests` | policy | native `server.max-queued-requests` (check the native level) |
| `limits.max_stop_bytes` | policy | native `models.<id>.max-stop-bytes` (check the native level) |
| `limits.max_stop_strings` | policy | native `models.<id>.max-stop-strings` (check the native level) |
| `limits.no_cors` | policy | native `server.no-cors` (check the native level) |
| `limits.read_timeout` | policy | native `server.read-timeout` (check the native level) |
| `limits.retry_after` | policy | native `server.retry-after` (check the native level) |
| `limits.served_model_name` | policy | native: the model id under `models:` plus `models.<id>.aliases` (no flag of its own in YAML) |
| `limits.write_timeout` | policy | native `server.write-timeout` (check the native level) |
| `lora.control_vector` | policy | native `models.<id>.control-vector` |
| `lora.lora_dir` | policy | native `models.<id>.lora-dir` |
| `lora.lora_target_modules` | policy | native `models.<id>.lora-target-modules` |
| `lora.max_lora_rank` | policy | native `models.<id>.max-lora-rank` |
| `lora.max_loras` | policy | native `models.<id>.max-loras` |
| `memory.helper_budgets_mib` | derived | x-bigcherry `memory.helper-budgets-mib` (explicit override of a derived value; normally omitted) |
| `memory.host_kv_mib` | derived | x-bigcherry `memory.host-kv-mib` (explicit override of a derived value; normally omitted) |
| `memory.host_pool_mib` | derived | x-bigcherry `memory.host-pool-mib` (explicit override of a derived value; normally omitted) |
| `memory.host_reserve_mib` | derived | x-bigcherry `memory.host-reserve-mib` (explicit override of a derived value; normally omitted) |
| `memory.tail_budget_mib` | derived | x-bigcherry `memory.tail-budget-mib` (explicit override of a derived value; normally omitted) |
| `sampling.default_max_tokens` | policy | native `models.<id>.default-max-tokens` |
| `sampling.dry_allowed_length` | policy | native `models.<id>.dry-allowed-length` |
| `sampling.dry_base` | policy | native `models.<id>.dry-base` |
| `sampling.dry_multiplier` | policy | native `models.<id>.dry-multiplier` |
| `sampling.dry_penalty_last_n` | policy | native `models.<id>.dry-penalty-last-n` |
| `sampling.dry_sequence_breakers` | policy | native `models.<id>.dry-sequence-breakers` |
| `sampling.frequency_penalty` | policy | native `models.<id>.frequency-penalty` |
| `sampling.generation_config` | policy | native `models.<id>.generation-config` |
| `sampling.max_n` | policy | native `models.<id>.max-n` |
| `sampling.max_tokens_cap` | policy | native `models.<id>.max-tokens-cap` |
| `sampling.min_p` | policy | native `models.<id>.min-p` |
| `sampling.presence_penalty` | policy | native `models.<id>.presence-penalty` |
| `sampling.repeat_last_n` | policy | native `models.<id>.repeat-last-n` |
| `sampling.repetition_penalty` | policy | native `models.<id>.repetition-penalty` |
| `sampling.temp` | policy | native `models.<id>.temp` |
| `sampling.top_k` | policy | native `models.<id>.top-k` |
| `sampling.top_p` | policy | native `models.<id>.top-p` |
| `sampling.typical_p` | policy | native `models.<id>.typical-p` |
| `sampling.xtc_probability` | policy | native `models.<id>.xtc-probability` |
| `sampling.xtc_threshold` | policy | native `models.<id>.xtc-threshold` |
| `service.max_context` | policy | deleted: is `max-model-len` |
| `service.max_sessions` | policy | deleted: is `max-num-seqs` |
| `service.objective` | policy | x-bigcherry `service.objective` |
| `service.quality` | policy | x-bigcherry `service.quality` |
| `service.retention` | policy | x-bigcherry `service.retention` |
| `tuning.attn_prefill_fp8` | policy | native `models.<id>.attn-prefill-fp8` |
| `tuning.busy_prefill_chunk` | policy | native `models.<id>.busy-prefill-chunk` |
| `tuning.checkpoint_policy` | policy | native `models.<id>.checkpoint-policy` |
| `tuning.draft_coupling` | policy | native `models.<id>.draft-coupling` |
| `tuning.draft_expert_placement` | policy | native `models.<id>.draft-expert-placement` |
| `tuning.draft_model` | policy | native `models.<id>.draft-model` |
| `tuning.draft_short_from` | policy | native `models.<id>.draft-short-from` |
| `tuning.draft_window` | policy | native `models.<id>.draft-window` |
| `tuning.embedding_placement` | policy | native `models.<id>.embedding-placement` |
| `tuning.expert_pin_top` | policy | native `models.<id>.expert-pin-top` |
| `tuning.expert_profile` | policy | native `models.<id>.expert-profile` |
| `tuning.expert_route_record` | policy | native `models.<id>.expert-route-record` on a build with the diagnostic patch (rad0044/rad0045) only; a production build refuses the key |
| `tuning.expert_route_replay` | policy | native `models.<id>.expert-route-replay` on a build with the diagnostic patch (rad0044/rad0045) only; a production build refuses the key |
| `tuning.expert_stats` | policy | native `models.<id>.expert-stats` |
| `tuning.live` | policy | native `models.<id>.live` |
| `tuning.p2p` | policy | native `models.<id>.p2p` |
| `tuning.pipelined_prefill` | policy | native `models.<id>.pipelined-prefill` |
| `tuning.pipelined_prefill_min_rows` | policy | native `models.<id>.pipelined-prefill-min-rows` |
| `tuning.prefill_governor` | policy | native `models.<id>.prefill-governor` |
| `tuning.prefill_interleave` | policy | native `models.<id>.prefill-interleave` |
| `tuning.prefill_span` | policy | native `models.<id>.prefill-span` |
| `tuning.prefix_cache_ttl` | policy | native `models.<id>.prefix-cache-ttl` |
| `tuning.state_cache_dtype` | policy | native `models.<id>.state-cache-dtype` |
| `tuning.tp_attn_split` | policy | native `models.<id>.tp-attn-split` |
| `tuning.tp_ownership` | policy | native `models.<id>.tp-ownership` |
| `tuning.tp_wire_min_kb` | policy | native `models.<id>.tp-wire-min-kb` |
| `tuning.tune_pin` | policy | native `models.<id>.tune-pin` |
| `tuning.weight_prefetch` | policy | native `models.<id>.weight-prefetch` |
| `tuning.weights_disk_tier` | policy | native `models.<id>.weights-disk-tier` |
| `vision.image_max_pixels` | policy | native `models.<id>.image-max-pixels` |
| `vision.image_min_pixels` | policy | native `models.<id>.image-min-pixels` |
| `vision.image_tokens` | policy | native `models.<id>.image-tokens` |
| `vision.max_source_pixels` | policy | native `models.<id>.max-source-pixels` |
| `vision.mm_max_patches` | policy | native `models.<id>.mm-max-patches` |
| `vision.video_fps` | policy | native `models.<id>.video-fps` |
| `vision.video_max_frame_tokens` | policy | native `models.<id>.video-max-frame-tokens` |
| `vision.video_max_frames` | policy | native `models.<id>.video-max-frames` |
| `vision.video_max_pixels` | policy | native `models.<id>.video-max-pixels` |
| `vision.video_min_frames` | policy | native `models.<id>.video-min-frames` |
| `vision.video_min_pixels` | policy | native `models.<id>.video-min-pixels` |
| `vision.vision_tower_placement` | policy | native `models.<id>.vision-tower-placement` |

153 keys.

## Callers to migrate (same change, then delete the old path)

- bc-serve: RSC12 catalogue reader (TOML configs, `base` merge), layout reader, `explain`/`--schema` (YAML paths),
  `--set` (refused on a managed start; deleted with the TOML resolver keys), plan/manifest keys (drop TOML-derived inputs; digest covers the file), bundled config
  selection (`<prefix>/configs/<name>.yaml`).
- Generated serving contract: delete the v2 contract (`components/serving/contract/serving.toml`, generated
  codec/types/schema) with the bump; v3 is the only contract on main after it.
- Python tooling: `tools/bigcherry/cli/layout_cmds.py`, `engines/radiance/layouts/*`, release packer and `runtime_assets.py` (ship the YAML files only).
- Lab harnesses: `tools/lab/radiance/flashnext-parity.sh` and arm runners (`FLASHNEXT_PARITY_SETS` becomes YAML-path
  `--set` for lab arms, or a variant file for anything kept).
- Docs: `engines/radiance/configs/README.md` (format = this file), `docs/reference/engines/radiance/SERVING_MANIFEST.md`,
  RSC12 and VTP36 plan text naming TOML keys.

## Tests

- Schema: one refusal per rule (unknown key, wrong type, a plain string where a bool/number is wanted, duplicate key
  inside x-bigcherry, bad schema version, unknown `differs` path,
  orphan model id, two models in a managed file), one pass per shipped config.
- Agreement: TP vs ranks count, cards order, kernels cover rank archs, loopback host, model path resolution; each a
  refusal naming both sources.
- Single source: no key is accepted in both halves (a native key inside `x-bigcherry` is refused, an `x-bigcherry`
  key under `models.<id>` is refused by Radiance's closed model keys).
- Equivalence: the resolved values of all 18 configs equal the reference captured from the TOML catalogue before the
  switch (captured by `bc-serve explain --json` for each config before the switch), except the deliberate deletions the mapping lists (`layout.name`, `service.max_context`, `service.max_sessions`) and
  `engine.port`, which leaves the files and is supplied by bc-serve (bundled start and preflight alike); the diff
  output lists each of these explicitly.
- Variants: the `variant-of`/`differs` check over every pair.
- Release: an installed bundle starts any config by name with no repository path; a stock Radiance binary parses
  each file (`radiance --config X --check`).

## Acceptance

All 18 configs are single YAML files with `x-bigcherry`; no TOML catalogue, layout file, v2 contract or reader
remains; all callers above migrated; the tests pass; the equivalence diff is empty; RUM09's per-config start, canary
and stop pass on the 2.2 build.
