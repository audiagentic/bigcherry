# Serving option bindings at the migration boundary

At BigCherry `1f1547e72953`, the [wire registry](../../../engines/radiance/components/serving/contract/serving-v2.toml) contains 113 fields with `radiance_option`. Each row below identifies one field, the target YAML spelling or local owner, a representative reader and the migration disposition. Upstream file and line references refer to [b8bca788](https://codeberg.org/StillDeadcode/radiance/src/commit/b8bca7885a5a118bfa65a62dde74e0a6e2d4e42e/core). Local patch IDs refer to [BigCherry patch packages](../../../engines/radiance/patches). This is a design inventory for removing the mirror. Production code must use upstream's parser and option metadata.

| Disposition | Count | Implementer's action |
| --- | ---: | --- |
| Native | 83 | Remove the setting from the BC operator schema and copied engine argv. Pass the authenticated YAML to the native reader. |
| Derived | 14 | Keep the consumer's assertion, path, identity or measured resource decision. Read native input from the resolved Config. Compare explicit conflicting native values before activation. |
| Local | 6 | Qualify or withhold a retained local value, kernel or scheduler behavior. A parsed flag alone fails to demonstrate it. |
| Diagnostic | 7 | Preserve existing authorization and reporting. An old lab-only option stays restricted on the managed path, even when native YAML accepts it. |
| Obsolete | 3 | Remove the old binding with its call sites, use its replacement or report the explicit incompatibility shown below. |

Every row currently passes through rad0009's `bc_serving_engine_argv`. The writer and resolver consume many entries generically in `engine_option_fields`, `engine_of` and `leaf_of_key`. Whole-subtree serialization and cache-key reads do not establish independent ownership of an option. Replace them with the immutable snapshot identity and native receipts described in the [handoff](IMPLEMENTATION-HANDOFF.md). Non-obvious BC reads are annotated in the table. The `Local` row for KV is value-specific. Diagnostic is a disposition for migration, not a change to existing profile permissions. No option binding exclusively represents helper policy. The actual helper policy is under `spec.handoff`, `helper_common` and per-helper entries consumed by [helper_settings.cpp](../../../engines/radiance/components/serving/v2/helper_settings.cpp). Keep those plus residual plugin settings, ownership, power and measured admission.

| Current JSON pointer | Old flag | Disposition | Target field or owner | Representative reader and migration detail |
| --- | --- | --- | --- | --- |
| `/spec/intent/model` | `--model` | Derived | `models.<id>.model` | `core/engine_bringup.cpp:802` — P1 producer, helper_settings, owner_map and S8 replay require the verified path and model identity. Derive the role view from cfg.model. |
| `/spec/intent/service/max_sessions` | `--max-num-seqs` | Derived | `models.<id>.max-num-seqs` | `core/engine_bringup.cpp:987` — P1 producer computes concurrency and retention bounds. |
| `/spec/intent/service/max_context_tokens` | `--max-model-len` | Derived | `models.<id>.max-model-len` | `core/engine_bringup.cpp:933` — P1 producer computes effective context target and retention bounds. |
| `/spec/resources/host/pool_mib` | `--host-pool-mib` | Derived | `models.<id>.host-pool-mib` | `core/engine_bringup.cpp:535` — S8 and plan_cache.cpp verify measured pool. Target flag is total, normalized Config is per rank. Keep units explicit. |
| `/spec/resources/host/kv_mib` | `--prefix-cache-mib` | Derived | `models.<id>.prefix-cache-mib` | `core/engine.cpp:3491` — S8/S10 and stored_manifest.cpp count admitted host KV and pinned bytes. |
| `/spec/resources/host/reserve_mib` | `--host-headroom-mib` | Derived | `memory.host-headroom-mib` | `core/engine.h:343` — P1, S8 and stored_manifest.cpp protect reserve and fresh host room. |
| `/spec/resources/host/checkpoint_slots` | `--checkpoint-slots` | Derived | `models.<id>.checkpoint-slots` | `core/engine_bringup.cpp:1965` — ledger.cpp fits measured slots. Strict reader requires slots only in measured plans. |
| `/spec/processes/engine/device/headroom_mib` | `--gpu-headroom-mib` | Native | `memory.gpu-headroom-mib` | `core/engine_bringup.cpp:5254` — Native request. Check rank claimable bytes and site floors independently at admission. |
| `/spec/processes/engine/device/rank_devices` | `--rank-devices` | Derived | `cards + cfg.devices` | `rad0016::rank selection` — Retire custom spelling after ordered BDF/mask verification in fleet and offline. Retain ISA/code checks for mixed ranks. |
| `/spec/processes/engine/runtime/radiance_home` | `--radiance-home` | Derived | `server.radiance-home` | `core/engine_bringup.cpp:954` — p1_producer and helper_settings use the approved runtime home. Verify its content and search-order identity. |
| `/spec/processes/engine/runtime/kernels` | `--kernels` | Derived | `server.kernels` | `core/engine_bringup.cpp:954` — Approved runtime/plugin identity and ordering. V2 plan_cache_key currently includes the hierarchy. Require explicit matched selection for the lane. |
| `/spec/processes/engine/server/port` | `--port` | Derived | `server.port` | `core/fleet/fleet.cpp:65` — session.cpp readiness and radiance_manifest.py/workloads.py read it. Obtain from native listener/session receipt. |
| `/spec/processes/engine/scheduler/tp` | `--tp` | Derived | `models.<id>.tp` | `core/fleet/fleet.cpp:713` — v2/manifest.cpp checks rank count and live-policy eligibility. helper_settings supplies TP to the worker. |
| `/spec/processes/engine/scheduler/max_num_batched_tokens` | `--max-num-batched-tokens` | Native | `models.<id>.max-num-batched-tokens` | `core/engine_bringup.cpp:986` — Move current helper cache-key and benchmark receipt reads to snapshot identity/engine readout. |
| `/spec/processes/engine/scheduler/checkpoint_interval` | `--checkpoint-interval` | Native | `models.<id>.checkpoint-interval` | `core/engine_bringup.cpp:4342` |
| `/spec/processes/engine/execution/placement` | `--placement` | Native | `models.<id>.placement` | `core/engine_bringup.cpp:1832` |
| `/spec/processes/engine/execution/ngram_placement` | `--ngram-placement` | Derived | `models.<id>.ngram-placement` | `core/engine_bringup.cpp:388` — plan_cache.cpp checks resident consumers against placement. Read resolved native setting and check measured bytes. |
| `/spec/processes/engine/execution/kv_cache_dtype` | `--kv-cache-dtype` | Local | `models.<id>.kv-cache-dtype` | `core/engine_bringup.cpp:1195` — Native key, value-dependent behavior. Target Qwen4exp refuses the selected E4M3 fp8. RUM07 owns equivalent layout, planner bytes and qualification. |
| `/spec/processes/engine/execution/num_speculative_tokens` | `--num-speculative-tokens` | Derived | `models.<id>.num-speculative-tokens` | `core/engine_bringup.cpp:989` — RuntimeGuards::startup and benchmark reports need effective depth. Keep a verified value for the guard and receipt. |
| `/spec/processes/engine/execution/rope_scaling` | `--rope-scaling` | Native | `models.<id>.rope-scaling` | `core/engine_bringup.cpp:909` |
| `/spec/processes/engine/execution/prefix_resume` | `--prefix-resume` | Native | `models.<id>.prefix-resume` | `core/sched/geometry.cpp:180` |
| `/spec/processes/engine/execution/tp_wire` | `--tp-wire` | Native | `models.<id>.tp-wire` | `core/engine_bringup.cpp:991` |
| `/spec/processes/engine/execution/no_prefix_cache` | `--no-prefix-cache` | Native | `models.<id>.no-prefix-cache` | `core/engine.cpp:405` |
| `/spec/processes/engine/sampling/default_max_tokens` | `--default-max-tokens` | Native | `models.<id>.default-max-tokens` | `core/engine.cpp:3142` |
| `/spec/processes/engine/sampling/dry_allowed_length` | `--dry-allowed-length` | Native | `models.<id>.dry-allowed-length` | `core/engine.cpp:3093` |
| `/spec/processes/engine/sampling/dry_base` | `--dry-base` | Native | `models.<id>.dry-base` | `core/engine.cpp:3092` |
| `/spec/processes/engine/sampling/dry_multiplier` | `--dry-multiplier` | Native | `models.<id>.dry-multiplier` | `core/engine.cpp:3091` |
| `/spec/processes/engine/sampling/dry_penalty_last_n` | `--dry-penalty-last-n` | Native | `models.<id>.dry-penalty-last-n` | `core/engine.cpp:3094` |
| `/spec/processes/engine/sampling/dry_sequence_breakers` | `--dry-sequence-breakers` | Native | `models.<id>.dry-sequence-breakers` | `core/engine.cpp:3095` |
| `/spec/processes/engine/sampling/frequency_penalty` | `--frequency-penalty` | Native | `models.<id>.frequency-penalty` | `core/engine.cpp:3089` |
| `/spec/processes/engine/sampling/generation_config` | `--generation-config` | Native | `models.<id>.generation-config` | `core/engine.cpp:3061` |
| `/spec/processes/engine/sampling/max_n` | `--max-n` | Native | `models.<id>.max-n` | `core/engine.cpp:3145` |
| `/spec/processes/engine/sampling/max_tokens_cap` | `--max-tokens-cap` | Native | `models.<id>.max-tokens-cap` | `core/engine.cpp:3144` |
| `/spec/processes/engine/sampling/min_p` | `--min-p` | Native | `models.<id>.min-p` | `core/engine.cpp:3081` |
| `/spec/processes/engine/sampling/presence_penalty` | `--presence-penalty` | Native | `models.<id>.presence-penalty` | `core/engine.cpp:3088` |
| `/spec/processes/engine/sampling/repeat_last_n` | `--repeat-last-n` | Native | `models.<id>.repeat-last-n` | `core/engine.cpp:3090` |
| `/spec/processes/engine/sampling/repetition_penalty` | `--repetition-penalty` | Native | `models.<id>.repetition-penalty` | `core/engine.cpp:3087` |
| `/spec/processes/engine/sampling/temp` | `--temp` | Native | `models.<id>.temp` | `core/engine.cpp:3079` |
| `/spec/processes/engine/sampling/top_k` | `--top-k` | Native | `models.<id>.top-k` | `core/engine.cpp:3082` |
| `/spec/processes/engine/sampling/top_p` | `--top-p` | Native | `models.<id>.top-p` | `core/engine.cpp:3080` |
| `/spec/processes/engine/sampling/typical_p` | `--typical-p` | Native | `models.<id>.typical-p` | `core/engine.cpp:3086` |
| `/spec/processes/engine/sampling/xtc_probability` | `--xtc-probability` | Native | `models.<id>.xtc-probability` | `core/engine.cpp:3096` |
| `/spec/processes/engine/sampling/xtc_threshold` | `--xtc-threshold` | Native | `models.<id>.xtc-threshold` | `core/engine.cpp:3097` |
| `/spec/processes/engine/limits/api_key` | `--api-key` | Native | `server.api-key-env` | `core/fleet/fleet.cpp:67` — Managed persistence accepts the native secret reference. Refuse authored inline keys before retaining bytes. |
| `/spec/processes/engine/limits/host` | `--host` | Native | `server.host` | `core/fleet/fleet.cpp:64` |
| `/spec/processes/engine/limits/http_threads` | `--http-threads` | Native | `server.http-threads` | `core/fleet/fleet.cpp:66` |
| `/spec/processes/engine/limits/keep_alive_timeout` | `--keep-alive-timeout` | Native | `server.keep-alive-timeout` | `core/fleet/fleet.cpp:71` |
| `/spec/processes/engine/limits/max_body_mib` | `--max-body-mib` | Native | `server.max-body-mib` | `core/fleet/fleet.cpp:72` |
| `/spec/processes/engine/limits/max_queued_requests` | `--max-queued-requests` | Native | `models.<id>.max-queued-requests` | `core/engine.cpp:3524` |
| `/spec/processes/engine/limits/max_stop_bytes` | `--max-stop-bytes` | Native | `models.<id>.max-stop-bytes` | `core/engine.cpp:3146` |
| `/spec/processes/engine/limits/max_stop_strings` | `--max-stop-strings` | Native | `models.<id>.max-stop-strings` | `core/engine.cpp:3146` |
| `/spec/processes/engine/limits/no_cors` | `--no-cors` | Native | `server.no-cors` | `core/fleet/fleet.cpp:68` |
| `/spec/processes/engine/limits/read_timeout` | `--read-timeout` | Native | `server.read-timeout` | `core/fleet/fleet.cpp:69` |
| `/spec/processes/engine/limits/retry_after` | `--retry-after` | Native | `server.retry-after` | `core/engine.cpp:3537` |
| `/spec/processes/engine/limits/served_model_name` | `--served-model-name` | Obsolete | `models.<id>.aliases` | `core/fleet/config.cpp:595` — Reader supplies the entry name. Replace the old flag spelling and preserve required aliases. |
| `/spec/processes/engine/limits/write_timeout` | `--write-timeout` | Native | `server.write-timeout` | `core/fleet/fleet.cpp:70` |
| `/spec/processes/engine/chat/chat_template_kwargs` | `--chat-template-kwargs` | Native | `models.<id>.chat-template-kwargs` | `core/engine.cpp:3149` |
| `/spec/processes/engine/chat/override_chat_template` | `--override-chat-template` | Native | `models.<id>.override-chat-template` | `core/engine.cpp:3389` |
| `/spec/processes/engine/chat/reasoning_effort` | `--reasoning-effort` | Native | `models.<id>.reasoning-effort` | `core/engine.cpp:3494` |
| `/spec/processes/engine/chat/reasoning_effort_map` | `--reasoning-effort-map` | Native | `models.<id>.reasoning-effort-map` | `core/engine.cpp:3495` |
| `/spec/processes/engine/chat/reasoning_format` | `--reasoning-format` | Native | `models.<id>.reasoning-format` | `core/engine.cpp:3150` |
| `/spec/processes/engine/chat/thinking_token_budget` | `--thinking-token-budget` | Native | `models.<id>.thinking-token-budget` | `core/engine.cpp:3151` |
| `/spec/processes/engine/lora/control_vector` | `--control-vector` | Native | `models.<id>.control-vector` | `core/engine_bringup.cpp:1427` |
| `/spec/processes/engine/lora/lora_dir` | `--lora-dir` | Native | `models.<id>.lora-dir` | `core/engine_lora.cpp:400` |
| `/spec/processes/engine/lora/lora_target_modules` | `--lora-target-modules` | Native | `models.<id>.lora-target-modules` | `core/engine_lora.cpp:90` |
| `/spec/processes/engine/lora/max_lora_rank` | `--max-lora-rank` | Native | `models.<id>.max-lora-rank` | `core/engine_lora.cpp:234` |
| `/spec/processes/engine/lora/max_loras` | `--max-loras` | Native | `models.<id>.max-loras` | `core/engine_lora.cpp:113` |
| `/spec/processes/engine/vision/image_max_pixels` | `--image-max-pixels` | Native | `models.<id>.image-max-pixels` | `core/engine.cpp:3618` |
| `/spec/processes/engine/vision/image_min_pixels` | `--image-min-pixels` | Native | `models.<id>.image-min-pixels` | `core/engine.cpp:3618` |
| `/spec/processes/engine/vision/image_tokens` | `--image-tokens` | Native | `models.<id>.image-tokens` | `core/engine.cpp:3626` |
| `/spec/processes/engine/vision/max_source_pixels` | `--max-source-pixels` | Native | `models.<id>.max-source-pixels` | `core/engine.cpp:3665` |
| `/spec/processes/engine/vision/mm_max_patches` | `--mm-max-patches` | Native | `models.<id>.mm-max-patches` | `core/engine.cpp:4139` |
| `/spec/processes/engine/vision/video_fps` | `--video-fps` | Native | `models.<id>.video-fps` | `core/engine.cpp:3612` |
| `/spec/processes/engine/vision/video_max_frame_tokens` | `--video-max-frame-tokens` | Native | `models.<id>.video-max-frame-tokens` | `core/engine.cpp:3612` |
| `/spec/processes/engine/vision/video_max_frames` | `--video-max-frames` | Native | `models.<id>.video-max-frames` | `core/engine.cpp:3612` |
| `/spec/processes/engine/vision/video_max_pixels` | `--video-max-pixels` | Native | `models.<id>.video-max-pixels` | `core/engine.cpp:3611` |
| `/spec/processes/engine/vision/video_min_frames` | `--video-min-frames` | Native | `models.<id>.video-min-frames` | `core/engine.cpp:3611` |
| `/spec/processes/engine/vision/video_min_pixels` | `--video-min-pixels` | Native | `models.<id>.video-min-pixels` | `core/engine.cpp:3611` |
| `/spec/processes/engine/vision/vision_tower_placement` | `--vision-tower-placement` | Native | `models.<id>.vision-tower-placement` | `core/engine.cpp:2972` |
| `/spec/processes/engine/tuning/attn_prefill_fp8` | `--attn-prefill-fp8` | Local feature, lost in ABI43 candidate | `off`, `qk`, `pv`, `both` in the v1.17.0 baseline; candidate parser currently has no control | Restore the exact mode contract and route it to `prefill_fp8` row matching. Historical lossy rows exist only at H256/GQA6/page16 for E4M3, fp8_v4 and k6_v4 (nine mode/format rows); unsupported geometry or B16 combinations must refuse, not fall through. See RUM07 review RV4241. |
| `/spec/processes/engine/tuning/busy_prefill_chunk` | `--busy-prefill-chunk` | Native | `models.<id>.busy-prefill-chunk` | `core/sched/scheduler.cpp:165` |
| `/spec/processes/engine/tuning/checkpoint_policy` | `--checkpoint-policy` | Native | `models.<id>.checkpoint-policy` | `core/engine_bringup.cpp:4255` |
| `/spec/processes/engine/tuning/draft_coupling` | `--draft-coupling` | Native | `models.<id>.draft-coupling` | `core/engine.cpp:883` |
| `/spec/processes/engine/tuning/draft_expert_placement` | `--draft-expert-placement` | Native | `models.<id>.draft-expert-placement` | `core/engine_bringup.cpp:2077` |
| `/spec/processes/engine/tuning/draft_model` | `--draft-model` | Obsolete | `native refusal` | `core/config.cpp:1006` — Even v1.17 rejected it. Native parser reports that the drafter must be merged into the container. |
| `/spec/processes/engine/tuning/draft_short_from` | `--draft-short-from` | Native | `models.<id>.draft-short-from` | `core/engine_bringup.cpp:1606` |
| `/spec/processes/engine/tuning/draft_window` | `--draft-window` | Native | `models.<id>.draft-window` | `core/engine_bringup.cpp:1325` |
| `/spec/processes/engine/tuning/embedding_placement` | `--embedding-placement` | Native | `models.<id>.embedding-placement` | `core/engine.cpp:2968` |
| `/spec/processes/engine/tuning/expert_pin_top` | `--expert-pin-top` | Native | `models.<id>.expert-pin-top` | `core/place/planner.cpp:1139` |
| `/spec/processes/engine/tuning/expert_profile` | `--expert-profile` | Native | `models.<id>.expert-profile` | `core/engine_bringup.cpp:2258` |
| `/spec/processes/engine/tuning/expert_stats` | `--expert-stats` | Native | `models.<id>.expert-stats` | `core/engine.cpp:1058` — Upstream owns the basic output. Check rad0043 protections before enabling its optional diagnostic variant. |
| `/spec/processes/engine/tuning/expert_route_record` | `--expert-route-record` | Diagnostic | `local diagnostic only` | `rad0044::route recorder` — Withhold from production. A diagnostic port needs observed records. |
| `/spec/processes/engine/tuning/live` | `--live` | Native | `server.live` | `core/engine.cpp:4226` |
| `/spec/processes/engine/tuning/p2p` | `--p2p` | Native | `models.<id>.p2p` | `core/engine_bringup.cpp:4838` |
| `/spec/processes/engine/tuning/prefill_governor` | `--prefill-governor` | Local | `local scheduler` | `rad0036::prefill governor` — Qwen long profile needs a policy comparison in RUM08. Withhold that layout in first candidate. |
| `/spec/processes/engine/tuning/prefill_interleave` | `--prefill-interleave` | Native | `models.<id>.prefill-interleave` | `core/engine_bringup.cpp:1414` |
| `/spec/processes/engine/tuning/prefill_span` | `--prefill-span` | Native | `models.<id>.prefill-span` | `core/engine_bringup.cpp:1016` |
| `/spec/processes/engine/tuning/prefix_cache_ttl` | `--prefix-cache-ttl` | Native | `models.<id>.prefix-cache-ttl` | `core/engine.cpp:2513` |
| `/spec/processes/engine/tuning/state_cache_dtype` | `--state-cache-dtype` | Native | `models.<id>.state-cache-dtype` | `core/engine_bringup.cpp:1004` |
| `/spec/processes/engine/tuning/tp_attn_split` | `--tp-attn-split` | Native | `models.<id>.tp-attn-split` | `core/engine.cpp:2963` |
| `/spec/processes/engine/tuning/tp_wire_min_kb` | `--tp-wire-min-kb` | Native | `models.<id>.tp-wire-min-kb` | `core/engine.cpp:2962` |
| `/spec/processes/engine/tuning/weight_prefetch` | `--weight-prefetch` | Native | `models.<id>.weight-prefetch` | `core/engine.cpp:2964` |
| `/spec/processes/engine/tuning/weights_disk_tier` | `--weights-disk-tier` | Native | `models.<id>.weights-disk-tier` | `core/engine_bringup.cpp:1007` |
| `/spec/processes/engine/tuning/tp_ownership` | `--tp-ownership` | Local | `local rank geometry` | `rad0018/19::geom_from` — Carry admitted per-rank spans when qualifying the mixed-TP layout. |
| `/spec/processes/engine/tuning/pipelined_prefill` | `--pipelined-prefill` | Local | `local pipeline policy` | `rad0020::BC_VTP14_PIPE_MODE` — Withhold with TP layouts until qualification. |
| `/spec/processes/engine/tuning/pipelined_prefill_min_rows` | `--pipelined-prefill-min-rows` | Local | `local pipeline policy` | `rad0020::BC_VTP14_PIPE_MODE` — Retain typed threshold alongside the pipeline mode for those layouts. |
| `/spec/processes/engine/tuning/tune_pin` | `--tune-pin` | Local | `local kernel selection` | `rad0022::BC_VTP31_TUNE_PIN_APPLY` — Bind to the matched kernel build and detect conflicting RADIANCE_TUNE. |
| `/spec/processes/engine/diagnostics/debug_accept_reference_kernels` | `--debug-accept-reference-kernels` | Diagnostic | `server.debug-accept-reference-kernels` | `core/engine_bringup.cpp:4878` — Target supports this at server level. Keep qualification checks for actual per-rank kernels. |
| `/spec/processes/engine/diagnostics/debug_placement` | `--debug-placement` | Diagnostic | `models.<id>.debug-placement` | `core/engine_bringup.cpp:2281` |
| `/spec/experiment/engine/profile_ops` | `--profile-ops` | Diagnostic | `models.<id>.profile-ops` | `core/engine.cpp:775` — RuntimeGuards::startup distinguishes a profiled run. Preserve that guard and lab-profile authorization. |
| `/spec/experiment/engine/debug_selection` | `--debug-selection` | Diagnostic | `models.<id>.debug-selection` | `core/engine_bringup.cpp:5348` |
| `/spec/experiment/engine/deterministic` | `--deterministic` | Diagnostic | `models.<id>.deterministic` | `core/engine.cpp:2977` — Lab key under the old contract. Keep the managed experiment gate. |
| `/spec/experiment/engine/kv_resident` | `--kv-resident` | Diagnostic | `models.<id>.kv-resident` | `core/engine.cpp:3311` — Lab key under the old contract. Keep the managed experiment gate. |

For an input audit, parse the TOML and compare all `(path, radiance_option)` pairs with columns one and two. Include fields under `spec.experiment`. Recheck target field levels in [fleet/config.cpp](https://codeberg.org/StillDeadcode/radiance/src/commit/b8bca7885a5a118bfa65a62dde74e0a6e2d4e42e/core/fleet/config.cpp), which also allows certain per-model card-memory overrides. Recheck actual values and readers when changing either SHA.
# Historical source inventory — not ABI43 implementation guidance

This inventory maps BigCherry's prior serving-v2 fields to intermediate
Radiance candidate `b8bca7885a5a118bfa65a62dde74e0a6e2d4e42e` (ABI41). The
frozen migration target is `3deabdaeca17b58bad90d332f4b204cf7bb8a10e`
(ABI43). Revalidate upstream field names and levels against the frozen source;
RUM04 and the [implementation handoff](IMPLEMENTATION-HANDOFF.md) govern the
current native YAML and caller migration. This is not runtime acceptance.
