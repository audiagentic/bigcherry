# radiance patch footprint — 2.2.0

Upstream pin: `b6b8751cccb0e834a13ae9e5afca824ee9a487aa`.

40 composed patches; **649 anchor edits** (614 upstream, 35 on patch-created files; 672 expected match sites); 681 total edits, 32 in created files; 82 upstream files.

Counts are lexical, not a measure of independent semantics, net code savings, or successful application. Regex anchor bytes and full replacement text bytes are counted as authored, not as actual upstream match or net diff bytes.

## Patches (ranked by non-create anchor edits)

| Patch | Upstream files | Created files | Anchors | Upstream anchors | Edits | Anchor bytes | Inserted bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rad0053_v4_kv_layouts` | 13 | 0 | 58 | 35 | 58 | 5825 | 27765 |
| `rad0051_b16_kv_kernels` | 11 | 3 | 44 | 41 | 47 | 3112 | 86822 |
| `rad0002_moe_remote_handoff` | 10 | 5 | 41 | 41 | 46 | 2309 | 52904 |
| `rad0016_vtp14_rank_code_selection` | 11 | 5 | 38 | 38 | 43 | 5730 | 52269 |
| `rad0045_expert_route_replay` | 15 | 0 | 36 | 36 | 36 | 3080 | 27756 |
| `rad0036_prefill_governor` | 11 | 1 | 29 | 29 | 30 | 7396 | 37516 |
| `rad0055_step_phase_trace` | 4 | 0 | 29 | 29 | 29 | 4694 | 15135 |
| `rad0048_abi44_kv_contract` | 9 | 1 | 28 | 28 | 29 | 1669 | 8863 |
| `rad0054_attn_prefill_fp8` | 11 | 0 | 25 | 24 | 25 | 1500 | 16621 |
| `rad0052_b16_kv_geometries` | 8 | 0 | 22 | 19 | 22 | 1725 | 21014 |
| `rad0025_multiseq_pipelined_prefill` | 7 | 0 | 21 | 21 | 21 | 3767 | 8728 |
| `rad0049_qwen4exp_e4m3_kv` | 7 | 1 | 21 | 21 | 22 | 1984 | 15986 |
| `rad0042_kv_b16_selftest` | 2 | 1 | 20 | 20 | 21 | 2698 | 10853 |
| `rad0032_ts3_duplex` | 5 | 0 | 18 | 16 | 18 | 3292 | 12008 |
| `rad0033_ts3_once` | 5 | 0 | 18 | 18 | 18 | 2540 | 6038 |
| `rad0018_vtp14_n2_tp_ownership` | 11 | 0 | 17 | 17 | 17 | 2130 | 14973 |
| `rad0050_b16_kv_runtime_contract` | 7 | 2 | 16 | 16 | 18 | 2615 | 12783 |
| `rad0044_expert_route_record` | 7 | 0 | 16 | 16 | 16 | 1285 | 13897 |
| `rad0019_vtp14_n2_tp_ownership_fp8` | 6 | 0 | 15 | 15 | 15 | 3519 | 11237 |
| `rad0070_managed_native_entry` | 4 | 0 | 14 | 14 | 14 | 572 | 3273 |
| `rad0020_vtp14_tp3_pipelined_prefill` | 7 | 0 | 13 | 13 | 13 | 982 | 4474 |
| `rad0027_mixed_one_pipelined_prefill` | 4 | 0 | 12 | 12 | 12 | 3219 | 5580 |
| `rad0046_retained_moe_rows` | 7 | 1 | 11 | 11 | 12 | 553 | 3854 |
| `rad0026_mixed_pipelined_prefill` | 4 | 0 | 10 | 10 | 10 | 2638 | 7332 |
| `rad0010_routing_evidence` | 2 | 1 | 10 | 10 | 11 | 488 | 14328 |
| `rad0023_ar_timeline` | 1 | 0 | 7 | 7 | 7 | 1677 | 4552 |
| `rad0017_vtp14_n3_shared_rendezvous` | 4 | 5 | 7 | 7 | 12 | 909 | 42219 |
| `rad0028_rank_consistent_block_drafts` | 3 | 0 | 7 | 7 | 7 | 702 | 2382 |
| `rad0022_tune_pin` | 4 | 0 | 7 | 7 | 7 | 574 | 2531 |
| `rad0011_plan_memory` | 1 | 4 | 7 | 7 | 11 | 300 | 23460 |
| `rad0031_ts_timeline` | 1 | 0 | 6 | 6 | 6 | 1982 | 5023 |
| `rad0021_vtp14_weighted_twoshot_3rank` | 3 | 0 | 6 | 3 | 6 | 1630 | 3196 |
| `rad0030_int8_pipelined_prefill` | 3 | 0 | 6 | 6 | 6 | 953 | 1416 |
| `rad0004_moe_remote_diagnostics` | 1 | 1 | 4 | 4 | 5 | 222 | 5777 |
| `rad0024_ts3_decode` | 1 | 0 | 3 | 3 | 3 | 684 | 1670 |
| `rad0012_helper_kernel_config` | 2 | 1 | 3 | 3 | 4 | 95 | 1943 |
| `rad0064_owned_weighted_head` | 1 | 0 | 1 | 1 | 1 | 591 | 878 |
| `rad0008_hc_upmix_rt_sweep` | 1 | 0 | 1 | 1 | 1 | 360 | 1249 |
| `rad0029_rank_own_split` | 1 | 0 | 1 | 1 | 1 | 311 | 4040 |
| `rad0043_expert_stats_protected` | 1 | 0 | 1 | 1 | 1 | 76 | 1910 |

## Upstream files (ranked by number of anchoring patches)

| Upstream file | Patches | Anchors | Anchor bytes | Patch IDs |
| --- | ---: | ---: | ---: | --- |
| `core/engine_bringup.cpp` | 13 | 33 | 2479 | `rad0002_moe_remote_handoff`, `rad0011_plan_memory`, `rad0016_vtp14_rank_code_selection`, `rad0017_vtp14_n3_shared_rendezvous`, `rad0018_vtp14_n2_tp_ownership`, `rad0020_vtp14_tp3_pipelined_prefill`, `rad0022_tune_pin`, `rad0029_rank_own_split`, `rad0044_expert_route_record`, `rad0045_expert_route_replay`, `rad0050_b16_kv_runtime_contract`, `rad0053_v4_kv_layouts`, `rad0054_attn_prefill_fp8` |
| `arch/qwen35_fp8/qwen35_fp8.cpp` | 11 | 43 | 5582 | `rad0019_vtp14_n2_tp_ownership_fp8`, `rad0020_vtp14_tp3_pipelined_prefill`, `rad0025_multiseq_pipelined_prefill`, `rad0026_mixed_pipelined_prefill`, `rad0027_mixed_one_pipelined_prefill`, `rad0028_rank_consistent_block_drafts`, `rad0030_int8_pipelined_prefill`, `rad0052_b16_kv_geometries`, `rad0053_v4_kv_layouts`, `rad0054_attn_prefill_fp8`, `rad0064_owned_weighted_head` |
| `core/config.cpp` | 10 | 34 | 4011 | `rad0016_vtp14_rank_code_selection`, `rad0018_vtp14_n2_tp_ownership`, `rad0020_vtp14_tp3_pipelined_prefill`, `rad0022_tune_pin`, `rad0036_prefill_governor`, `rad0044_expert_route_record`, `rad0045_expert_route_replay`, `rad0050_b16_kv_runtime_contract`, `rad0053_v4_kv_layouts`, `rad0054_attn_prefill_fp8` |
| `tests/config_test.cpp` | 9 | 12 | 1831 | `rad0016_vtp14_rank_code_selection`, `rad0018_vtp14_n2_tp_ownership`, `rad0020_vtp14_tp3_pipelined_prefill`, `rad0022_tune_pin`, `rad0036_prefill_governor`, `rad0044_expert_route_record`, `rad0045_expert_route_replay`, `rad0053_v4_kv_layouts`, `rad0054_attn_prefill_fp8` |
| `core/rad_core.h` | 8 | 9 | 545 | `rad0016_vtp14_rank_code_selection`, `rad0018_vtp14_n2_tp_ownership`, `rad0020_vtp14_tp3_pipelined_prefill`, `rad0022_tune_pin`, `rad0036_prefill_governor`, `rad0044_expert_route_record`, `rad0045_expert_route_replay`, `rad0054_attn_prefill_fp8` |
| `core/runtime/ctx.cpp` | 7 | 43 | 5333 | `rad0002_moe_remote_handoff`, `rad0010_routing_evidence`, `rad0025_multiseq_pipelined_prefill`, `rad0026_mixed_pipelined_prefill`, `rad0044_expert_route_record`, `rad0045_expert_route_replay`, `rad0055_step_phase_trace` |
| `libr4d/r4d_rows.cpp` | 7 | 20 | 1377 | `rad0012_helper_kernel_config`, `rad0032_ts3_duplex`, `rad0033_ts3_once`, `rad0045_expert_route_replay`, `rad0046_retained_moe_rows`, `rad0049_qwen4exp_e4m3_kv`, `rad0054_attn_prefill_fp8` |
| `tests/arch_test.cpp` | 7 | 12 | 1094 | `rad0027_mixed_one_pipelined_prefill`, `rad0028_rank_consistent_block_drafts`, `rad0049_qwen4exp_e4m3_kv`, `rad0051_b16_kv_kernels`, `rad0052_b16_kv_geometries`, `rad0053_v4_kv_layouts`, `rad0054_attn_prefill_fp8` |
| `libr4d/r4d_plugin.h` | 7 | 8 | 459 | `rad0045_expert_route_replay`, `rad0046_retained_moe_rows`, `rad0049_qwen4exp_e4m3_kv`, `rad0051_b16_kv_kernels`, `rad0052_b16_kv_geometries`, `rad0053_v4_kv_layouts`, `rad0054_attn_prefill_fp8` |
| `core/engine.cpp` | 6 | 21 | 1287 | `rad0002_moe_remote_handoff`, `rad0010_routing_evidence`, `rad0017_vtp14_n3_shared_rendezvous`, `rad0043_expert_stats_protected`, `rad0044_expert_route_record`, `rad0045_expert_route_replay` |
| `arch/common/rad_block_attn_gated_fp8.h` | 6 | 13 | 3725 | `rad0019_vtp14_n2_tp_ownership_fp8`, `rad0025_multiseq_pipelined_prefill`, `rad0026_mixed_pipelined_prefill`, `rad0027_mixed_one_pipelined_prefill`, `rad0051_b16_kv_kernels`, `rad0054_attn_prefill_fp8` |
| `libr4d/r4d_ar_entry.hip` | 5 | 8 | 2637 | `rad0017_vtp14_n3_shared_rendezvous`, `rad0021_vtp14_weighted_twoshot_3rank`, `rad0024_ts3_decode`, `rad0032_ts3_duplex`, `rad0033_ts3_once` |
| `libr4d/r4d_ar_twoshot_Nrank_exact.hip` | 4 | 26 | 5336 | `rad0021_vtp14_weighted_twoshot_3rank`, `rad0031_ts_timeline`, `rad0032_ts3_duplex`, `rad0033_ts3_once` |
| `core/runtime/ctx.h` | 4 | 16 | 1036 | `rad0002_moe_remote_handoff`, `rad0044_expert_route_record`, `rad0045_expert_route_replay`, `rad0055_step_phase_trace` |
| `libr4d/r4d_attn_paged_gqa8.hip` | 4 | 16 | 1673 | `rad0049_qwen4exp_e4m3_kv`, `rad0051_b16_kv_kernels`, `rad0052_b16_kv_geometries`, `rad0053_v4_kv_layouts` |
| `arch/qwen4exp_fp8/qwen4exp_fp8.cpp` | 4 | 13 | 1002 | `rad0049_qwen4exp_e4m3_kv`, `rad0051_b16_kv_kernels`, `rad0052_b16_kv_geometries`, `rad0053_v4_kv_layouts` |
| `arch/common/rad_arch.h` | 4 | 9 | 1560 | `rad0002_moe_remote_handoff`, `rad0018_vtp14_n2_tp_ownership`, `rad0019_vtp14_n2_tp_ownership_fp8`, `rad0054_attn_prefill_fp8` |
| `libr4d/r4d.h` | 4 | 4 | 614 | `rad0021_vtp14_weighted_twoshot_3rank`, `rad0032_ts3_duplex`, `rad0033_ts3_once`, `rad0052_b16_kv_geometries` |
| `tests/CMakeLists.txt` | 4 | 4 | 105 | `rad0042_kv_b16_selftest`, `rad0048_abi44_kv_contract`, `rad0050_b16_kv_runtime_contract`, `rad0070_managed_native_entry` |
| `libr4d/r4d_qk_norm_rope_gate.hip` | 3 | 22 | 1509 | `rad0049_qwen4exp_e4m3_kv`, `rad0051_b16_kv_kernels`, `rad0053_v4_kv_layouts` |
| `arch/common/rad_block_moe_fp8.h` | 3 | 17 | 1347 | `rad0002_moe_remote_handoff`, `rad0004_moe_remote_diagnostics`, `rad0045_expert_route_replay` |
| `arch/common/rad_block_gdn_fp8.h` | 3 | 11 | 4245 | `rad0025_multiseq_pipelined_prefill`, `rad0026_mixed_pipelined_prefill`, `rad0027_mixed_one_pipelined_prefill` |
| `core/build/rad_builder.cpp` | 3 | 9 | 1137 | `rad0016_vtp14_rank_code_selection`, `rad0050_b16_kv_runtime_contract`, `rad0053_v4_kv_layouts` |
| `libr4d/r4d_attn_paged_h256_gqa6.hip` | 3 | 9 | 372 | `rad0052_b16_kv_geometries`, `rad0053_v4_kv_layouts`, `rad0054_attn_prefill_fp8` |
| `abi/rad_builder.h` | 3 | 7 | 540 | `rad0018_vtp14_n2_tp_ownership`, `rad0020_vtp14_tp3_pipelined_prefill`, `rad0048_abi44_kv_contract` |
| `abi/rad_abi_layout_rows.h` | 3 | 6 | 256 | `rad0018_vtp14_n2_tp_ownership`, `rad0020_vtp14_tp3_pipelined_prefill`, `rad0048_abi44_kv_contract` |
| `arch/common/rad_fp8.h` | 3 | 6 | 831 | `rad0019_vtp14_n2_tp_ownership_fp8`, `rad0025_multiseq_pipelined_prefill`, `rad0030_int8_pipelined_prefill` |
| `libr4d/r4d_shapes.cpp` | 3 | 6 | 287 | `rad0045_expert_route_replay`, `rad0046_retained_moe_rows`, `rad0051_b16_kv_kernels` |
| `libr4d/r4d_args.h` | 3 | 4 | 165 | `rad0012_helper_kernel_config`, `rad0032_ts3_duplex`, `rad0033_ts3_once` |
| `libr4d/r4d_selftest.cpp` | 2 | 23 | 3037 | `rad0042_kv_b16_selftest`, `rad0049_qwen4exp_e4m3_kv` |
| `libr4d/r4d_moe.hip` | 2 | 7 | 535 | `rad0045_expert_route_replay`, `rad0046_retained_moe_rows` |
| `arch/common/rad_block_dflash2.h` | 2 | 6 | 682 | `rad0019_vtp14_n2_tp_ownership_fp8`, `rad0028_rank_consistent_block_drafts` |
| `core/mem/kv.cpp` | 2 | 6 | 477 | `rad0050_b16_kv_runtime_contract`, `rad0053_v4_kv_layouts` |
| `libr4d/r4d_model_bf16.hip` | 2 | 5 | 212 | `rad0051_b16_kv_kernels`, `rad0053_v4_kv_layouts` |
| `arch/common/rad_block_mlp_fp8.h` | 2 | 4 | 614 | `rad0019_vtp14_n2_tp_ownership_fp8`, `rad0030_int8_pipelined_prefill` |
| `abi/rad_runtime.h` | 2 | 2 | 129 | `rad0025_multiseq_pipelined_prefill`, `rad0045_expert_route_replay` |
| `core/runtime/issue.cpp` | 2 | 2 | 167 | `rad0045_expert_route_replay`, `rad0055_step_phase_trace` |
| `tests/arch_fakes.h` | 2 | 2 | 112 | `rad0002_moe_remote_handoff`, `rad0045_expert_route_replay` |
| `tests/abi_layout_table.h` | 1 | 9 | 375 | `rad0048_abi44_kv_contract` |
| `core/fleet/fleet.cpp` | 1 | 8 | 315 | `rad0070_managed_native_entry` |
| `core/plugin/select.cpp` | 1 | 8 | 1516 | `rad0016_vtp14_rank_code_selection` |
| `core/sched/scheduler.cpp` | 1 | 7 | 1823 | `rad0036_prefill_governor` |
| `libr4d/r4d_ar_oneshot_Nrank_exact.hip` | 1 | 7 | 1677 | `rad0023_ar_timeline` |
| `core/device/aql.cpp` | 1 | 5 | 775 | `rad0016_vtp14_rank_code_selection` |
| `core/plugin/registry.h` | 1 | 5 | 1094 | `rad0016_vtp14_rank_code_selection` |
| `tests/abi_layout_test.cpp` | 1 | 5 | 361 | `rad0048_abi44_kv_contract` |
| `core/kvstream/stage.cpp` | 1 | 4 | 638 | `rad0055_step_phase_trace` |
| `core/main.cpp` | 1 | 4 | 114 | `rad0070_managed_native_entry` |
| `core/place/planner.cpp` | 1 | 4 | 267 | `rad0002_moe_remote_handoff` |
| `core/server/metrics.cpp` | 1 | 4 | 1607 | `rad0036_prefill_governor` |
| `arch/qwen35_bf16/qwen35_bf16.cpp` | 1 | 3 | 447 | `rad0018_vtp14_n2_tp_ownership` |
| `core/plugin/loader.cpp` | 1 | 3 | 570 | `rad0016_vtp14_rank_code_selection` |
| `libref/ref_attn.cpp` | 1 | 3 | 152 | `rad0051_b16_kv_kernels` |
| `arch/common/rad_block_hc.h` | 1 | 2 | 96 | `rad0002_moe_remote_handoff` |
| `core/engine.h` | 1 | 2 | 84 | `rad0002_moe_remote_handoff` |
| `core/plugin/bands.cpp` | 1 | 2 | 423 | `rad0016_vtp14_rank_code_selection` |
| `core/sched/metrics.h` | 1 | 2 | 724 | `rad0036_prefill_governor` |
| `core/sched/scheduler.h` | 1 | 2 | 233 | `rad0036_prefill_governor` |
| `core/server/adapters.h` | 1 | 2 | 595 | `rad0036_prefill_governor` |
| `core/server/iface.h` | 1 | 2 | 826 | `rad0036_prefill_governor` |
| `core/server/metrics.h` | 1 | 2 | 860 | `rad0036_prefill_governor` |
| `abi/rad_abi.h` | 1 | 1 | 32 | `rad0048_abi44_kv_contract` |
| `arch/common/rad_block_attn_gated.h` | 1 | 1 | 223 | `rad0018_vtp14_n2_tp_ownership` |
| `arch/common/rad_block_gdn.h` | 1 | 1 | 239 | `rad0018_vtp14_n2_tp_ownership` |
| `arch/common/rad_block_mlp.h` | 1 | 1 | 229 | `rad0018_vtp14_n2_tp_ownership` |
| `core/CMakeLists.txt` | 1 | 1 | 87 | `rad0070_managed_native_entry` |
| `core/build/rad_build.h` | 1 | 1 | 50 | `rad0016_vtp14_rank_code_selection` |
| `core/kvstream/bringup.cpp` | 1 | 1 | 64 | `rad0050_b16_kv_runtime_contract` |
| `core/plugin/testplugin/CMakeLists.txt` | 1 | 1 | 189 | `rad0048_abi44_kv_contract` |
| `core/plugin/testplugin/testplugin.cc` | 1 | 1 | 45 | `rad0048_abi44_kv_contract` |
| `core/runtime/kvbind.cpp` | 1 | 1 | 85 | `rad0050_b16_kv_runtime_contract` |
| `core/sched/batch.cpp` | 1 | 1 | 113 | `rad0025_multiseq_pipelined_prefill` |
| `libr4d/r4d_ar_oneshot_2rank_exact.hip` | 1 | 1 | 303 | `rad0017_vtp14_n3_shared_rendezvous` |
| `libr4d/r4d_attn_decode_h256_gqa6.hip` | 1 | 1 | 250 | `rad0051_b16_kv_kernels` |
| `libr4d/r4d_attn_paged_h128_gqa4.hip` | 1 | 1 | 39 | `rad0052_b16_kv_geometries` |
| `libr4d/r4d_attn_prefill_h256_gqa6.hip` | 1 | 1 | 203 | `rad0051_b16_kv_kernels` |
| `libr4d/r4d_fp8_layout.cpp` | 1 | 1 | 47 | `rad0046_retained_moe_rows` |
| `libr4d/r4d_hc_bf16.hip` | 1 | 1 | 360 | `rad0008_hc_upmix_rt_sweep` |
| `libr4d/r4d_layout.cpp` | 1 | 1 | 42 | `rad0046_retained_moe_rows` |
| `libr4d/r4d_quant_act_fp8.hip` | 1 | 1 | 42 | `rad0046_retained_moe_rows` |
| `tests/plugin_test.cpp` | 1 | 1 | 63 | `rad0048_abi44_kv_contract` |
| `tests/sched_test.cpp` | 1 | 1 | 75 | `rad0036_prefill_governor` |

## Composed source memberships

| Source | Patch count |
| --- | ---: |
| `radiance` | 34 |
| `radiance-diag` | 35 |
| `radiance-native` | 0 |
| `radiance-routing-evidence` | 35 |
| `radiance-vtp44-evidence` | 38 |
