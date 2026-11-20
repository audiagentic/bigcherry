# Radiance

Radiance **v1.17.0** is BigCherry's primary upstream target. The custom
build uses RDNA2 and RDNA3 helpers alongside an RDNA4 engine GPU, with
expert placement adapted to differing GPU memory and bandwidth.

## Patches and extensions at a glance

BigCherry maintains **40 engine patch packages** and **6 components** for Radiance.
Each description says what the package is for, not that it is enabled or
performance-qualified. **Validated** means package-level evidence exists;
**untested** means qualification is not established; **diagnostic** is for
observability rather than production; **superseded** means a
newer patch replaces it. See the [full catalogue](ENHANCEMENTS.md) and
linked package pages for technical details and evidence.

### Engine patches

Patches modify upstream Radiance source.

| Patch | Simple purpose | State |
| --- | --- | --- |
| [rad0001_drain_mover_before_arena_free](../../../../engines/radiance/patches/archive/rad0001_drain_mover_before_arena_free/SUMMARY.md) | Drain GPU transfers before freeing memory | `superseded` |
| [rad0002_moe_remote_handoff](../../../../engines/radiance/patches/rad0002_moe_remote_handoff/SUMMARY.md) | Send expert work to helper GPUs | `validated` |
| [rad0003_mtp_full_head_switch](../../../../engines/radiance/patches/archive/rad0003_mtp_full_head_switch/SUMMARY.md) | Optionally draft with the full model head | `untested` |
| [rad0004_moe_remote_diagnostics](../../../../engines/radiance/patches/rad0004_moe_remote_diagnostics/SUMMARY.md) | Diagnose remote expert routing | `diagnostic` |
| [rad0005_mp4_ffmpeg6_writer](../../../../engines/radiance/patches/rad0005_mp4_ffmpeg6_writer/SUMMARY.md) | Support FFmpeg 6 builds | `validated` |
| [rad0006_heat_freeze_diagnostic](../../../../engines/radiance/patches/archive/rad0006_heat_freeze_diagnostic/SUMMARY.md) | Freeze expert movement for diagnosis | `superseded` |
| [rad0007_hc_pf2_fp8_down](../../../../engines/radiance/patches/archive/rad0007_hc_pf2_fp8_down/SUMMARY.md) | Unused experimental FP8 HyperConn kernels | `untested` |
| [rad0008_hc_upmix_rt_sweep](../../../../engines/radiance/patches/rad0008_hc_upmix_rt_sweep/SUMMARY.md) | Tune HyperConn launch size | `validated` |
| [rad0009_serving_manifest](../../../../engines/radiance/patches/rad0009_serving_manifest/SUMMARY.md) | Start engine from a serving manifest | `validated` |
| [rad0010_routing_evidence](../../../../engines/radiance/patches/rad0010_routing_evidence/SUMMARY.md) | Record expert-routing counters | `diagnostic` |
| [rad0011_plan_memory](../../../../engines/radiance/patches/rad0011_plan_memory/SUMMARY.md) | Expose engine memory plan | `validated` |
| [rad0012_helper_kernel_config](../../../../engines/radiance/patches/rad0012_helper_kernel_config/SUMMARY.md) | Pass settings to helper kernels | `validated` |
| [rad0013_mtp_pass_replay](../../../../engines/radiance/patches/rad0013_mtp_pass_replay/SUMMARY.md) | Replay speculative draft and verify passes | `validated` |
| [rad0016_vtp14_rank_code_selection](../../../../engines/radiance/patches/rad0016_vtp14_rank_code_selection/SUMMARY.md) | Select GPU-specific kernels per TP rank | `validated` |
| [rad0017_vtp14_n3_shared_rendezvous](../../../../engines/radiance/patches/rad0017_vtp14_n3_shared_rendezvous/SUMMARY.md) | Synchronise mixed-GPU ranks | `validated` |
| [rad0018_vtp14_n2_tp_ownership](../../../../engines/radiance/patches/rad0018_vtp14_n2_tp_ownership/SUMMARY.md) | Split model weights unevenly across GPUs | `validated` |
| [rad0019_vtp14_n2_tp_ownership_fp8](../../../../engines/radiance/patches/rad0019_vtp14_n2_tp_ownership_fp8/SUMMARY.md) | Uneven FP8 weight splits | `validated` |
| [rad0020_vtp14_tp3_pipelined_prefill](../../../../engines/radiance/patches/rad0020_vtp14_tp3_pipelined_prefill/SUMMARY.md) | Overlap computation and communication in prefill | `validated` |
| [rad0021_vtp14_weighted_twoshot_3rank](../../../../engines/radiance/patches/rad0021_vtp14_weighted_twoshot_3rank/SUMMARY.md) | Reduce traffic on slow PCIe links | `validated` |
| [rad0022_tune_pin](../../../../engines/radiance/patches/rad0022_tune_pin/SUMMARY.md) | Validate and pin kernel tuning | `validated` |
| [rad0023_ar_timeline](../../../../engines/radiance/patches/rad0023_ar_timeline/SUMMARY.md) | Trace one-shot reduction timings | `validated` |
| [rad0024_ts3_decode](../../../../engines/radiance/patches/rad0024_ts3_decode/SUMMARY.md) | Link-aware three-GPU decode reductions | `validated` |
| [rad0025_multiseq_pipelined_prefill](../../../../engines/radiance/patches/rad0025_multiseq_pipelined_prefill/SUMMARY.md) | Pipeline multi-prompt prefill | `validated` |
| [rad0026_mixed_pipelined_prefill](../../../../engines/radiance/patches/rad0026_mixed_pipelined_prefill/SUMMARY.md) | Pipeline mixed decode and multi-prompt prefill | `validated` |
| [rad0027_mixed_one_pipelined_prefill](../../../../engines/radiance/patches/rad0027_mixed_one_pipelined_prefill/SUMMARY.md) | Pipeline decode alongside one prefill | `validated` |
| [rad0028_rank_consistent_block_drafts](../../../../engines/radiance/patches/rad0028_rank_consistent_block_drafts/SUMMARY.md) | Keep draft decisions consistent across ranks | `validated` |
| [rad0029_rank_own_split](../../../../engines/radiance/patches/rad0029_rank_own_split/SUMMARY.md) | Per-GPU weight and KV memory budgets | `validated` |
| [rad0030_int8_pipelined_prefill](../../../../engines/radiance/patches/rad0030_int8_pipelined_prefill/SUMMARY.md) | Pipeline prefill for rotated INT8 | `validated` |
| [rad0031_ts_timeline](../../../../engines/radiance/patches/rad0031_ts_timeline/SUMMARY.md) | Trace two-shot reduction timings | `validated` |
| [rad0032_ts3_duplex](../../../../engines/radiance/patches/rad0032_ts3_duplex/SUMMARY.md) | Use both link directions in reductions | `validated` |
| [rad0033_ts3_once](../../../../engines/radiance/patches/rad0033_ts3_once/SUMMARY.md) | Avoid duplicate reduction writes | `validated` |
| [rad0034_prefill_concurrency](../../../../engines/radiance/patches/archive/rad0034_prefill_concurrency/SUMMARY.md) | Earlier long-prefill concurrency limit | `superseded` |
| [rad0035_prefill_order](../../../../engines/radiance/patches/archive/rad0035_prefill_order/SUMMARY.md) | Earlier prefill ordering and ageing controls | `superseded` |
| [rad0036_prefill_governor](../../../../engines/radiance/patches/rad0036_prefill_governor/SUMMARY.md) | Unified prefill admission and ordering | `validated` |

| [rad0037_bf16_fp8_kv](../../../../engines/radiance/patches/rad0037_bf16_fp8_kv/SUMMARY.md) | Keep BF16 keys; compress cache values to FP8 | `validated` |
| [rad0038_bf16_i8_kv](../../../../engines/radiance/patches/rad0038_bf16_i8_kv/SUMMARY.md) | Keep BF16 keys; quantise cache values to scaled INT8 | `validated` |

| [rad0039_moe_heat_observer](../../../../engines/radiance/patches/archive/rad0039_moe_heat_observer/SUMMARY.md) | Observe expert usage without changing placement; inactive | `untested` |
| [rad0040_kv_layout_table](../../../../engines/radiance/patches/rad0040_kv_layout_table/SUMMARY.md) | Keep packed KV-cache layout definitions in one table | `validated` |
| [rad0041_kv_b16_flashnext](../../../../engines/radiance/patches/rad0041_kv_b16_flashnext/SUMMARY.md) | Add packed BF16-key KV-cache kernels for Flash-Next | `validated` |
| [rad0042_kv_b16_selftest](../../../../engines/radiance/patches/rad0042_kv_b16_selftest/SUMMARY.md) | Add numeric selftests for Flash-Next packed KV caches | `untested` |

### Components (extensions)

Components are BigCherry-owned additions built alongside Radiance.

| Component | Simple purpose |
| --- | --- |
| [bc-gfx12](../../../../engines/radiance/components/bc-gfx12/README.md) | Selected RDNA4-optimised kernels |
| [libr2](../../../../engines/radiance/components/libr2/README.md) | RDNA2 expert computation and quantisation |
| [libr3](../../../../engines/radiance/components/libr3/README.md) | RDNA3 matrix, attention and reduction kernels |
| [moe-remote](../../../../engines/radiance/components/moe-remote/README.md) | Runs selected experts on helper GPUs |
| [ngram-helper](../../../../engines/radiance/components/ngram-helper/README.md) | Optional n-gram work on a spare GPU; off by default |
| [serving](../../../../engines/radiance/components/serving/README.md) | Serving configuration validation and manifests |

Individual component settings have their own qualification states. The validated
MTP replay patch supports fixed-shape draft and verify passes on v1.17.0.
Mixed-GPU remote-MoE layouts lack matched current-pin end-to-end throughput
and quality evidence here. Full manifest-driven serving startup remains under
qualification.

## Current-pin throughput

The pinned upstream is Radiance **v1.17.0**. Fresh current-pin aggregate throughput and
prefill measurements are not yet qualified for these layouts. The Flash-Next model page
records a dated cross-pin check, not a matched unmodified-stock throughput comparison.
The model workloads differ and must not be ranked against one another.

| Model and layout | Prefill tokens/s | Aggregate output tokens/s |
| --- | ---: | ---: |
| [Flash-Next](models/qwen3.8-flash-next.md), R9700 + two RX 7900 XTX + RX 6900 XT | Not qualified | Not qualified |
| [Qwen3.8-27B MXFP4](models/qwen3.8-27b.md), R9700; two-XTX alternative unqualified | Not qualified | Not qualified |
| [Qwen3.6-35B-A3B FP8](models/qwen3.6-35b-a3b.md), R9700 | Not qualified | Not qualified |

Earlier v1.14.0 results, including long-context and concurrent serving
measurements, are retained in the [archive](../../../archive/radiance/v1.14-model-results/).
The Flash-Next four-card layout has a documented quality trade-off.

## Further detail

The [build and serving runbook](RUNBOOK.md) covers composition and
measurement; [GPU layouts](LAYOUTS.md) covers helper card budgets;
[serving manifest](SERVING_MANIFEST.md) covers writing a manifest, starting with one and the `--set` key to engine flag table.
Manifest startup is still being qualified against the
[v2 specification](../../../design/serving-manifest-v2/spec.md).
