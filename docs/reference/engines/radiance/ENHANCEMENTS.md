# Radiance patches and components

Engine patches change upstream-owned source. Components are BigCherry-owned
code built against the pinned engine. Their package pages carry the technical
details, constraints and evidence; individual component settings may have
different statuses. See [the engine overview](README.md) first.

<!-- BEGIN GENERATED ENHANCEMENTS -->

_Generated from package documentation and metadata by `tools/docs/generate_engine_enhancement_summaries.py`. Do not edit this section by hand._

## Engine patches (10)

| Package | State | Summary |
| --- | --- | --- |
| [rad0001_drain_mover_before_arena_free](../../../../engines/radiance/patches/rad0001_drain_mover_before_arena_free/SUMMARY.md) | `superseded` | Drains each rank's mover stream before the activation arena is freed at shutdown. |
| [rad0002_moe_remote_handoff](../../../../engines/radiance/patches/rad0002_moe_remote_handoff/SUMMARY.md) | `untested` | Engine-side hooks (14 call sites, 4 files) for the whole-expert MoE remote hand-off. |
| [rad0003_mtp_full_head_switch](../../../../engines/radiance/patches/rad0003_mtp_full_head_switch/SUMMARY.md) | `untested` | RAD_MTP_FULL_HEAD=1 drafts MTP rounds through the trunk LM head instead of the container's 2-bit draft head. |
| [rad0004_moe_remote_diagnostics](../../../../engines/radiance/patches/rad0004_moe_remote_diagnostics/SUMMARY.md) | `untested` | Shadow-compare and mask-only diagnostics for the MoE remote hand-off, diagnostic build only (BC_MOE_REMOTE_DIAG). |
| [rad0005_mp4_ffmpeg6_writer](../../../../engines/radiance/patches/rad0005_mp4_ffmpeg6_writer/SUMMARY.md) | `untested` | Selects the FFmpeg 6 vs 7 avio_alloc_context write-callback signature so rad_core compiles on Ubuntu 24.04's FFmpeg 6.1. |
| [rad0006_heat_freeze_diagnostic](../../../../engines/radiance/patches/rad0006_heat_freeze_diagnostic/SUMMARY.md) | `untested` | RAD_DIAG_FREEZE_PROMOTION_AFTER=N stops HeatEngine expert-set swaps after N steps, diagnostic build only. |
| [rad0007_hc_pf2_fp8_down](../../../../engines/radiance/patches/rad0007_hc_pf2_fp8_down/SUMMARY.md) | `untested` | Inert scaffold for an FP8 WMMA down-product in hc_read's PF2 prefill form (gfx1201, Qwen3.8-Flash-Next). Adds three new kernels to `libr4d/r4d_hc_bf16.hip` -- none of them are called yet; this revision only proves the anchors apply and the new code compiles against the real pinned source. |
| [rad0008_hc_upmix_rt_sweep](../../../../engines/radiance/patches/rad0008_hc_upmix_rt_sweep/SUMMARY.md) | `validated` | Adds a runtime switch (`RAD_HC_UPMIX_RT`, values "128"/"256"/"384") at the PF2 up-mix launch call site inside `hc_pf2_launch` (vendor/radiance/libr4d/r4d_hc_bf16.hip, ~line 3167 as of the 1.9.0 pin). |
| [rad0009_serving_manifest](../../../../engines/radiance/patches/rad0009_serving_manifest/SUMMARY.md) | `untested` | Lets the engine start from a resolved serving manifest: `engine --manifest PATH [--role engine] [-v\|-vv]`, with no command-line converter in our tooling. |
| [rad0011_plan_memory](../../../../engines/radiance/patches/rad0011_plan_memory/SUMMARY.md) | `untested` | The engine's own memory plan (`Engine::plan()`) as a serving manifest v1 `EnginePlan`, measured or checked, through the shared v1 library `bc_serving_v1`. |

## Components (6)

| Package | State | Summary |
| --- | --- | --- |
| [bc-gfx12](../../../../engines/radiance/components/bc-gfx12/README.md) | *See component settings* | BigCherry's RDNA4 kernel component supplies targeted Radiance kernel rows alongside the upstream library. The engine uses a BigCherry row where its constraints match and otherwise keeps the upstream implementation. |
| [libr2](../../../../engines/radiance/components/libr2/README.md) | *See component settings* | A subset of Radiance device kernels adapted for RDNA2, built against the pinned engine sources. It currently supplies expert computation and activation quantisation for the remote-MoE worker; it is not a complete replacement for Radiance's upstream kernel library. |
| [libr3](../../../../engines/radiance/components/libr3/README.md) | *See component settings* | BigCherry's gfx1100 (RX 7900 XTX) kernel plugin for radiance (RAD01 PR 5): native WMMA forms for the GEMMs, attention and all-reduce, built against the pinned engine's libr4d sources via `rewrites.txt` + `native/*.rw` and loaded through `RADIANCE_HOME`. It is BigCherry source compiled against the installed engine, not a patch to the engine itself. |
| [moe-remote](../../../../engines/radiance/components/moe-remote/README.md) | *See component settings* | The remote-MoE hand-off (RAD02, RCS04): a separate `r3_moe_worker` process per card owns whole routed experts and computes them for the engine over a pinned- host mailbox with GPU-side flags; `libr3moe` is the engine-side mailbox plugin that fills the ops engine patch `rad0002_moe_remote_handoff` declares. Split out of `libr3`, which keeps the engine kernel overrides. |
| [ngram-helper](../../../../engines/radiance/components/ngram-helper/README.md) | *See component settings* | Opt-in n-gram row gather on a non-rank card (`R3_NGRAM_DEV`), split out of `libr3` (RCS06). |
| [serving](../../../../engines/radiance/components/serving/README.md) | *See component settings* | The `bc_serving` C++ library and `bc-serve` tool provide a strict typed TOML loader, hardware detection, configuration validation and a canonical serving manifest with provenance. |

<!-- END GENERATED ENHANCEMENTS -->
