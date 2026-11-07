# Radiance patches and components

Engine patches change upstream-owned source. Components are BigCherry-owned
code built against the pinned engine. Their package pages carry the technical
details, constraints and evidence; individual component settings may have
different statuses. See [the engine overview](README.md) first.

<!-- BEGIN GENERATED ENHANCEMENTS -->

_Generated from package documentation and metadata by `tools/docs/generate_engine_enhancement_summaries.py`. Do not edit this section by hand._

## Engine patches (13)

| Package | State | Summary |
| --- | --- | --- |
| [rad0001_drain_mover_before_arena_free](../../../../engines/radiance/patches/rad0001_drain_mover_before_arena_free/SUMMARY.md) | `superseded` | Drains each rank's mover stream before the activation arena is freed at shutdown. |
| [rad0002_moe_remote_handoff](../../../../engines/radiance/patches/rad0002_moe_remote_handoff/SUMMARY.md) | `validated` | Engine-side hooks (14 call sites, 4 files) for the whole-expert MoE remote hand-off. |
| [rad0003_mtp_full_head_switch](../../../../engines/radiance/patches/rad0003_mtp_full_head_switch/SUMMARY.md) | `untested` | RAD_MTP_FULL_HEAD=1 drafts MTP rounds through the trunk LM head instead of the container's 2-bit draft head. |
| [rad0004_moe_remote_diagnostics](../../../../engines/radiance/patches/rad0004_moe_remote_diagnostics/SUMMARY.md) | `untested` | Shadow-compare and mask-only diagnostics for the MoE remote hand-off, diagnostic build only (BC_MOE_REMOTE_DIAG). |
| [rad0005_mp4_ffmpeg6_writer](../../../../engines/radiance/patches/rad0005_mp4_ffmpeg6_writer/SUMMARY.md) | `untested` | Selects the FFmpeg 6 vs 7 avio_alloc_context write-callback signature so rad_core compiles on Ubuntu 24.04's FFmpeg 6.1. |
| [rad0006_heat_freeze_diagnostic](../../../../engines/radiance/patches/rad0006_heat_freeze_diagnostic/SUMMARY.md) | `untested` | RAD_DIAG_FREEZE_PROMOTION_AFTER=N stops HeatEngine expert-set swaps after N steps, diagnostic build only. |
| [rad0007_hc_pf2_fp8_down](../../../../engines/radiance/patches/rad0007_hc_pf2_fp8_down/SUMMARY.md) | `untested` | Inert scaffold for an FP8 WMMA down-product in hc_read's PF2 prefill form (gfx1201, Qwen3.8-Flash-Next). Adds three new kernels to `libr4d/r4d_hc_bf16.hip` -- none of them are called yet; this revision only proves the anchors apply and the new code compiles against the real pinned source. |
| [rad0008_hc_upmix_rt_sweep](../../../../engines/radiance/patches/rad0008_hc_upmix_rt_sweep/SUMMARY.md) | `validated` | Adds a runtime switch (`RAD_HC_UPMIX_RT`, values "128"/"256"/"384") at the PF2 up-mix launch call site inside `hc_pf2_launch` (vendor/radiance/libr4d/r4d_hc_bf16.hip, ~line 3167 as of the 1.9.0 pin). |
| [rad0009_serving_manifest](../../../../engines/radiance/patches/rad0009_serving_manifest/SUMMARY.md) | `validated` | Lets the engine start from a serving manifest v1: `engine --manifest PATH [--role engine] [-v\|-vv]`, with no command-line converter in our tooling. |
| [rad0010_routing_evidence](../../../../engines/radiance/patches/rad0010_routing_evidence/SUMMARY.md) | `untested` | Opt-in counters at the engine's existing per-layer routing-histogram producer and reader that say how many histograms were issued, completed, credited, overwritten or left pending, per rank and phase (prefill, decode, mixed, draft). Built only into the separate `radiance-routing-evidence` source; off unless `RAD_ROUTING_EVIDENCE=1`. |
| [rad0011_plan_memory](../../../../engines/radiance/patches/rad0011_plan_memory/SUMMARY.md) | `validated` | The engine's own memory plan (`Engine::plan()`) as a serving manifest v1 `EnginePlan`, measured or checked, through the shared v1 library `bc_serving_v1`. |
| [rad0012_helper_kernel_config](../../../../engines/radiance/patches/rad0012_helper_kernel_config/SUMMARY.md) | `untested` | Gives the kernel library a mandatory, size-checked, install-once configuration interface, so the helper process hands it the settings it read from the serving manifest instead of the library reading the environment. The engine keeps its own environment read until RSC07. |
| [rad0013_mtp_pass_replay](../../../../engines/radiance/patches/rad0013_mtp_pass_replay/SUMMARY.md) | `untested` | Restores recorded-pass replay for **fixed-shape decode MTP draft and speculative-verify passes** in the pinned Radiance v1.17.0 Qwen4-Exp architecture. It replaces the blanket draft/verify refusal with a guarded geometry fingerprint in `arch/qwen4exp_fp8/qwen4exp_fp8.cpp`. Does not change the core tape runtime, kernel selection or any GPU code. |

## Components (6)

| Package | State | Summary |
| --- | --- | --- |
| [bc-gfx12](../../../../engines/radiance/components/bc-gfx12/README.md) | *See component settings* | BigCherry's RDNA4 kernel component supplies targeted Radiance kernel rows alongside the upstream library. The engine uses a BigCherry row where its constraints match and otherwise keeps the upstream implementation. |
| [libr2](../../../../engines/radiance/components/libr2/README.md) | *See component settings* | A subset of Radiance device kernels adapted for RDNA2, built against the pinned engine sources. It currently supplies expert computation and activation quantisation for the remote-MoE worker; it is not a complete replacement for Radiance's upstream kernel library. |
| [libr3](../../../../engines/radiance/components/libr3/README.md) | *See component settings* | BigCherry's gfx1100 (RX 7900 XTX) kernel plugin for radiance (RAD01 PR 5): native WMMA forms for the GEMMs, attention and all-reduce, built against the pinned engine's libr4d sources via `rewrites.txt` + `native/*.rw` and loaded through `RADIANCE_HOME`. It is BigCherry source compiled against the installed engine, not a patch to the engine itself. |
| [moe-remote](../../../../engines/radiance/components/moe-remote/README.md) | *See component settings* | A separate `r3_moe_worker` process per helper card owns routed experts and computes their rows over a pinned-host mailbox with GPU-side signals. `libr3moe` is the engine-side plugin for the production `rad0002_moe_remote_handoff` hooks; `libr3` still owns the engine's kernel overrides. |
| [ngram-helper](../../../../engines/radiance/components/ngram-helper/README.md) | *See component settings* | Opt-in n-gram row gather on a non-rank card (`R3_NGRAM_DEV`), split out of `libr3` (RCS06). |
| [serving](../../../../engines/radiance/components/serving/README.md) | *See component settings* | The `bc_serving` C++ library and `bc-serve` tool provide a strict typed TOML loader, hardware detection, configuration validation and a canonical serving manifest with provenance. |

<!-- END GENERATED ENHANCEMENTS -->
