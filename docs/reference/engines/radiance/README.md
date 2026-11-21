# Radiance

The bump branch pins Radiance **v2.2.0**. Released v1.17.0 builds provide the
comparison baseline while this candidate is being qualified. The custom build uses RDNA2 and RDNA3 helpers alongside an RDNA4 engine GPU, with
expert placement adapted to differing GPU memory and bandwidth.

## Patches and extensions at a glance

BigCherry maintains **45 current engine patch packages** and **6 components** for Radiance.
Each description says what the package is for, not that it is enabled or
performance-qualified. **Validated** means package-level evidence exists;
**untested** means qualification is not established; **diagnostic** is for
observability rather than production; **superseded** means a
newer patch replaces it. See the [full catalogue](ENHANCEMENTS.md) and
linked package pages for technical details and evidence.

### Engine patches

The [generated catalogue](ENHANCEMENTS.md) lists every current package, including its state, scope, and evidence links.

### Archived patches

Retired patches are kept unchanged as evidence under [engines/radiance/patches/archive](../../../../engines/radiance/patches/archive/ARCHIVE.md) (reason, successor and evidence per patch in ARCHIVE.md); they are never built or composed.

| Package | What it was for | Final state |
|---|---|---|
| [rad0001_drain_mover_before_arena_free](../../../../engines/radiance/patches/archive/rad0001_drain_mover_before_arena_free/SUMMARY.md) | Drain GPU transfers before freeing memory | `superseded` |
| [rad0003_mtp_full_head_switch](../../../../engines/radiance/patches/archive/rad0003_mtp_full_head_switch/SUMMARY.md) | Optionally draft with the full model head | `rejected` |
| [rad0006_heat_freeze_diagnostic](../../../../engines/radiance/patches/archive/rad0006_heat_freeze_diagnostic/SUMMARY.md) | Freeze expert movement for diagnosis | `superseded` |
| [rad0007_hc_pf2_fp8_down](../../../../engines/radiance/patches/archive/rad0007_hc_pf2_fp8_down/SUMMARY.md) | Unused experimental FP8 HyperConn kernels | `rejected` |
| [rad0034_prefill_concurrency](../../../../engines/radiance/patches/archive/rad0034_prefill_concurrency/SUMMARY.md) | Earlier long-prefill concurrency limit | `superseded` |
| [rad0035_prefill_order](../../../../engines/radiance/patches/archive/rad0035_prefill_order/SUMMARY.md) | Earlier prefill ordering and ageing controls | `superseded` |
| [rad0039_moe_heat_observer](../../../../engines/radiance/patches/archive/rad0039_moe_heat_observer/SUMMARY.md) | Observe expert usage without changing placement; inactive | `untested` |

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

The **v2.2.0** candidate is undergoing matched quality and performance checks.
Fresh aggregate throughput and prefill measurements are pending for these layouts. The Flash-Next model page
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
[serving manifest](SERVING_MANIFEST.md) covers how the owner makes a manifest from a native config and how the engine starts from it.
Manifest startup is qualified against the
[serving manifest specification](../../../design/serving-manifest-v2/spec.md).
