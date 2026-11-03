# Radiance

BigCherry's main target is a custom **Radiance v1.14.0** build for
heterogeneous AMD GPUs. It runs the engine on one card and can send routed
MoE experts to RDNA3 and RDNA2 helper cards. Additional kernel libraries,
helper processes and narrow engine patch packages are built against the
pinned upstream source.

## Best measured models

These runs use the **v1.14.0** pin and different model-specific workloads.
Rates are tokens/s; detailed conditions and evidence are on each model page.

| Model | Hardware | Best prefill | Generated output |
| --- | --- | --- | --- |
| [Qwen3.8-Flash-Next](models/qwen3.8-flash-next.md) | R9700 + 2 × RX 7900 XTX + RX 6900 XT | **3,837** (fresh ~44K); **2,513** (8 sessions, 1.04M tokens) | **330–342** aggregate engine decode, 6 streams |
| [Qwen3.8-27B MXFP4](models/qwen3.8-27b.md) | R9700 | Not measured on current pin | **245–248** aggregate client output, 6 streams |
| [Qwen3.6-35B-A3B FP8](models/qwen3.6-35b-a3b.md) | R9700 | Not measured on current pin | **171–219** across prompts, 1 stream |

No matched stock v1.14.0 benchmark has been retained for these workloads;
the Flash-Next comparison against fewer helper cards is a BigCherry
configuration comparison, not a stock-engine speedup.

## Components

| Component | Role |
| --- | --- |
| [libr3](../../../../engines/radiance/components/libr3/README.md) | RDNA3 kernels, including attention and quantised GEMMs |
| [libr2](../../../../engines/radiance/components/libr2/README.md) | RDNA2 expert-kernel subset |
| [bc-gfx12](../../../../engines/radiance/components/bc-gfx12/README.md) | Selected RDNA4 kernel overrides |
| [moe-remote](../../../../engines/radiance/components/moe-remote/README.md) | Expert helpers, routing, mailbox and placement |
| [serving](../../../../engines/radiance/components/serving/README.md) | Strict configuration loader, inspection and manifest generation |
| [ngram-helper](../../../../engines/radiance/components/ngram-helper/README.md) | Shelved n-gram gather experiment |

The [patch and component catalogue](ENHANCEMENTS.md) lists all nine
Radiance patch packages, their recorded states and the six components.
Only one patch is currently marked validated. A component or experimental
setting is not automatically part of the production build.

## Architecture and current limits

The engine and remote helpers exchange activations and partial outputs
through pinned host memory. Expert placement reflects each card's memory
and throughput, and large prefills use chunked helper-row combine. The
configuration is described in [layouts](LAYOUTS.md); the
[build and test reference](RUNBOOK.md) describes source selections and
measurement.

The v1 manifest contract and its C++ reader have landed, as have
`bc-serve` and the engine's manifest entry point. Full role-specific
startup and resource-plan integration are still under development; see
the [v1 specification](../../../design/serving-manifest-v1/spec.md).
Do not treat legacy launcher flags as the final configuration interface.
