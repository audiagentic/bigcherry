# BigCherry

BigCherry is a platform for building, patching and testing LLM inference engines.
It pins upstream source revisions, applies selected patches, builds engine
components and records correctness and performance results.

The current focus is a custom **Radiance** build for heterogeneous AMD GPUs:
RDNA2 and RDNA3 cards can run MoE experts alongside an RDNA4 engine GPU,
without requiring a uniform multi-GPU setup. **llama.cpp** remains actively
maintained as the secondary engine, particularly its measured HIP dispatch
and kernel tuning.

## Current engines

| Engine | Pinned upstream | BigCherry packages | Focus |
| --- | --- | --- | --- |
| [Radiance](docs/reference/engines/radiance/README.md) | **v1.14.0** | **10 patches**, 6 components | Mixed-generation AMD kernels, remote experts and serving |
| [llama.cpp](docs/reference/engines/llamacpp/README.md) | **b11552** | **151 patches** | HIP kernel tuning, dispatch replay and engine fixes |

These counts are the catalogued packages, not the number included in every
build. Radiance has 2 validated patches; llama.cpp has 70.
See the [Radiance changes](docs/reference/engines/radiance/ENHANCEMENTS.md)
and [llama.cpp patches](docs/reference/engines/llamacpp/PATCHES.md) for
individual states and inclusion rules. Pins and source selections are defined
in `engines/radiance/recipes.toml` and `config/recipes.toml`.

## Best recorded Radiance results

All throughput figures are **tokens per second**. These are distinct workloads,
not interchangeable benchmark scores.

| Model and hardware | Prefill | Generated output |
| --- | --- | --- |
| [Qwen3.8-Flash-Next](docs/reference/engines/radiance/models/qwen3.8-flash-next.md), R9700 + 2 × RX 7900 XTX + RX 6900 XT | **3,837** (fresh ~44K prompt); **2,513** (eight sessions, 1.04M tokens) | **330–342 aggregate** at 6 streams |
| [Qwen3.8-27B MXFP4](docs/reference/engines/radiance/models/qwen3.8-27b.md), R9700 | Not measured on v1.14.0 in the recorded multi-stream run | **245–248 aggregate** at 6 streams |
| [Qwen3.6-35B-A3B FP8](docs/reference/engines/radiance/models/qwen3.6-35b-a3b.md), R9700 | Not measured on v1.14.0 in the recorded decode run | **171–219** at 1 stream, depending on prompt |

Flash-Next also recorded **3,165 tok/s** for a fresh 50K prefill *after*
about one million tokens had been retained by other sessions. Its six-stream
result is engine decode throughput; the 27B figure is client aggregate
generation throughput. The 35B range is from story, code and counting
prompts. Each model page gives the exact workload and limitations.

**Stock comparison:** there is no verified like-for-like throughput run
against unmodified Radiance v1.14.0 for these model and hardware
configurations. Improvements between BigCherry layouts are recorded on the
model pages, but are not presented as stock speedups.

## How it is built

Radiance adds RDNA2, RDNA3 and RDNA4 kernel components and remote-MoE workers.
Narrow engine patches provide the integration points; the runtime combines
compute across the cards through a pinned-host mailbox. Serving
configuration is moving to a strict v1 manifest; the complete process
handoff is not yet qualified.

llama.cpp uses patch packages for upstream-owned edits and an overlay for
BigCherry-owned files. Its HIP tuner measures eligible kernels and records
winners in a replay cache; unknown shapes fall back to upstream dispatch.

The [technical reference](docs/reference/README.md) covers architecture,
builds and validation. Individual [Radiance patches and components](docs/reference/engines/radiance/ENHANCEMENTS.md),
[llama.cpp patches](docs/reference/engines/llamacpp/PATCHES.md) and the
[model results](docs/reference/engines/radiance/README.md#best-measured-models) carry the
details. For current work, see [designs](docs/planning/) and
[evidence](docs/evidence/).
