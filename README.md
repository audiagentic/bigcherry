# BigCherry

BigCherry's primary work is a custom [Radiance](https://codeberg.org/StillDeadcode/radiance)
build for heterogeneous AMD GPU systems. The aim is to make RDNA2 and RDNA3
cards useful alongside newer hardware, rather than requiring every GPU to be
the same generation. That means additional kernels, remote MoE expert
execution, and placement and memory policies suited to mixed cards and
uneven PCIe links.

[llama.cpp](https://github.com/ggml-org/llama.cpp) remains under active
development in BigCherry, but is the secondary engine for now.

## Radiance: what changes

| Hardware in the measured layout | Role | BigCherry extension |
| --- | --- | --- |
| Radeon PRO R9700 (RDNA4) | Main engine | Radiance kernels, model execution and routed-expert hand-off |
| 2 × RX 7900 XTX (RDNA3) | Expert helpers | `libr3` kernels and remote-MoE workers |
| RX 6900 XT (RDNA2) | Additional expert helper | `libr2` kernel subset and remote-MoE worker |

The engine and helpers exchange routed activations and results through a
pinned-host mailbox. This matters on consumer boards without working GPU
peer-to-peer transfers. Expert placement, per-card capacity, helper kernels
and host-memory use are being tuned together. BigCherry carries these
changes as separate components and narrow upstream patches, not as a
permanent Radiance fork.

## Selected measured results

Qwen3.8-Flash-Next (113.6 GiB container, 512 experts, top-10 routing).
These are results from the recorded four-card layout, not general hardware
benchmarks.

| Workload | Earlier configuration | Measured result |
| --- | --- | --- |
| Eight sessions, 1.04M prompt tokens in total | R9700 + two XTX helpers: **1,353 tok/s** | Add the RDNA2 helper with row combine: **2,369 tok/s** |
| New 50K-token prefill after ~1M retained session tokens | No host KV tier: **1,901 tok/s** | 9 GiB host pool + 12 GiB KV tier: **3,165 tok/s** |
| Six-stream decode, 512-token prompts | Grouped helper GEMM: **55.1–56.3 ms/step** | Packed-f16 W4 GEMV: **45.9–48.5 ms/step** |
| Fresh-server prefill, roughly 44–49K tokens | — | **3,825–3,837 tok/s** |

The first row compares layouts; the second isolates retained-session memory
pressure; the third is decode step latency, not generated tokens per second.
These measurements were recorded on 1–2 November 2026. See the
[Flash-Next results and controls](docs/reference/engines/radiance/models/qwen3.8-flash-next.md)
for run conditions, quality checks and known limits.

## Other work

**Radiance** also has architecture-specific attention and GEMM kernels,
a strict C++ serving-configuration loader and an engine manifest entry
point. Helper manifest loading and managed process startup are still being
implemented. See the [Radiance overview](docs/reference/engines/radiance/README.md)
and [patch and component catalogue](docs/reference/engines/radiance/ENHANCEMENTS.md).

**llama.cpp** has measured HIP dispatch: record operation shapes, tune
eligible kernels, and replay selected winners using a compact cache.
Unmatched operations retain upstream dispatch. It also has individually
versioned engine patches. See the
[llama.cpp overview](docs/reference/engines/llamacpp/README.md) and
[patch catalogue](docs/reference/engines/llamacpp/PATCHES.md).

## Documentation

- [Build and use](GETTING_STARTED.md) — BigCherry commands and runtime modes.
- [Technical reference](docs/reference/README.md) — kernel architecture,
  dispatch, validation and build details.
- [Current designs](docs/planning/) and [recorded evidence](docs/evidence/)
  — active work and measurements.
