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

## Measured throughput

Qwen3.8-Flash-Next, 113.6 GiB (512 experts, top-10 routing). The measured
system uses an R9700 engine, two RX 7900 XTX helpers and an RX 6900 XT
helper. Results below are from 31 October–2 November 2026.

### Prefill

| Workload | Baseline | Best recorded |
| --- | ---: | ---: |
| Eight sessions, 1.04M prompt tokens in total | Two XTX helpers: **1,353 tok/s** | Three helpers, chunked combine: **2,513 tok/s** |
| Eight-session cold prefill after ~1M tokens retained | No host KV tier: **2,499 tok/s** | 12 GiB host KV tier: **2,945 tok/s** |
| New 50K-token prefill after ~1M tokens retained | No host KV tier: **1,901 tok/s** | 12 GiB host KV tier: **3,165 tok/s** |
| Fresh-server prefill, ~44K tokens | — | **3,837 tok/s** |
| Extended context, ~340K tokens | Unchunked combine: **3,089 tok/s** | Chunked combine: **3,397 tok/s** |
| Extended context, ~650–690K tokens | Unchunked combine: **2,663 tok/s** | Chunked combine: **2,796 tok/s** |

### Multi-stream decode

Aggregate *engine decode* throughput on the four-card layout, using the W4H2
helper GEMV. The workload uses 512-token prompts, 512 generated tokens and
three MTP draft tokens.

| Concurrent streams | Engine decode |
| ---: | ---: |
| 1 | **144–146 tok/s** |
| 2 | **176–201 tok/s** |
| 4 | **249–255 tok/s** |
| 6 | **330–342 tok/s** |

The prefill figures cover different conditions: cold sessions, retained KV
state, and extended-context prompts. The eight-session layout comparison
spans successive improvements rather than a single controlled A/B; the
~650–690K comparison also uses slightly different prompt lengths. Decode
throughput depends on draft acceptance and is not the same as step latency
or per-stream output speed. See the
[Flash-Next measurements](docs/reference/engines/radiance/models/qwen3.8-flash-next.md)
for test conditions, quality checks and individual runs.

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
