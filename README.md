# BigCherry

BigCherry builds, patches and tests LLM inference engines against pinned
upstream releases. The main focus is **Radiance** on heterogeneous AMD
systems: RDNA2 and RDNA3 GPUs run alongside RDNA4, with remote MoE experts
distributed across cards. **llama.cpp** remains an active secondary engine.

## Engines and changes

| Engine | Upstream pin | Catalogued changes | Main additions |
| --- | --- | --- | --- |
| [Radiance](docs/reference/engines/radiance/README.md) | **v1.17.0** | **15 patches**, 6 components | RDNA2/3/4 kernels, remote experts, placement and serving manifest |
| [llama.cpp](docs/reference/engines/llamacpp/README.md) | **b11552** | **150 patches** | HIP kernel tuning, measured dispatch, multi-GPU reductions |

**Radiance.** The `libr2`, `libr3` and `bc-gfx12` components add kernels
for different AMD generations. Remote-MoE workers exchange routed expert
results with the engine over pinned host memory. The main validated engine
patches provide the remote hand-off and a HyperConn up-mix kernel selector.
The strict serving loader and manifest are being integrated; full
manifest-driven startup is not yet qualified.
[Components and patches](docs/reference/engines/radiance/ENHANCEMENTS.md).

**llama.cpp.** BigCherry records real HIP matrix operations, measures kernel
candidates and replays selected implementations, falling back to upstream
dispatch on a miss. Patch packages add dispatch hooks, kernel geometry
controls, adaptive AllReduce and selected AMD kernel fixes.
[Main changes](docs/reference/engines/llamacpp/README.md) ·
[Patch catalogue](docs/reference/engines/llamacpp/PATCHES.md).

Package counts include experimental, superseded and rejected work, not only
patches selected for production. Radiance has 5 validated patches;
llama.cpp has 70.

## Current-pin throughput

Fresh aggregate output and prefill throughput on the current **v1.17.0** pin is not yet
qualified for these layouts. The Flash-Next model page records a dated cross-pin
check, not a matched unmodified-stock throughput A/B. Earlier results remain archived.

| Model and layout | Prefill tokens/s | Aggregate output tokens/s |
| --- | ---: | ---: |
| [Flash-Next](docs/reference/engines/radiance/models/qwen3.8-flash-next.md), R9700 + two RX 7900 XTX + RX 6900 XT | Not qualified | Not qualified |
| [Qwen3.8-27B MXFP4](docs/reference/engines/radiance/models/qwen3.8-27b.md), R9700; two-XTX alternative unqualified | Not qualified | Not qualified |
| [Qwen3.6-35B-A3B FP8](docs/reference/engines/radiance/models/qwen3.6-35b-a3b.md), R9700 | Not qualified | Not qualified |

Earlier v1.14.0 results, including long-context and concurrent serving
measurements, are retained in the [archive](docs/archive/radiance/v1.14-model-results/).
The Flash-Next four-card layout has a documented quality trade-off.

[Build and use](GETTING_STARTED.md) ·
[Technical reference](docs/reference/README.md) ·
[Evidence](docs/evidence/) ·
[Current designs](docs/planning/)
