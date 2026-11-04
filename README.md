# BigCherry

BigCherry builds, patches and tests LLM inference engines against pinned
upstream releases. The main focus is **Radiance** on heterogeneous AMD
systems: RDNA2 and RDNA3 GPUs run alongside RDNA4, with remote MoE experts
distributed across cards. **llama.cpp** remains an active secondary engine.

## Engines and changes

| Engine | Upstream pin | Catalogued changes | Main additions |
| --- | --- | --- | --- |
| [Radiance](docs/reference/engines/radiance/README.md) | **v1.16.0** | **10 patches**, 6 components | RDNA2/3/4 kernels, remote experts, placement and serving manifest |
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
patches selected for production. Radiance has 2 validated patches;
llama.cpp has 70.

## Best recorded Radiance throughput

These results were measured on **v1.14.0**, not yet revalidated on the
current v1.16.0 pin. Rates are tokens per second, with different workloads.

| Model | Prefill | Generated output |
| --- | --- | --- |
| [Qwen3.8-Flash-Next](docs/reference/engines/radiance/models/qwen3.8-flash-next.md), four AMD cards | **3,837** (fresh ~44K prompt) | **330–342**, 6-stream aggregate *engine decode* |
| [Qwen3.8-27B](docs/reference/engines/radiance/models/qwen3.8-27b.md), R9700 | — | **245–248**, 6-stream aggregate *client output* |
| [Qwen3.6-35B-A3B](docs/reference/engines/radiance/models/qwen3.6-35b-a3b.md), R9700 | — | **171–219**, single-stream client output across prompts |

There is no matched stock-engine comparison for these workloads. The
four-card Flash-Next layout also has a documented quality trade-off; see
the model page before using its throughput as a configuration choice.

[Build and use](GETTING_STARTED.md) ·
[Technical reference](docs/reference/README.md) ·
[Evidence](docs/evidence/) ·
[Current designs](docs/planning/)
