# BigCherry

BigCherry builds, patches and tests LLM inference engines against pinned
upstream releases. The main focus is **Radiance** on heterogeneous AMD
systems: RDNA2 and RDNA3 GPUs run alongside RDNA4, with remote MoE experts
distributed across cards. **llama.cpp** remains an active secondary engine.

## Engines and changes

| Engine | Upstream pin | Catalogued changes | Main additions |
| --- | --- | --- | --- |
| [Radiance](docs/reference/engines/radiance/README.md) | **v2.2.0** | **41 patches**, 6 components | RDNA2/3/4 kernels, remote experts, placement and serving manifest |
| [llama.cpp](docs/reference/engines/llamacpp/README.md) | **b11552** | **150 patches** | HIP kernel tuning, measured dispatch, multi-GPU reductions |

**Radiance.** The `libr2`, `libr3` and `bc-gfx12` components add kernels
for different AMD generations. Remote-MoE workers exchange routed expert
results with the engine over pinned host memory. The main validated engine
patches provide the remote hand-off and a HyperConn up-mix kernel selector.
The released bc-radiance 2020.8.0 line uses upstream Radiance v2.2.0 and
native Radiance YAML. The admitted configuration and device/helper plan are
recorded for replay; matched multi-card quality and performance qualification
remains outstanding.
[Components and patches](docs/reference/engines/radiance/ENHANCEMENTS.md).

**llama.cpp.** BigCherry records real HIP matrix operations, measures kernel
candidates and replays selected implementations, falling back to upstream
dispatch on a miss. Patch packages add dispatch hooks, kernel geometry
controls, adaptive AllReduce and selected AMD kernel fixes.
[Main changes](docs/reference/engines/llamacpp/README.md) ·
[Patch catalogue](docs/reference/engines/llamacpp/PATCHES.md).

Package counts include experimental, superseded and rejected work, not only
patches selected for production. Radiance has 33 validated patches;
llama.cpp has 70.

## Current-pin performance

The [Flash-Next four-GPU release gate](docs/evidence/radiance-latest-bump/RUM09-GATE-D.md)
measured engine-side prefill on Radiance v2.2.0 with a 262,144-token
context, eight sessions, matched prompts and topology, and stock power.
These are not client-output or aggregate decode rates.

| Workload | Baseline (v1.17.0), tokens/s | Best recorded (v2.2.0), tokens/s |
| --- | ---: | ---: |
| Document prefill | 3,254 | **3,662** |
| Long-7 prefill | 2,207 | **2,453** |
| Long-8 prefill | 2,099 | **2,430** |

See the [Flash-Next model record](docs/reference/engines/radiance/models/qwen3.8-flash-next.md)
for the measurement conditions and limits. The other model layouts
lack qualified current-pin output throughput; their
[27B](docs/reference/engines/radiance/models/qwen3.8-27b.md) and
[35B](docs/reference/engines/radiance/models/qwen3.6-35b-a3b.md)
records do not claim historical results as current measurements.

[Build and use](GETTING_STARTED.md) ·
[Technical reference](docs/reference/README.md) ·
[Evidence](docs/evidence/) ·
[Current designs](docs/planning/)
