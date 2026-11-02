# Radiance

BigCherry adds AMD kernel implementations, multi-GPU expert execution and
narrow engine integration patches to
[Radiance](https://codeberg.org/StillDeadcode/radiance). The pinned engine
version is declared in `engines/radiance/recipes.toml`; BigCherry's kernel
source is built separately and installed into the engine's runtime bundle.

## Extensions

| Component | Purpose |
| --- | --- |
| [libr3](../../../../engines/radiance/components/libr3/README.md) | RDNA3 device kernels built against Radiance's kernel sources |
| [libr2](../../../../engines/radiance/components/libr2/README.md) | RDNA2-compatible kernel subset, used by remote experts |
| [bc-gfx12](../../../../engines/radiance/components/bc-gfx12/README.md) | Focused RDNA4 kernel overrides where measured |
| [moe-remote](../../../../engines/radiance/components/moe-remote/README.md) | Remote MoE execution, mailbox and helper worker |
| [serving](../../../../engines/radiance/components/serving/README.md) | Strict serving schema, configuration inspection and manifest generation (RSC01) |
| [ngram-helper](../../../../engines/radiance/components/ngram-helper/README.md) | Optional n-gram gather experiment, currently shelved |

Engine modifications are separate from these components. See the
[patch and extension catalog](ENHANCEMENTS.md) for every package's recorded
state and its owning documentation page. A component's settings may have
different qualification states: read its `component.toml` and evidence before
enabling an experimental path.

## Model-specific results

- [Qwen3.8-Flash-Next](models/qwen3.8-flash-next.md) — routed experts and
  multi-GPU serving.
- [Qwen3.8-27B](models/qwen3.8-27b.md) — dense inference.
- [Qwen3.6-35B-A3B](models/qwen3.6-35b-a3b.md) — MoE inference.

These pages record dated results for particular model and hardware layouts;
they are not interchangeable performance guarantees.

## Build, measurement and serving

The [runbook](RUNBOOK.md) covers the build and measurement workflow.
[Platform design](PLATFORM_DESIGN.md) records the original engine/component
split; the [lab history](../../../../engines/radiance/README.md) retains
dated experiments.

**Serving configuration is being migrated to a manifest.** The strict C++
schema, loader and `bc-serve` inspection commands (RSC01), and the engine
manifest entry point (RSC07), have landed. Helper manifest consumption (RSC06)
and managed process start-up (RSC02) are still in progress. See the
[serving component](../../../../engines/radiance/components/serving/README.md)
and the [serving-configuration plan](../../../planning/active/run-serving-configuration/README.md).
The older launch instructions in the runbook and lab scripts are not the final
serving interface.
