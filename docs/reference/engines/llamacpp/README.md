# llama.cpp

BigCherry maintains a **b11552**-pinned llama.cpp build alongside Radiance.
It concentrates on AMD HIP kernel selection, measured dispatch and selected
multi-GPU improvements. Upstream-owned edits are applied as patch packages;
new source files live in the overlay.

## Main additions and patches

| Area | BigCherry change | Key packages |
| --- | --- | --- |
| Measured HIP dispatch | Record actual matrix-operation shapes, tune eligible kernels, replay winners from a cache; use native dispatch on a miss | [Dispatch hook](../../../../engines/llamacpp/patches/0200_dispatch_hook/README.md), [record/tune build](../../../../engines/llamacpp/patches/0110_campaign_tune_record_build/README.md) |
| Kernel selection | Explicit launch geometry for quantised and floating-point matrix operations without replacing upstream's fallback path | [MMQ](../../../../engines/llamacpp/patches/0300_mmq_forced_j/README.md), [MMVF](../../../../engines/llamacpp/patches/0400_mmvf_forced_block/README.md), [MMF](../../../../engines/llamacpp/patches/0500_mmf_forced_nwarps/README.md), [MMVQ](../../../../engines/llamacpp/patches/0600_mmvq_geometry/README.md) |
| Multi-GPU reductions | Choose internal or RCCL AllReduce by message size, keeping decode and prefill on different paths when useful | [Adaptive AllReduce](../../../../engines/llamacpp/patches/0840_hybrid_allreduce_dispatch/README.md), [provider selector](../../../../engines/llamacpp/patches/0860_allreduce_provider_cli/README.md) |
| AMD-specific improvements | Individually measured and gated kernel and MoE fixes alongside experiments that are not part of the default build | [Example: MoE MMQ compact grid](../../../../engines/llamacpp/patches/1237_rd30_moe_mmq_compact_grid/README.md) |

There are **150 patch packages** excluding the authoring template;
**70** are recorded as validated. A validated package is not necessarily
selected in every build. The [complete patch catalogue](PATCHES.md)
shows states and links to each change.

## Build and measurement

The engine composition is defined in `config/recipes.toml`:
source patch sets, build profiles and isolated experiments. A replay
build consumes promoted winners from a cache, rather than benchmarking
during inference.

See the [dispatch architecture](../../architecture/MMQ_CATALOG.md),
[patch system](../../patches/PATCH_SYSTEM.md) and
[build reference](../../build/BUILD.md) for the implementation and
validation boundaries.
