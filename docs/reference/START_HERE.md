# BigCherry technical reference

Radiance is BigCherry's primary engine target: a custom build for mixed
AMD GPUs, including RDNA2 and RDNA3 helper support alongside RDNA4.
llama.cpp remains active as the secondary engine.

## Radiance

- [Engine overview](engines/radiance/README.md) — kernel components,
  remote MoE and model results.
- [Patch and component catalogue](engines/radiance/ENHANCEMENTS.md) —
  integration patches and BigCherry-owned extensions.
- [Flash-Next measurements](engines/radiance/models/qwen3.8-flash-next.md) —
  four-card layout, prefill, multi-stream decode and retained-session results.
- [Serving configuration](../../engines/radiance/components/serving/README.md) —
  the current C++ loader and manifest work.

## llama.cpp

- [Engine overview](engines/llamacpp/README.md) — HIP measured dispatch and
  overlay architecture.
- [Patch catalogue](engines/llamacpp/PATCHES.md) — packages, recorded states
  and technical detail.
- [MMQ candidate families](architecture/MMQ_CATALOG.md) — measured-dispatch
  kernel families.
- [Multi-GPU dispatch](architecture/MULTI_GPU_DISPATCH.md) — tensor and
  layer splitting and reduction policies.

Replay uses a compact cache of promoted candidates. When no signature
matches, native upstream dispatch is used.

## Measurements and limits

- [Patch validation](testing/PATCH_VALIDATION.md) — how a patch's recorded
  state relates to the available evidence.
- [Experiment contracts](experiments/EXPERIMENT_CONTRACT.md) — workload
  identity, controls and measurements.
- [Radiance model pages](engines/radiance/README.md#model-specific-results) —
  dated performance and quality results for specific models and layouts.

An improvement for one model and hardware layout is not a general hardware
performance claim. Package state, build selection and measured coverage are
separate.

See the [reference index](README.md) for detailed build, kernel and testing
documents.
