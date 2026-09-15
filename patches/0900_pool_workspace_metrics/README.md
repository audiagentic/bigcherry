# 0900_pool_workspace_metrics framework validation

Adds requested-byte pool high-water accounting to `ggml_cuda_pool`'s base
struct: a per-`ggml_cuda_pool_alloc` `bc_requested_size` field plus
`bc_workspace_in_use`/`bc_workspace_peak` counters (with
`bc_workspace_note_alloc`/`_note_free`/`_reset_peak`), updated at the single
choke point (`ggml_cuda_pool_alloc`'s two call sites in `common.cuh`)
through which every pool alloc/free in the CUDA/HIP backend passes -- using
requested bytes, not the pool's own actual/cached-size bookkeeping, so
`measurement.workspace_bytes` reports each candidate's real demand instead
of its declared upper bound (a constant within a family that never
discriminated anything -- HI45's low-memory Pareto profile always reported
0% savings against it). Also adds a legacy-pool `bc_workspace_clear_cache()`
override, since the legacy pool implementation (unlike VMM) needs an
explicit cache-clear path. An earlier device-global `hipMemGetInfo`-delta
attempt failed structurally because the caching pool reuses a
high-water-mark allocation; the pool's own bookkeeping is the only place
that actually knows the per-candidate answer.

This local framework adapter has no Experiment Contract: it checks build
plumbing, not a claimed kernel speedup. Historical `validated` state is not
current qualification.

Validation is the universal `apply`/`build` checks only -- no patch-specific
custom check exists for this package. Passing proves the patch applies
cleanly to the pinned upstream source and the resulting tree compiles; it
does NOT prove the reported `bc_requested_size` values are correct for any
given workload.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework
(HI52 part 1).
