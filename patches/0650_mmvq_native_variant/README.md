# 0650_mmvq_native_variant framework validation

Makes the compiled MMVQ geometry instances from 0600 actually reachable.
Threads a forced-geometry struct down the existing native call chain
(`ggml_cuda_mul_mat_vec_q` -> `mul_mat_vec_q_switch_ncols_dst`) to the point
where quantization/strides/fastdiv triples are already computed, dispatching
to the compiled instance via `ggml_hip_mmvq_find_instance` only at the
launch call -- rather than reimplementing that computation beside it, which
would silently drift on every upstream release. Routing exists only under
`GGML_HIP_DISPATCH`; the split-buffer MoE entry point remains native and
untouched. Two cases call `GGML_ABORT` as fatal invariant checks, not
graceful eligibility rejection -- `ggml_hip_mmvq_can_execute` is expected to
reject both before this path is ever reached: no compiled instance exists
for the requested geometry (never silently falls back to native, which
would misattribute a tuner measurement to the wrong candidate); forced
`MUL_MAT_ID` with `ncols_dst > 1` (upstream routes that to a dedicated MoE
kernel with no geometry dimension). Fusion is left to the resolved instance,
since one compiled instance already serves both fused and unfused calls.

This local framework adapter has no Experiment Contract: it checks build
plumbing, not a claimed kernel speedup. Historical `validated` state is not
current qualification.

Validation is the universal `apply`/`build` checks only -- no patch-specific
custom check exists for this package. Passing proves the patch applies
cleanly to the pinned upstream source and the resulting tree compiles; it
does NOT prove that a forced-geometry launch actually resolves and produces
correct output at runtime.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework
(HI09 part 2).
