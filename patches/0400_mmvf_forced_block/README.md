# 0400_mmvf_forced_block framework validation

Threads a forced MMVF block-size/accumulator-mode value down to the launcher
via explicit, appended, defaulted parameters (`ggml_cuda_mul_mat_vec_f` ->
`mul_mat_vec_f_cuda` -> `..._switch_ncols_dst` -> `launch_..._cuda`), rather
than a thread-local override -- an override would make the native path pay,
on every production MMVF launch, for a feature only the tuner uses. With a
defaulted parameter the native path stays byte-identical to upstream; only a
forced launch carries the value. Accumulator mode never upgrades precision:
F16 accumulation is only selected where native policy would also select it.

This local framework adapter has no Experiment Contract: it checks build
plumbing, not a claimed kernel speedup. Historical `validated` state is not
current qualification.

Validation is the universal `apply`/`build` checks only -- no patch-specific
custom check exists for this package. Passing proves the patch applies
cleanly to the pinned upstream source and the resulting tree compiles; it
does NOT prove runtime activation, dispatch correctness (that is HI16's
job), or performance for this patch's actual behavior.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework
(HI07).
