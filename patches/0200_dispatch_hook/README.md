# 0200_dispatch_hook framework validation

Inserts a single guarded hook (`ggml_hip_dispatch_mul_mat`) at the top of
upstream's dense-matmul selector entry points (`ggml_cuda_mul_mat` and the
cuBLAS entry point, the latter newly exposed for this purpose). The hook
returns false whenever it declines, in which case upstream's own selector
ladder runs untouched -- a native-mode build with the layer compiled in
still executes upstream's real code path, not a reimplementation of it.

This local framework adapter has no Experiment Contract: it checks build
plumbing, not a claimed kernel speedup. Historical `validated` state is not
current qualification.

Validation is the universal `apply`/`build` checks only -- no patch-specific
custom check exists for this package. Passing proves the patch applies
cleanly to the pinned upstream source and the resulting tree compiles; it
does NOT prove runtime activation, dispatch correctness, or performance for
this patch's actual behavior.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework
(HI04).
