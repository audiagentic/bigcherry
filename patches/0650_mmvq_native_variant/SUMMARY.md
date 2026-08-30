# 0650_mmvq_native_variant: Route a forced MMVQ geometry to its compiled instance (HI09 part 2)

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Threads a forced-geometry struct down the existing native chain (ggml_cuda_mul_mat_vec_q -> mul_mat_vec_q_switch_ncols_dst) to the point where quantization/strides are already computed, diverging only at the launch call via ggml_hip_mmvq_find_instance.

## Why

Makes the geometry variants compiled by patch 0600 actually reachable, without duplicating upstream's quantization/stride logic (which would drift silently on every release). Refuses an unmatched geometry, MUL_MAT_ID width>1, and leaves fusion to the resolved instance rather than the forced path.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI09).
