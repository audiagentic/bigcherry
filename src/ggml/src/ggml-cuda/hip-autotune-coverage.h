// bigcherry: dispatch coverage counters.
//
// Answers one question that nothing else can: **what fraction of real matmul
// work actually reaches measured dispatch?**
//
// The dense selector is not the only way into the matmul families. The graph
// optimiser calls `ggml_cuda_mul_mat_vec_q` and `ggml_cuda_mul_mat_vec_f`
// directly for fused patterns, bypassing `ggml_cuda_mul_mat` entirely. Those
// launches execute, take time, and are invisible to a dispatch layer hooked
// only at the dense selector -- so a tuning run could optimise a minority of
// the work while reporting complete success.
//
// Two counters per family make the gap measurable rather than assumed:
//
//   executed  - launches that reached the family entry point, by any route
//   dispatched - launches that came through ggml_hip_dispatch_mul_mat
//
// The difference is exactly what is escaping through uncovered collection
// points (standards 9.1 lists five; HI04 covers two).
//
// Cost is one relaxed atomic increment per launch, on a path that is about to
// do a matrix multiply. It stays compiled in because a coverage number nobody
// can produce on demand is a coverage number nobody checks.

#pragma once

#include "hip-autotune-types.h"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

// Counted at the family entry point, regardless of how execution got there.
void ggml_hip_coverage_count_executed(ggml_hip_kernel_family family);

// Counted when an operation is handled by measured dispatch.
void ggml_hip_coverage_count_dispatched(ggml_hip_kernel_family family);

// Write the coverage report. Called from ggml_hip_autotune_flush; writes to
// GGML_HIP_DISPATCH_COVERAGE if set, otherwise logs a summary.
void ggml_hip_coverage_report();

#endif // GGML_USE_HIP && GGML_HIP_DISPATCH
