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
// Diagnostic builds pay one relaxed atomic increment per counted launch.
// Production call sites exclude both counting and its reentrancy probe with
// GGML_HIP_DISPATCH_DIAGNOSTICS; setting a report path cannot enable them.

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

// Defined in hip-autotune-dispatch.cu. Appends the hot-path counters to the
// coverage JSON as a "dispatch" object, or writes nothing when they are
// disabled.
//
// This exists because the log channel is unusable under llama-server: it
// installs a log callback that swallows the library's GGML_LOG_INFO lines
// entirely, so the counter report, the native-force report and the startup
// replay cache-load line NEVER reach stdout or stderr. Runs that looked
// completely silent -- and were read as "the dispatch layer never ran" --
// were in fact working the whole time. The JSON file is the only channel
// that survives, so anything needed to interpret a benchmark has to go here
// rather than into a log line.
void ggml_hip_dispatch_counters_write_json(void * out_file);

#endif // GGML_USE_HIP && GGML_HIP_DISPATCH
