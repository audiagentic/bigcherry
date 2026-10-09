// bigcherry: generated MMVQ geometry instances (HI09).
//
// One translation unit per candidate. That is deliberate rather than
// convenient: the bounded geometry matrix runs to thousands of instances, and
// compiling them into a handful of files serialises the build and produces
// object files large enough to trip linker limits. Per-candidate units also
// mean the catalog can add or drop one variant without recompiling the rest,
// and give the resource-report pass (HI09b) a clean mapping from a compiler
// remark back to a stable name.
//
// Each generated file is three lines:
//
//     #include "../mmvq-autotune.cuh"
//     DECL_MMVQ_AUTOTUNE_CASE(GGML_TYPE_Q8_0, 4, 2, 1);
//
// and nothing here is hand-written -- `bigcherry generate` emits them all from
// the manifest (standards 2.5).

#pragma once

// common.cuh only, deliberately. `mmvq.cuh` has no include guard upstream, so
// including it here as well as from mmvq.cu redeclares everything it contains
// -- which surfaces as "redefinition of default argument" and an ambiguous
// overload, some distance from the actual cause. Nothing below needs it:
// ggml_cuda_mm_fusion_args_device and ggml_type both come from common.cuh.
#include "common.cuh"

// A forced geometry request, threaded down through the native dispatch chain.
//
// Deliberately outside the HIP/dispatch guard below: it appears in the
// signatures of upstream's own mul_mat_vec_q_switch_type and
// mul_mat_vec_q_switch_ncols_dst, which are compiled in every build. It is a
// plain host struct with no HIP dependency, and in a build without measured
// dispatch it is simply always default-constructed and never read.
//
// One struct rather than three parameters because it is forwarded through 24
// call sites in mul_mat_vec_q_switch_type, one per quantised type. Adding a
// geometry dimension later then touches this header and two signatures, not
// every case of that switch -- which is the difference between a patch that
// survives an upstream release and one that does not.
//
// All-zero means "native policy", which is what every upstream caller passes by
// omission. `small_k` is only meaningful alongside a non-zero geometry: on the
// native path upstream derives it from the shape via should_use_small_k.
struct ggml_hip_mmvq_forced {
    int  nwarps         = 0;
    int  rows_per_block = 0;
    bool small_k        = false;

    bool requested() const { return nwarps != 0 || rows_per_block != 0; }
};

#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

// The argument bundle a generated instance is launched with.
//
// Passed as one struct rather than twenty-odd parameters because every
// generated instance repeats this signature, and a struct means upstream adding
// a stride does not require regenerating and recompiling every file -- only
// this header and the launcher change.
struct ggml_hip_mmvq_launch_args {
    const void *    vx;
    const void *    vy;
    const int32_t * ids;
    ggml_cuda_mm_fusion_args_device fusion;
    float *         dst;

    uint32_t ncols_x;
    uint3    nchannels_y;
    uint32_t stride_row_x;
    uint32_t stride_col_y;
    uint32_t stride_col_dst;
    uint3    channel_ratio;
    uint32_t stride_channel_x;
    uint32_t stride_channel_y;
    uint32_t stride_channel_dst;
    uint3    sample_ratio;
    uint32_t stride_sample_x;
    uint32_t stride_sample_y;
    uint32_t stride_sample_dst;
    uint32_t ids_stride;

    int      nrows_x;
    int      nchannels_dst;
    int      nsamples_or_ntokens;
    int      warp_size;
    int      nbytes_shared;
    cudaStream_t stream;
};

// Signature of every generated instance.
typedef void (*ggml_hip_mmvq_instance_fn)(const ggml_hip_mmvq_launch_args & args);

// Defined in mmvq.cu by the HI09 patch. Instantiates the kernel with the
// explicit geometry and derives launch parameters from the same numbers, so the
// grid and block shape can never disagree with what was compiled.
template <ggml_type type, int width, int nwarps, int rows_per_block, bool small_k>
void ggml_hip_mmvq_launch_instance(const ggml_hip_mmvq_launch_args & args);

// Emits one instance. The function name encodes the geometry so a compiler
// resource remark, a linker error and a catalog stable name all refer to
// recognisably the same thing.
#define GGML_HIP_MMVQ_INSTANCE_NAME(type, width, nwarps, rows, smallk) \
    ggml_hip_mmvq_instance_##type##_w##width##_nw##nwarps##_r##rows##_sk##smallk

#define DECL_MMVQ_AUTOTUNE_CASE_SK(type, width, nwarps, rows, smallk)          \
    void GGML_HIP_MMVQ_INSTANCE_NAME(type, width, nwarps, rows, smallk)(       \
            const ggml_hip_mmvq_launch_args & args) {                          \
        ggml_hip_mmvq_launch_instance<type, width, nwarps, rows,               \
                                      (smallk) != 0>(args);                    \
    }

// The common case: ordinary (non-small-K) geometry.
#define DECL_MMVQ_AUTOTUNE_CASE(type, width, nwarps, rows) \
    DECL_MMVQ_AUTOTUNE_CASE_SK(type, width, nwarps, rows, 0)

#endif // GGML_USE_HIP && GGML_HIP_DISPATCH
