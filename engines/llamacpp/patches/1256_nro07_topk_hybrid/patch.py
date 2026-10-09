"""PNRO06 (NRO07): hybrid HIP TOP_K kernels.

Port of nasone32/llama.cpp-RDNA3-7900xtx-opt 7f3e1e4d "ROCm: add hybrid
TOP_K kernels" plus its follow-up rename 10fdba9a, generated mechanically by
tools/bigcherry/patch/port_diff.py from the fork's file states (the fork's
pre-change top-k.cu is byte-identical to b11126's, so the port is exact) and
verified to reproduce the fork's top-k.cu byte for byte, plus BigCherry
activation markers.

What it does (HIP only): replaces the HIP TOP_K route with
  * k == 1, or (HIP >= 7.15) small rows      -> top_k_small_cuda
  * ncols > 1024                              -> top_k_parallel_radix_cuda
  * otherwise                                 -> bitonic argsort + copy
and compiles top-k.cu as wave64 when every HIP target is gfx11/gfx120x.

Toolchain note: the small-row route is compiled only for HIP >= 7.15; the
fleet's ROCm (7.2.4, 7.14) exercises only the k == 1 and radix routes.

Activation: BIGCHERRY_PATCH_TRACE=1 logs
  BIGCHERRY_PATCH_HIT patch=1256_nro07 path=topk_{small,parallel_radix}
once per route.
"""

from bigcherry.patcher import Edit, FilePatch

PROVENANCE = {
    "source-id": "nasone-rdna-optimizations",
    "plan-item": "NRO07",
    "fork-commit": "7f3e1e4d0b166cb681b2c01503370e610a5b423d",
    "fork-commits": [
        "7f3e1e4d0b166cb681b2c01503370e610a5b423d",
        "10fdba9ae2714f5b786374c43d306e460db66a1c",
    ],
    "port-mode": "port_diff-generated, verified byte-exact against the fork file",
}

TOPK_HYBRID = FilePatch(
    path='ggml/src/ggml-cuda/top-k.cu',
    description='hybrid TOP_K kernels (nasone 7f3e1e4d + rename 10fdba9a) with activation markers',
    edits=(
        Edit(
            id='nro07-topk-01',
            anchor='\\#include\\ "argsort\\.cuh"\\\n\\#include\\ "top\\-k\\.cuh"\\\n\\\n\\#ifdef\\ GGML_CUDA_USE_CUB\\\n',
            text='#include "argsort.cuh"\n#include "top-k.cuh"\n\n#if defined(GGML_USE_HIP)\n#include <hip/hip_version.h>\n#endif\n\n#ifdef GGML_CUDA_USE_CUB\n',
            mode="replace",
            guard='\\#include\\ <hip/hip_version\\.h>',
            rationale='nro07-topk hunk 1: upstream lines 3-2 -> result lines 3-6',
            max_span_lines=6,
        ),
        Edit(
            id='nro07-topk-02',
            anchor='\\#endif\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\\n\\#if\\ !defined\\(GGML_CUDA_USE_CUB\\)\\ \\&\\&\\ defined\\(GGML_USE_HIP\\)\\\n\\\nstatic\\ __device__\\ __forceinline__\\ uint32_t\\ top_k_float_to_ordered\\(float\\ value\\)\\ \\{\\\n',
            text='#endif                            // CUB_TOP_K_AVAILABLE\n\n#if defined(GGML_USE_HIP)\n\nstatic __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {\n',
            mode="replace",
            guard='\\#endif\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ //\\ CUB_TOP_K_AVAILABLE\\\n\\\n\\#if\\ defined\\(GGML_USE_HIP\\)\\\n\\\nstatic\\ __device__\\ __forceinline__\\ uint32_t\\ top_k_float_to_ordered\\(float\\ value\\)\\ \\{\\\n',
            rationale='nro07-topk hunk 2: upstream lines 51-51 -> result lines 55-55',
            max_span_lines=7,
        ),
        Edit(
            id='nro07-topk-03',
            anchor='\\ \\ \\ \\ const\\ uint32_t\\ mask\\ =\\ \\(uint32_t\\)\\ \\(\\-\\(int32_t\\)\\ \\(bits\\ >>\\ 31\\)\\)\\ \\|\\ 0x80000000U;\\\n\\ \\ \\ \\ return\\ bits\\ \\^\\ mask;\\\n\\}\\\n\\\nstruct\\ top_k_radix_state\\ \\{\\\n',
            text='    return (bits & 0x80000000U) != 0 ? ~bits : bits | 0x80000000U;\n}\n\ntemplate<int BLOCK_SIZE>\nstatic __global__ void top_k_nary_search_cuda(\n        const float * __restrict__ src,\n        const int2 * __restrict__ src_pairs,\n        int * __restrict__ dst,\n        int2 * __restrict__ dst_pairs,\n        int original_ncols,\n        int ncols_input,\n        int ncols_output,\n        int k,\n        int nrows,\n        bool first_pass,\n        bool last_pass) {\n    const int tid = threadIdx.x;\n    const int lane = tid % warpSize;\n    const int warp = tid / warpSize;\n    const int warp_count = BLOCK_SIZE / warpSize;\n\n    __shared__ int2 candidates[BLOCK_SIZE];\n    __shared__ uint32_t counts[64];\n    __shared__ uint32_t selected_bucket;\n    __shared__ uint32_t selected_total;\n    __shared__ uint32_t warp_offsets[32];\n    __shared__ uint32_t warp_equal_offsets[32];\n\n    for (int row = blockIdx.y; row < nrows; row += gridDim.y) {\n        const int col = blockIdx.x * BLOCK_SIZE + tid;\n        const bool valid = col < ncols_input;\n        int2 value;\n        if (valid) {\n            value = first_pass\n                ? make_int2(col, __float_as_int(src[(size_t) row * ncols_input + col]))\n                : src_pairs[(size_t) row * ncols_input + col];\n        } else {\n            value = make_int2(original_ncols, (int) 0xff800000U);\n        }\n        candidates[tid] = value;\n        __syncthreads();\n\n        const int limit = min(k, ncols_input - blockIdx.x * BLOCK_SIZE);\n        if (k == 1) {\n#pragma unroll\n            for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {\n                if (tid < stride) {\n                    const int2 a = candidates[tid];\n                    const int2 b = candidates[tid + stride];\n                    if (a.x >= original_ncols ||\n                        (b.x < original_ncols && __int_as_float(b.y) > __int_as_float(a.y))) {\n                        candidates[tid] = b;\n                    }\n                }\n                __syncthreads();\n            }\n        } else {\n            const int radix_bits = warpSize == 64 ? 6 : 5;\n            const int radix_size = 1 << radix_bits;\n            int shift = 32 - radix_bits;\n            uint32_t mask = ((1U << radix_bits) - 1) << shift;\n            uint32_t range_min = 0;\n            uint32_t range_max = 0xff800000U;\n            uint32_t total = 0;\n\n            while (mask != 0) {\n                if (tid < radix_size) {\n                    counts[tid] = 0;\n                }\n                __syncthreads();\n\n                const uint32_t key = top_k_float_to_ordered(__int_as_float(value.y));\n                if (valid && key >= range_min && key < range_max) {\n                    atomicAdd(&counts[(key & mask) >> shift], 1U);\n                }\n                __syncthreads();\n\n                if (tid < radix_size) {\n                    uint32_t partial = counts[radix_size - 1 - tid];\n                    for (int offset = 1; offset < radix_size; offset *= 2) {\n                        const uint32_t previous = __shfl_up(partial, offset, radix_size);\n                        if (tid >= offset) {\n                            partial += previous;\n                        }\n                    }\n                    partial += total;\n                    const unsigned long long selected = __ballot(partial >= (uint32_t) limit);\n                    const int first = __ffsll(selected) - 1;\n                    if (tid == first) {\n                        selected_bucket = radix_size - 1 - first;\n                        selected_total = partial;\n                    }\n                }\n                __syncthreads();\n\n                const uint32_t bucket = selected_bucket;\n                total = selected_total;\n                range_max = range_min + ((bucket + 1) << shift);\n                range_min = range_min + (bucket << shift);\n                if (total == (uint32_t) limit) {\n                    break;\n                }\n                total -= counts[bucket];\n                mask >>= radix_bits;\n                shift -= radix_bits;\n                if (shift < 0) {\n                    shift = 0;\n                }\n            }\n\n            const uint32_t key = top_k_float_to_ordered(__int_as_float(value.y));\n            const bool above = valid && key > range_min;\n            const bool equal = valid && key == range_min;\n            const unsigned long long above_mask = __ballot(above);\n            const unsigned long long equal_mask = __ballot(equal);\n            if (lane == 0) {\n                warp_offsets[warp] = __popcll(above_mask);\n                warp_equal_offsets[warp] = __popcll(equal_mask);\n            }\n            __syncthreads();\n\n            uint32_t above_base = 0;\n            uint32_t equal_base = 0;\n            uint32_t above_total = 0;\n            for (int i = 0; i < warp_count; ++i) {\n                if (i < warp) {\n                    above_base += warp_offsets[i];\n                    equal_base += warp_equal_offsets[i];\n                }\n                above_total += warp_offsets[i];\n            }\n            equal_base += above_total;\n\n            const unsigned long long lane_mask = lane == 0 ? 0 : (1ULL << lane) - 1;\n            if (above) {\n                candidates[above_base + __popcll(above_mask & lane_mask)] = value;\n            }\n            const uint32_t equal_index = equal_base + __popcll(equal_mask & lane_mask);\n            if (equal && equal_index < (uint32_t) limit) {\n                candidates[equal_index] = value;\n            }\n            __syncthreads();\n        }\n\n        if (tid < k) {\n            if (last_pass) {\n                dst[(size_t) row * k + tid] = candidates[tid].x;\n            } else {\n                const int output_col = blockIdx.x * k + tid;\n                if (output_col < ncols_output) {\n                    dst_pairs[(size_t) row * ncols_output + output_col] = candidates[tid];\n                }\n            }\n        }\n        __syncthreads();\n    }\n}\n\ntemplate<int BLOCK_SIZE>\nstatic __device__ __forceinline__ int2 top_k_one_reduce(\n        int2 * candidates,\n        int2 value,\n        int original_ncols) {\n    const int tid = threadIdx.x;\n    candidates[tid] = value;\n    __syncthreads();\n\n#pragma unroll\n    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {\n        if (tid < stride) {\n            const int2 a = candidates[tid];\n            const int2 b = candidates[tid + stride];\n            if (a.x >= original_ncols ||\n                (b.x < original_ncols && __int_as_float(b.y) > __int_as_float(a.y))) {\n                candidates[tid] = b;\n            }\n        }\n        __syncthreads();\n    }\n    return candidates[0];\n}\n\ntemplate<int BLOCK_SIZE>\nstatic __global__ void top_k_one_first_cuda(\n        const float * __restrict__ src,\n        int2 * __restrict__ dst_pairs,\n        int original_ncols,\n        int ncols_input,\n        int ncols_output,\n        int nrows) {\n    const int tid = threadIdx.x;\n    __shared__ int2 candidates[BLOCK_SIZE];\n\n    for (int row = blockIdx.y; row < nrows; row += gridDim.y) {\n        const int col = blockIdx.x * BLOCK_SIZE + tid;\n        const int2 value = col < ncols_input\n            ? make_int2(col, __float_as_int(src[(size_t) row * ncols_input + col]))\n            : make_int2(original_ncols, (int) 0xff800000U);\n        const int2 result = top_k_one_reduce<BLOCK_SIZE>(candidates, value, original_ncols);\n        if (tid == 0) {\n            dst_pairs[(size_t) row * ncols_output + blockIdx.x] = result;\n        }\n        __syncthreads();\n    }\n}\n\ntemplate<int BLOCK_SIZE>\nstatic __global__ void top_k_one_first_last_cuda(\n        const float * __restrict__ src,\n        int * __restrict__ dst,\n        int original_ncols,\n        int ncols_input,\n        int nrows) {\n    const int tid = threadIdx.x;\n    __shared__ int2 candidates[BLOCK_SIZE];\n\n    for (int row = blockIdx.y; row < nrows; row += gridDim.y) {\n        const int col = blockIdx.x * BLOCK_SIZE + tid;\n        const int2 value = col < ncols_input\n            ? make_int2(col, __float_as_int(src[(size_t) row * ncols_input + col]))\n            : make_int2(original_ncols, (int) 0xff800000U);\n        const int2 result = top_k_one_reduce<BLOCK_SIZE>(candidates, value, original_ncols);\n        if (tid == 0) {\n            dst[row] = result.x;\n        }\n        __syncthreads();\n    }\n}\n\ntemplate<int BLOCK_SIZE>\nstatic __global__ void top_k_one_middle_cuda(\n        const int2 * __restrict__ src_pairs,\n        int2 * __restrict__ dst_pairs,\n        int original_ncols,\n        int ncols_input,\n        int ncols_output,\n        int nrows) {\n    const int tid = threadIdx.x;\n    __shared__ int2 candidates[BLOCK_SIZE];\n\n    for (int row = blockIdx.y; row < nrows; row += gridDim.y) {\n        const int col = blockIdx.x * BLOCK_SIZE + tid;\n        const int2 value = col < ncols_input\n            ? src_pairs[(size_t) row * ncols_input + col]\n            : make_int2(original_ncols, (int) 0xff800000U);\n        const int2 result = top_k_one_reduce<BLOCK_SIZE>(candidates, value, original_ncols);\n        if (tid == 0) {\n            dst_pairs[(size_t) row * ncols_output + blockIdx.x] = result;\n        }\n        __syncthreads();\n    }\n}\n\ntemplate<int BLOCK_SIZE>\nstatic __global__ void top_k_one_last_cuda(\n        const int2 * __restrict__ src_pairs,\n        int * __restrict__ dst,\n        int original_ncols,\n        int ncols_input,\n        int nrows) {\n    const int tid = threadIdx.x;\n    __shared__ int2 candidates[BLOCK_SIZE];\n\n    for (int row = blockIdx.y; row < nrows; row += gridDim.y) {\n        const int col = blockIdx.x * BLOCK_SIZE + tid;\n        const int2 value = col < ncols_input\n            ? src_pairs[(size_t) row * ncols_input + col]\n            : make_int2(original_ncols, (int) 0xff800000U);\n        const int2 result = top_k_one_reduce<BLOCK_SIZE>(candidates, value, original_ncols);\n        if (tid == 0) {\n            dst[row] = result.x;\n        }\n        __syncthreads();\n    }\n}\n\ntemplate<int BLOCK_SIZE>\nstatic __global__ void top_k_radix_select_cuda(\n        const float * __restrict__ src,\n        int * __restrict__ dst,\n        int ncols,\n        int nrows,\n        int k) {\n    constexpr int RADIX_BITS = 8;\n    constexpr int RADIX_SIZE = 1 << RADIX_BITS;\n\n    const int tid = threadIdx.x;\n    __shared__ uint32_t histogram[RADIX_SIZE];\n    __shared__ uint32_t selected_bucket;\n    __shared__ uint32_t count_above;\n    __shared__ uint32_t output_count;\n\n    for (int row = blockIdx.x; row < nrows; row += gridDim.x) {\n        const float * row_src = src + (size_t) row * ncols;\n        int * row_dst = dst + (size_t) row * k;\n        uint32_t prefix = 0;\n        uint32_t desired = k;\n\n#pragma unroll\n        for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {\n            for (int bin = tid; bin < RADIX_SIZE; bin += BLOCK_SIZE) {\n                histogram[bin] = 0;\n            }\n            __syncthreads();\n\n            const uint32_t high_mask = shift == 32 - RADIX_BITS ? 0 : 0xffffffffU << (shift + RADIX_BITS);\n            const uint32_t prefix_high = prefix & high_mask;\n            for (int col = tid; col < ncols; col += BLOCK_SIZE) {\n                const uint32_t key = top_k_float_to_ordered(row_src[col]);\n                if ((key & high_mask) == prefix_high) {\n                    atomicAdd(&histogram[(key >> shift) & (RADIX_SIZE - 1)], 1U);\n                }\n            }\n            __syncthreads();\n\n            if (tid == 0) {\n                uint32_t above = 0;\n                uint32_t bucket = 0;\n                for (int bin = RADIX_SIZE - 1; bin >= 0; --bin) {\n                    const uint32_t count = histogram[bin];\n                    if (above + count >= desired) {\n                        bucket = bin;\n                        break;\n                    }\n                    above += count;\n                }\n                selected_bucket = bucket;\n                count_above = above;\n            }\n            __syncthreads();\n\n            prefix |= selected_bucket << shift;\n            desired -= count_above;\n            __syncthreads();\n        }\n\n        if (tid == 0) {\n            output_count = 0;\n        }\n        __syncthreads();\n\n        for (int col = tid; col < ncols; col += BLOCK_SIZE) {\n            if (top_k_float_to_ordered(row_src[col]) > prefix) {\n                row_dst[atomicAdd(&output_count, 1U)] = col;\n            }\n        }\n        __syncthreads();\n\n        for (int col = tid; col < ncols; col += BLOCK_SIZE) {\n            if (top_k_float_to_ordered(row_src[col]) == prefix) {\n                const uint32_t output = atomicAdd(&output_count, 1U);\n                if (output < (uint32_t) k) {\n                    row_dst[output] = col;\n                }\n            }\n        }\n        __syncthreads();\n    }\n}\n\nstatic void top_k_radix_select_cuda(\n        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {\n    constexpr int BLOCK_SIZE = 1024;\n    const int grid_size = std::min(nrows, 65535);\n    top_k_radix_select_cuda<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, stream>>>(src, dst, ncols, nrows, k);\n}\n\nstatic int top_k_floor_log2(int value) {\n    int result = 0;\n    while (value > 1) {\n        value >>= 1;\n        ++result;\n    }\n    return result;\n}\n\nstatic int top_k_ceil_log2(int value) {\n    const int floor = top_k_floor_log2(value);\n    return value == (1 << floor) ? floor : floor + 1;\n}\n\nstatic int top_k_nary_block_log2(int ncols, int k) {\n    const int min_block = std::max(top_k_floor_log2(k) + 1, 6);\n    if (min_block > 10) {\n        return -1;\n    }\n    const int max_block = std::min(std::max(top_k_floor_log2(k) + 2, 8), 10);\n    int block = std::min(std::max(top_k_ceil_log2(ncols), min_block), max_block);\n    if (ncols > (1 << block)) {\n        for (int candidate = block; candidate <= 10; ++candidate) {\n            if (ncols <= (1 << candidate)) {\n                block = candidate;\n                break;\n            }\n        }\n    }\n    return block;\n}\n\ntemplate<int BLOCK_SIZE>\nstatic void top_k_one_cuda_launch(\n        const float * src,\n        const int2 * src_pairs,\n        int * dst,\n        int2 * dst_pairs,\n        int original_ncols,\n        int ncols_input,\n        int ncols_output,\n        int nrows,\n        bool first_pass,\n        bool last_pass,\n        cudaStream_t stream) {\n    const dim3 grid((ncols_input + BLOCK_SIZE - 1) / BLOCK_SIZE, std::min(nrows, 65535), 1);\n    if (first_pass && last_pass) {\n        top_k_one_first_last_cuda<BLOCK_SIZE><<<grid, BLOCK_SIZE, 0, stream>>>(\n            src, dst, original_ncols, ncols_input, nrows);\n    } else if (first_pass) {\n        top_k_one_first_cuda<BLOCK_SIZE><<<grid, BLOCK_SIZE, 0, stream>>>(\n            src, dst_pairs, original_ncols, ncols_input, ncols_output, nrows);\n    } else if (last_pass) {\n        top_k_one_last_cuda<BLOCK_SIZE><<<grid, BLOCK_SIZE, 0, stream>>>(\n            src_pairs, dst, original_ncols, ncols_input, nrows);\n    } else {\n        top_k_one_middle_cuda<BLOCK_SIZE><<<grid, BLOCK_SIZE, 0, stream>>>(\n            src_pairs, dst_pairs, original_ncols, ncols_input, ncols_output, nrows);\n    }\n}\n\nstatic void top_k_one_cuda_launch(\n        int block_log2,\n        const float * src,\n        const int2 * src_pairs,\n        int * dst,\n        int2 * dst_pairs,\n        int original_ncols,\n        int ncols_input,\n        int ncols_output,\n        int nrows,\n        bool first_pass,\n        bool last_pass,\n        cudaStream_t stream) {\n    switch (block_log2) {\n        case 6:\n            top_k_one_cuda_launch<64>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, nrows, first_pass, last_pass, stream);\n            break;\n        case 7:\n            top_k_one_cuda_launch<128>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, nrows, first_pass, last_pass, stream);\n            break;\n        case 8:\n            top_k_one_cuda_launch<256>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, nrows, first_pass, last_pass, stream);\n            break;\n        case 9:\n            top_k_one_cuda_launch<512>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, nrows, first_pass, last_pass, stream);\n            break;\n        case 10:\n            top_k_one_cuda_launch<1024>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, nrows, first_pass, last_pass, stream);\n            break;\n        default:\n            GGML_ABORT("invalid HIP TOP_K block size");\n    }\n}\n\ntemplate<int BLOCK_SIZE>\nstatic void top_k_nary_search_cuda_launch(\n        const float * src,\n        const int2 * src_pairs,\n        int * dst,\n        int2 * dst_pairs,\n        int original_ncols,\n        int ncols_input,\n        int ncols_output,\n        int k,\n        int nrows,\n        bool first_pass,\n        bool last_pass,\n        cudaStream_t stream) {\n    const dim3 grid((ncols_input + BLOCK_SIZE - 1) / BLOCK_SIZE, std::min(nrows, 65535), 1);\n    top_k_nary_search_cuda<BLOCK_SIZE><<<grid, BLOCK_SIZE, 0, stream>>>(\n        src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output,\n        k, nrows, first_pass, last_pass);\n}\n\nstatic void top_k_nary_search_cuda_launch(\n        int block_log2,\n        const float * src,\n        const int2 * src_pairs,\n        int * dst,\n        int2 * dst_pairs,\n        int original_ncols,\n        int ncols_input,\n        int ncols_output,\n        int k,\n        int nrows,\n        bool first_pass,\n        bool last_pass,\n        cudaStream_t stream) {\n    switch (block_log2) {\n        case 6:\n            top_k_nary_search_cuda_launch<64>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, k, nrows, first_pass, last_pass, stream);\n            break;\n        case 7:\n            top_k_nary_search_cuda_launch<128>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, k, nrows, first_pass, last_pass, stream);\n            break;\n        case 8:\n            top_k_nary_search_cuda_launch<256>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, k, nrows, first_pass, last_pass, stream);\n            break;\n        case 9:\n            top_k_nary_search_cuda_launch<512>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, k, nrows, first_pass, last_pass, stream);\n            break;\n        case 10:\n            top_k_nary_search_cuda_launch<1024>(src, src_pairs, dst, dst_pairs, original_ncols, ncols_input, ncols_output, k, nrows, first_pass, last_pass, stream);\n            break;\n        default:\n            GGML_ABORT("invalid HIP TOP_K block size");\n    }\n}\n\nstatic void top_k_small_cuda(\n        ggml_cuda_pool & pool,\n        const float * src,\n        int * dst,\n        int ncols,\n        int nrows,\n        int k,\n        cudaStream_t stream) {\n    int block_log2 = top_k_nary_block_log2(ncols, k);\n    if (block_log2 < 0) {\n        top_k_radix_select_cuda(src, dst, ncols, nrows, k, stream);\n        return;\n    }\n\n    const int block_size = 1 << block_log2;\n    const int first_output = (ncols / block_size) * k + std::min(k, ncols % block_size);\n    const size_t scratch_elements = (size_t) first_output * nrows;\n    ggml_cuda_pool_alloc<int2> scratch_alloc(pool, 2 * scratch_elements);\n    int2 * scratch[2] = {scratch_alloc.get(), scratch_alloc.get() + scratch_elements};\n\n    int ncols_input = ncols;\n    int buffer = 0;\n    bool first_pass = true;\n    while (ncols_input > k || first_pass) {\n        block_log2 = top_k_nary_block_log2(ncols_input, k);\n        const int current_block_size = 1 << block_log2;\n        const int ncols_output = (ncols_input / current_block_size) * k + std::min(k, ncols_input % current_block_size);\n        const bool last_pass = ncols_output == k;\n        if (k == 1) {\n            top_k_one_cuda_launch(\n                block_log2, src, first_pass ? nullptr : scratch[buffer], dst,\n                last_pass ? nullptr : scratch[buffer ^ 1], ncols, ncols_input,\n                ncols_output, nrows, first_pass, last_pass, stream);\n        } else {\n            top_k_nary_search_cuda_launch(\n                block_log2, src, first_pass ? nullptr : scratch[buffer], dst,\n                last_pass ? nullptr : scratch[buffer ^ 1], ncols, ncols_input,\n                ncols_output, k, nrows, first_pass, last_pass, stream);\n        }\n        ncols_input = ncols_output;\n        first_pass = false;\n        buffer ^= 1;\n    }\n}\n\nstruct top_k_parallel_radix_state {\n',
            mode="replace",
            guard='return\\ \\(bits\\ \\&\\ 0x80000000U\\)\\ !=\\ 0\\ \\?\\ \\~bits\\ :\\ bits\\ \\|\\ 0x80000000U;',
            rationale='nro07-topk hunk 3: upstream lines 55-59 -> result lines 59-621',
            max_span_lines=7,
        ),
        Edit(
            id='nro07-topk-04',
            anchor='static\\ __global__\\ void\\ top_k_radix_init\\(top_k_radix_state\\ \\*\\ states,\\ int\\ nrows,\\ int\\ k\\)\\ \\{\\\n',
            text='static __global__ void top_k_parallel_radix_init(top_k_parallel_radix_state * states, int nrows, int k) {\n',
            mode="replace",
            guard='static\\ __global__\\ void\\ top_k_parallel_radix_init\\(top_k_parallel_radix_state\\ \\*\\ states,\\ int\\ nrows,\\ int\\ k\\)\\ \\{',
            rationale='nro07-topk hunk 4: upstream lines 67-67 -> result lines 629-629',
            max_span_lines=3,
        ),
        Edit(
            id='nro07-topk-05',
            anchor='static\\ __global__\\ void\\ top_k_radix_histogram\\(\\\n\\ \\ \\ \\ \\ \\ \\ \\ const\\ float\\ \\*\\ __restrict__\\ src,\\\n\\ \\ \\ \\ \\ \\ \\ \\ const\\ top_k_radix_state\\ \\*\\ __restrict__\\ states,\\\n',
            text='static __global__ void top_k_parallel_radix_histogram(\n        const float * __restrict__ src,\n        const top_k_parallel_radix_state * __restrict__ states,\n',
            mode="replace",
            guard='static\\ __global__\\ void\\ top_k_parallel_radix_histogram\\(',
            rationale='nro07-topk hunk 5: upstream lines 75-77 -> result lines 637-639',
            max_span_lines=5,
        ),
        Edit(
            id='nro07-topk-06',
            anchor='\\ \\ \\ \\ const\\ top_k_radix_state\\ state\\ =\\ states\\[row\\];\\\n',
            text='    const top_k_parallel_radix_state state = states[row];\n',
            mode="replace",
            guard='const\\ top_k_parallel_radix_state\\ state\\ =\\ states\\[row\\];',
            rationale='nro07-topk hunk 6: upstream lines 93-93 -> result lines 655-655',
            max_span_lines=3,
        ),
        Edit(
            id='nro07-topk-07',
            anchor='static\\ __global__\\ void\\ top_k_radix_select\\(\\\n\\ \\ \\ \\ \\ \\ \\ \\ const\\ int\\ \\*\\ __restrict__\\ block_histograms,\\\n\\ \\ \\ \\ \\ \\ \\ \\ top_k_radix_state\\ \\*\\ __restrict__\\ states,\\\n',
            text='static __global__ void top_k_parallel_radix_select(\n        const int * __restrict__ block_histograms,\n        top_k_parallel_radix_state * __restrict__ states,\n',
            mode="replace",
            guard='static\\ __global__\\ void\\ top_k_parallel_radix_select\\(',
            rationale='nro07-topk hunk 7: upstream lines 110-112 -> result lines 672-674',
            max_span_lines=5,
        ),
        Edit(
            id='nro07-topk-08',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ top_k_radix_state\\ state\\ =\\ states\\[row\\];\\\n',
            text='        top_k_parallel_radix_state state = states[row];\n',
            mode="replace",
            guard='\\ \\ \\ \\ \\ \\ \\ \\ top_k_parallel_radix_state\\ state\\ =\\ states\\[row\\];\\\n',
            rationale='nro07-topk hunk 8: upstream lines 130-130 -> result lines 692-692',
            max_span_lines=3,
        ),
        Edit(
            id='nro07-topk-09',
            anchor='static\\ __global__\\ void\\ top_k_radix_reset_counters\\(top_k_radix_state\\ \\*\\ states,\\ int\\ nrows\\)\\ \\{\\\n',
            text='static __global__ void top_k_parallel_radix_reset_counters(top_k_parallel_radix_state * states, int nrows) {\n',
            mode="replace",
            guard='static\\ __global__\\ void\\ top_k_parallel_radix_reset_counters\\(top_k_parallel_radix_state\\ \\*\\ states,\\ int\\ nrows\\)\\ \\{',
            rationale='nro07-topk hunk 9: upstream lines 141-141 -> result lines 703-703',
            max_span_lines=3,
        ),
        Edit(
            id='nro07-topk-10',
            anchor='static\\ __global__\\ void\\ top_k_radix_gather\\(\\\n\\ \\ \\ \\ \\ \\ \\ \\ const\\ float\\ \\*\\ __restrict__\\ src,\\\n\\ \\ \\ \\ \\ \\ \\ \\ int\\ \\*\\ __restrict__\\ dst,\\\n\\ \\ \\ \\ \\ \\ \\ \\ top_k_radix_state\\ \\*\\ __restrict__\\ states,\\\n',
            text='static __global__ void top_k_parallel_radix_gather(\n        const float * __restrict__ src,\n        int * __restrict__ dst,\n        top_k_parallel_radix_state * __restrict__ states,\n',
            mode="replace",
            guard='static\\ __global__\\ void\\ top_k_parallel_radix_gather\\(',
            rationale='nro07-topk hunk 10: upstream lines 150-153 -> result lines 712-715',
            max_span_lines=6,
        ),
        Edit(
            id='nro07-topk-11',
            anchor='\\ \\ \\ \\ top_k_radix_state\\ \\*\\ state\\ =\\ \\&states\\[row\\];\\\n',
            text='    top_k_parallel_radix_state * state = &states[row];\n',
            mode="replace",
            guard='top_k_parallel_radix_state\\ \\*\\ state\\ =\\ \\&states\\[row\\];',
            rationale='nro07-topk hunk 11: upstream lines 162-162 -> result lines 724-724',
            max_span_lines=3,
        ),
        Edit(
            id='nro07-topk-12',
            anchor='static\\ void\\ top_k_radix_cuda\\(\\\n',
            text='static void top_k_parallel_radix_cuda(\n',
            mode="replace",
            guard='static\\ void\\ top_k_parallel_radix_cuda\\(',
            rationale='nro07-topk hunk 12: upstream lines 180-180 -> result lines 742-742',
            max_span_lines=3,
        ),
        Edit(
            id='nro07-topk-13',
            anchor='\\ \\ \\ \\ ggml_cuda_pool_alloc<top_k_radix_state>\\ states_alloc\\(pool,\\ nrows\\);\\\n\\ \\ \\ \\ ggml_cuda_pool_alloc<int>\\ histograms_alloc\\(pool,\\ \\(size_t\\)\\ nrows\\ \\*\\ blocks_per_row\\ \\*\\ NBINS\\);\\\n\\ \\ \\ \\ top_k_radix_state\\ \\*\\ states\\ =\\ states_alloc\\.get\\(\\);\\\n\\ \\ \\ \\ int\\ \\*\\ histograms\\ =\\ histograms_alloc\\.get\\(\\);\\\n\\\n\\ \\ \\ \\ top_k_radix_init<<<\\(nrows\\ \\+\\ BLOCK_SIZE\\ \\-\\ 1\\)\\ /\\ BLOCK_SIZE,\\ BLOCK_SIZE,\\ 0,\\ stream>>>\\(states,\\ nrows,\\ k\\);\\\n',
            text='    ggml_cuda_pool_alloc<top_k_parallel_radix_state> states_alloc(pool, nrows);\n    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);\n    top_k_parallel_radix_state * states = states_alloc.get();\n    int * histograms = histograms_alloc.get();\n\n    top_k_parallel_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);\n',
            mode="replace",
            guard='ggml_cuda_pool_alloc<top_k_parallel_radix_state>\\ states_alloc\\(pool,\\ nrows\\);',
            rationale='nro07-topk hunk 13: upstream lines 188-193 -> result lines 750-755',
            max_span_lines=8,
        ),
        Edit(
            id='nro07-topk-14',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ top_k_radix_histogram<BLOCK_SIZE,\\ RADIX_BITS>\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ <<<row_grid,\\ BLOCK_SIZE,\\ 0,\\ stream>>>\\(\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ src,\\ states,\\ histograms,\\ ncols,\\ blocks_per_row,\\ shift\\);\\\n\\ \\ \\ \\ \\ \\ \\ \\ top_k_radix_select<BLOCK_SIZE,\\ RADIX_BITS>\\\n',
            text='        top_k_parallel_radix_histogram<BLOCK_SIZE, RADIX_BITS>\n            <<<row_grid, BLOCK_SIZE, 0, stream>>>(\n                src, states, histograms, ncols, blocks_per_row, shift);\n        top_k_parallel_radix_select<BLOCK_SIZE, RADIX_BITS>\n',
            mode="replace",
            guard='top_k_parallel_radix_histogram<BLOCK_SIZE,\\ RADIX_BITS>',
            rationale='nro07-topk hunk 14: upstream lines 197-200 -> result lines 759-762',
            max_span_lines=6,
        ),
        Edit(
            id='nro07-topk-15',
            anchor='\\ \\ \\ \\ top_k_radix_reset_counters\\\n\\ \\ \\ \\ \\ \\ \\ \\ <<<\\(nrows\\ \\+\\ BLOCK_SIZE\\ \\-\\ 1\\)\\ /\\ BLOCK_SIZE,\\ BLOCK_SIZE,\\ 0,\\ stream>>>\\(states,\\ nrows\\);\\\n\\ \\ \\ \\ top_k_radix_gather<BLOCK_SIZE>\\\n',
            text='    top_k_parallel_radix_reset_counters\n        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);\n    top_k_parallel_radix_gather<BLOCK_SIZE>\n',
            mode="replace",
            guard='top_k_parallel_radix_gather<BLOCK_SIZE>',
            rationale='nro07-topk hunk 15: upstream lines 204-206 -> result lines 766-768',
            max_span_lines=5,
        ),
        Edit(
            id='nro07-topk-16',
            anchor='\\#endif\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n',
            text='static bool top_k_use_small_kernel(int ncols, int nrows, int k) {\n    if (k == 1) {\n        return true;\n    }\n#if HIP_VERSION >= 71500000\n    if (ncols <= 1024) {\n        return true;\n    }\n    const uint64_t elements = (uint64_t) ncols * nrows;\n    if (k <= 32) {\n        return elements <= (1U << 20);\n    }\n    return nrows == 1 && ncols <= (1U << 17);\n#else\n    GGML_UNUSED(ncols);\n    GGML_UNUSED(nrows);\n    return false;\n#endif\n}\n#endif\n',
            mode="replace",
            guard='static\\ bool\\ top_k_use_small_kernel\\(int\\ ncols,\\ int\\ nrows,\\ int\\ k\\)\\ \\{',
            rationale='nro07-topk hunk 16: upstream lines 211-211 -> result lines 773-792',
            max_span_lines=3,
        ),
        Edit(
            id='nro07-topk-17',
            anchor='\\ \\ \\ \\ ggml_cuda_pool\\ \\&\\ pool\\ \\ =\\ ctx\\.pool\\(\\);\\\n\\#ifdef\\ CUB_TOP_K_AVAILABLE\\\n',
            text='#ifdef CUB_TOP_K_AVAILABLE\n    ggml_cuda_pool & pool = ctx.pool();\n',
            mode="replace",
            guard='ggml_cuda_pool\\ \\&\\ pool\\ =\\ ctx\\.pool\\(\\);',
            rationale='nro07-topk hunk 17: upstream lines 227-228 -> result lines 808-809',
            max_span_lines=4,
        ),
        Edit(
            id='nro07-topk-18',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ top_k_cub\\(pool,\\ src0_d\\ \\+\\ i\\ \\*\\ ncols,\\ dst_d\\ \\+\\ i\\ \\*\\ k,\\ ncols,\\ k,\\ stream\\);\\\n\\ \\ \\ \\ \\}\\\n\\#elif\\ defined\\(GGML_CUDA_USE_CUB\\)\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ const\\ int\\ \\ \\ \\ ncols_pad\\ \\ \\ \\ \\ \\ =\\ next_power_of_2\\(ncols\\);\\\n',
            text='        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);\n    }\n#elif defined(GGML_USE_HIP)\n    ggml_cuda_pool & pool = ctx.pool();\n    if (top_k_use_small_kernel(ncols, nrows, k)) {\n        if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {\n            static std::once_flag bigcherry_topk_small_logged;\n            std::call_once(bigcherry_topk_small_logged, [] {\n                GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1256_nro07 path=topk_small\\n");\n            });\n        }\n        top_k_small_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);\n    } else if (ncols > 1024) {\n        if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {\n            static std::once_flag bigcherry_topk_parallel_radix_logged;\n            std::call_once(bigcherry_topk_parallel_radix_logged, [] {\n                GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1256_nro07 path=topk_parallel_radix\\n");\n            });\n        }\n        top_k_parallel_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);\n    } else {\n        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);\n        int * tmp_dst = temp_dst_alloc.get();\n        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);\n        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,\n                                     cudaMemcpyDeviceToDevice, stream));\n    }\n#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE\n    ggml_cuda_pool & pool = ctx.pool();\n    // Fall back to argsort + copy\n    const int    ncols_pad      = next_power_of_2(ncols);\n',
            mode="replace",
            guard='\\#elif\\ defined\\(GGML_USE_HIP\\)',
            rationale='nro07-topk hunk 18: upstream lines 235-235 -> result lines 816-842',
            max_span_lines=7,
        ),
        Edit(
            id='nro07-topk-19',
            anchor='\\#if\\ defined\\(GGML_USE_HIP\\)\\\n\\ \\ \\ \\ if\\ \\(ncols\\ >\\ 1024\\)\\ \\{\\\n\\ \\ \\ \\ \\ \\ \\ \\ top_k_radix_cuda\\(pool,\\ src0_d,\\ dst_d,\\ ncols,\\ nrows,\\ k,\\ stream\\);\\\n\\ \\ \\ \\ \\}\\ else\\ \\{\\\n\\#endif\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ \\ \\ \\ \\ ggml_cuda_pool_alloc<int>\\ temp_dst_alloc\\(pool,\\ ncols\\ \\*\\ nrows\\);\\\n\\ \\ \\ \\ \\ \\ \\ \\ int\\ \\*\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ tmp_dst\\ =\\ temp_dst_alloc\\.get\\(\\);\\\n\\ \\ \\ \\ \\ \\ \\ \\ argsort_f32_i32_cuda_bitonic\\(src0_d,\\ tmp_dst,\\ ncols,\\ nrows,\\ GGML_SORT_ORDER_DESC,\\ stream\\);\\\n\\ \\ \\ \\ \\ \\ \\ \\ CUDA_CHECK\\(cudaMemcpy2DAsync\\(dst_d,\\ k\\ \\*\\ sizeof\\(int\\),\\ tmp_dst,\\ ncols\\ \\*\\ sizeof\\(int\\),\\ k\\ \\*\\ sizeof\\(int\\),\\ nrows,\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ cudaMemcpyDeviceToDevice,\\ stream\\)\\);\\\n\\#if\\ defined\\(GGML_USE_HIP\\)\\\n\\ \\ \\ \\ \\}\\\n\\#endif\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n',
            text='    ggml_cuda_pool & pool = ctx.pool();\n    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);\n    int *                     tmp_dst = temp_dst_alloc.get();\n    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);\n    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,\n                                 cudaMemcpyDeviceToDevice, stream));\n',
            mode="replace",
            guard='\\ \\ \\ \\ ggml_cuda_pool\\ \\&\\ pool\\ =\\ ctx\\.pool\\(\\);\\\n\\ \\ \\ \\ ggml_cuda_pool_alloc<int>\\ temp_dst_alloc\\(pool,\\ ncols\\ \\*\\ nrows\\);\\\n\\ \\ \\ \\ int\\ \\*\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ tmp_dst\\ =\\ temp_dst_alloc\\.get\\(\\);\\\n\\ \\ \\ \\ argsort_f32_i32_cuda_bitonic\\(src0_d,\\ tmp_dst,\\ ncols,\\ nrows,\\ GGML_SORT_ORDER_DESC,\\ stream\\);\\\n\\ \\ \\ \\ CUDA_CHECK\\(cudaMemcpy2DAsync\\(dst_d,\\ k\\ \\*\\ sizeof\\(int\\),\\ tmp_dst,\\ ncols\\ \\*\\ sizeof\\(int\\),\\ k\\ \\*\\ sizeof\\(int\\),\\ nrows,\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ cudaMemcpyDeviceToDevice,\\ stream\\)\\);\\\n',
            rationale='nro07-topk hunk 19: upstream lines 261-273 -> result lines 868-873',
            max_span_lines=15,
        ),
    ),
)

CMAKE_WAVE64 = FilePatch(
    path='ggml/src/ggml-hip/CMakeLists.txt',
    description='compile top-k.cu as wave64 on all-gfx11/gfx12 builds (nasone 7f3e1e4d)',
    edits=(
        Edit(
            id='nro07-cmake-01',
            anchor='endif\\(\\)\\\n\\\nif\\ \\(GGML_STATIC\\)\\\n\\ \\ \\ \\ message\\(FATAL_ERROR\\ "Static\\ linking\\ not\\ supported\\ for\\ HIP/ROCm"\\)\\\n',
            text='endif()\n\nset(_GGML_HIP_TOP_K_WAVE64 TRUE)\nif (WIN32 OR NOT CMAKE_HIP_ARCHITECTURES)\n    set(_GGML_HIP_TOP_K_WAVE64 FALSE)\nendif()\nforeach(_GGML_HIP_ARCH IN LISTS CMAKE_HIP_ARCHITECTURES)\n    if (NOT _GGML_HIP_ARCH MATCHES "^gfx11" AND NOT _GGML_HIP_ARCH MATCHES "^gfx120[01]$" AND NOT _GGML_HIP_ARCH STREQUAL "gfx12-generic")\n        set(_GGML_HIP_TOP_K_WAVE64 FALSE)\n    endif()\nendforeach()\nif (_GGML_HIP_TOP_K_WAVE64)\n    set_property(SOURCE ../ggml-cuda/top-k.cu APPEND PROPERTY COMPILE_OPTIONS -mwavefrontsize64)\nendif()\nunset(_GGML_HIP_ARCH)\nunset(_GGML_HIP_TOP_K_WAVE64)\n\nif (GGML_STATIC)\n    message(FATAL_ERROR "Static linking not supported for HIP/ROCm")\n',
            mode="replace",
            guard='set\\(_GGML_HIP_TOP_K_WAVE64\\ TRUE\\)',
            rationale='nro07-cmake hunk 1: upstream lines 136-135 -> result lines 136-150',
            max_span_lines=6,
        ),
    ),
)

PATCHES = [TOPK_HYBRID, CMAKE_WAVE64]
