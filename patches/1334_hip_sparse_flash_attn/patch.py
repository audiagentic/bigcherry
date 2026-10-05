"""1334 (QFP25): upstream's sparse flash attention on RDNA WMMA (BIGCHERRY_FA_SPARSE=1).

Qwen4Exp QSA attention is dense flash attention over all n_kv cells with a -inf mask; each query can see about
indexer_top_k (2048) cells. Upstream already has the sparse form for it: the mask is compacted into one index list per
tile of queries (the union of the cells those queries can see, flash_attn_mask_to_sparse_indices) and the MMA kernel
gathers only those K/V cells (flash_attn_ext_f16<..., use_sparse = true>, instances (256, 256, 1, 8) and
(256, 256, 8, 8)). It is compiled out for HIP in five places and its selection requires an NVIDIA card, so on AMD the
kernel reads the whole cache: flash attention costs 32 ms per 512-token ubatch averaged over the first 32K tokens of
a Flash-Next prefill and 97 ms averaged over 100K (21% of each XTX's kernel time there, and the R9700, which holds no
attention, idles in the all-reduce meanwhile).

This patch enables that path on HIP for RDNA3/RDNA4 WMMA behind BIGCHERRY_FA_SPARSE=1:

1. A HIP version of the index kernel. Upstream's uses __ballot_sync / __popc over 32-lane warps; the HIP one keeps the
   same output (ascending column order, -1 fill, per-list count) with a per-thread count and a block prefix sum, so it
   does not depend on the wave size.
2. ggml_cuda_flash_attn_ext_compact_mask and ..._shall_use_sparse are compiled for HIP; the selection accepts
   amd_wmma_available(cc) when the flag is set (NVIDIA selection unchanged).
3. The sparse kernels exist at ncols2 = 8 only. RDNA picks ncols2 by exact GQA divisibility (Qwen4Exp: 24 heads over
   2 KV heads, ratio 12 -> ncols2 = 4), so when the sparse path would be taken the RDNA branch selects ncols2 = 8 the
   way the generic rule does for a ratio above 4.
4. The two dispatch sites (switch_ncols1 and the kernel-pointer selection) are compiled for HIP.

Off (default): unchanged kernels and selection. The dense path remains the fallback for every shape the sparse
selection rejects (no mask, no n_kv_max, ALiBi, softcap, small caches).
"""

import re as _re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "core"
STATE = "untested"

_A_KERNEL = ('#include "fattn.cuh"\n'
             "\n"
             "#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)\n"
             "// one list per group of ncols1 queries: a column is selected if any query of the group can see it\n")
_N_KERNEL = r'''#include "fattn.cuh"

#if defined(GGML_USE_HIP)
#include <cstdlib>

// BigCherry 1334: sparse flash attention on RDNA WMMA is opt-in
static bool bc_fa_sparse_enabled() {
    static const bool on = getenv("BIGCHERRY_FA_SPARSE") != nullptr && atoi(getenv("BIGCHERRY_FA_SPARSE")) != 0;
    return on;
}

// BigCherry 1334: HIP version of the mask compaction below - one list per group of ncols1 queries, a column is
// selected if any query of the group can see it. Same output (ascending columns, -1 fill, count) without warp
// ballots, so it does not depend on the wave size: each thread owns 8 consecutive columns per pass and the block
// prefix-sums the per-thread counts.
template <int ncols1, bool oob>
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, int32_t * counts_ptr, const int ne30, const int n_queries,
        const int n_kv_max, const int64_t s31, const int64_t s33) {
    constexpr int values_per_thread = 8;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int group    = blockIdx.x;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    const half * mask = mask_ptr + sequence*s33 + q0*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + group)*n_kv_max;

    __shared__ int thread_offsets[256];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_thread) {
        uint32_t selected_bits = 0;
        int      thread_count  = 0;
#pragma unroll
        for (int item = 0; item < values_per_thread; ++item) {
            const int i = i0 + tid*values_per_thread + item;
            bool selected = false;
            if (i < ne30) {
#pragma unroll
                for (int q = 0; q < ncols1; ++q) {
                    selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q*s31 + i]));
                }
            }
            selected_bits |= uint32_t(selected) << item;
            thread_count  += selected;
        }
        thread_offsets[tid] = thread_count;
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
            for (int it = 0; it < (int) blockDim.x; ++it) {
                const int count = thread_offsets[it];
                thread_offsets[it] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        int dst = row_count + thread_offsets[tid];
#pragma unroll
        for (int item = 0; item < values_per_thread; ++item) {
            if (selected_bits & (uint32_t(1) << item)) {
                if (dst < n_kv_max) {
                    indices[dst] = i0 + tid*values_per_thread + item;
                }
                dst++;
            }
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = min(row_count, n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    if (tid == 0) {
        counts_ptr[int64_t(sequence)*gridDim.x + group] = count;
    }
}
#endif // defined(GGML_USE_HIP)

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// one list per group of ncols1 queries: a column is selected if any query of the group can see it
'''

_A_COMPACT = ("#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)\n"
              "    GGML_UNUSED_VARS(mask, indices, counts, n_queries, ncols1, n_kv_max, stream);\n"
              '    GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");\n')
_N_COMPACT = ("#if defined(GGML_USE_MUSA)  // BigCherry 1334: HIP has its own index kernel above\n"
              "    GGML_UNUSED_VARS(mask, indices, counts, n_queries, ncols1, n_kv_max, stream);\n"
              '    GGML_ABORT("sparse flash attention is not supported on MUSA");\n')

_A_SHALL = ("#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)\n"
            "    GGML_UNUSED_VARS(cc, dst, ncols1, ncols2);\n"
            "    return false;\n")
_N_SHALL = ("#if defined(GGML_USE_MUSA)  // BigCherry 1334: compiled for HIP\n"
            "    GGML_UNUSED_VARS(cc, dst, ncols1, ncols2);\n"
            "    return false;\n")

_A_ARCH = "    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&\n"
_N_ARCH = ("#if defined(GGML_USE_HIP)\n"
           "    // BigCherry 1334: the RDNA WMMA kernel runs the same sparse variant, opt-in (BIGCHERRY_FA_SPARSE=1)\n"
           "    const bool bc_arch_ok = amd_wmma_available(cc) && bc_fa_sparse_enabled();\n"
           "#else\n"
           "    const bool bc_arch_ok = GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc);\n"
           "#endif // defined(GGML_USE_HIP)\n"
           "    return bc_arch_ok &&\n")

_A_NCOLS1 = ("    const ggml_tensor * Q = dst->src[0];\n"
             "\n"
             "#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)\n"
             "    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {\n")
_N_NCOLS1 = ("    const ggml_tensor * Q = dst->src[0];\n"
             "\n"
             "#if !defined(GGML_USE_MUSA)  // BigCherry 1334: compiled for HIP\n"
             "    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {\n")

_A_RDNA = ("    // On RDNA it is preferable to minimize wasted compute vs. duplicate I/O for the mask.\n"
           "    if (amd_wmma_available(cc)) {\n")
_N_RDNA = (_A_RDNA +
           "        // BigCherry 1334: the sparse kernels exist at ncols2 = 8 only; when the sparse path would be taken, reading\n"
           "        // n_kv_max instead of n_kv cells outweighs the padded GQA tile (same choice as the generic rule below)\n"
           "        if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 8, 8)) {\n"
           "            if (use_gqa_opt && gqa_ratio > 4 && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 8, 8)) {\n"
           "                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);\n"
           "                return;\n"
           "            }\n"
           "        }\n")

_A_CASE = ("#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)\n"
           "        if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, ncols1, ncols2)) {\n"
           "            if (ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, ncols1, ncols2)) {\n")
_N_CASE = ("#if !defined(GGML_USE_MUSA)  // BigCherry 1334: compiled for HIP\n"
           "        if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, ncols1, ncols2)) {\n"
           "            if (ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, ncols1, ncols2)) {\n")

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/fattn.cu",
        description="1334: sparse flash attention index kernel and selection for HIP RDNA WMMA (BIGCHERRY_FA_SPARSE=1)",
        language="none",
        edits=(
            Edit(id="fa-sparse-hip-kernel", anchor=_re.escape(_A_KERNEL), mode="replace", text=_N_KERNEL,
                 guard=r"BigCherry 1334: HIP version of the mask compaction", rationale="Before upstream's NVIDIA-only index kernel.",
                 expect_matches=1, max_span_lines=5),
            Edit(id="fa-sparse-compact-hip", anchor=_re.escape(_A_COMPACT), mode="replace", text=_N_COMPACT,
                 guard=r"BigCherry 1334: HIP has its own index kernel above", rationale="ggml_cuda_flash_attn_ext_compact_mask HIP stub.",
                 expect_matches=1, max_span_lines=4),
            Edit(id="fa-sparse-shall-hip", anchor=_re.escape(_A_SHALL), mode="replace", text=_N_SHALL,
                 guard=r"defined\(GGML_USE_MUSA\)  // BigCherry 1334: compiled for HIP\n    GGML_UNUSED_VARS\(cc, dst, ncols1, ncols2\);",
                 rationale="shall_use_sparse HIP stub.", expect_matches=1, max_span_lines=4),
            Edit(id="fa-sparse-arch", anchor=_re.escape(_A_ARCH), mode="replace", text=_N_ARCH,
                 guard=r"const bool bc_arch_ok = amd_wmma_available\(cc\) && bc_fa_sparse_enabled\(\);",
                 rationale="Architecture term of the sparse selection.", expect_matches=1, max_span_lines=2),
            Edit(id="fa-sparse-switch-ncols1", anchor=_re.escape(_A_NCOLS1), mode="replace", text=_N_NCOLS1,
                 guard=r"defined\(GGML_USE_MUSA\)  // BigCherry 1334: compiled for HIP\n    if constexpr \(ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse\(DKQ, DV, 1, ncols2\)\)",
                 rationale="Single-query sparse dispatch in switch_ncols1.", expect_matches=1, max_span_lines=5),
            Edit(id="fa-sparse-rdna-ncols2", anchor=_re.escape(_A_RDNA), mode="replace", text=_N_RDNA,
                 guard=r"BigCherry 1334: the sparse kernels exist at ncols2 = 8 only", rationale="RDNA branch of switch_ncols2.",
                 expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/fattn-mma-f16.cuh",
        description="1334: sparse flash attention kernel selection compiled for HIP",
        language="none",
        edits=(
            Edit(id="fa-sparse-case-hip", anchor=_re.escape(_A_CASE), mode="replace", text=_N_CASE,
                 guard=r"defined\(GGML_USE_MUSA\)  // BigCherry 1334: compiled for HIP\n        if constexpr",
                 rationale="Kernel-pointer selection in ggml_cuda_flash_attn_ext_mma_f16_case.", expect_matches=1, max_span_lines=4),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_FA_SPARSE", "0|1", "0",
           "experimental: sparse flash attention on RDNA WMMA - masked attention with a per-query cell bound (Qwen4Exp QSA) "
           "reads only the cells its queries can see"),
)
