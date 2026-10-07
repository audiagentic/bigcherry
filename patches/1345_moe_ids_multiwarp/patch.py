"""1345: MoE routing helper with several warps per expert for large batches (QFP36).

ggml_cuda_launch_mm_ids_helper (ggml-cuda/mmid.cu) turns the routed expert ids of a batch into the compact row lists
the grouped MMQ needs (ids_src1, ids_dst, expert_bounds). It launches one block per expert and each block is ONE warp
that walks every token of the batch in sequence: for 512 tokens and top-10 routing that is 256 dependent iterations
per expert, on 32 threads, for each of the 512 experts. In the Flash-Next prefill profile (production build, 38.7K
tokens) mm_ids_helper<10> is about 3.8% of all kernel time on the three target cards.

This adds a multi-warp variant for the specialised top-k counts (2/4/6/8/10/16/32): a block has 8 warps, each warp
walks its own contiguous slice of the tokens with exactly the native per-warp code and keeps its matches in its own
part of the shared store; the warps then exchange their counts once and write their rows at the right offsets. The
rows of an expert stay in ascending token order, so ids_src1 (forward and inverse form), ids_dst and expert_bounds are
byte-identical to the native helper's - everything downstream sees the same input.

On by default for batches of at least 128 tokens (BIGCHERRY_MOE_IDS_MULTIWARP=0 restores the native helper); smaller
batches, the generic top-k path and devices whose warp size is not the compiled one use the native helper.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_A_LAUNCH_TEMPLATE = "template <int n_expert_used_template>\nstatic void launch_mm_ids_helper(\n"
_N_KERNEL = r"""// bigcherry 1345 (QFP36): mm_ids_helper with n_warps warps per expert. Warp w walks the tokens
// [w*chunk, (w + 1)*chunk) with the native per-warp code and keeps its matches in store[w*chunk ...]; the warps then
// exchange their counts and write their rows behind those of the lower warps, so every output is the native one.
template <int n_expert_used_template, int n_warps>
__launch_bounds__(n_warps*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void bc_mm_ids_helper_mw(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int chunk, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size     = ggml_cuda_get_physical_warp_size();
    constexpr int n_expert_used = n_expert_used_template;
    constexpr int neu_padded    = mm_ids_pow2<n_expert_used_template>::value;
    static_assert(n_expert_used_template > 0, "the generic top-k path stays on the native helper");
    static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");

    const int expert = blockIdx.x;
    const int lane   = threadIdx.x % warp_size;
    const int warp   = threadIdx.x / warp_size;

    extern __shared__ char data_bc_mm_ids_helper_mw[];
    mm_ids_helper_store * store  = (mm_ids_helper_store *) data_bc_mm_ids_helper_mw;
    int                 * counts = (int *) (store + n_warps*chunk); // [n_warps] rows, then [n_warps] lower-expert counts

    const int t_begin = warp*chunk;
    const int t_end   = t_begin + chunk < n_tokens ? t_begin + chunk : n_tokens;
    mm_ids_helper_store * store_w = store + t_begin;

    int nex_prev   = 0; // Number of columns for experts with a lower index, in this warp's tokens.
    int it_compact = 0; // Running index for this warp's part of the compact slice of this expert.

    for (int it0 = t_begin; it0 < t_end; it0 += warp_size/neu_padded) {
        const int it = it0 + lane / neu_padded;

        const int iex = lane % neu_padded; // The index at which the expert is used, if any.
        const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < t_end ?
            ids[it*si1 + iex] : INT_MAX;
        const int iex_used = expert_used == expert ? iex : -1;
        nex_prev += expert_used < expert;

        // Whether the threads at this token position have used the expert:
        const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

        // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
        int it_compact_add_lower = 0;
#pragma unroll
        for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
            const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
            if (lane >= offset) {
                it_compact_add_lower += tmp;
            }
        }

        if (iex_used != -1) {
            store_w[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
        }

        // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
        it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    if (lane == 0) {
        counts[warp]           = it_compact;
        counts[n_warps + warp] = nex_prev;
    }
    __syncthreads();

    int row_base     = 0; // rows of this expert written by the lower warps
    int row_total    = 0;
    int nex_prev_all = 0;
    for (int w = 0; w < n_warps; ++w) {
        row_base     += w < warp ? counts[w] : 0;
        row_total    += counts[w];
        nex_prev_all += counts[n_warps + w];
    }

    for (int itc = lane; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store_w[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        const int row      = nex_prev_all + row_base + itc;
        ids_dst[row] = it*n_expert_used + iex_used;
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = row;
        } else {
            ids_src1[row] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev_all;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev_all + row_total;
}

// bigcherry 1345: on by default, BIGCHERRY_MOE_IDS_MULTIWARP=0 restores the native helper
static bool bc_mm_ids_multiwarp() {
    static const bool on = [] {
        const char * s = getenv("BIGCHERRY_MOE_IDS_MULTIWARP");
        return s == nullptr || atoi(s) != 0;
    }();
    return on;
}

// bigcherry 1345: returns false when the batch has to take the native helper
template <int n_expert_used_template>
static bool bc_launch_mm_ids_helper_mw(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    constexpr int n_warps         = 8;
    constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int tokens_per_iter = warp_size/mm_ids_pow2<n_expert_used_template>::value;

    const int id = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    if (ggml_cuda_info().devices[id].warp_size != warp_size || n_tokens >= (1 << 22)) {
        return false;
    }

    int chunk = (n_tokens + n_warps - 1)/n_warps;
    chunk = (chunk + tokens_per_iter - 1)/tokens_per_iter*tokens_per_iter;
    const size_t nbytes_shared = (size_t) n_warps*chunk*sizeof(mm_ids_helper_store) + 2*n_warps*sizeof(int);
    if (nbytes_shared > smpbo) {
        return false;
    }
    CUDA_SET_SHARED_MEMORY_LIMIT((bc_mm_ids_helper_mw<n_expert_used_template, n_warps>), smpbo);

    static bool bc_logged = false;
    if (!bc_logged && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
        bc_logged = true;
        fprintf(stderr, "BIGCHERRY_PATCH_HIT patch=1345_moe_ids_multiwarp used=%d experts=%d tokens=%d warps=%d inverse=%d\n",
                n_expert_used_template, n_experts, n_tokens, n_warps, write_inverse ? 1 : 0);
    }

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(n_warps*warp_size, 1, 1);
    bc_mm_ids_helper_mw<n_expert_used_template, n_warps><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, chunk, nchannels_y, si1, sis1, write_inverse);
    return true;
}

"""

_A_SWITCH = "    switch (n_expert_used) {\n        case  2:\n            launch_mm_ids_helper< 2>("
_N_SWITCH = r"""    // bigcherry 1345 (QFP36): large batches group their rows with several warps per expert (same outputs)
    if (bc_mm_ids_multiwarp() && n_tokens >= 128) {
        bool bc_done = false;
        switch (n_expert_used) {
            case  2: bc_done = bc_launch_mm_ids_helper_mw< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, nchannels_y, si1, sis1, write_inverse, stream); break;
            case  4: bc_done = bc_launch_mm_ids_helper_mw< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, nchannels_y, si1, sis1, write_inverse, stream); break;
            case  6: bc_done = bc_launch_mm_ids_helper_mw< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, nchannels_y, si1, sis1, write_inverse, stream); break;
            case  8: bc_done = bc_launch_mm_ids_helper_mw< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, nchannels_y, si1, sis1, write_inverse, stream); break;
            case 10: bc_done = bc_launch_mm_ids_helper_mw<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, nchannels_y, si1, sis1, write_inverse, stream); break;
            case 16: bc_done = bc_launch_mm_ids_helper_mw<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, nchannels_y, si1, sis1, write_inverse, stream); break;
            case 32: bc_done = bc_launch_mm_ids_helper_mw<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, nchannels_y, si1, sis1, write_inverse, stream); break;
            default: break;
        }
        if (bc_done) {
            return;
        }
    }

""" + _A_SWITCH

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/mmid.cu",
        description="1345: MoE routing helper with 8 warps per expert for batches of 128 tokens and more",
        language="none",
        edits=(
            Edit(
                id="ids-mw-include",
                anchor=re.escape('#include "mmid.cuh"\n'),
                mode="insert_after",
                text="\n#include <cstdio>   // bigcherry 1345: fprintf\n#include <cstdlib>  // bigcherry 1345: getenv / atoi\n",
                guard=r"#include <cstdlib>  // bigcherry 1345",
                rationale="The file's own header include, the last include of mmid.cu.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="ids-mw-kernel",
                anchor=re.escape(_A_LAUNCH_TEMPLATE),
                mode="insert_before",
                text=_N_KERNEL,
                guard=r"static __global__ void bc_mm_ids_helper_mw\(",
                rationale="After the native helper kernel (and the store / pow2 helpers it shares) and before the native launcher.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="ids-mw-dispatch",
                anchor=re.escape(_A_SWITCH),
                mode="replace",
                text=_N_SWITCH,
                guard=r"bigcherry 1345 \(QFP36\): large batches group their rows with several warps per expert",
                rationale="The start of the public launcher's top-k switch; the multi-warp attempt goes in front of it.",
                expect_matches=1,
                max_span_lines=4,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_MOE_IDS_MULTIWARP", "0|1", "1 (on)",
           "MoE routing helper (expert row lists for grouped MMQ) runs 8 warps per expert for batches of 128 tokens and "
           "more, with byte-identical outputs; 0 restores the native one-warp helper"),
)
