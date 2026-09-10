"""NRO08 draft: wave32 TOP-1 reduction primitive, intentionally unwired."""

from bigcherry.patcher import Edit, FilePatch

DRAFT_SOURCE_CONTEXT = {
    "repo": "https://github.com/nasone32/llama.cpp-RDNA3-7900xtx-opt",
    "commit": "7f1d25f7e05cb34053d317a0f72f601d846c8662",
    "title": "ROCm: make TOP_K wave32-native",
    "parent": "NRO07",
}

_TEXT = r'''

#if defined(GGML_USE_HIP)
// BIGCHERRY_NRO08_TOPK_WAVE32_SCAFFOLD_BEGIN
template <int BLOCK_SIZE>
static __device__ __forceinline__ int2 bigcherry_nro08_top_k_one_wave32_reduce(
        int2 * wave_candidates, int2 value, int original_ncols) {
    static_assert(BLOCK_SIZE % 32 == 0, "NRO08 requires whole wave32 groups");
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    constexpr int wave_count = BLOCK_SIZE / 32;
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        const int2 other = make_int2(__shfl_down(value.x, offset, 32), __shfl_down(value.y, offset, 32));
        if (lane + offset < 32 &&
                (value.x >= original_ncols ||
                 (other.x < original_ncols && __int_as_float(other.y) > __int_as_float(value.y)))) {
            value = other;
        }
    }
    if (lane == 0) wave_candidates[wave] = value;
    __syncthreads();
    if (wave == 0) {
        value = lane < wave_count ? wave_candidates[lane] : make_int2(original_ncols, (int) 0xff800000U);
#pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            const int2 other = make_int2(__shfl_down(value.x, offset, 32), __shfl_down(value.y, offset, 32));
            if (lane + offset < 32 &&
                    (value.x >= original_ncols ||
                     (other.x < original_ncols && __int_as_float(other.y) > __int_as_float(value.y)))) {
                value = other;
            }
        }
    }
    return value;
}
// Runtime NRO07 dispatch is intentionally unchanged in this draft.
// BIGCHERRY_NRO08_TOPK_WAVE32_SCAFFOLD_END
#endif
'''

PATCHES = [FilePatch(
    path="ggml/src/ggml-cuda/top-k.cu",
    description="add an unwired wave32 TOP-1 reduction primitive after the NRO07 source unit",
    edits=(Edit(
        id="wave32-top1-primitive",
        anchor=r"^void ggml_cuda_op_top_k\(ggml_backend_cuda_context & ctx, ggml_tensor \* dst\) \{$",
        mode="insert_before",
        text=_TEXT + "\n\n",
        guard=r"BIGCHERRY_NRO08_TOPK_WAVE32_SCAFFOLD_BEGIN",
        rationale="NRO08 applies after NRO07 and uses the unchanged TOP_K dispatch function as a stable code anchor",
    ),),
)]
