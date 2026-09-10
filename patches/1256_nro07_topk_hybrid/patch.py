"""NRO07 draft: HIP hybrid TOP_K semantic/policy scaffold, runtime unwired."""

from bigcherry.patcher import Edit, FilePatch

DRAFT_SOURCE_CONTEXT = {
    "repo": "https://github.com/nasone32/llama.cpp-RDNA3-7900xtx-opt",
    "commit": "7f3e1e4d0b166cb681b2c01503370e610a5b423d",
    "title": "ROCm: add hybrid TOP_K kernels",
    "pin": "b10705",
}

_TEXT = r'''

#if defined(GGML_USE_HIP)
// BIGCHERRY_NRO07_TOPK_HYBRID_SCAFFOLD_BEGIN
static __device__ __forceinline__ uint32_t bigcherry_nro07_top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000U) != 0 ? ~bits : bits | 0x80000000U;
}

static bool bigcherry_nro07_top_k_hybrid_requested() {
    const char * env = getenv("GGML_CUDA_TOPK_HYBRID");
    return env != nullptr && atoi(env) != 0;
}
// Selection kernels/dispatch intentionally await exact-index fixture coverage.
// BIGCHERRY_NRO07_TOPK_HYBRID_SCAFFOLD_END
#endif
'''

PATCHES = [FilePatch(
    path="ggml/src/ggml-cuda/top-k.cu",
    description="stage HIP TOP_K ordered-key and selector policy without changing routing",
    edits=(Edit(
        id="hybrid-topk-scaffold",
        anchor=r"^void ggml_cuda_op_top_k\(ggml_backend_cuda_context & ctx, ggml_tensor \* dst\) \{$",
        mode="insert_before",
        text=_TEXT + "\n\n",
        guard=r"BIGCHERRY_NRO07_TOPK_HYBRID_SCAFFOLD_BEGIN",
        rationale="place HIP-only source primitives immediately before TOP_K dispatch",
    ),),
)]
