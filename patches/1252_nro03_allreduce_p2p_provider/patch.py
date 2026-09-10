"""NRO03 draft: source-current push primitive for a future HIP P2P AR provider."""

from bigcherry.patcher import Edit, FilePatch

DRAFT_SOURCE_CONTEXT = {
    "repo": "https://github.com/nasone32/llama.cpp-RDNA3-7900xtx-opt",
    "commit": "7c5bb5cb991670676b89cddc5077c9456c5cf70e",
    "adaptation": "fixed issuer rejected; all directions must be source-current push",
    "pin": "b10705",
}

_HELPER = r'''

// BIGCHERRY_NRO03_P2P_SCAFFOLD_BEGIN
// Correctness invariant from real gfx1100 evidence: issue every peer transfer
// with the SOURCE device current. Do not replace this with one fixed issuer.
static cudaError_t bigcherry_nro03_peer_push_async(
        void * dst, int dst_device,
        const void * src, int src_device,
        size_t bytes, cudaStream_t source_stream) {
    ggml_cuda_set_device(src_device);
    return cudaMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, source_stream);
}
// BIGCHERRY_NRO03_P2P_SCAFFOLD_END
'''

PATCHES = [FilePatch(
    path="ggml/src/ggml-cuda/allreduce.cu",
    description="add default-off P2P policy and source-current push helper",
    edits=(
        Edit(
            id="p2p-request-field",
            anchor=r"^    uint64_t call_count;$",
            mode="insert_before",
            text="    bool     nro03_p2p_requested; // request only; no live dispatch in draft\n",
            guard=r"nro03_p2p_requested",
            rationale="provider instance owns the opt-in state",
        ),
        Edit(
            id="p2p-request-init",
            anchor=r"^    p->bf16_threshold\s*=",
            mode="insert_after",
            text="\n    p->nro03_p2p_requested = ggml_cuda_ar_env_u64(\"GGML_CUDA_AR_P2P\", 0) != 0;",
            guard=r"p->nro03_p2p_requested = ggml_cuda_ar_env_u64",
            rationale="anchor on the uniquely named BF16 policy assignment; P2P remains explicitly opt-in",
        ),
        Edit(
            id="source-current-push-helper",
            anchor=r"^ggml_cuda_ar_pipeline \* ggml_cuda_ar_pipeline_init\(const int \* devices, size_t n_devices\) \{$",
            mode="insert_before",
            text=_HELPER + "\n",
            guard=r"BIGCHERRY_NRO03_P2P_SCAFFOLD_BEGIN",
            rationale="place the transport primitive immediately before provider initialization using a code anchor",
        ),
    ),
)]
