"""NRO02 draft: residual-capable AllReduce finish kernels.

Graph matching and compute-node skipping are intentionally not wired in this
first draft; the kernel primitives can be fixture-tested independently.
"""

from bigcherry.patcher import Edit, FilePatch

DRAFT_SOURCE_CONTEXT = {
    "parent": "NRO01",
    "source-commit": "e06dcf6300718227cb8cfda9e61fb12ccb693418",
    "scope": "residual finish arithmetic only",
}

_TEXT = r'''

// BIGCHERRY_NRO02_RESIDUAL_SCAFFOLD_BEGIN
// Draft only: Meta graph matching/node skipping is not yet reachable.
template <typename T_dst, typename T_src>
static __global__ void bigcherry_nro02_add_residual_kernel(
        T_dst * __restrict__ dst,
        const T_src * __restrict__ peer,
        const T_dst * __restrict__ residual,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const T_src local_low = ggml_cuda_cast<T_src>(dst[i]);
        const float sum = ggml_cuda_cast<float>(local_low) + ggml_cuda_cast<float>(peer[i]);
        dst[i] = ggml_cuda_cast<T_dst>(sum + ggml_cuda_cast<float>(residual[i]));
    }
}

static __global__ void bigcherry_nro02_q8_0_add_residual_kernel(
        float * __restrict__ dst,
        const block_q8_0 * __restrict__ rank0,
        const block_q8_0 * __restrict__ rank1,
        const float * __restrict__ residual,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const int ib = i / QK8_0;
        const int iq = i % QK8_0;
        const float a = (float) rank0[ib].d * (float) rank0[ib].qs[iq];
        const float b = (float) rank1[ib].d * (float) rank1[ib].qs[iq];
        dst[i] = a + b + residual[i];
    }
}
// BIGCHERRY_NRO02_RESIDUAL_SCAFFOLD_END
'''

PATCHES = [FilePatch(
    path="ggml/src/ggml-cuda/allreduce.cu",
    description="add fixture-testable residual AllReduce finish primitives",
    edits=(Edit(
        id="add-residual-kernel-primitives",
        anchor=r"^struct ggml_cuda_ar_pipeline \{$",
        mode="insert_before",
        text=_TEXT + "\n",
        guard=r"BIGCHERRY_NRO02_RESIDUAL_SCAFFOLD_BEGIN",
        rationale="insert the child primitives before provider state using a stable code anchor; NRO01 remains a declared dependency",
    ),),
)]
