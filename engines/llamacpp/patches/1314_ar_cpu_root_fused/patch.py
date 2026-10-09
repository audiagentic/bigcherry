"""1314 (QFP13/QFP11): one kernel per rank per small cpu-root AllReduce instead of produce + consume.

1291's small-message cpu-root AllReduce enqueues two kernels per rank per call: bc_cpu_root_produce (copy the slice to
pinned host memory, publish the arrive epoch) and bc_cpu_root_consume (spin on the CPU's done epoch, copy the result
back). Flash-Next decode makes ~30 such calls per generated token per GPU, so the pair costs ~30 extra launches per
token on a launch-gap-bound decode. With BIGCHERRY_AR_FUSED=1, bc_cpu_root_fused does both in one launch: the same
produce code, then thread 0 spins on the done epoch for the generation it just published (no ctr re-read), then the
same consume copy. In-place (dst == src) is safe because every read of src completes before the __syncthreads that
precedes the spin, and dst is written only after it. The CPU worker, epochs, slots and result buffers are unchanged,
so the arithmetic (CPU fixed-order f32 sum) and the output are identical. Requires 1291.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_KERNEL_ANCHOR = "static void bc_cpu_root_worker(bc_cpu_root * cr) {\n"

_KERNEL = r"""// bigcherry 1314: produce + consume in one launch (BIGCHERRY_AR_FUSED=1); same buffers, epochs and copies.
static __global__ void bc_cpu_root_fused(float * __restrict__ data, float * slots2, const float * result2,
        bc_cpu_root_line * ctr, bc_cpu_root_line * line, const bc_cpu_root_line * done, size_t max_elems, int n,
        int contributes) {
    __shared__ uint32_t s_gen;
    if (threadIdx.x == 0) {
        s_gen = ((volatile bc_cpu_root_line *) ctr)->v + 1;
        ((volatile bc_cpu_root_line *) ctr)->v = s_gen;
    }
    __syncthreads();
    const uint32_t gen = s_gen;
    float * slot = slots2 + (size_t) (gen & 1) * max_elems;
    if (!contributes) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            slot[i] = 0.0f;
        }
    } else {
        const bool aligned = ((((uintptr_t) data) | ((uintptr_t) slot)) & 15) == 0;
        const int n4 = aligned ? n / 4 : 0;
        const float4 * s4 = (const float4 *) data;
        float4 * d4 = (float4 *) slot;
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {
            d4[i] = s4[i];
        }
        for (int i = n4 * 4 + threadIdx.x; i < n; i += blockDim.x) {
            slot[i] = data[i];
        }
    }
    __syncthreads();  // every read of data (src) is done before anything writes it (dst) below
    if (threadIdx.x == 0) {
        line->n = (uint32_t) n;
        __threadfence_system();
        __hip_atomic_store((uint32_t *) &line->v, gen, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        while ((int32_t) (__hip_atomic_load((const uint32_t *) &done->v, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) - gen) < 0) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
    __threadfence_system();
    const float * res = result2 + (size_t) (gen & 1) * max_elems;
    const bool aligned = ((((uintptr_t) res) | ((uintptr_t) data)) & 15) == 0;
    const int n4 = aligned ? n / 4 : 0;
    const float4 * r4 = (const float4 *) res;
    float4 * d4 = (float4 *) data;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        d4[i] = r4[i];
    }
    for (int i = n4 * 4 + threadIdx.x; i < n; i += blockDim.x) {
        data[i] = res[i];
    }
}

"""

_LAUNCH_OLD = """        bc_cpu_root_produce<<<1, 1024, 0, stream>>>(data, cr->slots + i * 2 * cr->max_elems,
            cr->ctr + i, cr->arrive + i, cr->max_elems, n, (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0);
        bc_cpu_root_consume<<<1, 1024, 0, stream>>>(data, cr->result, cr->ctr + i, cr->done, cr->max_elems, n);
"""

_LAUNCH_NEW = """        static const bool bc_ar_fused = getenv("BIGCHERRY_AR_FUSED") != nullptr && atoi(getenv("BIGCHERRY_AR_FUSED")) != 0;
        if (bc_ar_fused) {  // bigcherry 1314: one launch per rank
            bc_cpu_root_fused<<<1, 1024, 0, stream>>>(data, cr->slots + i * 2 * cr->max_elems, cr->result,
                cr->ctr + i, cr->arrive + i, cr->done, cr->max_elems, n, (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0);
        } else {
        bc_cpu_root_produce<<<1, 1024, 0, stream>>>(data, cr->slots + i * 2 * cr->max_elems,
            cr->ctr + i, cr->arrive + i, cr->max_elems, n, (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0);
        bc_cpu_root_consume<<<1, 1024, 0, stream>>>(data, cr->result, cr->ctr + i, cr->done, cr->max_elems, n);
        }
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="1314: fused produce+consume kernel for 1291's small cpu-root AllReduce (BIGCHERRY_AR_FUSED=1)",
        language="none",
        edits=(
            Edit(
                id="ar-fused-kernel",
                anchor=re.escape(_KERNEL_ANCHOR),
                mode="insert_before",
                text=_KERNEL,
                guard=r"static __global__ void bc_cpu_root_fused\(",
                rationale="1291's CPU worker follows its produce/consume kernels; the fused kernel goes with them.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="ar-fused-launch",
                anchor=re.escape(_LAUNCH_OLD),
                mode="replace",
                text=_LAUNCH_NEW,
                guard=r"bigcherry 1314: one launch per rank",
                rationale="1291's per-rank small-path launch pair.",
                expect_matches=1,
                max_span_lines=4,
            ),
        ),
    ),
]
