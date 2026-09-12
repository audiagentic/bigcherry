"""NRO01 draft: Q8_0 wire primitives for the internal AllReduce.

This is intentionally a compile-time/scaffolding draft, not a live selector.
It creates the source primitives required for a correctness fixture before a
lossy wire format can become reachable. NRO02 owns residual fusion.
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

DRAFT_SOURCE_CONTEXT = {
    "repo": "https://github.com/nasone32/llama.cpp-RDNA3-7900xtx-opt",
    "commit": "e06dcf6300718227cb8cfda9e61fb12ccb693418",
    "title": "ggml: add Q8 wire, residual fusion, and AllReduce tracing",
    "pin": "b10705",
    "scope": "Q8_0 wire primitives only; residual fusion split to NRO02",
}

_Q8_KERNELS = r'''

// BIGCHERRY_NRO01_Q8_SCAFFOLD_BEGIN
// Draft only: these kernels are deliberately not dispatched until NRO01's
// synthetic numerical fixture and tolerance policy are committed.
static __global__ void bigcherry_nro01_quantize_q8_0_kernel(
        const float * __restrict__ src,
        block_q8_0 * __restrict__ dst,
        int64_t ne,
        int64_t nblocks) {
    const int lane = threadIdx.x % QK8_0;
    const int64_t warp = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) / QK8_0;
    const int64_t nwarps = ((int64_t) gridDim.x * blockDim.x) / QK8_0;
    for (int64_t ib = warp; ib < nblocks; ib += nwarps) {
        const int64_t i = ib * QK8_0 + lane;
        const float x = i < ne ? src[i] : 0.0f;
        const float amax = warp_reduce_max<QK8_0>(fabsf(x));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        dst[ib].qs[lane] = (int8_t) roundf(x * id);
        if (lane == 0) {
            dst[ib].d = d;
        }
    }
}

static __global__ void bigcherry_nro01_q8_0_add_kernel(
        float * __restrict__ dst,
        const block_q8_0 * __restrict__ rank0,
        const block_q8_0 * __restrict__ rank1,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const int ib = i / QK8_0;
        const int iq = i % QK8_0;
        dst[i] = (float) rank0[ib].d * (float) rank0[ib].qs[iq]
               + (float) rank1[ib].d * (float) rank1[ib].qs[iq];
    }
}
// BIGCHERRY_NRO01_Q8_SCAFFOLD_END
'''

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/allreduce.cu",
        description="add disabled Q8_0 AllReduce wire primitives and policy state",
        edits=(
            Edit(
                # Real bug found on real hardware (2026-09-12, gfx1201 build
                # attempt): the anchor previously ended at the `=` sign, mid-
                # statement -- insert_after splices immediately after the
                # MATCHED TEXT, not after the enclosing line/statement, so
                # this corrupted `... DEFAULT =\n<inserted>\n1024 * 1024;`
                # into unparseable C++. The real source is one line
                # (`... DEFAULT = 1024 * 1024; // 1 MB`); anchor through the
                # trailing `;` so the insertion lands after the complete
                # statement. The `// 1 MB` comment starts after the `;` so
                # no comment/string-literal noise-stripping concern here.
                id="q8-threshold-constant",
                anchor=r"^static constexpr size_t GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT = 1024 \* 1024;",
                mode="insert_after",
                text="\n// BigCherry NRO01: 0 keeps the draft Q8 path unreachable until qualified.\nstatic constexpr size_t BIGCHERRY_NRO01_Q8_THRESHOLD_DEFAULT = 0;",
                guard=r"^static constexpr size_t BIGCHERRY_NRO01_Q8_THRESHOLD_DEFAULT = 0;$",
                rationale="anchor through the complete single-line statement (not just up to '='), so insert_after lands after the full declaration instead of splicing mid-expression",
            ),
            Edit(
                id="q8-kernel-primitives",
                anchor=r"^struct ggml_cuda_ar_pipeline \{$",
                mode="insert_before",
                text=_Q8_KERNELS + "\n",
                guard=r"BIGCHERRY_NRO01_Q8_SCAFFOLD_BEGIN",
                rationale="insert Q8 primitives immediately before the provider state structure using a code anchor",
            ),
            Edit(
                id="q8-pipeline-field",
                anchor=r"^    size_t   bf16_threshold;",
                mode="insert_after",
                text="\n    size_t   nro01_q8_threshold; // draft: 0 disables Q8 wire dispatch",
                guard=r"nro01_q8_threshold",
                rationale="keep Q8 threshold in the provider instance alongside BF16 threshold",
            ),
            Edit(
                # Same real bug class as q8-threshold-constant above: the
                # anchor ended at `=`, mid-statement, corrupting
                # `p->bf16_threshold   =\n<inserted>\nggml_cuda_ar_env_u64(...)`.
                # The real source is one line:
                # `p->bf16_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 1);`
                # -- it contains a string literal, which the patcher blanks
                # before matching, so the LITERAL-placeholder technique
                # (patches/1222, patches/1225) is used: write the anchor
                # template with a placeholder token in place of the string,
                # then replace the escaped placeholder with a loose
                # same-line match.
                id="q8-init-policy",
                anchor=(
                    _re.escape('    p->bf16_threshold   = ggml_cuda_ar_env_u64(LITERAL1, 1);')
                    .replace(_re.escape('LITERAL1'), r'[^\n]*')
                ),
                mode="insert_after",
                text="\n    p->nro01_q8_threshold = ggml_cuda_ar_env_u64(\"GGML_CUDA_AR_Q8_THRESHOLD\", BIGCHERRY_NRO01_Q8_THRESHOLD_DEFAULT);",
                guard=r"p->nro01_q8_threshold = ggml_cuda_ar_env_u64",
                rationale="anchor through the complete single-line assignment (not just up to '='), using the LITERAL-placeholder technique to cross the noise-stripped string literal, so insert_after lands after the full statement instead of splicing mid-call",
            ),
        ),
    ),
]
