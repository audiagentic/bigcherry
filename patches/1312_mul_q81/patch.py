"""1312 (QFP13): fused UNARY*MUL (sigmoid/silu gate) also produces the Q8_1 activation its MMVQ consumer needs.

QFP13 Q8_1 miss trace after 1307-1311 (Flash-Next decode): two remaining quantize -> MMVQ chains per layer are fed by a
gate multiply - the full-attention output gate (attn_gated = attn * sigmoid(gate) -> wo) and the GatedDeltaNet gated
output norm (rms_norm(x)*w * sigmoid(z) -> reshape final_output -> ssm_out). Upstream fuses each UNARY+MUL pair into
ggml_cuda_op_unary_mul (one unary_gated launch writing the MUL node), so neither ggml_cuda_op_unary_gated (1310) nor
ggml_cuda_op_mul ever runs for them. This patch routes the F32 branch of ggml_cuda_op_unary_mul_impl through 1310's
bc_act_q81_try<op, true> with the MUL node as the output: same op(x) * g arithmetic (float multiply commutes, so
MUL(g, op(x)) is bit-identical), plus MMVQ-padded native Q8_1 published under the key its MMVQ consumer looks up
(1307's lookup resolves the final_output reshape to the MUL node). 1310's eligibility rules apply (BIGCHERRY_ACT_Q81=1,
cache on, decode-shaped graph, row length a multiple of QK8_1, contiguous output); otherwise the unchanged launch runs.
Requires 1310.
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

_IMPL_OLD = """    } else {
        unary_gated_cuda<op>((const float *) unary_src->data, (const float *) other_src->data,
                             (float *) mul_node->data, k, nc,
                             unary_stride / sizeof(float), other_stride / sizeof(float), stream);
    }
}
"""

_IMPL_NEW = """    } else {
#if defined(GGML_USE_HIP)
        // GDN gated norm [head_v_dim, heads, T] is read by ssm_out through reshape_3d(head_v_dim*heads, T): publish
        // that flattened layout when the per-head row would need MMVQ padding (1307 flattened-reshape lookup).
        const bool flatten01 = mul_node->ne[0] % MATRIX_ROW_PADDING != 0 && mul_node->ne[1] > 1;
        if (bc_act_q81_try<op, true>(ctx, mul_node, (const float *) unary_src->data, (const float *) other_src->data,
                                     unary_stride / sizeof(float), other_stride / sizeof(float), flatten01)) {  // bigcherry 1312
            return;
        }
#endif
        unary_gated_cuda<op>((const float *) unary_src->data, (const float *) other_src->data,
                             (float *) mul_node->data, k, nc,
                             unary_stride / sizeof(float), other_stride / sizeof(float), stream);
    }
}
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/unary.cu",
        description="1312: fused UNARY*MUL gate emits the Q8_1 activation for its MMVQ consumer (via 1310)",
        language="none",
        edits=(
            Edit(
                id="unary-mul-q81",
                anchor=re.escape(_IMPL_OLD),
                mode="replace",
                text=_IMPL_NEW,
                guard=r"bc_act_q81_try<op, true>\(ctx, mul_node,",
                rationale="F32 launch at the end of ggml_cuda_op_unary_mul_impl; 1310's helper is defined earlier in unary.cu.",
                expect_matches=1,
                max_span_lines=7,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc('BIGCHERRY_ACT_Q81', '0|1', '0',
           'also covers the elementwise MUL -> Q8_1 path (shared with 1310)'),
)
