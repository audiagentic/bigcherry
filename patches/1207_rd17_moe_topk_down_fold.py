"""RD17: fold the MoE topk-weights MUL into the down projection.

Provenance (group 'rdna-boosts' patches are external backports; the
machine-readable PROVENANCE dict below is cross-checked against
external-sources.toml by tools/tests/test_external_sources.py):

  source:            stew675-rdna-boosts
  repo:              https://github.com/stew675/llama.cpp
  locator:           rdna-boosts (branch name is a locator only, NOT identity)
  fork commit:       5e545b7da39f5abfb2b607922013009c38fd5cd0
                     (snapshot v2; v1 ledger item 19, 817cb6ba1, is the
                     pre-rebase identity of the SAME logical change,
                     content-identical per git patch-id)
                     "CUDA: fold the MoE topk-weights MUL into the down
                     projection"
  reviewed snapshot: v2 -- head 9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22
                     on base 4df29be4f4c3673f428170fda944a5b19f743bb8
  plan item:         RD17 (docs/planning/active/rdna-boost-experiments/RD17.md)
  mainline status:   NOT merged into ggml-org/llama.cpp master as of tip
                     6d0549831 (git cherry patch-id check, 2026-08-18)

What it does (performance, kernel-level fusion):
  The MoE output is scaled per token by the normalized topk softmax
  weights in a separate broadcast MUL kernel after the down projection.
  This folds the scale into the matmul epilogue: the mmvq kernel
  multiplies each result row by the weight of the destination channel
  (token). The new x_scale_channel_dst fusion flag indexes x_scale by
  channel_dst instead of the NVFP4 per-expert channel_x, and relaxes the
  NVFP4-only scale-fusion assertions for this second, per-token case.
  Fork-reported: decode on qwen35moe drops 40 kernels per token, PPL
  bit-identical.

Porting notes:
  - Ported from the fork commit across three files (7 hunks): common.cuh
    (x_scale_channel_dst in both fusion-args structs), ggml-cuda.cu (the
    MUL_MAT_ID + MUL detection in ggml_cuda_try_fuse), mmvq.cu (kernel
    scale-index plumbing, has_fusion bool, assertion relaxations).
  - ANCHOR OVERLAP WITH RD12 (expected, policy-driven): the RD17
    common.cuh hunks anchor on the SAME struct insertion slot that RD12
    uses (between gate_scale and glu_op), and the RD17 ggml-cuda.cu
    block anchors immediately before "fused_mul_mat_vec = false;" --
    the same insertion point as RD12's detection block (in the fork,
    RD17 sits above RD16's L2-norm block, which our base does not have,
    so the two insertion points coincide). If both patches are selected
    in one apply, the second one fails its anchor LOUDLY -- intended:
    experimental-vs-experimental composition is deferred until a patch
    proves its benefit (2026-08-19 policy). If that day comes, the fork
    ordering is RD17 above RD12.
  - Base co-tenancy: 0600/0650 edit mmvq.cu launch/geometry, not the
    scale-plumbing regions anchored here; all anchors verified
    byte-identical on the framework-patched base 2026-08-19.

Isolation and promotion (first-sweep policy, RD review 2026-08-18):
  - GROUP 'rdna-boosts' + STATE 'untested' keeps this OUT of the
    production 'framework' and 'validated-enhancements' patch-sets.
  - Bench: a MoE model is required (qwen35moe-class); output/PPL
    equality vs unfused is the primary gate, plus per-token kernel
    count and decode timing.

Maintenance (future pin bumps / fork movement):
  - mul_mat_vec_q's scale handling and the fusion-args structs are
    upstream-touched; re-derive from the tracked fork commit in
    external-sources.toml and run `python -m bigcherry sources check`
    before every pin bump.
"""

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

PROVENANCE = {
    "source-id": "stew675-rdna-boosts",
    "plan-item": "RD17",
    "fork-commit": "5e545b7da39f5abfb2b607922013009c38fd5cd0",
    "fork-commit-title": "CUDA: fold the MoE topk-weights MUL into the down projection",
    "original-commit": "817cb6ba178933f5b62399097f5848c79f4f34e8",
    "snapshot-head": "9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22",
    "snapshot-base": "4df29be4f4c3673f428170fda944a5b19f743bb8",
    "adaptations": [
        "ggml-cuda.cu detection block anchored before 'fused_mul_mat_vec "
        "= false;': in the fork it sits above RD16's L2-norm block, which "
        "our base does not have. Overlaps RD12's insertion point; "
        "double-selection fails loudly by policy (fork order: RD17 above "
        "RD12).",
    ],
}


# ----------------------------------------------------------------- common.cuh

_HOST_OLD = """    const ggml_tensor * gate_scale = nullptr;
    ggml_glu_op glu_op;"""

_HOST_NEW = """    const ggml_tensor * gate_scale = nullptr;
    // Index x_scale by the destination channel (token), not the source channel
    // (expert). Used for the MoE down x topk-weights fusion.
    bool x_scale_channel_dst = false;
    ggml_glu_op glu_op;"""

_DEV_OLD = """    const void * gate_scale = nullptr;
    ggml_glu_op glu_op;"""

_DEV_NEW = """    const void * gate_scale = nullptr;
    bool x_scale_channel_dst = false;
    ggml_glu_op glu_op;"""

# ------------------------------------------------------------- ggml-cuda.cu

_DETECT_ANCHOR = """    fused_mul_mat_vec = false;
    fused_node_count  = 0;"""

_DETECT_BLOCK = """    // MoE: ffn_moe_weighted = moe_down * topk_weights. The down projection
    // output is scaled per token (the topk softmax weights); fold the MUL into
    // the matmul epilogue. The pattern is [MUL_MAT_ID, MUL] with the MUL's
    // src1 being a contiguous per-channel F32 vector (a view of the
    // normalized weights).
    if (i + 1 < cgraph->n_nodes && cgraph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
        ggml_tensor * mm_node  = cgraph->nodes[i];
        ggml_tensor * mul_node = cgraph->nodes[i + 1];

        const int out_nodes[] = { i + 1 };
        if (mul_node->op == GGML_OP_MUL &&
                mul_node->src[0] == mm_node &&
                (mm_node->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                (mul_node->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                ggml_cuda_check_fusion_memory_ranges(cgraph, i, 2, out_nodes, 1)) {
            const ggml_tensor * weights = mul_node->src[1];
            if (weights->type == GGML_TYPE_F32 && ggml_is_contiguous(weights) &&
                    weights->ne[0] == 1 && weights->ne[1] == mm_node->ne[1] &&
                    ggml_are_same_shape(mm_node, mul_node) &&
                    ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.x_scale             = weights;
                fusion_data.x_scale_channel_dst = true;

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, mm_node->src[0], mm_node->src[1], mm_node->src[2], mul_node, &fusion_data);
                return 1;
            }
        }
    }

"""

_DETECT_NEW = _DETECT_BLOCK + _DETECT_ANCHOR

# -------------------------------------------------------------------- mmvq.cu

_SCALE_SETUP_OLD = """        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }"""

_SCALE_SETUP_NEW = """        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
        // Per-token scale (MoE down x topk weights). Indexed by channel_dst.
        if (fusion.x_scale_channel_dst) {
            use_scale = true;
            x_scale   = (const float *) fusion.x_scale;
        }"""

_PROLOGUE_OLD = """            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }"""

_PROLOGUE_NEW = """            if (use_scale) {
                x_scales = fusion.x_scale_channel_dst ? x_scale[channel_dst] : x_scale[ids ? channel_x : 0];
            }
            if (use_gate_scale) {
                gate_scales = gate_scale[ids ? channel_x : 0];
            }"""

_APPLY_OLD = """                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }"""

_APPLY_NEW = """                if constexpr (has_fusion) {
                    if (use_scale) {
                        result *= x_scales;
                    }"""

_HAS_FUSION_OLD = """    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;"""

_HAS_FUSION_NEW = """    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr || fusion.x_scale_channel_dst;"""

# The two comment lines are blanks in the noise-stripped anchor view, so the
# anchor covers them with space-run classes (same convention as 1206).
_ASSERT_OLD = ("        [ ]{80,110}\n"
               "        [ ]{40,90}\n"
               "        GGML_ASSERT\\(\\(fusion->x_scale == nullptr && fusion->gate_scale == nullptr\\) "
               "\\|\\| src0->type == GGML_TYPE_NVFP4\\);")

_ASSERT_NEW = """        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b. The per-token MoE scale is a second exception.
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4 || fusion->x_scale_channel_dst);"""

_NELEMS_OLD = """            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }"""

_NELEMS_NEW = """            if (fusion->x_scale_channel_dst) {
                GGML_ASSERT(ids);
                GGML_ASSERT(ggml_nelements(fusion->x_scale) == dst->ne[1]);
            } else {
                GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            }
            fusion_local.x_scale = fusion->x_scale->data;
        }
        fusion_local.x_scale_channel_dst = fusion->x_scale_channel_dst;"""


PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/common.cuh",
        description="Add x_scale_channel_dst to the mm fusion-args structs "
                    "(rdna-boosts 5e545b7da / RD17)",
        edits=(
            Edit(
                id="rd17-struct-host",
                anchor=re.escape(_HOST_OLD),
                rationale="ggml_cuda_mm_fusion_args_host: add "
                          "x_scale_channel_dst between gate_scale and glu_op "
                          "(fork position; same slot RD12 uses for dst_gate)",
                mode="replace",
                text=_HOST_NEW,
                guard=r"bool x_scale_channel_dst = false;\n    ggml_glu_op glu_op;",
            ),
            Edit(
                id="rd17-struct-device",
                anchor=re.escape(_DEV_OLD),
                rationale="ggml_cuda_mm_fusion_args_device: add "
                          "x_scale_channel_dst between gate_scale and glu_op",
                mode="replace",
                text=_DEV_NEW,
                # guard includes the device-struct gate_scale line so it can
                # never be satisfied by the host-struct edit above
                guard=r"const void \* gate_scale = nullptr;\n    bool x_scale_channel_dst = false;",
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="Detect [MUL_MAT_ID, MUL] MoE down+topk pattern and fold "
                    "the per-token scale into the mmvq epilogue "
                    "(rdna-boosts 5e545b7da / RD17)",
        edits=(
            Edit(
                id="rd17-moe-topk-detect",
                anchor=re.escape(_DETECT_ANCHOR),
                rationale="ggml_cuda_try_fuse: insertion before the "
                          "GLU/mul_mat_vec section (fork position, which in "
                          "the fork sits above RD16's L2-norm block)",
                mode="replace",
                text=_DETECT_NEW,
                guard=r"// MoE: ffn_moe_weighted = moe_down \* topk_weights\.",
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/mmvq.cu",
        description="Per-token x_scale (channel_dst) plumbing in "
                    "mul_mat_vec_q and its host wrapper "
                    "(rdna-boosts 5e545b7da / RD17)",
        edits=(
            Edit(
                id="rd17-kernel-scale-setup",
                anchor=re.escape(_SCALE_SETUP_OLD),
                rationale="mul_mat_vec_q: after the NVFP4 scale setup, "
                          "enable use_scale from x_scale_channel_dst",
                mode="replace",
                text=_SCALE_SETUP_NEW,
                guard=r"Per-token scale \(MoE down x topk weights\)\. Indexed by channel_dst\.",
            ),
            Edit(
                id="rd17-kernel-prologue-index",
                anchor=re.escape(_PROLOGUE_OLD),
                rationale="mul_mat_vec_q prologue: select the scale by "
                          "channel_dst for the per-token case (fork logic)",
                mode="replace",
                text=_PROLOGUE_NEW,
                guard=r"x_scales = fusion\.x_scale_channel_dst \? x_scale\[channel_dst\]",
            ),
            Edit(
                id="rd17-kernel-apply-scale",
                anchor=re.escape(_APPLY_OLD),
                rationale="mul_mat_vec_q epilogue: apply x_scales whenever "
                          "use_scale is set, not only for NVFP4 (fork logic)",
                mode="replace",
                text=_APPLY_NEW,
                guard=r"if \(use_scale\) \{\n                        result \*= x_scales;",
            ),
            Edit(
                id="rd17-switch-has-fusion",
                anchor=re.escape(_HAS_FUSION_OLD),
                rationale="mul_mat_vec_q_switch_fusion: x_scale_channel_dst "
                          "alone must select the fused instantiation",
                mode="replace",
                text=_HAS_FUSION_NEW,
                guard=r"fusion\.gate_scale != nullptr \|\| fusion\.x_scale_channel_dst;",
            ),
            Edit(
                id="rd17-assert-relax",
                anchor=_ASSERT_OLD,
                rationale="ggml_cuda_mul_mat_vec_q: allow the per-token MoE "
                          "scale as a second exception to the NVFP4-only "
                          "rule (fork logic)",
                mode="replace",
                text=_ASSERT_NEW,
                guard=r"src0->type == GGML_TYPE_NVFP4 \|\| fusion->x_scale_channel_dst\);",
            ),
            Edit(
                id="rd17-nelements-assert",
                anchor=re.escape(_NELEMS_OLD),
                rationale="ggml_cuda_mul_mat_vec_q: per-token x_scale is "
                          "checked against dst->ne[1] and requires ids; "
                          "copy the flag to the device args (fork logic)",
                mode="replace",
                text=_NELEMS_NEW,
                guard=r"fusion_local\.x_scale_channel_dst = fusion->x_scale_channel_dst;",
            ),
        ),
    ),
]
