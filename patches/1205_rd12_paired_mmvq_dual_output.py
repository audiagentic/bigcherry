"""RD12: fuse paired mmvq matmuls over a shared activation (dual-output).

Provenance (group 'rdna-boosts' patches are external backports; the
machine-readable PROVENANCE dict below is cross-checked against
external-sources.toml by tools/tests/test_external_sources.py):

  source:            stew675-rdna-boosts
  repo:              https://github.com/stew675/llama.cpp
  locator:           rdna-boosts (branch name is a locator only, NOT identity)
  fork commit:       44b51c66a210053b12bdfcf1183a6b175878b9c8
                     (snapshot v2; v1 ledger item 22, ba9e339ea, is the
                     pre-rebase identity of the SAME logical change,
                     content-identical per git patch-id)
                     "cuda : fuse paired mmvq matmuls over a shared
                     activation"
  reviewed snapshot: v2 -- head 9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22
                     on base 4df29be4f4c3673f428170fda944a5b19f743bb8
  plan item:         RD12 (docs/planning/active/rdna-boost-experiments/RD12.md)
  mainline status:   NOT merged into ggml-org/llama.cpp master as of tip
                     6d0549831 (git cherry patch-id check, 2026-08-18)

What it does (performance, kernel-level fusion):
  The K and V projections of an attention layer run two mmvq matmuls over
  the same activation with the same output shape. The first matmul now
  computes both in one launch: it treats the second weight as the fusion
  "gate" and writes that result to a SEPARATE destination (new dst_gate
  fusion arg) instead of combining it with the main output. Detection:
  two adjacent MUL_MAT nodes (view/noop nodes may sit between) with the
  same src1, same output shape, and mmvq-eligible quantized weights; the
  GLU fusions take precedence (they run first in ggml_cuda_try_fuse).
  Fork-reported validation: bit-identical vs unfused, 1194/1194
  MUL_MAT backend tests, small positive tg64 numbers on gfx1201.

Porting notes:
  - Ported from the fork commit across three files: common.cuh (dst_gate
    field in both fusion-args structs), ggml-cuda.cu (the dual-output
    detection block in ggml_cuda_try_fuse), mmvq.cu (kernel plumbing:
    use_dst_gate flag, dst_gate write branch, host-side dst_gate setup).
  - ADAPTATION (deviation from the fork hunk, required by first-sweep
    policy): the fork common.cuh hunk context includes RD17 x_scale_channel_dst
    field (RD17 landed earlier in the fork history). Our base struct does
    not have it, so the dst_gate fields are inserted against the base
    struct shape. If RD17 is ever selected TOGETHER with RD12, the
    RD12 common.cuh anchor will fail to match -- that conflict is
    intended to surface loudly (experimental-vs-experimental composition
    is deferred until a patch proves its benefit, per the 2026-08-19
    policy decision).
  - Base co-tenancy: 0600/0650/0700 edit mmvq.cu launch/geometry region,
    not the fusion-plumbing regions anchored here; all anchors verified
    byte-identical on the framework-patched base 2026-08-19.
  - The detection block anchors immediately before
    "fused_mul_mat_vec = false;" -- the same neighborhood RD13 edits
    (RD13 replaces the matmul+bias branch that ends just above).
    Again: loud failure on double-selection is the policy.

Isolation and promotion (first-sweep policy, RD review 2026-08-18):
  - GROUP 'rdna-boosts' + STATE 'untested' keeps this OUT of the
    production 'framework' and 'validated-enhancements' patch-sets.
  - Bench: standard attention models (K/V projections are the pattern);
    output equality vs unfused path is the primary gate (the fork
    claims bit-identity), plus decode/prefill timing.

Maintenance (future pin bumps / fork movement):
  - mul_mat_vec_q and the fusion-args structs are upstream-touched;
    re-derive from the tracked fork commit in external-sources.toml and
    run `python -m bigcherry sources check` before every pin bump.
"""

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

# RE40 (external patch-management review, 2026-08-20): 1205/1207 share a
# struct anchor in ggml-cuda.cu's fusion-detection block and cannot compose
# today (documented in both patches' own docstrings and
# docs/reference/PIN_REBASE_REVIEW_B10502.md section 2.2). CONFLICTS makes
# that real and enforced (patchset.resolve_exact() raises if both are
# explicitly selected together) instead of relying solely on the anchor
# collision to surface it at apply time. First-sweep policy: compatible
# composition is a future decision, not implied by this declaration.
CONFLICTS = ("1207_rd17_moe_topk_down_fold",)

PROVENANCE = {
    "source-id": "stew675-rdna-boosts",
    "plan-item": "RD12",
    "fork-commit": "44b51c66a210053b12bdfcf1183a6b175878b9c8",
    "fork-commit-title": "cuda : fuse paired mmvq matmuls over a shared activation",
    "original-commit": "ba9e339eaec9fafb9ae7d151bf95c82d13061a62",
    "snapshot-head": "9e46e1fdc7a880f9ae9a2f9a693ae3e14c142a22",
    "snapshot-base": "4df29be4f4c3673f428170fda944a5b19f743bb8",
    "adaptations": [
        "common.cuh dst_gate fields inserted against the BASE struct shape "
        "(no x_scale_channel_dst); the fork hunk context assumed RD17 had "
        "landed first. Double-selection of RD12+RD17 fails the anchor "
        "loudly by policy (exp-vs-exp composition deferred).",
    ],
}


# ----------------------------------------------------------------- common.cuh

_HOST_STRUCT_OLD = """    const ggml_tensor * gate_scale = nullptr;
    ggml_glu_op glu_op;"""

_HOST_STRUCT_NEW = """    const ggml_tensor * gate_scale = nullptr;
    // when set (with glu_op == GGML_GLU_OP_NONE), the gate result is written
    // to this separate destination instead of being combined into the main output
    const ggml_tensor * dst_gate = nullptr;
    ggml_glu_op glu_op;"""

_DEV_STRUCT_OLD = """    const void * gate_scale = nullptr;
    ggml_glu_op glu_op;"""

_DEV_STRUCT_NEW = """    const void * gate_scale = nullptr;
    const void * dst_gate = nullptr;
    ggml_glu_op glu_op;"""

# ------------------------------------------------------------- ggml-cuda.cu

_DETECT_ANCHOR = """    fused_mul_mat_vec = false;
    fused_node_count  = 0;"""

_DETECT_BLOCK = """    // Dual-output mmvq fusion: two matmuls over the same activation with the
    // same output shape (e.g. the K and V projections of an attention layer).
    // The first matmul computes both results; the gate result is written to the
    // second matmul's destination. Only view/noop nodes may sit between the pair.
    if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT) {
        ggml_tensor * mm_a = cgraph->nodes[i];
        if ((mm_a->flags & GGML_TENSOR_FLAG_COMPUTE) && ggml_cuda_should_fuse_mul_mat_vec_q(mm_a)) {
            for (int j = i + 1; j < std::min(cgraph->n_nodes, i + 8); ++j) {
                ggml_tensor * mid = cgraph->nodes[j];
                if (ggml_cuda_is_view_or_noop(mid)) {
                    continue;
                }
                if (mid->op != GGML_OP_MUL_MAT || !(mid->flags & GGML_TENSOR_FLAG_COMPUTE) ||
                        mid->src[1] != mm_a->src[1] || mid->ne[0] != mm_a->ne[0] ||
                        mid->ne[1] != mm_a->ne[1] || mid->ne[2] != mm_a->ne[2] ||
                        mid->src[0] == mm_a->src[0] || mid->src[0]->type != mm_a->src[0]->type ||
                        !ggml_cuda_should_fuse_mul_mat_vec_q(mid)) {
                    break;
                }
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate     = mid->src[0];
                fusion_data.dst_gate = mid;
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, mm_a->src[0], mm_a->src[1], mm_a->src[2], mm_a, &fusion_data);
                return j - i;
            }
        }
    }

"""

_DETECT_NEW = _DETECT_BLOCK + _DETECT_ANCHOR

# -------------------------------------------------------------------- mmvq.cu

_FLAGS_OLD = """    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;"""

_FLAGS_NEW = """    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    bool use_dst_gate = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    [[maybe_unused]] const void * dst_gate = nullptr;
    ggml_glu_op active_glu;"""

_FUSION_SETUP_OLD = """        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;"""

_FUSION_SETUP_NEW = """        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        use_dst_gate  = fusion.dst_gate != nullptr && use_gate;
        if (use_dst_gate) {
            dst_gate = fusion.dst_gate;
        }"""

_SWITCH_OLD = """                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }"""

_SWITCH_NEW = """                        gate_value += gate_biases[j];
                        if (use_dst_gate) {
                            // separate output: write the gate result to its own destination
                            // (dst was already offset by sample/channel/row; apply the same to dst_gate)
                            float * dst_gate_row = (float *) dst_gate + sample_dst*stride_sample_dst +
                                                   channel_dst*stride_channel_dst + row0 + j*stride_col_dst;
                            dst_gate_row[i] = gate_value;
                        } else {
                            switch (active_glu) {
                                case GGML_GLU_OP_SWIGLU:
                                    result *= ggml_cuda_op_silu_single(gate_value);
                                    break;
                                case GGML_GLU_OP_GEGLU:
                                    result *= ggml_cuda_op_gelu_single(gate_value);
                                    break;
                                case GGML_GLU_OP_SWIGLU_OAI:
                                    result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                    break;
                                default:
                                    result = result * gate_value;
                                    break;
                            }
                        }"""

_UNUSED_OLD = """        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);"""

_UNUSED_NEW = """        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, use_dst_gate, active_glu, gate_bias, x_bias, x_scale, gate_scale, tmp_gate, dst_gate);"""

_HOST_SIDE_OLD = """        fusion_local.glu_op = fusion->glu_op;"""

_HOST_SIDE_NEW = """        fusion_local.glu_op = fusion->glu_op;
        if (fusion->dst_gate) {
            GGML_ASSERT(fusion->dst_gate->type == GGML_TYPE_F32);
            fusion_local.dst_gate = fusion->dst_gate->data;
        }"""


PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/common.cuh",
        description="Add dst_gate to the mm fusion-args host/device structs "
                    "(rdna-boosts 44b51c66a / RD12)",
        edits=(
            Edit(
                id="rd12-struct-host",
                anchor=re.escape(_HOST_STRUCT_OLD),
                rationale="ggml_cuda_mm_fusion_args_host: add dst_gate "
                          "between gate_scale and glu_op (fork position)",
                mode="replace",
                text=_HOST_STRUCT_NEW,
                guard=r"const ggml_tensor \* dst_gate = nullptr;",
            ),
            Edit(
                id="rd12-struct-device",
                anchor=re.escape(_DEV_STRUCT_OLD),
                rationale="ggml_cuda_mm_fusion_args_device: add dst_gate "
                          "between gate_scale and glu_op (fork position)",
                mode="replace",
                text=_DEV_STRUCT_NEW,
                guard=r"const void \* dst_gate = nullptr;",
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="Detect paired mmvq matmuls over a shared activation and "
                    "run them as one dual-output launch "
                    "(rdna-boosts 44b51c66a / RD12)",
        edits=(
            Edit(
                id="rd12-dual-output-detect",
                anchor=re.escape(_DETECT_ANCHOR),
                rationale="ggml_cuda_try_fuse: insertion point between the "
                          "bias-add fusion and the GLU/mul_mat_vec section; "
                          "the fork places the block at exactly this spot",
                mode="replace",
                text=_DETECT_NEW,
                guard=r"// Dual-output mmvq fusion: two matmuls over the same activation with the",
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/mmvq.cu",
        description="Kernel plumbing for the dual-output dst_gate write in "
                    "mul_mat_vec_q (rdna-boosts 44b51c66a / RD12)",
        edits=(
            Edit(
                id="rd12-kernel-flags",
                anchor=re.escape(_FLAGS_OLD),
                rationale="mul_mat_vec_q: the fusion flag/pointer block; add "
                          "use_dst_gate and the dst_gate pointer",
                mode="replace",
                text=_FLAGS_NEW,
                guard=r"bool use_dst_gate = false;",
            ),
            Edit(
                id="rd12-kernel-fusion-setup",
                anchor=re.escape(_FUSION_SETUP_OLD),
                rationale="mul_mat_vec_q: the has_fusion setup block; enable "
                          "dst_gate when the host args carry one",
                mode="replace",
                text=_FUSION_SETUP_NEW,
                guard=r"use_dst_gate  = fusion\.dst_gate != nullptr && use_gate;",
            ),
            Edit(
                id="rd12-kernel-dst-write",
                anchor=re.escape(_SWITCH_OLD),
                rationale="mul_mat_vec_q: the gate-combine switch; when "
                          "dst_gate is set, write the raw gate value to its "
                          "own destination instead of combining (fork logic)",
                mode="replace",
                text=_SWITCH_NEW,
                guard=r"separate output: write the gate result to its own destination",
            ),
            Edit(
                id="rd12-kernel-unused-vars",
                anchor=re.escape(_UNUSED_OLD),
                rationale="mul_mat_vec_q: keep the non-fusion build compiling "
                          "-- add the new variables to GGML_UNUSED_VARS",
                mode="replace",
                text=_UNUSED_NEW,
                guard=r"use_dst_gate, active_glu",
            ),
            Edit(
                id="rd12-kernel-host-side",
                anchor=re.escape(_HOST_SIDE_OLD),
                rationale="ggml_cuda_mul_mat_vec_q: copy the host dst_gate "
                          "pointer into the device fusion args with the "
                          "F32-type assertion (fork logic)",
                mode="replace",
                text=_HOST_SIDE_NEW,
                guard=r"fusion_local\.dst_gate = fusion->dst_gate->data;",
            ),
        ),
    ),
]
