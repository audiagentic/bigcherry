"""1295: QSA decode attends over the gathered n_sel cells instead of the masked full KV cache.

build_attn_qsa expresses the QSA selection as a -inf mask over all n_kv cells and hands the full K/V cache
to flash attention. Upstream's sparse flash-attention path (mask compaction + index gather, n_kv_max) is
NVIDIA-only; on HIP the tile/vec kernels read every cell, so QSA attention costs O(n_kv): rocprof decode
windows show target verify attention 0.44 -> 3.36 ms/token and draft attention 0.18 -> 1.41 ms/token from
10K to 80K context (7.6x for 8x), see RNX02 review RV4213.

For small batches (n_tokens <= 8: decode, MTP verify, MTP draft) this patch gathers each token's selected
cells (top pools + tail, sentinel n_kv for missing cells) from the K/V cache with get_rows, gathers the
matching kq_mask entries (the sentinel maps to an appended -inf column), and runs flash attention per token
over n_sel cells with the token as the batch dimension. Same attention, n_sel instead of n_kv cells read.
Prefill keeps the masked path. Falls back to the masked path when the cache view is not plain (multiple
streams, transposed V) or has no spare row for the sentinel. BIGCHERRY_QSA_GATHER=0 disables it.
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "core"
STATE = "untested"

_GATHER = """    // BigCherry 1295: small batches attend over the gathered n_sel cells, not the masked n_kv cells
    if (bc_qsa_idx != nullptr && n_tokens <= 8 && bc_qsa_gather_enabled()) {
        ggml_tensor * k_all = mctx_cur->get_k(ctx0, il); // [hd, n_head_kv, n_kv, n_stream]
        ggml_tensor * v_all = mctx_cur->get_v(ctx0, il);
        const int64_t hd    = k_all->ne[0];
        const int64_t nh_kv = k_all->ne[1];
        const int64_t n_kv  = k_all->ne[2];
        // pad the cell list to a multiple of 256 with the sentinel so the vec kernel and the GQA path apply
        const int64_t n_pad     = (256 - bc_qsa_idx->ne[0] % 256) % 256;
        const int64_t n_sel_all = bc_qsa_idx->ne[0] + n_pad;
        const bool plain = k_all->ne[3] == 1 && v_all->ne[3] == 1 && v_all->nb[1] <= v_all->nb[2] &&
            k_all->view_src != nullptr && v_all->view_src != nullptr &&
            k_all->view_offs + (size_t) (n_kv + 1) * k_all->nb[2] <= ggml_nbytes(k_all->view_src) &&
            v_all->view_offs + (size_t) (n_kv + 1) * v_all->nb[2] <= ggml_nbytes(v_all->view_src) &&
            v_all->ne[0] == hd && v_all->ne[1] == nh_kv;
        if (plain) {
            // rows n_kv cells + 1: the sentinel index n_kv reads a spare row, masked to -inf below
            ggml_tensor * k_rows = ggml_view_2d(ctx0, k_all->view_src, hd*nh_kv, n_kv + 1, k_all->nb[2], k_all->view_offs);
            ggml_tensor * v_rows = ggml_view_2d(ctx0, v_all->view_src, hd*nh_kv, n_kv + 1, v_all->nb[2], v_all->view_offs);
            ggml_tensor * cells = bc_qsa_idx;
            if (n_pad > 0) {
                ggml_tensor * pad = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pad, n_tokens), (float) n_kv);
                cells = ggml_concat(ctx0, cells, ggml_cast(ctx0, pad, GGML_TYPE_I32), 0);
            }
            ggml_tensor * idx    = ggml_reshape_1d(ctx0, cells, n_sel_all*n_tokens);

            ggml_tensor * k_sel = ggml_cast(ctx0, ggml_get_rows(ctx0, k_rows, idx), GGML_TYPE_F16);
            ggml_tensor * v_sel = ggml_cast(ctx0, ggml_get_rows(ctx0, v_rows, idx), GGML_TYPE_F16);
            k_sel = ggml_permute(ctx0, ggml_reshape_4d(ctx0, k_sel, hd, nh_kv, n_sel_all, n_tokens), 0, 2, 1, 3);
            v_sel = ggml_permute(ctx0, ggml_reshape_4d(ctx0, v_sel, hd, nh_kv, n_sel_all, n_tokens), 0, 2, 1, 3);

            // each token's mask entries at its own cells; the appended column makes the sentinel -inf
            ggml_tensor * kq_mask_all = inp->get_kq_mask();
            ggml_tensor * m    = ggml_cast(ctx0, ggml_reshape_2d(ctx0, kq_mask_all, n_kv, n_tokens), GGML_TYPE_F32);
            ggml_tensor * ninf = ggml_fill(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, m, 1, n_tokens, m->nb[1], 0)), -INFINITY);
            m = ggml_reshape_3d(ctx0, ggml_concat(ctx0, m, ninf, 0), 1, n_kv + 1, n_tokens);
            ggml_tensor * m_sel = ggml_get_rows(ctx0, m, cells); // [1, n_sel_all, n_tokens]
            m_sel = ggml_cast(ctx0, ggml_reshape_4d(ctx0, m_sel, n_sel_all, 1, 1, n_tokens), GGML_TYPE_F16);

            ggml_tensor * q = ggml_permute(ctx0, ggml_reshape_4d(ctx0, q_cur, q_cur->ne[0], q_cur->ne[1], 1, n_tokens), 0, 2, 1, 3);
            ggml_tensor * cur = ggml_flash_attn_ext(ctx0, q, k_sel, v_sel, m_sel, kq_scale, 0.0f, 0.0f);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], n_tokens);
            cb(cur, "kqv_out_qsa_gather", il);

            if (inp->self_v_rot) {
                cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
            }
            return cur;
        }
    }

"""

_ENABLED = """// BigCherry 1295: QSA gathered-cell attention for small batches, on unless BIGCHERRY_QSA_GATHER=0
static bool bc_qsa_gather_enabled() {
    static const bool on = [] {
        const char * e = getenv("BIGCHERRY_QSA_GATHER");
        const bool enabled = e == nullptr || strcmp(e, "0") != 0;
        if (enabled && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
            LLAMA_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1295_qsa_gather_decode\\n");
        }
        return enabled;
    }();
    return on;
}

"""

MODEL = FilePatch(
    path="src/models/qwen4exp.cpp",
    language="none",
    description="QSA: gather the selected cells for small batches instead of masking the full cache.",
    edits=(
        Edit(
            id="qsa-gather-keep-idx",
            anchor=_re.escape("    sel_idx = ggml_concat(ctx0, sel_idx, inp_kpool->tail_idxs, 0);\n"),
            text=(
                "    sel_idx = ggml_concat(ctx0, sel_idx, inp_kpool->tail_idxs, 0);\n"
                "    bc_qsa_idx = sel_idx; // BigCherry 1295: build_attn_qsa gathers these cells for small batches\n"
            ),
            mode="replace",
            guard=r"bc_qsa_idx = sel_idx;",
            expect_matches=1,
            rationale="The one place build_qsa_sel finalizes the per-token cell list (top pools + tail).",
        ),
        Edit(
            id="qsa-gather-path",
            anchor=_re.escape("    // the selection mask already carries the causal mask\n"),
            text=_GATHER,
            mode="insert_before",
            guard=r"BigCherry 1295: small batches attend over the gathered n_sel cells",
            expect_matches=1,
            rationale="In build_attn_qsa after the K/V cache writes, before the masked path builds its mask.",
        ),
        Edit(
            id="qsa-gather-enabled",
            anchor=r"(?m)^ggml_tensor \* llama_model_qwen4exp::graph::build_attn_qsa\(",
            text=_ENABLED,
            mode="insert_before",
            guard=r"static bool bc_qsa_gather_enabled\(\)",
            expect_matches=1,
            rationale="File-local switch defined before its only user.",
        ),
    ),
)

MODELS_H = FilePatch(
    path="src/models/models.h",
    language="none",
    description="Qwen4Exp graph keeps the current layer's QSA cell list for build_attn_qsa.",
    edits=(
        Edit(
            id="qsa-gather-member",
            anchor=_re.escape("        // dense self-attention over the cells the QSA mask keeps\n"),
            text=(
                "        // BigCherry 1295: the current layer's QSA cells [n_sel, n_tokens], set by build_qsa_sel\n"
                "        ggml_tensor * bc_qsa_idx = nullptr;\n\n"
                "        // dense self-attention over the cells the QSA mask keeps\n"
            ),
            mode="replace",
            guard=r"ggml_tensor \* bc_qsa_idx = nullptr;",
            expect_matches=1,
            rationale="Comment heads the build_attn_qsa declaration inside the qwen4exp graph struct only.",
        ),
    ),
)

PATCHES = [MODEL, MODELS_H]
