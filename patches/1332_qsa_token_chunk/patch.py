"""1332 (QFP17 chunked scratch): build the QSA attention masks and run QSA flash attention per token chunk.

1331 (live set at the allocator peak, v6, 240K f16) put the main compute-buffer peak at the first QSA layer's
indexer_sel: at ub1024, 1748 MiB live = the kq_mask input [n_kv, T] 480 MiB + mask_all [n_kv+n_sel, T] 484 MiB +
indexer_sel = mask_all view + kq_mask [n_kv, T] 480 MiB (+ kpool mask 120). The two QSA masks scale with the ubatch.

With BIGCHERRY_QSA_CHUNK=C (tokens), build_qsa_sel stops after the dead-slot remap and returns the selection indices
(I32 [n_sel, T], ~8 MiB) instead of the dense mask, and build_attn_qsa builds, per chunk of C query tokens, the same
mask_all (-inf, zeros scattered at the selected cells, dead slots to their dump rows) and the same out-of-place add of
the causal kq_mask rows, then runs flash attention for that chunk's queries; the chunk outputs are concatenated
along the token dimension. Mask values are the same as the unchunked path (same fill/set_rows/add ops on a row
subset). Chunks are built in order, so each chunk's masks are freed before the next chunk's are allocated: the peak
holds one chunk's pair (C/T of 964 MiB) next to the kq_mask input. Flash attention may select a different kernel
tiling for fewer query rows, so results can differ in the last bits; greedy identity is checked on hardware.
Conflicts with 1330 (which rewrites the same mask code).
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_A_INC = "#include <algorithm>\n"
_N_INC = r"""#include <algorithm>
#include <cstdlib>  // bigcherry 1332: getenv/atoi

// bigcherry 1332: QSA masks + attention per chunk of this many query tokens (BIGCHERRY_QSA_CHUNK, 0 = off)
static int64_t bc_qsa_chunk() {
    static const int64_t c = getenv("BIGCHERRY_QSA_CHUNK") ? (int64_t) atoi(getenv("BIGCHERRY_QSA_CHUNK")) : 0;
    return c > 0 ? c : 0;
}
"""

_A_SEL = "    ggml_tensor * sel = ggml_set_rows(ctx0, mask_all, zeros, ggml_reshape_3d(ctx0, sel_idx, n_sel, n_tokens, 1));\n"
_N_SEL = ("    if (bc_qsa_chunk() > 0) {  // bigcherry 1332: build_attn_qsa builds the masks per token chunk from the indices\n"
          "        return sel_idx;        // I32 [n_sel, n_tokens], dead slots already remapped to their dump rows\n"
          "    }\n"
          + _A_SEL)

_A_ATTN = """    // the selection mask already carries the causal mask
    ggml_tensor * kq_mask = inp->get_kq_mask();
    ggml_tensor * mask    = ggml_reshape_4d(ctx0, sel, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);
    cb(mask, "kq_mask_qsa", il);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, mask, nullptr, nullptr, n_sel, kq_scale, il);
"""
_N_ATTN = """    ggml_tensor * kq_mask = inp->get_kq_mask();
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = nullptr;
    if (sel->type == GGML_TYPE_I32) {
        // bigcherry 1332: sel holds the selection indices [n_sel, n_tokens]; build each chunk's mask exactly as
        // build_qsa_sel builds the whole one (fill -inf, scatter zeros, add the causal rows) and attend per chunk
        const int64_t n_kv  = kq_mask->ne[0];
        const int64_t n_tok = sel->ne[1];
        GGML_ASSERT(sel->ne[0] == n_sel && q_cur->ne[2] == n_tok);
        GGML_ASSERT(kq_mask->ne[1]*kq_mask->ne[2]*kq_mask->ne[3] == n_tok && kq_mask->ne[2] == 1 && kq_mask->ne[3] == 1);
        const int64_t chunk = bc_qsa_chunk();
        for (int64_t t0 = 0; t0 < n_tok; t0 += chunk) {
            const int64_t nt = std::min<int64_t>(chunk, n_tok - t0);

            ggml_tensor * m = ggml_new_tensor_4d(ctx0, kq_mask->type, n_kv + n_sel, 1, 1, 1);
            m = ggml_fill(ctx0, m, -INFINITY);
            m = ggml_repeat_4d(ctx0, m, n_kv + n_sel, nt, 1, 1);
            m = ggml_reshape_3d(ctx0, m, 1, n_kv + n_sel, nt);

            ggml_tensor * z = ggml_new_tensor_4d(ctx0, kq_mask->type, n_sel, 1, 1, 1);
            z = ggml_fill(ctx0, z, 0.0f);
            z = ggml_repeat_4d(ctx0, z, n_sel, nt, 1, 1);
            z = ggml_reshape_3d(ctx0, z, 1, n_sel, nt);

            ggml_tensor * idx = ggml_view_2d(ctx0, sel, n_sel, nt, sel->nb[1], (size_t) t0*sel->nb[1]);
            m = ggml_set_rows(ctx0, m, z, ggml_reshape_3d(ctx0, idx, n_sel, nt, 1));

            const size_t row = m->nb[2];
            ggml_tensor * mv  = ggml_view_4d(ctx0, m, n_kv, nt, 1, 1, row, row*nt, row*nt, 0);
            ggml_tensor * kqm = ggml_view_4d(ctx0, kq_mask, n_kv, nt, 1, 1, kq_mask->nb[1], kq_mask->nb[1]*nt,
                    kq_mask->nb[1]*nt, (size_t) t0*kq_mask->nb[1]);
            ggml_tensor * mask = ggml_add(ctx0, mv, kqm);
            cb(mask, "kq_mask_qsa", il);

            ggml_tensor * qc = ggml_view_3d(ctx0, q_cur, q_cur->ne[0], q_cur->ne[1], nt, q_cur->nb[1], q_cur->nb[2],
                    (size_t) t0*q_cur->nb[2]);
            ggml_tensor * oc = build_attn_mha(qc, k, v, nullptr, mask, nullptr, nullptr, n_sel, kq_scale, il);
            cur = cur ? ggml_concat(ctx0, cur, oc, 1) : oc;
        }
    } else {
        // the selection mask already carries the causal mask
        ggml_tensor * mask = ggml_reshape_4d(ctx0, sel, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);
        cb(mask, "kq_mask_qsa", il);

        cur = build_attn_mha(q_cur, k, v, nullptr, mask, nullptr, nullptr, n_sel, kq_scale, il);
    }
"""

PATCHES = [
    FilePatch(
        path="src/models/qwen4exp.cpp",
        description="1332: QSA masks + flash attention per token chunk (BIGCHERRY_QSA_CHUNK=<tokens>)",
        language="none",
        edits=(
            Edit(id="qsa-chunk-helper", anchor=re.escape(_A_INC), mode="replace", text=_N_INC,
                 guard=r"bigcherry 1332: QSA masks \+ attention per chunk", rationale="Standard include block of qwen4exp.cpp.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="qsa-chunk-sel", anchor=re.escape(_A_SEL), mode="replace", text=_N_SEL,
                 guard=r"bigcherry 1332: build_attn_qsa builds the masks per token chunk", rationale="build_qsa_sel scatter.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="qsa-chunk-attn", anchor=re.escape(_A_ATTN), mode="replace", text=_N_ATTN,
                 guard=r"bigcherry 1332: sel holds the selection indices", rationale="build_attn_qsa mask + attention.",
                 expect_matches=1, max_span_lines=11),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_QSA_CHUNK", "<tokens>", "0 (off)",
           "experimental: build QSA masks and run QSA attention per chunk of this many query tokens (lower ubatch peak)"),
)
