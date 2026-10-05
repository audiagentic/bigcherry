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

// bigcherry 1332: number of QSA chunks K for this context - a constant (from n_ubatch), so every chunked graph has the
// same topology as the reserved one; 0 = chunking off
static int64_t bc_qsa_chunks(uint32_t n_ubatch) {
    const int64_t c = bc_qsa_chunk();
    return c > 0 ? ((int64_t) n_ubatch + c - 1) / c : 0;
}

// bigcherry 1332: batches of at most 8 tokens (decode, MTP verify and draft) keep the dense mask: splitting a
// handful of tokens into K attention calls cost ~8% per decode step at 24K depth and saves no scratch
static bool bc_qsa_chunked(uint32_t n_ubatch, int64_t n_tokens) {
    const int64_t k = bc_qsa_chunks(n_ubatch);
    return k >= 2 && n_tokens >= k && n_tokens > 8;
}
"""

_A_CAUSAL = "    ggml_tensor * idx_f = ggml_cast(ctx0, sel_idx, GGML_TYPE_F32);\n"
_N_CAUSAL = (
    "    if (bc_qsa_chunked(cparams.n_ubatch, n_tokens)) {\n"
    "        // bigcherry 1332: apply the causal mask while the selection is compact - gather kq_mask at each token's\n"
    "        // selected cells (0 visible, -inf invisible), mark invisible selections dead so they go to their dump rows;\n"
    "        // the scattered mask is then final (no dense [n_kv, T] ADD and no per-chunk views of the kq_mask input)\n"
    "        ggml_tensor * bc_safe = ggml_cast(ctx0, ggml_clamp(ctx0, ggml_cast(ctx0, sel_idx, GGML_TYPE_F32), 0.0f, (float) (n_kv - 1)),\n"
    "                GGML_TYPE_I32);\n"
    "        if (bc_qsa_kq_rows == nullptr) {  // one view of the input per graph, shared by all QSA layers\n"
    "            bc_qsa_kq_rows = ggml_reshape_3d(ctx0, kq_mask, 1, n_kv, n_tokens);\n"
    "        }\n"
    "        GGML_ASSERT(bc_qsa_kq_rows->view_src == kq_mask || bc_qsa_kq_rows == kq_mask);\n"
    "        ggml_tensor * bc_kv = ggml_get_rows(ctx0, bc_qsa_kq_rows, ggml_reshape_2d(ctx0, bc_safe, n_sel, n_tokens));\n"
    "        live = ggml_mul(ctx0, live, ggml_reshape_2d(ctx0, ggml_step(ctx0, ggml_exp(ctx0, bc_kv)), n_sel, n_tokens));\n"
    "    }\n"
    + _A_CAUSAL)

_A_SEL = "    ggml_tensor * sel = ggml_set_rows(ctx0, mask_all, zeros, ggml_reshape_3d(ctx0, sel_idx, n_sel, n_tokens, 1));\n"
_N_SEL = ("    // bigcherry 1332: build_attn_qsa builds the masks per token chunk from the indices. The graph topology depends\n"
          "    // only on context constants (llama.cpp #29958): every batch of more than 8 and at least K = ceil(n_ubatch /\n"
          "    // chunk) tokens uses exactly K chunks, smaller batches (decode, MTP verify) the dense mask - a varying chunk\n"
          "    // count after reserve forces a scheduler reallocation, which crashed the meta backend\n"
          "    if (bc_qsa_chunked(cparams.n_ubatch, n_tokens)) {\n"
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
        // bigcherry 1332: sel holds the causally-filtered selection indices [n_sel, n_tokens]; build each chunk's mask
        // (fill -inf, scatter zeros - already final) and attend per chunk
        const int64_t n_kv  = kq_mask->ne[0];
        const int64_t n_tok = sel->ne[1];
        GGML_ASSERT(sel->ne[0] == n_sel && q_cur->ne[2] == n_tok);
        GGML_ASSERT(kq_mask->ne[1]*kq_mask->ne[2]*kq_mask->ne[3] == n_tok && kq_mask->ne[2] == 1 && kq_mask->ne[3] == 1);
        // exactly K chunks for every batch (fixed topology); sizes differ by at most one token
        const int64_t n_chunks = bc_qsa_chunks(cparams.n_ubatch);
        GGML_ASSERT(n_chunks >= 2 && n_tok >= n_chunks);
        int64_t t0 = 0;
        for (int64_t ci = 0; ci < n_chunks; ++ci) {
            const int64_t nt = n_tok / n_chunks + (ci < n_tok % n_chunks ? 1 : 0);

            // rows padded to 256 so the strided mask view keeps aligned rows for the flash-attention mask loads
            const int64_t rows = GGML_PAD(n_kv + n_sel, 256);
            ggml_tensor * m = ggml_new_tensor_4d(ctx0, kq_mask->type, rows, 1, 1, 1);
            m = ggml_fill(ctx0, m, -INFINITY);
            m = ggml_repeat_4d(ctx0, m, rows, nt, 1, 1);
            m = ggml_reshape_3d(ctx0, m, 1, rows, nt);

            ggml_tensor * z = ggml_new_tensor_4d(ctx0, kq_mask->type, n_sel, 1, 1, 1);
            z = ggml_fill(ctx0, z, 0.0f);
            z = ggml_repeat_4d(ctx0, z, n_sel, nt, 1, 1);
            z = ggml_reshape_3d(ctx0, z, 1, n_sel, nt);

            ggml_tensor * idx = ggml_view_2d(ctx0, sel, n_sel, nt, sel->nb[1], (size_t) t0*sel->nb[1]);
            m = ggml_set_rows(ctx0, m, z, ggml_reshape_3d(ctx0, idx, n_sel, nt, 1));

            // the causal mask was applied to the compact selection in build_qsa_sel: the first n_kv rows are final
            const size_t row = m->nb[2];
            ggml_tensor * mask = ggml_view_4d(ctx0, m, n_kv, nt, 1, 1, row, row*nt, row*nt, 0);
            cb(mask, "kq_mask_qsa", il);

            ggml_tensor * qc = ggml_view_3d(ctx0, q_cur, q_cur->ne[0], q_cur->ne[1], nt, q_cur->nb[1], q_cur->nb[2],
                    (size_t) t0*q_cur->nb[2]);
            ggml_tensor * oc = build_attn_mha(qc, k, v, nullptr, mask, nullptr, nullptr, n_sel, kq_scale, il);
            cur = cur ? ggml_concat(ctx0, cur, oc, 1) : oc;
            t0 += nt;
        }
    } else {
        // the selection mask already carries the causal mask
        ggml_tensor * mask = ggml_reshape_4d(ctx0, sel, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);
        cb(mask, "kq_mask_qsa", il);

        cur = build_attn_mha(q_cur, k, v, nullptr, mask, nullptr, nullptr, n_sel, kq_scale, il);
    }
"""

_A_HDR = ("        // dense self-attention over the cells the QSA mask keeps\n"
          "        ggml_tensor * build_attn_qsa(\n")
_N_HDR = ("        // bigcherry 1332: the kq_mask input as [1, n_kv, n_tokens] rows, one view shared by all QSA layers of this graph\n"
          "        ggml_tensor * bc_qsa_kq_rows = nullptr;\n"
          "\n"
          + _A_HDR)

_A_FA = ("        GGML_ASSERT(mask->type == GGML_TYPE_F16);\n"
         "        GGML_ASSERT(ggml_is_contiguous(mask));\n"
         "        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));\n")
_N_FA = ("        GGML_ASSERT(mask->type == GGML_TYPE_F16);\n"
         "        // bigcherry 1332: rows must be contiguous; backends index mask rows by nb[1..3], so a strided row view (the\n"
         "        // chunked QSA mask) is valid\n"
         "        GGML_ASSERT(mask->nb[0] == ggml_type_size(mask->type));\n"
         "        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));\n")

PATCHES = [
    FilePatch(
        path="src/models/models.h",
        description="1332: qwen4exp graph member for the shared kq_mask row view",
        language="none",
        edits=(
            Edit(id="qsa-chunk-member", anchor=re.escape(_A_HDR), mode="replace", text=_N_HDR,
                 guard=r"bigcherry 1332: the kq_mask input as \[1, n_kv, n_tokens\] rows", rationale="qwen4exp graph struct, before build_attn_qsa.",
                 expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="src/models/qwen4exp.cpp",
        description="1332: QSA masks + flash attention per token chunk (BIGCHERRY_QSA_CHUNK=<tokens>)",
        language="none",
        edits=(
            Edit(id="qsa-chunk-helper", anchor=re.escape(_A_INC), mode="replace", text=_N_INC,
                 guard=r"bigcherry 1332: QSA masks \+ attention per chunk", rationale="Standard include block of qwen4exp.cpp.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="qsa-chunk-causal", anchor=re.escape(_A_CAUSAL), mode="replace", text=_N_CAUSAL,
                 guard=r"bigcherry 1332: apply the causal mask while the selection is compact",
                 rationale="build_qsa_sel, before the dead-slot remap that consumes live.", expect_matches=1, max_span_lines=2),
            Edit(id="qsa-chunk-sel", anchor=re.escape(_A_SEL), mode="replace", text=_N_SEL,
                 guard=r"bigcherry 1332: build_attn_qsa builds the masks per token chunk", rationale="build_qsa_sel scatter.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="qsa-chunk-attn", anchor=re.escape(_A_ATTN), mode="replace", text=_N_ATTN,
                 guard=r"bigcherry 1332: sel holds the causally-filtered selection indices", rationale="build_attn_qsa mask + attention.",
                 expect_matches=1, max_span_lines=11),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml.c",
        description="1332: ggml_flash_attn_ext accepts a mask with contiguous rows and any row stride",
        language="none",
        edits=(
            Edit(id="qsa-chunk-fa-mask-rows", anchor=re.escape(_A_FA), mode="replace", text=_N_FA,
                 guard=r"bigcherry 1332: rows must be contiguous", rationale="ggml_flash_attn_ext mask checks.",
                 expect_matches=1, max_span_lines=4),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_QSA_CHUNK", "<tokens>", "0 (off)",
           "experimental: build QSA masks and run QSA attention per chunk of this many query tokens (lower ubatch peak)"),
)
