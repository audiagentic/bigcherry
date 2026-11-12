"""rad0027 (VTP31): the pipelined prefill over a mixed step of decode rows and one prefill sequence.

rad0026 pipelines a mixed step with two or more prefill sequences, cut between them. A mixed step with one prefill
sequence (a long prompt beside streams that are generating: every chunk after a long prompt's first, and the usual
shape of a server whose requests arrive one at a time) stays serial. This patch cuts that step inside its one prefill
sequence, as rad0020 cuts a one-sequence prefill step:

- `pipe_ready` takes a mixed step with one prefill sequence when its decode rows are under 64 and both pieces have 64
  rows or more. The first piece is the decode rows and the prefill sequence's first `pipe_rows(T - DT)` rows, the second
  the rest. The bounds {DT, R, T} come from a pinned table per decode height, made on first use, so a recorded step
  replays the same copy.
- The attention and delta-net one-sequence piece paths take that step. In the first piece each block issues the serial
  path's decode launches over the decode sequences (rad0026), then the prefill piece over the sequence's slice of the
  per-sequence arrays (block table row, seqused, convolution and state indices) and its own row count.

The prefill pieces are rad0020's one-sequence pieces at an offset row, and the decode launches are the serial path's.
Source-pinned to radiance v1.17.0 (19d4e8fe), after rad0026.
"""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch


def _edit(id: str, before: str, after: str, *, guard: str) -> Edit:
    return Edit(
        id=id,
        anchor=re.escape(before),
        mode="replace",
        text=after,
        guard=re.escape(guard),
        expect_matches=1,
        max_span_lines=max(2, before.count("\n") + 1),
        rationale="VTP31: pipelined prefill over a mixed step of one prefill sequence",
    )


PLUGIN = FilePatch(path="arch/qwen35_fp8/qwen35_fp8.cpp", language="none", description="VTP31 mixed one-sequence cut",
                   edits=(
    _edit(
        "mx1-field",
        """    bool      pipe_tbl_tried = false;
""",
        """    bool      pipe_tbl_tried = false;
    /* BC_VTP31_MX1_FIELD (rad0027): {DT, DT + pipe_rows(T - DT), T} for every T, one pinned table
     * per decode height DT of a mixed step with one prefill sequence, made on first use. */
    int32_t*  pipe_tbl_d[64] = {};
""",
        guard="BC_VTP31_MX1_FIELD",
    ),
    _edit(
        "mx1-free",
        """    if (m.pipe_tbl) rad_dev_free(m.pipe_tbl, RAD_MEM_HOST_PINNED);
""",
        """    if (m.pipe_tbl) rad_dev_free(m.pipe_tbl, RAD_MEM_HOST_PINNED);
    for (int32_t* t : m.pipe_tbl_d) /* BC_VTP31_MX1_FREE (rad0027) */
        if (t) rad_dev_free(t, RAD_MEM_HOST_PINNED);
""",
        guard="BC_VTP31_MX1_FREE",
    ),
    _edit(
        "mx1-ready",
        """    if (batch->phase == RAD_PHASE_MIXED && (dS < 1 || batch->n_seq - dS < 2)) return nullptr;
    if (batch->n_seq > 1) {
""",
        """    /* BC_VTP31_MX1_READY (rad0027): a mixed step with one prefill sequence is cut inside it, the
     * first piece the decode rows and the sequence's first pipe_rows(T - DT) rows. */
    if (batch->phase == RAD_PHASE_MIXED && dS < 1) return nullptr;
    const bool mix1 = batch->phase == RAD_PHASE_MIXED && batch->n_seq - dS == 1;
    if (mix1) {
        if (dT >= 64 || T - dT < 128) return nullptr;
        R = dT + pipe_rows(T - dT);
        if (T - R < 64) return nullptr;
    }
    if (batch->n_seq > 1 && !mix1) {
""",
        guard="BC_VTP31_MX1_READY",
    ),
    _edit(
        "mx1-tbl",
        """    if (!m.pipe_tbl ||
        rad_memcpy_async(d, m.pipe_tbl + 3 * T, 3 * (int64_t)sizeof(int32_t), rad_stream(c)) !=
""",
        """    /* BC_VTP31_MX1_TBL (rad0027): a mixed one-sequence step's bounds come from its decode height's
     * table, made here on first use (so a pass recorded after it replays the same copy). */
    const int32_t* src = m.pipe_tbl ? m.pipe_tbl + 3 * T : nullptr;
    if (mix1) {
        int32_t*& tb = m.pipe_tbl_d[dT];
        if (!tb) {
            const int64_t n = m.g.max_tok + 1;
            tb = (int32_t*)rad_dev_alloc(3 * n * (int64_t)sizeof(int32_t), RAD_MEM_HOST_PINNED);
            if (tb) {
                for (int64_t t = 0; t < n; ++t) {
                    tb[3 * t]     = (int32_t)dT;
                    tb[3 * t + 1] = (int32_t)(t > dT ? dT + pipe_rows(t - dT) : dT);
                    tb[3 * t + 2] = (int32_t)t;
                }
            }
        }
        src = tb ? tb + 3 * T : nullptr;
    }
    if (!src ||
        rad_memcpy_async(d, src, 3 * (int64_t)sizeof(int32_t), rad_stream(c)) !=
""",
        guard="BC_VTP31_MX1_TBL",
    ),
))

ATTN = FilePatch(path="arch/common/rad_block_attn_gated_fp8.h", language="none",
                 description="VTP31 mixed one-sequence cut", edits=(
    _edit(
        "mx1-attn-one",
        """        const bool one = pieces && pieces->seq_cut == 0 && pieces->cu && T <= 2 * pieces->rows &&
                         batch->n_seq == 1;
""",
        """        /* BC_VTP31_MX1_ATTN_ONE (rad0027): or a mixed step's one prefill sequence. */
        const bool one = pieces && pieces->seq_cut == 0 && pieces->cu && T <= 2 * pieces->rows &&
                         (batch->n_seq == 1 ||
                          (batch->phase == RAD_PHASE_MIXED && batch->n_seq_decode > 0 &&
                           batch->n_seq - batch->n_seq_decode == 1));
""",
        guard="BC_VTP31_MX1_ATTN_ONE",
    ),
    _edit(
        "mx1-attn-phase",
        """               (batch->phase == RAD_PHASE_PREFILL || (seqs && batch->phase == RAD_PHASE_MIXED)) &&
""",
        """               (batch->phase == RAD_PHASE_PREFILL ||
                ((seqs || one) && batch->phase == RAD_PHASE_MIXED)) && /* BC_VTP31_MX1_ATTN_PHASE */
""",
        guard="BC_VTP31_MX1_ATTN_PHASE",
    ),
    _edit(
        "mx1-attn-launch",
        """            RAD_ISSUE_N(c, op_attn, n,
                        brows(w.q, T), kv_cache(kv, layer), praw2(table, RAD_I32, 1, tpitch),
                        praw(p == 0 ? batch->positions + R : used, RAD_I32, 1),
                        RAD_NONE, RAD_NONE, praw(pieces->cu + p, RAD_I32, 2),
                        brows(w.attn.x, T));
""",
        """            /* BC_VTP31_MX1_ATTN_LAUNCH (rad0027): a mixed step's decode sequences [0, D) first, as the
             * serial path launches them; the prefill sequence is sequence D and its first piece
             * starts at row DT. */
            int64_t oD = 0, oDT = 0;
            batch_split(batch, &oD, &oDT);
            if (p == 0 && oD > 0)
                RAD_ISSUE_N(c, op_attn, batch->max_q_len_decode,
                            brows(w.q, oDT), kv_cache(kv, layer),
                            praw2(table, RAD_I32, oD, tpitch), praw(used, RAD_I32, oD),
                            RAD_NONE, RAD_NONE, praw(batch->cu_seqlens, RAD_I32, oD + 1),
                            brows(w.attn.x, oDT));
            RAD_ISSUE_N(c, op_attn, p == 0 ? n - oDT : n,
                        brows(w.q, T), kv_cache(kv, layer),
                        praw2(table + oD * tpitch, RAD_I32, 1, tpitch),
                        praw(p == 0 ? batch->positions + R : used + oD, RAD_I32, 1),
                        RAD_NONE, RAD_NONE, praw(pieces->cu + p, RAD_I32, 2),
                        brows(w.attn.x, T));
""",
        guard="BC_VTP31_MX1_ATTN_LAUNCH",
    ),
))

GDN = FilePatch(path="arch/common/rad_block_gdn_fp8.h", language="none", description="VTP31 mixed one-sequence cut",
                edits=(
    _edit(
        "mx1-gdn-one",
        """        const bool one = pieces && pieces->seq_cut == 0 && pieces->cu &&
                         batch->n_tok <= 2 * pieces->rows && batch->n_seq == 1;
""",
        """        /* BC_VTP31_MX1_GDN_ONE (rad0027): or a mixed step's one prefill sequence. */
        const bool one = pieces && pieces->seq_cut == 0 && pieces->cu &&
                         batch->n_tok <= 2 * pieces->rows &&
                         (batch->n_seq == 1 ||
                          (batch->phase == RAD_PHASE_MIXED && batch->n_seq_decode > 0 &&
                           batch->n_seq - batch->n_seq_decode == 1));
""",
        guard="BC_VTP31_MX1_GDN_ONE",
    ),
    _edit(
        "mx1-gdn-phase",
        """               (batch->phase == RAD_PHASE_PREFILL || (seqs && batch->phase == RAD_PHASE_MIXED));
""",
        """               (batch->phase == RAD_PHASE_PREFILL ||
                ((seqs || one) && batch->phase == RAD_PHASE_MIXED)); /* BC_VTP31_MX1_GDN_PHASE */
""",
        guard="BC_VTP31_MX1_GDN_PHASE",
    ),
    _edit(
        "mx1-gdn-prep",
        """            RAD_ISSUE_N(c, op_conv_prep, n,
                        bcol(w.in.x, 0, conv_dim, T), RAD_W(w_conv), RAD_NONE, RAD_W(w_a_log),
                        RAD_W(w_dt_bias), kv_cache(kv_conv, layer), praw(cu, RAD_I32, 2),
                        bcol(w.ab, 0, H, T), bcol(w.ab, H, H, T),
                        praw(cv_idx, RAD_I32, 1), praw(batch->positions + r0, RAD_I32, 1),
""",
        """            /* BC_VTP31_MX1_GDN_PREP (rad0027): the prefill sequence is sequence D (rad0026's decode
             * half ran above), its first piece from row DT. */
            const int64_t o0 = p == 0 ? mDT : 0;
            RAD_ISSUE_N(c, op_conv_prep, n - o0,
                        bcol(w.in.x, 0, conv_dim, T), RAD_W(w_conv), RAD_NONE, RAD_W(w_a_log),
                        RAD_W(w_dt_bias), kv_cache(kv_conv, layer), praw(cu, RAD_I32, 2),
                        bcol(w.ab, 0, H, T), bcol(w.ab, H, H, T),
                        praw(cv_idx ? cv_idx + mD : nullptr, RAD_I32, 1),
                        praw(batch->positions + r0 + o0, RAD_I32, 1),
""",
        guard="BC_VTP31_MX1_GDN_PREP",
    ),
    _edit(
        "mx1-gdn-kkt",
        """            RAD_ISSUE_N(c, op_kkt, n,
""",
        """            RAD_ISSUE_N(c, op_kkt, n - o0, /* BC_VTP31_MX1_GDN_KKT */
""",
        guard="BC_VTP31_MX1_GDN_KKT",
    ),
    _edit(
        "mx1-gdn-scan",
        """            RAD_ISSUE_N(c, op_scan, n,
""",
        """            RAD_ISSUE_N(c, op_scan, n - o0, /* BC_VTP31_MX1_GDN_SCAN */
""",
        guard="BC_VTP31_MX1_GDN_SCAN",
    ),
    _edit(
        "mx1-gdn-state",
        """                        praw2(st_idx, RAD_I32, 1, st_w));
""",
        """                        praw2(st_idx ? st_idx + mD * st_w : nullptr, RAD_I32, 1, st_w)); /* BC_VTP31_MX1_GDN_ST */
""",
        guard="BC_VTP31_MX1_GDN_ST",
    ),
))

PATCHES = (PLUGIN, ATTN, GDN)
