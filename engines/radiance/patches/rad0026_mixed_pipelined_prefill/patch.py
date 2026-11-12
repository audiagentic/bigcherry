"""rad0026 (VTP31): the pipelined prefill over a mixed step, its decode sequences in the first piece.

rad0025 pipelines a pure prefill step of several sequences, cut between them. A step that also carries decode rows (a
mixed step: every multi-stream run once the first streams are generating while later prompts still prefill) stays
serial, and at 8 streams that is most of the prefill. Each block's serial path already splits a mixed step into a
decode prefix (sequences `[0, D)`, rows `[0, DT)`, the decode and verify launches) and a prefill tail (`[D, S)`, the
chunked launches over the step's absolute bounds). This patch keeps that split and cuts the tail between two prefill
sequences: the first piece is the decode prefix plus the prefill sequences `[D, seq_cut)`, the second the rest.

- `pipe_ready` takes a mixed step with two or more prefill sequences; the cut is the prefill boundary nearest half
  the rows (both pieces at least 64 rows, as before). A mixed step with one prefill sequence stays serial, silently.
- Attention: the first piece issues the serial path's decode launch over `[0, D)` and then its prefill sequences;
  where the serial mixed path launches its prefill half at the top band (`g.max_tok`), both pieces do.
- Delta net: the first piece issues the serial path's decode-half ops over `[0, D)` (the fused or split recurrent
  update, which gates and quantises its rows), then the chunked ops over its prefill sequences; the gated norm and the
  quantiser run over the prefill rows only, as in the serial path.
- `pass_key` keys a mixed pass of several sequences by the host bounds too (rad0025 keys a prefill pass).

Every launch is one the serial step issues, over the same sequences and band, so the output is the same bits.
Source-pinned to radiance v1.17.0 (19d4e8fe), after rad0025.
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
        rationale="VTP31: pipelined prefill over a mixed step",
    )


KEY = FilePatch(path="core/runtime/ctx.cpp", language="none", description="VTP31 mixed pass key", edits=(
    _edit(
        "mxp-key",
        "    if (b->phase == RAD_PHASE_PREFILL && b->n_seq > 1 && b->cu_seqlens_h)\n",
        "    if ((b->phase == RAD_PHASE_PREFILL || b->phase == RAD_PHASE_MIXED) && /* BC_VTP31_MXP_KEY (rad0026) */\n"
        "        b->n_seq > 1 && b->cu_seqlens_h)\n",
        guard="BC_VTP31_MXP_KEY",
    ),
))

ATTN = FilePatch(path="arch/common/rad_block_attn_gated_fp8.h", language="none", description="VTP31 mixed cut",
                 edits=(
    _edit(
        "mxp-attn-takes",
        "        const bool seqs = pieces && pieces->seq_cut > 0 && pieces->seq_cut < batch->n_seq &&\n"
        "                          batch->n_seq_decode == 0;\n"
        "        return pieces && pieces->in_ready && pieces->cuts(T) && (one || seqs) &&\n"
        "               !op_add && !op_fill && op_gq && !op_sig && !qsa_sel &&\n"
        "               batch->phase == RAD_PHASE_PREFILL &&\n",
        "        /* BC_VTP31_MXP_ATTN_TAKES (rad0026): a mixed step's decode sequences ride in the first piece. */\n"
        "        int64_t dS = 0, dT = 0;\n"
        "        batch_split(batch, &dS, &dT);\n"
        "        const bool seqs = pieces && pieces->seq_cut > dS && pieces->seq_cut < batch->n_seq;\n"
        "        return pieces && pieces->in_ready && pieces->cuts(T) && (one || seqs) &&\n"
        "               !op_add && !op_fill && op_gq && !op_sig && !qsa_sel &&\n"
        "               (batch->phase == RAD_PHASE_PREFILL || (seqs && batch->phase == RAD_PHASE_MIXED)) &&\n",
        guard="BC_VTP31_MXP_ATTN_TAKES",
    ),
    _edit(
        "mxp-attn-seqs",
        "                const int64_t s0 = p == 0 ? 0 : pieces->seq_cut;\n"
        "                const int64_t ns = p == 0 ? pieces->seq_cut : batch->n_seq - pieces->seq_cut;\n"
        "                RAD_ISSUE_N(c, op_attn, batch->max_q_len,\n",
        "                int64_t s0 = p == 0 ? 0 : pieces->seq_cut;\n"
        "                int64_t ns = p == 0 ? pieces->seq_cut : batch->n_seq - pieces->seq_cut;\n"
        "                /* BC_VTP31_MXP_ATTN_SEQS (rad0026): a mixed step as the serial path launches it, the\n"
        "                 * decode sequences [0, D) at their own band in the first piece, the prefill sequences\n"
        "                 * at the top band where the serial path splits the two. */\n"
        "                int64_t mD = 0, mDT = 0;\n"
        "                batch_split(batch, &mD, &mDT);\n"
        "                const int32_t mqp = batch->max_q_len_prefill > 0 ? batch->max_q_len_prefill\n"
        "                                                                 : batch->max_q_len;\n"
        "                const bool msplit = batch->phase == RAD_PHASE_MIXED && mD > 0 && mD < batch->n_seq &&\n"
        "                                    batch->max_q_len_decode > 0 && mqp > batch->max_q_len_decode;\n"
        "                if (msplit && p == 0) {\n"
        "                    RAD_ISSUE_N(c, op_attn, batch->max_q_len_decode,\n"
        "                                brows(w.q, mDT), kv_cache(kv, layer),\n"
        "                                praw2(table, RAD_I32, mD, tpitch), praw(used, RAD_I32, mD),\n"
        "                                RAD_NONE, RAD_NONE, praw(batch->cu_seqlens, RAD_I32, mD + 1),\n"
        "                                brows(w.attn.x, mDT));\n"
        "                    s0 = mD;\n"
        "                    ns = pieces->seq_cut - mD;\n"
        "                }\n"
        "                RAD_ISSUE_N(c, op_attn, msplit ? (int64_t)g.max_tok : (int64_t)batch->max_q_len,\n",
        guard="BC_VTP31_MXP_ATTN_SEQS",
    ),
))

GDN = FilePatch(path="arch/common/rad_block_gdn_fp8.h", language="none", description="VTP31 mixed cut", edits=(
    _edit(
        "mxp-gdn-takes",
        "        const bool seqs = pieces && pieces->seq_cut > 0 && pieces->seq_cut < batch->n_seq &&\n"
        "                          batch->n_seq_decode == 0;\n"
        "        return pieces && pieces->cuts(batch->n_tok) && (one || seqs) &&\n"
        "               (ext_in || pieces->in_ready) && !op_add && batch->phase == RAD_PHASE_PREFILL;\n",
        "        /* BC_VTP31_MXP_GDN_TAKES (rad0026): a mixed step's decode sequences ride in the first piece. */\n"
        "        int64_t dS = 0, dT = 0;\n"
        "        batch_split(batch, &dS, &dT);\n"
        "        const bool seqs = pieces && pieces->seq_cut > dS && pieces->seq_cut < batch->n_seq;\n"
        "        return pieces && pieces->cuts(batch->n_tok) && (one || seqs) &&\n"
        "               (ext_in || pieces->in_ready) && !op_add &&\n"
        "               (batch->phase == RAD_PHASE_PREFILL || (seqs && batch->phase == RAD_PHASE_MIXED));\n",
        guard="BC_VTP31_MXP_GDN_TAKES",
    ),
    _edit(
        "mxp-gdn-dec",
        "            if (pieces->seq_cut > 0) {\n"
        "                /* BC_VTP31_MSP_GDN_SEQS (rad0025): whole sequences [s0, s0 + ns), as the serial path's\n",
        "            /* BC_VTP31_MXP_GDN_DEC (rad0026): a mixed step's decode half first, in the first piece, as\n"
        "             * the serial path issues it over sequences [0, D) and rows [0, DT). */\n"
        "            int64_t mD = 0, mDT = 0;\n"
        "            batch_split(batch, &mD, &mDT);\n"
        "            if (p == 0 && mD > 0) {\n"
        "                const int32_t qd = batch->max_q_len_decode;\n"
        "                if (op_cr && qd <= cr_rows) {\n"
        "                    RAD_ISSUE_N(c, op_cr, qd,\n"
        "                                bcol(w.in.x, 0, conv_dim, mDT), RAD_W(w_conv), RAD_NONE,\n"
        "                                kv_cache(kv_conv, layer), praw(cv_idx, RAD_I32, mD),\n"
        "                                bcol(w.ab, 0, H, mDT), bcol(w.ab, H, H, mDT),\n"
        "                                RAD_W(w_a_log), RAD_W(w_dt_bias), kv_cache(kv_state, layer),\n"
        "                                praw2(st_idx, RAD_I32, mD, st_w),\n"
        "                                praw(batch->num_accepted, RAD_I32, mD),\n"
        "                                praw(batch->cu_seqlens, RAD_I32, mD + 1),\n"
        "                                bcol(w.in.x, conv_dim, v_dim, mDT), RAD_W(w_out_norm),\n"
        "                                brows(w.o.x, mDT),\n"
        "                                w.o.cq() ? brows(w.o.cq(), mDT) : RAD_NONE,\n"
        "                                w.o.cq() ? brows(w.o.cs(), mDT) : RAD_NONE);\n"
        "                } else {\n"
        "                    RAD_ISSUE_N(c, op_conv_update, qd,\n"
        "                                bcol(w.in.x, 0, conv_dim, mDT), RAD_W(w_conv), RAD_NONE,\n"
        "                                kv_cache(kv_conv, layer), praw(cv_idx, RAD_I32, mD),\n"
        "                                praw(batch->num_accepted, RAD_I32, mD),\n"
        "                                praw(batch->cu_seqlens, RAD_I32, mD + 1),\n"
        "                                brows(w.q, mDT), brows(w.k, mDT), brows(w.v, mDT));\n"
        "                    RAD_ISSUE_N(c, op_recur, qd,\n"
        "                                brows(w.q, mDT), brows(w.k, mDT), brows(w.v, mDT),\n"
        "                                bcol(w.ab, 0, H, mDT), bcol(w.ab, H, H, mDT),\n"
        "                                RAD_W(w_a_log), RAD_W(w_dt_bias), kv_cache(kv_state, layer),\n"
        "                                praw2(st_idx, RAD_I32, mD, st_w),\n"
        "                                praw(batch->num_accepted, RAD_I32, mD),\n"
        "                                praw(batch->cu_seqlens, RAD_I32, mD + 1),\n"
        "                                bcol(w.in.x, conv_dim, v_dim, mDT), RAD_W(w_out_norm),\n"
        "                                brows(w.o.x, mDT),\n"
        "                                w.o.cq() && !w.o.q8_rot ? brows(w.o.cq(), mDT) : RAD_NONE,\n"
        "                                w.o.cq() && !w.o.q8_rot ? brows(w.o.cs(), mDT) : RAD_NONE);\n"
        "                    if (w.o.q8_rot) q_o.step(c, T, 0, mDT);\n"
        "                }\n"
        "            }\n"
        "            if (pieces->seq_cut > 0) {\n"
        "                /* BC_VTP31_MSP_GDN_SEQS (rad0025): whole sequences [s0, s0 + ns), as the serial path's\n",
        guard="BC_VTP31_MXP_GDN_DEC",
    ),
    _edit(
        "mxp-gdn-seqs",
        "                const int64_t s0 = p == 0 ? 0 : pieces->seq_cut;\n"
        "                const int64_t ns = p == 0 ? pieces->seq_cut : batch->n_seq - pieces->seq_cut;\n"
        "                const int32_t* scu = batch->cu_seqlens + s0;\n",
        "                int64_t s0 = p == 0 ? 0 : pieces->seq_cut;\n"
        "                int64_t ns = p == 0 ? pieces->seq_cut : batch->n_seq - pieces->seq_cut;\n"
        "                if (p == 0 && mD > 0) { /* BC_VTP31_MXP_GDN_SEQS (rad0026): after the decode half */\n"
        "                    s0 = mD;\n"
        "                    ns = pieces->seq_cut - mD;\n"
        "                }\n"
        "                const int32_t* scu = batch->cu_seqlens + s0;\n",
        guard="BC_VTP31_MXP_GDN_SEQS",
    ),
    _edit(
        "mxp-gdn-gnorm",
        "            } /* BC_VTP31_MSP_GDN_CLOSE (rad0025) */\n"
        "            RAD_ISSUE_N(c, op_gnorm, n,\n"
        "                        brow_slice(w.o.x, r0, n, v_dim),\n"
        "                        bcol_at(w.in.x, r0, in_w, conv_dim, v_dim, n),\n"
        "                        RAD_W(w_out_norm), brow_slice(w.o.x, r0, n, v_dim));\n"
        "            q_o.step(c, T, r0, n);\n",
        "            } /* BC_VTP31_MSP_GDN_CLOSE (rad0025) */\n"
        "            /* BC_VTP31_MXP_GDN_GNORM (rad0026): the recurrent update gated and quantised a mixed\n"
        "             * step's decode rows, so the first piece norms and quantises its prefill rows only. */\n"
        "            int64_t gD = 0, gDT = 0;\n"
        "            batch_split(batch, &gD, &gDT);\n"
        "            const int64_t g0 = p == 0 ? r0 + gDT : r0;\n"
        "            const int64_t gn = p == 0 ? n - gDT : n;\n"
        "            RAD_ISSUE_N(c, op_gnorm, gn,\n"
        "                        brow_slice(w.o.x, g0, gn, v_dim),\n"
        "                        bcol_at(w.in.x, g0, in_w, conv_dim, v_dim, gn),\n"
        "                        RAD_W(w_out_norm), brow_slice(w.o.x, g0, gn, v_dim));\n"
        "            q_o.step(c, T, g0, gn);\n",
        guard="BC_VTP31_MXP_GDN_GNORM",
    ),
))

PLUGIN = FilePatch(path="arch/qwen35_fp8/qwen35_fp8.cpp", language="none", description="VTP31 mixed cut", edits=(
    _edit(
        "mxp-ready-phase",
        "        batch->phase != RAD_PHASE_PREFILL)\n"
        "        return nullptr;\n",
        "        (batch->phase != RAD_PHASE_PREFILL && batch->phase != RAD_PHASE_MIXED)) /* BC_VTP31_MXP_READY_PHASE */\n"
        "        return nullptr;\n",
        guard="BC_VTP31_MXP_READY_PHASE",
    ),
    _edit(
        "mxp-ready-dec",
        "    if (batch->n_seq_decode != 0) return serial(\"decode rows in the step\");\n"
        "    if (batch->n_seq > 1) {\n",
        "    /* BC_VTP31_MXP_READY_DEC (rad0026): a mixed step's decode sequences [0, D) go in the first\n"
        "     * piece and the cut is between two of its prefill sequences; a mixed step with one prefill\n"
        "     * sequence stays serial (no note: the next pure prefill step's says more). */\n"
        "    int64_t dS = 0, dT = 0;\n"
        "    batch_split(batch, &dS, &dT);\n"
        "    if (batch->phase == RAD_PHASE_MIXED && (dS < 1 || batch->n_seq - dS < 2)) return nullptr;\n"
        "    if (batch->n_seq > 1) {\n",
        guard="BC_VTP31_MXP_READY_DEC",
    ),
    _edit(
        "mxp-ready-loop",
        "        for (int64_t s = 1; s < batch->n_seq; ++s) {\n",
        "        for (int64_t s = dS + 1; s < batch->n_seq; ++s) { /* BC_VTP31_MXP_READY_LOOP */\n",
        guard="BC_VTP31_MXP_READY_LOOP",
    ),
))

PATCHES = (KEY, ATTN, GDN, PLUGIN)
