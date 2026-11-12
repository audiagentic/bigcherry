"""rad0025 (VTP31): the pipelined prefill over a step of several whole sequences, cut between them.

`qwen35_fp8` (and `qwen35_w8a8`, built from it) pipelines a wide prefill step (step_pipelined, rad0020): two pieces,
each block's all-reduce for one piece on the second lane beside the other piece's work. It takes a step of one
sequence only, cut inside it; a step that prefills several prompts at once (every multi-stream burst) runs serially,
and at three ranks such a step spends about 70% of its device time in collectives (VTP31 profile).

A step of several whole prefill sequences can be cut between them instead. The pieces then share no attention keys and
no recurrent state, and each block's serial path already runs over a contiguous run of sequences by offsetting the
per-sequence arrays over the step's absolute bounds (the delta net's prefill half `[D, S)`, the attention's mixed-step
launches). This patch:

- gives `RadBatch` a host copy of `cu_seqlens` (`cu_seqlens_h`, trailing, filled by the scheduler beside
  `positions_h`; null from any other driver), so the plugin places the cut without reading the card;
- lets `pipe_ready` take a step of two or more prefill sequences and no decode rows, cut at the sequence boundary
  nearest half the rows, both pieces at least 64 rows (`ArPieces::seq_cut`); a step of one sequence is cut as before;
- passes the first piece's rows through step_pipelined and the hand-off instead of recomputing `pipe_rows(T)`;
- runs the attention and delta-net piece paths over each piece's slice of the per-sequence arrays (query bounds,
  seqused, block table rows; convolution and state indices, context lengths) at the serial launch's band.

Every row's arithmetic is the serial step's (the per-sequence kernels see the same sequences, the per-row ops the same
rows, in the band of the whole step), so the output is the same bits. Source-pinned to radiance v1.17.0 (19d4e8fe),
after rad0020.
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
        rationale="VTP31: pipelined prefill over several sequences",
    )


ABI = FilePatch(path="abi/rad_runtime.h", language="none", description="VTP31 host cu_seqlens", edits=(
    _edit(
        "msp-abi",
        "    const int32_t* bidi_used;          /* device */\n"
        "} RadBatch;\n",
        "    const int32_t* bidi_used;          /* device */\n"
        "    /* BC_VTP31_MSP_ABI (rad0025): [n_seq + 1] host copy of cu_seqlens, or null (a driver that does\n"
        "     * not fill it), so a plugin can cut a step between sequences without reading the card. */\n"
        "    const int32_t* cu_seqlens_h;\n"
        "} RadBatch;\n",
        guard="BC_VTP31_MSP_ABI",
    ),
))

CORE = FilePatch(path="core/sched/batch.cpp", language="none", description="VTP31 host cu_seqlens", edits=(
    _edit(
        "msp-core",
        "        B.positions_h = s.h_pos;\n"
        "        B.cu_seqlens = on_rank(base, b0, s.d_cu);\n",
        "        B.positions_h = s.h_pos;\n"
        "        B.cu_seqlens = on_rank(base, b0, s.d_cu);\n"
        "        B.cu_seqlens_h = s.h_cu; /* BC_VTP31_MSP_CORE (rad0025) */\n",
        guard="BC_VTP31_MSP_CORE",
    ),
))

PIECES = FilePatch(path="arch/common/rad_fp8.h", language="none", description="VTP31 sequence cut", edits=(
    _edit(
        "msp-pieces",
        "    bool in_ready = false;\n"
        "    bool cuts(int64_t T) const { return done && rows > 0 && T > rows; }\n",
        "    bool in_ready = false;\n"
        "    /* BC_VTP31_MSP_PIECES (rad0025): a step of several whole prefill sequences cut between them,\n"
        "     * the first piece sequences [0, seq_cut) and `rows` rows, the second the rest; each block runs\n"
        "     * a piece over its slice of the per-sequence arrays. 0 for a cut inside one sequence (cu). */\n"
        "    int64_t seq_cut = 0;\n"
        "    bool cuts(int64_t T) const { return done && rows > 0 && T > rows; }\n",
        guard="BC_VTP31_MSP_PIECES",
    ),
))

ATTN = FilePatch(path="arch/common/rad_block_attn_gated_fp8.h", language="none", description="VTP31 sequence cut",
                 edits=(
    _edit(
        "msp-attn-takes",
        "        return pieces && pieces->in_ready && pieces->cu && pieces->cuts(T) &&\n"
        "               T <= 2 * pieces->rows && !op_add && !op_fill && op_gq && !op_sig && !qsa_sel &&\n"
        "               batch->n_seq == 1 && batch->phase == RAD_PHASE_PREFILL &&\n",
        "        /* BC_VTP31_MSP_ATTN_TAKES (rad0025): or several whole prefill sequences cut between them. */\n"
        "        const bool one = pieces && pieces->seq_cut == 0 && pieces->cu && T <= 2 * pieces->rows &&\n"
        "                         batch->n_seq == 1;\n"
        "        const bool seqs = pieces && pieces->seq_cut > 0 && pieces->seq_cut < batch->n_seq &&\n"
        "                          batch->n_seq_decode == 0;\n"
        "        return pieces && pieces->in_ready && pieces->cuts(T) && (one || seqs) &&\n"
        "               !op_add && !op_fill && op_gq && !op_sig && !qsa_sel &&\n"
        "               batch->phase == RAD_PHASE_PREFILL &&\n",
        guard="BC_VTP31_MSP_ATTN_TAKES",
    ),
    _edit(
        "msp-attn-seqs",
        "            RAD_ISSUE_N(c, op_attn, n,\n"
        "                        brows(w.q, T), kv_cache(kv, layer), praw2(table, RAD_I32, 1, tpitch),\n"
        "                        praw(p == 0 ? batch->positions + R : used, RAD_I32, 1),\n"
        "                        RAD_NONE, RAD_NONE, praw(pieces->cu + p, RAD_I32, 2),\n"
        "                        brows(w.attn.x, T));\n",
        "            if (pieces->seq_cut > 0) {\n"
        "                /* BC_VTP31_MSP_ATTN_SEQS (rad0025): whole sequences [s0, s0 + ns), their slice of the\n"
        "                 * per-sequence arrays over the step's absolute bounds, at the serial launch's band. */\n"
        "                const int64_t s0 = p == 0 ? 0 : pieces->seq_cut;\n"
        "                const int64_t ns = p == 0 ? pieces->seq_cut : batch->n_seq - pieces->seq_cut;\n"
        "                RAD_ISSUE_N(c, op_attn, batch->max_q_len,\n"
        "                            brows(w.q, T), kv_cache(kv, layer),\n"
        "                            praw2(table + s0 * tpitch, RAD_I32, ns, tpitch),\n"
        "                            praw(used + s0, RAD_I32, ns),\n"
        "                            RAD_NONE, RAD_NONE, praw(batch->cu_seqlens + s0, RAD_I32, ns + 1),\n"
        "                            brows(w.attn.x, T));\n"
        "            } else {\n"
        "            RAD_ISSUE_N(c, op_attn, n,\n"
        "                        brows(w.q, T), kv_cache(kv, layer), praw2(table, RAD_I32, 1, tpitch),\n"
        "                        praw(p == 0 ? batch->positions + R : used, RAD_I32, 1),\n"
        "                        RAD_NONE, RAD_NONE, praw(pieces->cu + p, RAD_I32, 2),\n"
        "                        brows(w.attn.x, T));\n"
        "            }\n",
        guard="BC_VTP31_MSP_ATTN_SEQS",
    ),
))

GDN = FilePatch(path="arch/common/rad_block_gdn_fp8.h", language="none", description="VTP31 sequence cut", edits=(
    _edit(
        "msp-gdn-takes",
        "        return pieces && pieces->cu && pieces->cuts(batch->n_tok) &&\n"
        "               batch->n_tok <= 2 * pieces->rows && (ext_in || pieces->in_ready) && !op_add &&\n"
        "               batch->n_seq == 1 && batch->phase == RAD_PHASE_PREFILL;\n",
        "        /* BC_VTP31_MSP_GDN_TAKES (rad0025): or several whole prefill sequences cut between them. */\n"
        "        const bool one = pieces && pieces->seq_cut == 0 && pieces->cu &&\n"
        "                         batch->n_tok <= 2 * pieces->rows && batch->n_seq == 1;\n"
        "        const bool seqs = pieces && pieces->seq_cut > 0 && pieces->seq_cut < batch->n_seq &&\n"
        "                          batch->n_seq_decode == 0;\n"
        "        return pieces && pieces->cuts(batch->n_tok) && (one || seqs) &&\n"
        "               (ext_in || pieces->in_ready) && !op_add && batch->phase == RAD_PHASE_PREFILL;\n",
        guard="BC_VTP31_MSP_GDN_TAKES",
    ),
    _edit(
        "msp-gdn-seqs",
        "            RAD_ISSUE_N(c, op_conv_prep, n,\n"
        "                        bcol(w.in.x, 0, conv_dim, T), RAD_W(w_conv), RAD_NONE, RAD_W(w_a_log),\n"
        "                        RAD_W(w_dt_bias), kv_cache(kv_conv, layer), praw(cu, RAD_I32, 2),\n",
        "            if (pieces->seq_cut > 0) {\n"
        "                /* BC_VTP31_MSP_GDN_SEQS (rad0025): whole sequences [s0, s0 + ns), as the serial path's\n"
        "                 * prefill half: their slice of the per-sequence arrays, token operands at their base\n"
        "                 * (every kernel here reads its rows from the absolute bounds). */\n"
        "                const int64_t s0 = p == 0 ? 0 : pieces->seq_cut;\n"
        "                const int64_t ns = p == 0 ? pieces->seq_cut : batch->n_seq - pieces->seq_cut;\n"
        "                const int32_t* scu = batch->cu_seqlens + s0;\n"
        "                RAD_ISSUE(c, op_conv_prep,\n"
        "                          bcol(w.in.x, 0, conv_dim, T), RAD_W(w_conv), RAD_NONE, RAD_W(w_a_log),\n"
        "                          RAD_W(w_dt_bias), kv_cache(kv_conv, layer), praw(scu, RAD_I32, ns + 1),\n"
        "                          bcol(w.ab, 0, H, T), bcol(w.ab, H, H, T),\n"
        "                          praw(cv_idx ? cv_idx + s0 : nullptr, RAD_I32, ns),\n"
        "                          praw(batch->ctx_lens + s0, RAD_I32, ns),\n"
        "                          brows(w.q, T), brows(w.k, T), brows(w.v, T),\n"
        "                          brows(w.gdec, T), brows(w.beta, T));\n"
        "                RAD_ISSUE(c, op_kkt,\n"
        "                          brows(w.k, T), brows(w.beta, T), brows(w.gdec, T),\n"
        "                          praw(scu, RAD_I32, ns + 1), brows(w.kkt, T));\n"
        "                RAD_ISSUE(c, op_scan,\n"
        "                          brows(w.q, T), brows(w.k, T), brows(w.v, T), brows(w.kkt, T),\n"
        "                          brows(w.gdec, T), brows(w.beta, T),\n"
        "                          kv_cache(kv_state, layer), praw(scu, RAD_I32, ns + 1),\n"
        "                          brows(w.o.x, T), kv_cache(kv_state, layer),\n"
        "                          praw2(st_idx ? st_idx + s0 * st_w : nullptr, RAD_I32, ns, st_w));\n"
        "            } else {\n"
        "            RAD_ISSUE_N(c, op_conv_prep, n,\n"
        "                        bcol(w.in.x, 0, conv_dim, T), RAD_W(w_conv), RAD_NONE, RAD_W(w_a_log),\n"
        "                        RAD_W(w_dt_bias), kv_cache(kv_conv, layer), praw(cu, RAD_I32, 2),\n",
        guard="BC_VTP31_MSP_GDN_SEQS",
    ),
    _edit(
        "msp-gdn-close",
        "                        brows(w.o.x, T), kv_cache(kv_state, layer),\n"
        "                        praw2(st_idx, RAD_I32, 1, st_w));\n",
        "                        brows(w.o.x, T), kv_cache(kv_state, layer),\n"
        "                        praw2(st_idx, RAD_I32, 1, st_w));\n"
        "            } /* BC_VTP31_MSP_GDN_CLOSE (rad0025) */\n",
        guard="BC_VTP31_MSP_GDN_CLOSE",
    ),
))

PLUGIN = FilePatch(path="arch/qwen35_fp8/qwen35_fp8.cpp", language="none", description="VTP31 sequence cut", edits=(
    _edit(
        "msp-hand-struct",
        "    int64_t      T = 0, s = 0;\n"
        "};\n"
        "static void pipe_hand(void* u, int64_t r0, int64_t n) {\n",
        "    int64_t      T = 0, s = 0;\n"
        "    int64_t      R = 0; /* BC_VTP31_MSP_HAND (rad0025): the first piece's rows */\n"
        "};\n"
        "static void pipe_hand(void* u, int64_t r0, int64_t n) {\n",
        guard="BC_VTP31_MSP_HAND",
    ),
    _edit(
        "msp-hand-rows",
        "        const int64_t R = pipe_rows(h.T);\n"
        "        pipe_taps(h.c, *h.m, h.s / 2 - 1, R, h.T - R);\n",
        "        const int64_t R = h.R; /* BC_VTP31_MSP_HAND_ROWS (rad0025) */\n"
        "        pipe_taps(h.c, *h.m, h.s / 2 - 1, R, h.T - R);\n",
        guard="BC_VTP31_MSP_HAND_ROWS",
    ),
    _edit(
        "msp-ready-sig",
        "static const int32_t* pipe_ready(RadCtx* c, const RadBatch* batch, Model& m, bool* failed) {\n",
        "static const int32_t* pipe_ready(RadCtx* c, const RadBatch* batch, Model& m, bool* failed,\n"
        "                                 int64_t* rows_out, int64_t* seq_cut_out) { /* BC_VTP31_MSP_READY_SIG */\n",
        guard="BC_VTP31_MSP_READY_SIG",
    ),
    _edit(
        "msp-ready-seqs",
        "    if (batch->n_seq != 1 || batch->n_seq_decode != 0) return serial(\"more than one sequence\");\n",
        "    /* BC_VTP31_MSP_READY (rad0025): a step of several whole prefill sequences is cut between them,\n"
        "     * at the boundary nearest half the rows, both pieces at least 64 rows; one sequence is cut\n"
        "     * inside it as before. Every rank reads the same bounds, so all cut alike. */\n"
        "    int64_t seq_cut = 0, R = pipe_rows(T);\n"
        "    if (batch->n_seq_decode != 0) return serial(\"decode rows in the step\");\n"
        "    if (batch->n_seq > 1) {\n"
        "        const int32_t* hc = batch->cu_seqlens_h;\n"
        "        if (!hc) return serial(\"more than one sequence and no host bounds\");\n"
        "        int64_t best = -1;\n"
        "        for (int64_t s = 1; s < batch->n_seq; ++s) {\n"
        "            const int64_t d = 2 * (int64_t)hc[s] - T;\n"
        "            const int64_t ad = d < 0 ? -d : d;\n"
        "            if (best < 0 || ad < best) { best = ad; seq_cut = s; }\n"
        "        }\n"
        "        R = hc[seq_cut];\n"
        "        if (R < 64 || T - R < 64) return serial(\"no sequence boundary leaves both pieces 64 rows\");\n"
        "    }\n",
        guard="BC_VTP31_MSP_READY ",
    ),
    _edit(
        "msp-ready-pieces",
        "    const int64_t R = pipe_rows(T);\n"
        "    ArPieces pc{};\n"
        "    pc.rows = R; pc.done = pipe_hand; pc.cu = d; pc.in_ready = true;\n",
        "    ArPieces pc{};\n"
        "    pc.rows = R; pc.done = pipe_hand; pc.cu = d; pc.in_ready = true;\n"
        "    pc.seq_cut = seq_cut; /* BC_VTP31_MSP_READY_PIECES */\n",
        guard="BC_VTP31_MSP_READY_PIECES",
    ),
    _edit(
        "msp-ready-out",
        "        *failed = true;\n"
        "        return nullptr;\n"
        "    }\n"
        "    return d;\n"
        "}\n",
        "        *failed = true;\n"
        "        return nullptr;\n"
        "    }\n"
        "    *rows_out = R; *seq_cut_out = seq_cut; /* BC_VTP31_MSP_READY_OUT */\n"
        "    return d;\n"
        "}\n",
        guard="BC_VTP31_MSP_READY_OUT",
    ),
    _edit(
        "msp-step-sig",
        "static void step_pipelined(RadCtx* c, const RadBatch* batch, Model& m, const int32_t* pcu) {\n"
        "    const Geom& g = m.g;\n"
        "    const int64_t T = batch->n_tok, R = pipe_rows(T);\n",
        "static void step_pipelined(RadCtx* c, const RadBatch* batch, Model& m, const int32_t* pcu,\n"
        "                           int64_t R, int64_t seq_cut) { /* BC_VTP31_MSP_STEP (rad0025) */\n"
        "    const Geom& g = m.g;\n"
        "    const int64_t T = batch->n_tok;\n",
        guard="BC_VTP31_MSP_STEP",
    ),
    _edit(
        "msp-step-loop",
        "        PipeHand h{c, &m, T, 2 * li};\n"
        "        ArPieces pc{};\n"
        "        pc.rows = R; pc.done = pipe_hand; pc.u = &h; pc.cu = pcu; pc.in_ready = true;\n",
        "        PipeHand h{c, &m, T, 2 * li, R};\n"
        "        ArPieces pc{};\n"
        "        pc.rows = R; pc.done = pipe_hand; pc.u = &h; pc.cu = pcu; pc.in_ready = true;\n"
        "        pc.seq_cut = seq_cut; /* BC_VTP31_MSP_LOOP */\n",
        guard="BC_VTP31_MSP_LOOP",
    ),
    _edit(
        "msp-call",
        "    if (const int32_t* pcu = pipe_ready(c, batch, m, &pipe_failed)) {\n"
        "        step_pipelined(c, batch, m, pcu);\n",
        "    int64_t pipe_r = 0, pipe_cut = 0; /* BC_VTP31_MSP_CALL (rad0025) */\n"
        "    if (const int32_t* pcu = pipe_ready(c, batch, m, &pipe_failed, &pipe_r, &pipe_cut)) {\n"
        "        step_pipelined(c, batch, m, pcu, pipe_r, pipe_cut);\n",
        guard="BC_VTP31_MSP_CALL",
    ),
))

PATCHES = (ABI, CORE, PIECES, ATTN, GDN, PLUGIN)
