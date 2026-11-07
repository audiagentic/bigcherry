"""Geometry-safe recorded-pass replay of Qwen4-Exp MTP decode draft/verify passes."""

import re

from bigcherry.patcher import Edit, FilePatch

OLD = """    /* MTP owns another attention path on the same KV group. The present batch key cannot prove
     * whether that head's attention row takes a split grid, so keep only those passes live. */
    if (batch->draft_pass != 0) return 0;
    /* Mixed batches split the query rows into decode and prefill subsets before attention, and a
     * speculative verify can choose another q_len band. The whole-batch host summary cannot key
     * those subset geometries exactly. */
    if (batch->phase == RAD_PHASE_MIXED || batch->n_spec > 0) return 0;
    if (batch->phase != RAD_PHASE_PREFILL) return 1;"""

NEW = """    /* BC_MTP_DECODE_REPLAY_GEOMETRY: MTP uses the same KV group as the trunk. The h256
     * gqa6/gqa12 decode rows accept at most 10/5 query tokens respectively. Wider
     * verification can choose the live-context prefill splitter: NEVER tape it. Mixed
     * batches split into two independently shaped subsets, so they stay live as well.
     * Core's pass key additionally covers all batch/pointer/KV-view inputs. */
    if (batch->phase == RAD_PHASE_MIXED) return 0;
    if (batch->draft_pass != 0 || batch->n_spec > 0) {
        if (batch->phase != RAD_PHASE_DECODE || batch->draft_pass == 1 ||
            batch->n_spec < 0 || batch->n_tok <= 0 || batch->n_seq <= 0 ||
            batch->n_tok < batch->n_seq ||
            batch->max_q_len <= 0 || batch->max_q_len != batch->max_q_len_decode ||
            batch->max_q_len_prefill != 0 ||
            (batch->n_seq_decode > 0 && batch->n_seq_decode != batch->n_seq) ||
            (batch->n_tok_decode > 0 && batch->n_tok_decode != batch->n_tok) ||
            (batch->n_tok - 1) / batch->n_seq + 1 > batch->max_q_len)
            return 0;
        const Model& m = g_model[rad_rank(c)];
        const int gqa = (int)m.g.gqa();
        if (m.g.head_dim != 256 || (gqa != 6 && gqa != 12) ||
            m.g.n_head_kv <= 0 || m.g.n_head_kv > 64 ||
            m.g.max_ctx <= 0 || m.g.max_ctx > 0x7fffffff - 16 ||
            m.g.max_tok <= 0 || batch->n_tok > m.g.max_tok ||
            batch->n_seq > 0x7fffffff || !m.kv_full || !batch->kv ||
            batch->n_kv_groups <= 0 || batch->max_q_len > 64 / gqa)
            return 0;
        const RadKVGroupBatch* k = nullptr;
        for (int i = 0; i < batch->n_kv_groups; ++i) {
            if (batch->kv[i].group == m.kv_full) { k = batch->kv + i; break; }
        }
        if (!k || k->block_size != 4 || k->block_table_pitch <= 0) return 0;

        /* The actual h256 decode split-KV law, tile 16. Dense attention uses the
         * group's block-table pitch; QSA's q_len=1 launches use the selected-table
         * pitch and each chunk's own row count. Both are known on the host. */
        auto decode_splits = [&](int q_len, int64_t rows, int64_t blocks) -> int {
            if (q_len <= 0 || rows <= 0 || blocks <= 0 ||
                rows > 0x7fffffff / m.g.n_head_kv ||
                blocks > 0x7fffffff / 4) return 0;
            const int units = (int)(rows * m.g.n_head_kv);
            const int tiles = ((int)m.g.max_ctx + 15) / 16;
            int splits = 256 / units;
            if (q_len * gqa > 32) {
                const int64_t depth = blocks * 4 < m.g.max_ctx ? blocks * 4 : m.g.max_ctx;
                splits = (depth > 65536 ? 512 : 128) / units;
                if (splits > 256) splits = 256;
                if (depth > 16384) {
                    const int floor = 1024 / units < 64 ? 1024 / units : 64;
                    if (splits < floor) splits = floor;
                }
            }
            const int cap = tiles / 2;
            if (splits > cap) splits = cap;
            if (splits > tiles) splits = tiles;
            return splits < 1 ? 1 : splits;
        };

        uint64_t h = 0x9e3779b97f4a7c15ull;
        auto mix = [&](uint64_t v) {
            h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
            h *= 0xbf58476d1ce4e5b9ull;
        };
        const RadKVStream* st = k->stream;
        const bool ring_addressed = st && st->block_bytes > 0 && st->slots > 0 &&
                                     st->host_layer_stride / st->block_bytes > st->slots;
        const int64_t reach = (int64_t)batch->max_ctx_len + batch->max_q_len;
        auto attention_key = [&](const AttnGatedFP8& a) -> bool {
            if (a.kv_block != 4 || !a.op_attn) return false;
            const bool has_qsa = a.qsa_sel && a.qsa_sequ;
            const bool sparse = has_qsa &&
                (batch->n_tok == batch->n_seq || a.qsa_exact_to <= 0 ||
                 reach > a.qsa_exact_to);
            /* A ring can force sparse attention below the QSA exactness bound.
             * If that choice depends on the runtime stage view, refuse capture. */
            if (ring_addressed && !sparse) return false;
            mix((uint64_t)a.op_attn);
            mix((uint64_t)a.op_attn_gq);
            mix((uint64_t)sparse);
            mix((uint64_t)a.qsa_topk);  // Selected-table pitch is geometry too.
            if (sparse) {
                const int64_t chunk = a.op_attn_gq ? a.qsa_gq_rows() : a.qsa_rows();
                if (chunk <= 0 || a.qsa_topk <= 0) return false;
                for (int64_t off = 0; off < batch->n_tok; off += chunk) {
                    const int64_t rows = batch->n_tok - off < chunk ?
                                          batch->n_tok - off : chunk;
                    const int splits = decode_splits(1, rows, a.qsa_topk + 1);
                    if (splits <= 0) return false;
                    mix((uint64_t)rows);
                    mix((uint64_t)splits);
                }
            } else {
                const int splits = decode_splits(batch->max_q_len, batch->n_seq,
                                                   k->block_table_pitch);
                if (splits <= 0) return false;
                mix((uint64_t)splits);
            }
            return true;
        };
        bool have_full = false;
        for (const Layer& l : m.layers) {
            if (!l.full) continue;
            have_full = true;
            if (!attention_key(l.attn)) return 0;
        }
        if (!have_full) return 0;
        if (batch->draft_pass != 0 &&
            (!m.have_mtp || !attention_key(m.mtp.attn))) return 0;

        /* Core also keys n_out, n_draft_out, all operand pointers, stream view,
         * context/pitch and layer/arena epochs. This fingerprint adds every
         * attention launch's decode split count and the MTP/verify geometry. */
        mix((uint64_t)(int64_t)batch->draft_pass);
        mix((uint64_t)batch->n_spec);
        mix((uint64_t)batch->n_tok);
        mix((uint64_t)batch->n_seq);
        mix((uint64_t)batch->max_q_len);
        mix((uint64_t)batch->max_q_len_decode);
        mix((uint64_t)batch->max_ctx_len);
        mix((uint64_t)k->block_table_pitch);
        mix((uint64_t)gqa);
        mix((uint64_t)m.g.n_head_kv);
        mix((uint64_t)m.g.max_ctx);
        mix((uint64_t)k->block_size);
        return h | 1;
    }
    if (batch->phase != RAD_PHASE_PREFILL) return 1;"""

PATCH = FilePatch(
    path="arch/qwen4exp_fp8/qwen4exp_fp8.cpp",
    language="none",  # Exact full hunk, including its guard comments, is intentional.
    description="Replay only provable fixed-geometry MTP decode draft/verify passes.",
    edits=(
        Edit(
            id="mtp-decode-recording-geometry",
            anchor=re.escape(OLD),
            mode="replace",
            text=NEW,
            guard=r"BC_MTP_DECODE_REPLAY_GEOMETRY",
            expect_matches=1,
            max_span_lines=8,
            rationale="long-anchor: exactly the v1.17.0 seven-line pass_key rejection guard, matched whole so any upstream drift in it refuses to apply (the anti-regression contract)",
        ),
    ),
)
