"""rad0019 (VTP14 N2, fp8): --tp-ownership qwen38-27b-6-6-12 for the fp8 Qwen3.8 plugin (qwen35_fp8).

rad0018 adds the profile, the Geom fields and the BF16 plugin. The fp8 plugin serves the block-fp8, w8a8 and other
quantised 27B containers, the ones that fit two RX 7900 XTX, so the matched TP2-vs-TP3 comparison (VTP14 N0/N7)
needs it. Its blocks already take per-rank spans: attention spans q|gate, k/v and the output by head
(rad_block_attn_gated_fp8.h), the delta net takes uneven heads through Config::k_first / v_first
(rad_block_gdn_fp8.h), and LinearFP8 takes a row or column span. Every span is whole 128-row scale blocks here
(Q 3072/3072/6144 rows of q|gate, KV 256/256/512, FFN 5120/4864/7424, GDN K 512/512/1024, V 1536/1536/3072).

- qwen35_fp8: opts in (Geom::tp_owned_ok); delta-net heads and first heads from the profile; explicit vocabulary
  spans for token_embd and output. Refused under a profile: a vocabulary-weighted head (output.k<B>, rows
  whole 256-row classes of an even split), and a speculative drafter (its geometry is not owned).
- rad_block_attn_gated_fp8.h / rad_block_mlp_fp8.h: the spans under Geom::tp_owned only. The blocks are shared by
  eight plugins; only one that opted in can set tp_owned.

Source-pinned to radiance v1.17.0 (19d4e8fe), applied after rad0018.
"""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch


def _edit(id: str, before: str, after: str, *, guard: str, max_span: int | None = None) -> Edit:
    return Edit(
        id=id,
        anchor=re.escape(before),
        mode="replace",
        text=after,
        guard=re.escape(guard),
        expect_matches=1,
        max_span_lines=max_span or max(6, before.count("\n") + 1),
        rationale="VTP14 N2: asymmetric tensor-parallel ownership, fp8 plugin",
    )


QWEN = FilePatch(path="arch/qwen35_fp8/qwen35_fp8.cpp", language="none", description="VTP14 N2 fp8 Qwen opt-in", edits=(
    _edit(
        "n2f-optin",
        "    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, \"bf16\", RAD_BF16));\n",
        "    /* BC_VTP14_N2F_OPTIN: this plugin declares --tp-ownership spans (rad0019). */\n"
        "    m.g.tp_owned_ok = true;\n"
        "    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, \"bf16\", RAD_BF16));\n",
        guard="BC_VTP14_N2F_OPTIN",
        max_span=2,
    ),
    _edit(
        "n2f-gdn-heads",
        "    int s;\n"
        "    if ((s = divide_or_fail(n_head_k_all, g.world, \"linear_num_key_heads\",   &gc.n_head_k)) < 0)\n"
        "        return s;\n"
        "    if ((s = divide_or_fail(n_head_v_all, g.world, \"linear_num_value_heads\", &gc.n_head_v)) < 0)\n"
        "        return s;",
        "    int s;\n"
        "    if (g.tp_owned) { /* BC_VTP14_N2F_GDN_HEADS: the delta net's uneven split (Config::k_first) */\n"
        "        if (n_head_k_all != g.own_gdn_k_all || n_head_v_all != g.own_gdn_v_all) {\n"
        "            fprintf(stderr, \"radiance: qwen35: --tp-ownership expects %lld/%lld delta-net key/value \"\n"
        "                            \"heads, the container has %lld/%lld\\n\", (long long)g.own_gdn_k_all,\n"
        "                    (long long)g.own_gdn_v_all, (long long)n_head_k_all, (long long)n_head_v_all);\n"
        "            return RAD_E_SHAPE;\n"
        "        }\n"
        "        gc.n_head_k     = g.own_gdn_k;\n"
        "        gc.n_head_v     = g.own_gdn_v;\n"
        "        gc.k_first      = g.own_gdn_k_lo;\n"
        "        gc.v_first      = g.own_gdn_v_lo;\n"
        "        gc.n_head_k_all = n_head_k_all;\n"
        "        gc.n_head_v_all = n_head_v_all;\n"
        "    } else {\n"
        "        if ((s = divide_or_fail(n_head_k_all, g.world, \"linear_num_key_heads\",   &gc.n_head_k)) < 0)\n"
        "            return s;\n"
        "        if ((s = divide_or_fail(n_head_v_all, g.world, \"linear_num_value_heads\", &gc.n_head_v)) < 0)\n"
        "            return s;\n"
        "    }",
        guard="BC_VTP14_N2F_GDN_HEADS",
    ),
    _edit(
        "n2f-vocab",
        "    m.op_embed = rw(b, RAD_OP(b, \"embed_lookup\",",
        "    /* BC_VTP14_N2F_VOCAB: the vocabulary does not divide by three; this rank's rows are\n"
        "     * [vocab_off, vocab_off + n_vocab), whole 128-row blocks (vocab_per_rank). */\n"
        "    if (g.tp_owned) {\n"
        "        if (m.n_vcls || !m.w_lm_head) {\n"
        "            fprintf(stderr, \"radiance: qwen35: --tp-ownership reads a plain output.weight, not a \"\n"
        "                            \"vocabulary-weighted head\\n\");\n"
        "            return RAD_E_UNSUPPORTED;\n"
        "        }\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_tok, g.vocab_off, g.vocab_off + g.n_vocab));\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_lm_head, g.vocab_off, g.vocab_off + g.n_vocab));\n"
        "    }\n"
        "\n"
        "    m.op_embed = rw(b, RAD_OP(b, \"embed_lookup\",",
        guard="BC_VTP14_N2F_VOCAB",
        max_span=1,
    ),
))

ATTN = FilePatch(path="arch/common/rad_block_attn_gated_fp8.h", language="none", description="VTP14 N2 fp8 attention spans", edits=(
    _edit(
        "n2f-attn-spans",
        "    const bool    a_spanned  = g.attn_world != g.world;\n"
        "    const bool    kv_spanned = g.world > g.n_head_kv_all || a_spanned;\n"
        "    const int64_t kv_span_lo = kv_spanned ? g.kv_off : 0;\n"
        "    const int64_t kv_span_hi = kv_spanned ? g.kv_off + kv_dim : 0;\n"
        "    const int64_t qg_span_lo = a_spanned ? g.attn_rank * qg_dim : 0;\n"
        "    const int64_t qg_span_hi = a_spanned ? (g.attn_rank + 1) * qg_dim : 0;\n"
        "    const int64_t o_span_lo  = a_spanned ? g.attn_rank * q_dim : 0;\n"
        "    const int64_t o_span_hi  = a_spanned ? (g.attn_rank + 1) * q_dim : 0;\n",
        "    /* BC_VTP14_N2F_ATTN_SPANS: under --tp-ownership (Geom::tp_owned) every rank spans its own query\n"
        "     * heads [own_q_lo, +n_head) and KV heads [kv_off, +kv_dim). */\n"
        "    const bool    a_spanned  = g.attn_world != g.world || g.tp_owned;\n"
        "    const bool    kv_spanned = g.world > g.n_head_kv_all || a_spanned;\n"
        "    const int64_t q_first    = g.tp_owned ? g.own_q_lo : g.attn_rank * g.n_head;\n"
        "    const int64_t kv_span_lo = kv_spanned ? g.kv_off : 0;\n"
        "    const int64_t kv_span_hi = kv_spanned ? g.kv_off + kv_dim : 0;\n"
        "    const int64_t qg_span_lo = a_spanned ? q_first * 2 * g.head_dim : 0;\n"
        "    const int64_t qg_span_hi = a_spanned ? q_first * 2 * g.head_dim + qg_dim : 0;\n"
        "    const int64_t o_span_lo  = a_spanned ? q_first * g.head_dim : 0;\n"
        "    const int64_t o_span_hi  = a_spanned ? q_first * g.head_dim + q_dim : 0;\n",
        guard="BC_VTP14_N2F_ATTN_SPANS",
    ),
))

MLP = FilePatch(path="arch/common/rad_block_mlp_fp8.h", language="none", description="VTP14 N2 fp8 feed-forward spans", edits=(
    _edit(
        "n2f-mlp-gate-up",
        "                                 { src.gate, src.up }, w.h, w.gate_up, 0, { g.n_ff, g.n_ff }));\n",
        "                                 { src.gate, src.up }, w.h, w.gate_up, 0, { g.n_ff, g.n_ff },\n"
        "                                 /* BC_VTP14_N2F_MLP_GATE_UP: each of gate and up spans alike */\n"
        "                                 g.tp_owned ? g.own_ff_lo : 0, g.tp_owned ? g.own_ff_lo + g.n_ff : 0));\n",
        guard="BC_VTP14_N2F_MLP_GATE_UP",
        max_span=2,
    ),
    _edit(
        "n2f-mlp-down",
        "                              g.n_embd, g.n_ff, RAD_SHARD_COL, grp_layer(l),\n"
        "                              { src.down }, w.ffn, w.h.x));\n",
        "                              g.n_embd, g.n_ff, RAD_SHARD_COL, grp_layer(l),\n"
        "                              { src.down }, w.ffn, w.h.x, 0, {},\n"
        "                              /* BC_VTP14_N2F_MLP_DOWN */\n"
        "                              g.tp_owned ? g.own_ff_lo : 0, g.tp_owned ? g.own_ff_lo + g.n_ff : 0));\n",
        guard="BC_VTP14_N2F_MLP_DOWN",
        max_span=3,
    ),
))

LINEAR = FilePatch(path="arch/common/rad_fp8.h", language="none", description="VTP14 N2 span_parts covers MXFP4 wref", edits=(
    _edit(
        "n2f-dense-span",
        "    /* Rows [lo, hi) of the stored block-fp8 weight `src_wn` (codes and scale plane; mapped by\n",
        "    /* BC_VTP14_N2F_DENSE_SPAN: this rank's uneven slice [lo, hi) of the split dimension (rows of each row\n"
        "     * part for RAD_SHARD_ROW, columns for RAD_SHARD_COL), applied to every plane: block fp8's scale\n"
        "     * blocks are 128 each way; MXFP4's scale and reference planes keep the rows and take K in 32s.\n"
        "     * Call after declare(). */\n"
        "    int span(RadBuilder* b, int shard, int64_t lo, int64_t hi) const {\n"
        "        const bool rows = shard == RAD_SHARD_ROW;\n"
        "        const int64_t div = (mx || m2x) ? (rows ? 1 : RAD_MX_GROUP) : RAD_FP8_BLOCK;\n"
        "        if (lo < 0 || hi <= lo || lo % div || hi % div || ((mx || m2x) && rows && (lo % 16 || hi % 16))) {\n"
        "            fprintf(stderr, \"radiance: a span [%lld, %lld) cuts this weight's scale blocks\\n\",\n"
        "                    (long long)lo, (long long)hi);\n"
        "            return RAD_E_INVAL;\n"
        "        }\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span(b, w, lo, hi));\n"
        "        if (s) RAD_ARCH_TRY(rad_weight_shard_span(b, s, lo / div, hi / div));\n"
        "        if (ref) RAD_ARCH_TRY(rad_weight_shard_span(b, ref, lo / div, hi / div));\n"
        "        return RAD_OK;\n"
        "    }\n"
        "    /* Rows [lo, hi) of the stored block-fp8 weight `src_wn` (codes and scale plane; mapped by\n",
        guard="BC_VTP14_N2F_DENSE_SPAN",
        max_span=2,
    ),
    _edit(
        "n2f-span-parts-wref",
        "        RAD_ARCH_TRY(rad_weight_shard_span_parts(b, w, lo, hi, total, n));\n"
        "        if (!ws) return RAD_OK;\n",
        "        RAD_ARCH_TRY(rad_weight_shard_span_parts(b, w, lo, hi, total, n));\n"
        "        /* BC_VTP14_N2F_SPAN_PARTS_WREF: an MXFP4 linear's reference scale plane has the weight's rows\n"
        "         * (as declare()'s single span already treats it); without its span an uneven head split\n"
        "         * (the delta net's k_first) leaves it on the even shard and the load refuses it. */\n"
        "        if (wref) RAD_ARCH_TRY(rad_weight_shard_span_parts(b, wref, lo, hi, total, n));\n"
        "        if (!ws) return RAD_OK;\n",
        guard="BC_VTP14_N2F_SPAN_PARTS_WREF",
        max_span=3,
    ),
))

DRAFT_GEOM = FilePatch(path="arch/common/rad_arch.h", language="none", description="VTP14 N2 drafter ownership", edits=(
    _edit(
        "n2f-vocab-head-sharded",
        "inline bool vocab_head_sharded(const Geom& g) {\n"
        "    return g.world > 1 && g.n_vocab > 0 && (g.n_vocab % 16) == 0 &&\n",
        "inline bool vocab_head_sharded(const Geom& g) {\n"
        "    /* BC_VTP14_N2F_VOCAB_HEAD: under --tp-ownership the vocabulary is the ragged vocab_per_rank split\n"
        "     * (82816/82816/82688 on the 27B). The candidate merge reads (global id, value) pairs, which\n"
        "     * row_topk and logit_rerank form from vocab_off, so a contiguous split in whole 16-row tiles\n"
        "     * shards the draft head as well as an equal one; replicated, it loses the exact rerank. */\n"
        "    if (g.world > 1 && g.tp_owned)\n"
        "        return g.n_vocab > 0 && (g.n_vocab % 16) == 0 && (g.vocab_off % 16) == 0 &&\n"
        "               g.vocab_off == shard_begin(g.n_vocab_all, g.world, g.rank) &&\n"
        "               g.n_vocab == shard_extent(g.n_vocab_all, g.world, g.rank);\n"
        "    return g.world > 1 && g.n_vocab > 0 && (g.n_vocab % 16) == 0 &&\n",
        guard="BC_VTP14_N2F_VOCAB_HEAD",
        max_span=3,
    ),
    _edit(
        "n2f-drafter-geom",
        "    RAD_ARCH_TRY(divide_or_fail(dg.n_head_all,    dg.world, \"draft num_attention_heads\", &dg.n_head));\n"
        "    RAD_ARCH_TRY(divide_or_fail(dg.n_head_kv_all, dg.world, \"draft num_key_value_heads\", &dg.n_head_kv));\n"
        "    RAD_ARCH_TRY(divide_or_fail(dg.n_ff_all,      dg.world, \"draft intermediate_size\",   &dg.n_ff));\n",
        "    if (target.tp_owned) {\n"
        "        /* BC_VTP14_N2F_DRAFTER_GEOM: under --tp-ownership the drafter takes the profile's 1:1:2 shares of\n"
        "         * its own heads (DFlash2 on the 27B: 8/8/16 query and 2/2/4 KV heads of 128) and the target's\n"
        "         * feed-forward rows when its width is the target's, so every rank's share of each all-reduce\n"
        "         * stays its own. Anything else is refused rather than split unevenly. */\n"
        "        static constexpr int64_t share[3] = { 1, 1, 2 }, first[3] = { 0, 1, 2 };\n"
        "        if (dg.world != 3 || dg.n_head_all % 4 || dg.n_head_kv_all % 4 || dg.n_ff_all != target.n_ff_all) {\n"
        "            fprintf(stderr, \"radiance: --tp-ownership: the %s drafter (%lld/%lld heads, %lld FFN) does not \"\n"
        "                            \"split 1:1:2 over three ranks like its target (%lld FFN)\\n\", what,\n"
        "                    (long long)dg.n_head_all, (long long)dg.n_head_kv_all, (long long)dg.n_ff_all,\n"
        "                    (long long)target.n_ff_all);\n"
        "            return RAD_E_UNSUPPORTED;\n"
        "        }\n"
        "        const int r = dg.rank;\n"
        "        const int64_t q4 = dg.n_head_all / 4, kv4 = dg.n_head_kv_all / 4;\n"
        "        dg.n_head    = q4 * share[r];\n"
        "        dg.n_head_kv = kv4 * share[r];\n"
        "        dg.own_q_lo  = q4 * first[r];\n"
        "        dg.own_kv_lo = kv4 * first[r];\n"
        "        dg.kv_off    = dg.own_kv_lo * dg.head_dim;\n"
        "        dg.n_ff      = target.n_ff;\n"
        "        dg.own_ff_lo = target.own_ff_lo;\n"
        "    } else {\n"
        "        RAD_ARCH_TRY(divide_or_fail(dg.n_head_all,    dg.world, \"draft num_attention_heads\", &dg.n_head));\n"
        "        RAD_ARCH_TRY(divide_or_fail(dg.n_head_kv_all, dg.world, \"draft num_key_value_heads\", &dg.n_head_kv));\n"
        "        RAD_ARCH_TRY(divide_or_fail(dg.n_ff_all,      dg.world, \"draft intermediate_size\",   &dg.n_ff));\n"
        "    }\n",
        guard="BC_VTP14_N2F_DRAFTER_GEOM",
    ),
))

DFLASH = FilePatch(path="arch/common/rad_block_dflash2.h", language="none", description="VTP14 N2 drafter spans", edits=(
    _edit(
        "n2f-dflash-head-span",
        "    if (!w_dh || !w_dhs) return RAD_E_INVAL;\n",
        "    if (!w_dh || !w_dhs) return RAD_E_INVAL;\n"
        "    /* BC_VTP14_N2F_DFLASH_HEAD_SPAN: a ragged --tp-ownership share is not 1/tp, so the sharded draft head\n"
        "     * reads this rank's rows [vocab_off, +n_vocab) of the stored head and of its scale planes. */\n"
        "    if (dh_shard && g.tp_owned) {\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span(b, w_dh, g.vocab_off, g.vocab_off + g.n_vocab));\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span(b, w_dhs, g.vocab_off, g.vocab_off + g.n_vocab));\n"
        "    }\n",
        guard="BC_VTP14_N2F_DFLASH_HEAD_SPAN",
        max_span=2,
    ),
    _edit(
        "n2f-dflash-spans",
        "                                     { nm.ckpt(\"%s.layers.%d.mlp.down_proj.weight\", ck, i) }));\n",
        "                                     { nm.ckpt(\"%s.layers.%d.mlp.down_proj.weight\", ck, i) }));\n"
        "        /* BC_VTP14_N2F_DFLASH_SPANS: --tp-ownership (drafter_geom set this rank's first heads). */\n"
        "        if (g.tp_owned) {\n"
        "            const int64_t ql = g.own_q_lo * g.head_dim, kl = g.own_kv_lo * g.head_dim;\n"
        "            RAD_ARCH_TRY(w_q[i].span(b, RAD_SHARD_ROW, ql, ql + qd));\n"
        "            RAD_ARCH_TRY(w_k[i].span(b, RAD_SHARD_ROW, kl, kl + kvd));\n"
        "            RAD_ARCH_TRY(w_v[i].span(b, RAD_SHARD_ROW, kl, kl + kvd));\n"
        "            RAD_ARCH_TRY(w_o[i].span(b, RAD_SHARD_COL, ql, ql + qd));\n"
        "            RAD_ARCH_TRY(w_gu[i].span(b, RAD_SHARD_ROW, g.own_ff_lo, g.own_ff_lo + g.n_ff));\n"
        "            RAD_ARCH_TRY(w_dn[i].span(b, RAD_SHARD_COL, g.own_ff_lo, g.own_ff_lo + g.n_ff));\n"
        "        }\n",
        guard="BC_VTP14_N2F_DFLASH_SPANS",
        max_span=2,
    ),
))

PATCHES = (QWEN, ATTN, MLP, LINEAR, DRAFT_GEOM, DFLASH)
