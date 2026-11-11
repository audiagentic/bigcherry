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
        "        if (ctx->max_spec > 0) {\n"
        "            fprintf(stderr, \"radiance: qwen35: --tp-ownership declares no drafter geometry; run \"\n"
        "                            \"without speculation (--num-speculative-tokens 0)\\n\");\n"
        "            return RAD_E_UNSUPPORTED;\n"
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

PATCHES = (QWEN, ATTN, MLP, LINEAR)
