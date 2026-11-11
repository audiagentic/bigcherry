"""rad0018 (VTP14 N2): an asymmetric tensor-parallel ownership profile for the dense Qwen3.8-27B.

Radiance divides a model evenly over --tp ranks (arch/common/rad_arch.h geom_from). The 27B has 4 KV heads and a
17408-wide feed-forward, which three ranks cannot split evenly, and its three cards are unequal: two RX 7900 XTX
(24 GiB) and an R9700 (32 GiB). `--tp-ownership qwen38-27b-6-6-12` gives the three ranks, in --rank-devices order:

    rank   Q heads   KV heads   GDN K heads   GDN V heads   FFN width   vocabulary rows
    0      6  (0)    1 (0)      4  (0)        12 (0)        5120 (0)    [0, 82816)
    1      6  (6)    1 (1)      4  (4)        12 (12)       4864 (5120) [82816, 165632)
    2      12 (12)   2 (2)      8  (8)        24 (24)       7424 (9984) [165632, 248320)

(first head / first row in brackets). Every rank keeps whole heads and Q:KV = 6, GDN V:K = 3. The weights are the
container's own, read through radiance's shard-span ABI (rad_weight_shard_span[_parts]); the residual stream, the
all-reduces (M x 5120 on every rank), KV and GDN state, the loader and the sampler are radiance's.

- abi: RadBuildCtx gains a trailing `tp_ownership` (null: even division, as upstream).
- core: `--tp-ownership NAME` (config, help, explain, config_test value) into the build context.
- rad_arch.h: Geom carries the profile's per-rank first heads and rows. geom_from refuses a profile for an
  architecture that has not opted in (Geom::tp_owned_ok), an unknown profile, a model other than the original dense
  27B, a world other than 3, and calibration (whose taps assume even global widths).
- qwen35_bf16: opts in; the GDN heads, and explicit vocabulary spans for token_embd and output (248320 is not a
  multiple of 3, and the loader refuses an uneven split without a span).
- rad_block_attn_gated.h / rad_block_gdn.h / rad_block_mlp.h: the spans, under Geom::tp_owned only.

Without --tp-ownership nothing changes. Source-pinned to radiance v1.17.0 (19d4e8fe), applied after rad0016.
The span arithmetic follows the VTP30 experiment (wt-xtx27-prefill rad0016_vtp30_native_tp3_ownership); the profile
is typed configuration instead of an environment variable, and vocabulary spans and the calibration refusal are added.
"""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

PROFILE = "qwen38-27b-6-6-12"


def _edit(id: str, before: str, after: str, *, guard: str, max_span: int | None = None) -> Edit:
    return Edit(
        id=id,
        anchor=re.escape(before),
        mode="replace",
        text=after,
        guard=re.escape(guard),
        expect_matches=1,
        max_span_lines=max_span or max(6, before.count("\n") + 1),
        rationale="VTP14 N2: asymmetric tensor-parallel ownership",
    )


ABI = FilePatch(path="abi/rad_builder.h", language="none", description="VTP14 N2 build context", edits=(
    _edit(
        "n2-ctx-field",
        "    int64_t  gen_max_pixels;\n"
        "    int64_t  gen_max_refs;\n"
        "} RadBuildCtx;",
        "    int64_t  gen_max_pixels;\n"
        "    int64_t  gen_max_refs;\n"
        "\n"
        "    /* ---- BC_VTP14_N2_CTX (BigCherry rad0018) */\n"
        "    /* --tp-ownership NAME: an asymmetric tensor-parallel ownership profile (rad_arch.h geom_from), or\n"
        "     * null for the even division. An architecture that does not opt in refuses a profile.\n"
        "     * Trailing and zero-defaulted. */\n"
        "    const char* tp_ownership;\n"
        "} RadBuildCtx;",
        guard="BC_VTP14_N2_CTX",
    ),
))

RAD_CORE = FilePatch(path="core/rad_core.h", language="none", description="VTP14 N2 config", edits=(
    _edit(
        "n2-config",
        "    int   p2p = RAD_P2P_AUTO;         /* --p2p auto|on|off */\n",
        "    /* BC_VTP14_N2_CONFIG: --tp-ownership, an asymmetric ownership profile; empty is the even split. */\n"
        "    std::string tp_ownership;\n"
        "    int   p2p = RAD_P2P_AUTO;         /* --p2p auto|on|off */\n",
        guard="BC_VTP14_N2_CONFIG",
        max_span=2,
    ),
))

CONFIG = FilePatch(path="core/config.cpp", language="none", description="VTP14 N2 --tp-ownership", edits=(
    _edit(
        "n2-help",
        "    { \"--p2p\",                \"MODE\",   \"how the collectives reach the other cards: auto (default)\\n\"",
        "    { \"--tp-ownership\",       \"NAME\",   \"BigCherry VTP14: an asymmetric tensor-parallel ownership\\n\"\n"
        "                                        \"profile instead of the even split; " + PROFILE + "\\n\"\n"
        "                                        \"(the dense Qwen3.8-27B on --tp 3)\" }, /* BC_VTP14_N2_HELP */\n"
        "    { \"--p2p\",                \"MODE\",   \"how the collectives reach the other cards: auto (default)\\n\"",
        guard="BC_VTP14_N2_HELP",
        max_span=1,
    ),
    _edit(
        "n2-parse",
        "        else if (S(\"--p2p\")) {\n"
        "            if (!(v = need(i))) return RAD_E_INVAL;",
        "        else if (S(\"--tp-ownership\")) { /* BC_VTP14_N2_PARSE */\n"
        "            if (!(v = need(i))) return RAD_E_INVAL;\n"
        "            if (std::strcmp(v, \"" + PROFILE + "\") != 0) {\n"
        "                RAD_ERR(\"--tp-ownership: '%s' is not a profile (" + PROFILE + ")\", v);\n"
        "                return RAD_E_INVAL;\n"
        "            }\n"
        "            c->tp_ownership = v;\n"
        "        }\n"
        "        else if (S(\"--p2p\")) {\n"
        "            if (!(v = need(i))) return RAD_E_INVAL;",
        guard="BC_VTP14_N2_PARSE",
    ),
    _edit(
        "n2-explain",
        "    { \"--p2p\",                [](const Config& c) {",
        "    { \"--tp-ownership\",       [](const Config& c) { /* BC_VTP14_N2_EXPLAIN */\n"
        "          return c.tp_ownership.empty() ? std::string(\"the even split\") : c.tp_ownership; } },\n"
        "    { \"--p2p\",                [](const Config& c) {",
        guard="BC_VTP14_N2_EXPLAIN",
        max_span=1,
    ),
))

CONFIG_TEST = FilePatch(path="tests/config_test.cpp", language="none", description="VTP14 N2 option inventory", edits=(
    _edit(
        "n2-config-test",
        "        { \"--max-source-pixels\", { \"1000000\" } },\n",
        "        { \"--max-source-pixels\", { \"1000000\" } },\n"
        "        { \"--tp-ownership\", { \"" + PROFILE + "\" } }, /* BC_VTP14_N2_CONFIG_TEST */\n",
        guard="BC_VTP14_N2_CONFIG_TEST",
        max_span=2,
    ),
))

ENGINE = FilePatch(path="core/engine_bringup.cpp", language="none", description="VTP14 N2 build context", edits=(
    _edit(
        "n2-build-ctx",
        "    bc.tp_wire_lossy = cfg_.tp_wire_exact ? 0 : 1;\n",
        "    bc.tp_wire_lossy = cfg_.tp_wire_exact ? 0 : 1;\n"
        "    /* BC_VTP14_N2_BUILD_CTX: the ownership profile, or null for the even split. */\n"
        "    bc.tp_ownership = cfg_.tp_ownership.empty() ? nullptr : cfg_.tp_ownership.c_str();\n",
        guard="BC_VTP14_N2_BUILD_CTX",
        max_span=2,
    ),
))

RAD_ARCH = FilePatch(path="arch/common/rad_arch.h", language="none", description="VTP14 N2 geometry", edits=(
    _edit(
        "n2-geom-fields",
        "    bool    attn_zero  = false;\n"
        "    int64_t n_ff       = 0;    /* per rank */\n",
        "    bool    attn_zero  = false;\n"
        "    /* BC_VTP14_N2_GEOM: --tp-ownership. `tp_owned_ok` is the plugin's opt-in, set before geom_from;\n"
        "     * `tp_owned` says a profile is in force, and the own_* fields are this rank's first query head,\n"
        "     * KV head, feed-forward row and delta-net key/value heads, with the delta net's per-rank heads\n"
        "     * (own_gdn_k/v) and model totals (own_gdn_k_all/v_all) for the plugin to apply. */\n"
        "    bool    tp_owned_ok  = false;\n"
        "    bool    tp_owned     = false;\n"
        "    int64_t own_q_lo = 0, own_kv_lo = 0, own_ff_lo = 0;\n"
        "    int64_t own_gdn_k_lo = 0, own_gdn_v_lo = 0, own_gdn_k = 0, own_gdn_v = 0;\n"
        "    int64_t own_gdn_k_all = 0, own_gdn_v_all = 0;\n"
        "    int64_t n_ff       = 0;    /* per rank */\n",
        guard="BC_VTP14_N2_GEOM",
    ),
    _edit(
        "n2-geom-owned",
        "    int s;\n"
        "    /* The attention world (see Geom::attn_world): the world itself when it is at most the KV head",
        "    /* BC_VTP14_N2_GEOM_OWNED: --tp-ownership replaces the even division below. */\n"
        "    if (c->tp_ownership && *c->tp_ownership) {\n"
        "        if (!g.tp_owned_ok) {\n"
        "            fprintf(stderr, \"radiance: --tp-ownership %s: this architecture declares no ownership \"\n"
        "                            \"spans; it serves only the even split\\n\", c->tp_ownership);\n"
        "            return RAD_E_UNSUPPORTED;\n"
        "        }\n"
        "        if (std::strcmp(c->tp_ownership, \"" + PROFILE + "\") != 0) {\n"
        "            fprintf(stderr, \"radiance: --tp-ownership '%s' is not a profile\\n\", c->tp_ownership);\n"
        "            return RAD_E_INVAL;\n"
        "        }\n"
        "        if (g.world != 3 || g.n_layer != 64 || g.n_embd != 5120 || g.head_dim != 256 ||\n"
        "            g.n_head_all != 24 || g.n_head_kv_all != 4 || g.n_ff_all != 17408 ||\n"
        "            g.n_vocab_all != 248320) {\n"
        "            fprintf(stderr, \"radiance: --tp-ownership " + PROFILE + " is the dense Qwen3.8-27B \"\n"
        "                            \"(64 layers, 5120 wide, 24 Q / 4 KV heads of 256, 17408 FFN, 248320 \"\n"
        "                            \"vocabulary) on --tp 3; this is world %d, %lld layers, %lld wide, %lld/%lld \"\n"
        "                            \"heads of %lld, %lld FFN, %lld vocabulary\\n\", g.world,\n"
        "                    (long long)g.n_layer, (long long)g.n_embd, (long long)g.n_head_all,\n"
        "                    (long long)g.n_head_kv_all, (long long)g.head_dim, (long long)g.n_ff_all,\n"
        "                    (long long)g.n_vocab_all);\n"
        "            return RAD_E_UNSUPPORTED;\n"
        "        }\n"
        "        if (c->calib_dir && *c->calib_dir) {\n"
        "            fprintf(stderr, \"radiance: --tp-ownership: calibration taps assume the even split's \"\n"
        "                            \"global widths; record calibration without a profile\\n\");\n"
        "            return RAD_E_UNSUPPORTED;\n"
        "        }\n"
        "        static constexpr int64_t qh[3] = { 6, 6, 12 }, q_lo[3] = { 0, 6, 12 };\n"
        "        static constexpr int64_t kvh[3] = { 1, 1, 2 }, kv_lo[3] = { 0, 1, 2 };\n"
        "        static constexpr int64_t ff[3] = { 5120, 4864, 7424 }, ff_lo[3] = { 0, 5120, 9984 };\n"
        "        static constexpr int64_t gk[3] = { 4, 4, 8 }, gk_lo[3] = { 0, 4, 8 };\n"
        "        static constexpr int64_t gv[3] = { 12, 12, 24 }, gv_lo[3] = { 0, 12, 24 };\n"
        "        const int r = g.rank;\n"
        "        g.tp_owned   = true;\n"
        "        g.attn_world = g.world;\n"
        "        g.attn_rank  = r;\n"
        "        g.attn_zero  = false;\n"
        "        g.n_head     = qh[r];\n"
        "        g.n_head_kv  = kvh[r];\n"
        "        g.kv_off     = kv_lo[r] * g.head_dim;\n"
        "        g.n_ff       = ff[r];\n"
        "        g.own_q_lo   = q_lo[r];\n"
        "        g.own_kv_lo  = kv_lo[r];\n"
        "        g.own_ff_lo  = ff_lo[r];\n"
        "        g.own_gdn_k  = gk[r];\n"
        "        g.own_gdn_v  = gv[r];\n"
        "        g.own_gdn_k_lo = gk_lo[r];\n"
        "        g.own_gdn_v_lo = gv_lo[r];\n"
        "        g.own_gdn_k_all = 16;\n"
        "        g.own_gdn_v_all = 48;\n"
        "        g.n_vocab    = shard_extent(g.n_vocab_all, g.world, r);\n"
        "        g.vocab_off  = shard_begin (g.n_vocab_all, g.world, r);\n"
        "        if (g.n_vocab <= 0 || g.n_head != 6 * g.n_head_kv) return RAD_E_SHAPE;\n"
        "        return RAD_OK;\n"
        "    }\n"
        "\n"
        "    int s;\n"
        "    /* The attention world (see Geom::attn_world): the world itself when it is at most the KV head",
        guard="BC_VTP14_N2_GEOM_OWNED",
    ),
))

QWEN = FilePatch(path="arch/qwen35_bf16/qwen35_bf16.cpp", language="none", description="VTP14 N2 Qwen opt-in", edits=(
    _edit(
        "n2-qwen-optin",
        "    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, \"bf16\", RAD_BF16));\n",
        "    /* BC_VTP14_N2_QWEN_OPTIN: this plugin declares --tp-ownership spans (rad0018). */\n"
        "    m.g.tp_owned_ok = true;\n"
        "    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, \"bf16\", RAD_BF16));\n",
        guard="BC_VTP14_N2_QWEN_OPTIN",
        max_span=2,
    ),
    _edit(
        "n2-qwen-gdn-heads",
        "    int s;\n"
        "    if ((s = divide_or_fail(n_head_k_all, g.world, \"linear_num_key_heads\",   &gc.n_head_k)) < 0)\n"
        "        return s;\n"
        "    if ((s = divide_or_fail(n_head_v_all, g.world, \"linear_num_value_heads\", &gc.n_head_v)) < 0)\n"
        "        return s;",
        "    int s;\n"
        "    if (g.tp_owned) { /* BC_VTP14_N2_QWEN_GDN_HEADS */\n"
        "        if (n_head_k_all != g.own_gdn_k_all || n_head_v_all != g.own_gdn_v_all) {\n"
        "            fprintf(stderr, \"radiance: qwen35: --tp-ownership expects %lld/%lld delta-net key/value \"\n"
        "                            \"heads, the container has %lld/%lld\\n\", (long long)g.own_gdn_k_all,\n"
        "                    (long long)g.own_gdn_v_all, (long long)n_head_k_all, (long long)n_head_v_all);\n"
        "            return RAD_E_SHAPE;\n"
        "        }\n"
        "        gc.n_head_k = g.own_gdn_k;\n"
        "        gc.n_head_v = g.own_gdn_v;\n"
        "    } else {\n"
        "        if ((s = divide_or_fail(n_head_k_all, g.world, \"linear_num_key_heads\",   &gc.n_head_k)) < 0)\n"
        "            return s;\n"
        "        if ((s = divide_or_fail(n_head_v_all, g.world, \"linear_num_value_heads\", &gc.n_head_v)) < 0)\n"
        "            return s;\n"
        "    }",
        guard="BC_VTP14_N2_QWEN_GDN_HEADS",
    ),
    _edit(
        "n2-qwen-vocab",
        "    m.op_embed = rw(b, RAD_OP(b, \"embed_lookup\",",
        "    /* BC_VTP14_N2_QWEN_VOCAB: the vocabulary does not divide by three, and the loader refuses an uneven\n"
        "     * row split without a span: this rank's rows are [vocab_off, vocab_off + n_vocab). */\n"
        "    if (g.tp_owned) {\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_tok, g.vocab_off, g.vocab_off + g.n_vocab));\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_lm_head, g.vocab_off, g.vocab_off + g.n_vocab));\n"
        "    }\n"
        "\n"
        "    m.op_embed = rw(b, RAD_OP(b, \"embed_lookup\",",
        guard="BC_VTP14_N2_QWEN_VOCAB",
        max_span=1,
    ),
))

ATTN = FilePatch(path="arch/common/rad_block_attn_gated.h", language="none", description="VTP14 N2 attention spans", edits=(
    _edit(
        "n2-attn-spans",
        "    w_o      = decl_w(b, nm.f(\"blk.%d.attn_output.weight\", l), RAD_BF16, {g.n_embd, q_dim},\n"
        "                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));\n",
        "    w_o      = decl_w(b, nm.f(\"blk.%d.attn_output.weight\", l), RAD_BF16, {g.n_embd, q_dim},\n"
        "                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));\n"
        "    /* BC_VTP14_N2_ATTN_SPANS: --tp-ownership. Each query head's gate rows sit beside it (2 * head_dim\n"
        "     * rows a head), and each rank's query heads share its own KV heads. */\n"
        "    if (g.tp_owned) {\n"
        "        const int64_t qlo = g.own_q_lo * g.head_dim, klo = g.own_kv_lo * g.head_dim;\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span(b, w_qg, 2 * qlo, 2 * (qlo + q_dim)));\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span(b, w_k, klo, klo + kv_dim));\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span(b, w_v, klo, klo + kv_dim));\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span(b, w_o, qlo, qlo + q_dim));\n"
        "    }\n",
        guard="BC_VTP14_N2_ATTN_SPANS",
        max_span=3,
    ),
))

GDN = FilePatch(path="arch/common/rad_block_gdn.h", language="none", description="VTP14 N2 delta-net spans", edits=(
    _edit(
        "n2-gdn-spans",
        "    w_out      = decl_w(b, nm.f(\"blk.%d.ssm_out.weight\", l),        RAD_BF16, {g.n_embd, v_dim},\n"
        "                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));\n",
        "    w_out      = decl_w(b, nm.f(\"blk.%d.ssm_out.weight\", l),        RAD_BF16, {g.n_embd, v_dim},\n"
        "                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));\n"
        "    /* BC_VTP14_N2_GDN_SPANS: --tp-ownership. ssm_in is q|k|v|a|b (five row parts) and the convolution\n"
        "     * q|k|v (three); this rank's key heads start at own_gdn_k_lo and value heads at own_gdn_v_lo. */\n"
        "    if (g.tp_owned) {\n"
        "        const int64_t k = g.own_gdn_k_lo * cfg.head_k, v = g.own_gdn_v_lo * cfg.head_v;\n"
        "        const int64_t vh = g.own_gdn_v_lo;\n"
        "        const int64_t lo5[5]  = { k, k, v, vh, vh };\n"
        "        const int64_t hi5[5]  = { k + cfg.k_dim(), k + cfg.k_dim(), v + cfg.v_dim(), vh + cfg.n_head_v,\n"
        "                                  vh + cfg.n_head_v };\n"
        "        const int64_t all5[5] = { g.own_gdn_k_all * cfg.head_k, g.own_gdn_k_all * cfg.head_k,\n"
        "                                  g.own_gdn_v_all * cfg.head_v, g.own_gdn_v_all, g.own_gdn_v_all };\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span_parts(b, w_in, lo5, hi5, all5, 5));\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span_parts(b, w_conv, lo5, hi5, all5, 3));\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span(b, w_z, v, v + cfg.v_dim()));\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span(b, w_a_log, vh, vh + cfg.n_head_v));\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span(b, w_dt_bias, vh, vh + cfg.n_head_v));\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span(b, w_out, v, v + cfg.v_dim()));\n"
        "    }\n",
        guard="BC_VTP14_N2_GDN_SPANS",
        max_span=3,
    ),
))

MLP = FilePatch(path="arch/common/rad_block_mlp.h", language="none", description="VTP14 N2 feed-forward spans", edits=(
    _edit(
        "n2-mlp-spans",
        "    w_down    = decl_w(b, nm.f(\"blk.%d.ffn_down.weight\", l),    RAD_BF16, {g.n_embd, g.n_ff},\n"
        "                       RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));\n",
        "    w_down    = decl_w(b, nm.f(\"blk.%d.ffn_down.weight\", l),    RAD_BF16, {g.n_embd, g.n_ff},\n"
        "                       RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));\n"
        "    /* BC_VTP14_N2_MLP_SPANS: --tp-ownership. gate|up are two equal row parts, each spanned alike. */\n"
        "    if (g.tp_owned) {\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span(b, w_gate_up, g.own_ff_lo, g.own_ff_lo + g.n_ff));\n"
        "        RAD_ARCH_TRY(rad_weight_shard_span(b, w_down, g.own_ff_lo, g.own_ff_lo + g.n_ff));\n"
        "    }\n",
        guard="BC_VTP14_N2_MLP_SPANS",
        max_span=3,
    ),
))

PATCHES = (ABI, RAD_CORE, CONFIG, CONFIG_TEST, ENGINE, RAD_ARCH, QWEN, ATTN, GDN, MLP)
