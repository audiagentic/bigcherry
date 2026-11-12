"""rad0018 (VTP14 N2): asymmetric tensor-parallel ownership, configured by a split spec rather than a built-in table.

Radiance divides a model evenly over --tp ranks (arch/common/rad_arch.h geom_from). Unequal cards (two RX 7900 XTX and an
R9700) and widths that do not divide by the rank count (the dense Qwen3.8-27B's 4 KV heads and 17408-wide feed-forward
over three ranks) need an uneven, whole-head split. `--tp-ownership SPEC` (the serving manifest's
`scheduler.tp_ownership`) gives it, one count a rank in --rank-devices order:

    kv=1,1,2;ffn=5120,4864,7424;lk=4,4,8;lv=12,12,24      (query heads follow kv; q= may be given)

    kv   KV heads                 q    query heads (default: kv x the model's query group)
    ffn  feed-forward rows        lk   delta-net key heads, lv  delta-net value heads (models with a delta net)

Nothing about a model is built in: geom_from checks the shares against the loaded model's own totals (they must sum
to its heads and rows, every share positive, each rank's query heads whole groups of its own KV heads), and the plugin
checks the delta-net shares against the container's heads and its value:key ratio. The weights are the container's
own, read through radiance's shard-span ABI (rad_weight_shard_span[_parts]); the residual stream, the all-reduces
(M x n_embd on every rank), KV and delta-net state, the loader and the sampler are radiance's. The vocabulary keeps the
even split, given as explicit spans because an uneven row split needs one.

- abi: RadBuildCtx gains a trailing `tp_ownership` (null: even division, as upstream).
- core: `--tp-ownership SPEC` (config, help, explain, config_test value) into the build context. The option only checks
  the spec's form; the model checks happen at build, where the model is known.
- rad_arch.h: Geom carries this rank's first heads and rows and every rank's KV and feed-forward shares (for a drafter
  that follows the target's split). geom_from refuses a spec for an architecture that has not opted in
  (Geom::tp_owned_ok), a malformed spec, shares that do not fit this model or world, and calibration (whose taps
  assume even global widths). bc_own_linear_heads checks a plugin's delta-net heads.
- qwen35_bf16: opts in; the delta-net heads through bc_own_linear_heads, and explicit vocabulary spans for token_embd
  and output.
- rad_block_attn_gated.h / rad_block_gdn.h / rad_block_mlp.h: the spans, under Geom::tp_owned only.

Without --tp-ownership nothing changes. Source-pinned to radiance v1.17.0 (19d4e8fe), applied after rad0016.
"""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

# The config_test inventory value: the 6/6/12 split of the dense Qwen3.8-27B over (XTX, XTX, R9700). Test data only.
EXAMPLE = "kv=1,1,2;ffn=5120,4864,7424;lk=4,4,8;lv=12,12,24"
KEYS = ("q", "kv", "ffn", "lk", "lv")


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
        "    { \"--tp-ownership\",       \"SPEC\",   \"BigCherry VTP14: an uneven tensor-parallel split, one\\n\"\n"
        "                                        \"count a rank: kv=..;ffn=..[;q=..][;lk=..;lv=..]\\n\"\n"
        "                                        \"(KV heads, feed-forward rows, query heads, delta-net\\n\"\n"
        "                                        \"key/value heads), checked against the model at load\" }, /* BC_VTP14_N2_HELP */\n"
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
        "            /* The form only (keys, digits, separators); the shares are checked against the model\n"
        "             * at build (rad_arch.h geom_from), where it is known. */\n"
        "            if (!*v || std::strspn(v, \"abcdefghijklmnopqrstuvwxyz0123456789=,;\") != std::strlen(v)) {\n"
        "                RAD_ERR(\"--tp-ownership: '%s' is not a split spec (kv=..;ffn=..[;q=..][;lk=..;lv=..])\", v);\n"
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
        "        { \"--tp-ownership\", { \"" + EXAMPLE + "\" } }, /* BC_VTP14_N2_CONFIG_TEST */\n",
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
        "     * `tp_owned` says a split is in force, and the own_* fields are this rank's first query head,\n"
        "     * KV head, feed-forward row and delta-net key/value heads, with the delta net's per-rank heads\n"
        "     * (own_gdn_k/v) and totals (own_gdn_k_all/v_all, 0 when the spec gives none) for the plugin to\n"
        "     * check (bc_own_linear_heads). own_kv_split / own_ff_split are every rank's KV heads and rows,\n"
        "     * for a drafter that follows the target's split. */\n"
        "    bool    tp_owned_ok  = false;\n"
        "    bool    tp_owned     = false;\n"
        "    int64_t own_q_lo = 0, own_kv_lo = 0, own_ff_lo = 0;\n"
        "    int64_t own_gdn_k_lo = 0, own_gdn_v_lo = 0, own_gdn_k = 0, own_gdn_v = 0;\n"
        "    int64_t own_gdn_k_all = 0, own_gdn_v_all = 0;\n"
        "    int64_t own_kv_split[MAX_RANKS] = {}, own_ff_split[MAX_RANKS] = {};\n"
        "    int64_t n_ff       = 0;    /* per rank */\n",
        guard="BC_VTP14_N2_GEOM:",
    ),
    _edit(
        "n2-spec-helpers",
        "/* Fills `g` from model metadata and the build context.\n",
        "/* BC_VTP14_N2_SPEC (rad0018): --tp-ownership's split spec, `key=n,n,..;key=..`, one count a rank. The\n"
        " * keys are q, kv, ffn, lk and lv (rad0018 README); any other key, a repeated key, an empty or\n"
        " * non-numeric count, or more than MAX_RANKS counts is refused. `n` is the count read, -1 when the key\n"
        " * is absent. */\n"
        "inline bool bc_own_spec_ok(const char* s) {\n"
        "    static const char* const keys[] = { \"q\", \"kv\", \"ffn\", \"lk\", \"lv\" };\n"
        "    for (const char* p = s; p; ) {\n"
        "        const char* e = std::strchr(p, ';');\n"
        "        const size_t len = e ? (size_t)(e - p) : std::strlen(p);\n"
        "        const char* eq = (const char*)std::memchr(p, '=', len);\n"
        "        if (!eq) return false;\n"
        "        bool known = false;\n"
        "        for (const char* k : keys)\n"
        "            if (std::strlen(k) == (size_t)(eq - p) && std::strncmp(p, k, (size_t)(eq - p)) == 0) known = true;\n"
        "        if (!known) return false;\n"
        "        p = e ? e + 1 : nullptr;\n"
        "    }\n"
        "    return true;\n"
        "}\n"
        "inline bool bc_own_spec_get(const char* s, const char* key, int64_t* out, int* n) {\n"
        "    *n = -1;\n"
        "    const size_t kl = std::strlen(key);\n"
        "    for (const char* p = s; p; ) {\n"
        "        const char* e = std::strchr(p, ';');\n"
        "        const char* end = e ? e : p + std::strlen(p);\n"
        "        if ((size_t)(end - p) > kl && std::strncmp(p, key, kl) == 0 && p[kl] == '=') {\n"
        "            if (*n >= 0) return false;\n"
        "            int cnt = 0;\n"
        "            for (const char* q = p + kl + 1; q < end; ) {\n"
        "                if (cnt == MAX_RANKS || *q < '0' || *q > '9') return false;\n"
        "                char* x = nullptr;\n"
        "                const long long v = std::strtoll(q, &x, 10);\n"
        "                if (x == q || x > end) return false;\n"
        "                out[cnt++] = (int64_t)v;\n"
        "                q = x;\n"
        "                if (q < end && (*q != ',' || ++q == end)) return false;\n"
        "            }\n"
        "            if (cnt == 0) return false;\n"
        "            *n = cnt;\n"
        "        }\n"
        "        p = e ? e + 1 : nullptr;\n"
        "    }\n"
        "    return true;\n"
        "}\n"
        "\n"
        "/* BC_VTP14_N2_LINEAR_HEADS (rad0018): a plugin with a delta net checks its heads against the spec's lk/lv\n"
        " * shares: they must be given, sum to the container's key and value heads, and keep the container's\n"
        " * value:key ratio on every rank (each rank's value heads whole groups of its own key heads). */\n"
        "inline int bc_own_linear_heads(const Geom& g, int64_t k_all, int64_t v_all, const char* who) {\n"
        "    if (g.own_gdn_k_all <= 0 || g.own_gdn_v_all <= 0) {\n"
        "        fprintf(stderr, \"radiance: %s: --tp-ownership: this model has a delta net (%lld key / %lld value \"\n"
        "                        \"heads); give its split as lk=..;lv=..\\n\", who, (long long)k_all, (long long)v_all);\n"
        "        return RAD_E_INVAL;\n"
        "    }\n"
        "    if (k_all != g.own_gdn_k_all || v_all != g.own_gdn_v_all || k_all <= 0 || v_all % k_all ||\n"
        "        g.own_gdn_v != g.own_gdn_k * (v_all / k_all)) {\n"
        "        fprintf(stderr, \"radiance: %s: --tp-ownership: lk/lv sum to %lld/%lld delta-net key/value heads with \"\n"
        "                        \"%lld/%lld on rank %d; the container has %lld/%lld, and each rank's value heads must \"\n"
        "                        \"be %lld a key head\\n\", who, (long long)g.own_gdn_k_all, (long long)g.own_gdn_v_all,\n"
        "                (long long)g.own_gdn_k, (long long)g.own_gdn_v, g.rank, (long long)k_all, (long long)v_all,\n"
        "                (long long)(k_all > 0 ? v_all / k_all : 0));\n"
        "        return RAD_E_SHAPE;\n"
        "    }\n"
        "    return RAD_OK;\n"
        "}\n"
        "\n"
        "/* Fills `g` from model metadata and the build context.\n",
        guard="BC_VTP14_N2_SPEC",
        max_span=2,
    ),
    _edit(
        "n2-geom-owned",
        "    int s;\n"
        "    /* The attention world (see Geom::attn_world): the world itself when it is at most the KV head",
        "    /* BC_VTP14_N2_GEOM_OWNED: --tp-ownership SPEC replaces the even division below. Every share is\n"
        "     * checked against this model's own totals and this world; nothing about a model is assumed. */\n"
        "    if (c->tp_ownership && *c->tp_ownership) {\n"
        "        const char* spec = c->tp_ownership;\n"
        "        if (!g.tp_owned_ok) {\n"
        "            fprintf(stderr, \"radiance: --tp-ownership %s: this architecture declares no ownership \"\n"
        "                            \"spans; it serves only the even split\\n\", spec);\n"
        "            return RAD_E_UNSUPPORTED;\n"
        "        }\n"
        "        if (c->calib_dir && *c->calib_dir) {\n"
        "            fprintf(stderr, \"radiance: --tp-ownership: calibration taps assume the even split's \"\n"
        "                            \"global widths; record calibration without a split\\n\");\n"
        "            return RAD_E_UNSUPPORTED;\n"
        "        }\n"
        "        int64_t q[MAX_RANKS] = {}, kv[MAX_RANKS] = {}, ff[MAX_RANKS] = {}, lk[MAX_RANKS] = {}, lv[MAX_RANKS] = {};\n"
        "        int nq = -1, nkv = -1, nff = -1, nlk = -1, nlv = -1;\n"
        "        const int W = g.world;\n"
        "        const char* why = nullptr;\n"
        "        if (!bc_own_spec_ok(spec) || !bc_own_spec_get(spec, \"q\", q, &nq) ||\n"
        "            !bc_own_spec_get(spec, \"kv\", kv, &nkv) || !bc_own_spec_get(spec, \"ffn\", ff, &nff) ||\n"
        "            !bc_own_spec_get(spec, \"lk\", lk, &nlk) || !bc_own_spec_get(spec, \"lv\", lv, &nlv))\n"
        "            why = \"it is not kv=..;ffn=..[;q=..][;lk=..;lv=..] with whole counts\";\n"
        "        else if (W < 2 || nkv != W || nff != W || (nq >= 0 && nq != W) || (nlk >= 0) != (nlv >= 0) ||\n"
        "                 (nlk >= 0 && (nlk != W || nlv != W)))\n"
        "            why = \"kv and ffn (and q, lk, lv when given) need one count a rank; lk and lv come together\";\n"
        "        else if (g.n_head_kv_all <= 0 || g.n_head_all % g.n_head_kv_all)\n"
        "            why = \"this model's query heads are not whole groups of its KV heads\";\n"
        "        if (!why) {\n"
        "            const int64_t grp = g.n_head_all / g.n_head_kv_all;\n"
        "            int64_t sq = 0, skv = 0, sff = 0;\n"
        "            for (int i = 0; i < W && !why; ++i) {\n"
        "                if (nq < 0) q[i] = kv[i] * grp;\n"
        "                if (q[i] <= 0 || kv[i] <= 0 || ff[i] <= 0 || (nlk >= 0 && (lk[i] <= 0 || lv[i] <= 0)))\n"
        "                    why = \"every share must be positive\";\n"
        "                else if (q[i] != kv[i] * grp)\n"
        "                    why = \"each rank's query heads must be whole groups of its own KV heads\";\n"
        "                sq += q[i]; skv += kv[i]; sff += ff[i];\n"
        "            }\n"
        "            if (!why && (sq != g.n_head_all || skv != g.n_head_kv_all || sff != g.n_ff_all))\n"
        "                why = \"the shares must sum to this model's query heads, KV heads and feed-forward rows\";\n"
        "        }\n"
        "        if (why) {\n"
        "            fprintf(stderr, \"radiance: --tp-ownership '%s': %s (this is world %d; the model has %lld query / \"\n"
        "                            \"%lld KV heads and %lld feed-forward rows)\\n\", spec, why, W,\n"
        "                    (long long)g.n_head_all, (long long)g.n_head_kv_all, (long long)g.n_ff_all);\n"
        "            return RAD_E_INVAL;\n"
        "        }\n"
        "        const int r = g.rank;\n"
        "        const auto first = [r](const int64_t* n) {  /* this rank's first head or row */\n"
        "            int64_t lo = 0;\n"
        "            for (int i = 0; i < r; ++i) lo += n[i];\n"
        "            return lo;\n"
        "        };\n"
        "        g.tp_owned   = true;\n"
        "        g.attn_world = g.world;\n"
        "        g.attn_rank  = r;\n"
        "        g.attn_zero  = false;\n"
        "        g.n_head     = q[r];\n"
        "        g.n_head_kv  = kv[r];\n"
        "        g.kv_off     = first(kv) * g.head_dim;\n"
        "        g.n_ff       = ff[r];\n"
        "        g.own_q_lo   = first(q);\n"
        "        g.own_kv_lo  = first(kv);\n"
        "        g.own_ff_lo  = first(ff);\n"
        "        for (int i = 0; i < W; ++i) {\n"
        "            g.own_kv_split[i] = kv[i];\n"
        "            g.own_ff_split[i] = ff[i];\n"
        "        }\n"
        "        if (nlk >= 0) {\n"
        "            g.own_gdn_k    = lk[r];\n"
        "            g.own_gdn_v    = lv[r];\n"
        "            g.own_gdn_k_lo = first(lk);\n"
        "            g.own_gdn_v_lo = first(lv);\n"
        "            g.own_gdn_k_all = g.own_gdn_v_all = 0;\n"
        "            for (int i = 0; i < W; ++i) {\n"
        "                g.own_gdn_k_all += lk[i];\n"
        "                g.own_gdn_v_all += lv[i];\n"
        "            }\n"
        "        }\n"
        "        g.n_vocab    = shard_extent(g.n_vocab_all, g.world, r);\n"
        "        g.vocab_off  = shard_begin (g.n_vocab_all, g.world, r);\n"
        "        if (g.n_vocab <= 0) return RAD_E_SHAPE;\n"
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
        "        if ((s = bc_own_linear_heads(g, n_head_k_all, n_head_v_all, \"qwen35\")) != RAD_OK) return s;\n"
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
