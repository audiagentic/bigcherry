"""VTP30: opt-in source-pinned Qwen3.8-27B asymmetric TP3 ownership in Radiance.

Radiance already owns the model graph, tensors, state, and host all-reduce.
Only declare the three unequal source weight spans using the existing
RadBuilder shard-span ABI. This patch does NOT enable heterogeneous devices
or bypass first-card code selection. Experimental v1.17.0 source only.
"""
import re

from bigcherry.patcher import Edit, FilePatch


def replace(tag, old, new):
    marker = "BC_VTP30_NATIVE_TP3_" + tag.upper().replace("-", "_")
    assert marker in new
    return Edit(
        id="vtp30-" + tag, anchor=re.escape(old), mode="replace", text=new,
        guard=re.escape(marker), expect_matches=1,
        max_span_lines=max(6, old.count("\n") + 1),
        rationale="Source-pinned original-BF16 27B TP3 geometry; refuse pin drift",
    )


PATCHES = (
    FilePatch(
        language="none",  # exact, source-pinned C++ code+literal anchor
        path="arch/common/rad_arch.h",
        description="Opt-in exact 27B three-rank geometry, preserving every existing split.",
        edits=(
            replace(
                "geometry-flag",
                """    bool    attn_zero  = false;
    int64_t n_ff       = 0;""",
                """    bool    attn_zero  = false;
    /* BC_VTP30_NATIVE_TP3_GEOMETRY_FLAG: original Qwen3.8 27B only.
     * This is declaration/weight ownership, NOT mixed-GPU admission. */
    bool    native_tp3_27b = false;
    int64_t n_ff       = 0;""",
            ),
            replace(
                "geometry-signature",
                """inline int geom_from(Geom& g, const RadModelMeta* m, const RadBuildCtx* c,
                     const char* dtype, uint32_t act_dtype, bool arch_splits_ff = false,
                     bool attn_world_ok = false) {""",
                """inline int geom_from(Geom& g, const RadModelMeta* m, const RadBuildCtx* c,
                     const char* dtype, uint32_t act_dtype, bool arch_splits_ff = false,
                     bool attn_world_ok = false, bool native_tp3_qwen = false) {
    /* BC_VTP30_NATIVE_TP3_GEOMETRY_SIGNATURE: only the BF16 Qwen plugin opts in. */""",
            ),
            replace(
                "exact-head-layout",
                """    int s;
    /* The attention world (see Geom::attn_world): the world itself when it is at most the KV head""",
                """    /* BC_VTP30_NATIVE_TP3_EXACT_HEAD_LAYOUT: do not let --tp 3
     * ever silently split four original KV heads into fractional heads.
     * All other deployments keep the exact upstream geometry. The
     * switch affects model DECLARATION only: the core still refuses
     * mixed-architecture devices until rank-local code selection exists. */
    const char* tp3 = native_tp3_qwen ? std::getenv("BC_VTP_NATIVE_TP3") : nullptr;
    if (tp3 && *tp3) {
        if (std::strcmp(tp3, "1") != 0 || g.world != 3 ||
            g.n_layer != 64 || g.n_embd != 5120 ||
            g.head_dim != 256 || g.n_head_all != 24 ||
            g.n_head_kv_all != 4 || g.n_ff_all != 17408 ||
            g.n_vocab_all != 248320 || c->is_draft ||
            g.rank < 0 || g.rank >= 3) {
            fprintf(stderr, "radiance: VTP30 TP3 requires original dense 27B BF16 "
                            "(64L/5120/24Q/4KV/256D/17408FF/248320vocab) "
                            "on exactly three ranks; got world=%d rank=%d\\n",
                    g.world, g.rank);
            return RAD_E_UNSUPPORTED;
        }
        constexpr int64_t qh[3] = {6, 6, 12};
        constexpr int64_t kvh[3] = {1, 1, 2};
        constexpr int64_t ff[3] = {5120, 4864, 7424};
        constexpr int64_t kv_begin[3] = {0, 1, 2};
        g.n_head = qh[g.rank];
        g.n_head_kv = kvh[g.rank];
        g.kv_off = kv_begin[g.rank] * g.head_dim;
        g.n_ff = ff[g.rank];
        g.attn_world = g.world;
        g.attn_rank = g.rank;
        g.attn_zero = false;
        g.n_vocab = shard_extent(g.n_vocab_all, g.world, g.rank);
        g.vocab_off = shard_begin(g.n_vocab_all, g.world, g.rank);
        g.native_tp3_27b = true;
        if (g.n_vocab <= 0 || g.n_head != 6 * g.n_head_kv)
            return RAD_E_SHAPE;
        return RAD_OK;
    }

    int s;
    /* The attention world (see Geom::attn_world): the world itself when it is at most the KV head""",
            ),
        ),
    ),
    FilePatch(
        language="none",  # exact, source-pinned C++ code+literal anchor
        path="arch/qwen35_bf16/qwen35_bf16.cpp",
        description="Select exact TP3 geometry only for the existing native BF16 Qwen graph.",
        edits=(
            replace(
                "optin-qwen",
                """    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));""",
                """    /* BC_VTP30_NATIVE_TP3_OPTIN_QWEN: reuse Radiance's entire Qwen
     * graph, scheduler, KV/GDN state, embed/head and collectives. */
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16,
                           false, false, true));""",
            ),
            replace(
                "gdn-heads",
                """    int s;
    if ((s = divide_or_fail(n_head_k_all, g.world, "linear_num_key_heads",   &gc.n_head_k)) < 0)
        return s;
    if ((s = divide_or_fail(n_head_v_all, g.world, "linear_num_value_heads", &gc.n_head_v)) < 0)
        return s;""",
                """    /* BC_VTP30_NATIVE_TP3_GDN_HEADS: 4/4/8 K and 12/12/24 V
     * head ownership follows the original BF16 Qwen checkpoint. */
    int s;
    if (g.native_tp3_27b) {
        if (n_head_k_all != 16 || n_head_v_all != 48) return RAD_E_SHAPE;
        gc.n_head_k = g.rank == 2 ? 8 : 4;
        gc.n_head_v = g.rank == 2 ? 24 : 12;
    } else {
        if ((s = divide_or_fail(n_head_k_all, g.world, "linear_num_key_heads",   &gc.n_head_k)) < 0)
            return s;
        if ((s = divide_or_fail(n_head_v_all, g.world, "linear_num_value_heads", &gc.n_head_v)) < 0)
            return s;
    }""",
            ),
        ),
    ),
    FilePatch(
        language="none",  # exact, source-pinned C++ code+literal anchor
        path="arch/common/rad_block_mlp.h",
        description="Use Radiance's row/column shard-span API for 5120/4864/7424 BF16 FFN.",
        edits=(
            replace(
                "ffn-spans",
                """    w_down    = decl_w(b, nm.f("blk.%d.ffn_down.weight", l),    RAD_BF16, {g.n_embd, g.n_ff},
                       RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));

    op_norm = rw(b, RAD_OP(b, "rmsnorm",""",
                """    w_down    = decl_w(b, nm.f("blk.%d.ffn_down.weight", l),    RAD_BF16, {g.n_embd, g.n_ff},
                       RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));
    /* BC_VTP30_NATIVE_TP3_FFN_SPANS: native RadBuilder weight loader
     * reads each existing source matrix, never a pre-packed VTP file.
     * gate|up are two 17408-row parts, each with this rank's same span. */
    if (g.native_tp3_27b) {
        const int64_t lo = g.rank == 0 ? 0 : (g.rank == 1 ? 5120 : 9984);
        const int64_t hi = lo + g.n_ff;
        RAD_ARCH_TRY(rad_weight_shard_span(b, w_gate_up, lo, hi));
        RAD_ARCH_TRY(rad_weight_shard_span(b, w_down, lo, hi));
    }

    op_norm = rw(b, RAD_OP(b, "rmsnorm",""",
            ),
        ),
    ),
    FilePatch(
        language="none",  # exact, source-pinned C++ code+literal anchor
        path="arch/common/rad_block_attn_gated.h",
        description="Declare per-rank causal 6/6/12-Q,1/1/2-KV weights and their exact spans.",
        edits=(
            replace(
                "attention-spans",
                """    w_o      = decl_w(b, nm.f("blk.%d.attn_output.weight", l), RAD_BF16, {g.n_embd, q_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));

    /* ---- ops, in run order except attn_paged (see there). Each states its read and write sets,""",
                """    w_o      = decl_w(b, nm.f("blk.%d.attn_output.weight", l), RAD_BF16, {g.n_embd, q_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));
    /* BC_VTP30_NATIVE_TP3_ATTENTION_SPANS: 6/6/12 query heads
     * belong to 1/1/2 original KV heads. Each query's output gate
     * is interleaved beside that query (2*256 rows per Q head).
     * The same native KV group already retains causal decode state. */
    if (g.native_tp3_27b) {
        const int64_t qh_lo = g.rank == 0 ? 0 : (g.rank == 1 ? 6 : 12);
        const int64_t kvh_lo = g.rank == 0 ? 0 : (g.rank == 1 ? 1 : 2);
        const int64_t qlo = qh_lo * g.head_dim;
        const int64_t klo = kvh_lo * g.head_dim;
        RAD_ARCH_TRY(rad_weight_shard_span(b, w_qg, 2 * qlo, 2 * (qlo + q_dim)));
        RAD_ARCH_TRY(rad_weight_shard_span(b, w_k, klo, klo + kv_dim));
        RAD_ARCH_TRY(rad_weight_shard_span(b, w_v, klo, klo + kv_dim));
        RAD_ARCH_TRY(rad_weight_shard_span(b, w_o, qlo, qlo + q_dim));
    }

    /* ---- ops, in run order except attn_paged (see there). Each states its read and write sets,""",
            ),
        ),
    ),
    FilePatch(
        language="none",  # exact, source-pinned C++ code+literal anchor
        path="arch/common/rad_block_gdn.h",
        description="Preserve 5-part GDN Q|K|V|A|B and 3-part conv source for uneven native TP3.",
        edits=(
            replace(
                "gdn-spans",
                """    w_out      = decl_w(b, nm.f("blk.%d.ssm_out.weight", l),        RAD_BF16, {g.n_embd, v_dim},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));

    op_norm = rw(b, RAD_OP(b, "rmsnorm",""",
                """    w_out      = decl_w(b, nm.f("blk.%d.ssm_out.weight", l),        RAD_BF16, {g.n_embd, v_dim},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));
    /* BC_VTP30_NATIVE_TP3_GDN_SPANS: the existing Radiance loader
     * already accepts per-part row spans and column spans; retain
     * Q|K|V|A|B ordering and all native conv/recurrent KV pools. */
    if (g.native_tp3_27b) {
        const int64_t khead = g.rank == 0 ? 0 : (g.rank == 1 ? 4 : 8);
        const int64_t vhead = g.rank == 0 ? 0 : (g.rank == 1 ? 12 : 24);
        const int64_t k = khead * cfg.head_k, v = vhead * cfg.head_v;
        const int64_t lo5[5] = {k, k, v, vhead, vhead};
        const int64_t hi5[5] = {k + cfg.k_dim(), k + cfg.k_dim(),
                                v + cfg.v_dim(), vhead + cfg.n_head_v,
                                vhead + cfg.n_head_v};
        const int64_t all5[5] = {16 * cfg.head_k, 16 * cfg.head_k,
                                 48 * cfg.head_v, 48, 48};
        RAD_ARCH_TRY(rad_weight_shard_span_parts(b, w_in, lo5, hi5, all5, 5));
        RAD_ARCH_TRY(rad_weight_shard_span_parts(b, w_conv, lo5, hi5, all5, 3));
        RAD_ARCH_TRY(rad_weight_shard_span(b, w_z, v, v + v_dim));
        RAD_ARCH_TRY(rad_weight_shard_span(b, w_a_log, vhead, vhead + cfg.n_head_v));
        RAD_ARCH_TRY(rad_weight_shard_span(b, w_dt_bias, vhead, vhead + cfg.n_head_v));
        RAD_ARCH_TRY(rad_weight_shard_span(b, w_out, v, v + v_dim));
    }

    op_norm = rw(b, RAD_OP(b, "rmsnorm",""",
            ),
        ),
    ),
)
