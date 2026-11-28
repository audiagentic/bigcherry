"""Complete the packed B16 geometry set and keep Qwen35's DFlash2 cache native E4M3."""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch


def _edit(edit_id: str, before: str, text: str, guard: str | None = None, *,
          mode: str = "replace", count: int = 1) -> Edit:
    if count > 1:
        mode = "replace_all"
    return Edit(
        id=edit_id, anchor=re.escape(before), mode=mode, expect_matches=count,
        text=text.replace("\\", "\\\\") if mode == "replace_all" else text,
        guard=(guard[3:] if guard and guard.startswith("re:") else re.escape(guard or text)), max_span_lines=before.count("\n") + 1,
        rationale="Attach the remaining historical B16 and independent DFlash2 geometry",
    )


_PAGE16_WRAPPERS = r"""/* Resident B16, Qwen35's H256/GQA6/page16 dense attention. */
template<RadKVLayout Layout>
static int b16_page16_launch(const RadArgs* a, RadStream s, bool prefill) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    const RadTensor* cache = r4d_opt(a, AP_KV);
    if (!cache || cache->dtype != RAD_U8) return R4D_NO(RAD_E_DTYPE);
    constexpr int row = 16 * R4dKvBytes<Layout>::row16;
    R4DArgs r;
    const int rc = prefill ? fill_paged_row(a, &r, A_HEAD_DIM, row)
                           : decode_common(a, &r, A_HEAD_DIM, 0, row);
    if (rc != RAD_OK) return rc;
    if (prefill) {
        r.splits = prefill_splits(a, &r);
        return prefill_launch<0, 0, Layout>(&r, (hipStream_t)s);
    }
    return decode_launch<0, 0, Layout>(&r, (hipStream_t)s);
}

#define BC_B16_PAGE16(name, layout) \
extern "C" int r4d_rad_attn_prefill_h256_gqa6_page16_##name(const RadArgs* a, RadStream s) { \
    return b16_page16_launch<layout>(a, s, true); \
} \
extern "C" int64_t r4d_rad_attn_prefill_h256_gqa6_page16_##name##_scratch(const RadArgs* a) { \
    return prefill_scratch(a, 16 * R4dKvBytes<layout>::row16); \
} \
extern "C" int r4d_rad_attn_decode_h256_gqa6_page16_##name(const RadArgs* a, RadStream s) { \
    return b16_page16_launch<layout>(a, s, false); \
} \
extern "C" int64_t r4d_rad_attn_decode_h256_gqa6_page16_##name##_scratch(const RadArgs* a) { \
    return decode_scratch(a, A_HEAD_DIM, 0, 16 * R4dKvBytes<layout>::row16); \
}
BC_B16_PAGE16(b16f8, RAD_KVQ_B16F8)
BC_B16_PAGE16(b16i8, RAD_KVQ_B16I8)
#undef BC_B16_PAGE16

"""

_DRAFTER_WRAPPER = """/* Qwen35 DFlash2: the H128/GQA4 native cache stays E4M3 over a packed trunk. */
extern "C" int r4d_rad_attn_decode_h128_gqa4_fp8(const RadArgs* a, RadStream s) {
    const RadTensor* cache = r4d_opt(a, AP_KV);
    if (!cache || cache->dtype != RAD_F8E4M3) return R4D_NO(RAD_E_DTYPE);
    R4DArgs r;
    const int rc = decode_common(a, &r, 128, 1);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h128_gqa4_fp8kv(&r, (hipStream_t)s);
}

"""

_DRAFTER_CTX = """    RadBuildCtx cache_ctx = *ctx;
    if (ctx->kv_layout != RAD_KVQ_NONE) {
        if (dg.head_dim != 128 || dg.gqa() != 4 || dc.block > 16) {
            fprintf(stderr, "radiance: qwen35: DFlash2 over a B16 trunk requires "
                            "H128/GQA4/page16 and at most 16 query rows\\n");
            return RAD_E_UNSUPPORTED;
        }
        cache_ctx.kv_layout = RAD_KVQ_NONE;
        cache_ctx.kv_dtype = RAD_F8E4M3;
    }
    RAD_ARCH_TRY(drafter_kv_group(b, nm, &cache_ctx, dg, dc.window, base_layer, &m.kv_draft));"""

_Q35_GROUP = """    RadKVGroupDecl fd{};
    fd.kind      = RAD_KV_FULL;
    fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    fd.n_head_kv = g.n_head_kv;
    fd.head_dim  = g.head_dim;
    m.kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
    /* The op parameter must use the spelling the rows band on, which is rad_dtype_name's. */
    g.kv_dtype = rad_dtype_name(fd.dtype);"""

_Q35_PACKED = """    if (ctx->kv_layout != RAD_KVQ_NONE &&
        (ctx->kv_dtype != RAD_U8 || g.head_dim != 256 || g.gqa() != 6 ||
         (ctx->kv_layout != RAD_KVQ_B16F8 && ctx->kv_layout != RAD_KVQ_B16I8))) {
        fprintf(stderr, "radiance: qwen35: the packed B16 cache requires H256/GQA6/page16\\n");
        return RAD_E_UNSUPPORTED;
    }
    RadKVGroupDecl fd{};
    fd.kind      = RAD_KV_FULL;
    fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    fd.kv_layout = ctx->kv_layout;
    fd.n_head_kv = g.n_head_kv;
    fd.head_dim  = g.head_dim;
    m.kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
    const RadKVLayoutDesc* packed = rad_kvq_desc(fd.kv_layout);
    g.kv_dtype = packed ? packed->name : rad_dtype_name(fd.dtype);"""


_Q35_PROBE = """    const auto hd = meta->head_dim ? meta->head_dim :
                     (meta->n_head ? meta->n_embd / meta->n_head : 0);
    const auto kv_heads = meta->n_head_kv ? meta->n_head_kv : meta->n_head;
    out->kv_layout_ok = hd == 256 && kv_heads > 0 && meta->n_head % kv_heads == 0 &&
                        meta->n_head / kv_heads == 6
        ? ((1u << RAD_KVQ_B16F8) | (1u << RAD_KVQ_B16I8)) : 0;
"""


def _declarations() -> str:
    lines = []
    for fmt in ("b16f8", "b16i8"):
        for symbol in ("h256_gqa6", "h256_gqa6_page16"):
            for stage in ("prefill", "decode"):
                name = f"r4d_rad_attn_{stage}_{symbol}_{fmt}"
                lines += [f"int     {name}(const RadArgs*, RadStream);",
                          f"int64_t {name}_scratch(const RadArgs*);"]
        for stage in ("decode_gq", "decode_ghq"):
            lines.append(f"int     r4d_rad_attn_{stage}_h256_gqa6_{fmt}(const RadArgs*, RadStream);")
    lines.append("int     r4d_rad_attn_decode_h128_gqa4_fp8(const RadArgs*, RadStream);")
    return "\n".join(lines) + "\n"


def _rows() -> tuple[str, str]:
    """Generate the explicit page4 and page16 tuples plus their matching row bindings."""
    constraints = []
    rows = []
    for fmt, dtype in (("b16f8", "bf16_fp8"), ("b16i8", "bf16_i8")):
        for page, geo, sym in ((4, "h256_gqa6_bs4", "h256_gqa6"), (16, "h256_gqa6", "h256_gqa6_page16")):
            for stage in ("prefill", "decode", *(("decode_gq", "decode_ghq") if page == 4 else ())):
                name = f"attn_{stage}_{geo}_{fmt}"
                check = f"c_{name}"
                gate = stage in ("decode_gq", "decode_ghq")
                rotated = stage == "decode_ghq"
                prefill = stage == "prefill"
                op = "attn_paged_gate_had_quant" if rotated else "attn_paged_gate_quant" if gate else "attn_paged"
                constraints.append(f"""static const RadConstraint {check}[] = {{
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", {page}),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "{dtype}"),
""")
                constraints.append('    C_EQ("causal", 1),\n' if prefill else
                                   '    C_GE("causal", 0), C_LE("q_len", 10),\n')
                if gate:
                    constraints.append('    C_EQ("group", 128), C_IN("act", "sigmoid"),\n')
                constraints.append("};\n")
                tune = "NOTUNE" if prefill else "ROW(tAttnSplit), nullptr"
                priority = "R4D_PRIO_ATTN_PREFILL" if prefill else "R4D_PRIO_ATTN_DECODE"
                scratch = "prefill" if prefill else "decode"
                shape = "nullptr" if rotated else f"r4d_shape_{op}"
                extra = ", nullptr, nullptr, RAD_KERNEL_CAP_DYNAMIC_PREFILL_SPLIT" if prefill else (
                    f", nullptr, r4d_fuse_{op}" if gate else "")
                rows.append(f"""{{ "{name}", "{op}", "attn",
  "Resident packed KV {stage} attention", "H256/GQA6/page{page}", "{dtype}",
  RAD_DOMAIN_DEVICE, {priority}, ROW({check}), {tune},
  nullptr, nullptr, r4d_rad_attn_{stage}_{sym}_{fmt},
  r4d_rad_attn_{scratch}_{sym}_{fmt}_scratch, nullptr, nullptr, {shape}{extra} }},

""")
        for fused in (False, True):
            name = f"{'qk_norm_rope_gate_' if fused else ''}kv_store_page16_{fmt}"
            op = "qk_norm_rope_gate_kv_store" if fused else "kv_store"
            check = f"c_{name}"
            q_dtype = ' C_IN("q_dtype", "bf16"),' if fused else ""
            constraints.append(f"""static const RadConstraint {check}[] = {{
    C_EQ("head_dim", 256), C_EQ("block_size", 16), C_IN("kv_dtype", "{dtype}"),{q_dtype}
}};
""")
            launch = "r4d_rad_qk_norm_rope_gate_kv_store" if fused else f"r4d_rad_kv_store_{fmt}"
            extra = ", nullptr, r4d_fuse_qk_norm_rope_gate_kv_store" if fused else ""
            rows.append(f"""{{ "{name}", "{op}", "attn",
  "Resident packed KV store", "H256/page16", "{dtype}",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW({check}), NOTUNE,
  nullptr, nullptr, {launch}, nullptr, nullptr, nullptr, r4d_shape_{op}{extra} }},

""")
    return "".join(constraints), "".join(rows)


_GEOMETRY_CONSTRAINTS, _GEOMETRY_ROWS = _rows()
_DRAFTER_CONSTRAINTS = """static const RadConstraint cDflashH128Gqa4Fp8[] = {
    C_GE("window", 0), C_EQ("head_dim", 128), C_EQ("gqa", 4), C_EQ("block_size", 16),
    C_GE("causal", 0), C_LE("q_len", 16), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3"),
};
"""
_DRAFTER_ROW = """{ "attn_decode_h128_gqa4_fp8kv", "attn_paged", "attn",
  "Native E4M3 windowed split-KV for the Qwen35 DFlash2 drafter",
  "H128/GQA4/page16, query length <= 16, window optional", "bf16 query, E4M3 K and V",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE, ROW(cDflashH128Gqa4Fp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h128_gqa4_fp8, r4d_rad_attn_decode_h128_scratch,
  nullptr, nullptr, r4d_shape_attn_paged },

"""

PATCHES = (
    FilePatch(path="libr4d/r4d_attn_paged_gqa8.hip", language="none", edits=(
        _edit("b16-gqa6-page4", "#if G8_BLOCK_SIZE == 4 && R4D_ATTN_GQA == 12\nG8_BYTE_ENTRIES(b16f8,",
              "#if G8_BLOCK_SIZE == 4 && (R4D_ATTN_GQA == 12 || R4D_ATTN_GQA == 6)\nG8_BYTE_ENTRIES(b16f8,"),
    )),
    FilePatch(path="libr4d/r4d_attn_paged_h256_gqa6.hip", language="none", edits=(
        # Only the two supported chain/prefill entries, not the upstream tree launcher.
        _edit("b16-page16-layout-template", "template <int KVP, int XOPT = 0>\nstatic int prefill_launch(",
              "template <int KVP, int XOPT = 0, RadKVLayout Layout = RAD_KVQ_NONE>\nstatic int prefill_launch("),
        _edit("b16-page16-decode-layout-template", "template <int KVP, int XOPT = 0>\nstatic int decode_launch(",
              "template <int KVP, int XOPT = 0, RadKVLayout Layout = RAD_KVQ_NONE>\nstatic int decode_launch("),
        _edit("b16-page16-prefill-split", "KVP, P_OPT | XOPT | O_PSPLIT>",
              "KVP, P_OPT | XOPT | O_PSPLIT, 0, Layout>"),
        _edit("b16-page16-prefill", "KVP, P_OPT | XOPT>", "KVP, P_OPT | XOPT, 0, Layout>"),
        _edit("b16-page16-decode", "A_GQA, A_BLOCK_SIZE, KVP, D_OPT | XOPT>",
              "A_GQA, A_BLOCK_SIZE, KVP, D_OPT | XOPT, 0, Layout>"),
        _edit("b16-page16-wrappers", "/* The i8 cache: a token's row", _PAGE16_WRAPPERS,
              "BC_B16_PAGE16(b16f8,", mode="insert_before"),
        _edit("b16-drafter-native-plugin", "/* The i8 cache at this geometry:", _DRAFTER_WRAPPER,
              "int r4d_rad_attn_decode_h128_gqa4_fp8(", mode="insert_before"),
    )),
    FilePatch(path="libr4d/r4d_attn_paged_h128_gqa4.hip", language="none", edits=(
        _edit("b16-drafter-native-window", "// The int8 cache (int8 K and V",
              """extern "C" int r4d_attn_decode_h128_gqa4_fp8kv(const R4DArgs* a, hipStream_t s) {
    return r4d_decode_launch_win<B_HEAD_DIM, B_GQA, B_OPT, 0>(a, s);
}
""", "int r4d_attn_decode_h128_gqa4_fp8kv(", mode="insert_before"),
    )),
    FilePatch(path="libr4d/r4d.h", language="none", edits=(
        _edit("b16-drafter-native-declaration", "int  r4d_attn_decode_h128_gqa4_i8kv",
              "int  r4d_attn_decode_h128_gqa4_fp8kv (const R4DArgs* a, hipStream_t stream);\n",
              "int  r4d_attn_decode_h128_gqa4_fp8kv", mode="insert_before"),
    )),
    FilePatch(path="libr4d/r4d_plugin.h", language="none", edits=(
        _edit("b16-remaining-symbol-declarations", "int     r4d_rad_attn_prefill_h256_gqa12_b16f8(",
              _declarations(), "r4d_rad_attn_prefill_h256_gqa6_page16_b16i8(", mode="insert_before"),
    )),
    FilePatch(path="libr4d/bc_kv_rows.h", language="none", edits=(
        _edit("b16-geometry-constraints", "#define BC_B16_QSA_BASE(",
              _GEOMETRY_CONSTRAINTS + _DRAFTER_CONSTRAINTS, "cDflashH128Gqa4Fp8[]", mode="insert_before"),
        _edit("b16-geometry-rows", "#define BC_B16_QSA_ROWS(",
              _GEOMETRY_ROWS + _DRAFTER_ROW, '{ "attn_decode_h128_gqa4_fp8kv",', mode="insert_before"),
        _edit("bc-kv-shared-chain-rad0052", "/* BC_KV_CHAIN_APPEND */",
              '        "qk_norm_rope_gate_kv_store_page16_b16f8",\n        "qk_norm_rope_gate_kv_store_page16_b16i8",\n        "attn_decode_h256_gqa6_bs4_b16f8",\n        "attn_decode_h256_gqa6_bs4_b16i8",\n        "attn_decode_h256_gqa6_b16f8",\n        "attn_decode_h256_gqa6_b16i8",\n        "attn_decode_h128_gqa4_fp8kv",\n/* BC_KV_CHAIN_rad0052 */\n', '/* BC_KV_CHAIN_rad0052 */', mode="insert_before"),
    )),
    FilePatch(path="arch/qwen4exp_fp8/qwen4exp_fp8.cpp", language="none", edits=(
        _edit("b16-q4-gqa6-admit", "ctx->kv_dtype != RAD_U8 || g.head_dim != 256 || g.gqa() != 12 ||",
              "ctx->kv_dtype != RAD_U8 || g.head_dim != 256 || (g.gqa() != 12 && g.gqa() != 6) ||"),
        _edit("b16-q4-gqa6-error", "the packed B16 cache requires H256/GQA12/page4",
              "the packed B16 cache requires H256, GQA6 or GQA12, and page4", guard="re:the packed (?:B16|KV) cache requires H256, GQA6 or GQA12, and page4"),
        _edit("b16-q4-gqa6-probe",
              "    out->kv_layout_ok = hd == 256 && kv_heads > 0 && meta->n_head % kv_heads == 0 &&\n"
              "                        meta->n_head / kv_heads == 12",
              "    out->kv_layout_ok = native_e4m3_geometry(hd, meta->n_head, kv_heads)"),
    )),
    FilePatch(path="arch/qwen35_fp8/qwen35_fp8.cpp", language="none", edits=(
        _edit("b16-qwen35-trunk", _Q35_GROUP, _Q35_PACKED, guard="re:the packed (?:B16|KV) cache requires H256/GQA6/page16"),
        _edit("b16-qwen35-native-draft-context",
              "    RAD_ARCH_TRY(drafter_kv_group(b, nm, ctx, dg, dc.window, base_layer, &m.kv_draft));",
              _DRAFTER_CTX, "cache_ctx.kv_dtype = RAD_F8E4M3;"),
        _edit("b16-qwen35-probe",
              "    /* The delta-net state cache is declared at RadBuildCtx::state_dtype. */",
              _Q35_PROBE, "out->kv_layout_ok = hd == 256", mode="insert_before"),
    )),
)

_Q4_TP4_FIXTURE = r"""TEST(qwen4exp_b16_tp4_selects_gqa6_page4) {
    const RadModelMeta meta = qwen4exp_meta();
    RadModelMeta gqa6 = meta;
    gqa6.n_head_kv = 2 * meta.n_head_kv;
    RadArchProbe probe{};
    REQUIRE_EQ(qwen4exp_fp8::probe(&gqa6, &probe), RAD_OK);
    CHECK_EQ(probe.kv_layout_ok, (1u << RAD_KVQ_B16F8) | (1u << RAD_KVQ_B16I8));
    for (const auto layout : {RAD_KVQ_B16F8, RAD_KVQ_B16I8}) {
        RadBuildCtx ctx = build_ctx(0, 4);
        ctx.max_tok = 64; ctx.max_seqs = 2; ctx.max_spec = 3; ctx.max_ctx = 4096;
        ctx.kv_dtype = RAD_U8;
        ctx.kv_layout = layout;
        RadBuilder b;
        b.kv_block = 4;
        REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
        const auto& model = qwen4exp_fp8::g_model[0];
        REQUIRE(model.kv_full > 0);
        CHECK_EQ(b.kv_groups[model.kv_full - 1].d.n_head_kv, 1);
        CHECK_EQ(b.kv_groups[model.kv_full - 1].d.kv_layout, (RadKVLayout)layout);
        int count = 0;
        for (const auto& op : b.ops)
            if (op.op == "attn_paged" && pstr(&op, "kv_dtype") == rad_kvq_desc(layout)->name) {
                CHECK_EQ(pint(&op, "gqa"), 6);
                CHECK_EQ(pint(&op, "block_size"), 4);
                ++count;
            }
        CHECK(count >= 2);
    }
}

"""

_Q35_FIXTURE = r"""TEST(qwen35_b16_trunk_keeps_the_dflash2_cache_native_e4m3) {
    MetaKV mk = with_dflash2(q35fp8_meta());
    mk.set("draft.use_sliding_window", "1").set("draft.sliding_window", "2048");
    RadArchProbe probe{};
    REQUIRE_EQ(qwen35_fp8::probe(mk.meta(), &probe), RAD_OK);
    CHECK_EQ(probe.kv_layout_ok, (1u << RAD_KVQ_B16F8) | (1u << RAD_KVQ_B16I8));
    RadModelMeta unsupported = *mk.meta();
    unsupported.head_dim = 128;
    REQUIRE_EQ(qwen35_fp8::probe(&unsupported, &probe), RAD_OK);
    CHECK_EQ(probe.kv_layout_ok, 0u);
    for (const auto layout : {RAD_KVQ_B16F8, RAD_KVQ_B16I8}) {
        RadBuildCtx ctx = spec_ctx(0, 1, 7);
        ctx.kv_dtype = RAD_U8;
        ctx.kv_layout = layout;
        RadBuilder b;
        b.kv_block = 16;
        REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        REQUIRE(b.kv_groups.size() >= 4);
        const auto& trunk = b.kv_groups.front();
        const auto& draft = b.kv_groups.back();
        CHECK_EQ(trunk.name, std::string("kv_attn"));
        CHECK_EQ(trunk.d.kv_layout, (RadKVLayout)layout);
        CHECK_EQ(trunk.d.dtype, (uint32_t)RAD_U8);
        CHECK_EQ(trunk.d.head_dim, 256);
        CHECK_EQ(draft.name, std::string("kv_draft"));
        CHECK_EQ(draft.d.dtype, (uint32_t)RAD_F8E4M3);
        CHECK_EQ(draft.d.kv_layout, (RadKVLayout)RAD_KVQ_NONE);
        CHECK_EQ(draft.d.head_dim, 128);
        CHECK_EQ(draft.d.window, 2048);
        int trunk_readers = 0, draft_readers = 0, draft_writers = 0;
        for (const auto& op : b.ops) {
            if (op.op == "attn_paged" && pstr(&op, "kv_dtype") == rad_kvq_desc(layout)->name) {
                CHECK_EQ(pint(&op, "head_dim"), 256);
                CHECK_EQ(pint(&op, "gqa"), 6);
                CHECK_EQ(pint(&op, "block_size"), 16);
                ++trunk_readers;
            }
            if (op.op == "attn_paged" && pstr(&op, "kv_dtype") == "fp8_e4m3") {
                CHECK_EQ(pint(&op, "head_dim"), 128);
                CHECK_EQ(pint(&op, "gqa"), 4);
                CHECK_EQ(pint(&op, "block_size"), 16);
                CHECK_EQ(pint(&op, "window"), 2048);
                ++draft_readers;
            }
            if (op.op == "kv_store" && pstr(&op, "kv_dtype") == "fp8_e4m3") ++draft_writers;
        }
        CHECK(trunk_readers > 0);
        CHECK(draft_readers > 0);
        CHECK(draft_writers > 0);
    }
}

"""

PATCHES += (
    FilePatch(path="tests/arch_test.cpp", language="none", edits=(
        _edit("b16-q4-unsupported-geometry", "        c.world_size = 4;\n        RadBuilder unsupported;",
              "        c.world_size = 6;   /* post-shard GQA4 remains unsupported */\n"
              "        RadBuilder unsupported;"),
        _edit("b16-q4-tp4-fixture", "TEST(qwen4exp_native_e4m3_uses_the_page4_attention_and_shared_mtp_group)",
              _Q4_TP4_FIXTURE, "TEST(qwen4exp_b16_tp4_selects_gqa6_page4)", mode="insert_before"),
        _edit("b16-qwen35-drafter-fixture", "TEST(qwen35_i8_cache_is_an_int8_group_at_the_models_head_dim)",
              _Q35_FIXTURE, "TEST(qwen35_b16_trunk_keeps_the_dflash2_cache_native_e4m3)", mode="insert_before"),
    )),
)
