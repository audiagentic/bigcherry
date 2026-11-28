"""Restore independent, opt-in E4M3 prefill products for the historical Qwen35 rows."""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch


def _replace(edit_id: str, previous: str, updated: str, guard: str | None = None) -> Edit:
    return Edit(
        id=edit_id, anchor=re.escape(previous), mode="replace", expect_matches=1,
        text=updated, guard=re.escape(guard or updated),
        max_span_lines=previous.count("\n") + 1,
        rationale="Restore independent prefill precision control on ABI44",
    )


def _before(edit_id: str, anchor: str, snippet: str, guard: str) -> Edit:
    return Edit(
        id=edit_id, anchor=re.escape(anchor), mode="insert_before", expect_matches=1,
        text=snippet, guard=re.escape(guard), max_span_lines=anchor.count("\n") + 1,
        rationale="Restore the historical Qwen35 E4M3 prefill options",
    )


_TUNING_CATALOG = '''    { "--attn-prefill-fp8",   "MODE",   "off (default) | qk | pv | both -- E4M3 prefill products\\n"
                                        "on supported fp8, fp8_v4 and k6_v4 caches (lossy)" },
'''

_PARSE = '''        else if (S("--attn-prefill-fp8")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if      (!std::strcmp(v, "off"))  c->attn_prefill_fp8 = 0;
            else if (!std::strcmp(v, "qk"))   c->attn_prefill_fp8 = RAD_ATTN_FP8_QK;
            else if (!std::strcmp(v, "pv"))   c->attn_prefill_fp8 = RAD_ATTN_FP8_PV;
            else if (!std::strcmp(v, "both")) c->attn_prefill_fp8 = RAD_ATTN_FP8_QK | RAD_ATTN_FP8_PV;
            else {
                RAD_ERR("--attn-prefill-fp8 takes off, qk, pv or both, not '%s'", v);
                return RAD_E_INVAL;
            }
        }
'''

_NAMES = '''static const char* attn_prefill_fp8_name(uint32_t mode) {
    switch (mode) {
        case 0: return "off";
        case RAD_ATTN_FP8_QK: return "qk";
        case RAD_ATTN_FP8_PV: return "pv";
        case RAD_ATTN_FP8_QK | RAD_ATTN_FP8_PV: return "both";
    }
    return "?";
}

'''

_ROW_VALIDATION = '''/* Decode bands stay exact. At least one device prefill band on each rank
 * must select a row that explicitly serves the requested combination of legs. */
static bool supports_prefill_fp8(const Program& p, uint32_t mode) {
    for (const OpInfo& op : p.ops)
        for (const Band& band : op.bands) {
            const Resolved& r = band.dom[RAD_DOMAIN_DEVICE];
            if (!r || !r.row->info) continue;
            const RadKernelInfo* k = r.row->info;
            for (int c = 0; c < k->n_constraints; ++c) {
                const RadConstraint& constraint = k->constraints[c];
                if (constraint.key && !std::strcmp(constraint.key, "prefill_fp8") &&
                    constraint.op == RAD_C_EQ && constraint.ival == mode)
                    return true;
            }
        }
    return false;
}

'''

_MODE_ADMISSION = '''    if (cfg_.attn_prefill_fp8 != 0) {
        for (const auto& r : ranks_) {
            if (!supports_prefill_fp8(r->program, cfg_.attn_prefill_fp8) && !cfg_.debug_graph) {
                RAD_ERR("rank %d: --attn-prefill-fp8 has no matching prefill row for this model, "
                        "--kv-cache-dtype %s, and the loaded kernel libraries",
                        r->index, cfg_.kv_cache_dtype.c_str());
                return RAD_E_UNSUPPORTED;
            }
        }
        RAD_WARN("attention: lossy prefill (--attn-prefill-fp8). Selected products use E4M3 WMMA");
    }
'''

_Q35_ADMISSION = '''    const bool fp8_geometry = g.head_dim == 256 && g.gqa() == 6;
    if (ctx->kv_dtype == RAD_F8E4M3 && !fp8_geometry) {
        fprintf(stderr, "radiance: qwen35: the E4M3 cache requires H256/GQA6/page16\\n");
        return RAD_E_UNSUPPORTED;
    }
    if (ctx->attn_prefill_fp8 != 0) {
        constexpr uint32_t modes = RAD_ATTN_FP8_QK | RAD_ATTN_FP8_PV;
        if (!fp8_geometry || (ctx->attn_prefill_fp8 & ~modes) ||
            (ctx->kv_dtype != RAD_F8E4M3 && ctx->kv_layout != RAD_KVQ_V4 && ctx->kv_layout != RAD_KVQ_K6V4)) {
            fprintf(stderr, "radiance: qwen35: --attn-prefill-fp8 requires fp8, fp8_v4 or k6_v4 "
                            "on H256/GQA6/page16\\n");
            return RAD_E_UNSUPPORTED;
        }
        g.attn_prefill_fp8 = ctx->attn_prefill_fp8;
    }
'''

_CACHE_GEOMETRY = """template<RadKVLayout Layout>
static int h256_fill_byte_prefill(const RadArgs* a, R4DArgs* r) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    const RadTensor* cache = r4d_opt(a, AP_KV);
    constexpr uint32_t dtype = Layout == RAD_KVQ_NONE ? RAD_F8E4M3 : RAD_U8;
    if (!cache || cache->dtype != dtype) return R4D_NO(RAD_E_DTYPE);
    constexpr int row = Layout == RAD_KVQ_NONE ? 512 : 16 * R4dKvBytes<Layout>::row16;
    return fill_paged_row(a, r, A_HEAD_DIM, row);
}

extern "C" int r4d_rad_attn_prefill_h256_fp8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = h256_fill_byte_prefill<RAD_KVQ_NONE>(a, &r);
    if (rc != RAD_OK) return rc;
    r.splits = prefill_splits(a, &r);
    return prefill_launch<0>(&r, (hipStream_t)s);
}

extern "C" int r4d_rad_attn_decode_h256_fp8(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    const RadTensor* cache = r4d_opt(a, AP_KV);
    if (!cache || cache->dtype != RAD_F8E4M3) return R4D_NO(RAD_E_DTYPE);
    R4DArgs r;
    const int rc = decode_common(a, &r, A_HEAD_DIM, 0);
    if (rc != RAD_OK) return rc;
    return decode_launch<0>(&r, (hipStream_t)s);
}

template<RadKVLayout Layout, int Options>
static int h256_prefill_fp8_products(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = h256_fill_byte_prefill<Layout>(a, &r);
    if (rc != RAD_OK) return rc;
    return prefill_launch<0, Options, Layout>(&r, (hipStream_t)s);
}

"""

_MODES = (
    ("qk8", "RAD_ATTN_FP8_QK", "O_QK8"),
    ("pv8", "RAD_ATTN_FP8_PV", "O_PV8"),
    ("qk8pv8", "RAD_ATTN_FP8_QK | RAD_ATTN_FP8_PV", "O_QK8 | O_PV8"),
)
_FORMATS = (
    ("fp8", "fp8_e4m3", "RAD_KVQ_NONE"),
    ("fp8v4", "fp8_v4", "RAD_KVQ_V4"),
    ("k6v4", "k6_v4", "RAD_KVQ_K6V4"),
)


def _bindings() -> tuple[str, str, str, str]:
    declarations = [
        "int r4d_rad_attn_prefill_h256_fp8(const RadArgs*, RadStream);\n",
        "int r4d_rad_attn_decode_h256_fp8(const RadArgs*, RadStream);\n",
    ]
    definitions = [_CACHE_GEOMETRY]
    constraints = ['''#define BC_H256_PREFILL_FP8(kv) \\
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16), \\
    C_EQ("causal", 1), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", kv)
static const RadConstraint cBcPrefillFp8[] = { BC_H256_PREFILL_FP8("fp8_e4m3") };
static const RadConstraint cBcDecodeFp8[] = { BC_H256_PREFILL_FP8("fp8_e4m3"), C_LE("q_len", 10) };
''']
    rows = ['''{ "attn_prefill_h256_gqa6_fp8kv", "attn_paged", "attn",
  "Native E4M3 paged prefill", "H256/GQA6/page16", "BF16 Q, native E4M3 K and V",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL, ROW(cBcPrefillFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_fp8, r4d_rad_attn_prefill_h256_scratch,
  nullptr, nullptr, r4d_shape_attn_paged, nullptr, nullptr, RAD_KERNEL_CAP_DYNAMIC_PREFILL_SPLIT },

{ "attn_decode_h256_gqa6_fp8kv", "attn_paged", "attn",
  "Native E4M3 paged decode", "H256/GQA6/page16", "BF16 Q, native E4M3 K and V",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE, ROW(cBcDecodeFp8), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_fp8, r4d_rad_attn_decode_h256_scratch,
  nullptr, nullptr, r4d_shape_attn_paged },

''']
    for fmt, dtype, layout in _FORMATS:
        for name, bits, opts in _MODES:
            tag = f"{fmt}_{name}"
            symbol = f"r4d_rad_attn_prefill_h256_{tag}"
            declarations.append(f"int {symbol}(const RadArgs*, RadStream);\n")
            definitions.append(f'''extern "C" int {symbol}(const RadArgs* a, RadStream s) {{
    return h256_prefill_fp8_products<{layout}, {opts}>(a, s);
}}
''')
            constraints.append(f'''static const RadConstraint c_bc_{tag}[] = {{
    BC_H256_PREFILL_FP8("{dtype}"), C_EQ("prefill_fp8", {bits}),
}};
''')
            rows.append(f'''{{ "attn_prefill_h256_gqa6_{tag}", "attn_paged", "attn",
  "Opt-in unsplit E4M3 prefill products", "H256/GQA6/page16", "{dtype}, {name}",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL_FP8, ROW(c_bc_{tag}), NOTUNE,
  nullptr, nullptr, {symbol}, nullptr, nullptr, nullptr, r4d_shape_attn_paged }},

''')
    constraints.append("#undef BC_H256_PREFILL_FP8\n\n")
    return "".join(declarations), "".join(definitions) + "\n", "".join(constraints), "".join(rows)


_DECLARATIONS, _DEFINITIONS, _CONSTRAINTS, _ROWS = _bindings()

_CONFIG_TEST = r'''TEST(attn_prefill_fp8_option_round_trips_independently_of_cache_format) {
    Config defaults;
    REQUIRE_EQ(parse({}, &defaults), RAD_OK);
    CHECK_EQ(defaults.attn_prefill_fp8, 0u);
    for (const auto& test : {std::pair{"off", 0u}, std::pair{"qk", (uint32_t)RAD_ATTN_FP8_QK},
                            std::pair{"pv", (uint32_t)RAD_ATTN_FP8_PV},
                            std::pair{"both", uint32_t(RAD_ATTN_FP8_QK | RAD_ATTN_FP8_PV)}}) {
        Config c;
        REQUIRE_EQ(parse({"--kv-cache-dtype", "k6_v4", "--attn-prefill-fp8", test.first}, &c), RAD_OK);
        CHECK_EQ(c.kv_cache_dtype, std::string("k6_v4"));
        CHECK_EQ(c.attn_prefill_fp8, test.second);
    }
    CHECK_EQ(parse({"--attn-prefill-fp8", "qk+pv"}, &defaults), RAD_E_INVAL);
}

'''
_ARCH_TEST = '''TEST(qwen35_prefill_fp8_modes_follow_only_supported_trunk_caches) {
    MetaKV meta = q35fp8_meta();
    constexpr uint32_t both = RAD_ATTN_FP8_QK | RAD_ATTN_FP8_PV;
    RadArchProbe probe{};
    REQUIRE_EQ(qwen35_fp8::probe(meta.meta(), &probe), RAD_OK);
    CHECK_EQ(probe.kv_fp8_ok, 1);
    for (const auto layout : {RAD_KVQ_NONE, RAD_KVQ_V4, RAD_KVQ_K6V4}) {
        for (uint32_t mode : {0u, (uint32_t)RAD_ATTN_FP8_QK, (uint32_t)RAD_ATTN_FP8_PV, both}) {
            RadBuildCtx ctx = build_ctx(0, 1);
            ctx.kv_dtype = layout == RAD_KVQ_NONE ? RAD_F8E4M3 : RAD_U8;
            ctx.kv_layout = layout;
            ctx.attn_prefill_fp8 = mode;
            RadBuilder b;
            b.kv_block = 16;
            REQUIRE_EQ(qwen35_fp8::declare(&b, meta.meta(), &ctx), RAD_OK);
            int prefill = 0;
            for (const auto& op : b.ops) {
                if (op.op != "attn_paged") continue;
                CHECK_EQ(pint(&op, "head_dim"), 256);
                CHECK_EQ(pint(&op, "block_size"), 16);
                if (param(&op, "prefill_fp8")) {
                    CHECK_EQ(pint(&op, "prefill_fp8"), mode);
                    ++prefill;
                }
            }
            CHECK_EQ(prefill > 0, mode != 0);
        }
    }
    RadBuildCtx ctx = build_ctx(0, 1);
    ctx.kv_dtype = RAD_U8;
    ctx.kv_layout = RAD_KVQ_B16I8;
    ctx.attn_prefill_fp8 = both;
    RadBuilder unsupported;
    CHECK_EQ(stderr_of_rc([&] { return qwen35_fp8::declare(&unsupported, meta.meta(), &ctx); }),
             RAD_E_UNSUPPORTED);
    RadModelMeta wrong_geometry = *meta.meta();
    wrong_geometry.head_dim = 128;
    ctx.kv_layout = RAD_KVQ_V4;
    RadBuilder wrong_head;
    CHECK_EQ(stderr_of_rc([&] { return qwen35_fp8::declare(&wrong_head, &wrong_geometry, &ctx); }),
             RAD_E_UNSUPPORTED);
    CHECK(wrong_head.kv_groups.empty());
}

'''

PATCHES = (
    FilePatch(path="core/rad_core.h", language="none", edits=(
        _replace("prefill-mode-config-field", '    std::string kv_cache_dtype = "bf16";',
                 '    std::string kv_cache_dtype = "bf16";\n    uint32_t attn_prefill_fp8 = 0;'),
    )),
    FilePatch(path="core/config.cpp", language="none", edits=(
        _before("prefill-mode-option", '    { "--state-cache-dtype",  "T",', _TUNING_CATALOG,
                '{ "--attn-prefill-fp8",   "MODE",'),
        _before("prefill-mode-parse", '        else if (S("--kv-resident"))', _PARSE,
                'else if (S("--attn-prefill-fp8"))'),
        _before("prefill-mode-names", "std::string onoff(bool b)", _NAMES, "const char* attn_prefill_fp8_name("),
        _before("prefill-mode-show", '    { "--state-cache-dtype",  [](const Config&',
                '    { "--attn-prefill-fp8", [](const Config& c) { return std::string(attn_prefill_fp8_name(c.attn_prefill_fp8)); } },\n',
                "attn_prefill_fp8_name(c.attn_prefill_fp8)"),
    )),
    FilePatch(path="core/engine_bringup.cpp", language="none", edits=(
        _replace("prefill-mode-context", "    set_build_kv_format(cfg_.kv_cache_dtype.c_str(), &bc);",
                 "    set_build_kv_format(cfg_.kv_cache_dtype.c_str(), &bc);\n"
                 "    bc.attn_prefill_fp8 = cfg_.attn_prefill_fp8;"),
        _before("prefill-mode-row-evidence", "static MissReport miss_report", _ROW_VALIDATION,
                "static bool supports_prefill_fp8("),
        _before("prefill-mode-validate-ranks", '    RAD_INFO("declare: %zu ops, %zu weights, %zu buffers, %zu KV groups",',
                _MODE_ADMISSION, "if (!supports_prefill_fp8(r->program"),
    )),
    FilePatch(path="arch/common/rad_arch.h", language="none", edits=(
        _replace("prefill-mode-geometry", '    const char* kv_dtype = "bf16";',
                 '    const char* kv_dtype = "bf16";\n    uint32_t attn_prefill_fp8 = 0;'),
    )),
    FilePatch(path="arch/common/rad_block_attn_gated_q.h", language="none", edits=(
        _replace("prefill-mode-op-params", '    op_attn = rad_decl_op(b, "attn_paged",\n                     RAD_PARAMS(',
                 "    const RadParam attn_params[] = {"),
        _replace("prefill-mode-op-key",
                 '                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),\n'
                 '                     RAD_NOWEIGHTS);',
                 '                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype),\n'
                 '                                RAD_INT("prefill_fp8", g.attn_prefill_fp8) };\n'
                 '    const int attn_param_count = sizeof(attn_params) / sizeof(attn_params[0]);\n'
                 '    op_attn = rad_decl_op(b, "attn_paged", attn_params,\n'
                 '                          attn_param_count - (g.attn_prefill_fp8 == 0), RAD_NOWEIGHTS);'),
    )),
    FilePatch(path="arch/qwen35_fp8/qwen35_fp8.cpp", language="none", edits=(
        _before("prefill-mode-q35-geometry", "    if (ctx->kv_layout != RAD_KVQ_NONE &&",
                _Q35_ADMISSION, "g.attn_prefill_fp8 = ctx->attn_prefill_fp8;"),
        _replace("prefill-native-q35-drafter", "    if (ctx->kv_layout != RAD_KVQ_NONE) {",
                 "    if (ctx->kv_layout != RAD_KVQ_NONE || ctx->kv_dtype == RAD_F8E4M3) {"),
        _replace("prefill-native-q35-drafter-error", "DFlash2 over a packed trunk", "the native E4M3 DFlash2 cache"),
        _before("prefill-native-q35-probe", "    /* The delta-net state cache is declared",
                "    out->kv_fp8_ok = hd == 256 && kv_heads > 0 && meta->n_head % kv_heads == 0 &&\n"
                "                     meta->n_head / kv_heads == 6;\n", "out->kv_fp8_ok = hd == 256"),
    )),
    FilePatch(path="libr4d/r4d_attn_paged_h256_gqa6.hip", language="none", edits=(
        _replace("prefill-fp8-products-stay-unsplit", "    constexpr bool CAN_SPLIT = true;",
                 "    constexpr bool CAN_SPLIT = !(XOPT & (O_QK8 | O_PV8));"),
        _before("prefill-fp8-launch-bindings", "/* Resident B16, Qwen35's H256/GQA6/page16 dense attention. */",
                _DEFINITIONS, "static int h256_prefill_fp8_products("),
    )),
    FilePatch(path="libr4d/r4d_plugin.h", language="none", edits=(
        _before("prefill-fp8-launch-symbols", "int     r4d_rad_attn_prefill_h256_gqa12_b16f8(",
                _DECLARATIONS, "int r4d_rad_attn_prefill_h256_k6v4_qk8pv8("),
    )),
    FilePatch(path="libr4d/r4d_rows.cpp", language="none", edits=(
        _before("prefill-fp8-row-constraints", "static const RadConstraint cAttnPrefillBf16[]", _CONSTRAINTS,
                "static const RadConstraint c_bc_k6v4_qk8pv8[]"),
        _before("prefill-fp8-rows", '{ "attn_prefill_h256_gqa6_bf16kv",', _ROWS,
                '{ "attn_prefill_h256_gqa6_k6v4_qk8pv8",'),
        _replace("prefill-native-fp8-chain", '        "attn_decode_h256_gqa6_i8", "attn_decode_h256_gqa6_bf16kv",\n',
                 '        "attn_decode_h256_gqa6_i8", "attn_decode_h256_gqa6_bf16kv",\n'
                 '        "attn_decode_h256_gqa6_fp8kv",\n'),
    )),
    FilePatch(path="tests/arch_test.cpp", language="none", edits=(
        _before("prefill-mode-arch-checks", "TEST(qwen35_b16_trunk_keeps_the_dflash2_cache_native_e4m3)",
                _ARCH_TEST, "TEST(qwen35_prefill_fp8_modes_follow_only_supported_trunk_caches)"),
    )),
    FilePatch(path="tests/config_test.cpp", language="none", edits=(
        _before("prefill-mode-config-inventory", '        { "--state-cache-dtype", { "f16" } },',
                '        { "--attn-prefill-fp8", { "both" } },\n',
                '{ "--attn-prefill-fp8", { "both" } },'),
        _before("prefill-mode-config-checks", "TEST(the_kv_cache_dtype_uses_canonical_names)",
                _CONFIG_TEST, "TEST(attn_prefill_fp8_option_round_trips_independently_of_cache_format)"),
    )),
    FilePatch(path="tests/bc_b16_kv_runtime_test.cpp", language="none", edits=(
        _before("prefill-mode-native-yaml", "TEST(config_to_resident_page)", r'''TEST(prefill_fp8_native_yaml_round_trips) {
    fleet::FleetConfig yaml;
    REQUIRE_EQ(fleet::parse("models:\n  main:\n    model: unused.rad\n"
                           "    kv-cache-dtype: fp8_v4\n    attn-prefill-fp8: both\n",
                           "prefill-fp8-fixture", &yaml), RAD_OK);
    REQUIRE_EQ(yaml.models.size(), size_t{1});
    const Config& cfg = yaml.models.front().cfg;
    CHECK_EQ(cfg.kv_cache_dtype, std::string("fp8_v4"));
    CHECK_EQ(cfg.attn_prefill_fp8, uint32_t(RAD_ATTN_FP8_QK | RAD_ATTN_FP8_PV));
    CHECK_EQ(cfg.given.at("--attn-prefill-fp8"), "models.main.attn-prefill-fp8");
}

''', "TEST(prefill_fp8_native_yaml_round_trips)"),
    )),
)
