"""Local ABI44 contract layered on the reviewed Radiance ABI43 source.

rad0018 and rad0020 declare and fingerprint the TP ownership and prefill
pipeline fields first. This package makes the single version transition and
appends the KV layout and prefill-precision fields. Separate runtime patches
own format selection, kernel dispatch and capability advertisement.
"""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch


def _replace(edit_id: str, before: str, after: str, guard: str) -> Edit:
    return Edit(
        id=edit_id,
        anchor=re.escape(before),
        mode="replace",
        text=after,
        guard=re.escape(guard),
        expect_matches=1,
        max_span_lines=before.count("\n") + 1,
        rationale="Extend the reviewed Radiance ABI43 headers once for BigCherry ABI44",
    )


_KV_ROW_ELEMS = """static inline int64_t rad_kv_row_elems(uint32_t dtype, int64_t head_dim) {
    return 2 * head_dim + (dtype == RAD_I8 ? 4 : 0);
}
"""

_KV_LAYOUTS = r"""
/* BigCherry ABI44 packed KV formats. Layout 0 uses the native dtype:
 * E4M3 has no stored scales and native I8 has two scales per head. Packed
 * rows use opaque RAD_U8 storage and keep head_dim as the logical width.
 *
 * The offsets and row lengths are bytes per 16 logical elements. All
 * group-scaled formats use 32 elements per FP16 scale. A zero scale
 * offset denotes no stored scale for that operand. */
#define RAD_KV_LAYOUTS(X) \
    X(V4,     1, "fp8_v4",   25, 16,  0, 24, 32) \
    X(K6V4,   2, "k6_v4",    22, 12, 20, 21, 32) \
    X(B16F8,  3, "bf16_fp8", 48, 32,  0,  0,  0) \
    X(B16I8,  4, "bf16_i8",  49, 32,  0, 48, 32)

typedef uint32_t RadKVLayout;
enum {
    RAD_KVQ_NONE = 0,
#define RAD_KV_LAYOUT_ENUM(id, number, name, row, value, ks, vs, group) RAD_KVQ_##id = number,
    RAD_KV_LAYOUTS(RAD_KV_LAYOUT_ENUM)
#undef RAD_KV_LAYOUT_ENUM
    RAD_KVQ_COUNT
};

enum {
    RAD_ATTN_FP8_QK = 1,
    RAD_ATTN_FP8_PV = 2
};

/* Header-local lookup data. Only RadKVLayout crosses the plugin interface. */
typedef struct RadKVLayoutDesc {
    const char* name;
    int row16;
    int value16;
    int key_scale16;
    int value_scale16;
    int scale_group;
} RadKVLayoutDesc;

static inline const RadKVLayoutDesc* rad_kvq_desc(RadKVLayout layout) {
    static const RadKVLayoutDesc formats[] = {
        {0, 0, 0, 0, 0, 0},
#define RAD_KV_LAYOUT_DESC(id, number, name, row, value, ks, vs, group) {name, row, value, ks, vs, group},
        RAD_KV_LAYOUTS(RAD_KV_LAYOUT_DESC)
#undef RAD_KV_LAYOUT_DESC
    };
    return layout > RAD_KVQ_NONE && layout < RAD_KVQ_COUNT ? &formats[layout] : 0;
}

static inline RadKVLayout rad_kvq_from_name(const char* name) {
    if (name) {
        for (RadKVLayout layout = RAD_KVQ_NONE + 1; layout < RAD_KVQ_COUNT; ++layout)
            if (!strcmp(name, rad_kvq_desc(layout)->name)) return layout;
    }
    return RAD_KVQ_NONE;
}

/* Model admission owns the supported head and page geometry. */
static inline int64_t rad_kv_group_row_elems(RadKVLayout layout, uint32_t dtype, int64_t head_dim) {
    if (layout == RAD_KVQ_NONE) return rad_kv_row_elems(dtype, head_dim);
    const RadKVLayoutDesc* format = rad_kvq_desc(layout);
    if (!format || dtype != RAD_U8 || head_dim <= 0) return 0;
    const int group = format->scale_group ? format->scale_group : 16;
    if (head_dim % group != 0 || head_dim / 16 > INT64_MAX / format->row16) return 0;
    return (head_dim / 16) * format->row16;
}
"""

_FIXTURE = r"""/* ABI44 layout and packed KV byte contract, runnable without a GPU. */
#include "rad_abi_layout.h"

#include <inttypes.h>
#include <stdio.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(void) {
    const RadBuildCtx ctx = {0};
    const RadKVGroupDecl group = {0};
    const RadArchProbe probe = {0};
    const int64_t hd = 256;
    CHECK(RAD_ABI_VERSION == 45u);
    CHECK(!ctx.tp_ownership && ctx.pipelined_prefill == 0 && ctx.pipelined_prefill_min_rows == 0);
    CHECK(ctx.kv_layout == RAD_KVQ_NONE && ctx.attn_prefill_fp8 == 0);
    CHECK(group.kv_layout == RAD_KVQ_NONE && probe.kv_layout_ok == 0);
    CHECK(RAD_ATTN_FP8_QK == 1 && RAD_ATTN_FP8_PV == 2);

    CHECK(rad_kv_group_row_elems(RAD_KVQ_NONE, RAD_F8E4M3, hd) == 512);
    CHECK(rad_kv_group_row_elems(RAD_KVQ_NONE, RAD_I8, hd) == 516);
    CHECK(rad_kv_group_row_elems(RAD_KVQ_NONE, RAD_BF16, hd) == 512);
    CHECK(!rad_kvq_desc(RAD_KVQ_NONE) && !rad_kvq_desc(RAD_KVQ_COUNT));
    CHECK(rad_kvq_from_name("fp8") == RAD_KVQ_NONE && rad_kvq_from_name("i8") == RAD_KVQ_NONE);

    const struct {
        RadKVLayout layout;
        const char* name;
        int row_bytes;
        int value;
        int key_scale;
        int value_scale;
        int scale_group;
    } expected[] = {
        { RAD_KVQ_V4,    "fp8_v4",   400, 256,   0, 384, 32 },
        { RAD_KVQ_K6V4,  "k6_v4",    352, 192, 320, 336, 32 },
        { RAD_KVQ_B16F8, "bf16_fp8", 768, 512,   0,   0,  0 },
        { RAD_KVQ_B16I8, "bf16_i8",  784, 512,   0, 768, 32 },
    };
    for (unsigned i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i) {
        const RadKVLayout layout = expected[i].layout;
        const RadKVLayoutDesc* format = rad_kvq_desc(layout);
        CHECK(layout == i + 1 && format);
        CHECK(strcmp(format->name, expected[i].name) == 0);
        CHECK(rad_kvq_from_name(expected[i].name) == layout);
        CHECK(rad_kv_group_row_elems(layout, RAD_U8, hd) == expected[i].row_bytes);
        CHECK(rad_kv_group_row_elems(layout, RAD_I8, hd) == 0);
        CHECK(rad_kv_group_row_elems(layout, RAD_U8, hd - 1) == 0);
        CHECK(format->value16 * (hd / 16) == expected[i].value);
        CHECK(format->key_scale16 * (hd / 16) == expected[i].key_scale);
        CHECK(format->value_scale16 * (hd / 16) == expected[i].value_scale);
        CHECK(format->scale_group == expected[i].scale_group);
    }
    printf("%" PRIx64 "\n", rad_abi_layout());
    return 0;
}
"""

_CTX_END = "    int64_t pipelined_prefill_min_rows;\n} RadBuildCtx;"
_GROUP_END = "     * rad_kv_block_size() after declaring the attention op that consumes this group. */\n} RadKVGroupDecl;"
_PROBE_END = "    int32_t ckpt_snap_ok;\n} RadArchProbe;"


_ABI44_LOADER_TESTS = r"""/* ABI45 migration gates: a real ABI44 plugin and an ABI45 plugin without the
 * required layout export must both be refused by the loader. */
TEST(an_abi44_plugin_is_refused_by_the_abi45_engine) {
    const char* path = std::getenv("RADTEST_ABI44_PLUGIN");
    if (!path || !*path) return; /* the dedicated migration gate supplies a retained ABI44 DSO */
    Registry r;
    const int st = r.load_plugin(path, 0);
    CHECK_EQ(st, RAD_E_ABI);
    CHECK_EQ(r.plugins().size(), (size_t)0);
    const std::string e = errs(r);
    CHECK(has(e, path));
    CHECK(has(e, "ABI version 44"));
    CHECK(has(e, "this engine is version 45"));
}

TEST(an_abi45_plugin_without_layout_export_is_refused) {
    Registry r;
    const std::string path = bad_dir() + "/radtest_nolayout.so";
    const int st = r.load_plugin(path, 0);
    CHECK_EQ(st, RAD_E_ABI);
    CHECK_EQ(r.plugins().size(), (size_t)0);
    const std::string e = errs(r);
    CHECK(has(e, "radtest_nolayout.so"));
    CHECK(has(e, "exports no rad_plugin_abi_layout"));
}

"""

PATCHES = (
    FilePatch(path="abi/rad_builder.h", language="none", description="Local ABI44 KV layout contract", edits=(
        _replace("abi44-string-header", '#include "rad_abi.h"\n',
                 '#include "rad_abi.h"\n#include <string.h>\n', '#include <string.h>'),
        _replace("abi44-kv-formats", _KV_ROW_ELEMS, _KV_ROW_ELEMS + _KV_LAYOUTS,
                 "#define RAD_KV_LAYOUTS(X)"),
        _replace("abi44-build-context", _CTX_END,
                 "    int64_t pipelined_prefill_min_rows;\n"
                 "\n"
                 "    RadKVLayout kv_layout;       /* zero selects the native kv_dtype */\n"
                 "    uint32_t attn_prefill_fp8;   /* independent RAD_ATTN_FP8_* bits, zero is exact */\n"
                 "} RadBuildCtx;", "RadKVLayout kv_layout;       /* zero selects the native kv_dtype */"),
        _replace("abi44-kv-group", _GROUP_END,
                 "     * rad_kv_block_size() after declaring the attention op that consumes this group. */\n"
                 "    RadKVLayout kv_layout;  /* packed rows keep logical head_dim and use dtype RAD_U8 */\n"
                 "} RadKVGroupDecl;", "RadKVLayout kv_layout;  /* packed rows keep logical head_dim"),
        _replace("abi44-arch-probe", _PROBE_END,
                 "    int32_t ckpt_snap_ok;\n"
                 "    uint32_t kv_layout_ok;  /* bit (1u << layout) for each supported packed format */\n"
                 "} RadArchProbe;", "uint32_t kv_layout_ok;"),
    )),
    FilePatch(path="abi/rad_abi_layout_rows.h", language="none", description="Fingerprint the ABI44 fields", edits=(
        _replace("abi44-layout-context", "ABI_FIELD(RadBuildCtx, pipelined_prefill_min_rows)\n",
                 "ABI_FIELD(RadBuildCtx, pipelined_prefill_min_rows)\n"
                 "ABI_FIELD(RadBuildCtx, kv_layout)\n"
                 "ABI_FIELD(RadBuildCtx, attn_prefill_fp8)\n", "ABI_FIELD(RadBuildCtx, attn_prefill_fp8)"),
        _replace("abi44-layout-group", "ABI_FIELD(RadKVGroupDecl, conv_width)\n",
                 "ABI_FIELD(RadKVGroupDecl, conv_width)\n"
                 "ABI_FIELD(RadKVGroupDecl, kv_layout)\n", "ABI_FIELD(RadKVGroupDecl, kv_layout)"),
        _replace("abi44-layout-probe", "ABI_FIELD(RadArchProbe, ckpt_snap_ok)\n",
                 "ABI_FIELD(RadArchProbe, ckpt_snap_ok)\n"
                 "ABI_FIELD(RadArchProbe, kv_layout_ok)\n", "ABI_FIELD(RadArchProbe, kv_layout_ok)"),
        _replace("abi44-layout-batch", "ABI_FIELD(RadBatch, vl_mask)\n",
                 "ABI_FIELD(RadBatch, vl_mask)\n"
                 "ABI_FIELD(RadBatch, cu_seqlens_h)\n", "ABI_FIELD(RadBatch, cu_seqlens_h)"),
    )),
    FilePatch(path="tests/abi_layout_test.cpp", language="none", description="Check ABI45 layout", edits=(
        _replace("abi44-test-doc", "taken at RAD_ABI_VERSION 38 and RAD_FORMAT_VER 2",
                 "taken at RAD_ABI_VERSION 45 and RAD_FORMAT_VER 2",
                 "taken at RAD_ABI_VERSION 45 and RAD_FORMAT_VER 2"),
        _replace("abi44-test-version", "/* The layout at RAD_ABI_VERSION 44 / RAD_FORMAT_VER 2 (see the file header). */\n"
                 "constexpr uint32_t kTableAbi = 44, kTableFormat = 2;",
                 "/* The layout at RAD_ABI_VERSION 45 / RAD_FORMAT_VER 2 (see the file header). */\n"
                 "constexpr uint32_t kTableAbi = 45, kTableFormat = 2;", "constexpr uint32_t kTableAbi = 45,"),
        _replace("abi44-test-context-size", "CHECK_EQ(sizeof(RadBuildCtx), (size_t)224);",
                 "CHECK_EQ(sizeof(RadBuildCtx), (size_t)256);", "CHECK_EQ(sizeof(RadBuildCtx), (size_t)256);"),
        _replace("abi44-test-probe-size", "CHECK_EQ(sizeof(RadArchProbe), (size_t)128);",
                 "CHECK_EQ(sizeof(RadArchProbe), (size_t)136);", "CHECK_EQ(sizeof(RadArchProbe), (size_t)136);"),
        _replace("abi44-test-batch-size", "CHECK_EQ(sizeof(RadBatch), (size_t)504);",
                 "CHECK_EQ(sizeof(RadBatch), (size_t)512);", "CHECK_EQ(sizeof(RadBatch), (size_t)512);"),
    )),
    FilePatch(path="tests/abi_layout_table.h", language="none", description="Pin ABI45 layout extensions", edits=(
        _replace("abi44-table-doc", "at RAD_ABI_VERSION 43 and", "at RAD_ABI_VERSION 45 and",
                 "at RAD_ABI_VERSION 45 and"),
        _replace("abi44-table-probe-size", '{ "RadArchProbe", "", 0, 128 },',
                 '{ "RadArchProbe", "", 0, 136 },', '{ "RadArchProbe", "", 0, 136 },'),
        _replace("abi44-table-probe", '{ "RadArchProbe", "ckpt_snap_ok", 124, 4 },\n',
                 '{ "RadArchProbe", "ckpt_snap_ok", 124, 4 },\n'
                 '{ "RadArchProbe", "kv_layout_ok", 128, 4 },\n',
                 '{ "RadArchProbe", "kv_layout_ok", 128, 4 },'),
        _replace("abi44-table-batch-size", '{ "RadBatch", "", 0, 504 },',
                 '{ "RadBatch", "", 0, 512 },', '{ "RadBatch", "", 0, 512 },'),
        _replace("abi44-table-batch", '{ "RadBatch", "vl_mask", 496, 8 },\n',
                 '{ "RadBatch", "vl_mask", 496, 8 },\n'
                 '{ "RadBatch", "cu_seqlens_h", 504, 8 },\n',
                 '{ "RadBatch", "cu_seqlens_h", 504, 8 },'),
        _replace("abi44-table-context-size", '{ "RadBuildCtx", "", 0, 208 },',
                 '{ "RadBuildCtx", "", 0, 256 },', '{ "RadBuildCtx", "", 0, 256 },'),
        _replace("abi44-table-context", '{ "RadBuildCtx", "draft_head_off", 204, 4 },\n',
                 '{ "RadBuildCtx", "draft_head_off", 204, 4 },\n'
                 '{ "RadBuildCtx", "expert_split", 208, 8 },\n'
                 '{ "RadBuildCtx", "dspark_walk", 216, 4 },\n'
                 '{ "RadBuildCtx", "tp_ownership", 224, 8 },\n'
                 '{ "RadBuildCtx", "pipelined_prefill", 232, 4 },\n'
                 '{ "RadBuildCtx", "pipelined_prefill_min_rows", 240, 8 },\n'
                 '{ "RadBuildCtx", "kv_layout", 248, 4 },\n'
                 '{ "RadBuildCtx", "attn_prefill_fp8", 252, 4 },\n',
                 '{ "RadBuildCtx", "attn_prefill_fp8", 252, 4 },'),
        _replace("abi44-table-group-size", '{ "RadKVGroupDecl", "", 0, 56 },',
                 '{ "RadKVGroupDecl", "", 0, 64 },', '{ "RadKVGroupDecl", "", 0, 64 },'),
        _replace("abi44-table-group", '{ "RadKVGroupDecl", "conv_width", 48, 8 },\n',
                 '{ "RadKVGroupDecl", "conv_width", 48, 8 },\n'
                 '{ "RadKVGroupDecl", "kv_layout", 56, 4 },\n',
                 '{ "RadKVGroupDecl", "kv_layout", 56, 4 },'),
    )),
    FilePatch(path="abi/rad_abi.h", language="none", description="Advance composed ABI from native 44 to 45", edits=(
        _replace("abi44-version", "#define RAD_ABI_VERSION 44u\n", "#define RAD_ABI_VERSION 45u\n",
                 "#define RAD_ABI_VERSION 45u\n"),
    )),
    FilePatch(path="tests/bc_kv_layout_contract_test.c", language="none", create=True,
              description="Always-on ABI44 descriptor and row byte fixture", edits=(
        Edit(id="abi44-contract-fixture", anchor=r"\A", mode="insert_after", text=_FIXTURE,
             guard=re.escape("/* ABI44 layout and packed KV byte contract, runnable without a GPU. */")),
    )),
    FilePatch(path="tests/CMakeLists.txt", language="none", description="Register the ABI44 CPU test", edits=(
        Edit(id="abi44-register-contract", anchor=r"\Z", mode="insert_after", text="""
add_executable(bc_kv_layout_contract_test bc_kv_layout_contract_test.c)
target_include_directories(bc_kv_layout_contract_test PRIVATE ${CMAKE_SOURCE_DIR}/abi)
add_test(NAME bc_kv_layout_contract_test COMMAND bc_kv_layout_contract_test)
set_tests_properties(bc_kv_layout_contract_test PROPERTIES LABELS "unit;fast")
""", guard=re.escape("add_executable(bc_kv_layout_contract_test "),
             rationale="Run the layout and row checks in any build type"),
    )),
    FilePatch(path="core/plugin/testplugin/testplugin.cc", language="none", description="Test missing ABI44 layout export refusal", edits=(
        _replace("abi44-missing-layout-export",
                 "#else\nRAD_PLUGIN_ABI_LAYOUT_EXPORT\n#endif",
                 "#elif !defined(RADTEST_NOLAYOUT)\nRAD_PLUGIN_ABI_LAYOUT_EXPORT\n#endif",
                 "#elif !defined(RADTEST_NOLAYOUT)\nRAD_PLUGIN_ABI_LAYOUT_EXPORT"),
    )),
    FilePatch(path="core/plugin/testplugin/CMakeLists.txt", language="none", description="Build a plugin missing the ABI44 layout export", edits=(
        _replace("abi44-nolayout-plugin",
                 "rad_add_plugin(radtest_badlayout testkernels/bad SOURCES " + chr(36) + chr(123) + "CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)\ntarget_compile_definitions(radtest_badlayout PRIVATE RADTEST_BADLAYOUT=1)\n",
                 "rad_add_plugin(radtest_badlayout testkernels/bad SOURCES " + chr(36) + chr(123) + "CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)\ntarget_compile_definitions(radtest_badlayout PRIVATE RADTEST_BADLAYOUT=1)\n\nrad_add_plugin(radtest_nolayout testkernels/bad SOURCES " + chr(36) + chr(123) + "CMAKE_CURRENT_SOURCE_DIR}/testplugin.cc)\ntarget_compile_definitions(radtest_nolayout PRIVATE RADTEST_BASE=1 RADTEST_NOLAYOUT=1)\n",
                 "target_compile_definitions(radtest_nolayout PRIVATE RADTEST_BASE=1 RADTEST_NOLAYOUT=1)"),
    )),
    FilePatch(path="tests/plugin_test.cpp", language="none", description="Refuse stale ABI43 and missing-layout plugins", edits=(
        Edit(id="abi44-loader-migration-cases",
             anchor=re.escape("TEST(a_foreign_layout_under_the_right_version_is_refused) {"),
             mode="insert_before", expect_matches=1, text=_ABI44_LOADER_TESTS,
             guard=re.escape("TEST(an_abi45_plugin_without_layout_export_is_refused)"),
             max_span_lines=2,
             rationale="Prove ABI43 mismatch and missing ABI44 fingerprint refusal"),
    )),

)
