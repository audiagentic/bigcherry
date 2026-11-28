"""VTP44 packed KV selftests on the Radiance 2.2 B16/V4 implementation."""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch


def _edit(name, before, after, guard, *, count=1, mode="replace"):
    span = max(4, before.count("\n") + 2)
    return Edit(
        id=name, anchor=re.escape(before), text=after, guard=re.escape(guard),
        mode=mode, expect_matches=count, max_span_lines=span,
        rationale=("long-anchor: distinguish the selftest hunk; " if span > 6 else "") +
                  "VTP44 diagnostic packed-KV fixture",
    )


_LAYOUTS = r"""/* The stock q8 selector 3 remains int8; packed rows use ABI44 opaque storage. */
static RadKVLayout selftest_packed_kv(int kvq) {
    switch (kvq) {
        case 1: return RAD_KVQ_V4;
        case 2: return RAD_KVQ_K6V4;
        case 4: return RAD_KVQ_B16F8;
        case 5: return RAD_KVQ_B16I8;
        default: return RAD_KVQ_NONE;
    }
}

static uint32_t selftest_kv_dtype(int kvq, bool byte_cache) {
    if (selftest_packed_kv(kvq) != RAD_KVQ_NONE) return RAD_U8;
    return kvq == 3 ? RAD_I8 : byte_cache ? RAD_F8E4M3 : RAD_BF16;
}

static int64_t kvq_row_bytes(int64_t head_dim, int kvq);

"""

_B16_FILL = r"""        if (kvq == 4 || kvq == 5) {
            for (int64_t d = 0; d < head_dim; ++d) {
                const uint16_t key = st_rand_bf16((uint32_t)(r0 + d));
                std::memcpy(row + 2 * d, &key, sizeof(key));
                uint8_t& value = row[2 * head_dim + d];
                if (kvq == 5 && value == 0x80) value = 0x81;
            }
            if (kvq == 5) {
                const int group = rad_kvq_desc(RAD_KVQ_B16I8)->scale_group;
                for (int64_t g = 0; g < head_dim / group; ++g) {
                    const _Float16 scale = (_Float16)(0.0078125f * (1 + (st_rand_e4m3(r0 + g + 5) & 15)));
                    std::memcpy(row + 3 * head_dim + 2 * g, &scale, sizeof(scale));
                }
            }
            continue;
        }
"""

_ROW_INVENTORY = r"""static void bf16_i8_rows_case(const Plugin& pl) {
    if (!sel("bf16_i8_rows")) return;
    struct Row { const char* name; int gqa, page; };
    static const Row expected[] = {
        { "attn_prefill_h256_gqa6_b16i8", 6, 16 }, { "attn_decode_h256_gqa6_b16i8", 6, 16 },
        { "kv_store_page16_b16i8", 0, 16 }, { "qk_norm_rope_gate_kv_store_page16_b16i8", 0, 16 },
        { "kv_store_b16i8", 0, 4 }, { "qk_norm_rope_gate_kv_store_b16i8", 0, 4 },
        { "attn_prefill_h256_gqa12_b16i8", 12, 4 }, { "attn_decode_h256_gqa12_b16i8", 12, 4 },
        { "attn_decode_gq_h256_gqa12_b16i8", 12, 4 }, { "attn_decode_ghq_h256_gqa12_b16i8", 12, 4 },
        { "attn_prefill_h256_gqa6_bs4_b16i8", 6, 4 }, { "attn_decode_h256_gqa6_bs4_b16i8", 6, 4 },
        { "attn_decode_gq_h256_gqa6_bs4_b16i8", 6, 4 }, { "attn_decode_ghq_h256_gqa6_bs4_b16i8", 6, 4 },
    };
    const auto pins = [](const RadKernelInfo* row, const char* key, int value) {
        for (int i = 0; i < row->n_constraints; ++i) {
            const auto& c = row->constraints[i];
            if (c.op == RAD_C_EQ && !std::strcmp(c.key, key) && c.ival == value) return true;
        }
        return false;
    };
    std::string errors;
    for (const auto& want : expected) {
        const auto* row = find(pl, want.name);
        if (!row || !pins(row, "head_dim", 256) || !pins(row, "block_size", want.page) ||
            (want.gqa && (!pins(row, "gqa", want.gqa) || !pins(row, "window", 0))))
            errors += std::string(" missing or wrong geometry: ") + want.name;
    }
    int packed_rows = 0;
    for (int k = 0; k < pl.kernel_count(); ++k) {
        const auto* row = pl.kernel_at(k);
        for (int i = 0; i < row->n_constraints; ++i) {
            const auto& c = row->constraints[i];
            if (c.op != RAD_C_IN || std::strcmp(c.key, "kv_dtype") || !c.sval ||
                (std::string(" ") + c.sval + " ").find(" bf16_i8 ") == std::string::npos) continue;
            ++packed_rows;
            if (std::none_of(std::begin(expected), std::end(expected), [&](const Row& want) {
                    return !std::strcmp(want.name, row->name);
                })) errors += std::string(" unexpected row: ") + row->name;
        }
    }
    ok("bf16_i8_rows", errors.empty() && packed_rows == int(sizeof(expected) / sizeof(expected[0])), errors);
}

"""

# Preserve both Flash-Next geometries. V4 uses the same ABI44 opaque plane.
_CASES = (("bf16_fp8", "b16f8"), ("bf16_i8", "b16i8"), ("fp8_v4", "fp8v4"), ("k6_v4", "k6v4"))
_GEOMETRIES = (("kSplitG12", "gqa12"), ("kSplitG6Bs4", "gqa6_bs4"))


def _matrix(mixed=False):
    rows = []
    for geo, tag in _GEOMETRIES:
        for layout, suffix in _CASES:
            if tag == "gqa6_bs4" and layout == ("fp8_v4" if mixed else "k6_v4"):
                continue
            prefix = f'        {{ &{geo}, "attn_prefill_h256_{tag}_{suffix}", '
            if mixed:
                prefix += f'"attn_decode_h256_{tag}_{suffix}", '
            rows.append(prefix + f'"{layout}" }},\n')
    return "".join(rows)


_NUMERIC = r"""    bf16_i8_rows_case(pl);
    for (const auto& layout : { std::make_pair("bf16_fp8", "b16f8"), std::make_pair("bf16_i8", "b16i8"),
                               std::make_pair("fp8_v4", "fp8v4"), std::make_pair("k6_v4", "k6v4") }) {
        const char* kv = layout.first;
        const std::string p12 = std::string("attn_prefill_h256_gqa12_") + layout.second;
        const std::string d12 = std::string("attn_decode_h256_gqa12_") + layout.second;
        const std::string p6 = std::string("attn_prefill_h256_gqa6_bs4_") + layout.second;
        const std::string d6 = std::string("attn_decode_h256_gqa6_bs4_") + layout.second;
        attn_numeric_case(pl, p12.c_str(), 256, 12, 64, kv, 0, 1000, 4);
        attn_numeric_case(pl, d12.c_str(), 256, 12, 1, kv, 0, 1000, 4);
        attn_numeric_case(pl, d12.c_str(), 256, 12, 5, kv, 0, 2048, 4);
        attn_numeric_case(pl, p6.c_str(), 256, 6, 64, kv, 0, 1000, 4);
        attn_numeric_case(pl, d6.c_str(), 256, 6, 1, kv, 0, 1000, 4);
        attn_numeric_case(pl, d6.c_str(), 256, 6, 10, kv, 0, 2048, 4);
        if (kv[0] == 'b') {
            attn_numeric_case(pl, p12.c_str(), 256, 12, 100, kv, 0, 1024, 4);
            for (int rows : {2, 3, 4})
                attn_numeric_case(pl, d12.c_str(), 256, 12, rows, kv, 0, 1000, 4);
            attn_numeric_case(pl, p6.c_str(), 256, 6, 100, kv, 0, 1024, 4);
            for (int rows : {4, 8})
                attn_numeric_case(pl, d6.c_str(), 256, 6, rows, kv, 0, 1000, 4);
        }
    }
    for (int kvq : {1, 2, 4, 5})
        for (int rows : {3, 24}) {
            attn_gq_fold_case(pl, rows, false, false, kvq);
            if (kvq == 4 || kvq == 5)
                attn_gq_fold_case(pl, rows, false, false, kvq, false, true);
        }
"""

_DTYPE = "const uint32_t kvdt = kvq == 3 ? RAD_I8 : kv_is_fp8 ? RAD_F8E4M3 : RAD_BF16;"

PATCHES = (
    FilePatch(path="libr4d/r4d_selftest.cpp", language="none", description="2.2 Flash-Next packed KV cases", edits=(
        _edit("host-packed-layout", "/* Inline paged-attention oracle: a port of ref_attn_paged", _LAYOUTS,
              "static RadKVLayout selftest_packed_kv(", mode="insert_before"),
        _edit("oracle-row-size", """    const int64_t so = kvq == 3 ? 2 * head_dim + 4
                     : kvq == 2 ? 22 * head_dim / 16 : v4 ? 25 * head_dim / 16 : 2 * head_dim;""",
              "    const int64_t so = kvq_row_bytes(head_dim, kvq);",
              "const int64_t so = kvq_row_bytes(head_dim, kvq);"),
        _edit("oracle-bf16-key", "    const auto kslot = [&](int64_t blk, int64_t slot, int64_t d) {\n",
              """    const auto kslot = [&](int64_t blk, int64_t slot, int64_t d) {
        if (kvq == 4 || kvq == 5) {
            const uint8_t* row = kv + blk * sb + kh * sh + slot * so;
            uint16_t key;
            std::memcpy(&key, row + 2 * d, sizeof(key));
            return h_bf16(key);
        }
""", "std::memcpy(&key, row + 2 * d, sizeof(key));"),
        _edit("oracle-b16-value", "    const auto vslot = [&](int64_t blk, int64_t slot, int64_t d) {\n",
              """    const auto vslot = [&](int64_t blk, int64_t slot, int64_t d) {
        if (kvq == 4 || kvq == 5) {
            const uint8_t* row = kv + blk * sb + kh * sh + slot * so;
            if (kvq == 4) return h_e4m3(row[2 * head_dim + d]);
            return (float)(int8_t)row[2 * head_dim + d] * f16at(row + 3 * head_dim + 2 * (d / 32));
        }
""", "if (kvq == 4) return h_e4m3(row[2 * head_dim + d]);"),
        _edit("host-row-size", """    return kvq == 3 ? 2 * head_dim + 4
         : kvq == 2 ? 22 * head_dim / 16 : kvq == 1 ? 25 * head_dim / 16 : 2 * head_dim;""",
              """    const auto layout = selftest_packed_kv(kvq);
    if (layout != RAD_KVQ_NONE) return rad_kv_group_row_elems(layout, RAD_U8, head_dim);
    return kvq == 3 ? 2 * head_dim + 4 : 2 * head_dim;""",
              "return rad_kv_group_row_elems(layout, RAD_U8, head_dim);"),
        _edit("host-layout-selector", """         : !std::strcmp(kv_dtype, "i8") ? 3 : 0;""",
              """         : !std::strcmp(kv_dtype, "i8") ? 3 : !std::strcmp(kv_dtype, "bf16_fp8") ? 4
         : !std::strcmp(kv_dtype, "bf16_i8") ? 5 : 0;""", 'std::strcmp(kv_dtype, "bf16_i8") ? 5'),
        _edit("host-packed-values", "    for (int64_t r0 = 0; r0 < nelem; r0 += rowe) {\n        uint8_t* row = &kv[(size_t)r0];\n",
              _B16_FILL, "value == 0x80", mode="insert_after"),
        _edit("packed-tensor-dtype", _DTYPE, "const uint32_t kvdt = selftest_kv_dtype(kvq, kv_is_fp8);",
              "const uint32_t kvdt = selftest_kv_dtype(kvq, kv_is_fp8);", count=4),
        _edit("mixed-packed-tensor", "T(d_kv, RAD_I8, { ns * mb, kv_heads, bs, rowe }),",
              "T(d_kv, selftest_kv_dtype(kvq, true), { ns * mb, kv_heads, bs, rowe }),",
              "T(d_kv, selftest_kv_dtype(kvq, true)"),
        _edit("fold-selector", """                              int kvq = 0, bool rot = false) {
    const std::string kvs = bf16 ? "bf16kv" : kvq == 1 ? "fp8v4" : kvq == 2 ? "k6v4"
                          : kvq == 3 ? "i8" : "fp8kv";
    const std::string fold = (rot ? "attn_decode_ghq_h256_gqa12_" : "attn_decode_gq_h256_gqa12_")
                           + kvs;
    const std::string pair = "attn_decode_h256_gqa12_" + kvs;""",
              """                              int kvq = 0, bool rot = false, bool g6 = false) {
    const std::string kvs = bf16 ? "bf16kv" : kvq == 1 ? "fp8v4" : kvq == 2 ? "k6v4"
                          : kvq == 3 ? "i8" : kvq == 4 ? "b16f8" : kvq == 5 ? "b16i8" : "fp8kv";
    const std::string geometry = g6 ? "h256_gqa6_bs4_" : "h256_gqa12_";
    const std::string fold = (rot ? "attn_decode_ghq_" : "attn_decode_gq_") + geometry + kvs;
    const std::string pair = "attn_decode_" + geometry + kvs;""", "bool rot = false, bool g6 = false"),
        _edit("fold-heads", "const int64_t nq = 12, nkv = 1, hd = 256, bs = 4, nb = 96, maxb = 40;",
              "const int64_t nq = g6 ? 6 : 12, nkv = 1, hd = 256, bs = 4, nb = 96, maxb = 40;",
              "const int64_t nq = g6 ? 6 : 12, nkv = 1"),
        _edit("fold-packed-size", """    const int64_t rowb = kvq == 3 ? 2 * hd + 4 : kvq == 2 ? 22 * hd / 16 : kvq == 1 ? 25 * hd / 16
                                                                                      : 2 * hd;""",
              "    const int64_t rowb = kvq_row_bytes(hd, kvq);", "const int64_t rowb = kvq_row_bytes(hd, kvq)"),
        _edit("fold-packed-bytes", """        if (kvq == 3) {
            /* i8: random codes without -128""",
              """        if (kvq == 4 || kvq == 5) {
            kvq_fill_rows(hc, (int64_t)cbytes, rowb, hd, kvq);
        } else if (kvq == 3) {
            /* i8: random codes without -128""", "kvq_fill_rows(hc, (int64_t)cbytes"),
        _edit("fold-packed-type", "T(d_c, bf16 ? RAD_BF16 : RAD_F8E4M3, { nb, nkv, bs, rowb }),",
              "T(d_c, selftest_packed_kv(kvq) != RAD_KVQ_NONE ? RAD_U8 : bf16 ? RAD_BF16 : RAD_F8E4M3,"
              " { nb, nkv, bs, rowb }),",
              "T(d_c, selftest_packed_kv(kvq) != RAD_KVQ_NONE"),
        _edit("fold-params", """        PS("kv_dtype", bf16 ? "bf16" : kvq == 1 ? "fp8_v4" : kvq == 2 ? "k6_v4"
                                    : kvq == 3 ? "i8" : "fp8_e4m3"),""",
              """        PS("kv_dtype", selftest_packed_kv(kvq) != RAD_KVQ_NONE ?
                           rad_kvq_desc(selftest_packed_kv(kvq))->name :
                           bf16 ? "bf16" : kvq == 3 ? "i8" : "fp8_e4m3"),""",
              'PS("kv_dtype", selftest_packed_kv(kvq) != RAD_KVQ_NONE'),
        _edit("packed-inventory", "/* --------------------------- the attention with the output gate and quantiser",
              _ROW_INVENTORY,
              "static void bf16_i8_rows_case(", mode="insert_before"),
        _edit("packed-split", '        { &kSplitG6, "attn_prefill_h256_gqa6_i8", "i8" },\n',
              _matrix(), '        { &kSplitG12, "attn_prefill_h256_gqa12_b16f8", "bf16_fp8" },',
              mode="insert_before"),
        _edit("packed-mixed", '        { &kSplitG6, "attn_prefill_h256_gqa6_i8", "attn_decode_h256_gqa6_i8", "i8" },\n',
              _matrix(True), '        { &kSplitG12, "attn_prefill_h256_gqa12_b16f8", "attn_decode_h256_gqa12_b16f8",',
              mode="insert_before"),
        _edit("packed-numeric", "    attn_decode_scratch_band_case(pl);\n", _NUMERIC,
              "    bf16_i8_rows_case(pl);", mode="insert_after"),
    )),
    FilePatch(path="tests/bc_kv_layout_diagnostic_test.c", language="none", create=True,
              description="ABI44 host descriptor test", edits=(Edit(
        id="abi44-packed-fixture", anchor=r"\A", guard="BC_RUM07_ABI44_TEST", max_span_lines=1,
        text=r"""/* BC_RUM07_ABI44_TEST: representation and bounds only, no device activation. */
#include "rad_builder.h"
#undef NDEBUG
#include <assert.h>
#include <string.h>

int main(void) {
    assert(RAD_ABI_VERSION == 44);
    assert(rad_kv_group_row_elems(RAD_KVQ_NONE, RAD_F8E4M3, 256) == 512);
    assert(rad_kv_group_row_elems(RAD_KVQ_NONE, RAD_I8, 256) == 516);
    struct Case {
        RadKVLayout layout;
        const char* name;
        int row, value, key_scale, value_scale, group;
    } cases[] = {
        { RAD_KVQ_V4, "fp8_v4", 400, 256, 0, 384, 32 },
        { RAD_KVQ_K6V4, "k6_v4", 352, 192, 320, 336, 32 },
        { RAD_KVQ_B16F8, "bf16_fp8", 768, 512, 0, 0, 0 },
        { RAD_KVQ_B16I8, "bf16_i8", 784, 512, 0, 768, 32 },
    };
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const struct Case* c = &cases[i];
        const RadKVLayoutDesc* d = rad_kvq_desc(c->layout);
        assert(d && !strcmp(c->name, d->name) && rad_kvq_from_name(c->name) == c->layout);
        assert(d->row16 * 16 == c->row && d->value16 * 16 == c->value);
        assert(d->key_scale16 * 16 == c->key_scale && d->value_scale16 * 16 == c->value_scale);
        assert(d->scale_group == c->group);
        assert(rad_kv_group_row_elems(c->layout, RAD_U8, 256) == c->row);
        assert(!rad_kv_group_row_elems(c->layout, RAD_I8, 256));
        assert(!rad_kv_group_row_elems(c->layout, RAD_U8, 255));
    }
    assert(!rad_kvq_desc(RAD_KVQ_COUNT));
    assert(rad_kvq_from_name("i8") == RAD_KVQ_NONE);
    assert(rad_kvq_from_name("fp8") == RAD_KVQ_NONE);
    return 0;
}
""", rationale="Card-free fixture for exact ABI44 packed rows",
    ),)),
    FilePatch(path="tests/CMakeLists.txt", language="none", description="Register ABI44 descriptor CTest",
              edits=(Edit(
        id="register-abi44-packed-fixture", anchor=r"\Z", guard=r"add_executable\(bc_kv_layout_diagnostic_test", max_span_lines=1,
        text="""\nadd_executable(bc_kv_layout_diagnostic_test bc_kv_layout_diagnostic_test.c)
target_include_directories(bc_kv_layout_diagnostic_test PRIVATE ${CMAKE_SOURCE_DIR}/abi)
add_test(NAME bc_kv_layout_diagnostic_test COMMAND bc_kv_layout_diagnostic_test)
set_tests_properties(bc_kv_layout_diagnostic_test PROPERTIES LABELS "unit;fast")
""", rationale="Register the diagnostic host fixture alongside upstream tests",
    ),)),
)
