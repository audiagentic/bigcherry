"""rad0042 (VTP44 patch B2): r4d_selftest cases for bf16_fp8 and bf16_i8 on Flash-Next's geometry. The gqa12 and gqa6
block-4 numeric cases (ragged prefill tile, decode at every wave count), the split-KV prefill and mixed-step cases and the fused
gate+quantiser byte-identity case (bf16 K, finite E4M3 or int8 V and f16 group scales) take the two layouts.
"""
from __future__ import annotations

import re
from bigcherry.patcher import Edit, FilePatch


def _edit(id: str, before: str, after: str, guard: str) -> Edit:
    span = max(4, before.count("\n") + 2)
    return Edit(id=id, anchor=re.escape(before), mode="replace", text=after,
                guard=re.escape(guard), expect_matches=1, max_span_lines=span,
                rationale=("long-anchor: the whole diff hunk is the anchor; " if span > 6 else "")
                + "VTP44 B2: bf16 packed KV selftest on Flash-Next")


PATCHES = (
    FilePatch(path='libr4d/r4d_selftest.cpp', language="none", description='VTP44 B2: bf16 packed KV selftest on Flash-Next', edits=(
        _edit('r4d_selftest_cpp-0', ' * finite f16 group scales. */\nstatic void attn_gq_fold_case(const Plugin& pl, int64_t S, bool bf16, bool i8 = false,\n                              int kvq = 0) {\n    const std::string kvs = bf16 ? "bf16kv" : kvq == 1 ? "fp8v4" : kvq == 2 ? "k6v4" : "fp8kv";\n    const std::string fold = "attn_decode_gq_h256_gqa12_" + kvs;\n    const std::string pair = "attn_decode_h256_gqa12_" + kvs;\n    if (!sel(fold.c_str())) return;\n', ' * finite f16 group scales. */\nstatic void attn_gq_fold_case(const Plugin& pl, int64_t S, bool bf16, bool i8 = false,\n                              int kvq = 0, bool g6 = false) {\n    const std::string kvs = bf16 ? "bf16kv" : kvq == 1 ? "fp8v4" : kvq == 2 ? "k6v4"\n                          : kvq == 3 ? "b16f8" : kvq == 4 ? "b16i8" : "fp8kv";\n    const std::string geo = g6 ? "gqa6_bs4_" : "gqa12_";   /* BC_VTP44_FOLD_GEO: four ranks six queries per KV head, or twelve */\n    const std::string fold = "attn_decode_gq_h256_" + geo + kvs;\n    const std::string pair = "attn_decode_h256_" + geo + kvs;\n    if (!sel(fold.c_str())) return;\n', ': kvq == 3 ? "b16f8" : kvq == 4 ? "b16i8" : "fp8kv";'),
        _edit('r4d_selftest_cpp-1', "    for (auto& x : hqg) x = h_to_bf16(rnd(6.0f));\n    /* A token's row in a page: 2 * head_dim elements, or the v4 layouts' 25 / 22 * head_dim / 16\n     * bytes (RadBuildCtx::kv_quant). */\n    const int64_t rowb = kvq == 2 ? 22 * hd / 16 : kvq == 1 ? 25 * hd / 16 : 2 * hd;\n    const size_t cbytes = (size_t)(nb * nkv * bs * rowb) * (bf16 ? 2 : 1);\n    std::vector<uint8_t> hc(cbytes);\n    if (bf16) {\n", "    for (auto& x : hqg) x = h_to_bf16(rnd(6.0f));\n    /* A token's row in a page: 2 * head_dim elements, or the v4 layouts' 25 / 22 * head_dim / 16\n     * bytes (RadBuildCtx::kv_quant). */\n    const int64_t rowb = kvq == 4 ? RAD_KVQ_B16I8_ROW16 * hd / 16 : kvq == 3 ? RAD_KVQ_B16F8_ROW16 * hd / 16\n                       : kvq == 2 ? 22 * hd / 16 : kvq == 1 ? 25 * hd / 16 : 2 * hd;\n    const size_t cbytes = (size_t)(nb * nkv * bs * rowb) * (bf16 ? 2 : 1);\n    std::vector<uint8_t> hc(cbytes);\n    if (bf16) {\n", 'const int64_t rowb = kvq == 4 ? RAD_KVQ_B16I8_ROW16 * hd / 16 : kvq == 3 ? RAD_KVQ_B16F8_ROW16 * hd / 16'),
        _edit('r4d_selftest_cpp-2', "        }\n        /* The group scales: K's (k6_v4) and V's, finite and positive. */\n        const int64_t vs = kvq == 2 ? hd * 21 / 16 : hd * 3 / 2, ks = hd * 5 / 4;\n        if (kvq)\n            for (size_t r0 = 0; r0 < cbytes; r0 += (size_t)rowb)\n                for (int64_t g = 0; g < hd / 32; ++g) {\n                    const _Float16 v = (_Float16)(0.03125f * (1.0f + (rnd(1.0f) + 1.0f) * 2.0f));\n", "        }\n        /* The group scales: K's (k6_v4) and V's, finite and positive. */\n        const int64_t vs = kvq == 2 ? hd * 21 / 16 : hd * 3 / 2, ks = hd * 5 / 4;\n        /* bf16_fp8 and bf16_i8: K is bf16 over [0, 2HD) and must be finite; V's bytes follow (int8\n         * codes stay in -127..127), then, for bf16_i8, an f16 scale per group at 3HD (finite, positive). */\n        if (kvq >= 3)\n            for (size_t r0 = 0; r0 < cbytes; r0 += (size_t)rowb) {\n                for (int64_t d = 0; d < hd; ++d) {\n                    const uint16_t h = h_to_bf16(rnd(3.0f));\n                    std::memcpy(&hc[r0 + (size_t)(2 * d)], &h, 2);\n                }\n                if (kvq == 4) {\n                    for (int64_t d = 0; d < hd; ++d)\n                        if (hc[r0 + (size_t)(2 * hd + d)] == 0x80) hc[r0 + (size_t)(2 * hd + d)] = 0x81;\n                    for (int64_t g = 0; g < hd / RAD_KVQ_B16I8_GROUP; ++g) {\n                        const _Float16 v = (_Float16)(0.03125f * (1.0f + (rnd(1.0f) + 1.0f) * 2.0f));\n                        std::memcpy(&hc[r0 + (size_t)(3 * hd + g * 2)], &v, 2);\n                    }\n                }\n            }\n        if (kvq == 1 || kvq == 2)\n            for (size_t r0 = 0; r0 < cbytes; r0 += (size_t)rowb)\n                for (int64_t g = 0; g < hd / 32; ++g) {\n                    const _Float16 v = (_Float16)(0.03125f * (1.0f + (rnd(1.0f) + 1.0f) * 2.0f));\n", "/* bf16_fp8 and bf16_i8: K is bf16 over [0, 2HD) and must be finite; V's bytes follow (int8"),
        _edit('r4d_selftest_cpp-3', '    std::vector<RadParam> pa = {\n        PI("q_len", 1), PI("head_dim", hd), PI("gqa", nq / nkv), PI("block_size", bs),\n        PI("causal", 1), PI("window", 0), PS("q_dtype", "bf16"),\n        PS("kv_dtype", bf16 ? "bf16" : kvq == 1 ? "fp8_v4" : kvq == 2 ? "k6_v4" : "fp8_e4m3"),\n        PI("n_head", nq), PI("max_seqs", S), PI("max_ctx", maxb * bs),\n    };\n\n', '    std::vector<RadParam> pa = {\n        PI("q_len", 1), PI("head_dim", hd), PI("gqa", nq / nkv), PI("block_size", bs),\n        PI("causal", 1), PI("window", 0), PS("q_dtype", "bf16"),\n        PS("kv_dtype", bf16 ? "bf16" : kvq == 1 ? "fp8_v4" : kvq == 2 ? "k6_v4"\n                       : kvq == 3 ? "bf16_fp8" : kvq == 4 ? "bf16_i8" : "fp8_e4m3"),\n        PI("n_head", nq), PI("max_seqs", S), PI("max_ctx", maxb * bs),\n    };\n\n', ': kvq == 3 ? "bf16_fp8" : kvq == 4 ? "bf16_i8" : "fp8_e4m3"),'),
        _edit('r4d_selftest_cpp-4', '        { &kSplitG6Bs4, "attn_prefill_h256_gqa6_bs4_fp8v4", "fp8_v4" },\n        { &kSplitG6Bs4, "attn_prefill_h256_gqa6_bs4_fp8kv", "fp8_e4m3" },\n        { &kSplitG6Bs4, "attn_prefill_h256_gqa6_bs4_bf16kv", "bf16" },\n        { &kSplitG8, "attn_prefill_h256_gqa8_fp8kv", "fp8_e4m3" },\n        { &kSplitG8, "attn_prefill_h256_gqa8_bf16kv", "bf16" },\n        { &kSplitG4, "attn_prefill_h256_gqa4_fp8kv", "fp8_e4m3" },\n', '        { &kSplitG6Bs4, "attn_prefill_h256_gqa6_bs4_fp8v4", "fp8_v4" },\n        { &kSplitG6Bs4, "attn_prefill_h256_gqa6_bs4_fp8kv", "fp8_e4m3" },\n        { &kSplitG6Bs4, "attn_prefill_h256_gqa6_bs4_bf16kv", "bf16" },\n        { &kSplitG12, "attn_prefill_h256_gqa12_b16f8", "bf16_fp8" },\n        { &kSplitG12, "attn_prefill_h256_gqa12_b16i8", "bf16_i8" },\n        { &kSplitG6Bs4, "attn_prefill_h256_gqa6_bs4_b16f8", "bf16_fp8" },\n        { &kSplitG6Bs4, "attn_prefill_h256_gqa6_bs4_b16i8", "bf16_i8" },\n        { &kSplitG8, "attn_prefill_h256_gqa8_fp8kv", "fp8_e4m3" },\n        { &kSplitG8, "attn_prefill_h256_gqa8_bf16kv", "bf16" },\n        { &kSplitG4, "attn_prefill_h256_gqa4_fp8kv", "fp8_e4m3" },\n', '{ &kSplitG12, "attn_prefill_h256_gqa12_b16f8", "bf16_fp8" },'),
        _edit('r4d_selftest_cpp-5', '        { &kSplitG12, "attn_prefill_h256_gqa12_k6v4", "attn_decode_h256_gqa12_k6v4", "k6_v4" },\n        { &kSplitG12, "attn_prefill_h256_gqa12_fp8v4", "attn_decode_h256_gqa12_fp8v4", "fp8_v4" },\n        { &kSplitG12, "attn_prefill_h256_gqa12_fp8kv", "attn_decode_h256_gqa12_fp8kv", "fp8_e4m3" },\n        { &kSplitG6Bs4, "attn_prefill_h256_gqa6_bs4_k6v4", "attn_decode_h256_gqa6_bs4_k6v4", "k6_v4" },\n        { &kSplitG8, "attn_prefill_h256_gqa8_fp8kv", "attn_decode_h256_gqa8_fp8kv", "fp8_e4m3" },\n        { &kSplitG4, "attn_prefill_h256_gqa4_fp8kv", "attn_decode_h256_gqa4_fp8kv", "fp8_e4m3" },\n        { &kSplitG2Kv2, "attn_prefill_h256_gqa2_fp8kv", "attn_decode_h256_gqa2_fp8kv", "fp8_e4m3" },\n', '        { &kSplitG12, "attn_prefill_h256_gqa12_k6v4", "attn_decode_h256_gqa12_k6v4", "k6_v4" },\n        { &kSplitG12, "attn_prefill_h256_gqa12_fp8v4", "attn_decode_h256_gqa12_fp8v4", "fp8_v4" },\n        { &kSplitG12, "attn_prefill_h256_gqa12_fp8kv", "attn_decode_h256_gqa12_fp8kv", "fp8_e4m3" },\n        { &kSplitG12, "attn_prefill_h256_gqa12_b16f8", "attn_decode_h256_gqa12_b16f8", "bf16_fp8" },\n        { &kSplitG12, "attn_prefill_h256_gqa12_b16i8", "attn_decode_h256_gqa12_b16i8", "bf16_i8" },\n        { &kSplitG6Bs4, "attn_prefill_h256_gqa6_bs4_k6v4", "attn_decode_h256_gqa6_bs4_k6v4", "k6_v4" },\n        { &kSplitG6Bs4, "attn_prefill_h256_gqa6_bs4_b16f8", "attn_decode_h256_gqa6_bs4_b16f8", "bf16_fp8" },\n        { &kSplitG6Bs4, "attn_prefill_h256_gqa6_bs4_b16i8", "attn_decode_h256_gqa6_bs4_b16i8", "bf16_i8" },\n        { &kSplitG8, "attn_prefill_h256_gqa8_fp8kv", "attn_decode_h256_gqa8_fp8kv", "fp8_e4m3" },\n        { &kSplitG4, "attn_prefill_h256_gqa4_fp8kv", "attn_decode_h256_gqa4_fp8kv", "fp8_e4m3" },\n        { &kSplitG2Kv2, "attn_prefill_h256_gqa2_fp8kv", "attn_decode_h256_gqa2_fp8kv", "fp8_e4m3" },\n', '{ &kSplitG12, "attn_prefill_h256_gqa12_b16f8", "attn_decode_h256_gqa12_b16f8", "bf16_fp8" },'),
        _edit('r4d_selftest_cpp-6', '    /* Flash-Next\'s QSA layers: twelve queries per KV head over 4-token pages, and six at\n     * four ranks, each layout\'s prefill (a long history and a ragged last tile), decode, and the\n     * widest verify (q_len 5 at gqa 12, 10 at gqa 6). */\n    for (const char* kv : { "fp8_v4", "k6_v4" }) {\n        const std::string t = kv[0] == \'f\' ? "fp8v4" : "k6v4";\n        const std::string p12 = "attn_prefill_h256_gqa12_" + t, d12 = "attn_decode_h256_gqa12_" + t;\n        const std::string p6 = "attn_prefill_h256_gqa6_bs4_" + t, d6 = "attn_decode_h256_gqa6_bs4_" + t;\n        attn_numeric_case(pl, p12.c_str(), 256, 12, 64, kv, 0, 1000, 4);\n', '    /* Flash-Next\'s QSA layers: twelve queries per KV head over 4-token pages, and six at\n     * four ranks, each layout\'s prefill (a long history and a ragged last tile), decode, and the\n     * widest verify (q_len 5 at gqa 12, 10 at gqa 6). */\n    /* bf16_fp8 and bf16_i8 (rad0041) take the same shapes, and the decode at each wave count the\n     * gqa12 tile has (1 to 5 rows: 1 to 4 waves), as the single-wave variants spill the most. */\n    for (const auto& lay : { std::make_pair("fp8_v4", "fp8v4"), std::make_pair("k6_v4", "k6v4"),\n                             std::make_pair("bf16_fp8", "b16f8"), std::make_pair("bf16_i8", "b16i8") }) {\n        const char* kv = lay.first;\n        const std::string t = lay.second;\n        const bool b16 = kv[0] == \'b\';\n        const std::string p12 = "attn_prefill_h256_gqa12_" + t, d12 = "attn_decode_h256_gqa12_" + t;\n        const std::string p6 = "attn_prefill_h256_gqa6_bs4_" + t, d6 = "attn_decode_h256_gqa6_bs4_" + t;\n        attn_numeric_case(pl, p12.c_str(), 256, 12, 64, kv, 0, 1000, 4);\n', '/* bf16_fp8 and bf16_i8 (rad0041) take the same shapes, and the decode at each wave count the'),
        _edit('r4d_selftest_cpp-7', "        attn_numeric_case(pl, p6.c_str(),  256, 6,  64, kv, 0, 1000, 4);\n        attn_numeric_case(pl, d6.c_str(),  256, 6,  1,  kv, 0, 1000, 4);\n        attn_numeric_case(pl, d6.c_str(),  256, 6,  10, kv, 0, 2048, 4);\n    }\n    /* A ctx past a whole block (a partial last tile), and q_len 7: three waves, where the V\n     * staging's unit count is not a multiple of the threads. */\n", "        attn_numeric_case(pl, p6.c_str(),  256, 6,  64, kv, 0, 1000, 4);\n        attn_numeric_case(pl, d6.c_str(),  256, 6,  1,  kv, 0, 1000, 4);\n        attn_numeric_case(pl, d6.c_str(),  256, 6,  10, kv, 0, 2048, 4);\n        if (b16) {\n            attn_numeric_case(pl, p12.c_str(), 256, 12, 100, kv, 0, 1024, 4);   /* a ragged last tile */\n            attn_numeric_case(pl, d12.c_str(), 256, 12, 2,   kv, 0, 1000, 4);\n            attn_numeric_case(pl, d12.c_str(), 256, 12, 3,   kv, 0, 1000, 4);\n            attn_numeric_case(pl, d12.c_str(), 256, 12, 4,   kv, 0, 1000, 4);\n            attn_numeric_case(pl, p6.c_str(),  256, 6,  100, kv, 0, 1024, 4);\n            attn_numeric_case(pl, d6.c_str(),  256, 6,  4,   kv, 0, 1000, 4);\n            attn_numeric_case(pl, d6.c_str(),  256, 6,  8,   kv, 0, 1000, 4);\n        }\n    }\n    /* A ctx past a whole block (a partial last tile), and q_len 7: three waves, where the V\n     * staging's unit count is not a multiple of the threads. */\n", 'if (b16) {'),
        _edit('r4d_selftest_cpp-8', '    attn_numeric_case(pl, "attn_decode_h128_gqa2_bf16kv",  128, 2, 24, "bf16",     0, 256);\n    attn_gq_fold_case(pl, 3, false);\n    attn_gq_fold_case(pl, 24, false);\n    /* And over the fp8_v4 and k6_v4 caches (Flash-Next\'s --kv-cache-dtype fp8_v4 / k6_v4). */\n    for (int kvq : { 1, 2 }) {\n        attn_gq_fold_case(pl, 3, false, false, kvq);\n        attn_gq_fold_case(pl, 24, false, false, kvq);\n    }\n', '    attn_numeric_case(pl, "attn_decode_h128_gqa2_bf16kv",  128, 2, 24, "bf16",     0, 256);\n    attn_gq_fold_case(pl, 3, false);\n    attn_gq_fold_case(pl, 24, false);\n    /* And over the fp8_v4, k6_v4, bf16_fp8 and bf16_i8 caches (Flash-Next\'s --kv-cache-dtype). */\n    for (int kvq : { 1, 2, 3, 4 }) {\n        attn_gq_fold_case(pl, 3, false, false, kvq);\n        attn_gq_fold_case(pl, 24, false, false, kvq);\n    }\n    /* The packed layouts also at four ranks geometry: six queries per KV head over 4-token pages. */\n    for (int kvq : { 3, 4 }) {\n        attn_gq_fold_case(pl, 3, false, false, kvq, true);\n        attn_gq_fold_case(pl, 24, false, false, kvq, true);\n    }\n', "/* And over the fp8_v4, k6_v4, bf16_fp8 and bf16_i8 caches (Flash-Next's --kv-cache-dtype). */"),

        _edit('r4d_selftest_cpp-9', "    const int64_t nq = 12, nkv = 1, hd = 256, bs = 4, nb = 96, maxb = 40;\n",
              "    const int64_t nq = g6 ? 6 : 12, nkv = 1, hd = 256, bs = 4, nb = 96, maxb = 40;   /* BC_VTP44_FOLD_NQ */\n",
              "BC_VTP44_FOLD_NQ"),
        _edit('r4d_selftest_cpp-10',
              '    static const char* const want[] = { "attn_prefill_h256_gqa6_b16i8", "attn_decode_h256_gqa6_b16i8",\n'
              '                                        "kv_store_b16i8", "qk_norm_rope_gate_kv_store_b16i8" };\n',
              "    /* BC_VTP44_B16I8_ROWS: every row the layout has and the geometry each pins: the gqa 6 / 16-token-page rows of the 27B,\n"
              "     * the stores, and the QSA rows of Flash-Next (gqa 12 over 4-token pages, and gqa 6 at four ranks). gqa 0: no attention pin. */\n"
              "    struct Want { const char* name; long long gqa, block; };\n"
              "    static const Want want[] = {\n"
              '        { "attn_prefill_h256_gqa6_b16i8", 6, 16 }, { "attn_decode_h256_gqa6_b16i8", 6, 16 },\n'
              '        { "kv_store_b16i8", 0, 0 }, { "qk_norm_rope_gate_kv_store_b16i8", 0, 0 },\n'
              '        { "attn_prefill_h256_gqa12_b16i8", 12, 4 }, { "attn_decode_h256_gqa12_b16i8", 12, 4 },\n'
              '        { "attn_decode_gq_h256_gqa12_b16i8", 12, 4 },\n'
              '        { "attn_prefill_h256_gqa6_bs4_b16i8", 6, 4 }, { "attn_decode_h256_gqa6_bs4_b16i8", 6, 4 },\n'
              '        { "attn_decode_gq_h256_gqa6_bs4_b16i8", 6, 4 } };\n',
              "BC_VTP44_B16I8_ROWS"),
        _edit('r4d_selftest_cpp-11',
              "        bool known = false;\n        for (const char* w : want) known = known || !std::strcmp(w, r->name);\n",
              "        const Want* wr = nullptr;   /* BC_VTP44_B16I8_KNOWN */\n        for (const Want& w : want) if (!std::strcmp(w.name, r->name)) wr = &w;\n        const bool known = wr != nullptr;\n",
              "BC_VTP44_B16I8_KNOWN"),
        _edit('r4d_selftest_cpp-12',
              '        if (!std::strcmp(r->op, "attn_paged") &&\n'
              '            !(eq(r, "head_dim", 256) && eq(r, "gqa", 6) && eq(r, "block_size", 16) && eq(r, "window", 0)))\n'
              '            why += std::string("; ") + r->name + " does not pin head_dim 256, gqa 6, block 16, no window";\n',
              '        if (!std::strcmp(r->op, "attn_paged") && wr &&   /* BC_VTP44_B16I8_PIN */\n'
              '            !(eq(r, "head_dim", 256) && eq(r, "gqa", wr->gqa) && eq(r, "block_size", wr->block) && eq(r, "window", 0)))\n'
              '            why += std::string("; ") + r->name + " does not pin head_dim 256, its gqa and block size, no window";\n',
              "BC_VTP44_B16I8_PIN"),
    )),
)


# RUM07: target-only ABI43 card-free contract fixture. This does not imply a
# functional GPU reader, fused-store or a numerically qualified hybrid cache.
_ABI43_CPU_FIXTURE = r"""/* BC_RUM07_ABI43_TEST -- KV row bytes and refusal boundaries without cards.
 * This tests the descriptor, NOT hardware store/attention correctness.
 */
#include "rad_builder.h"
#include <assert.h>
#include <string.h>

int main(void) {
    const int64_t hd = 256;
    assert(RAD_ABI_VERSION == 43);
    assert(rad_kv_row_elems(RAD_F8E4M3, hd) == 512);
    assert(rad_kv_row_elems(RAD_I8, hd) == 516);
    assert(rad_kvq_row_bytes(RAD_KVQ_B16F8, hd) == 768);
    assert(rad_kvq_row_bytes(RAD_KVQ_B16I8, hd) == 784);
    assert(rad_kvq_value_offset(RAD_KVQ_B16F8, hd) == 512);
    assert(rad_kvq_value_offset(RAD_KVQ_B16I8, hd) == 512);
    assert(rad_kvq_scale_offset(RAD_KVQ_B16F8, hd) == -1);
    assert(rad_kvq_scale_offset(RAD_KVQ_B16I8, hd) == 768);
    assert(rad_kvq_row16(RAD_KVQ_B16F8) == 48);
    assert(rad_kvq_row16(RAD_KVQ_B16I8) == 49);
    assert(strcmp(rad_kvq_name(RAD_KVQ_B16F8), "bf16_fp8") == 0);
    assert(strcmp(rad_kvq_name(RAD_KVQ_B16I8), "bf16_i8") == 0);
    assert(rad_kvq_from_name("bf16_fp8") == RAD_KVQ_B16F8);
    assert(rad_kvq_from_name("bf16_i8") == RAD_KVQ_B16I8);
    assert(rad_kvq_from_name("i8") == RAD_KVQ_NONE);
    assert(rad_kvq_from_name("fp8") == RAD_KVQ_NONE);
    assert(rad_kvq_from_name("fp8_v4") == RAD_KVQ_NONE);
    assert(rad_kvq_geometry_ok(RAD_KVQ_B16F8, hd, 12, 4));
    assert(rad_kvq_geometry_ok(RAD_KVQ_B16I8, hd, 12, 4));
    assert(!rad_kvq_geometry_ok(RAD_KVQ_B16F8, hd, 6, 4));
    assert(!rad_kvq_geometry_ok(RAD_KVQ_B16I8, hd, 12, 16));
    assert(!rad_kvq_geometry_ok(RAD_KVQ_B16F8, 128, 12, 4));
    assert(rad_kvq_row_bytes(RAD_KVQ_NONE, hd) == 0);
    assert(rad_kvq_row_bytes(RAD_KVQ_B16F8, 128) == 0);
    return 0;
}
"""
PATCHES += (
    FilePatch(
        path="tests/bc_kv_layout_contract_test.c",
        language="none", create=True,
        description="RUM07 ABI43: card-free local hybrid row-format and refusal fixture",
        edits=(Edit(
            id="rum07-abi43-cpu-contract-fixture",
            anchor=r"\A", mode="insert_after", text=_ABI43_CPU_FIXTURE,
            guard="BC_RUM07_ABI43_TEST",
            rationale="New CPU contract fixture: no GPU runtime or kernel activation",
        ),),
    ),
    FilePatch(
        path="tests/CMakeLists.txt", language="none",
        description="RUM07 ABI43: register card-free KV descriptor contract CTest",
        edits=(Edit(
            id="rum07-abi43-register-cpu-contract",
            anchor=r"\Z", mode="insert_after",
            text="""\n# RUM07 ABI43: descriptor-level CPU test; kernel and model proof remain separate.
# Keep the v1.17 pin's test build unchanged. This fixture uses the ABI43
# descriptor and runs only against the exact reviewed checkpoint's ABI.
file(STRINGS "${CMAKE_SOURCE_DIR}/abi/rad_abi.h" BC_RUM07_ABI43_LINE
     REGEX "^#define RAD_ABI_VERSION 43u$")
if(BC_RUM07_ABI43_LINE)
  add_executable(bc_kv_layout_contract_test bc_kv_layout_contract_test.c)
  target_include_directories(bc_kv_layout_contract_test PRIVATE ${CMAKE_SOURCE_DIR}/abi)
  add_test(NAME bc_kv_layout_contract_test COMMAND bc_kv_layout_contract_test)
  set_tests_properties(bc_kv_layout_contract_test PROPERTIES LABELS "unit;fast")
endif()
""",
            guard=r"add_executable\(bc_kv_layout_contract_test",
            rationale="Keep native KV contract rejection tested on CPU with no hardware",
        ),),
    ),
)
