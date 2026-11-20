"""Restore native E4M3 on the frozen Qwen4exp H256 page-4 attention paths."""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch


def _edit(edit_id: str, before: str, text: str, guard: str, *, mode: str = "replace") -> Edit:
    return Edit(
        id=edit_id,
        anchor=re.escape(before),
        mode=mode,
        text=text,
        guard=re.escape(guard),
        max_span_lines=before.count("\n") + 1,
        rationale="Restore native E4M3 KV for the Flash-Next page-4 attention geometries",
    )


_GEOMETRY = """static bool native_e4m3_geometry(int64_t head_dim, int64_t n_head, int64_t n_head_kv) {
    if (head_dim != 256 || n_head_kv <= 0 || n_head % n_head_kv != 0) return false;
    const int64_t gqa = n_head / n_head_kv;
    return gqa == 6 || gqa == 12;
}

"""

_ARCH_TEST = r"""TEST(qwen4exp_native_e4m3_uses_the_page4_attention_and_shared_mtp_group) {
    RadModelMeta meta = qwen4exp_meta();
    RadArchProbe probe{};
    REQUIRE_EQ(qwen4exp_fp8::probe(&meta, &probe), RAD_OK);
    CHECK_EQ(probe.kv_fp8_ok, 1);
    meta.head_dim = 128;
    REQUIRE_EQ(qwen4exp_fp8::probe(&meta, &probe), RAD_OK);
    CHECK_EQ(probe.kv_fp8_ok, 0);

    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 64; c.max_seqs = 2; c.max_spec = 3; c.max_ctx = 4096;
    c.kv_dtype = RAD_F8E4M3;
    RadBuilder bad;
    CHECK_EQ(stderr_of_rc([&] { return qwen4exp_fp8::declare(&bad, &meta, &c); }), RAD_E_UNSUPPORTED);
    CHECK(bad.kv_groups.empty());

    meta = qwen4exp_meta();
    for (const int world : {1, 2, 4}) {
        const int64_t gqa = world == 4 ? 6 : 12;
        const int64_t local_kv_heads = world == 1 ? 2 : 1;
        c.world_size = world;
        RadBuilder b;
        b.kv_block = 4;
        REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
        const auto& m = qwen4exp_fp8::g_model[0];
        REQUIRE(m.kv_full > 0);
        REQUIRE(m.kv_qsa_bkey > 0);
        CHECK(m.kv_full != m.kv_qsa_bkey);
        const auto& cache = b.kv_groups[m.kv_full - 1];
        CHECK_EQ(cache.d.dtype, (uint32_t)RAD_F8E4M3);
        CHECK_EQ(cache.d.kv_layout, (RadKVLayout)RAD_KVQ_NONE);
        CHECK_EQ(cache.d.head_dim, 256);
        CHECK_EQ(cache.d.n_head_kv, local_kv_heads);
        CHECK_EQ(rad_kv_layer_bytes(&b, m.kv_full), local_kv_heads * 4 * 512);
        CHECK_EQ(b.kv_groups[m.kv_qsa_bkey - 1].d.dtype, (uint32_t)RAD_BF16);
        CHECK(std::find(b.binds.begin(), b.binds.end(),
                        std::make_pair((int)meta.n_layers, m.kv_full)) != b.binds.end());
        CHECK(std::count_if(b.binds.begin(), b.binds.end(),
                            [&](const auto& p) { return p.second == m.kv_full; }) > 1);

        int readers = 0;
        int writers = 0;
        int gates = 0;
        for (const RecOp& o : b.ops) {
            const bool reads_cache = o.op == "attn_paged";
            const bool gate = o.op == "attn_paged_gate_quant" || o.op == "attn_paged_gate_had_quant";
            const bool writes_cache = o.op == "kv_store";
            if (!reads_cache && !gate && !writes_cache) continue;
            const RecParam* kv_dtype = nullptr;
            const RecParam* block_size = nullptr;
            const RecParam* ratio = nullptr;
            for (const RecParam& p : o.p) {
                if (p.key == "kv_dtype") kv_dtype = &p;
                if (p.key == "block_size") block_size = &p;
                if (p.key == "gqa") ratio = &p;
            }
            REQUIRE(kv_dtype && block_size);
            CHECK_EQ(kv_dtype->sval, std::string("fp8_e4m3"));
            CHECK_EQ(block_size->ival, 4);
            if (!writes_cache) {
                REQUIRE(ratio != nullptr);
                CHECK_EQ(ratio->ival, gqa);
            }
            readers += reads_cache;
            writers += writes_cache;
            gates += gate;
        }
        CHECK(readers >= 2);
        CHECK_EQ(writers, readers);
        CHECK(gates >= 2);
    }
}

"""

_BYTE_LAUNCH = r"""/* The byte caches share the split/merge shims. E4M3 uses the native
 * two-head-width row and the plain decode path, including for QSA selection. */
static bool g8_e4m3_cache(const RadArgs* a) {
    const RadTensor* kv = r4d_opt(a, AP_KV);
    return kv && kv->dtype == RAD_F8E4M3;
}
#define G8_BYTE_ENTRIES(TAG, ROWB, XOPT)"""
_FIRST_BYTE_LAUNCH = """    R4DArgs r;                                                                                 \\
    const int rc = fill_paged_row(a, &r, 256, ROWB);                                           \\
"""
_BYTE_DECODE_LAUNCH = """    R4DArgs r;                                                                                 \\
    const int rc = g8_decode_common(a, &r, 256, ROWB);                                         \\
"""
_DTYPE_CHECK = """    if ((XOPT) == 0 && !g8_e4m3_cache(a)) return R4D_NO(RAD_E_DTYPE);                           \\
"""

def _dtype_check_guard(launch: str) -> str:
    native = re.escape(_DTYPE_CHECK)
    packed = re.escape(_DTYPE_CHECK.replace("g8_e4m3_cache(a)", "g8_byte_cache(a, LAYOUT)"))
    return "(?:" + native + "|" + packed + ")" + re.escape(launch)


_CONSTRAINTS = r"""/* Native E4M3 on the two Flash-Next page-4 geometries. */
#define BC_Q4_FP8_ATTN_BASE(gqa) \
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", gqa), C_EQ("block_size", 4), \
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_e4m3")
#define BC_Q4_FP8_ATTN_C(tag, gqa, max_decode) \
    static const RadConstraint cAttnPrefill##tag##Fp8[] = { \
        BC_Q4_FP8_ATTN_BASE(gqa), C_EQ("causal", 1), \
    }; \
    static const RadConstraint cAttnDecode##tag##Fp8[] = { \
        BC_Q4_FP8_ATTN_BASE(gqa), C_GE("causal", 0), C_LE("q_len", max_decode), \
    }; \
    static const RadConstraint cAttnDecodeGq##tag##Fp8[] = { \
        BC_Q4_FP8_ATTN_BASE(gqa), C_GE("causal", 0), C_LE("q_len", max_decode), \
        C_EQ("group", 128), C_IN("act", "sigmoid"), \
    };
BC_Q4_FP8_ATTN_C(H256G12, 12, 5)
BC_Q4_FP8_ATTN_C(H256G6Bs4, 6, 10)
#undef BC_Q4_FP8_ATTN_C
#undef BC_Q4_FP8_ATTN_BASE

"""

_ATTN_ROWS = r"""#define BC_Q4_FP8_ATTN_ROWS(name, sym, tag, geometry) \
{ "attn_prefill_" #name "_fp8kv", "attn_paged", "attn", \
  "Paged E4M3 cache attention with dynamic prefill splits", geometry, \
  "bf16 query and output, E4M3 K and V", RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL, \
  ROW(cAttnPrefill##tag##Fp8), NOTUNE, \
  nullptr, nullptr, r4d_rad_attn_prefill_##sym##_fp8, r4d_rad_attn_prefill_##sym##_fp8_scratch, \
  nullptr, nullptr, r4d_shape_attn_paged, nullptr, nullptr, RAD_KERNEL_CAP_DYNAMIC_PREFILL_SPLIT }, \
{ "attn_decode_" #name "_fp8kv", "attn_paged", "attn", \
  "Paged E4M3 cache attention with split-KV decode and verification", geometry, \
  "bf16 query and output, E4M3 K and V", RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE, \
  ROW(cAttnDecode##tag##Fp8), ROW(tAttnSplit), nullptr, \
  nullptr, nullptr, r4d_rad_attn_decode_##sym##_fp8, r4d_rad_attn_decode_##sym##_fp8_scratch, \
  nullptr, nullptr, r4d_shape_attn_paged }, \
{ "attn_decode_gq_" #name "_fp8kv", "attn_paged_gate_quant", "attn", \
  "Paged split-KV decode with the output gate and quantiser in the merge", geometry, \
  "bf16 query and gated output, E4M3 K and V, group-128 quantised output", \
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE, \
  ROW(cAttnDecodeGq##tag##Fp8), ROW(tAttnSplit), nullptr, \
  nullptr, nullptr, r4d_rad_attn_decode_gq_##sym##_fp8, r4d_rad_attn_decode_##sym##_fp8_scratch, \
  nullptr, nullptr, r4d_shape_attn_paged_gate_quant, nullptr, r4d_fuse_attn_paged_gate_quant }, \
{ "attn_decode_ghq_" #name "_fp8kv", "attn_paged_gate_had_quant", "attn", \
  "Paged split-KV decode with the output gate and rotated int8 quantiser in the merge", geometry, \
  "bf16 query and gated output, E4M3 K and V, group-128 rotated int8 output", \
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE, \
  ROW(cAttnDecodeGq##tag##Fp8), ROW(tAttnSplit), nullptr, \
  nullptr, nullptr, r4d_rad_attn_decode_ghq_##sym##_fp8, r4d_rad_attn_decode_##sym##_fp8_scratch, \
  nullptr, nullptr, nullptr, nullptr, r4d_fuse_attn_paged_gate_had_quant },
BC_Q4_FP8_ATTN_ROWS(h256_gqa12, h256_gqa12, H256G12, "H256/GQA12/page4")
BC_Q4_FP8_ATTN_ROWS(h256_gqa6_bs4, h256_gqa6, H256G6Bs4, "H256/GQA6/page4")
#undef BC_Q4_FP8_ATTN_ROWS

"""

_FUSED_ROWS = r"""{ "qk_norm_rope_gate_kv_store_fp8", "qk_norm_rope_gate_kv_store", "attn",
  "Fused QK normalization, RoPE, gate and E4M3 KV store",
  "Flash-Next H256 with page4, negative slots store nothing",
  "bf16 Q, K and V, native E4M3 cache with caller-constant scales",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cQkNormRopeGateKvQ4Fp8), NOTUNE,
  nullptr, nullptr, r4d_rad_qk_norm_rope_gate_kv_store, nullptr, nullptr, nullptr,
  r4d_shape_qk_norm_rope_gate_kv_store, nullptr, r4d_fuse_qk_norm_rope_gate_kv_store },

{ "qk_norm_rope_gate_kv_store_streamed_fp8", "qk_norm_rope_gate_kv_store_streamed", "attn",
  "Fused QK normalization, RoPE, gate and streamed E4M3 KV store",
  "Flash-Next H256 with page4, host mirror and resident ring slot",
  "bf16 Q, K and V, native E4M3 cache with caller-constant scales",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cQkNormRopeGateKvQ4Fp8), NOTUNE,
  nullptr, nullptr, r4d_rad_qk_norm_rope_gate_kv_store_streamed, nullptr, nullptr, nullptr,
  r4d_shape_qk_norm_rope_gate_kv_store_streamed, nullptr, r4d_fuse_qk_norm_rope_gate_kv_store_streamed },

"""

_PAGE16_CONSTRAINT = """/* Resident native E4M3 page-16 fused store at the compiled H256 width.
 * The fused kernel validates head counts and rotary width at launch; these vary by rank.
 * Leave the existing Flash-Next page-4 and streamed rows unchanged. */
static const RadConstraint cQkNormRopeGateKvH256Page16Fp8[] = {
    C_IN("q_dtype", "bf16"), C_EQ("head_dim", 256),
    C_EQ("block_size", 16), C_IN("kv_dtype", "fp8_e4m3"),
};

"""

_PAGE16_ROW = """{ "qk_norm_rope_gate_kv_store_h256_page16_fp8", "qk_norm_rope_gate_kv_store", "attn",
  "Fused QK normalization, RoPE, gate and native E4M3 page-16 KV store",
  "H256/page16 with rank-local head counts, negative slots store nothing",
  "bf16 Q, K and V, native E4M3 K and V with caller-constant scales",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cQkNormRopeGateKvH256Page16Fp8), NOTUNE,
  nullptr, nullptr, r4d_rad_qk_norm_rope_gate_kv_store, nullptr, nullptr, nullptr,
  r4d_shape_qk_norm_rope_gate_kv_store, nullptr, r4d_fuse_qk_norm_rope_gate_kv_store },

"""

_FP8_WRITE_BYTES = """  } else if constexpr (KV == QK_KV_FP8 || KV == QK_KV_STREAMED_FP8) {
    if (isk && t < HD) {
      constexpr bool STREAMED = KV == QK_KV_STREAMED_FP8;
      const int32_t sl = kvs.slot[tok];
      const uint32_t block_size = (uint32_t)kvs.block_size;
      const uint32_t blocks = (uint32_t)(STREAMED ? kvs.blocks : kvs.n_blocks);
      if (sl >= 0 && (uint32_t)sl / block_size < blocks) {
        const uint32_t blk = (uint32_t)sl / block_size, off = (uint32_t)sl % block_size;
        const size_t block_bytes = (size_t)nkv * block_size * (2 * HD);
        const size_t offset = ((size_t)lh * block_size + off) * (2 * HD);
        // Match kv_store_fp8's BF16 rounding point, scale, clamp and byte conversion.
        const float kf = __bfloat162float(__float2bfloat16(res)) * kvs.k_inv;
        const float vf = __bfloat162float(kvs.v[(size_t)tok * kvs.v_pitch + (size_t)lh * HD + t]) *
                         kvs.v_inv;
        const int packed = __builtin_amdgcn_cvt_pk_fp8_f32(
            fminf(fmaxf(kf, -448.0f), 448.0f), fminf(fmaxf(vf, -448.0f), 448.0f), 0, false);
        auto put = [&](void* pool, uint32_t block) {
          auto* dst = (uint8_t*)pool + (size_t)block * block_bytes + offset;
          dst[t] = (uint8_t)packed;
          dst[HD + t] = (uint8_t)(packed >> 8);
        };
        if constexpr (STREAMED) {
          put(kvs.host, blk);
          const int32_t slot = kvs.res[blk];
          if (slot >= 0 && (uint32_t)slot < (uint32_t)kvs.n_blocks)
            put(kvs.cache, (uint32_t)slot);
        } else {
          put(kvs.cache, blk);
        }
      }
    }
"""

def _declarations() -> str:
    return "".join(
        f"int     r4d_rad_attn_prefill_h256_gqa{ratio}_fp8(const RadArgs*, RadStream);\n"
        f"int64_t r4d_rad_attn_prefill_h256_gqa{ratio}_fp8_scratch(const RadArgs*);\n"
        f"int     r4d_rad_attn_decode_h256_gqa{ratio}_fp8(const RadArgs*, RadStream);\n"
        f"int     r4d_rad_attn_decode_gq_h256_gqa{ratio}_fp8(const RadArgs*, RadStream);\n"
        f"int     r4d_rad_attn_decode_ghq_h256_gqa{ratio}_fp8(const RadArgs*, RadStream);\n"
        f"int64_t r4d_rad_attn_decode_h256_gqa{ratio}_fp8_scratch(const RadArgs*);\n"
        for ratio in (12, 6)
    )


_GEOM = "    Geom& g = m.g;\n\n    /* Qwen4-Exp's norms"
_PROBE = "static int probe(const RadModelMeta* meta, RadArchProbe* out) {\n    if (!meta || !out) return RAD_E_INVAL;\n"
_BYTE_LAUNCH_ANCHOR = "#define G8_I8_ENTRIES(TAG, ROWB, XOPT)"
_BYTE_LAUNCH_END = """G8_I8_ENTRIES(i8, 516, O_KVI8)
#undef G8_I8_ENTRIES
"""
_FUSED_ENUM = """enum { QK_KV_NONE = 0, QK_KV_BF16 = 2, QK_KV_STREAMED_BF16 = 6, QK_KV_I8 = 7,
       QK_KV_STREAMED_I8 = 8 };
"""

PATCHES = (
    FilePatch(path="arch/qwen4exp_fp8/qwen4exp_fp8.cpp", language="none", edits=(
        _edit("q4-native-e4m3-geometry", "static int declare(RadBuilder* b,",
              _GEOMETRY, "static bool native_e4m3_geometry(", mode="insert_before"),
        _edit("q4-native-e4m3-declare", _GEOM,
              "    Geom& g = m.g;\n"
              "    if (ctx->kv_dtype == RAD_F8E4M3 && !native_e4m3_geometry(g.head_dim, g.n_head, g.n_head_kv)) {\n"
              "        fprintf(stderr, \"radiance: qwen4exp: the E4M3 cache requires H256, GQA6 or GQA12, and page4\\n\");\n"
              "        return RAD_E_UNSUPPORTED;\n"
              "    }\n\n"
              "    /* Qwen4-Exp's norms", "the E4M3 cache requires H256, GQA6 or GQA12, and page4"),
        _edit("q4-native-e4m3-probe", _PROBE,
              _PROBE +
              "    const int64_t hd = meta->head_dim ? meta->head_dim :\n"
              "                       (meta->n_head > 0 ? meta->n_embd / meta->n_head : 0);\n"
              "    const int64_t kv_heads = meta->n_head_kv ? meta->n_head_kv : meta->n_head;\n"
              "    out->kv_fp8_ok = native_e4m3_geometry(hd, meta->n_head, kv_heads);\n",
              "out->kv_fp8_ok = native_e4m3_geometry("),
    )),
    FilePatch(path="libr4d/r4d_attn_paged_gqa8.hip", language="none", edits=(
        _edit("q4-e4m3-byte-launch-macro", _BYTE_LAUNCH_ANCHOR,
              _BYTE_LAUNCH, "#define G8_BYTE_ENTRIES(TAG, ROWB, XOPT"),
        Edit(id="q4-e4m3-prefill-check", anchor=re.escape(_FIRST_BYTE_LAUNCH),
             mode="replace", text=_DTYPE_CHECK + _FIRST_BYTE_LAUNCH,
             guard=_dtype_check_guard(_FIRST_BYTE_LAUNCH), max_span_lines=3,
             rationale="Require the native E4M3 tensor at the prefill shim"),
        Edit(id="q4-e4m3-decode-checks", anchor=re.escape(_BYTE_DECODE_LAUNCH),
             mode="replace_all", text=(_DTYPE_CHECK + _BYTE_DECODE_LAUNCH).replace("\\", "\\\\"),
             guard=_dtype_check_guard(_BYTE_DECODE_LAUNCH), expect_matches=3,
             max_span_lines=3, rationale="Require the native E4M3 tensor at the decode shims"),
        _edit("q4-e4m3-page4-wrappers", _BYTE_LAUNCH_END,
              "G8_BYTE_ENTRIES(i8, 516, O_KVI8)\n"
              "#if G8_BLOCK_SIZE == 4 && (R4D_ATTN_GQA == 12 || R4D_ATTN_GQA == 6)\n"
              "G8_BYTE_ENTRIES(fp8, 512, 0)\n"
              "#endif\n"
              "#undef G8_BYTE_ENTRIES\n", "G8_BYTE_ENTRIES(fp8, 512, 0"),
    )),
    FilePatch(path="libr4d/r4d_plugin.h", language="none", edits=(
        _edit("q4-e4m3-wrapper-declarations", "int     r4d_rad_attn_prefill_h256_gqa12_i8",
              _declarations(), "r4d_rad_attn_prefill_h256_gqa12_fp8_scratch(", mode="insert_before"),
    )),
    FilePatch(path="libr4d/r4d_qk_norm_rope_gate.hip", language="none", edits=(
        _edit("q4-e4m3-fused-fields", "  int   v_pitch, block_size, n_blocks;\n",
              "  float k_inv, v_inv;       /* E4M3 caller-constant inverse descales */\n",
              "float k_inv, v_inv;", mode="insert_after"),
        _edit("q4-e4m3-fused-types", _FUSED_ENUM,
              "enum { QK_KV_NONE = 0, QK_KV_FP8 = 1, QK_KV_BF16 = 2, QK_KV_STREAMED_FP8 = 5,\n"
              "       QK_KV_STREAMED_BF16 = 6, QK_KV_I8 = 7, QK_KV_STREAMED_I8 = 8 };\n",
              "QK_KV_STREAMED_FP8 = 5"),
        _edit("q4-e4m3-fused-byte-encoding", "  } else if constexpr (KV != QK_KV_NONE) {\n",
              _FP8_WRITE_BYTES, "KV == QK_KV_FP8 || KV == QK_KV_STREAMED_FP8", mode="insert_before"),
        _edit("q4-e4m3-fused-validate-tensor", "      (kv->dtype != RAD_I8 && kv->dtype != RAD_BF16))\n",
              "      (kv->dtype != RAD_I8 && kv->dtype != RAD_BF16 && kv->dtype != RAD_F8E4M3))\n",
              "kv->dtype != RAD_I8 && kv->dtype != RAD_BF16 && kv->dtype != RAD_F8E4M3"),
        _edit("q4-e4m3-fused-resident-launch", "  if (!streamed)\n    return i8 ? qk_body<QK_KV_I8>",
              """  const bool fp8 = kv->dtype == RAD_F8E4M3;
  if (fp8) {
    const double ks = r4d_getf(args, "k_scale", 1.0);
    const double vs = r4d_getf(args, "v_scale", 1.0);
    if (!(ks > 0.0) || !(vs > 0.0)) return R4D_NO(RAD_E_INVAL);
    kvs.k_inv = 1.0f / (float)ks;
    kvs.v_inv = 1.0f / (float)vs;
    if (!streamed) return qk_body<QK_KV_FP8>(args, stream, kvs);
  }
  if (!streamed)
    return i8 ? qk_body<QK_KV_I8>""", "if (!streamed) return qk_body<QK_KV_FP8>"),
        _edit("q4-e4m3-fused-byte-pitch", " * rowe * (i8 ? 1 : 2);",
              " * rowe * ((i8 || fp8) ? 1 : 2);", "rowe * ((i8 || fp8) ? 1 : 2);"),
        _edit("q4-e4m3-fused-stream-launch", "  return i8 ? qk_body<QK_KV_STREAMED_I8>",
              "  if (fp8) return qk_body<QK_KV_STREAMED_FP8>(args, stream, kvs);\n",
              "if (fp8) return qk_body<QK_KV_STREAMED_FP8>", mode="insert_before"),
    )),
    FilePatch(path="libr4d/r4d_rows.cpp", language="none", edits=(
        _edit("q4-e4m3-attention-constraints", "static const RadConstraint cAttnPrefillH256G12I8[]",
              _CONSTRAINTS, "#define BC_Q4_FP8_ATTN_C(", mode="insert_before"),
        _edit("q4-e4m3-attention-rows", '{ "attn_prefill_h256_gqa12_i8",',
              _ATTN_ROWS, "#define BC_Q4_FP8_ATTN_ROWS(", mode="insert_before"),
        _edit("q4-e4m3-fused-constraint", "static const RadConstraint cQkNormRopeGateKvI8[]",
              "static const RadConstraint cQkNormRopeGateKvQ4Fp8[] = {\n"
              '    C_IN("q_dtype", "bf16"), C_EQ("head_dim", 256), C_EQ("block_size", 4),\n'
              '    C_IN("kv_dtype", "fp8_e4m3"),\n'
              "};\n", "static const RadConstraint cQkNormRopeGateKvQ4Fp8[]", mode="insert_before"),
        _edit("q4-e4m3-page16-constraint", "static const RadConstraint cQkNormRopeGateKvI8[]",
              _PAGE16_CONSTRAINT, "static const RadConstraint cQkNormRopeGateKvH256Page16Fp8[]",
              mode="insert_before"),
        _edit("q4-e4m3-fused-rows", '{ "qk_norm_rope_gate_kv_store_i8",',
              _FUSED_ROWS, '{ "qk_norm_rope_gate_kv_store_fp8",', mode="insert_before"),
        _edit("q4-e4m3-page16-row", '{ "qk_norm_rope_gate_kv_store_i8",',
              _PAGE16_ROW, '{ "qk_norm_rope_gate_kv_store_h256_page16_fp8",',
              mode="insert_before"),
        _edit("q4-e4m3-chain-fused", '        "qk_norm_rope_gate_kv_store_i8", "qk_norm_rope_gate_kv_store_bf16",\n',
              '        "qk_norm_rope_gate_kv_store_fp8",\n', '        "qk_norm_rope_gate_kv_store_fp8",\n',
              mode="insert_after"),
        _edit("q4-e4m3-chain-decode", '        "attn_decode_h256_gqa6_bs4_bf16kv", "attn_decode_h256_gqa6_bs4_i8",\n',
              '        "attn_decode_h256_gqa12_fp8kv", "attn_decode_h256_gqa6_bs4_fp8kv",\n',
              '        "attn_decode_h256_gqa12_fp8kv", "attn_decode_h256_gqa6_bs4_fp8kv",\n', mode="insert_after"),
    )),
    FilePatch(path="libr4d/r4d_selftest.cpp", language="none", edits=(
        _edit("q4-e4m3-page16-selftest-signature",
              "static void qk_kv_fold_case(const Plugin& pl, bool bf16, int kvq = 0) {",
              "static void qk_kv_fold_case(const Plugin& pl, bool bf16, int kvq = 0,\n"
              "                            int page16_nq = 0, int page16_nkv = 0) {",
              "int page16_nq = 0, int page16_nkv = 0"),
        _edit("q4-e4m3-page16-selftest-row",
              '    const char* row = bf16     ? "qk_norm_rope_gate_kv_store_bf16"',
              '    const char* row = page16_nq ? "qk_norm_rope_gate_kv_store_h256_page16_fp8"\n'
              '                    : bf16     ? "qk_norm_rope_gate_kv_store_bf16"',
              'row = page16_nq ? "qk_norm_rope_gate_kv_store_h256_page16_fp8"'),
        _edit("q4-e4m3-page16-selftest-geometry",
              "    const int64_t M = 48, nq = 12, nkv = 2, hd = 256, rot = 64, ctx = 4096, bs = 16, nb = 6;",
              "    const int64_t M = 48, nq = page16_nq ? page16_nq : 12;\n"
              "    const int64_t nkv = page16_nq ? page16_nkv : 2;\n"
              "    const int64_t hd = 256, rot = 64, ctx = 4096, bs = 16, nb = 6;",
              "nq = page16_nq ? page16_nq : 12"),
        _edit("q4-e4m3-page16-selftest-invocation",
              "        qk_kv_fold_case(pl, false, 3);\n",
              "        qk_kv_fold_case(pl, false, 0, 6, 1);\n"
              "        qk_kv_fold_case(pl, false, 0, 12, 2);\n",
              "qk_kv_fold_case(pl, false, 0, 12, 2)", mode="insert_after"),
    )),
    FilePatch(path="tests/arch_test.cpp", language="none", edits=(
        _edit("q4-e4m3-host-fixture", "TEST(a_streamed_qwen4exp_group_stores_then_resolves_then_attends)",
              _ARCH_TEST, "TEST(qwen4exp_native_e4m3_uses_the_page4_attention_and_shared_mtp_group)",
              mode="insert_before"),
    )),
)
