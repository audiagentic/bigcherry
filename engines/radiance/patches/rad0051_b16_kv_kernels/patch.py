"""The resident B16 cache kernels and declarations on the composed ABI44 source."""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch


def _replace(edit_id: str, before: str, after: str, guard: str | None = None, *,
             count: int = 1) -> Edit:
    return Edit(
        id=edit_id, anchor=re.escape(before), text=after if count == 1 else after.replace("\\", "\\\\"),
        mode="replace" if count == 1 else "replace_all",
        guard=(guard[3:] if guard and guard.startswith("re:") else re.escape(guard or after)), expect_matches=count,
        max_span_lines=before.count("\n") + 1,
        rationale="Attach the packed KV contract to the reviewed ABI44 composition",
    )


def _before(edit_id: str, anchor: str, text: str, guard: str) -> Edit:
    return Edit(
        id=edit_id, anchor=re.escape(anchor), mode="insert_before", text=text,
        guard=re.escape(guard), expect_matches=1, max_span_lines=anchor.count("\n") + 1,
        rationale="Add the resident packed KV path at the matching native attachment",
    )


_TRAITS = r"""/* Resident hybrid KV byte positions and shared store codecs. */
#pragma once

#include "rad_builder.h"
#include "r4d_common.h"

template<RadKVLayout Layout> struct R4dKvBytes {
    static constexpr int row16 = 0, value16 = 0, value_scale16 = 0, group = 32;
};
#define R4D_KV_BYTES(id, number, name, row, value, ks, vs, scale_group) \
    template<> struct R4dKvBytes<RAD_KVQ_##id> { \
        static constexpr int row16 = row, value16 = value, value_scale16 = vs, group = scale_group; \
    };
RAD_KV_LAYOUTS(R4D_KV_BYTES)
#undef R4D_KV_BYTES

constexpr bool r4d_b16_layout(RadKVLayout layout) {
    return layout == RAD_KVQ_B16F8 || layout == RAD_KVQ_B16I8;
}

/* v1.17 B16I8 quantization. Positive scales round upward to FP16 and stay
 * between 2^-14 and 515.5. fmaxf ignores NaNs in the maximum; the code clamp
 * maps a NaN to -127 when its group has a scale and to 0 otherwise. */
__device__ __forceinline__ uint32_t r4d_b16_scale_bits(float amax) {
    const _Float16 rn = (_Float16)fminf(amax / 127.0f, 515.5f);
    uint32_t b = (uint32_t)__builtin_bit_cast(uint16_t, rn);
    b += ((float)rn * 127.0f < amax) ? 1u : 0u;
    b = b < 0x0400u ? 0x0400u : b > 0x6007u ? 0x6007u : b;
    return amax > 0.0f ? b : 0u;
}

__device__ __forceinline__ uint32_t r4d_b16_i8_code(float v, float scale) {
    const float r = scale > 0.0f ? rintf(v / scale) : 0.0f;
    return (uint32_t)(int)fminf(fmaxf(r, -127.0f), 127.0f) & 0xffu;
}

__device__ __forceinline__ uint32_t r4d_b16_f8_pair(float a, float b) {
    return (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(
        fminf(fmaxf(a, -448.0f), 448.0f), fminf(fmaxf(b, -448.0f), 448.0f), 0, false) & 0xffffu;
}
"""

_STANDALONE = r"""// Resident B16 stores: BF16 K bytes followed by V codes and optional group scales.
template<RadKVLayout Layout>
__global__ __launch_bounds__(kThreads) void kv_store_b16_kernel(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const int32_t* __restrict__ slot, uint8_t* __restrict__ kv,
    uint32_t heads, uint32_t page, uint32_t blocks, uint32_t k_ld, uint32_t v_ld) {
    using Format = R4dKvBytes<Layout>;
    constexpr uint32_t hd = 256, row = hd / 16 * Format::row16;
    constexpr uint32_t value = hd / 16 * Format::value16;
    constexpr uint32_t scale = hd / 16 * Format::value_scale16;
    const uint32_t t = blockIdx.x;
    const int32_t sl = slot[t];
    if (sl < 0 || (uint32_t)sl / page >= blocks) return;
    const uint32_t blk = (uint32_t)sl / page, off = (uint32_t)sl % page;
    for (uint32_t j = threadIdx.x * 4u; j < heads * hd; j += kThreads * 4u) {
        const uint32_t h = j / hd, d = j % hd;
        uint8_t* dst = kv + (((size_t)blk * heads + h) * page + off) * row;
        const uint16_t* key = k + (size_t)t * k_ld + j;
        *(uint2*)(dst + 2 * d) = make_uint2((uint32_t)key[0] | ((uint32_t)key[1] << 16),
                                            (uint32_t)key[2] | ((uint32_t)key[3] << 16));
        const uint16_t* src = v + (size_t)t * v_ld + j;
        const float a = bf16_to_f32(src[0]), b = bf16_to_f32(src[1]);
        const float c = bf16_to_f32(src[2]), d4 = bf16_to_f32(src[3]);
        uint32_t codes;
        if constexpr (Layout == RAD_KVQ_B16I8) {
            constexpr uint32_t group = Format::group;
            float amax = fmaxf(fmaxf(fabsf(a), fabsf(b)), fmaxf(fabsf(c), fabsf(d4)));
            for (int mask = group / 8; mask >= 1; mask >>= 1)
                amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
            const uint32_t sb = r4d_b16_scale_bits(amax);
            const float sc = (float)__builtin_bit_cast(_Float16, (uint16_t)sb);
            codes = r4d_b16_i8_code(a, sc) | (r4d_b16_i8_code(b, sc) << 8) |
                    (r4d_b16_i8_code(c, sc) << 16) | (r4d_b16_i8_code(d4, sc) << 24);
            if (d % group == 0) *(uint16_t*)(dst + scale + 2 * (d / group)) = (uint16_t)sb;
        } else {
            codes = r4d_b16_f8_pair(a, b) | (r4d_b16_f8_pair(c, d4) << 16);
        }
        *(uint32_t*)(dst + value + d) = codes;
    }
}

"""

_STORE_LAUNCH = r"""template<RadKVLayout Layout>
static int kv_store_b16(const RadArgs* a, RadStream stream) {
    KvStoreGeom g{};
    const int rc = kv_store_geom(a, &g, R4dKvBytes<Layout>::row16);
    if (rc != RAD_OK) return rc;
    if (g.kv->dtype != RAD_U8) return R4D_NO(RAD_E_DTYPE);
    if (g.head_dim != 256 || (g.block_size != 4 && g.block_size != 16))
        return R4D_NO(RAD_E_SHAPE);
    return R4D_EMIT(stream, kv_store_b16_kernel<Layout>, dim3((uint32_t)g.T), dim3(kThreads), 0,
                    (const uint16_t*)r4d_p(g.k), (const uint16_t*)r4d_p(g.v),
                    (const int32_t*)r4d_p(g.sm), (uint8_t*)r4d_wp(g.kv),
                    (uint32_t)g.kv_heads, (uint32_t)g.block_size, (uint32_t)g.n_blocks,
                    (uint32_t)g.k_ld, (uint32_t)g.v_ld);
}
extern "C" int r4d_rad_kv_store_b16f8(const RadArgs* a, RadStream stream) {
    return kv_store_b16<RAD_KVQ_B16F8>(a, stream);
}
extern "C" int r4d_rad_kv_store_b16i8(const RadArgs* a, RadStream stream) {
    return kv_store_b16<RAD_KVQ_B16I8>(a, stream);
}

"""

_FUSED = r"""  } else if constexpr (r4d_b16_layout(Layout)) {
    if (isk && t < HD) {
      using Format = R4dKvBytes<Layout>;
      constexpr size_t row = HD / 16 * Format::row16;
      constexpr size_t value = HD / 16 * Format::value16;
      const int32_t sl = kvs.slot[tok];
      const uint32_t page = (uint32_t)kvs.block_size;
      if (sl >= 0 && (uint32_t)sl / page < (uint32_t)kvs.n_blocks) {
        const uint32_t blk = (uint32_t)sl / page, off = (uint32_t)sl % page;
        auto* dst = (uint8_t*)kvs.cache + (((size_t)blk * nkv + lh) * page + off) * row;
        ((__hip_bfloat16*)dst)[t] = __float2bfloat16(res);
        const float v = __bfloat162float(kvs.v[(size_t)tok * kvs.v_pitch + (size_t)lh * HD + t]);
        if constexpr (Layout == RAD_KVQ_B16I8) {
          constexpr int group = Format::group;
          constexpr size_t scale = HD / 16 * Format::value_scale16;
          float amax = fabsf(v);
          for (int mask = group / 2; mask >= 1; mask >>= 1)
            amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
          const uint32_t sb = r4d_b16_scale_bits(amax);
          dst[value + t] = (uint8_t)r4d_b16_i8_code(v, (float)__builtin_bit_cast(_Float16, (uint16_t)sb));
          if (t % group == 0) *(uint16_t*)(dst + scale + (t / group) * 2) = (uint16_t)sb;
        } else {
          dst[value + t] = (uint8_t)r4d_b16_f8_pair(v, 0.0f);
        }
      }
    }
"""

_FUSED_DTYPE = """  const char* kvd = r4d_gets(args, "kv_dtype");
  const uint32_t want = kvd ? rad_dtype_parse(kvd) : (uint32_t)RAD_DT_INVALID;
  if (v->dtype != RAD_BF16 || sm->dtype != RAD_I32 || kv->dtype != want ||
      (kv->dtype != RAD_I8 && kv->dtype != RAD_BF16 && kv->dtype != RAD_F8E4M3))
    return R4D_NO(RAD_E_DTYPE);
"""

PATCHES = (
    FilePatch(path="libr4d/r4d_b16_kv.h", language="none", create=True, edits=(
        Edit(id="b16-traits-and-codecs", anchor=r"\A", mode="insert_after", text=_TRAITS,
             guard=r"r4d_b16_scale_bits\("),
    )),
    FilePatch(path="libr4d/r4d_model_bf16.hip", language="none", edits=(
        _replace("b16-store-header", '#include "r4d_common.h"\n', '#include "r4d_b16_kv.h"\n'),
        _before("b16-store-kernel", "// The store of a streamed KV group", _STANDALONE,
                "void kv_store_b16_kernel("),
        _before("b16-store-launch", "/* The int8 store: the cache's last extent", _STORE_LAUNCH,
                "static int kv_store_b16("),
    )),
    FilePatch(path="libr4d/r4d_qk_norm_rope_gate.hip", language="none", edits=(
        _replace("b16-fold-header", '#include "r4d_common.h"\n', '#include "r4d_b16_kv.h"\n'),
        _replace("b16-fold-kv-type", "template <int HD, int KV = QK_KV_NONE>",
                 "template <int HD, int KV = QK_KV_NONE, RadKVLayout Layout = RAD_KVQ_NONE>"),
        _before("b16-fold-store", "  } else if constexpr (KV != QK_KV_NONE) {\n", _FUSED,
                "else if constexpr (r4d_b16_layout(Layout))"),
        _replace("b16-fold-kernel-type", "template <int HD, int KV, class CH>",
                 "template <int HD, int KV, class CH, RadKVLayout Layout = RAD_KVQ_NONE>"),
        _replace("b16-fold-body-call", "r4d_qk_norm_rope_gate_body<HD, KV>",
                 "r4d_qk_norm_rope_gate_body<HD, KV, Layout>"),
        _replace("b16-fold-host-type", "template <int KVF>\n",
                 "template <int KVF, RadKVLayout Layout = RAD_KVQ_NONE>\n"),
        _replace("b16-fold-emit-type", "r4d_qk_norm_rope_gate_kernel<H, KVF, CH_>",
                 "r4d_qk_norm_rope_gate_kernel<H, KVF, CH_, Layout>"),
        _replace("b16-fold-dtype", _FUSED_DTYPE, """  const char* kvd = r4d_gets(args, "kv_dtype");
  const RadKVLayout layout = rad_kvq_from_name(kvd);
  const bool b16 = r4d_b16_layout(layout);
  if (b16 && streamed) return R4D_NO(RAD_E_UNSUPPORTED);
  const uint32_t want = b16 ? (uint32_t)RAD_U8 : rad_dtype_parse(kvd);
  if (v->dtype != RAD_BF16 || sm->dtype != RAD_I32 || kv->dtype != want ||
      (!b16 && kv->dtype != RAD_I8 && kv->dtype != RAD_BF16 && kv->dtype != RAD_F8E4M3))
    return R4D_NO(RAD_E_DTYPE);
""", "const RadKVLayout layout = rad_kvq_from_name(kvd);"),
        _replace("b16-fold-shape",
                 "  const long long rowe = i8 ? 2 * hd + 4 : 2 * hd;       /* a token's row in a head, in elements */",
                 """  if (b16 && (hd != 256 || (bs != 4 && bs != 16))) return R4D_NO(RAD_E_SHAPE);
  const long long rowe = rad_kv_group_row_elems(layout, kv->dtype, hd);""", "rad_kv_group_row_elems(layout, kv->dtype, hd)"),
        _before("b16-fold-launch", "  const bool fp8 = kv->dtype == RAD_F8E4M3;\n",
                """  if (layout == RAD_KVQ_B16F8) return qk_body<QK_KV_NONE, RAD_KVQ_B16F8>(args, stream, kvs);
  if (layout == RAD_KVQ_B16I8) return qk_body<QK_KV_NONE, RAD_KVQ_B16I8>(args, stream, kvs);
""", "qk_body<QK_KV_NONE, RAD_KVQ_B16I8>"),
    )),
)

_TPL_BODY = "int GQA, int BS, int KVP, int OPT, int WIN = 0>"
_TPL_TYPED = "int GQA, int BS, int KVP, int OPT, int WIN = 0, RadKVLayout Layout = RAD_KVQ_NONE>"
_TPL_DECODE = "int GQA, int BS, int KVP, int OPT,\n         int WIN = 0>"
_TPL_DECODE_TYPED = _TPL_DECODE.replace("int WIN = 0>", "int WIN = 0, RadKVLayout Layout = RAD_KVQ_NONE>")
_I8_TEST = "    constexpr int I8    = ((OPT & O_I8) && K6) ? 1 : 0;\n"
_HYBRID_CONSTS = """    constexpr bool KB   = r4d_b16_layout(Layout);
    constexpr bool KI   = Layout == RAD_KVQ_B16I8;
    constexpr bool KBF  = KVP || KB;
    constexpr bool VI8  = I8 || KI;
    constexpr bool VSTAGE = V4 || KI;
    constexpr int VG    = KI ? R4dKvBytes<Layout>::group : 32;
    static_assert(!KB || (HEAD_DIM == 256 && !KVP && !V4 && F16), "hybrid rows stage BF16 K");
"""
_ROWBEFORE = "    constexpr int ROWB  = I8 ? 2 * HEAD_DIM + 4"
_ROWAFTER = """    constexpr int ROWB  = KB ? HEAD_DIM / 16 * R4dKvBytes<Layout>::row16
                        : I8 ? 2 * HEAD_DIM + 4"""
_VOFFSET = """    constexpr int VBOFF = KB ? HEAD_DIM / 16 * R4dKvBytes<Layout>::value16 : HEAD_DIM;
    constexpr int VISOFF = HEAD_DIM / 16 * R4dKvBytes<Layout>::value_scale16;
"""
_VSTAGE_START = """                    if constexpr (I8) {
                        const uint2 cw = *(const uint2*)(base + VOFF + o * 8);"""
_VSCALE = "                        sc[j]             = *(const uint16_t*)(base + VSOFF);"

def _v_staging() -> tuple[Edit, ...]:
    """Carry the typed B16I8 group scales through the existing byte-code V stage."""
    return (
        _replace("b16-template-row", _ROWBEFORE, _ROWAFTER, "re:constexpr int ROWB(?:  = KB \\?| = Layout != RAD_KVQ_NONE)"),
        _before("b16-template-value-offsets", "    constexpr int VOFF  = I8 ?", _VOFFSET,
                "constexpr int VBOFF = KB"),
        _replace("b16-value-stage-rounds", "constexpr int U4  = V4 ?", "constexpr int U4  = VSTAGE ?"),
        _replace("b16-value-stage-registers", "uint32_t v4c[U4][I8 ? 4 : 2]", "uint32_t v4c[U4][VI8 ? 4 : 2]"),
        _replace("b16-value-code-load", _VSTAGE_START, """                    if constexpr (VI8) {
                        const uint2 cw = *(const uint2*)(base + (KI ? VBOFF : VOFF) + o * 8);"""),
        _replace("b16-value-group-scale", _VSCALE,
                 "                        sc[j] = *(const uint16_t*)(base + (KI ? VISOFF + 2 * (o * 8 / VG) : VSOFF));"),
        _replace("b16-byte-code-transform", "I8 ? v4c[r][", "VI8 ? v4c[r][", count=4),
        _replace("b16-byte-code-k1-low", "[I8 ? 2 : 1]", "[VI8 ? 2 : 1]"),
        _replace("b16-byte-code-k1-high", "[I8 ? 3 : 1]", "[VI8 ? 3 : 1]"),
        _replace("b16-i8-widen-bias", "(I8 ? 1152.0f : 1032.0f)", "(VI8 ? 1152.0f : 1032.0f)", count=2),
        _replace("b16-byte-stage-index-lo", "(I8 ? b : 2 * b)", "(VI8 ? b : 2 * b)"),
        _replace("b16-byte-stage-index-hi", "(I8 ? 4 + b : 2 * b + 1)", "(VI8 ? 4 + b : 2 * b + 1)"),
        _replace("b16-fp8-value-offset", "* ROWB + HEAD_DIM + d0;", "* ROWB + VBOFF + d0;"),
    )

PATCHES += (
    FilePatch(path="libr4d/r4d_attn_prefill_h256_gqa6.hip", language="none", edits=(
        _replace("b16-prefill-header", '#include "r4d_common.h"\n', '#include "r4d_b16_kv.h"\n'),
        _replace("b16-prefill-kernel-type", _TPL_BODY, _TPL_TYPED),
        _replace("b16-prefill-layout-constants", _I8_TEST, _I8_TEST + _HYBRID_CONSTS, "constexpr bool KB   = r4d_b16_layout(Layout);"),
        *_v_staging(),
        _replace("b16-prefill-kreg-width", "constexpr int KREGS   = KPASS * (KVP ? 2 : 1);",
                 "constexpr int KREGS   = KPASS * (KBF ? 2 : 1);"),
        _replace("b16-prefill-kprefetch", "&& !KVP && !OLDV4) ?", "&& !KBF && !OLDV4) ?"),
        _replace("b16-prefill-value-prefetch", "((OPT & O_GPREV) && !V4) ?", "((OPT & O_GPREV) && !VSTAGE) ?"),
        _replace("b16-prefill-precision",
                 '''    static_assert(!(OPT & (O_QK8 | O_PV8)) || !KVP,
                  "the e4m3 legs need an 8-bit cache, or they compile to the exact form");''',
                 '''    static_assert(!(OPT & (O_QK8 | O_PV8)) || (!KVP && !KB),
                  "lossy prefill products require the supported native or V4/K6 cache rows");'''),
        _replace("b16-prefill-read-k",
                 """                if (KVP) {
                    const uint16_t* g = (const uint16_t*)a.kv + o + ch * KEL;
                    #pragma unroll
                    for (int q = 0; q < (KVP ? (KEL / 8) : 1); ++q)""",
                 """                if constexpr (KBF) {
                    const auto* row = (const uint8_t*)a.kv + o * (KVP ? 2 : 1);
                    const uint16_t* g = (const uint16_t*)row + ch * KEL;
                    #pragma unroll
                    for (int q = 0; q < KEL / 8; ++q)"""),
        _replace("b16-prefill-stage-k", "if (KVP) {\n                        const uint4 r = kreg[p",
                 "if constexpr (KBF) {\n                        const uint4 r = kreg[p"),
        _replace("b16-prefill-fetch-first-v", "if (V4) fetchV4(tstart", "if (VSTAGE) fetchV4(tstart"),
        _replace("b16-prefill-stage-v", "if constexpr (V4) { storeV4(); }",
                 "if constexpr (VSTAGE) { storeV4(); }"),
        _replace("b16-prefill-fetch-next-v", "if (V4 && ti + 1 < ntiles) fetchV4",
                 "if (VSTAGE && ti + 1 < ntiles) fetchV4"),
    )),
    FilePatch(path="libr4d/r4d_attn_decode_h256_gqa6.hip", language="none", edits=(
        _replace("b16-decode-header", '#include "r4d_common.h"\n', '#include "r4d_b16_kv.h"\n'),
        _replace("b16-decode-kernel-type", _TPL_DECODE, _TPL_DECODE_TYPED, count=3),
        _replace("b16-decode-kernel-forward",
                 "r4d_attn_decode_body<NWARPS, TILE, HEAD_DIM, GQA, BS, KVP, OPT, WIN>",
                 "r4d_attn_decode_body<NWARPS, TILE, HEAD_DIM, GQA, BS, KVP, OPT, WIN, Layout>", count=2),
        _replace("b16-decode-layout-constants", _I8_TEST, _I8_TEST + _HYBRID_CONSTS, "constexpr bool KB   = r4d_b16_layout(Layout);"),
        *_v_staging(),
        _replace("b16-decode-v8-guard", "((OPT & O_V8) && !KVP && !V4)", "((OPT & O_V8) && !KVP && !VSTAGE)"),
        _replace("b16-decode-gprev-guard", "VREGS <= 3 && !V4)", "VREGS <= 3 && !VSTAGE)"),
        _replace("b16-decode-vsk", "(V4 && TILE == 16", "(VSTAGE && TILE == 16"),
        _replace("b16-decode-vstage-waves", "if constexpr (K6) __builtin_assume(tid < NTHR);",
                 "if constexpr (K6 || KI) __builtin_assume(tid < NTHR);"),
        _replace("b16-decode-vstage-round", "(K6 ? ((r + 1) * NTHR <= NP4",
                 "((K6 || KI) ? ((r + 1) * NTHR <= NP4", count=2),
        _replace("b16-decode-stage-v", "if constexpr (V4) { fetchV4(k0, blk); storeV4(); }",
                 "if constexpr (VSTAGE) { fetchV4(k0, blk); storeV4(); }"),
        _replace("b16-decode-stage-k-lds",
                 """                if (KVP) {
                    const uint4 r = *(const uint4*)((const uint16_t*)a.kv + o + ch * 8);""",
                 """                if constexpr (KBF) {
                    const auto* row = (const uint8_t*)a.kv + o * (KVP ? 2 : 1);
                    const uint4 r = *(const uint4*)((const uint16_t*)row + ch * 8);"""),
        _replace("b16-decode-k-register-width", "std::conditional<KVP != 0, uint4, uint2>",
                 "std::conditional<KBF, uint4, uint2>"),
        _replace("b16-decode-k-fragment-stride", "(16 * t + 8 * h) * (KVP ? 2 : 1)",
                 "(16 * t + 8 * h) * (KBF ? 2 : 1)"),
        _replace("b16-decode-k-from-bf16", "} else if (KVP) {\n                        const uint16_t* kp",
                 "} else if constexpr (KBF) {\n                        const uint16_t* kp"),
        _replace("b16-decode-kraw-bf16", "if constexpr (KRAW && KVP != 0)", "if constexpr (KRAW && KBF)"),
        _replace("b16-decode-kraw-fp8", "if constexpr (KRAW && KVP == 0)", "if constexpr (KRAW && !KBF)"),
    )),
)

_G8_MACRO = """G8_BYTE_ENTRIES(i8, 516, O_KVI8)
#if G8_BLOCK_SIZE == 4 && (R4D_ATTN_GQA == 12 || R4D_ATTN_GQA == 6)
G8_BYTE_ENTRIES(fp8, 512, 0)
#endif
#undef G8_BYTE_ENTRIES
"""

PATCHES += (
    FilePatch(path="libr4d/r4d_attn_paged_gqa8.hip", language="none", edits=(
        _replace("b16-page4-prefill-template", "template <int HEAD_DIM, int OPT, int KVP>",
                 "template <int HEAD_DIM, int OPT, int KVP, RadKVLayout Layout = RAD_KVQ_NONE>"),
        _replace("b16-page4-prefill-split", "G8_BLOCK_SIZE, KVP, OPT | O_PSPLIT>",
                 "G8_BLOCK_SIZE, KVP, OPT | O_PSPLIT, 0, Layout>"),
        _replace("b16-page4-prefill-unsplit", "G8_BLOCK_SIZE, KVP, OPT>",
                 "G8_BLOCK_SIZE, KVP, OPT, 0, Layout>"),
        _replace("b16-page4-decode-template", "template <int HEAD_DIM, int KVP, int XOPT = 0>",
                 "template <int HEAD_DIM, int KVP, int XOPT = 0, RadKVLayout Layout = RAD_KVQ_NONE>"),
        _replace("b16-page4-decode-body", "G8_D_OPT | OPT_ | XOPT>",
                 "G8_D_OPT | OPT_ | XOPT, 0, Layout>"),
        _replace("b16-page4-macro-cache-type",
                 """static bool g8_e4m3_cache(const RadArgs* a) {
    const RadTensor* kv = r4d_opt(a, AP_KV);
    return kv && kv->dtype == RAD_F8E4M3;
}
#define G8_BYTE_ENTRIES(TAG, ROWB, XOPT)""",
                 """static bool g8_byte_cache(const RadArgs* a, RadKVLayout layout) {
    const RadTensor* kv = r4d_opt(a, AP_KV);
    return kv && kv->dtype == (layout == RAD_KVQ_NONE ? RAD_F8E4M3 : RAD_U8);
}
#define G8_BYTE_ENTRIES(TAG, ROWB, XOPT, LAYOUT)"""),
        _replace("b16-page4-macro-cache-checks", "!g8_e4m3_cache(a)",
                 "!g8_byte_cache(a, LAYOUT)", count=4),
        _replace("b16-page4-macro-prefill-call", "g8_prefill_launch<256, G8_P_OPT_256 | (XOPT), 0>",
                 "g8_prefill_launch<256, G8_P_OPT_256 | (XOPT), 0, LAYOUT>"),
        _replace("b16-page4-macro-decode-calls", "g8_decode_launch<256, 0, (XOPT)>",
                 "g8_decode_launch<256, 0, (XOPT), LAYOUT>", count=3),
        _replace("b16-page4-instantiations", _G8_MACRO,
                 """G8_BYTE_ENTRIES(i8, 516, O_KVI8, RAD_KVQ_NONE)
#if G8_BLOCK_SIZE == 4 && (R4D_ATTN_GQA == 12 || R4D_ATTN_GQA == 6)
G8_BYTE_ENTRIES(fp8, 512, 0, RAD_KVQ_NONE)
#endif
#if G8_BLOCK_SIZE == 4 && R4D_ATTN_GQA == 12
G8_BYTE_ENTRIES(b16f8, (16 * R4dKvBytes<RAD_KVQ_B16F8>::row16), 0, RAD_KVQ_B16F8)
G8_BYTE_ENTRIES(b16i8, (16 * R4dKvBytes<RAD_KVQ_B16I8>::row16), 0, RAD_KVQ_B16I8)
#endif
#undef G8_BYTE_ENTRIES
""", "G8_BYTE_ENTRIES(b16i8,"),
    )),
)

_B16_CONSTRAINTS = r"""#define BC_B16_QSA_BASE(dtype) \
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 12), C_EQ("block_size", 4), \
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", dtype)
#define BC_B16_QSA_CONSTRAINTS(tag, dtype) \
    static const RadConstraint cB16Prefill##tag[] = { BC_B16_QSA_BASE(dtype), C_EQ("causal", 1) }; \
    static const RadConstraint cB16Decode##tag[] = { \
        BC_B16_QSA_BASE(dtype), C_GE("causal", 0), C_LE("q_len", 5), \
    }; \
    static const RadConstraint cB16Gate##tag[] = { \
        BC_B16_QSA_BASE(dtype), C_GE("causal", 0), C_LE("q_len", 5), \
        C_EQ("group", 128), C_IN("act", "sigmoid"), \
    }; \
    static const RadConstraint cB16Store##tag[] = { \
        C_EQ("head_dim", 256), C_EQ("block_size", 4), C_IN("kv_dtype", dtype), \
    }; \
    static const RadConstraint cB16FusedStore##tag[] = { \
        C_IN("q_dtype", "bf16"), C_EQ("head_dim", 256), C_EQ("block_size", 4), C_IN("kv_dtype", dtype), \
    };
BC_B16_QSA_CONSTRAINTS(B16F8, "bf16_fp8")
BC_B16_QSA_CONSTRAINTS(B16I8, "bf16_i8")
#undef BC_B16_QSA_CONSTRAINTS
#undef BC_B16_QSA_BASE

"""

_B16_ROWS = r"""#define BC_B16_QSA_ROWS(tag, sym, description) \
{ "attn_prefill_h256_gqa12_" #sym, "attn_paged", "attn", \
  "Resident packed KV prefill with dynamic splits", "H256/GQA12/page4", description, \
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL, ROW(cB16Prefill##tag), NOTUNE, \
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa12_##sym, r4d_rad_attn_prefill_h256_gqa12_##sym##_scratch, \
  nullptr, nullptr, r4d_shape_attn_paged, nullptr, nullptr, RAD_KERNEL_CAP_DYNAMIC_PREFILL_SPLIT }, \
{ "attn_decode_h256_gqa12_" #sym, "attn_paged", "attn", \
  "Resident packed split-KV decode and verification", "H256/GQA12/page4, query length <= 5", description, \
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE, ROW(cB16Decode##tag), ROW(tAttnSplit), nullptr, \
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa12_##sym, r4d_rad_attn_decode_h256_gqa12_##sym##_scratch, \
  nullptr, nullptr, r4d_shape_attn_paged }, \
{ "attn_decode_gq_h256_gqa12_" #sym, "attn_paged_gate_quant", "attn", \
  "Resident packed split-KV decode with gated output quantisation", "H256/GQA12/page4", description, \
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE, ROW(cB16Gate##tag), ROW(tAttnSplit), nullptr, \
  nullptr, nullptr, r4d_rad_attn_decode_gq_h256_gqa12_##sym, \
  r4d_rad_attn_decode_h256_gqa12_##sym##_scratch, nullptr, nullptr, \
  r4d_shape_attn_paged_gate_quant, nullptr, r4d_fuse_attn_paged_gate_quant }, \
{ "attn_decode_ghq_h256_gqa12_" #sym, "attn_paged_gate_had_quant", "attn", \
  "Resident packed split-KV decode with gated rotated output quantisation", "H256/GQA12/page4", description, \
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE, ROW(cB16Gate##tag), ROW(tAttnSplit), nullptr, \
  nullptr, nullptr, r4d_rad_attn_decode_ghq_h256_gqa12_##sym, \
  r4d_rad_attn_decode_h256_gqa12_##sym##_scratch, nullptr, nullptr, \
  nullptr, nullptr, r4d_fuse_attn_paged_gate_had_quant }, \
{ "kv_store_" #sym, "kv_store", "attn", "Resident packed KV store", "H256/page4", description, \
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cB16Store##tag), NOTUNE, \
  nullptr, nullptr, r4d_rad_kv_store_##sym, nullptr, nullptr, nullptr, r4d_shape_kv_store }, \
{ "qk_norm_rope_gate_kv_store_" #sym, "qk_norm_rope_gate_kv_store", "attn", \
  "Fused QK normalization and resident packed KV store", "H256/page4", description, \
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cB16FusedStore##tag), NOTUNE, \
  nullptr, nullptr, r4d_rad_qk_norm_rope_gate_kv_store, nullptr, nullptr, nullptr, \
  r4d_shape_qk_norm_rope_gate_kv_store, nullptr, r4d_fuse_qk_norm_rope_gate_kv_store },
BC_B16_QSA_ROWS(B16F8, b16f8, "BF16 K, E4M3 V")
BC_B16_QSA_ROWS(B16I8, b16i8, "BF16 K, int8 V with FP16 group scales")
#undef BC_B16_QSA_ROWS

"""

def _b16_declarations() -> str:
    return "".join(
        f"int     r4d_rad_attn_prefill_h256_gqa12_{sym}(const RadArgs*, RadStream);\n"
        f"int64_t r4d_rad_attn_prefill_h256_gqa12_{sym}_scratch(const RadArgs*);\n"
        f"int     r4d_rad_attn_decode_h256_gqa12_{sym}(const RadArgs*, RadStream);\n"
        f"int     r4d_rad_attn_decode_gq_h256_gqa12_{sym}(const RadArgs*, RadStream);\n"
        f"int     r4d_rad_attn_decode_ghq_h256_gqa12_{sym}(const RadArgs*, RadStream);\n"
        f"int64_t r4d_rad_attn_decode_h256_gqa12_{sym}_scratch(const RadArgs*);\n"
        f"int     r4d_rad_kv_store_{sym}(const RadArgs*, RadStream);\n"
        for sym in ("b16f8", "b16i8")
    )


_B16_SHAPE = """/* Keep the logical head width and physical packed operand extent distinct. */
static RadOpdDesc r4d_kv_storage_operand(const RadParam* p, int n_p, uint32_t native_dtype,
                                         int64_t blocks, int64_t heads, int64_t page, int64_t hd) {
    const RadKVLayout layout = rad_kvq_from_name(r4d_pgets(p, n_p, "kv_dtype"));
    if (layout == RAD_KVQ_NONE)
        return opd(native_dtype, blocks, heads, page, rad_kv_row_elems(native_dtype, hd));
    return opd(RAD_U8, blocks, heads, page, rad_kv_group_row_elems(layout, RAD_U8, hd));
}

"""

_Q4_B16_VALIDATE = """    if (ctx->kv_layout != RAD_KVQ_NONE &&
        (ctx->kv_dtype != RAD_U8 || g.head_dim != 256 || g.gqa() != 12 ||
         (ctx->kv_layout != RAD_KVQ_B16F8 && ctx->kv_layout != RAD_KVQ_B16I8))) {
        fprintf(stderr, "radiance: qwen4exp: the packed B16 cache requires H256/GQA12/page4\\n");
        return RAD_E_UNSUPPORTED;
    }
"""

PATCHES += (
    FilePatch(path="libr4d/r4d_plugin.h", language="none", edits=(
        _before("b16-plugin-declarations", "int     r4d_rad_attn_prefill_h256_gqa12_fp8(",
                _b16_declarations(), "r4d_rad_kv_store_b16i8("),
    )),
    FilePatch(path="libr4d/r4d_rows.cpp", language="none", edits=(
        _before("b16-register-constraints", "#define BC_Q4_FP8_ATTN_BASE(",
                _B16_CONSTRAINTS, "#define BC_B16_QSA_CONSTRAINTS("),
        _before("b16-register-rows", "#define BC_Q4_FP8_ATTN_ROWS(",
                _B16_ROWS, "#define BC_B16_QSA_ROWS("),
        _replace("b16-register-chain-fused", '        "qk_norm_rope_gate_kv_store_fp8",\n',
                 '        "qk_norm_rope_gate_kv_store_fp8",\n'
                 '        "qk_norm_rope_gate_kv_store_b16f8", "qk_norm_rope_gate_kv_store_b16i8",\n'),
        _replace("b16-register-chain-decode",
                 '        "attn_decode_h256_gqa12_fp8kv", "attn_decode_h256_gqa6_bs4_fp8kv",\n',
                 '        "attn_decode_h256_gqa12_fp8kv", "attn_decode_h256_gqa6_bs4_fp8kv",\n'
                 '        "attn_decode_h256_gqa12_b16f8", "attn_decode_h256_gqa12_b16i8",\n'),
    )),
    FilePatch(path="libr4d/r4d_shapes.cpp", language="none", edits=(
        _before("b16-shapes-header", '#include "r4d_plugin.h"\n', '#include "rad_builder.h"\n',
                '#include "rad_builder.h"'),
        _before("b16-shapes-token-row", "/* kv_store, both rows:", _B16_SHAPE,
                "static RadOpdDesc r4d_kv_storage_operand("),
        _replace("b16-shapes-stores", "opd(KV, nb, kvh, bs, 2 * hd)",
                 "r4d_kv_storage_operand(p, n_p, KV, nb, kvh, bs, hd)", count=3),
        _replace("b16-shapes-attention", "opd(KV, nblk, kvh, bs, 2 * hd)",
                 "r4d_kv_storage_operand(p, n_p, KV, nblk, kvh, bs, hd)"),
    )),
    FilePatch(path="arch/qwen4exp_fp8/qwen4exp_fp8.cpp", language="none", edits=(
        _before("b16-arch-q4-validate", "    /* Qwen4-Exp's norms", _Q4_B16_VALIDATE,
                "ctx->kv_layout != RAD_KVQ_NONE"),
        _replace("b16-arch-q4-layout", "        fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);\n",
                 "        fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);\n"
                 "        fd.kv_layout = ctx->kv_layout;\n"),
        _replace("b16-arch-q4-reader-spelling", "        g.kv_dtype = rad_dtype_name(fd.dtype);",
                 "        const RadKVLayoutDesc* packed = rad_kvq_desc(fd.kv_layout);\n"
                 "        g.kv_dtype = packed ? packed->name : rad_dtype_name(fd.dtype);"),
        _before("b16-arch-q4-probe", "    out->kv_fp8_ok = native_e4m3_geometry(",
                """    /* Probe sees the model geometry; declare checks the resulting rank geometry. */
    out->kv_layout_ok = hd == 256 && kv_heads > 0 && meta->n_head % kv_heads == 0 &&
                        meta->n_head / kv_heads == 12
        ? ((1u << RAD_KVQ_B16F8) | (1u << RAD_KVQ_B16I8)) : 0;
""", "out->kv_layout_ok"),
    )),
    FilePatch(path="arch/common/rad_block_attn_gated_fp8.h", language="none", edits=(
        _replace("b16-arch-q4-no-resolve", "    if (qsa_sel && qsa_nsel)\n        op_resolve = kv_resolve_declare",
                 "    const bool resident_packed = rad_kvq_from_name(g.kv_dtype) != RAD_KVQ_NONE;\n"
                 "    if (!resident_packed && qsa_sel && qsa_nsel)\n        op_resolve = kv_resolve_declare"),
        _replace("b16-arch-no-stream-store", '    if (rad_op_declared(b, "kv_store_streamed"))',
                 '    if (!resident_packed && rad_op_declared(b, "kv_store_streamed"))'),
        _replace("b16-arch-no-stream-fold", '    if (op_fused && rad_op_declared(b, "qk_norm_rope_gate_kv_store_streamed"))',
                 '    if (!resident_packed && op_fused && rad_op_declared(b, "qk_norm_rope_gate_kv_store_streamed"))'),
    )),
    FilePatch(path="libref/ref_attn.cpp", language="none", edits=(
        _before("b16-ref-header", '#include "ref_common.h"\n', '#include "rad_builder.h"\n',
                '#include "rad_builder.h"'),
        _before("b16-ref-attention-refusal", "    const int64_t q_len = cus ? 0 : rows / n_seq;\n",
                """    if (rad_kvq_from_name(p_str(a, "kv_dtype", nullptr)) != RAD_KVQ_NONE)
        return RAD_E_UNSUPPORTED;   /* the host oracle reads native layouts */
""", "return RAD_E_UNSUPPORTED;   /* the host oracle reads native layouts */"),
        _replace("b16-ref-store-refusal", "int ref_kv_store(const RadArgs* a, RadStream) {\n",
                 """int ref_kv_store(const RadArgs* a, RadStream) {
    if (rad_kvq_from_name(p_str(a, "kv_dtype", nullptr)) != RAD_KVQ_NONE)
        return RAD_E_UNSUPPORTED;   /* a packed row needs its matching writer */
"""),
    )),
)

_Q4_B16_TEST = r"""TEST(qwen4exp_packed_b16_e4m3_and_i8_keep_mtp_and_qsa_caches_distinct) {
    const RadModelMeta meta = qwen4exp_meta();
    RadArchProbe probe{};
    REQUIRE_EQ(qwen4exp_fp8::probe(&meta, &probe), RAD_OK);
    CHECK_EQ(probe.kv_layout_ok, (1u << RAD_KVQ_B16F8) | (1u << RAD_KVQ_B16I8));
    RadModelMeta smaller = meta;
    smaller.head_dim = 128;
    REQUIRE_EQ(qwen4exp_fp8::probe(&smaller, &probe), RAD_OK);
    CHECK_EQ(probe.kv_layout_ok, 0u);
    for (const auto layout : { RAD_KVQ_B16F8, RAD_KVQ_B16I8 }) {
        const auto* row = rad_kvq_desc(layout);
        REQUIRE(row != nullptr);
        RadBuildCtx c = build_ctx(0, 1);
        c.max_tok = 64; c.max_seqs = 2; c.max_spec = 3; c.max_ctx = 4096;
        c.kv_dtype = RAD_U8;
        c.kv_layout = layout;
        c.kv_resident_blocks = 64;
        c.kv_stream_ok = 1;
        RadBuilder b;
        b.kv_block = 4;
        REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
        const auto& model = qwen4exp_fp8::g_model[0];
        REQUIRE(model.kv_full > 0);
        REQUIRE(model.kv_qsa_bkey > 0);
        CHECK(model.kv_qsa_bkey != model.kv_full);
        const auto& cache = b.kv_groups[model.kv_full - 1].d;
        CHECK_EQ(cache.kv_layout, (RadKVLayout)layout);
        CHECK_EQ(cache.dtype, (uint32_t)RAD_U8);
        CHECK_EQ(cache.head_dim, 256);
        CHECK_EQ(cache.n_head_kv, 2);
        CHECK_EQ(rad_kv_group_row_elems(cache.kv_layout, cache.dtype, cache.head_dim),
                 layout == RAD_KVQ_B16F8 ? 768 : 784);
        CHECK_EQ(b.kv_groups[model.kv_qsa_bkey - 1].d.kv_layout, (RadKVLayout)RAD_KVQ_NONE);
        CHECK_EQ(b.kv_groups[model.kv_qsa_bkey - 1].d.dtype, (uint32_t)RAD_BF16);
        CHECK(std::find(b.binds.begin(), b.binds.end(),
                        std::make_pair((int)meta.n_layers, model.kv_full)) != b.binds.end());
        CHECK_EQ(count_op(b, "kv_resolve"), 0);
        CHECK_EQ(count_op(b, "kv_store_streamed"), 0);
        CHECK_EQ(count_op(b, "qk_norm_rope_gate_kv_store_streamed"), 0);

        int reads = 0, writes = 0, fused_writes = 0;
        for (const auto& op : b.ops) {
            const bool read = op.op == "attn_paged";
            const bool write = op.op == "kv_store";
            const bool fused = op.op == "qk_norm_rope_gate_kv_store";
            if (!read && !write && !fused) continue;
            bool dtype = false, page = false;
            for (const auto& p : op.p) {
                if (p.key == "kv_dtype") { CHECK_EQ(p.sval, std::string(row->name)); dtype = true; }
                if (p.key == "block_size") { CHECK_EQ(p.ival, 4); page = true; }
                if (read && p.key == "gqa") CHECK_EQ(p.ival, 12);
            }
            CHECK(dtype && page);
            reads += read;
            writes += write;
            fused_writes += fused;
        }
        CHECK(reads >= 2);
        CHECK_EQ(writes, reads);
        CHECK_EQ(fused_writes, reads);

        c.world_size = 4;
        RadBuilder unsupported;
        CHECK_EQ(stderr_of_rc([&] { return qwen4exp_fp8::declare(&unsupported, &meta, &c); }),
                 RAD_E_UNSUPPORTED);
        CHECK(unsupported.kv_groups.empty());
    }
}

"""

PATCHES += (
    FilePatch(path="tests/arch_test.cpp", language="none", edits=(
        _before("b16-q4-host-fixture",
                "TEST(qwen4exp_native_e4m3_uses_the_page4_attention_and_shared_mtp_group)",
                _Q4_B16_TEST, "TEST(qwen4exp_packed_b16_e4m3_and_i8_keep_mtp_and_qsa_caches_distinct)"),
    )),
)
