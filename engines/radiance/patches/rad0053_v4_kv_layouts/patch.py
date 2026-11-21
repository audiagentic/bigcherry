"""Restore the two historical int4-V caches on the canonical ABI44 packed-KV path."""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch


def _edit(edit_id: str, before: str, after: str, *, mode: str = "replace",
          guard: str | None = None, count: int = 1) -> Edit:
    return Edit(
        id=edit_id, anchor=re.escape(before), mode=mode, expect_matches=count,
        text=after.replace("\\", "\\\\") if mode == "replace_all" else after,
        guard=(guard[3:] if guard and guard.startswith("re:") else re.escape(guard if guard is not None else after)) if after or guard else None,
        max_span_lines=before.count("\n") + 1,
        rationale="Restore historical packed KV without changing native cache layout",
    )


PATCHES = (
    FilePatch(path='core/mem/packed_kv.h', language="none", edits=(
        _edit('packed-descriptor-selection',
              """inline bool b16_kv_layout(RadKVLayout layout) {
    return layout == RAD_KVQ_B16F8 || layout == RAD_KVQ_B16I8;
}

inline RadKVLayout configured_b16_kv_layout(const char* name) {
    const RadKVLayout layout = rad_kvq_from_name(name);
    return b16_kv_layout(layout) ? layout : RAD_KVQ_NONE;
}

""",
              '',
              guard='inline void set_build_kv_format(const char* name, RadBuildCtx* out) {\n    out->kv_layout = rad_kvq_from_name(name);'),
        _edit('packed-context',
              'out->kv_layout = configured_b16_kv_layout(name);',
              'out->kv_layout = rad_kvq_from_name(name);',
        ),
        _edit('packed-capability',
              'return b16_kv_layout(layout) &&',
              'return rad_kvq_desc(layout) &&',
        ),
        _edit('packed-geometry',
              'return b16_kv_layout(d.kv_layout) &&',
              'return rad_kvq_desc(d.kv_layout) &&',
        ),
        _edit('generic-model-supported',
              'b16_kv_model_supported',
              'packed_kv_model_supported',
        ),
        _edit('generic-group-supported',
              'b16_kv_group_supported',
              'packed_kv_group_supported',
              mode='replace_all', count=2),
        _edit('generic-block-bytes',
              'b16_kv_block_bytes',
              'packed_kv_block_bytes',
        ),
    )),
    FilePatch(path='core/config.cpp', language="none", edits=(
        _edit('restore-v4-cli',
              """            if (!std::strcmp(v, "fp8_v4") || !std::strcmp(v, "k6_v4")) {
                RAD_ERR("--kv-cache-dtype %s: the packed caches are no longer served; i8 is the "
                        "8-bit cache (int8 K and V, a scale each per token and head)", v);
                return RAD_E_INVAL;
            }
""",
              '',
              guard='rad_kvq_from_name(v) == RAD_KVQ_NONE'),
        _edit('accept-canonical-names',
              'configured_b16_kv_layout(v) == RAD_KVQ_NONE',
              'rad_kvq_from_name(v) == RAD_KVQ_NONE',
        ),
        _edit('list-cli-cache-names',
              'bf16 (default) | i8 | fp8 | bf16_fp8 | bf16_i8 -- ',
              'bf16 (default) | i8 | fp8 | fp8_v4 | k6_v4 | bf16_fp8 | bf16_i8 -- ',
        ),
        _edit('list-invalid-cache-formats',
              'takes bf16, i8, fp8, bf16_fp8 or bf16_i8',
              'takes bf16, i8, fp8, fp8_v4, k6_v4, bf16_fp8 or bf16_i8',
        ),
    )),
    FilePatch(path='core/engine_bringup.cpp', language="none", edits=(
        _edit('use-rad-kvq-from-name',
              'configured_b16_kv_layout',
              'rad_kvq_from_name',
        ),
        _edit('use-packed-kv-model-supported',
              'b16_kv_model_supported',
              'packed_kv_model_supported',
        ),
    )),
    FilePatch(path='core/build/rad_builder.cpp', language="none", edits=(
        _edit('use-packed-kv-group-supported',
              'b16_kv_group_supported',
              'packed_kv_group_supported',
        ),
        _edit('use-packed-kv-block-bytes',
              'b16_kv_block_bytes',
              'packed_kv_block_bytes',
              mode='replace_all', count=2),
        _edit('use-packed-kv',
              'packed B16',
              'packed KV',
              mode='replace_all', count=2),
    )),
    FilePatch(path='core/mem/kv.cpp', language="none", edits=(
        _edit('use-packed-kv-group-supported',
              'b16_kv_group_supported',
              'packed_kv_group_supported',
        ),
        _edit('use-packed-kv-block-bytes',
              'b16_kv_block_bytes',
              'packed_kv_block_bytes',
        ),
        _edit('use-packed-kv',
              'packed B16',
              'packed KV',
              mode='replace_all', count=2),
    )),
    FilePatch(path='tests/bc_b16_kv_runtime_test.cpp', language="none", edits=(
        _edit('use-packed-kv-model-supported',
              'b16_kv_model_supported',
              'packed_kv_model_supported',
              mode='replace_all', count=3),
        _edit('v4-runtime-format-fixture',
              '        {"bf16_i8", RAD_KVQ_B16I8, RAD_U8, 784, 6272},',
              """        {"bf16_i8", RAD_KVQ_B16I8, RAD_U8, 784, 6272},
        {"fp8_v4", RAD_KVQ_V4, RAD_U8, 400, 3200},
        {"k6_v4", RAD_KVQ_K6V4, RAD_U8, 352, 2816},""",
              guard='{"k6_v4", RAD_KVQ_K6V4, RAD_U8, 352, 2816},'),
    )),
    FilePatch(path='arch/qwen4exp_fp8/qwen4exp_fp8.cpp', language="none", edits=(
        _edit('admit-packed-descriptor',
              '(ctx->kv_layout != RAD_KVQ_B16F8 && ctx->kv_layout != RAD_KVQ_B16I8)',
              '!rad_kvq_desc(ctx->kv_layout)',
        ),
        _edit('packed-arch-shape',
              'the packed B16 cache requires H256, GQA6 or GQA12, and page4',
              'the packed KV cache requires H256, GQA6 or GQA12, and page4',
        ),
        _edit('old-formats-probe',
              '((1u << RAD_KVQ_B16F8) | (1u << RAD_KVQ_B16I8)) : 0',
              """((1u << RAD_KVQ_B16F8) | (1u << RAD_KVQ_B16I8) |
           (1u << RAD_KVQ_V4) | (1u << RAD_KVQ_K6V4)) : 0""",
        ),
    )),
    FilePatch(path='arch/qwen35_fp8/qwen35_fp8.cpp', language="none", edits=(
        _edit('admit-packed-descriptor',
              '(ctx->kv_layout != RAD_KVQ_B16F8 && ctx->kv_layout != RAD_KVQ_B16I8)',
              '!rad_kvq_desc(ctx->kv_layout)',
        ),
        _edit('packed-arch-shape',
              'the packed B16 cache requires H256/GQA6/page16',
              'the packed KV cache requires H256/GQA6/page16',
        ),
        _edit('old-formats-probe',
              '((1u << RAD_KVQ_B16F8) | (1u << RAD_KVQ_B16I8)) : 0',
              """((1u << RAD_KVQ_B16F8) | (1u << RAD_KVQ_B16I8) |
           (1u << RAD_KVQ_V4) | (1u << RAD_KVQ_K6V4)) : 0""",
        ),
        _edit('packed-dflash-shape',
              'DFlash2 over a B16 trunk',
              'DFlash2 over a packed trunk',
              guard='re:(?:DFlash2 over a packed trunk|the native E4M3 DFlash2 cache)',
        ),
    )),
    FilePatch(path='libr4d/r4d_b16_kv.h', language="none", edits=(
        _edit('descriptor-key-scale',
              'value16 = 0, value_scale16 = 0',
              'value16 = 0, key_scale16 = 0, value_scale16 = 0',
        ),
        _edit('descriptor-scale-positions',
              'value16 = value, value_scale16 = vs',
              'value16 = value, key_scale16 = ks, value_scale16 = vs',
        ),
        _edit('legacy-v4-predicate',
              '/* v1.17 B16I8 quantization.',
              """constexpr bool r4d_v4_layout(RadKVLayout layout) {
    return layout == RAD_KVQ_V4 || layout == RAD_KVQ_K6V4;
}

""",
              mode='insert_before', guard='constexpr bool r4d_v4_layout('),
    )),
    FilePatch(path='libr4d/bc_b16_prefill_impl.h', language="none", edits=(
        _edit('typed-v4-cache',
              """    constexpr int V4    = ((OPT & O_V4) && !KVP) ? 1 : 0;
    constexpr int K6    = ((OPT & O_K6) && V4) ? 1 : 0;
    constexpr int I8    = ((OPT & O_I8) && K6) ? 1 : 0;
""",
              """    constexpr bool PackedV4 = r4d_v4_layout(Layout);
    constexpr int V4 = PackedV4 || ((OPT & O_V4) && !KVP);
    constexpr int K6 = Layout == RAD_KVQ_K6V4 || ((OPT & O_K6) && V4);
    constexpr int I8 = !PackedV4 && (OPT & O_I8) && K6;
""",
        ),
        _edit('descriptor-row-bytes',
              """    constexpr int ROWB  = KB ? HEAD_DIM / 16 * R4dKvBytes<Layout>::row16
                        : I8 ? 2 * HEAD_DIM + 4
                        : K6 ? HEAD_DIM * 22 / 16 : V4 ? HEAD_DIM * 25 / 16 : 2 * HEAD_DIM;
""",
              """    using Format = R4dKvBytes<Layout>;
    constexpr int ROWB = Layout != RAD_KVQ_NONE ? HEAD_DIM / 16 * Format::row16
                       : I8 ? 2 * HEAD_DIM + 4
                       : K6 ? HEAD_DIM * 22 / 16 : V4 ? HEAD_DIM * 25 / 16 : 2 * HEAD_DIM;
""",
        ),
        _edit('descriptor-v4-offsets',
              """    constexpr int VOFF  = I8 ? HEAD_DIM : K6 ? HEAD_DIM * 3 / 4 : HEAD_DIM;
    constexpr int VSOFF = I8 ? 2 * HEAD_DIM + 2 : K6 ? HEAD_DIM * 21 / 16 : HEAD_DIM * 3 / 2;
    constexpr int KSOFF = I8 ? 2 * HEAD_DIM : HEAD_DIM * 5 / 4;
""",
              """    constexpr int VOFF = PackedV4 ? HEAD_DIM / 16 * Format::value16
                       : I8 ? HEAD_DIM : K6 ? HEAD_DIM * 3 / 4 : HEAD_DIM;
    constexpr int VSOFF = PackedV4 ? HEAD_DIM / 16 * Format::value_scale16
                        : I8 ? 2 * HEAD_DIM + 2 : K6 ? HEAD_DIM * 21 / 16 : HEAD_DIM * 3 / 2;
    constexpr int KSOFF = PackedV4 ? HEAD_DIM / 16 * Format::key_scale16
                        : I8 ? 2 * HEAD_DIM : HEAD_DIM * 5 / 4;
""",
        ),
    )),
    FilePatch(path='libr4d/bc_b16_decode_impl.h', language="none", edits=(
        _edit('typed-v4-cache',
              """    constexpr int V4    = ((OPT & O_V4) && !KVP) ? 1 : 0;
    constexpr int K6    = ((OPT & O_K6) && V4) ? 1 : 0;
    constexpr int I8    = ((OPT & O_I8) && K6) ? 1 : 0;
""",
              """    constexpr bool PackedV4 = r4d_v4_layout(Layout);
    constexpr int V4 = PackedV4 || ((OPT & O_V4) && !KVP);
    constexpr int K6 = Layout == RAD_KVQ_K6V4 || ((OPT & O_K6) && V4);
    constexpr int I8 = !PackedV4 && (OPT & O_I8) && K6;
""",
        ),
        _edit('descriptor-row-bytes',
              """    constexpr int ROWB  = KB ? HEAD_DIM / 16 * R4dKvBytes<Layout>::row16
                        : I8 ? 2 * HEAD_DIM + 4
                        : K6 ? HEAD_DIM * 22 / 16 : V4 ? HEAD_DIM * 25 / 16 : 2 * HEAD_DIM;
""",
              """    using Format = R4dKvBytes<Layout>;
    constexpr int ROWB = Layout != RAD_KVQ_NONE ? HEAD_DIM / 16 * Format::row16
                       : I8 ? 2 * HEAD_DIM + 4
                       : K6 ? HEAD_DIM * 22 / 16 : V4 ? HEAD_DIM * 25 / 16 : 2 * HEAD_DIM;
""",
        ),
        _edit('descriptor-v4-offsets',
              """    constexpr int VOFF  = I8 ? HEAD_DIM : K6 ? HEAD_DIM * 3 / 4 : HEAD_DIM;
    constexpr int VSOFF = I8 ? 2 * HEAD_DIM + 2 : K6 ? HEAD_DIM * 21 / 16 : HEAD_DIM * 3 / 2;
    constexpr int KSOFF = I8 ? 2 * HEAD_DIM : HEAD_DIM * 5 / 4;
""",
              """    constexpr int VOFF = PackedV4 ? HEAD_DIM / 16 * Format::value16
                       : I8 ? HEAD_DIM : K6 ? HEAD_DIM * 3 / 4 : HEAD_DIM;
    constexpr int VSOFF = PackedV4 ? HEAD_DIM / 16 * Format::value_scale16
                        : I8 ? 2 * HEAD_DIM + 2 : K6 ? HEAD_DIM * 21 / 16 : HEAD_DIM * 3 / 2;
    constexpr int KSOFF = PackedV4 ? HEAD_DIM / 16 * Format::key_scale16
                        : I8 ? 2 * HEAD_DIM : HEAD_DIM * 5 / 4;
""",
        ),
    )),
    FilePatch(path='libr4d/r4d_attn_paged_gqa8.hip', language="none", edits=(
        _edit('v4-qwen4exp-page4',
              'G8_BYTE_ENTRIES(b16i8, (16 * R4dKvBytes<RAD_KVQ_B16I8>::row16), 0, RAD_KVQ_B16I8)',
              """G8_BYTE_ENTRIES(b16i8, (16 * R4dKvBytes<RAD_KVQ_B16I8>::row16), 0, RAD_KVQ_B16I8)
G8_BYTE_ENTRIES(fp8v4, (16 * R4dKvBytes<RAD_KVQ_V4>::row16), 0, RAD_KVQ_V4)
G8_BYTE_ENTRIES(k6v4, (16 * R4dKvBytes<RAD_KVQ_K6V4>::row16), 0, RAD_KVQ_K6V4)""",
              guard='G8_BYTE_ENTRIES(k6v4, (16 *'),
    )),
    FilePatch(path='libr4d/r4d_attn_paged_h256_gqa6.hip', language="none", edits=(
        _edit('v4-qwen35-page16',
              'BC_B16_PAGE16(b16i8, RAD_KVQ_B16I8)',
              """BC_B16_PAGE16(b16i8, RAD_KVQ_B16I8)
BC_B16_PAGE16(fp8v4, RAD_KVQ_V4)
BC_B16_PAGE16(k6v4, RAD_KVQ_K6V4)""",
              guard='BC_B16_PAGE16(k6v4, RAD_KVQ_K6V4)'),
    )),
    FilePatch(path='libr4d/r4d_model_bf16.hip', language="none", edits=(
        _edit('v4-standalone-codec',
              '// The store of a streamed KV group',
              """// The fp8_v4 store: K as kv_store_fp8 writes it, V as int4 in groups of 32 with an f16 scale
// (absmax / 7). Four columns a thread as above, so a group of 32 is 8 lanes; head_dim is a multiple
// of 32, so a group never straddles two heads. A token's row is `row` bytes. K6 is the k6_v4 store:
// K as int6 in groups of 32 (absmax / 31) in the row layout of abi/rad_builder.h's RAD_KVQ_K6V4, V
// as above after it.
template<RadKVLayout Layout>
__global__ __launch_bounds__(kThreads) void kv_store_fp8v4_kernel(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const int32_t* __restrict__ slot, uint8_t* __restrict__ kv,
    uint32_t head_dim, uint32_t kv_heads, uint32_t block_size, uint32_t n_blocks,
    uint32_t k_ld, uint32_t v_ld, float k_inv, float v_inv, uint32_t row) {
    using Format = R4dKvBytes<Layout>;
    constexpr bool K6 = Layout == RAD_KVQ_K6V4;
    const uint32_t t = blockIdx.x;
    const int32_t s = slot[t];
    if (s < 0 || (uint32_t)s / block_size >= n_blocks) return;
    const uint32_t blk = (uint32_t)s / block_size, off = (uint32_t)s % block_size;
    const size_t slot_base = ((size_t)blk * kv_heads * block_size + (size_t)off) * row;
    const size_t head_step = (size_t)block_size * row;

    for (uint32_t j = threadIdx.x * 4u; j < kv_heads * head_dim; j += kThreads * 4u) {
        const uint32_t h = j / head_dim, d = j % head_dim;
        uint8_t* __restrict__ c = kv + slot_base + (size_t)h * head_step;
        float kf[4], vf[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            kf[e] = bf16_to_f32(k[(size_t)t * k_ld + j + (uint32_t)e]) * k_inv;
            vf[e] = bf16_to_f32(v[(size_t)t * v_ld + j + (uint32_t)e]) * v_inv;
        }
        const uint32_t voff = head_dim / 16 * Format::value16;
        const uint32_t vsoff = head_dim / 16 * Format::value_scale16;
        if constexpr (K6) {
            // A group of 32 is 8 lanes; its scale absmax / 31 in f16, codes 6-bit two's complement.
            float kmax = fmaxf(fmaxf(fabsf(kf[0]), fabsf(kf[1])), fmaxf(fabsf(kf[2]), fabsf(kf[3])));
            for (int m = 4; m >= 1; m >>= 1) kmax = fmaxf(kmax, __shfl_xor(kmax, m, 32));
            // Clamped so 31 times it stays an f16 (65504): a group gone inf
            // or NaN decodes finite, as the E4M3 form's clamp to 448 does.
            const __fp16 k16 = (__fp16)fminf(kmax / 31.0f, 2112.0f);
            const float ksc = (float)k16;
            uint32_t q[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float r = ksc > 0.0f ? rintf(kf[e] / ksc) : 0.0f;
                q[e] = (uint32_t)((int)fminf(fmaxf(r, -31.0f), 31.0f)) & 0x3Fu;
            }
            *(uint16_t*)(c + d / 2) = (uint16_t)((q[0] & 15u) | ((q[1] & 15u) << 4) |
                                                 ((q[2] & 15u) << 8) | ((q[3] & 15u) << 12));
            // The high bits of dims x..x+3 of a 16-dim chunk (x = d % 16 in 0, 4, 8, 12): bits
            // 0-3 of bytes x/2 and x/2+1 for x < 8, bits 4-7 of bytes (x-8)/2 and (x-8)/2+1 for
            // x >= 8. The lane two over holds the dims eight on, so the lower lane writes both.
            const uint32_t hb = ((q[0] >> 4) | ((q[1] >> 4) << 2)) | ((((q[2] >> 4) | ((q[3] >> 4) << 2))) << 8);
            const uint32_t up = (uint32_t)__shfl_xor((int)hb, 2, 32);
            if ((d & 15u) < 8u)
                *(uint16_t*)(c + head_dim / 2 + (d / 16) * 4 + (d & 7u) / 2) = (uint16_t)(hb | (up << 4));
            if ((d & 31u) == 0)
                *(uint16_t*)(c + head_dim / 16 * Format::key_scale16 + (d / 32) * 2) = __builtin_bit_cast(uint16_t, k16);
        } else {
            int pk = 0;
            pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(kf[0], -448.0f), 448.0f),
                                                 fminf(fmaxf(kf[1], -448.0f), 448.0f), pk, false);
            pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(kf[2], -448.0f), 448.0f),
                                                 fminf(fmaxf(kf[3], -448.0f), 448.0f), pk, true);
            *(uint32_t*)(c + d) = (uint32_t)pk;
        }

        float amax = fmaxf(fmaxf(fabsf(vf[0]), fabsf(vf[1])), fmaxf(fabsf(vf[2]), fabsf(vf[3])));
        for (int m = 4; m >= 1; m >>= 1) amax = fmaxf(amax, __shfl_xor(amax, m, 32));
        // Clamped so 7 times it stays an f16: see the K scale above.
        const __fp16 s16 = (__fp16)fminf(amax / 7.0f, 8188.0f);
        const float sc = (float)s16;
        uint32_t q[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float r = sc > 0.0f ? rintf(vf[e] / sc) : 0.0f;
            q[e] = (uint32_t)((int)fminf(fmaxf(r, -7.0f), 7.0f)) & 0xFu;
        }
        *(uint16_t*)(c + voff + d / 2) = (uint16_t)(q[0] | (q[1] << 4) | (q[2] << 8) | (q[3] << 12));
        if ((d & 31u) == 0)
            *(uint16_t*)(c + vsoff + (d / 32) * 2) = __builtin_bit_cast(uint16_t, s16);
    }
}

""",
              mode='insert_before', guard='void kv_store_fp8v4_kernel('),
        _edit('v4-standalone-launch',
              "/* The int8 store: the cache's last extent",
              """template<RadKVLayout Layout>
static int kv_store_v4(const RadArgs* a, RadStream stream) {
    using Format = R4dKvBytes<Layout>;
    KvStoreGeom g{};
    const int rc = kv_store_geom(a, &g, Format::row16);
    if (rc != RAD_OK) return rc;
    if (g.kv->dtype != RAD_U8) return R4D_NO(RAD_E_DTYPE);
    if (g.head_dim != 256 || (g.block_size != 4 && g.block_size != 16))
        return R4D_NO(RAD_E_SHAPE);
    const double ks = r4d_getf(a, "k_scale", 1.0), vs = r4d_getf(a, "v_scale", 1.0);
    if (!(ks > 0.0) || !(vs > 0.0)) return R4D_NO(RAD_E_INVAL);
    // v1.17 kv_store_q4 takes the reciprocal in double before rounding to float.
    return R4D_EMIT(stream, kv_store_fp8v4_kernel<Layout>, dim3((uint32_t)g.T), dim3(kThreads), 0,
                    (const uint16_t*)r4d_p(g.k), (const uint16_t*)r4d_p(g.v),
                    (const int32_t*)r4d_p(g.sm), (uint8_t*)r4d_wp(g.kv),
                    (uint32_t)g.head_dim, (uint32_t)g.kv_heads, (uint32_t)g.block_size,
                    (uint32_t)g.n_blocks, (uint32_t)g.k_ld, (uint32_t)g.v_ld,
                    (float)(1.0 / ks), (float)(1.0 / vs),
                    (uint32_t)(g.head_dim / 16 * Format::row16));
}

extern "C" int r4d_rad_kv_store_fp8v4(const RadArgs* a, RadStream stream) {
    return kv_store_v4<RAD_KVQ_V4>(a, stream);
}
extern "C" int r4d_rad_kv_store_k6v4(const RadArgs* a, RadStream stream) {
    return kv_store_v4<RAD_KVQ_K6V4>(a, stream);
}

""",
              mode='insert_before', guard='static int kv_store_v4('),
    )),
    FilePatch(path='libr4d/r4d_qk_norm_rope_gate.hip', language="none", edits=(
        _edit('v4-fused-codec',
              '  } else if constexpr (r4d_b16_layout(Layout))',
              """  } else if constexpr (r4d_v4_layout(Layout)) {
    if (isk && t < HD) {
      using Format = R4dKvBytes<Layout>;
      constexpr size_t row = HD / 16 * Format::row16;
      const int32_t sl = kvs.slot[tok];
      const uint32_t page = (uint32_t)kvs.block_size;
      if (sl >= 0 && (uint32_t)sl / page < (uint32_t)kvs.n_blocks) {
        const uint32_t blk = (uint32_t)sl / page, off = (uint32_t)sl % page;
        // fp8_v4: K as the E4M3 form writes it. k6_v4: K as int6 in groups of 32 (one wave)
        // with an f16 scale, absmax / 31, in kv_store_k6v4's layout. V as int4 in groups of 32
        // with an f16 scale, absmax / 7. The even lane of a pair writes the pair's nibble byte.
        const __hip_bfloat16 kb = __float2bfloat16(res);
        const __hip_bfloat16 vb = kvs.v[(size_t)tok * kvs.v_pitch + (size_t)lh * HD + t];
        auto* c = (uint8_t*)kvs.cache + (((size_t)blk * nkv + lh) * page + off) * row;
        const float kf = __bfloat162float(kb) * kvs.k_inv;
        constexpr int voff = HD / 16 * Format::value16;
        constexpr int vsoff = HD / 16 * Format::value_scale16;
        if constexpr (Layout == RAD_KVQ_K6V4) {
          float kmax = fabsf(kf);
          for (int m = 16; m >= 1; m >>= 1) kmax = fmaxf(kmax, __shfl_xor(kmax, m, 32));
          const __fp16 k16 = (__fp16)fminf(kmax / 31.0f, 2112.0f);   // finite codes, as kv_store_k6v4
          const float ksc = (float)k16;
          const float r = ksc > 0.0f ? rintf(kf / ksc) : 0.0f;
          const int q = ((int)fminf(fmaxf(r, -31.0f), 31.0f)) & 0x3F;
          const int qn = __shfl_xor(q & 15, 1, 32);
          if ((t & 1) == 0) c[t / 2] = (unsigned char)((q & 15) | (qn << 4));
          // High bits: a pair's four into bits 0-3 at its even lane, then the pair eight dims on
          // into bits 4-7; lane 2j of a 16-dim chunk writes byte j.
          const int h01 = (q >> 4) | (__shfl_xor(q >> 4, 1, 32) << 2);
          const int hb  = h01 | (__shfl_xor(h01, 8, 32) << 4);
          if ((t & 1) == 0 && (t & 15) < 8) c[HD / 2 + (t / 16) * 4 + (t & 7) / 2] = (unsigned char)hb;
          if ((t & 31) == 0) *(uint16_t*)(c + HD / 16 * Format::key_scale16 + (t / 32) * 2) = __builtin_bit_cast(uint16_t, k16);
        } else {
          const int pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(kf, -448.0f), 448.0f), 0.0f,
                                                         0, false);
          c[t] = (unsigned char)(pk & 0xFF);
        }
        const float vf = __bfloat162float(vb) * kvs.v_inv;
        float amax = fabsf(vf);
        for (int m = 16; m >= 1; m >>= 1) amax = fmaxf(amax, __shfl_xor(amax, m, 32));
        const __fp16 s16 = (__fp16)fminf(amax / 7.0f, 8188.0f);   // finite codes, as kv_store_fp8v4
        const float sc = (float)s16;
        const float r = sc > 0.0f ? rintf(vf / sc) : 0.0f;
        const int q = ((int)fminf(fmaxf(r, -7.0f), 7.0f)) & 0xF;
        const int qn = __shfl_xor(q, 1, 32);
        if ((t & 1) == 0) c[voff + t / 2] = (unsigned char)(q | (qn << 4));
        if ((t & 31) == 0) *(uint16_t*)(c + vsoff + (t / 32) * 2) = __builtin_bit_cast(uint16_t, s16);
      }
    }
""",
              mode='insert_before', guard='else if constexpr (r4d_v4_layout(Layout))'),
        _edit('fused-packed-type',
              '  const bool b16 = r4d_b16_layout(layout);',
              '  const bool packed = rad_kvq_desc(layout) != nullptr;',
        ),
        _edit('fused-packed-geometry',
              'b16 &&',
              'packed &&',
              mode='replace_all', count=3),
        _edit('fused-packed-dtype',
              'want = b16 ?',
              'want = packed ?',
        ),
        _edit('v4-fused-launch',
              '  const bool fp8 = kv->dtype == RAD_F8E4M3;',
              """  if (r4d_v4_layout(layout)) {
    const double ks = r4d_getf(args, "k_scale", 1.0), vs = r4d_getf(args, "v_scale", 1.0);
    if (!(ks > 0.0) || !(vs > 0.0)) return R4D_NO(RAD_E_INVAL);
    // v1.17 qk_kv_q4 rounds the scale first and takes the reciprocal in float.
    kvs.k_inv = 1.0f / (float)ks;
    kvs.v_inv = 1.0f / (float)vs;
    return layout == RAD_KVQ_K6V4 ? qk_body<QK_KV_NONE, RAD_KVQ_K6V4>(args, stream, kvs)
                                 : qk_body<QK_KV_NONE, RAD_KVQ_V4>(args, stream, kvs);
  }
""",
              mode='insert_before', guard='return layout == RAD_KVQ_K6V4 ? qk_body'),
    )),
    FilePatch(path='libr4d/r4d_plugin.h', language="none", edits=(
        _edit('v4-abi-launchers',
              'int     r4d_rad_attn_prefill_h256_gqa12_b16f8(',
              """int     r4d_rad_kv_store_fp8v4(const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_gqa6_fp8v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_prefill_h256_gqa6_fp8v4_scratch(const RadArgs*);
int     r4d_rad_attn_decode_h256_gqa6_fp8v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa6_fp8v4_scratch(const RadArgs*);
int     r4d_rad_attn_decode_gq_h256_gqa6_fp8v4(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_ghq_h256_gqa6_fp8v4(const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_gqa12_fp8v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_prefill_h256_gqa12_fp8v4_scratch(const RadArgs*);
int     r4d_rad_attn_decode_h256_gqa12_fp8v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa12_fp8v4_scratch(const RadArgs*);
int     r4d_rad_attn_decode_gq_h256_gqa12_fp8v4(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_ghq_h256_gqa12_fp8v4(const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_gqa6_page16_fp8v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_prefill_h256_gqa6_page16_fp8v4_scratch(const RadArgs*);
int     r4d_rad_attn_decode_h256_gqa6_page16_fp8v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa6_page16_fp8v4_scratch(const RadArgs*);
int     r4d_rad_kv_store_k6v4(const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_gqa6_k6v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_prefill_h256_gqa6_k6v4_scratch(const RadArgs*);
int     r4d_rad_attn_decode_h256_gqa6_k6v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa6_k6v4_scratch(const RadArgs*);
int     r4d_rad_attn_decode_gq_h256_gqa6_k6v4(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_ghq_h256_gqa6_k6v4(const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_gqa12_k6v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_prefill_h256_gqa12_k6v4_scratch(const RadArgs*);
int     r4d_rad_attn_decode_h256_gqa12_k6v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa12_k6v4_scratch(const RadArgs*);
int     r4d_rad_attn_decode_gq_h256_gqa12_k6v4(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_ghq_h256_gqa12_k6v4(const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_gqa6_page16_k6v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_prefill_h256_gqa6_page16_k6v4_scratch(const RadArgs*);
int     r4d_rad_attn_decode_h256_gqa6_page16_k6v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa6_page16_k6v4_scratch(const RadArgs*);
""",
              mode='insert_before', guard='r4d_rad_kv_store_k6v4('),
    )),
    FilePatch(path='libr4d/bc_kv_rows.h', language="none", edits=(
        _edit('v4-gqa12-constraints',
              '#undef BC_B16_QSA_CONSTRAINTS',
              """BC_B16_QSA_CONSTRAINTS(FP8V4, "fp8_v4")
BC_B16_QSA_CONSTRAINTS(K6V4, "k6_v4")
""",
              mode='insert_before', guard='BC_B16_QSA_CONSTRAINTS(K6V4,'),
        _edit('v4-gqa6-constraints',
              'static const RadConstraint cDflashH128Gqa4Fp8[]',
              """static const RadConstraint c_attn_prefill_h256_gqa6_bs4_fp8v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_v4"),
    C_EQ("causal", 1),
};
static const RadConstraint c_attn_decode_h256_gqa6_bs4_fp8v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_v4"),
    C_GE("causal", 0), C_LE("q_len", 10),
};
static const RadConstraint c_attn_decode_gq_h256_gqa6_bs4_fp8v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_v4"),
    C_GE("causal", 0), C_LE("q_len", 10),
    C_EQ("group", 128), C_IN("act", "sigmoid"),
};
static const RadConstraint c_attn_decode_ghq_h256_gqa6_bs4_fp8v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_v4"),
    C_GE("causal", 0), C_LE("q_len", 10),
    C_EQ("group", 128), C_IN("act", "sigmoid"),
};
static const RadConstraint c_attn_prefill_h256_gqa6_fp8v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_v4"),
    C_EQ("causal", 1),
};
static const RadConstraint c_attn_decode_h256_gqa6_fp8v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "fp8_v4"),
    C_GE("causal", 0), C_LE("q_len", 10),
};
static const RadConstraint c_kv_store_page16_fp8v4[] = {
    C_EQ("head_dim", 256), C_EQ("block_size", 16), C_IN("kv_dtype", "fp8_v4"),
};
static const RadConstraint c_qk_norm_rope_gate_kv_store_page16_fp8v4[] = {
    C_EQ("head_dim", 256), C_EQ("block_size", 16), C_IN("kv_dtype", "fp8_v4"), C_IN("q_dtype", "bf16"),
};
static const RadConstraint c_attn_prefill_h256_gqa6_bs4_k6v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "k6_v4"),
    C_EQ("causal", 1),
};
static const RadConstraint c_attn_decode_h256_gqa6_bs4_k6v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "k6_v4"),
    C_GE("causal", 0), C_LE("q_len", 10),
};
static const RadConstraint c_attn_decode_gq_h256_gqa6_bs4_k6v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "k6_v4"),
    C_GE("causal", 0), C_LE("q_len", 10),
    C_EQ("group", 128), C_IN("act", "sigmoid"),
};
static const RadConstraint c_attn_decode_ghq_h256_gqa6_bs4_k6v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 4),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "k6_v4"),
    C_GE("causal", 0), C_LE("q_len", 10),
    C_EQ("group", 128), C_IN("act", "sigmoid"),
};
static const RadConstraint c_attn_prefill_h256_gqa6_k6v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "k6_v4"),
    C_EQ("causal", 1),
};
static const RadConstraint c_attn_decode_h256_gqa6_k6v4[] = {
    C_EQ("window", 0), C_EQ("head_dim", 256), C_EQ("gqa", 6), C_EQ("block_size", 16),
    C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "k6_v4"),
    C_GE("causal", 0), C_LE("q_len", 10),
};
static const RadConstraint c_kv_store_page16_k6v4[] = {
    C_EQ("head_dim", 256), C_EQ("block_size", 16), C_IN("kv_dtype", "k6_v4"),
};
static const RadConstraint c_qk_norm_rope_gate_kv_store_page16_k6v4[] = {
    C_EQ("head_dim", 256), C_EQ("block_size", 16), C_IN("kv_dtype", "k6_v4"), C_IN("q_dtype", "bf16"),
};
""",
              mode='insert_before', guard='c_attn_decode_h256_gqa6_k6v4[]'),
        _edit('v4-gqa12-rows',
              '#undef BC_B16_QSA_ROWS',
              """BC_B16_QSA_ROWS(FP8V4, fp8v4, "E4M3 K, int4 V with FP16 scales")
BC_B16_QSA_ROWS(K6V4, k6v4, "int6 K, int4 V with FP16 scales")
""",
              mode='insert_before', guard='BC_B16_QSA_ROWS(K6V4,'),
        _edit('v4-gqa6-rows',
              '{ "attn_decode_h128_gqa4_fp8kv",',
              """{ "attn_prefill_h256_gqa6_bs4_fp8v4", "attn_paged", "attn",
  "Resident packed KV prefill attention", "H256/GQA6/page4", "fp8_v4",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL, ROW(c_attn_prefill_h256_gqa6_bs4_fp8v4), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa6_fp8v4,
  r4d_rad_attn_prefill_h256_gqa6_fp8v4_scratch, nullptr, nullptr, r4d_shape_attn_paged, nullptr, nullptr, RAD_KERNEL_CAP_DYNAMIC_PREFILL_SPLIT },

{ "attn_decode_h256_gqa6_bs4_fp8v4", "attn_paged", "attn",
  "Resident packed KV decode attention", "H256/GQA6/page4", "fp8_v4",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE, ROW(c_attn_decode_h256_gqa6_bs4_fp8v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa6_fp8v4,
  r4d_rad_attn_decode_h256_gqa6_fp8v4_scratch, nullptr, nullptr, r4d_shape_attn_paged },

{ "attn_decode_gq_h256_gqa6_bs4_fp8v4", "attn_paged_gate_quant", "attn",
  "Resident packed KV decode_gq attention", "H256/GQA6/page4", "fp8_v4",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE, ROW(c_attn_decode_gq_h256_gqa6_bs4_fp8v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_gq_h256_gqa6_fp8v4,
  r4d_rad_attn_decode_h256_gqa6_fp8v4_scratch, nullptr, nullptr, r4d_shape_attn_paged_gate_quant, nullptr, r4d_fuse_attn_paged_gate_quant },

{ "attn_decode_ghq_h256_gqa6_bs4_fp8v4", "attn_paged_gate_had_quant", "attn",
  "Resident packed KV decode_ghq attention", "H256/GQA6/page4", "fp8_v4",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE, ROW(c_attn_decode_ghq_h256_gqa6_bs4_fp8v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_ghq_h256_gqa6_fp8v4,
  r4d_rad_attn_decode_h256_gqa6_fp8v4_scratch, nullptr, nullptr, nullptr, nullptr, r4d_fuse_attn_paged_gate_had_quant },

{ "attn_prefill_h256_gqa6_fp8v4", "attn_paged", "attn",
  "Resident packed KV prefill attention", "H256/GQA6/page16", "fp8_v4",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL, ROW(c_attn_prefill_h256_gqa6_fp8v4), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa6_page16_fp8v4,
  r4d_rad_attn_prefill_h256_gqa6_page16_fp8v4_scratch, nullptr, nullptr, r4d_shape_attn_paged, nullptr, nullptr, RAD_KERNEL_CAP_DYNAMIC_PREFILL_SPLIT },

{ "attn_decode_h256_gqa6_fp8v4", "attn_paged", "attn",
  "Resident packed KV decode attention", "H256/GQA6/page16", "fp8_v4",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE, ROW(c_attn_decode_h256_gqa6_fp8v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa6_page16_fp8v4,
  r4d_rad_attn_decode_h256_gqa6_page16_fp8v4_scratch, nullptr, nullptr, r4d_shape_attn_paged },

{ "kv_store_page16_fp8v4", "kv_store", "attn",
  "Resident packed KV store", "H256/page16", "fp8_v4",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(c_kv_store_page16_fp8v4), NOTUNE,
  nullptr, nullptr, r4d_rad_kv_store_fp8v4, nullptr, nullptr, nullptr, r4d_shape_kv_store },

{ "qk_norm_rope_gate_kv_store_page16_fp8v4", "qk_norm_rope_gate_kv_store", "attn",
  "Resident packed KV store", "H256/page16", "fp8_v4",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(c_qk_norm_rope_gate_kv_store_page16_fp8v4), NOTUNE,
  nullptr, nullptr, r4d_rad_qk_norm_rope_gate_kv_store, nullptr, nullptr, nullptr, r4d_shape_qk_norm_rope_gate_kv_store, nullptr, r4d_fuse_qk_norm_rope_gate_kv_store },

{ "attn_prefill_h256_gqa6_bs4_k6v4", "attn_paged", "attn",
  "Resident packed KV prefill attention", "H256/GQA6/page4", "k6_v4",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL, ROW(c_attn_prefill_h256_gqa6_bs4_k6v4), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa6_k6v4,
  r4d_rad_attn_prefill_h256_gqa6_k6v4_scratch, nullptr, nullptr, r4d_shape_attn_paged, nullptr, nullptr, RAD_KERNEL_CAP_DYNAMIC_PREFILL_SPLIT },

{ "attn_decode_h256_gqa6_bs4_k6v4", "attn_paged", "attn",
  "Resident packed KV decode attention", "H256/GQA6/page4", "k6_v4",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE, ROW(c_attn_decode_h256_gqa6_bs4_k6v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa6_k6v4,
  r4d_rad_attn_decode_h256_gqa6_k6v4_scratch, nullptr, nullptr, r4d_shape_attn_paged },

{ "attn_decode_gq_h256_gqa6_bs4_k6v4", "attn_paged_gate_quant", "attn",
  "Resident packed KV decode_gq attention", "H256/GQA6/page4", "k6_v4",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE, ROW(c_attn_decode_gq_h256_gqa6_bs4_k6v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_gq_h256_gqa6_k6v4,
  r4d_rad_attn_decode_h256_gqa6_k6v4_scratch, nullptr, nullptr, r4d_shape_attn_paged_gate_quant, nullptr, r4d_fuse_attn_paged_gate_quant },

{ "attn_decode_ghq_h256_gqa6_bs4_k6v4", "attn_paged_gate_had_quant", "attn",
  "Resident packed KV decode_ghq attention", "H256/GQA6/page4", "k6_v4",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE, ROW(c_attn_decode_ghq_h256_gqa6_bs4_k6v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_ghq_h256_gqa6_k6v4,
  r4d_rad_attn_decode_h256_gqa6_k6v4_scratch, nullptr, nullptr, nullptr, nullptr, r4d_fuse_attn_paged_gate_had_quant },

{ "attn_prefill_h256_gqa6_k6v4", "attn_paged", "attn",
  "Resident packed KV prefill attention", "H256/GQA6/page16", "k6_v4",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_PREFILL, ROW(c_attn_prefill_h256_gqa6_k6v4), NOTUNE,
  nullptr, nullptr, r4d_rad_attn_prefill_h256_gqa6_page16_k6v4,
  r4d_rad_attn_prefill_h256_gqa6_page16_k6v4_scratch, nullptr, nullptr, r4d_shape_attn_paged, nullptr, nullptr, RAD_KERNEL_CAP_DYNAMIC_PREFILL_SPLIT },

{ "attn_decode_h256_gqa6_k6v4", "attn_paged", "attn",
  "Resident packed KV decode attention", "H256/GQA6/page16", "k6_v4",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ATTN_DECODE, ROW(c_attn_decode_h256_gqa6_k6v4), ROW(tAttnSplit), nullptr,
  nullptr, nullptr, r4d_rad_attn_decode_h256_gqa6_page16_k6v4,
  r4d_rad_attn_decode_h256_gqa6_page16_k6v4_scratch, nullptr, nullptr, r4d_shape_attn_paged },

{ "kv_store_page16_k6v4", "kv_store", "attn",
  "Resident packed KV store", "H256/page16", "k6_v4",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(c_kv_store_page16_k6v4), NOTUNE,
  nullptr, nullptr, r4d_rad_kv_store_k6v4, nullptr, nullptr, nullptr, r4d_shape_kv_store },

{ "qk_norm_rope_gate_kv_store_page16_k6v4", "qk_norm_rope_gate_kv_store", "attn",
  "Resident packed KV store", "H256/page16", "k6_v4",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(c_qk_norm_rope_gate_kv_store_page16_k6v4), NOTUNE,
  nullptr, nullptr, r4d_rad_qk_norm_rope_gate_kv_store, nullptr, nullptr, nullptr, r4d_shape_qk_norm_rope_gate_kv_store, nullptr, r4d_fuse_qk_norm_rope_gate_kv_store },

""",
              mode='insert_before', guard='{ "attn_decode_h256_gqa6_k6v4",'),
        _edit("bc-kv-shared-chain-rad0053", "/* BC_KV_CHAIN_APPEND */",
              '        "qk_norm_rope_gate_kv_store_fp8v4",\n        "qk_norm_rope_gate_kv_store_k6v4",\n        "qk_norm_rope_gate_kv_store_page16_fp8v4",\n        "qk_norm_rope_gate_kv_store_page16_k6v4",\n        "attn_decode_h256_gqa12_fp8v4",\n        "attn_decode_h256_gqa12_k6v4",\n        "attn_decode_h256_gqa6_bs4_fp8v4",\n        "attn_decode_h256_gqa6_bs4_k6v4",\n        "attn_decode_h256_gqa6_fp8v4",\n        "attn_decode_h256_gqa6_k6v4",\n/* BC_KV_CHAIN_rad0053 */\n', guard='/* BC_KV_CHAIN_rad0053 */', mode="insert_before"),
    )),
    FilePatch(path='tests/config_test.cpp', language="none", edits=(
        _edit('describe-cache-format-test',
              'TEST(the_kv_cache_dtype_is_one_of_three_names)',
              'TEST(the_kv_cache_dtype_uses_canonical_names)',
        ),
        _edit('update-retired-format-expectation',
              """    /* Three values only: an "auto" resolving to bf16 would be a second name for the default, and
     * each value should be unambiguous in a log. The packed caches are refused by name, with the
     * cache that serves in their place. */
    CHECK_EQ(parse({ "--kv-cache-dtype", "auto" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--kv-cache-dtype", "fp16" }, &c), RAD_E_INVAL);
    for (const char* gone : { "fp8_v4", "k6_v4" }) {
        int rc = RAD_OK;
        const std::string err = stderr_of([&] { rc = parse({ "--kv-cache-dtype", gone }, &c); });
        CHECK_EQ(rc, RAD_E_INVAL);
        CHECK(err.find("i8 is the 8-bit cache") != std::string::npos);
    }
    CHECK_EQ(c.kv_cache_dtype, std::string("i8"));
""",
              """    CHECK_EQ(parse({ "--kv-cache-dtype", "auto" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--kv-cache-dtype", "fp16" }, &c), RAD_E_INVAL);
    for (const char* packed : { "fp8_v4", "k6_v4", "bf16_fp8", "bf16_i8" }) {
        CHECK_EQ(parse({ "--kv-cache-dtype", packed }, &c), RAD_OK);
        CHECK_EQ(c.kv_cache_dtype, std::string(packed));
    }
""",
        ),
    )),
    FilePatch(path='tests/arch_test.cpp', language="none", edits=(
        _edit('packed-probe-matrix',
              'CHECK_EQ(probe.kv_layout_ok, (1u << RAD_KVQ_B16F8) | (1u << RAD_KVQ_B16I8));',
              """CHECK_EQ(probe.kv_layout_ok, (1u << RAD_KVQ_B16F8) | (1u << RAD_KVQ_B16I8) |
          (1u << RAD_KVQ_V4) | (1u << RAD_KVQ_K6V4));""",
              mode='replace_all', count=3),
        _edit('packed-gqa6-matrix',
              'for (const auto layout : {RAD_KVQ_B16F8, RAD_KVQ_B16I8})',
              'for (const auto layout : {RAD_KVQ_B16F8, RAD_KVQ_B16I8, RAD_KVQ_V4, RAD_KVQ_K6V4})',
              mode='replace_all', count=2),
        _edit('packed-gqa12-matrix',
              """    for (const auto layout : { RAD_KVQ_B16F8, RAD_KVQ_B16I8 }) {
        const auto* row = rad_kvq_desc(layout);""",
              """    for (const auto layout : { RAD_KVQ_B16F8, RAD_KVQ_B16I8, RAD_KVQ_V4, RAD_KVQ_K6V4 }) {
        const auto* row = rad_kvq_desc(layout);""",
        ),
        _edit('packed-gqa12-extent',
              """        CHECK_EQ(rad_kv_group_row_elems(cache.kv_layout, cache.dtype, cache.head_dim),
                 layout == RAD_KVQ_B16F8 ? 768 : 784);""",
              '        CHECK_EQ(rad_kv_group_row_elems(cache.kv_layout, cache.dtype, cache.head_dim), 16 * row->row16);',
        ),
    )),
)
