"""Native-WMMA single-band IQR prefill, selected only by form and tile geometry.

The fused route decodes into the same Stage8 fragments as the ordinary
fragment-plane producer. The native gfx12 WMMA, integer K accumulation,
scale folds and BF16 stores remain those of the existing prefill body.
"""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch


def replace(name: str, old: str, new: str, *, guard: str | None = None) -> Edit:
    return Edit(
        id=name,
        anchor=re.escape(old),
        mode="replace",
        text=new,
        guard=re.escape(guard if guard is not None else new),
        expect_matches=1,
        max_span_lines=max(4, old.count("\n") + 2),
        rationale="Keep native WMMA's K order and folds while sourcing decoded bytes from block-local storage.",
    )


FUSED_TILE = """#ifndef R3_TARGETS   // libr4d (gfx12) only: libr3 compiles with R3_TARGETS defined and has its own gfx11 tile
// One-band IQR prefill: each of eight waves decodes one 16-row unit.
// The transpose is local to that wave; a block-wide barrier publishes the
// native fragment bytes for all consumers, without a global decoded plane.
template <class F>
struct IqrFused12Pre {
    Ctx c;
    uint8_t* tr;

    __device__ void begin(uint32_t, uint32_t, uint8_t*) const {}
    __device__ void end(uint32_t) const {}

    __device__ __forceinline__ void prepare(uint32_t n0, uint32_t kb) const {
        using D = Dec<F>;
        static_assert(kPWaves == kPWW * kPTW);
        static_assert(kTrBytes >= 2u * kTileBytes);
        static_assert(kTrBytes % 16u == 0u);
        const uint32_t tid = threadIdx.x;
        const uint32_t lane = tid % kWave, wave = tid / kWave;
        const uint32_t rt = n0 / 16u + wave;
        uint8_t* const t = tr + wave * kTrBytes;
        // All waves have finished reading the preceding K block.
        __syncthreads();
        uint32_t w[16];
        const typename D::Raw code = D::fetch(c, rt, kb, lane, true);
        D::rows(code, c, rb_ratio<F>(c, rt * 16u + (lane & 15u), kb), w);
        D::unzero(c, rt, kb, w);
        Stage8 f[2];
        frag(w, t, lane, f);
        // frag has completed its wave-local transpose before the copy.
#pragma unroll
        for (uint32_t h = 0; h < 2; ++h) {
            uint8_t* const d = t + h * kTileBytes + lane * kLaneBytes;
            *(uint4*)d = f[h].v;
            *(uint4*)(d + 16) = f[h].w;
        }
        __syncthreads();
    }

    __device__ __forceinline__ Stage8 load(uint32_t part, uint32_t half,
                                            uint32_t lane) const {
        const uint8_t* const p = tr + part * kTrBytes + half * kTileBytes
                                 + lane * kLaneBytes;
        Stage8 f;
        f.v = *(const uint4*)p;
        f.w = *(const uint4*)(p + 16);
        return f;
    }
};

// Compile-time certificate is solely the supported form and the tile geometry;
// the band-count check is at dispatch. Other forms retain the plane route.
template <class F>
struct IqrFused12Certificate {
    static constexpr uint32_t lds = kPrefillLds + 2u * kPBM * sizeof(float)
                                  + kPWaves * kTrBytes + Dec<F>::kTab;
    static constexpr bool geometry = kPWaves == kPWW * kPTW &&
        kTrBytes >= 2u * kTileBytes && kTrBytes % 16u == 0u;
    static constexpr bool rowf = F::SC == kRowF32 &&
        ((F::L == 16u && F::B >= 2u && F::B <= 7u) ||
         (F::L == 15u && F::B == 2u));
    static constexpr bool bf16 = F::SC == kBf16 && F::L == 16u &&
        F::B >= 4u && F::B <= 7u;
    static constexpr bool verified = geometry && F::KIND == kTrellisMad &&
        F::LV == 0u && F::LRB == 0u && (rowf || bf16);
};

template <class F>
__global__ __launch_bounds__(kPWaves * kWave) R4D_WAVES(6)
void iqr_fused12_prefill_kernel(Ctx c, const uint8_t* __restrict__ A,
                              const float* __restrict__ Sa,
                              const uint16_t* __restrict__ Sw,
                              uint16_t* __restrict__ C,
                              uint32_t M, uint32_t K, uint32_t N,
                              uint32_t a_ld, uint32_t sa_ld) {
    using D = Dec<F>;
    __shared__ __attribute__((aligned(16))) int8_t s_tab[D::kTab];
    __shared__ __attribute__((aligned(16))) uint8_t s_tr[kPWaves * kTrBytes];
    c.tab = s_tab;
    D::stage(s_tab, c, threadIdx.x, kPWaves * kWave);
    __syncthreads();
    const IqrFused12Pre<F> pre{c, s_tr};
    if constexpr (F::SC == kRowF32) {
        constexpr Form f{F::KIND, F::B, F::LV, F::L, F::LRB, F::SC, 0};
        const uint8_t* const sb = c.S;
        gemm_8a8_prefill_body<true, 32, IqrFused12Pre<F>, true, kBlock, true>(
            A, Sa, nullptr, Sw, C, M, K, N, a_ld, sa_ld, c.nkb, pre,
            (const float*)sb, (const float*)(sb + cf_off(f, N, K)));
    } else {
        static_assert(F::SC == kBf16 && F::LRB == 0u);
        gemm_8a8_prefill_body<true, 0, IqrFused12Pre<F>, false, kBlock, true>(
            A, Sa, nullptr, Sw, C, M, K, N, a_ld, sa_ld, c.nkb, pre);
    }
}
#endif  // R3_TARGETS

"""

FORM_ANCHOR = """template <class F, int RB>
__global__ __launch_bounds__(kPWaves * kWave) R4D_WAVES(6)
void iqr_prefill_kernel"""

LAUNCH_ANCHOR = """    unsigned int* const ctl = r4d_8a8_split_ctr(stream, g.y->data);"""
FUSED_LAUNCH = """#ifndef R3_TARGETS
    if constexpr (IqrFused12Certificate<F>::verified) {
        const uint32_t bands = (uint32_t)((g.M + kPBM - 1u) / kPBM);
        if (bands == 1u) {
            const Ctx c = ctx_of(g);
            const int rc = R4D_EMIT(stream, (iqr_fused12_prefill_kernel<F>),
                dim3(bands, (uint32_t)(g.N / kPBN), 1), dim3(kPWaves * kWave), 0,
                c, (const uint8_t*)g.a->data, (const float*)g.as->data,
                (const uint16_t*)g.bs->data, (uint16_t*)g.y->data,
                (uint32_t)g.M, (uint32_t)g.K, (uint32_t)g.N,
                (uint32_t)g.a_ld, (uint32_t)g.sa_ld);
            return rc == RAD_OK && hipGetLastError() == hipSuccess
                ? RAD_OK : R4D_NO(RAD_E_DEVICE);
        }
    }
#endif  // R3_TARGETS
"""

PATCHES = (
    FilePatch(
        path="libr4d/r4d_gemm_8a8_prefill.h",
        language="none",
        description="Read fused fragments from per-block storage, retaining native gfx12 WMMA and all folds.",
        edits=(
            replace(
                "rad0065-template",
                "template <bool I8, int RB, class Pre, bool ROWF = false, uint32_t SB = kBlock>",
                "template <bool I8, int RB, class Pre, bool ROWF = false, uint32_t SB = kBlock, bool Fused = false>",
            ),
            replace(
                "rad0065-global-weight",
                "    const uint8_t* const wsrc = W + (size_t)wt0 * ksuper * kTileBytes + lane * kLaneBytes;",
                """    const uint8_t* wsrc = nullptr;
    if constexpr (!Fused) wsrc = W + (size_t)wt0 * ksuper * kTileBytes + lane * kLaneBytes;""",
            ),
            replace(
                "rad0065-first-prefetch",
                """#pragma unroll
    for (uint32_t i = 0; i < kPTW; ++i)
        w0[i] = stage8_load(wsrc + (size_t)i * ksuper * kTileBytes, true);""",
                """    if constexpr (!Fused) {
#pragma unroll
        for (uint32_t i = 0; i < kPTW; ++i)
            w0[i] = stage8_load(wsrc + (size_t)i * ksuper * kTileBytes, true);
    }""",
            ),
            replace(
                "rad0065-stage-native-fragment",
                """#pragma unroll
        for (uint32_t i = 0; i < kPTW; ++i)
            nxt[i] = stage8_load(wsrc + ((size_t)i * ksuper + c1 / kSuperK) * kTileBytes, true);""",
                """        if constexpr (Fused) {
#pragma unroll
            for (uint32_t i = 0; i < kPTW; ++i)
                cur[i] = pre.load(ww * kPTW + i, half, lane);
        } else {
#pragma unroll
            for (uint32_t i = 0; i < kPTW; ++i)
                nxt[i] = stage8_load(wsrc + ((size_t)i * ksuper + c1 / kSuperK) * kTileBytes, true);
        }""",
            ),
            replace(
                "rad0065-block-decode",
                """    for (uint32_t kb = 0; kb < nkb; ++kb) {
        // The weight's scales""",
                """    for (uint32_t kb = 0; kb < nkb; ++kb) {
        if constexpr (Fused) {
            static_assert(I8 && SB == kBlock && kTPB == 2u);
            pre.prepare(n0, kb);
        }
        // The weight's scales""",
            ),
        ),
    ),
    FilePatch(
        path="libr4d/r4d_gemm_iqr.hip",
        language="none",
        description="Fused weight decoding, constexpr form certificate and single-band dispatch.",
        edits=(
            replace(
                "rad0065-native-fused-tile",
                FORM_ANCHOR,
                FUSED_TILE + FORM_ANCHOR,
                guard="struct IqrFused12Pre {",
            ),
            replace(
                "rad0065-one-band-dispatch",
                LAUNCH_ANCHOR,
                FUSED_LAUNCH + LAUNCH_ANCHOR,
                guard="if constexpr (IqrFused12Certificate<F>::verified)",
            ),
        ),
    ),
)
