#pragma once

// R3_MOE_WORKER_COMBINE (RAD18): the combined row's fp16 encoding, shared by the helper's combine and its test.

#include <cstdint>
#include <cstring>
#include <cmath>
#include <immintrin.h>

/* f32 to IEEE binary16, round to nearest even; *overflow set when the value is finite but beyond fp16 range. A NaN
 * becomes the canonical 0x7E00, never kCombineFallback. */
inline uint16_t combine_f32_f16_rn(float f, bool* overflow) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint16_t sign = (uint16_t)((u >> 16) & 0x8000u);
    const uint32_t a = u & 0x7fffffffu;
    if (a >= 0x7f800000u) return (uint16_t)(sign | (a > 0x7f800000u ? 0x7e00u : 0x7c00u));
    if (a >= 0x477ff000u) { *overflow = true; return (uint16_t)(sign | 0x7c00u); }   /* rounds past 65504 */
    if (a < 0x38800000u) {                                     /* fp16 subnormal or zero */
        if (a < 0x33000000u) return sign;                      /* below half the smallest subnormal */
        const uint32_t m = (a & 0x7fffffu) | 0x800000u;
        const uint32_t shift = 126u - (a >> 23);               /* 14..24 */
        const uint32_t r = m >> shift, rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1u);
        return (uint16_t)(sign | (r + (rem > half || (rem == half && (r & 1u)))));
    }
    uint32_t v = a - 0x38000000u;                              /* rebias exponent 127 -> 15 */
    v += 0xfffu + ((v >> 13) & 1u);
    return (uint16_t)(sign | (v >> 13));
}

inline float combine_bf16_to_f32(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; std::memcpy(&f, &u, 4); return f; }

/* The CPU the combine needs: F16C converts 8 floats to fp16 in one instruction with the same rounding as
 * combine_f32_f16_rn. Checked once when the combine is enabled. */
inline bool combine_cpu_supported() { return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c"); }

/* One token's combined row: the fp32 sum of the present helper rows (mask bit k: src[k]), written as fp16. Helper 0's row
 * is the starting value and the others are added in helper order, so two helpers give exactly what the two-row form did.
 * True when some value is infinite or at least 65520 in magnitude (it would not fit fp16): the caller then marks the row
 * for the engine's per-helper path. A NaN converts to a quiet NaN whose low mantissa bits are zero (bf16 inputs carry
 * none), so it never equals kCombineFallback. */
__attribute__((target("avx2,f16c"))) inline bool combine_row_f16(const uint16_t* const* src, uint32_t nsrc, uint8_t mask,
                                                                  uint16_t* out, uint32_t n) {
    const __m256 limit = _mm256_set1_ps(65520.0f);
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 over = _mm256_setzero_ps();
    uint32_t c = 0;
    for (; c + 8 <= n; c += 8) {
        __m256 v = _mm256_setzero_ps();
        if (mask & 1u)
            v = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src[0] + c))), 16));
        for (uint32_t k = 1; k < nsrc; ++k)
            if (mask & (1u << k))
                v = _mm256_add_ps(v, _mm256_castsi256_ps(_mm256_slli_epi32(
                    _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src[k] + c))), 16)));
        over = _mm256_or_ps(over, _mm256_cmp_ps(_mm256_and_ps(v, abs_mask), limit, _CMP_GE_OQ));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + c), _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
    }
    bool overflow = _mm256_movemask_ps(over) != 0;
    for (; c < n; ++c) {
        float v = (mask & 1u) ? combine_bf16_to_f32(src[0][c]) : 0.0f;
        for (uint32_t k = 1; k < nsrc; ++k)
            if (mask & (1u << k)) v += combine_bf16_to_f32(src[k][c]);
        if (std::fabs(v) >= 65520.0f) overflow = true;
        bool unused = false;
        out[c] = combine_f32_f16_rn(v, &unused);
    }
    return overflow;
}
