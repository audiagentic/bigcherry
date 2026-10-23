#pragma once

// R3_MOE_WORKER_COMBINE (RAD18): the combined row's fp16 encoding, shared by the helper's combine and its test.

#include <cstdint>
#include <cstring>

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
