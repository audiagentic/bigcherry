// libr3 compatibility layer: the gfx12-only instructions libr4d's kernels use, supplied for gfx11 (RDNA3).
//
// Every libr4d unit libr3 builds is compiled through a wrapper that includes this header first. A builtin gfx11 does
// not have is replaced by a device function of the same meaning, so the unit's own text is unchanged. What is here
// is correct first and fast second: a kernel that matters gets a native gfx11 version, developed and compared in
// kdev/ against the version built through this layer.
//
// Covered so far:
//   fp8 (OCP E4M3) pack conversions  v_cvt_pk_fp8_f32 / v_cvt_pk_f32_fp8     -> software, same rounding
// Not covered yet (units that need them are listed in exclude.txt):
//   the gfx12 WMMA builtins, global_load_tr_b128, and the all-reduce units' gfx12 instructions.
#ifndef R3_COMPAT_H
#define R3_COMPAT_H

#include <hip/hip_runtime.h>

#include <cstdint>

typedef float r3_v2f __attribute__((ext_vector_type(2)));  // what v_cvt_pk_f32_fp8 returns

// One E4M3 code to f32: 1-4-3, bias 7, no infinity, S1111111 the only NaN (rad_plugin.h, rad_fp8e4m3_to_f32).
__device__ __forceinline__ float r3_e4m3_to_f32(uint32_t b) {
    const uint32_t s = (b & 0x80u) << 24;
    const uint32_t e = (b >> 3) & 0xfu;
    uint32_t m = b & 0x7u;
    uint32_t u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            // subnormal: m/8 * 2^-6; normalise the 3-bit mantissa
            const uint32_t sh = m >= 4u ? 1u : (m >= 2u ? 2u : 3u);
            m = (m << sh) & 0x7u;
            u = s | ((127u - 6u - sh) << 23) | (m << 20);
        }
    } else if (e == 0xfu && m == 0x7u) {
        u = s | 0x7fc00000u;
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    return __uint_as_float(u);
}

// f32 to one E4M3 code, round to nearest even. A NaN stays the NaN code. The instruction answers an out-of-range
// input with NaN and every caller clamps to +-448 first; here an out-of-range input saturates, which is what the
// reference conversion does (rad_plugin.h, rad_f32_to_fp8e4m3), so a clamped caller sees the same bytes.
__device__ __forceinline__ uint32_t r3_f32_to_e4m3(float f) {
    const uint32_t u = __float_as_uint(f);
    const uint32_t s = (u >> 24) & 0x80u;
    const uint32_t a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return s | 0x7fu;
    if (a >= 0x43e00000u) return s | 0x7eu;  // >= 448
    const int e = static_cast<int>(a >> 23) - 127;
    uint32_t m = a & 0x7fffffu;
    if (e < -10) return s;
    if (e < -6) {  // subnormal code
        const uint32_t sh = static_cast<uint32_t>(-6 - e + 20);
        m |= 0x800000u;
        const uint32_t lo = m & ((1u << sh) - 1u);
        uint32_t q = m >> sh;
        const uint32_t half = 1u << (sh - 1u);
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return s | q;
    }
    uint32_t q = m >> 20;
    const uint32_t lo = m & 0xfffffu;
    if (lo > 0x80000u || (lo == 0x80000u && (q & 1u))) ++q;
    int ee = e + 7;
    if (q == 0x8u) { q = 0; ++ee; }
    if (ee > 15 || (ee == 15 && q >= 7u)) return s | 0x7eu;
    return s | (static_cast<uint32_t>(ee) << 3) | q;
}

// v_cvt_pk_fp8_f32: x and y to two codes, x in the low byte, written into the low word of `old` (hi false) or the
// high word (hi true); the other word of `old` is kept.
__device__ __forceinline__ int r3_cvt_pk_fp8_f32(float x, float y, int old, bool hi) {
    const uint32_t pair = r3_f32_to_e4m3(x) | (r3_f32_to_e4m3(y) << 8);
    const uint32_t keep = static_cast<uint32_t>(old);
    return static_cast<int>(hi ? ((keep & 0x0000ffffu) | (pair << 16)) : ((keep & 0xffff0000u) | pair));
}

// v_cvt_pk_f32_fp8: the two codes of the low word (hi false) or the high word (hi true) to two f32, low byte first.
__device__ __forceinline__ r3_v2f r3_cvt_pk_f32_fp8(int src, bool hi) {
    const uint32_t word = static_cast<uint32_t>(src) >> (hi ? 16 : 0);
    r3_v2f out;
    out.x = r3_e4m3_to_f32(word & 0xffu);
    out.y = r3_e4m3_to_f32((word >> 8) & 0xffu);
    return out;
}

#define __builtin_amdgcn_cvt_pk_fp8_f32 r3_cvt_pk_fp8_f32
#define __builtin_amdgcn_cvt_pk_f32_fp8 r3_cvt_pk_f32_fp8

#endif
