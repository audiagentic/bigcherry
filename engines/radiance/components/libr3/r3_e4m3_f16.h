// Direct OCP FP8 E4M3 to IEEE binary16 conversion for libr3 (gfx1100 has no fp8 conversion instruction).
//
// E4M3 is 1 sign, 4 exponent bits (bias 7), 3 mantissa bits, no infinity; S.1111.111 is the only NaN. Every E4M3
// value is exactly representable in binary16 (5 exponent bits, bias 15, 10 mantissa bits), so no rounding is
// involved and the f32 detour of r3_cvt_pk_f32_fp8 is unnecessary. For a code b, s = b >> 7, e = (b >> 3) & 15,
// m = b & 7:
//   normal (e != 0, not the NaN code): f16 = (s << 15) | (((b & 0x7f) << 7) + 0x2000)
//                                       exponent field e + 8 (= e - 7 + 15), mantissa m << 7
//   e == 0, m == 0                    : signed zero, s << 15
//   e == 0, m in 1..7 (subnormal)     : the value m * 2^-9 is a NORMAL f16: with p = floor(log2 m),
//                                       exponent field p + 6, mantissa (m - 2^p) << (10 - p), that is
//                                       m = 1..7 -> 0x1800 0x1c00 0x1e00 0x2000 0x2100 0x2200 0x2300
//   (b & 0x7f) == 0x7f                : a quiet f16 NaN with the sign, 0x7e00 | (s << 15), which is also what
//                                       r3_e4m3_to_f32's NaN (0x7fc00000 | sign) becomes in f16.
//
// r3_e4m3_to_f16_bits is the readable reference. The x2 forms are branch-free integer code on both codes of a
// dword at once (SWAR, two 16-bit lanes). Count of 32-bit integer operations (shift, and, or, add, xor, mul) of
// r3_e4m3x2_to_f16x2, mul counted as one:
//   spread the two codes into the lanes ........................ 5
//   normal path (sign, magnitude, +0x2000, merge sign) ......... 5   (10 with the spread)
//   NaN fix-up (flag, 0x7e00 fill) ............................. 5
//   e == 0 fix-up (zero flag, e flag and mask, 3-bit lookup in a
//   nibble table, select) ...................................... 23
//   total ...................................................... 38
// The hi form has the same count (the spread uses the high half). The compiler fuses several of these into single
// instructions (and-or, shift-or, add3, mad24), so the instruction count is lower.
//
// Usable from HIP device code and from a plain host C++17 translation unit (the host test checks every input).
#ifndef R3_E4M3_F16_H
#define R3_E4M3_F16_H

#include <cstdint>

#ifndef R3_E4M3_FN
#if defined(__HIPCC__)
#define R3_E4M3_FN __device__ __forceinline__
#else
#define R3_E4M3_FN inline
#endif
#endif

// One E4M3 code (low 8 bits of `code`) to f16 bits.
R3_E4M3_FN uint32_t r3_e4m3_to_f16_bits(uint32_t code) {
    const uint32_t s = (code & 0x80u) << 8;
    const uint32_t mag = code & 0x7fu;
    if (mag == 0x7fu) return s | 0x7e00u;
    if (mag >= 8u) return s | ((mag << 7) + 0x2000u);
    if (mag == 0u) return s;
    const uint32_t p = mag >= 4u ? 2u : (mag >= 2u ? 1u : 0u);  // floor(log2 m)
    return s | (((p + 5u) << 10) + (mag << (10u - p)));
}

// Two codes in lane form, one per 16-bit lane at bits 7..14 of the lane (u has no other bit set), to two f16.
R3_E4M3_FN uint32_t r3_e4m3_lanes_to_f16x2(uint32_t u) {
    const uint32_t mag = u & 0x3f803f80u;                            // (b & 0x7f) << 7
    uint32_t r = mag + 0x20002000u;                                  // the normal path
    // NaN: mag == 0x3f80 is the only lane that reaches bit 14 when 0x80 is added; fill 0x7e00 = 0x5f80 + 0x1e80.
    const uint32_t nan = ((mag + 0x00800080u) >> 14) & 0x00010001u;
    r += nan * 0x1e80u;
    // e == 0: ne is 1 in the lanes with mag >= 0x400 (e != 0), nz in the lanes with mag != 0; each is the carry
    // into bit 15 of mag plus (0x8000 - bound), which cannot reach the next lane (mag <= 0x3f80).
    const uint32_t ne = ((mag + 0x7c007c00u) >> 15) & 0x00010001u;
    const uint32_t nz = ((mag + 0x7f807f80u) >> 15) & 0x00010001u;
    // Subnormal m in 1..7: high byte of the f16 is 0x18 + g(m), g = 0 4 6 8 9 10 11 (a nibble table).
    const uint32_t i0 = (u >> 5) & 0x1cu;                            // 4 * m of lane 0
    const uint32_t i1 = (u >> 21) & 0x1cu;                           // 4 * m of lane 1
    const uint32_t g0 = ((0xba986400u >> i0) << 8) & 0x00000f00u;
    const uint32_t g1 = ((0xba986400u >> i1) << 24) & 0x0f000000u;
    const uint32_t w = (g0 | g1) + nz * 0x1800u;                     // 0 for zero, else the subnormal's f16
    const uint32_t keep = ne * 0xffffu;                              // lane mask of the normal lanes
    r = w ^ ((w ^ r) & keep);
    return r | ((u << 1) & 0x80008000u);                             // sign: bit 14 of the lane to bit 15
}

// The two codes in the low 16 bits of w (byte 0 = element 0) to two f16, element 0 in the low half.
R3_E4M3_FN uint32_t r3_e4m3x2_to_f16x2(uint32_t w) {
    return r3_e4m3_lanes_to_f16x2(((w << 7) & 0x00007f80u) | ((w << 15) & 0x7f800000u));
}

// The two codes in the high 16 bits of w (the `hi` flag of v_cvt_pk_f32_fp8).
R3_E4M3_FN uint32_t r3_e4m3x2hi_to_f16x2(uint32_t w) {
    return r3_e4m3_lanes_to_f16x2(((w >> 9) & 0x00007f80u) | ((w >> 1) & 0x7f800000u));
}

// Eight codes (lo = codes 0..3, hi = codes 4..7, memory order) to four dwords of two f16 each:
// w4[0] = codes 0,1; w4[1] = codes 2,3; w4[2] = codes 4,5; w4[3] = codes 6,7.
R3_E4M3_FN void r3_e4m3x8_to_f16x4w(uint32_t lo, uint32_t hi, uint32_t* w4) {
    w4[0] = r3_e4m3x2_to_f16x2(lo);
    w4[1] = r3_e4m3x2hi_to_f16x2(lo);
    w4[2] = r3_e4m3x2_to_f16x2(hi);
    w4[3] = r3_e4m3x2hi_to_f16x2(hi);
}

#endif  // R3_E4M3_F16_H
