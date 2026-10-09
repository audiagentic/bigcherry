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

// ------------------------------------------------------------------ the workgroup barrier
//
// gfx12 splits the barrier into s_barrier_signal and s_barrier_wait. libr4d issues them as a pair on the workgroup
// barrier (-1), signal then wait, which is gfx11's one s_barrier (r4d_gemm_w4a8_prefill.hip). The signal becomes
// nothing and the wait becomes the barrier; a signal without its wait would be wrong and no unit has one.
#define __builtin_amdgcn_s_barrier_signal(id) ((void) 0)
#define __builtin_amdgcn_s_barrier_wait(id) __builtin_amdgcn_s_barrier()

// ------------------------------------------------------------------ WMMA, 16-bit floats
//
// libr4d builds gfx12 fragments and calls the gfx12 instruction. gfx11 has the instruction with other layouts
// (both probed on gfx1100 for BigCherry patch 1253, gated_delta_net_chunked_bf16_gfx11.cu):
//
//             gfx12                                        gfx11
//   A, B      8 elements a lane: row = lane & 15;          16 elements a lane: row = lane & 15, k = element;
//             lanes 0-15 hold k 0-3 and 8-11,              lanes 16-31 repeat lanes 0-15
//             lanes 16-31 hold k 4-7 and 12-15
//   C, D      column = lane & 15,                          column = lane & 15,
//             row = 8 * (lane >> 4) + element              row = 2 * element + (lane >> 4)
//
// The functions below take gfx12 fragments, exchange the missing half with the partner lane (lane ^ 16) through
// ds_bpermute, run the gfx11 instruction on a zero accumulator, bring the product back to the gfx12 layout and add
// the caller's accumulator in f32. That is 12 cross-lane moves a call: correct, not fast. The sum is rounded once
// more than the instruction rounds it (product, then add), so results agree with gfx12 to rounding, not bit for bit.

typedef unsigned r3_u4 __attribute__((ext_vector_type(4)));
typedef unsigned r3_u8 __attribute__((ext_vector_type(8)));
typedef float    r3_v8f __attribute__((ext_vector_type(8)));
typedef __bf16   r3_v8bf __attribute__((ext_vector_type(8)));
typedef __bf16   r3_v16bf __attribute__((ext_vector_type(16)));
typedef __fp16   r3_v8hf __attribute__((ext_vector_type(8)));
typedef _Float16 r3_v16h __attribute__((ext_vector_type(16)));

// This thread's lane in its wave: threads fill waves in order of their flattened index.
__device__ __forceinline__ unsigned r3_lane() {
    return (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31u;
}

// The value `v` holds in lane `src`.
__device__ __forceinline__ unsigned r3_from_lane(unsigned src, unsigned v) {
    return static_cast<unsigned>(__builtin_amdgcn_ds_bpermute(static_cast<int>(src << 2), static_cast<int>(v)));
}

// A or B: a gfx12 fragment (4 dwords, 8 elements) to the gfx11 fragment (8 dwords, the whole row).
__device__ __forceinline__ r3_u8 r3_frag_12_to_11(r3_u4 own, unsigned lane) {
    const unsigned partner = lane ^ 16u;
    const bool hi = (lane >> 4) != 0;
    r3_u4 other;
    other.x = r3_from_lane(partner, own.x);
    other.y = r3_from_lane(partner, own.y);
    other.z = r3_from_lane(partner, own.z);
    other.w = r3_from_lane(partner, own.w);
    const r3_u4 k_0_3_8_11 = hi ? other : own;
    const r3_u4 k_4_7_12_15 = hi ? own : other;
    r3_u8 row;
    row[0] = k_0_3_8_11.x;  row[1] = k_0_3_8_11.y;   // k 0-3
    row[2] = k_4_7_12_15.x; row[3] = k_4_7_12_15.y;  // k 4-7
    row[4] = k_0_3_8_11.z;  row[5] = k_0_3_8_11.w;   // k 8-11
    row[6] = k_4_7_12_15.z; row[7] = k_4_7_12_15.w;  // k 12-15
    return row;
}

// C or D: a gfx11 accumulator to the gfx12 layout.
__device__ __forceinline__ r3_v8f r3_acc_11_to_12(r3_v8f p, unsigned lane) {
    const unsigned partner = lane ^ 16u;
    const bool hi = (lane >> 4) != 0;
    // A lower lane needs the partner's rows 1,3,5,7 (its elements 0-3); an upper lane the partner's rows
    // 8,10,12,14 (its elements 4-7). So an upper lane sends elements 0-3 and a lower lane elements 4-7.
    float got[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float send = hi ? p[i] : p[4 + i];
        got[i] = __uint_as_float(r3_from_lane(partner, __float_as_uint(send)));
    }
    r3_v8f out;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int h = e >> 1;
        out[e] = hi ? ((e & 1) ? p[4 + h] : got[h])   // rows 8-15: even rows from the partner, odd rows own
                    : ((e & 1) ? got[h] : p[h]);      // rows 0-7:  even rows own, odd rows from the partner
    }
    return out;
}

// Templates on the operand types: libr4d passes the fragments as whichever 128-bit vector its unit built them in
// (bf16, fp16, shorts, dwords) and the accumulator as its own 8 x f32 typedef.
template <class A, class B, class C>
__device__ __forceinline__ r3_v8f r3_wmma_f32_16x16x16_bf16_w32_gfx12(A a, B b, C acc) {
    static_assert(sizeof(A) == 16 && sizeof(B) == 16 && sizeof(C) == 32, "gfx12 WMMA operands: 2 x 128 bits, 8 x f32");
    const r3_v8f c = __builtin_bit_cast(r3_v8f, acc);
    const unsigned lane = r3_lane();
    const r3_v16bf a11 = __builtin_bit_cast(r3_v16bf, r3_frag_12_to_11(__builtin_bit_cast(r3_u4, a), lane));
    const r3_v16bf b11 = __builtin_bit_cast(r3_v16bf, r3_frag_12_to_11(__builtin_bit_cast(r3_u4, b), lane));
    const r3_v8f zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    return c + r3_acc_11_to_12(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a11, b11, zero), lane);
}

template <class A, class B, class C>
__device__ __forceinline__ r3_v8f r3_wmma_f32_16x16x16_f16_w32_gfx12(A a, B b, C acc) {
    static_assert(sizeof(A) == 16 && sizeof(B) == 16 && sizeof(C) == 32, "gfx12 WMMA operands: 2 x 128 bits, 8 x f32");
    const r3_v8f c = __builtin_bit_cast(r3_v8f, acc);
    const unsigned lane = r3_lane();
    const r3_v16h a11 = __builtin_bit_cast(r3_v16h, r3_frag_12_to_11(__builtin_bit_cast(r3_u4, a), lane));
    const r3_v16h b11 = __builtin_bit_cast(r3_v16h, r3_frag_12_to_11(__builtin_bit_cast(r3_u4, b), lane));
    const r3_v8f zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    return c + r3_acc_11_to_12(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a11, b11, zero), lane);
}

// ------------------------------------------------------------------ WMMA, 8-bit and 4-bit integers, and fp8
//
// The integer forms have the same shape one size down: gfx12 passes 8 bytes a lane (2 dwords), gfx11 the whole
// row of 16 bytes (4 dwords). The split of K between a lane and its partner is ASSUMED to follow the 16-bit
// fragments (lanes 0-15 hold k 0-3 and 8-11, lanes 16-31 k 4-7 and 12-15); no source states it for bytes, and
// rad-kbench decides: the integer sums are exact, so a wrong mapping fails every case rather than hiding.
// The int32 accumulator has the f32 one's layout, and adding the caller's accumulator afterwards is exact.
typedef int r3_i2 __attribute__((ext_vector_type(2)));
typedef int r3_i4 __attribute__((ext_vector_type(4)));
typedef int r3_i8 __attribute__((ext_vector_type(8)));

// A or B, bytes: a gfx12 fragment (2 dwords) to the gfx11 row (4 dwords).
__device__ __forceinline__ r3_i4 r3_bytes_12_to_11(r3_i2 own, unsigned lane) {
    const unsigned partner = lane ^ 16u;
    const bool hi = (lane >> 4) != 0;
    const int other0 = static_cast<int>(r3_from_lane(partner, static_cast<unsigned>(own.x)));
    const int other1 = static_cast<int>(r3_from_lane(partner, static_cast<unsigned>(own.y)));
    r3_i4 row;
    row.x = hi ? other0 : own.x;   // k 0-3
    row.y = hi ? own.x : other0;   // k 4-7
    row.z = hi ? other1 : own.y;   // k 8-11
    row.w = hi ? own.y : other1;   // k 12-15
    return row;
}

__device__ __forceinline__ r3_i8 r3_acc_i32_11_to_12(r3_i8 p, unsigned lane) {
    return __builtin_bit_cast(r3_i8, r3_acc_11_to_12(__builtin_bit_cast(r3_v8f, p), lane));
}

// The sign flags are instruction immediates, so they are template arguments; every libr4d call site passes
// constants. `clamp` is false at every call site and is not implemented.
template <bool SA, bool SB, class A, class B, class C>
__device__ __forceinline__ r3_i8 r3_wmma_i32_16x16x16_iu8_w32_gfx12(A a, B b, C acc) {
    static_assert(sizeof(A) == 8 && sizeof(B) == 8 && sizeof(C) == 32, "gfx12 iu8 WMMA operands: 2 x 64 bits, 8 x i32");
    const unsigned lane = r3_lane();
    const r3_i4 a11 = r3_bytes_12_to_11(__builtin_bit_cast(r3_i2, a), lane);
    const r3_i4 b11 = r3_bytes_12_to_11(__builtin_bit_cast(r3_i2, b), lane);
    const r3_i8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    const r3_i8 product = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(SA, a11, SB, b11, zero, false);
    return __builtin_bit_cast(r3_i8, acc) + r3_acc_i32_11_to_12(product, lane);
}

// gfx12's 4-bit form covers K = 32 in one instruction (16 nibbles a lane); gfx11's covers K = 16 (the whole row of
// 16 nibbles in 2 dwords). ASSUMED split, by the same pattern: dword 0 is k 0-7 in lanes 0-15 and k 8-15 in lanes
// 16-31, dword 1 is k 16-23 and k 24-31. Two gfx11 instructions, one for each half of K.
template <bool SA, bool SB, class A, class B, class C>
__device__ __forceinline__ r3_i8 r3_wmma_i32_16x16x32_iu4_w32_gfx12(A a, B b, C acc) {
    static_assert(sizeof(A) == 8 && sizeof(B) == 8 && sizeof(C) == 32, "gfx12 iu4 WMMA operands: 2 x 64 bits, 8 x i32");
    const unsigned lane = r3_lane();
    const r3_i4 a11 = r3_bytes_12_to_11(__builtin_bit_cast(r3_i2, a), lane);   // x,y = k 0-15; z,w = k 16-31
    const r3_i4 b11 = r3_bytes_12_to_11(__builtin_bit_cast(r3_i2, b), lane);
    const r3_i8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    const r3_i2 a_lo = {a11.x, a11.y}, a_hi = {a11.z, a11.w};
    const r3_i2 b_lo = {b11.x, b11.y}, b_hi = {b11.z, b11.w};
    r3_i8 product = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(SA, a_lo, SB, b_lo, zero, false);
    product = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(SA, a_hi, SB, b_hi, product, false);
    return __builtin_bit_cast(r3_i8, acc) + r3_acc_i32_11_to_12(product, lane);
}

// fp8 x fp8: gfx11 has no fp8 WMMA. Every E4M3 value is exactly a bf16 value (3 mantissa bits, exponents inside
// bf16's range), so each code is widened to bf16 and the bf16 form does the multiply: the same products, summed in
// f32. 8 codes a lane (2 dwords) become the gfx12 bf16 fragment's 8 elements, in the same order.
__device__ __forceinline__ r3_u4 r3_e4m3x8_to_bf16(r3_i2 codes) {
    r3_u4 out;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
        const unsigned word = static_cast<unsigned>(d < 2 ? codes.x : codes.y) >> ((d & 1) * 16);
        const unsigned lo = __float_as_uint(r3_e4m3_to_f32(word & 0xffu)) >> 16;
        const unsigned hi = __float_as_uint(r3_e4m3_to_f32((word >> 8) & 0xffu)) >> 16;
        out[d] = lo | (hi << 16);
    }
    return out;
}

template <class A, class B, class C>
__device__ __forceinline__ r3_v8f r3_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(A a, B b, C acc);

#define __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(sa, a, sb, b, c, clamp) \
    r3_wmma_i32_16x16x16_iu8_w32_gfx12<(sa), (sb)>((a), (b), (c))
#define __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(sa, a, sb, b, c, clamp) \
    r3_wmma_i32_16x16x32_iu4_w32_gfx12<(sa), (sb)>((a), (b), (c))
#define __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12 r3_wmma_f32_16x16x16_fp8_fp8_w32_gfx12

template <class A, class B, class C>
__device__ __forceinline__ r3_v8f r3_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(A a, B b, C acc) {
    static_assert(sizeof(A) == 8 && sizeof(B) == 8 && sizeof(C) == 32, "gfx12 fp8 WMMA operands: 2 x 64 bits, 8 x f32");
    return r3_wmma_f32_16x16x16_bf16_w32_gfx12(r3_e4m3x8_to_bf16(__builtin_bit_cast(r3_i2, a)),
                                               r3_e4m3x8_to_bf16(__builtin_bit_cast(r3_i2, b)), acc);
}

#define __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12 r3_wmma_f32_16x16x16_bf16_w32_gfx12
#define __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12 r3_wmma_f32_16x16x16_f16_w32_gfx12

// ------------------------------------------------------------------ native gfx11 forms
//
// Used by the units that have block rules in native/ (r3_rewrite.py): there the kernel's own inner loop is
// rewritten to build gfx11 fragments directly, so no fragment or accumulator crosses lanes inside the K loop. A
// gfx11 lane holds its row's 16 values, so every lane reads 16 values where the gfx12 kernel reads 8, and the
// accumulator stays in the gfx11 layout until the kernel's epilogue, where r3_acc_native_to_gfx12 moves it once.

typedef unsigned r3_u4u __attribute__((ext_vector_type(4), aligned(1)));

// One E4M3 code as the bf16 with the same value (exact: 3 mantissa bits, exponents inside bf16's range). Normal
// codes shift into place; the eight subnormal magnitudes m * 2^-9 come from a packed table. 0x7f / 0xff (NaN) are
// not special-cased: weights and activations never hold them.
__device__ __forceinline__ unsigned r3_e4m3_to_bf16_bits(unsigned code) {
    const unsigned mag = code & 0x7fu;
    const unsigned normal = (mag << 4) + (120u << 7);
    const unsigned long long sub = (mag & 4u) ? 0x3c603c403c203c00ull : 0x3bc03b803b000000ull;
    const unsigned subnormal = static_cast<unsigned>(sub >> ((mag & 3u) * 16)) & 0xffffu;
    return ((code & 0x80u) << 8) | (mag < 8u ? subnormal : normal);
}

// 16 E4M3 codes at p (one fragment row, K ascending) as a gfx11 bf16 fragment.
__device__ __forceinline__ r3_v16bf r3_row16_e4m3_to_bf16(const unsigned char* p) {
    const r3_u4u codes = *reinterpret_cast<const r3_u4u*>(p);
    r3_u8 out;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
        const unsigned w = codes[d];
        out[2 * d] = r3_e4m3_to_bf16_bits(w & 0xffu) | (r3_e4m3_to_bf16_bits((w >> 8) & 0xffu) << 16);
        out[2 * d + 1] = r3_e4m3_to_bf16_bits((w >> 16) & 0xffu) | (r3_e4m3_to_bf16_bits(w >> 24) << 16);
    }
    return __builtin_bit_cast(r3_v16bf, out);
}

// D = A x B + C on gfx11 fragments, accumulator in the gfx11 layout.
__device__ __forceinline__ r3_v8f r3_wmma_bf16_native(r3_v16bf a, r3_v16bf b, r3_v8f acc) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
}

// A gfx11-layout accumulator to the gfx12 layout the kernel's epilogue reads.
template <class C>
__device__ __forceinline__ r3_v8f r3_acc_native_to_gfx12(C acc) {
    static_assert(sizeof(C) == 32, "accumulator: 8 x f32");
    return r3_acc_11_to_12(__builtin_bit_cast(r3_v8f, acc), r3_lane());
}

// ------------------------------------------------------------------ the gfx12 transposed load
//
// global_load_tr_b128: every lane names 128 bits (8 x 16-bit) at its own address, and the instruction returns
// them transposed within each group of 8 lanes -- lane j of a group receives element j of each of the group's 8
// lanes, in lane order (libr4d r4d_common.h, load_tr_b128). gfx11 has no such load. Here each lane fetches the 8
// addresses of its group through ds_bpermute and reads its element from each: 16 cross-lane moves and 8 loads.
typedef short r3_v8s __attribute__((ext_vector_type(8)));

__device__ __forceinline__ r3_v8s r3_global_load_tr_b128_v8i16(uint64_t address) {
    const unsigned lane = r3_lane();
    const unsigned group = lane & ~7u, j = lane & 7u;
    const unsigned lo = static_cast<unsigned>(address), hi = static_cast<unsigned>(address >> 32);
    r3_v8s out;
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const uint64_t src = (static_cast<uint64_t>(r3_from_lane(group + i, hi)) << 32) | r3_from_lane(group + i, lo);
        out[i] = reinterpret_cast<const short*>(src)[j];
    }
    return out;
}

// The callers pass a global (address_space(1)) pointer, which only a C-style cast turns into an integer.
#define __builtin_amdgcn_global_load_tr_b128_v8i16(p) r3_global_load_tr_b128_v8i16((uint64_t)(p))

#endif
