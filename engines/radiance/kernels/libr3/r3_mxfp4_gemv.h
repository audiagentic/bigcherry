// r3_mxfp4_gemv.h - the MXFP4 x fp8 product for ONE activation row on gfx11, as a vector product (RR07).
//
// libr4d's decode kernel (r4d_gemm_mxfp4a8_decode) gives a single token a 16-row WMMA fragment: 15 of every 16
// products the matrix instruction forms are for rows nobody asked for. On gfx12 the instruction is fast enough that
// the kernel still runs near the memory rate. On gfx11 it does not, so for M = 1 libr3 can launch this kernel from
// the same entry point, on the same operands, instead (native/r4d_gemm_mxfp4a8_decode.hip.rw). It is opt-in,
// R3_GEMV=1 in the environment, until it passes the selftest's M = 1 cases and beats the WMMA form.
//
// Structure after hipfire's gfx1100 decode kernels (gemv_hfp4g32.gfx1100.hip, Apache-2.0, Kaden Schutt): no matrix
// instruction and no shared-memory staging of the weight; the weight is decoded in registers as it streams.
//
// History. The first form (commit b7b15039) was wrong and slow (r3-serve8, r3-serve10). Slow, measured: it gave one
// lane a whole column's K, so a wave ran K/16 iterations in sequence and the time followed K, not the bytes (465 us
// at N=5120 K=17408 against 113 us for the WMMA form; 138 us for a 2.6 MB weight). Wrong, cause NOT established: it
// multiplied with v_dot2_f32_bf16, and the suspicion is that this instruction is not gfx11's (v_dot2_f32_f16 is),
// although the compiler accepted it for gfx1100. This form uses f16 and splits K over the block's 8 waves; whether
// that also fixes the output is for the selftest's M = 1 cases to say.
//
// A BLOCK is one tile of 16 output columns; its 8 waves each take an eighth of K (whole 128-of-K groups, so an
// activation scale group never straddles two waves). libr4d stores the weight in fragment order: for a tile and a
// K step of 16, 32 dwords, lane (half << 4 | column) holding 8 codes, k = 8 * half .. + 7. So lane (half, column)
// reads ONE dword a K step, in memory order. Each dword is decoded to 8 f16 by table lookup (libr4d's magnitude
// table widened to f16 at compile time, the block's exponent difference folded in) and multiplied with 8
// activations through four v_dot2_f32_f16, accumulating in f32. The 16 partial sums of a column (8 waves x 2
// halves) meet in shared memory and wave 0 writes the result.
//
// The activation row is widened from E4M3 to f16 once a block into shared memory (2 bytes x K; K up to
// R3_GEMV_MAX_K = 24576 is 48 KiB). E4M3 values are exact in f16, and the activation scale is applied to the sum:
// once at the end for a per-row scale, once per 128 of K for the (row, 128) grid, which is the sum libr4d forms
// after its carried-scale bookkeeping. Sums are formed in another order than libr4d's, so results agree to
// rounding, not bit for bit.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdlib>

#define R3_GEMV_MAX_K 24576
#define R3_GEMV_WAVES 8

static inline bool r3_gemv_on() {
    static const bool on = [] {
        const char* e = std::getenv("R3_GEMV");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return on;
}

// One E4M3 code as the f16 with the same value (exact). Normal codes shift into place; the eight subnormal
// magnitudes m * 2^-9 are f16 normals.
constexpr unsigned r3c_e4m3_to_f16(unsigned code) {
    const unsigned mag = code & 0x7fu;
    const unsigned sub[8] = {0x0000u, 0x1800u, 0x1c00u, 0x1e00u, 0x2000u, 0x2100u, 0x2200u, 0x2300u};
    return mag < 8u ? sub[mag] : (mag << 7) + (8u << 10);
}
constexpr unsigned r3c_f16_bytes(unsigned e4m3x4, unsigned shift) {
    return ((r3c_e4m3_to_f16(e4m3x4 & 0xffu) >> shift) & 0xffu)
         | (((r3c_e4m3_to_f16((e4m3x4 >> 8) & 0xffu) >> shift) & 0xffu) << 8)
         | (((r3c_e4m3_to_f16((e4m3x4 >> 16) & 0xffu) >> shift) & 0xffu) << 16)
         | (((r3c_e4m3_to_f16(e4m3x4 >> 24) >> shift) & 0xffu) << 24);
}
#define R3_MX_LO(a, b) {r3c_f16_bytes(a, 0), r3c_f16_bytes(b, 0)},
#define R3_MX_HI(a, b) {r3c_f16_bytes(a, 8), r3c_f16_bytes(b, 8)},
static __device__ __constant__ unsigned int kR3MxF16Lo[16][2] = { R3_MXFP4_MAG(R3_MX_LO) };
static __device__ __constant__ unsigned int kR3MxF16Hi[16][2] = { R3_MXFP4_MAG(R3_MX_HI) };
#undef R3_MX_LO
#undef R3_MX_HI

__device__ __forceinline__ unsigned r3_e4m3_to_f16_bits(unsigned code) {
    const unsigned mag = code & 0x7fu;
    // subnormal magnitudes 1..7 as f16: 0x1800 0x1c00 0x1e00 0x2000 0x2100 0x2200 0x2300, 16 bits each
    const unsigned long long sub = (mag & 4u) ? 0x2300220021002000ull : 0x1e001c0018000000ull;
    const unsigned subnormal = static_cast<unsigned>(sub >> ((mag & 3u) * 16)) & 0xffffu;
    return ((code & 0x80u) << 8) | (mag < 8u ? subnormal : (mag << 7) + (8u << 10));
}

// 16 E4M3 codes (four dwords) as 16 f16 at dst.
template <class V>
__device__ __forceinline__ void r3_store16_e4m3_as_f16(unsigned short* dst, V codes) {
    static_assert(sizeof(V) == 16, "16 E4M3 codes");
    typedef unsigned r3_u8h __attribute__((ext_vector_type(8), aligned(2)));
    const r3_u4 c = __builtin_bit_cast(r3_u4, codes);
    r3_u8 out;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned w = c[j];
        out[2 * j] = r3_e4m3_to_f16_bits(w & 0xffu) | (r3_e4m3_to_f16_bits((w >> 8) & 0xffu) << 16);
        out[2 * j + 1] = r3_e4m3_to_f16_bits((w >> 16) & 0xffu) | (r3_e4m3_to_f16_bits(w >> 24) << 16);
    }
    *reinterpret_cast<r3_u8h*>(dst) = out;
}

// Eight E2M1 codes with exponent difference d as eight f16 in K order, two a dword (r3_mxfp4_unpack8_bf16's lookup
// with the f16 tables).
__device__ __forceinline__ r3_u4 r3_mxfp4_unpack8_f16(unsigned wv, int d) {
    const unsigned ev = wv & 0x0f0f0f0fu, od = (wv >> 4) & 0x0f0f0f0fu;
    const unsigned ei = ev & 0x07070707u, oi = od & 0x07070707u;
    const unsigned e_lo = __builtin_amdgcn_perm(kR3MxF16Lo[d][1], kR3MxF16Lo[d][0], ei);
    const unsigned e_hi = __builtin_amdgcn_perm(kR3MxF16Hi[d][1], kR3MxF16Hi[d][0], ei) | ((ev & 0x08080808u) << 4);
    const unsigned o_lo = __builtin_amdgcn_perm(kR3MxF16Lo[d][1], kR3MxF16Lo[d][0], oi);
    const unsigned o_hi = __builtin_amdgcn_perm(kR3MxF16Hi[d][1], kR3MxF16Hi[d][0], oi) | ((od & 0x08080808u) << 4);
    const unsigned e01 = __builtin_amdgcn_perm(e_hi, e_lo, 0x05010400u), e23 = __builtin_amdgcn_perm(e_hi, e_lo, 0x07030602u);
    const unsigned o01 = __builtin_amdgcn_perm(o_hi, o_lo, 0x05010400u), o23 = __builtin_amdgcn_perm(o_hi, o_lo, 0x07030602u);
    r3_u4 out;
    out[0] = __builtin_amdgcn_perm(o01, e01, 0x05040100u);  // k 0, 1
    out[1] = __builtin_amdgcn_perm(o01, e01, 0x07060302u);  // k 2, 3
    out[2] = __builtin_amdgcn_perm(o23, e23, 0x05040100u);  // k 4, 5
    out[3] = __builtin_amdgcn_perm(o23, e23, 0x07060302u);  // k 6, 7
    return out;
}

typedef _Float16 r3_v2h __attribute__((ext_vector_type(2)));

template <bool NT, bool ABLK>
__global__ __launch_bounds__(R3_GEMV_WAVES * 32) void r3_gemm_mxfp4a8_gemv1_kernel(
        const unsigned char* __restrict__ A, const float* __restrict__ As, const unsigned int* __restrict__ W,
        const unsigned char* __restrict__ Ws, const unsigned char* __restrict__ Wref,
        __hip_bfloat16* __restrict__ C, int K, int N) {
    __shared__ unsigned short sA[R3_GEMV_MAX_K];
    __shared__ float sP[R3_GEMV_WAVES * 32];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int col = lane & 15, half = lane >> 4;

    // the activation row, E4M3 to f16, 16 codes a thread a pass (K is a multiple of 64)
    for (int k0 = tid * 16; k0 < K; k0 += R3_GEMV_WAVES * 32 * 16) {
        r3_store16_e4m3_as_f16(&sA[k0], *reinterpret_cast<const r3_u4u*>(A + k0));
    }
    __syncthreads();

    const int n = blockIdx.x * 16 + col;  // N is a multiple of 16, so every column of the tile exists
    const int ksteps = K / 16;
    // this wave's share of K, in whole 128-of-K groups (8 K steps)
    const int groups = (ksteps + 7) / 8, share = (groups + R3_GEMV_WAVES - 1) / R3_GEMV_WAVES;
    const int ks_lo = wave * share * 8;
    const int ks_hi = (wave + 1) * share * 8 < ksteps ? (wave + 1) * share * 8 : ksteps;
    const unsigned int* __restrict__ wp = W + ((size_t) blockIdx.x * ksteps) * 32 + ((half << 4) | col);
    const int ref = (int) Wref[n];
    const unsigned short* __restrict__ ap = sA + half * 8;

    float acc = 0.f, total = 0.f;
    int d = 0;
    for (int ks = ks_lo; ks < ks_hi; ++ks) {
        if ((ks & 1) == 0) {  // one E8M0 exponent per 32 of K
            d = r4d_mxfp4_fold_d(ref, (int) Ws[(size_t) (ks >> 1) * N + n]);
        }
        const unsigned int codes = NT ? __builtin_nontemporal_load(wp + (size_t) ks * 32) : wp[(size_t) ks * 32];
        const r3_u4 w = r3_mxfp4_unpack8_f16(codes, d);
        const r3_u4 a = *reinterpret_cast<const r3_u4*>(ap + ks * 16);  // byte offset 16 * half + 32 * ks
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const r3_v2h wv = __builtin_bit_cast(r3_v2h, w[p]), av = __builtin_bit_cast(r3_v2h, a[p]);
#if defined(R3_GEMV_DOT2)
            acc = __builtin_amdgcn_fdot2(wv, av, acc, false);
#else
            // Plain f32 multiply-adds. Both dot-product forms (v_dot2_f32_bf16 in the first form, v_dot2_f32_f16 in
            // the second) failed the selftest's M = 1 cases with the SAME error (rel_l2 1.519 at N=272 K=640), so
            // this isolates the instruction from the indexing: if this passes, the dot product was at fault.
            acc += (float) wv[0] * (float) av[0] + (float) wv[1] * (float) av[1];
#endif
        }
        if constexpr (ABLK) {
            if ((ks & 7) == 7 || ks == ks_hi - 1) {  // end of a 128 group (or of K): its own activation scale
                total += acc * As[ks >> 3];
                acc = 0.f;
            }
        }
    }
    if constexpr (!ABLK) {
        total = acc * As[0];
    }
    sP[tid] = total;
    __syncthreads();
    if (wave == 0 && half == 0) {
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < R3_GEMV_WAVES; ++j) {
            sum += sP[j * 32 + col] + sP[j * 32 + 16 + col];
        }
        C[n] = (__hip_bfloat16) (sum * __int_as_float(ref << 23));  // 2^(Wref - 127)
    }
}

// Launch for M = 1. Operands are r4d_gemm_mxfp4a8_decode's; `ablk` is the (row, 128) activation scale grid.
static inline void r3_mxfp4_gemv1_launch(long a, long ascale, long wq, long ws, long wref, long c, int K, int N,
                                         bool nt, bool ablk, long stream) {
    const dim3 grid(N / 16, 1, 1), block(R3_GEMV_WAVES * 32);
    hipStream_t st = (hipStream_t) stream;
    auto Ap = (const unsigned char*) a;
    auto Sp = (const float*) ascale;
    auto Wp = (const unsigned int*) wq;
    auto Zp = (const unsigned char*) ws;
    auto Rp = (const unsigned char*) wref;
    auto Cp = (__hip_bfloat16*) c;
    if (nt) {
        if (ablk) hipLaunchKernelGGL((r3_gemm_mxfp4a8_gemv1_kernel<true, true>), grid, block, 0, st, Ap, Sp, Wp, Zp, Rp, Cp, K, N);
        else      hipLaunchKernelGGL((r3_gemm_mxfp4a8_gemv1_kernel<true, false>), grid, block, 0, st, Ap, Sp, Wp, Zp, Rp, Cp, K, N);
    } else {
        if (ablk) hipLaunchKernelGGL((r3_gemm_mxfp4a8_gemv1_kernel<false, true>), grid, block, 0, st, Ap, Sp, Wp, Zp, Rp, Cp, K, N);
        else      hipLaunchKernelGGL((r3_gemm_mxfp4a8_gemv1_kernel<false, false>), grid, block, 0, st, Ap, Sp, Wp, Zp, Rp, Cp, K, N);
    }
}
