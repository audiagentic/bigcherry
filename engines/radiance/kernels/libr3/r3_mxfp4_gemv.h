// r3_mxfp4_gemv.h - the MXFP4 x fp8 product for ONE activation row on gfx11, as a vector product (RR07).
//
// libr4d's decode kernel (r4d_gemm_mxfp4a8_decode) gives a single token a 16-row WMMA fragment: 15 of every 16
// products the matrix instruction forms are for rows nobody asked for. On gfx12 the instruction is fast enough that
// the kernel still runs near the memory rate. On gfx11 it does not, so for M = 1 libr3 launches this kernel from
// the same entry point, on the same operands, instead (native/r4d_gemm_mxfp4a8_decode.hip.rw). R3_GEMV=0 in the
// environment keeps libr4d's kernel.
//
// Structure after hipfire's gfx1100 decode kernels (gemv_hfp4g32.gfx1100.hip, Apache-2.0, Kaden Schutt): no matrix
// instruction and no shared-memory staging of the weight; the weight is decoded in registers as it streams.
//
// A block is 8 waves over 128 output columns, 16 columns a wave. libr4d stores the weight in fragment order: for a
// tile of 16 columns and a K step of 16, 32 dwords, lane (half << 4 | column) holding 8 codes, k = 8 * half .. + 7.
// So lane (half, column) owns half of its column's K and reads ONE dword a K step, in memory order; the two halves
// of a column are added once at the end with one cross-lane move. Each dword is decoded to 8 bf16 through
// r3_mxfp4_unpack8_bf16 (the table lookup the native WMMA form uses, with the block's exponent difference folded
// in) and multiplied with 8 activations through four v_dot2_f32_bf16, accumulating in f32.
//
// The activation row is widened from E4M3 to bf16 once a block into shared memory (2 bytes x K; K up to
// R3_GEMV_MAX_K = 24576 is 48 KiB). E4M3 values are exact in bf16, and the activation scale is applied to the sum:
// once at the end for a per-row scale, once per 128 of K for the (row, 128) grid, which is the sum libr4d forms
// after its carried-scale bookkeeping. Sums are formed in another order than libr4d's, so results agree to
// rounding, not bit for bit.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdlib>

#define R3_GEMV_MAX_K 24576

static inline bool r3_gemv_on() {
    static const bool on = [] {
        const char* e = std::getenv("R3_GEMV");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return on;
}

typedef __bf16 r3_v2bf __attribute__((ext_vector_type(2)));

template <bool NT, bool ABLK>
__global__ __launch_bounds__(256) void r3_gemm_mxfp4a8_gemv1_kernel(
        const unsigned char* __restrict__ A, const float* __restrict__ As, const unsigned int* __restrict__ W,
        const unsigned char* __restrict__ Ws, const unsigned char* __restrict__ Wref,
        __hip_bfloat16* __restrict__ C, int K, int N) {
    __shared__ unsigned short sA[R3_GEMV_MAX_K];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int col = lane & 15, half = lane >> 4;

    // the activation row, E4M3 to bf16, 16 codes a thread a pass (K is a multiple of 64)
    for (int k0 = tid * 16; k0 < K; k0 += 256 * 16) {
        r3_store16_e4m3_as_bf16(&sA[k0], *reinterpret_cast<const r3_u4u*>(A + k0));
    }
    __syncthreads();

    const int n = blockIdx.x * 128 + wave * 16 + col;
    const int nc = n < N ? n : N - 16 + col;  // N is a multiple of 16: a column past the end reads the last tile
    const int ksteps = K / 16;
    const unsigned int* __restrict__ wp = W + ((size_t) (nc >> 4) * ksteps) * 32 + ((half << 4) | (nc & 15));
    const int ref = (int) Wref[nc];
    const unsigned short* __restrict__ ap = sA + half * 8;

    float acc = 0.f, total = 0.f;
    int d = 0;
    for (int ks = 0; ks < ksteps; ++ks) {
        if ((ks & 1) == 0) {  // one E8M0 exponent per 32 of K
            d = r4d_mxfp4_fold_d(ref, (int) Ws[(size_t) (ks >> 1) * N + nc]);
        }
        const unsigned int codes = NT ? __builtin_nontemporal_load(wp + (size_t) ks * 32) : wp[(size_t) ks * 32];
        const r3_u4 w = r3_mxfp4_unpack8_bf16(codes, d);
        const r3_u4 a = *reinterpret_cast<const r3_u4*>(ap + ks * 16);  // 16-byte aligned: ks * 32 or + 16 bytes
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            acc = __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(r3_v2bf, w[p]), __builtin_bit_cast(r3_v2bf, a[p]),
                                                  acc, false);
        }
        if constexpr (ABLK) {
            if ((ks & 7) == 7) {  // end of a 128 block: its own activation scale
                total += acc * As[ks >> 3];
                acc = 0.f;
            }
        }
    }
    if constexpr (ABLK) {
        if (ksteps & 7) {  // K is a multiple of 64, so the last 128 block may be half one
            total += acc * As[ksteps >> 3];
        }
    } else {
        total = acc * As[0];
    }
    // the other half of this column's K
    total += __uint_as_float(r3_from_lane((unsigned) lane ^ 16u, __float_as_uint(total)));
    if (half == 0 && n < N) {
        C[n] = (__hip_bfloat16) (total * __int_as_float(ref << 23));  // 2^(Wref - 127)
    }
}

// Launch for M = 1. Operands are r4d_gemm_mxfp4a8_decode's; `ablk` is the (row, 128) activation scale grid.
static inline void r3_mxfp4_gemv1_launch(long a, long ascale, long wq, long ws, long wref, long c, int K, int N,
                                         bool nt, bool ablk, long stream) {
    const dim3 grid((N + 127) / 128, 1, 1), block(256);
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
