// r3_mxfp4_dec11.h - the MXFP4 x fp8 decode GEMM for M <= 16 with gfx11's own fragments, the weight never staged.
//
// WHY. libr4d's decode kernel stages a slab of the weight in shared memory (unpack, store, barrier, load) because
// on gfx12 a lane holds half a fragment row and eight waves share the staged slab. libr3's native form of it
// (native/r4d_gemm_mxfp4a8_decode.hip.rw) keeps that structure and reaches 28.4 tok/s on one RX 7900 XTX with the
// MXFP4 Qwen3.8-27B, about 430 GB/s of weight on a 960 GB/s card; the split, slab and load hint make no difference
// to it. On gfx11 a lane holds a WHOLE fragment row, so the staging is not needed at all:
//
//   A operand (weight): lane l holds row n0 + (l & 15), 16 values of K. libr4d stores the weight in fragment order,
//     32 dwords a (16-column tile, 16-of-K step), dword (half << 4 | row) holding the 8 codes k = 8 * half .. + 7.
//     A lane reads its row's two dwords (8 bytes) and widens them to 16 bf16 in registers by table lookup, with the
//     block's exponent difference folded in. No shared memory, no barrier, nothing crosses lanes.
//   B operand (activation): lane l holds token min(l & 15, M - 1), the same 16 values of K, as bf16. The activation
//     is small (M x K bytes); each wave widens the 128 of K it is about to use from E4M3 into its own strip of
//     shared memory, written and read by the same wave, so there is no block barrier either.
//   D: lane (column j, half h) element e is output row 2e + h of column j, that is C[token j][n0 + 2e + h].
//
// A BLOCK is 8 waves, each owning one tile of 16 output columns over ALL of K, and writing its own result: no
// barrier in or after the K loop. The first two forms gave a block one tile and split K over its waves, with the
// partial sums combined after a barrier. Their ablation (README, 2026-10-10) showed the time followed the NUMBER OF
// BLOCKS, about 50 ns each whatever was in them (N=1024 18 us, N=34816 124 us with every load, unpack and multiply
// switched off): workgroup launch and barrier cost, 2176 blocks for one gate/up GEMM. This form launches an eighth
// as many blocks, and a wave's K loop keeps its loads one group ahead so its length is not a memory wait a step.
//
// SCALES. The activation scale is one f32 a row, applied at the end, or one a (row, 128 of K), applied to each
// group's accumulator as it closes: the same sum libr4d forms through its carried-scale bookkeeping. The weight's
// 2^(Wref - 127) is applied at the end, per output row. Sums are formed in another order than libr4d's, so results
// agree to rounding, not bit for bit.
//
// Opt-in with R3_DEC11=1 in the environment until it passes the selftest's decode cases; then the default.
// Structure after hipfire's gfx1100 kernels (gemm_mq4g256v2_residual_wmma_gfx11_bt.hip, Apache-2.0, Kaden Schutt):
// the lane owns a weight row and dequantises it in registers.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdlib>

#define R3_D11_WAVES 8

static inline bool r3_dec11_on() {
    static const bool on = [] {
        const char* e = std::getenv("R3_DEC11");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return on;
}

// r3_mxfp4_unpack8_bf16 with the table rows already in registers (they change once per 32 of K).
__device__ __forceinline__ r3_u4 r3_mxfp4_unpack8_bf16_t(unsigned wv, unsigned lo0, unsigned lo1, unsigned hi0,
                                                         unsigned hi1) {
    const unsigned ev = wv & 0x0f0f0f0fu, od = (wv >> 4) & 0x0f0f0f0fu;
    const unsigned ei = ev & 0x07070707u, oi = od & 0x07070707u;
    const unsigned e_lo = __builtin_amdgcn_perm(lo1, lo0, ei);
    const unsigned e_hi = __builtin_amdgcn_perm(hi1, hi0, ei) | ((ev & 0x08080808u) << 4);
    const unsigned o_lo = __builtin_amdgcn_perm(lo1, lo0, oi);
    const unsigned o_hi = __builtin_amdgcn_perm(hi1, hi0, oi) | ((od & 0x08080808u) << 4);
    const unsigned e01 = __builtin_amdgcn_perm(e_hi, e_lo, 0x05010400u), e23 = __builtin_amdgcn_perm(e_hi, e_lo, 0x07030602u);
    const unsigned o01 = __builtin_amdgcn_perm(o_hi, o_lo, 0x05010400u), o23 = __builtin_amdgcn_perm(o_hi, o_lo, 0x07030602u);
    r3_u4 out;
    out[0] = __builtin_amdgcn_perm(o01, e01, 0x05040100u);  // k 0, 1
    out[1] = __builtin_amdgcn_perm(o01, e01, 0x07060302u);  // k 2, 3
    out[2] = __builtin_amdgcn_perm(o23, e23, 0x05040100u);  // k 4, 5
    out[3] = __builtin_amdgcn_perm(o23, e23, 0x07060302u);  // k 6, 7
    return out;
}

// MR: activation rows a wave stages (1, 4 or 16), the smallest that covers M.
template <int MR, bool NT, bool ABLK>
__global__ __launch_bounds__(R3_D11_WAVES * 32) void r3_gemm_mxfp4a8_dec11_kernel(
        const unsigned char* __restrict__ A, const float* __restrict__ As, const unsigned int* __restrict__ W,
        const unsigned char* __restrict__ Ws, const unsigned char* __restrict__ Wref,
        __hip_bfloat16* __restrict__ C, int M, int K, int N, int as_rs, int mode) {
    typedef float r3_v8fa __attribute__((ext_vector_type(8), aligned(4)));
    __shared__ unsigned short sA[R3_D11_WAVES * MR * 128];  // a wave's strip: MR rows x 128 of K, bf16
    __shared__ unsigned sT[64];                             // the unpack tables: [d] -> lo0, lo1, hi0, hi1
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int col = lane & 15, half = lane >> 4;
    const int tile = blockIdx.x * R3_D11_WAVES + wave;  // each wave owns one 16-column tile, over all of K
    const bool live = tile * 16 < N;                    // N is a multiple of 16; the last block may have spare waves
    const int n0 = live ? tile * 16 : N - 16;           // a spare wave reads the last tile and writes nothing
    const int nrow = n0 + col;             // the weight row this lane's A fragment holds
    const int m = col < M ? col : M - 1;   // the token this lane's B fragment holds (clamped, never predicated)
    const int mr = m < MR ? m : MR - 1;    // its row in the strip (MR >= M, so this is m)

    const int ksteps = K / 16;
    const int groups = (ksteps + 7) / 8;
    const int g_lo = 0, g_hi = live ? groups : 0;

    const unsigned int* __restrict__ wp = W + ((size_t) (n0 >> 4) * ksteps) * 32;
    const int ref = (int) Wref[nrow];
    unsigned short* __restrict__ strip = sA + wave * (MR * 128);
    const float* __restrict__ asrow = As + (size_t) m * as_rs;

    // The unpack tables go to shared memory once: read from the constant segment they would be a second load that
    // depends on the exponent just loaded, in every 32 of K, and the wave would wait on memory twice in sequence.
    if (tid < 64) {
        sT[tid] = (tid & 2) ? kR3MxBf16Hi[tid >> 2][tid & 1] : kR3MxBf16Lo[tid >> 2][tid & 1];
    }
    __syncthreads();

    // A GROUP'S LOADS ARE ISSUED ONE GROUP AHEAD. The timing ablation of the first form (README, 2026-10-10) left
    // 124 of 227 us with the weight loads, the unpack and the matrix instruction all switched off: the wave waited on
    // memory once every two K steps. Here the 16 code dwords and 4 exponents of group g + 1 are requested before
    // group g is computed, so they are in flight across its eight multiplies and a wave waits once a group at most.
    const int kblocks = K / 32;
    auto request = [&](int g, unsigned (&cw)[16], unsigned (&e8)[4]) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int ks = g * 8 + s < ksteps ? g * 8 + s : ksteps - 1;  // clamped: the tail's extra reads are unused
            const unsigned int* __restrict__ slot = wp + (size_t) ks * 32 + col;
            if (mode & 4) {  // timing only: no weight loads
                cw[2 * s] = (unsigned) ks * 0x01010101u; cw[2 * s + 1] = ~cw[2 * s];
            } else {
                cw[2 * s] = NT ? __builtin_nontemporal_load(slot) : slot[0];                // k 0-7
                cw[2 * s + 1] = NT ? __builtin_nontemporal_load(slot + 16) : slot[16];      // k 8-15
            }
        }
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int blk = g * 4 + b < kblocks ? g * 4 + b : kblocks - 1;
            e8[b] = (unsigned) Ws[(size_t) blk * N + nrow];
        }
    };
    unsigned cw[16] = {}, e8[4] = {}, ncw[16] = {}, ne8[4] = {};
    if (g_lo < g_hi) request(g_lo, cw, e8);

    r3_v8f acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, total = acc;
    for (int g = g_lo; g < g_hi; ++g) {
        const int ks0 = g * 8;
        const int nsteps = ks0 + 8 <= ksteps ? 8 : ksteps - ks0;  // K is a multiple of 64: the last group may be half
        if (g + 1 < g_hi) request(g + 1, ncw, ne8);
        // this wave's activations for the group, E4M3 to bf16: chunk c is 16 of K of strip row c / 8
        for (int c = lane; c < MR * 8; c += 32) {
            const int row = c >> 3, kc = (c & 7) * 16;
            if (kc < nsteps * 16) {
                const int rc = row < M ? row : M - 1;
                r3_store16_e4m3_as_bf16(strip + row * 128 + kc,
                                        *reinterpret_cast<const r3_u4u*>(A + (size_t) rc * K + ks0 * 16 + kc));
            }
        }
        unsigned lo0 = 0, lo1 = 0, hi0 = 0, hi1 = 0;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < nsteps) {
                if ((s & 1) == 0) {  // one E8M0 exponent per 32 of K
                    const int d = r4d_mxfp4_fold_d(ref, (int) e8[s >> 1]);
                    lo0 = sT[d * 4]; lo1 = sT[d * 4 + 1]; hi0 = sT[d * 4 + 2]; hi1 = sT[d * 4 + 3];
                }
                const unsigned int c0 = cw[2 * s], c1 = cw[2 * s + 1];
                r3_u8 wrow;
                if (mode & 2) {  // timing only: no unpack
                    wrow[0] = c0; wrow[1] = c1; wrow[2] = c0; wrow[3] = c1; wrow[4] = c0; wrow[5] = c1; wrow[6] = c0; wrow[7] = c1;
                } else {
                    const r3_u4 w0 = r3_mxfp4_unpack8_bf16_t(c0, lo0, lo1, hi0, hi1);
                    const r3_u4 w1 = r3_mxfp4_unpack8_bf16_t(c1, lo0, lo1, hi0, hi1);
                    wrow[0] = w0[0]; wrow[1] = w0[1]; wrow[2] = w0[2]; wrow[3] = w0[3];
                    wrow[4] = w1[0]; wrow[5] = w1[1]; wrow[6] = w1[2]; wrow[7] = w1[3];
                }
                if (mode & 1) {  // timing only: no matrix instruction
                    acc += __builtin_bit_cast(r3_v8f, wrow);
                } else {
                    acc = r3_wmma_bf16_native(__builtin_bit_cast(r3_v16bf, wrow),
                                              r3_row16_e4m3_to_bf16(strip + mr * 128 + s * 16), acc);
                }
            }
        }
        if constexpr (ABLK) {  // the group's own activation scale, per column
            total += acc * asrow[g];
            acc = r3_v8f{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) cw[i] = ncw[i];
#pragma unroll
        for (int i = 0; i < 4; ++i) e8[i] = ne8[i];
    }
    if constexpr (!ABLK) {
        total = acc;
    }
    if (live && col < M) {
        const r3_v8f sum = total;
        const float sc = ABLK ? 1.f : asrow[0];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int n = n0 + 2 * e + half;  // gfx11 accumulator: row 2e + half, column `col`
            C[(size_t) col * N + n] = (__hip_bfloat16) (sum[e] * __int_as_float((int) Wref[n] << 23) * sc);
        }
    }
}

// Launch for M <= 16. Operands are r4d_gemm_mxfp4a8_decode's; `ablk` is the (row, 128) activation scale grid.
static inline void r3_mxfp4_dec11_launch(long a, long ascale, long wq, long ws, long wref, long c, int M, int K,
                                         int N, bool nt, bool ablk, int as_rs, long stream) {
    const dim3 grid((N / 16 + R3_D11_WAVES - 1) / R3_D11_WAVES, 1, 1), block(R3_D11_WAVES * 32);
    hipStream_t st = (hipStream_t) stream;
    auto Ap = (const unsigned char*) a;
    auto Sp = (const float*) ascale;
    auto Wp = (const unsigned int*) wq;
    auto Zp = (const unsigned char*) ws;
    auto Rp = (const unsigned char*) wref;
    auto Cp = (__hip_bfloat16*) c;
    // R3_D11_MODE: timing ablation, results are WRONG when set. 1 = no matrix instruction, 2 = no unpack, 4 = no
    // weight loads; add them to combine. For locating the kernel's cost with --profile-ops, nothing else.
    static const int mode = std::getenv("R3_D11_MODE") ? std::atoi(std::getenv("R3_D11_MODE")) : 0;
#define R3_D11_L(MR_, NT_, AB_) \
    hipLaunchKernelGGL((r3_gemm_mxfp4a8_dec11_kernel<MR_, NT_, AB_>), grid, block, 0, st, Ap, Sp, Wp, Zp, Rp, Cp, M, K, N, as_rs, mode)
#define R3_D11_AB(MR_, NT_) { if (ablk) R3_D11_L(MR_, NT_, true); else R3_D11_L(MR_, NT_, false); }
#define R3_D11_NT(MR_) { if (nt) R3_D11_AB(MR_, true) else R3_D11_AB(MR_, false) }
    if (M == 1) R3_D11_NT(1) else if (M <= 4) R3_D11_NT(4) else R3_D11_NT(16)
#undef R3_D11_NT
#undef R3_D11_AB
#undef R3_D11_L
}
