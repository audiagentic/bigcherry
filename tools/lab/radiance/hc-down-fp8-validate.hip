// RAD15: standalone correctness validation for an FP8 WMMA down-product kernel, before any radiance patch.
//
// Validates the real shape (n=2560, hc=4 -> HN=10240, lowrank=320) at a 64-row tile: d[m,n] = sum_k A[m,k]*B[n,k],
// A and B both E4M3 with an f32 (A) / bf16 (B) scale per (row, 128 of K) -- B's layout matches W_down's existing
// row-major [lr,HN] storage (one contiguous K-row of codes per output column), which rocwmma's col_major load
// reads directly with ldm=HN, no rearrangement.
//
// Uses rocwmma::fragment (NOT the raw __builtin_amdgcn_wmma_* with a hand-rolled lane mapping): the high-level API
// was proven correct on this hardware via single-nonzero-element probes (RAD15 notes) before trusting it here.
//
// Build:  /opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx1201 -O2 -I/opt/rocm-7.2.4/include hc-down-fp8-validate.hip \
//           -o hc-down-fp8-validate
// Run:    ROCR_VISIBLE_DEVICES=2 ./hc-down-fp8-validate
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

using namespace rocwmma;

constexpr int M = 64, N = 320, K = 10240, KB = 128, NKB = K / KB;  // 80 scale blocks
constexpr int NT = N / 16;  // 20 output column tiles

using Afrag = fragment<matrix_a, 16, 16, 16, float8_t, row_major>;
using Bfrag = fragment<matrix_b, 16, 16, 16, float8_t, col_major>;
using Cfrag = fragment<accumulator, 16, 16, 16, float>;

// One workgroup = one 16-row tile x all 20 column tiles (one wave per column tile, 20 waves -> 640 threads;
// clamped to kThreads-sized launch bounds by using 4 waves and looping 5 column tiles per wave, matching the
// existing codebase's kThreads=256 convention elsewhere in r4d_hc_bf16.hip).
constexpr int WAVES = 8;              // 256 threads
constexpr int TILES_PER_WAVE = (NT + WAVES - 1) / WAVES;  // ceil(20/8) = 3

__global__ __launch_bounds__(256) void hc_down_fp8(
    const uint8_t* __restrict__ Aq, const float* __restrict__ Asc,     // [M,K] codes, [M,NKB] scales
    const uint8_t* __restrict__ Bq, const float* __restrict__ Bsc,     // [N,K] codes, [N,NKB] scales (f32 here; bf16 in production)
    float* __restrict__ D) {                                           // [M,N]
    const int wave = threadIdx.x / 32;
    const int mrow_base = blockIdx.x * 16;

    for (int t = 0; t < TILES_PER_WAVE; ++t) {
        const int ntile = wave * TILES_PER_WAVE + t;
        if (ntile >= NT) break;
        const int ncol_base = ntile * 16;
        float acc[16][16] = {};  // only lane 0 of the wave accumulates the unpacked tile into this (see below)

        // KB=128 is the SCALE block width, not the WMMA K-step (16): each scale block needs KB/16=8 WMMA
        // calls accumulating raw (unscaled) products into one fragment before that block's scale is applied
        // once. Looping kb*KB as if it were a 16-wide WMMA step (an earlier version of this harness did
        // exactly that) silently consumes only 1/8 of K -- caught here because this harness compares against
        // a reference over the FULL K, not because the kernel errored.
        for (int kb = 0; kb < NKB; ++kb) {
            Cfrag fc;
            fill_fragment(fc, 0.0f);
            for (int s = 0; s < KB / 16; ++s) {
                const int k0 = kb * KB + s * 16;
                Afrag fa; Bfrag fb;
                load_matrix_sync(fa, reinterpret_cast<const float8_t*>(Aq + (size_t)mrow_base * K + k0), K);
                load_matrix_sync(fb, reinterpret_cast<const float8_t*>(Bq + (size_t)ncol_base * K + k0), K);
                mma_sync(fc, fa, fb, fc);
            }
            // Indexed by wave: each wave computes a different column tile concurrently, so a single shared
            // buffer here would have every wave overwrite the others' partial sums (caught by this harness).
            __shared__ float part[WAVES][16][16];
            store_matrix_sync(&part[wave][0][0], fc, 16, mem_row_major);
            __syncwarp();
            if (threadIdx.x % 32 == 0) {
                for (int i = 0; i < 16; ++i)
                    for (int j = 0; j < 16; ++j) {
                        const float sa = Asc[(mrow_base + i) * NKB + kb];
                        const float sb = Bsc[(ncol_base + j) * NKB + kb];
                        acc[i][j] += part[wave][i][j] * sa * sb;
                    }
            }
        }
        if (threadIdx.x % 32 == 0)
            for (int i = 0; i < 16; ++i)
                for (int j = 0; j < 16; ++j) D[(size_t)(mrow_base + i) * N + ncol_base + j] = acc[i][j];
    }
}

// Round-trips a float through E4M3 at the given scale (same convention as the production quantiser: x/scale
// clamped to the format's range, reconstructed as code*scale) so the host reference sees exactly what the device
// kernel computes from, isolating WMMA/layout/scale-application bugs from ordinary quantisation error.
float8_t to_e4m3(float x) { return float8_t(x); }
float from_e4m3(float8_t c) { return float(c); }

int main() {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> Af(M * K), Bf(N * K);
    for (auto& v : Af) v = dist(rng);
    for (auto& v : Bf) v = dist(rng);

    std::vector<uint8_t> Aq(M * K), Bq(N * K);
    std::vector<float> Asc(M * NKB), Bsc(N * NKB);
    std::vector<float> Adq(M * K), Bdq(N * K);  // dequantised, for the reference

    for (int m = 0; m < M; ++m)
        for (int kb = 0; kb < NKB; ++kb) {
            float amax = 0.0f;
            for (int k = 0; k < KB; ++k) amax = std::max(amax, std::fabs(Af[m * K + kb * KB + k]));
            const float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
            Asc[m * NKB + kb] = scale;
            for (int k = 0; k < KB; ++k) {
                const float8_t c = to_e4m3(Af[m * K + kb * KB + k] / scale);
                Aq[m * K + kb * KB + k] = *reinterpret_cast<const uint8_t*>(&c);
                Adq[m * K + kb * KB + k] = from_e4m3(c) * scale;
            }
        }
    for (int n = 0; n < N; ++n)
        for (int kb = 0; kb < NKB; ++kb) {
            float amax = 0.0f;
            for (int k = 0; k < KB; ++k) amax = std::max(amax, std::fabs(Bf[n * K + kb * KB + k]));
            const float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
            Bsc[n * NKB + kb] = scale;
            for (int k = 0; k < KB; ++k) {
                const float8_t c = to_e4m3(Bf[n * K + kb * KB + k] / scale);
                Bq[n * K + kb * KB + k] = *reinterpret_cast<const uint8_t*>(&c);
                Bdq[n * K + kb * KB + k] = from_e4m3(c) * scale;
            }
        }

    // Host reference 1: against the quantised operands (isolates WMMA/layout/scale bugs from quant error).
    std::vector<float> Dref_q(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double s = 0;
            for (int k = 0; k < K; ++k) s += (double)Adq[m * K + k] * Bdq[n * K + k];
            Dref_q[m * N + n] = (float)s;
        }
    // Host reference 2: against the original fp32 operands (the real end-to-end quantisation error we'll see).
    std::vector<float> Dref_fp32(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double s = 0;
            for (int k = 0; k < K; ++k) s += (double)Af[m * K + k] * Bf[n * K + k];
            Dref_fp32[m * N + n] = (float)s;
        }

    uint8_t *dAq, *dBq; float *dAsc, *dBsc, *dD;
    hipMalloc(&dAq, Aq.size()); hipMalloc(&dBq, Bq.size());
    hipMalloc(&dAsc, Asc.size() * 4); hipMalloc(&dBsc, Bsc.size() * 4);
    hipMalloc(&dD, M * N * 4);
    hipMemcpy(dAq, Aq.data(), Aq.size(), hipMemcpyHostToDevice);
    hipMemcpy(dBq, Bq.data(), Bq.size(), hipMemcpyHostToDevice);
    hipMemcpy(dAsc, Asc.data(), Asc.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dBsc, Bsc.data(), Bsc.size() * 4, hipMemcpyHostToDevice);

    dim3 grid(M / 16);
    hc_down_fp8<<<grid, 256>>>(dAq, dAsc, dBq, dBsc, dD);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("FAIL launch: %s\n", hipGetErrorString(e)); return 1; }

    std::vector<float> D(M * N);
    hipMemcpy(D.data(), dD, M * N * 4, hipMemcpyDeviceToHost);

    double maxabs_q = 0, maxrel_q = 0, maxabs_fp32 = 0, sumsq_fp32 = 0, sumsq_ref = 0;
    for (int i = 0; i < M * N; ++i) {
        const double dq = std::fabs(D[i] - Dref_q[i]);
        maxabs_q = std::max(maxabs_q, dq);
        if (std::fabs(Dref_q[i]) > 1e-3) maxrel_q = std::max(maxrel_q, dq / std::fabs(Dref_q[i]));
        const double df = D[i] - Dref_fp32[i];
        maxabs_fp32 = std::max(maxabs_fp32, std::fabs(df));
        sumsq_fp32 += df * df;
        sumsq_ref += (double)Dref_fp32[i] * Dref_fp32[i];
    }
    printf("against quantised-operand reference (isolates WMMA/layout/scale bugs):\n");
    printf("  max abs diff=%.6g max rel diff=%.6g  (expect ~0, i.e. exact modulo fp32 accumulation order)\n",
           maxabs_q, maxrel_q);
    printf("against the original fp32 operands (the real end-to-end quantisation error):\n");
    printf("  max abs diff=%.6g  RMS relative error=%.4f%%\n", maxabs_fp32,
           100.0 * std::sqrt(sumsq_fp32 / (M * N)) / std::sqrt(sumsq_ref / (M * N)));
    printf("D[0]=%f Dref_q[0]=%f Dref_fp32[0]=%f\n", D[0], Dref_q[0], Dref_fp32[0]);
    printf("PASS if max rel diff against the quantised reference is near zero (< 1e-3); the fp32-reference RMS\n");
    printf("error is the real quantisation cost to validate against the KL harness, not a bug signal.\n");
    return 0;
}
