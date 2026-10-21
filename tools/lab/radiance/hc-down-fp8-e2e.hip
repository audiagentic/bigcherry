// RAD15: end-to-end combination of the two separately-validated pieces -- the cross-thread amax reduction
// (hc-pf2-quant-reduce.hip, proven bit-for-bit against offline ground truth) quantising REAL ground-truth N
// (ndump.bin, from a live hc_read call), fed into the WMMA down-product kernel (hc-down-fp8-validate.hip,
// proven correct against a quantised-operand reference). W_down itself stays synthetic (random, same as
// hc-down-fp8-validate.hip) since extracting the real weight needs a second ndump-style hook not yet built --
// this test proves the ACTIVATION side's full real pipeline, not the weight side.
//
// Build: /opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx1201 -O2 -I/opt/rocm-7.2.4/include hc-down-fp8-e2e.hip \
//          -o hc-down-fp8-e2e
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

using namespace rocwmma;

constexpr int M = 64, N = 320, K = 10240, KB = 128, NKB = K / KB;
constexpr int NT = N / 16;
constexpr int kThreads = 256, kKV = 8, kTKT = 64;
constexpr int kRS = kThreads / kKV, RT = 64, kAV = RT / kRS;

using Afrag = fragment<matrix_a, 16, 16, 16, float8_t, row_major>;
using Bfrag = fragment<matrix_b, 16, 16, 16, float8_t, col_major>;
using Cfrag = fragment<accumulator, 16, 16, 16, float>;

// Stage 1: quantise all M rows of real N using the proven cross-thread reduction (hc-pf2-quant-reduce.hip),
// one workgroup per row since M=64 here is small -- the real kernel does this inline per row-tile; here it's
// a separate kernel for clarity, which is equally valid as the "separate quantisation pre-pass" option from
// RAD15's scope notes (option (a), vs. (b) fusing into the slice loop).
__global__ void quantise_n(const float* __restrict__ n_in, uint8_t* __restrict__ codes_out,
                           float* __restrict__ scale_out) {
    __shared__ float red[kThreads];
    const int row = blockIdx.x, tid = threadIdx.x;
    const int ar = tid / kKV, k8a = (tid % kKV) * 8;
    const float* n_row = n_in + (size_t)row * K;
    for (int k0 = 0; k0 < K; k0 += 2 * kTKT) {
        float local_max = 0.0f;
        float vals[16];
#pragma unroll
        for (int e = 0; e < 8; ++e) { vals[e] = n_row[k0 + k8a + e]; local_max = std::max(local_max, std::fabs(vals[e])); }
#pragma unroll
        for (int e = 0; e < 8; ++e) { vals[8 + e] = n_row[k0 + kTKT + k8a + e]; local_max = std::max(local_max, std::fabs(vals[8 + e])); }
        red[tid] = local_max;
        __syncthreads();
        float group_max = 0.0f;
        const int base = ar * kKV;
        for (int i = 0; i < kKV; ++i) group_max = std::max(group_max, red[base + i]);
        __syncthreads();
        const float scale = group_max > 0.0f ? group_max / 448.0f : 1.0f;
        if (tid == 0) scale_out[row * NKB + k0 / KB] = scale;
#pragma unroll
        for (int e = 0; e < 16; ++e) {
            const int k = (e < 8) ? k0 + k8a + e : k0 + kTKT + k8a + (e - 8);
            const float8_t c = float8_t(vals[e] / scale);
            codes_out[(size_t)row * K + k] = *reinterpret_cast<const uint8_t*>(&c);
        }
    }
}

// Stage 2: the WMMA down-product, same kernel as hc-down-fp8-validate.hip.
constexpr int WAVES = 8, TILES_PER_WAVE = (NT + WAVES - 1) / WAVES;
__global__ __launch_bounds__(256) void hc_down_fp8(
    const uint8_t* __restrict__ Aq, const float* __restrict__ Asc,
    const uint8_t* __restrict__ Bq, const float* __restrict__ Bsc, float* __restrict__ D) {
    const int wave = threadIdx.x / 32;
    const int mrow_base = blockIdx.x * 16;
    for (int t = 0; t < TILES_PER_WAVE; ++t) {
        const int ntile = wave * TILES_PER_WAVE + t;
        if (ntile >= NT) break;
        const int ncol_base = ntile * 16;
        float acc[16][16] = {};
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
            __shared__ float part[WAVES][16][16];
            store_matrix_sync(&part[wave][0][0], fc, 16, mem_row_major);
            __syncwarp();
            if (threadIdx.x % 32 == 0)
                for (int i = 0; i < 16; ++i)
                    for (int j = 0; j < 16; ++j)
                        acc[i][j] += part[wave][i][j] * Asc[(mrow_base + i) * NKB + kb] * Bsc[(ncol_base + j) * NKB + kb];
        }
        if (threadIdx.x % 32 == 0)
            for (int i = 0; i < 16; ++i)
                for (int j = 0; j < 16; ++j) D[(size_t)(mrow_base + i) * N + ncol_base + j] = acc[i][j];
    }
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/mnt/data/bigcherry-work/ndump.bin";
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    std::vector<float> one_row(K);
    size_t got = fread(one_row.data(), sizeof(float), K, f);
    fclose(f);
    if (got != (size_t)K) { printf("short read\n"); return 1; }
    // Only one real row is available (ndump dumped row 0); replicate it across all M rows so the WMMA kernel's
    // M=64 shape is exercised with real activation VALUES even though the "rows" aren't 64 distinct real tokens.
    std::vector<float> Af(M * K);
    for (int m = 0; m < M; ++m) std::copy(one_row.begin(), one_row.end(), Af.begin() + (size_t)m * K);

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> Bf(N * K);
    for (auto& v : Bf) v = dist(rng);

    float* dN; uint8_t* dAq; float* dAsc;
    hipMalloc(&dN, (size_t)M * K * sizeof(float));
    hipMalloc(&dAq, (size_t)M * K);
    hipMalloc(&dAsc, (size_t)M * NKB * sizeof(float));
    hipMemcpy(dN, Af.data(), Af.size() * sizeof(float), hipMemcpyHostToDevice);
    quantise_n<<<M, kThreads>>>(dN, dAq, dAsc);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("FAIL quantise_n: %s\n", hipGetErrorString(e)); return 1; }

    std::vector<uint8_t> Bq(N * K);
    std::vector<float> Bsc(N * NKB), Bdq(N * K);
    for (int n = 0; n < N; ++n)
        for (int kb = 0; kb < NKB; ++kb) {
            float amax = 0.0f;
            for (int k = 0; k < KB; ++k) amax = std::max(amax, std::fabs(Bf[n * K + kb * KB + k]));
            const float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
            Bsc[n * NKB + kb] = scale;
            for (int k = 0; k < KB; ++k) {
                const float8_t c = float8_t(Bf[n * K + kb * KB + k] / scale);
                Bq[n * K + kb * KB + k] = *reinterpret_cast<const uint8_t*>(&c);
                Bdq[n * K + kb * KB + k] = float(c) * scale;
            }
        }
    uint8_t* dBq; float* dBsc, * dD;
    hipMalloc(&dBq, Bq.size()); hipMalloc(&dBsc, Bsc.size() * sizeof(float)); hipMalloc(&dD, M * N * sizeof(float));
    hipMemcpy(dBq, Bq.data(), Bq.size(), hipMemcpyHostToDevice);
    hipMemcpy(dBsc, Bsc.data(), Bsc.size() * sizeof(float), hipMemcpyHostToDevice);

    hc_down_fp8<<<M / 16, 256>>>(dAq, dAsc, dBq, dBsc, dD);
    e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("FAIL hc_down_fp8: %s\n", hipGetErrorString(e)); return 1; }
    std::vector<float> D(M * N);
    hipMemcpy(D.data(), dD, M * N * sizeof(float), hipMemcpyDeviceToHost);

    // Reference: the real N (fp32, replicated) against the real weight's quantised reconstruction (isolates
    // the activation-side pipeline's correctness; Bdq already includes weight quantisation, same as before).
    std::vector<float> Dref(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double s = 0;
            for (int k = 0; k < K; ++k) s += (double)Af[m * K + k] * Bdq[n * K + k];
            Dref[m * N + n] = (float)s;
        }

    double sumsq_err = 0, sumsq_ref = 0, maxabs = 0;
    for (int i = 0; i < M * N; ++i) {
        const double d = D[i] - Dref[i];
        sumsq_err += d * d; sumsq_ref += (double)Dref[i] * Dref[i];
        maxabs = std::max(maxabs, std::fabs(d));
    }
    printf("end-to-end (real N, proven cross-thread quantiser, proven WMMA down-product) vs reference:\n");
    printf("  RMS relative error: %.4f%%  max abs diff: %.6g\n",
           100.0 * std::sqrt(sumsq_err / (M * N)) / std::sqrt(sumsq_ref / (M * N)), maxabs);
    printf("D[0]=%f Dref[0]=%f\n", D[0], Dref[0]);
    printf("This isolates the ACTIVATION-side real-data pipeline; the weight side stays synthetic (no real-weight\n");
    printf("dump built yet). A small residual RMS error here is expected (activation E4M3 quantisation, ~2.5%%\n");
    printf("measured earlier on this exact row); a LARGE one would mean the two pieces don't compose correctly.\n");
    return 0;
}
