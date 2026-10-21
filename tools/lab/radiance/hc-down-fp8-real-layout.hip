// RAD17: validates the FP8 down-product WMMA kernel against the REAL on-disk weight layout (GPT dev-agent
// design, req_bigcherry_757a112f3c07): each weight row is one packed [K E4M3 codes][K/128 f32 scales][pad]
// blob, real stride dld=row_bytes(K,128)=10560 for K=10240 (confirmed against the live container, #366) --
// NOT a clean [N,K] codes array at ldm=K like the earlier validation harness (hc-down-fp8-validate.hip)
// assumed. This is the first test combining: real ground-truth N (quantised by the validated
// hc_pf2_quant_a8_kernel design), real ground-truth W_down bytes (dumped from the live container, #366),
// and the proven rocwmma::fragment WMMA mechanics -- end to end, on the real packed weight format.
//
// matrix_b's col_major load still works directly on the real weight buffer: for a 16-column tile, consecutive
// weight rows (=columns in col-major terms) are dld bytes apart, so ldm=dld (not K) is the only change needed
// from the earlier harness. The group scale for output column n, K-group g is a plain f32 at
// weight_row_n_base + K + g*4 (immediately after that row's K code bytes) -- read directly, not from a
// separate scale array.
//
// Build: /opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx1201 -O2 -I/opt/rocm-7.2.4/include \
//          hc-down-fp8-real-layout.hip -o hc-down-fp8-real-layout
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace rocwmma;

constexpr int M = 16, N = 16, K = 10240, KB = 128, NKB = K / KB;
constexpr long DLD = 10560;  // real row stride: K codes + K/128 f32 scales, confirmed against the container

using Afrag = fragment<matrix_a, 16, 16, 16, float8_t, row_major>;
using Bfrag = fragment<matrix_b, 16, 16, 16, float8_t, col_major>;
using Cfrag = fragment<accumulator, 16, 16, 16, float>;

// One 16x16 output tile, one workgroup, one wave (32 threads) -- minimal, to isolate the real-layout question
// from the tiling/scheduling questions already answered elsewhere. A (activation) stays a clean [M,K] array
// (that side's real layout -- what the quant_a8 pre-pass writes -- was already validated separately); only B
// (the weight) uses the real packed-row format here.
__global__ void hc_down_fp8_real_b(const uint8_t* __restrict__ Aq, const float* __restrict__ Asc,
                                   const uint8_t* __restrict__ Wraw, float* __restrict__ D) {
    float acc[16][16] = {};
    for (int kb = 0; kb < NKB; ++kb) {
        Cfrag fc;
        fill_fragment(fc, 0.0f);
        for (int s = 0; s < KB / 16; ++s) {
            const int k0 = kb * KB + s * 16;
            Afrag fa; Bfrag fb;
            load_matrix_sync(fa, reinterpret_cast<const float8_t*>(Aq + k0), K);
            // The real weight layout: row n's codes start at Wraw + n*DLD; col_major with ldm=DLD reads
            // row k, col n at Wraw + n*DLD + k, exactly where that row's codes live.
            load_matrix_sync(fb, reinterpret_cast<const float8_t*>(Wraw + k0), DLD);
            mma_sync(fc, fa, fb, fc);
        }
        __shared__ float part[16][16];
        store_matrix_sync(&part[0][0], fc, 16, mem_row_major);
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j) {
                // Weight scale for column j, group kb: f32 immediately after row j's K code bytes.
                const float* wsc = reinterpret_cast<const float*>(Wraw + j * DLD + K);
                acc[i][j] += part[i][j] * Asc[i * NKB + kb] * wsc[kb];
            }
    }
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) D[i * N + j] = acc[i][j];
}

int main(int argc, char** argv) {
    const char* npath = argc > 1 ? argv[1] : "/mnt/data/bigcherry-work/ndump.bin";
    const char* wpath = argc > 2 ? argv[2] : "/mnt/data/bigcherry-work/wdump.bin";
    FILE* fn = fopen(npath, "rb");
    FILE* fw = fopen(wpath, "rb");
    if (!fn || !fw) { printf("cannot open inputs\n"); return 1; }
    std::vector<float> n_row(K);
    fread(n_row.data(), sizeof(float), K, fn);
    fclose(fn);
    std::vector<uint8_t> w_raw(N * DLD);
    const size_t got = fread(w_raw.data(), 1, N * DLD, fw);
    fclose(fw);
    if (got != (size_t)(N * DLD)) { printf("short weight read: got %zu want %ld\n", got, N * DLD); return 1; }

    // Replicate the real N across all M rows (one real row available, same approach as hc-down-fp8-e2e.hip).
    std::vector<float> Af(M * K);
    for (int m = 0; m < M; ++m) std::copy(n_row.begin(), n_row.end(), Af.begin() + (size_t)m * K);

    // Quantise A with the already-validated per-128-group scheme (host side, for this test's reference build).
    std::vector<uint8_t> Aq(M * K);
    std::vector<float> Asc(M * NKB);
    for (int m = 0; m < M; ++m)
        for (int g = 0; g < NKB; ++g) {
            float amax = 0.0f;
            for (int k = 0; k < KB; ++k) amax = fmaxf(amax, fabsf(Af[m * K + g * KB + k]));
            const float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
            Asc[m * NKB + g] = scale;
            for (int k = 0; k < KB; ++k) {
                const float8_t c = float8_t(Af[m * K + g * KB + k] / scale);
                Aq[m * K + g * KB + k] = *reinterpret_cast<const uint8_t*>(&c);
            }
        }

    // Reference: dequantise A (as the kernel will) and the REAL weight bytes (as actually stored), multiply.
    std::vector<float> Wdq(N * K);
    for (int n = 0; n < N; ++n)
        for (int g = 0; g < NKB; ++g) {
            const float wsc = *reinterpret_cast<const float*>(w_raw.data() + n * DLD + K + g * 4);
            for (int k = 0; k < KB; ++k) {
                const float8_t c = *reinterpret_cast<const float8_t*>(&w_raw[n * DLD + g * KB + k]);
                Wdq[n * K + g * KB + k] = float(c) * wsc;
            }
        }
    std::vector<float> Adq(M * K);
    for (int m = 0; m < M; ++m)
        for (int g = 0; g < NKB; ++g) {
            for (int k = 0; k < KB; ++k) {
                const float8_t c = *reinterpret_cast<const float8_t*>(&Aq[m * K + g * KB + k]);
                Adq[m * K + g * KB + k] = float(c) * Asc[m * NKB + g];
            }
        }
    std::vector<float> Dref(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double s = 0;
            for (int k = 0; k < K; ++k) s += (double)Adq[m * K + k] * Wdq[n * K + k];
            Dref[m * N + n] = (float)s;
        }

    uint8_t *dAq, *dW; float *dAsc, *dD;
    hipMalloc(&dAq, Aq.size()); hipMalloc(&dW, w_raw.size());
    hipMalloc(&dAsc, Asc.size() * sizeof(float)); hipMalloc(&dD, M * N * sizeof(float));
    hipMemcpy(dAq, Aq.data(), Aq.size(), hipMemcpyHostToDevice);
    hipMemcpy(dW, w_raw.data(), w_raw.size(), hipMemcpyHostToDevice);
    hipMemcpy(dAsc, Asc.data(), Asc.size() * sizeof(float), hipMemcpyHostToDevice);

    hc_down_fp8_real_b<<<1, 32>>>(dAq, dAsc, dW, dD);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("FAIL launch: %s\n", hipGetErrorString(e)); return 1; }
    std::vector<float> D(M * N);
    hipMemcpy(D.data(), dD, M * N * sizeof(float), hipMemcpyDeviceToHost);

    double maxabs = 0, maxrel = 0;
    for (int i = 0; i < M * N; ++i) {
        const double d = fabs(D[i] - Dref[i]);
        maxabs = std::max(maxabs, d);
        if (fabs(Dref[i]) > 1e-3) maxrel = std::max(maxrel, d / fabs(Dref[i]));
    }
    printf("real-layout FP8 down-product vs quantised-operand reference (real N, real W_down bytes):\n");
    printf("  max abs diff=%.6g max rel diff=%.6g (expect ~0; a large value means ldm=DLD or the scale offset is wrong)\n",
           maxabs, maxrel);
    printf("D[0]=%f Dref[0]=%f\n", D[0], Dref[0]);
    return maxrel < 1e-2 ? 0 : 1;
}
