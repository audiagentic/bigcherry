// RAD17 (GPT dev-agent review, strategic pivot, step 2 of the decision framework): diagnostic
// GEMM + separate HIP epilogue kernel for hc_up_mix_wide_kernel's real shape (M=2048, K=lr=320,
// HC=4 streams, n=2560 per stream, HN=HC*n=10240), to measure the FULL combined latency against
// the existing fused kernel's measured 261us -- a GEMM win alone (hc-upmix-gemm-bench.hip, 148.12us)
// is necessary but not sufficient per GPT's own criteria.
//
// Epilogue formula (reproduced from r4d_hc_bf16.hip's hc_up_mix_wide_kernel, vendor/radiance/libr4d,
// not re-derived): for each row r and output column j in [0, n):
//   x[r,j] = (1/HC) * sum_{c=0}^{HC-1} bf16(sigmoid(U[r, c*n+j])) * bf16(h[r,c,j] * scalein[r,c] * (w[c*n+j] + wadd))
// where U is the up-product GEMM output (fp32, matching the production kernel's fp32 WMMA
// accumulator -- narrowing to bf16 happens only after sigmoid, same as upstream). wadd is fixed to
// 0 here (upstream passes a weight-decay-style add-on that does not affect the timing question).
//
// LIMITATION (explicit, per GPT's own decision criteria): this compares a hand-written epilogue
// against the *formula*, not byte-for-byte against the production hc_up_mix_wide_kernel binary --
// doing that needs a full radiance build+link, which is a separate, heavier harness. Numerical
// correctness here is checked against a host-computed double-precision reference of the SAME
// formula, which validates the epilogue kernel's own arithmetic, not bit-identity with production.
// Treat a correctness pass here as necessary, not sufficient, before calling this validated against
// upstream.
//
// Build: /opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx1201 -O2 -I/opt/rocm-7.2.4/include \
//          hc-upmix-epilogue-diag.hip -o hc-upmix-epilogue-diag -lhipblaslt -L/opt/rocm-7.2.4/lib
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define CHECK_HIP(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); return 1; } } while (0)
#define CHECK_HIPBLASLT(x) do { hipblasStatus_t st = (x); if (st != HIPBLAS_STATUS_SUCCESS) { \
    printf("hipBLASLt error %s:%d: status=%d\n", __FILE__, __LINE__, (int)st); return 1; } } while (0)

__device__ __forceinline__ unsigned short f2bf_(float f) {
    return (unsigned short)((__builtin_bit_cast(unsigned, f) + 0x8000u) >> 16);
}
__device__ __forceinline__ float bf2f_(unsigned short h) {
    return __builtin_bit_cast(float, (unsigned)h << 16);
}
__device__ __forceinline__ float sigmoidf__(float v) { return 1.0f / (1.0f + __expf(-v)); }

// One block per row, blockDim.x columns per pass (looped if n > blockDim.x).
__global__ void hc_upmix_epilogue_diag_kernel(
        const float* __restrict__ U,            // [rows, HN], HN = HC*n, fp32 GEMM output
        const unsigned short* __restrict__ h,    // [rows, HC, n] bf16
        const float* __restrict__ scalein,       // [rows, HC]
        const float* __restrict__ w,             // [HC*n]
        unsigned short* __restrict__ x,           // [rows, n] bf16 output
        long rows, long n, int HC) {
    const long r = blockIdx.y;
    if (r >= rows) return;
    const long HN = (long)HC * n;
    for (long j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += (long)gridDim.x * blockDim.x) {
        float acc = 0.0f;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            if (c >= HC) break;
            const float u = U[r * HN + (long)c * n + j];
            const float sig = bf2f_(f2bf_(sigmoidf__(u)));
            const float hv = bf2f_(h[(r * HC + c) * n + j]);
            const float nv = bf2f_(f2bf_(hv * scalein[r * HC + c] * w[(long)c * n + j]));
            acc += sig * nv;
        }
        x[r * n + j] = f2bf_(acc * (1.0f / (float)HC));
    }
}

static float host_bf2f(unsigned short h) {
    unsigned u = (unsigned)h << 16;
    float f; __builtin_memcpy(&f, &u, 4); return f;
}
static unsigned short host_f2bf(float f) {
    unsigned u; __builtin_memcpy(&u, &f, 4);
    return (unsigned short)((u + 0x8000u) >> 16);
}

int main(int argc, char** argv) {
    const int64_t M = 2048, K = 320, HC = 4, n = 2560;
    const int64_t HN = HC * n;
    const int iters = argc > 1 ? atoi(argv[1]) : 50;

    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.0f, 0.3f);

    std::vector<hip_bfloat16> h_dbuf((size_t)M * K), h_upv((size_t)HN * K), h_h((size_t)M * HC * n);
    std::vector<float> h_scalein((size_t)M * HC), h_w((size_t)HC * n);
    for (auto& v : h_dbuf) v = hip_bfloat16(nd(rng));
    for (auto& v : h_upv) v = hip_bfloat16(nd(rng));
    for (auto& v : h_h) v = hip_bfloat16(nd(rng));
    for (auto& v : h_scalein) v = std::fabs(nd(rng)) + 0.1f;
    for (auto& v : h_w) v = nd(rng);

    void *dA, *dB, *dU;
    CHECK_HIP(hipMalloc(&dA, (size_t)M * K * sizeof(hip_bfloat16)));
    CHECK_HIP(hipMalloc(&dB, (size_t)HN * K * sizeof(hip_bfloat16)));
    CHECK_HIP(hipMalloc(&dU, (size_t)M * HN * sizeof(float)));
    CHECK_HIP(hipMemcpy(dA, h_dbuf.data(), h_dbuf.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, h_upv.data(), h_upv.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));

    unsigned short *dH, *dX;
    float *dScale, *dW;
    CHECK_HIP(hipMalloc(&dH, h_h.size() * sizeof(unsigned short)));
    CHECK_HIP(hipMalloc(&dScale, h_scalein.size() * sizeof(float)));
    CHECK_HIP(hipMalloc(&dW, h_w.size() * sizeof(float)));
    CHECK_HIP(hipMalloc(&dX, (size_t)M * n * sizeof(unsigned short)));
    CHECK_HIP(hipMemcpy(dH, h_h.data(), h_h.size() * sizeof(unsigned short), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dScale, h_scalein.data(), h_scalein.size() * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dW, h_w.data(), h_w.size() * sizeof(float), hipMemcpyHostToDevice));

    hipblasLtHandle_t handle;
    CHECK_HIPBLASLT(hipblasLtCreate(&handle));
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_16BF, K, M, K));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16BF, HN, K, HN));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_32F, HN, M, HN));

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t opN = HIPBLAS_OP_N;
    CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT(hipblasLtMatmulPreferenceCreate(&pref));
    size_t workspaceSize = 32 * 1024 * 1024;
    void* workspace;
    CHECK_HIP(hipMalloc(&workspace, workspaceSize));
    CHECK_HIPBLASLT(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    const int kReqAlgos = 8;
    hipblasLtMatmulHeuristicResult_t heuristics[kReqAlgos];
    int returnedAlgoCount = 0;
    CHECK_HIPBLASLT(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmul, layoutA, layoutB, layoutC, layoutC, pref, kReqAlgos, heuristics, &returnedAlgoCount));
    if (returnedAlgoCount == 0) { printf("FAIL: no hipBLASLt algo for fp32-output GEMM\n"); return 1; }

    const float alpha = 1.0f, beta = 0.0f;
    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));

    dim3 epBlock(256);
    dim3 epGrid((unsigned)((n + epBlock.x - 1) / epBlock.x), (unsigned)M);

    hipEvent_t e0, e1, e2;
    CHECK_HIP(hipEventCreate(&e0));
    CHECK_HIP(hipEventCreate(&e1));
    CHECK_HIP(hipEventCreate(&e2));

    double best_gemm_us = 1e18, best_ep_us = 1e18, best_combined_us = 1e18;
    int best_algo = -1;

    for (int a = 0; a < returnedAlgoCount; ++a) {
        hipblasStatus_t st = hipblasLtMatmul(handle, matmul, &alpha, dB, layoutB, dA, layoutA,
                                             &beta, dU, layoutC, dU, layoutC,
                                             &heuristics[a].algo, workspace, workspaceSize, stream);
        if (st != HIPBLAS_STATUS_SUCCESS) continue;
        hc_upmix_epilogue_diag_kernel<<<epGrid, epBlock, 0, stream>>>(
            (const float*)dU, dH, dScale, dW, dX, M, n, (int)HC);
        CHECK_HIP(hipStreamSynchronize(stream));

        CHECK_HIP(hipEventRecord(e0, stream));
        for (int i = 0; i < iters; ++i)
            hipblasLtMatmul(handle, matmul, &alpha, dB, layoutB, dA, layoutA, &beta, dU, layoutC,
                             dU, layoutC, &heuristics[a].algo, workspace, workspaceSize, stream);
        CHECK_HIP(hipEventRecord(e1, stream));
        for (int i = 0; i < iters; ++i)
            hc_upmix_epilogue_diag_kernel<<<epGrid, epBlock, 0, stream>>>(
                (const float*)dU, dH, dScale, dW, dX, M, n, (int)HC);
        CHECK_HIP(hipEventRecord(e2, stream));
        CHECK_HIP(hipStreamSynchronize(stream));

        float gemm_ms, ep_ms, total_ms;
        CHECK_HIP(hipEventElapsedTime(&gemm_ms, e0, e1));
        CHECK_HIP(hipEventElapsedTime(&ep_ms, e1, e2));
        CHECK_HIP(hipEventElapsedTime(&total_ms, e0, e2));
        const double gemm_us = gemm_ms * 1000.0 / iters;
        const double ep_us = ep_ms * 1000.0 / iters;
        const double combined_us = total_ms * 1000.0 / iters;
        printf("  algo %d: gemm=%.2f us, epilogue=%.2f us, combined=%.2f us\n", a, gemm_us, ep_us, combined_us);
        if (combined_us < best_combined_us) {
            best_combined_us = combined_us; best_gemm_us = gemm_us; best_ep_us = ep_us; best_algo = a;
        }
    }
    if (best_algo < 0) { printf("FAIL: no algo ran successfully\n"); return 1; }

    // Correctness: host double-precision reference of the SAME formula (see LIMITATION above --
    // this is not a check against the production hc_up_mix_wide_kernel binary).
    std::vector<float> h_U((size_t)M * HN);
    std::vector<unsigned short> h_X((size_t)M * n);
    CHECK_HIP(hipMemcpy(h_U.data(), dU, h_U.size() * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_X.data(), dX, h_X.size() * sizeof(unsigned short), hipMemcpyDeviceToHost));

    double max_abs = 0.0, sum_sq = 0.0, sum_ref_sq = 0.0;
    const int64_t check_rows = 64;  // sample, not the full 2048 rows -- host ref is O(rows*n*HC)
    for (int64_t r = 0; r < check_rows; ++r) {
        for (int64_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int64_t c = 0; c < HC; ++c) {
                double u = (double)h_U[r * HN + c * n + j];
                double sig = 1.0 / (1.0 + std::exp(-u));
                double sig_bf = host_bf2f(host_f2bf((float)sig));
                double hv = host_bf2f((unsigned short)h_h[(r * HC + c) * n + j]);
                double nv = hv * (double)h_scalein[r * HC + c] * (double)h_w[c * n + j];
                double nv_bf = host_bf2f(host_f2bf((float)nv));
                acc += sig_bf * nv_bf;
            }
            double ref = acc / (double)HC;
            double got = host_bf2f(h_X[r * n + j]);
            double d = got - ref;
            max_abs = std::max(max_abs, std::fabs(d));
            sum_sq += d * d;
            sum_ref_sq += ref * ref;
        }
    }
    const double rms_rel = sum_ref_sq > 0.0 ? std::sqrt(sum_sq / sum_ref_sq) : 0.0;

    printf("\nBEST: algo %d -- gemm=%.2f us, epilogue=%.2f us, COMBINED=%.2f us\n",
           best_algo, best_gemm_us, best_ep_us, best_combined_us);
    printf("Compare against hc_up_mix_wide_kernel's measured 261us (production, fused, bit-identical).\n");
    printf("Correctness vs host double-precision reference of the SAME formula (%lld rows sampled):\n",
           (long long)check_rows);
    printf("  max_abs=%.6g rms_rel=%.6g (bf16 ULP-scale error expected; large values indicate a real bug)\n",
           max_abs, rms_rel);
    printf("NOTE: this does NOT establish bit-identity with the production hc_up_mix_wide_kernel binary.\n");
    return 0;
}
