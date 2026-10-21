// RAD17 (GPT dev-agent review, strategic pivot): benchmark a tuned hipBLASLt BF16 GEMM at the up-product's
// real shape (M=2048, K=320, N=10240) against the existing fused hc_up_mix_wide_kernel's measured 261us, to
// find out whether the shared BF16 up/mix kernel is leaving real performance on the table before chasing
// any more FP8-path optimisation. This is matrix-multiply-only -- it does NOT include the sigmoid/mean
// epilogue hc_up_mix_wide_kernel also does in the same kernel, so a GEMM win here is necessary but not
// sufficient; see the harness's own printed caveat.
//
// Build: /opt/rocm-7.2.4/bin/hipcc --offload-arch=gfx1201 -O2 -I/opt/rocm-7.2.4/include \
//          hc-upmix-gemm-bench.hip -o hc-upmix-gemm-bench -lhipblaslt -L/opt/rocm-7.2.4/lib
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

#define CHECK_HIP(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); return 1; } } while (0)
#define CHECK_HIPBLASLT(x) do { hipblasStatus_t st = (x); if (st != HIPBLAS_STATUS_SUCCESS) { \
    printf("hipBLASLt error %s:%d: status=%d\n", __FILE__, __LINE__, (int)st); return 1; } } while (0)

int main(int argc, char** argv) {
    const int64_t M = 2048, K = 320, N = 10240;
    const int iters = argc > 1 ? atoi(argv[1]) : 50;

    hipblasLtHandle_t handle;
    printf("step: create handle\n"); fflush(stdout);
    CHECK_HIPBLASLT(hipblasLtCreate(&handle));
    printf("step: handle created\n"); fflush(stdout);

    // A: [M, K] row-major (the "d" operand, bf16). B: [K, N] row-major (W_up, bf16). C/D: [M, N] bf16.
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_16BF, K, M, K));   // col-major storage trick: lda=K
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16BF, N, K, N));   // ldb=N
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_16BF, N, M, N));   // ldc=N

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t opN = HIPBLAS_OP_N;
    CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    void *dA, *dB, *dC;
    CHECK_HIP(hipMalloc(&dA, (size_t)M * K * sizeof(hip_bfloat16)));
    CHECK_HIP(hipMalloc(&dB, (size_t)K * N * sizeof(hip_bfloat16)));
    CHECK_HIP(hipMalloc(&dC, (size_t)M * N * sizeof(hip_bfloat16)));
    std::vector<hip_bfloat16> hA((size_t)M * K), hB((size_t)K * N);
    for (auto& v : hA) v = hip_bfloat16(0.01f);
    for (auto& v : hB) v = hip_bfloat16(0.01f);
    CHECK_HIP(hipMemcpy(dA, hA.data(), hA.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, hB.data(), hB.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));

    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT(hipblasLtMatmulPreferenceCreate(&pref));
    size_t workspaceSize = 32 * 1024 * 1024;
    void* workspace;
    CHECK_HIP(hipMalloc(&workspace, workspaceSize));
    CHECK_HIPBLASLT(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    printf("step: about to call hipblasLtMatmulAlgoGetHeuristic\n"); fflush(stdout);
    const int kReqAlgos = 8;
    hipblasLtMatmulHeuristicResult_t heuristics[kReqAlgos];
    int returnedAlgoCount = 0;
    CHECK_HIPBLASLT(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmul, layoutA, layoutB, layoutC, layoutC, pref, kReqAlgos, heuristics, &returnedAlgoCount));
    printf("step: heuristic search returned\n"); fflush(stdout);
    if (returnedAlgoCount == 0) { printf("FAIL: no hipBLASLt algo found for this shape\n"); return 1; }
    printf("hipBLASLt found %d candidate algo(s) for M=%lld K=%lld N=%lld bf16\n",
           returnedAlgoCount, (long long)M, (long long)K, (long long)N);

    const float alpha = 1.0f, beta = 0.0f;
    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));

    // Warmup + correctness smoke (just check it runs without error), then time the best heuristic.
    double best_us = 1e18;
    int best_idx = -1;
    for (int a = 0; a < returnedAlgoCount; ++a) {
        printf("step: trying algo %d, about to call hipblasLtMatmul (warmup)\n", a); fflush(stdout);
        // Row-major A[M,K]@B[K,N]=C[M,N] via the standard col-major-library trick: describe each matrix
        // as its OWN transpose (col-major layouts above already do this), then compute C^T = B^T @ A^T,
        // i.e. swap the operand order passed here relative to the row-major A,B,C naming.
        hipblasStatus_t st = hipblasLtMatmul(
            handle, matmul, &alpha, dB, layoutB, dA, layoutA, &beta, dC, layoutC, dC, layoutC,
            &heuristics[a].algo, workspace, workspaceSize, stream);
        printf("step: algo %d warmup call returned, status=%d\n", a, (int)st); fflush(stdout);
        if (st != HIPBLAS_STATUS_SUCCESS) continue;
        printf("step: algo %d about to sync\n", a); fflush(stdout);
        CHECK_HIP(hipStreamSynchronize(stream));
        printf("step: algo %d sync done, starting timing loop\n", a); fflush(stdout);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            hipblasLtMatmul(handle, matmul, &alpha, dB, layoutB, dA, layoutA, &beta, dC, layoutC, dC, layoutC,
                             &heuristics[a].algo, workspace, workspaceSize, stream);
        }
        printf("step: algo %d timing loop launches issued, about to final sync\n", a); fflush(stdout);
        CHECK_HIP(hipStreamSynchronize(stream));
        printf("step: algo %d final sync done\n", a); fflush(stdout);
        auto t1 = std::chrono::high_resolution_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        printf("  algo %d: %.2f us/call\n", a, us);
        if (us < best_us) { best_us = us; best_idx = a; }
    }
    if (best_idx < 0) { printf("FAIL: no algo ran successfully\n"); return 1; }

    printf("\nBEST: algo %d, %.2f us/call (matmul only, no sigmoid/mean epilogue)\n", best_idx, best_us);
    printf("Compare against hc_up_mix_wide_kernel's measured 261us (fused matmul+sigmoid+mean).\n");
    printf("A GEMM win here is necessary but not sufficient -- the real question is whether a\n");
    printf("GEMM + separate epilogue beats 261us end to end, which this harness does not measure.\n");
    return 0;
}
