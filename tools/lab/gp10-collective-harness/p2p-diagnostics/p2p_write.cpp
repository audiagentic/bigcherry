// GP11 deep-dive part 2: peer WRITES work where peer reads return zero.
// Characterize the write path at scale + measure validated bandwidth.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>

#define CHK(call)                                                              \
    do {                                                                       \
        hipError_t _rc = (call);                                               \
        if (_rc != hipSuccess) {                                               \
            printf("  !! HIP error at %d: %s\n", __LINE__, hipGetErrorString(_rc)); \
        }                                                                      \
    } while (0)

// scalar peer store
__global__ void k_push(float * dst_peer, const float * src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = gridDim.x * blockDim.x;
    for (; i < n; i += s) dst_peer[i] = src[i];
}
// vectorized (float4) peer store
__global__ void k_push4(float4 * dst_peer, const float4 * src, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = gridDim.x * blockDim.x;
    for (; i < n4; i += s) dst_peer[i] = src[i];
}

int main() {
    const int D0 = 0, D1 = 1;
    CHK(hipSetDevice(D0)); CHK(hipDeviceEnablePeerAccess(D1, 0));
    CHK(hipSetDevice(D1)); CHK(hipDeviceEnablePeerAccess(D0, 0));

    const std::vector<int> sizes = {7680, 30720, 122880, 1048576, 16777216};

    printf("%-12s %-10s %-12s %-10s %-12s %-10s\n",
           "elements", "MB", "scalar_ok", "sc_GB/s", "vec4_ok", "v4_GB/s");

    for (int n : sizes) {
        size_t bytes = (size_t) n * sizeof(float);
        CHK(hipSetDevice(D0));
        float * src = nullptr; CHK(hipMalloc(&src, bytes));
        CHK(hipSetDevice(D1));
        float * dst = nullptr; CHK(hipMalloc(&dst, bytes));

        std::vector<float> h(n), back(n);
        for (int i = 0; i < n; ++i) h[i] = (float) (i % 1000) * 0.5f + 1.0f;
        CHK(hipSetDevice(D0));
        CHK(hipMemcpy(src, h.data(), bytes, hipMemcpyHostToDevice));

        auto zero_dst = [&]() {
            CHK(hipSetDevice(D1));
            CHK(hipMemset(dst, 0, bytes));
            CHK(hipDeviceSynchronize());
        };
        auto verify = [&]() -> bool {
            CHK(hipSetDevice(D1));
            CHK(hipMemcpy(back.data(), dst, bytes, hipMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i)
                if (std::fabs(back[i] - h[i]) > 1e-6f) return false;
            return true;
        };
        auto timeit = [&](bool vec) -> double {
            CHK(hipSetDevice(D0));
            hipEvent_t a, b; CHK(hipEventCreate(&a)); CHK(hipEventCreate(&b));
            const int iters = 20;
            // warmup
            if (vec) k_push4<<<256, 256>>>((float4 *) dst, (const float4 *) src, n / 4);
            else     k_push<<<256, 256>>>(dst, src, n);
            CHK(hipDeviceSynchronize());
            CHK(hipEventRecord(a));
            for (int it = 0; it < iters; ++it) {
                if (vec) k_push4<<<256, 256>>>((float4 *) dst, (const float4 *) src, n / 4);
                else     k_push<<<256, 256>>>(dst, src, n);
            }
            CHK(hipEventRecord(b));
            CHK(hipDeviceSynchronize());
            float ms = 0; CHK(hipEventElapsedTime(&ms, a, b));
            CHK(hipEventDestroy(a)); CHK(hipEventDestroy(b));
            return (double) bytes * iters / (ms / 1000.0) / 1e9;
        };

        zero_dst();
        CHK(hipSetDevice(D0));
        k_push<<<256, 256>>>(dst, src, n);
        CHK(hipDeviceSynchronize());
        bool ok_s = verify();
        double gbs_s = timeit(false);

        zero_dst();
        CHK(hipSetDevice(D0));
        k_push4<<<256, 256>>>((float4 *) dst, (const float4 *) src, n / 4);
        CHK(hipDeviceSynchronize());
        bool ok_v = verify();
        double gbs_v = timeit(true);

        printf("%-12d %-10.2f %-12s %-10.2f %-12s %-10.2f\n",
               n, bytes / 1e6, ok_s ? "OK" : "**FAIL**", gbs_s,
               ok_v ? "OK" : "**FAIL**", gbs_v);

        CHK(hipSetDevice(D0)); CHK(hipFree(src));
        CHK(hipSetDevice(D1)); CHK(hipFree(dst));
    }

    // Does hipMemcpyPeer work in the WRITE direction (dst on peer)?
    printf("\n-- hipMemcpyPeer direction test (1 float) --\n");
    CHK(hipSetDevice(D0)); float * a0; CHK(hipMalloc(&a0, 4));
    CHK(hipSetDevice(D1)); float * a1; CHK(hipMalloc(&a1, 4));
    float v = 77.0f, r = -1.0f;
    CHK(hipSetDevice(D0)); CHK(hipMemcpy(a0, &v, 4, hipMemcpyHostToDevice));
    CHK(hipSetDevice(D1)); CHK(hipMemset(a1, 0, 4));
    CHK(hipDeviceSynchronize());
    CHK(hipMemcpyPeer(a1, D1, a0, D0, 4));   // WRITE direction: dev0 -> dev1
    CHK(hipDeviceSynchronize());
    CHK(hipSetDevice(D1)); CHK(hipMemcpy(&r, a1, 4, hipMemcpyDeviceToHost));
    printf("  hipMemcpyPeer dev0->dev1 (push): wrote 77.0, got %.3f %s\n",
           r, r == 77.0f ? "OK" : "**FAIL**");

    return 0;
}
