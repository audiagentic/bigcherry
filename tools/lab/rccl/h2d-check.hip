// Pinned host <-> device copy bandwidth per visible GPU (reference for p2p-check).
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
    int n = 0; hipGetDeviceCount(&n);
    const size_t bytes = 256u << 20;
    void *h = nullptr; hipHostMalloc(&h, bytes, 0);
    for (int d = 0; d < n; d++) {
        hipSetDevice(d); void *g = nullptr; hipMalloc(&g, bytes);
        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1); float ms;
        hipMemcpy(g, h, bytes, hipMemcpyHostToDevice);
        hipEventRecord(e0); for (int i = 0; i < 10; i++) hipMemcpy(g, h, bytes, hipMemcpyHostToDevice); hipEventRecord(e1); hipEventSynchronize(e1);
        hipEventElapsedTime(&ms, e0, e1); printf("dev %d H2D %.1f GB/s", d, 10.0 * bytes / (ms * 1e6));
        hipEventRecord(e0); for (int i = 0; i < 10; i++) hipMemcpy(h, g, bytes, hipMemcpyDeviceToHost); hipEventRecord(e1); hipEventSynchronize(e1);
        hipEventElapsedTime(&ms, e0, e1); printf("  D2H %.1f GB/s\n", 10.0 * bytes / (ms * 1e6));
        hipFree(g);
    }
    return 0;
}
