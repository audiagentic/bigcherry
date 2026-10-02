// HIP peer-access and peer-copy bandwidth check between every visible GPU pair.
// Build: hipcc -O2 p2p-check.hip -o p2p-check ; run with HIP_VISIBLE_DEVICES set as needed.
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
    int n = 0; hipGetDeviceCount(&n);
    const size_t bytes = 256u << 20;
    for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) {
        if (a == b) continue;
        int can = 0; hipDeviceCanAccessPeer(&can, a, b);
        void *src = nullptr, *dst = nullptr;
        hipSetDevice(a); hipMalloc(&src, bytes);
        hipError_t pe = can ? hipDeviceEnablePeerAccess(b, 0) : hipSuccess;
        hipSetDevice(b); hipMalloc(&dst, bytes);
        hipSetDevice(a);
        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
        hipMemcpyPeer(dst, b, src, a, bytes);
        hipEventRecord(e0);
        for (int i = 0; i < 10; i++) hipMemcpyPeer(dst, b, src, a, bytes);
        hipEventRecord(e1); hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, e0, e1);
        printf("%d -> %d canAccessPeer=%d enable=%s copy %.1f GB/s\n", a, b, can, hipGetErrorName(pe), 10.0 * bytes / (ms * 1e6));
        hipFree(src); hipSetDevice(b); hipFree(dst);
    }
    return 0;
}
