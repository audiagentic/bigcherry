// Per-GPU pinned-host copy bandwidth: H2D alone, D2H alone, and both directions at once on two streams.
// Answers whether a card's link (e.g. the R9700 on PCIe4 x4) overlaps D2H and H2D.
// Build: hipcc -O2 bidir-check.hip -o bidir-check ; run with HIP_VISIBLE_DEVICES set.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
int main() {
    int n = 0; (void) hipGetDeviceCount(&n);
    const size_t bytes = 64u << 20; const int reps = 20;
    void * hsrc; void * hdst;
    (void) hipHostMalloc(&hsrc, bytes, hipHostMallocPortable); (void) hipHostMalloc(&hdst, bytes, hipHostMallocPortable);
    for (int d = 0; d < n; d++) {
        (void) hipSetDevice(d);
        void * a; void * b; hipStream_t s1, s2;
        (void) hipMalloc(&a, bytes); (void) hipMalloc(&b, bytes);
        (void) hipStreamCreateWithFlags(&s1, hipStreamNonBlocking); (void) hipStreamCreateWithFlags(&s2, hipStreamNonBlocking);
        auto bw = [&](bool up, bool down) {
            (void) hipDeviceSynchronize();
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; r++) {
                if (up)   (void) hipMemcpyAsync(a, hsrc, bytes, hipMemcpyHostToDevice, s1);
                if (down) (void) hipMemcpyAsync(hdst, b, bytes, hipMemcpyDeviceToHost, s2);
            }
            (void) hipStreamSynchronize(s1); (void) hipStreamSynchronize(s2);
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            return double(bytes) * reps / s / 1e9;
        };
        bw(true, true);
        const double h2d = bw(true, false), d2h = bw(false, true), both = bw(true, true);
        hipDeviceProp_t p; (void) hipGetDeviceProperties(&p, d);
        printf("dev %d %-28s H2D %.1f GB/s  D2H %.1f GB/s  both-at-once %.1f GB/s per direction (%.1f total)\n",
               d, p.name, h2d, d2h, both, 2 * both);
        (void) hipFree(a); (void) hipFree(b);
    }
    return 0;
}
