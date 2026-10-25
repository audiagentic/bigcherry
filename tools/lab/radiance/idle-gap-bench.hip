// RAD20 decode diagnosis: the helper's VRAM-bound decode GEMV runs 2.2x slower when its lane thread blocks (interrupt
// wait) instead of spinning, with identical code. This repeats the decode pattern on one GPU - a short VRAM-streaming
// kernel, then the GPU idle while the host waits and sleeps `gap_us` - with the host waiting either by spinning on
// hipEventQuery (as polling) or in hipEventSynchronize on a BlockingSync event (as RAD20). Kernel time on the GPU clock.
// Build: hipcc -O3 --offload-arch=gfx1100 idle-gap-bench.hip -o idle-gap-bench
// Run:   HIP_VISIBLE_DEVICES=<xtx> ./idle-gap-bench [MiB a request=8] [gap_us=60] [requests=3000]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

__global__ void stream_read(const uint4* w, size_t n, unsigned long long* st, float* sink) {
    __shared__ unsigned long long t0;
    if (threadIdx.x == 0) t0 = wall_clock64();
    __syncthreads();
    float acc = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = w[i];
        acc += __uint_as_float(v.x) + __uint_as_float(v.w);
    }
    if (acc == 1234.5f) *sink = acc;
    __syncthreads();
    if (threadIdx.x == 0) { atomicMin(&st[0], t0); atomicMax(&st[1], wall_clock64()); }
}

int main(int argc, char** argv) {
    const size_t mib = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 8;
    const int gap_us = argc > 2 ? std::atoi(argv[2]) : 60;
    const int reqs = argc > 3 ? std::atoi(argv[3]) : 3000;
    CK(hipSetDevice(0));
    int khz = 0; CK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0));
    const size_t sets = 16, n = mib * 1024 * 1024 / sizeof(uint4);   // 16 distinct slices: no request hits L2/MALL warm
    uint4* w; float* sink; unsigned long long* st;
    CK(hipMalloc(&w, n * sizeof(uint4) * sets)); CK(hipMemset(w, 1, n * sizeof(uint4) * sets));
    CK(hipMalloc(&sink, 4)); CK(hipMalloc(&st, 16 * reqs));
    hipStream_t s; CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    hipEvent_t spin_ev, block_ev;
    CK(hipEventCreateWithFlags(&spin_ev, hipEventDisableTiming));
    CK(hipEventCreateWithFlags(&block_ev, hipEventBlockingSync | hipEventDisableTiming));
    for (int mode : {0, 1, 0, 1}) {
        std::vector<unsigned long long> init(2 * reqs);
        for (int i = 0; i < reqs; ++i) { init[2 * i] = ~0ull; init[2 * i + 1] = 0; }
        CK(hipMemcpy(st, init.data(), 16 * reqs, hipMemcpyHostToDevice));
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reqs; ++i) {
            hipLaunchKernelGGL(stream_read, dim3(384), dim3(256), 0, s, w + (i % sets) * n, n, st + 2 * i, sink);
            hipEvent_t e = mode ? block_ev : spin_ev;
            CK(hipEventRecord(e, s));
            if (mode) CK(hipEventSynchronize(e));
            else while (hipEventQuery(e) == hipErrorNotReady) {}
            const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(gap_us);
            if (mode) std::this_thread::sleep_until(until);
            else while (std::chrono::steady_clock::now() < until) {}
        }
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::vector<unsigned long long> v(2 * reqs);
        CK(hipMemcpy(v.data(), st, 16 * reqs, hipMemcpyDeviceToHost));
        std::vector<double> d;
        for (int i = 100; i < reqs; ++i) d.push_back((v[2 * i + 1] - v[2 * i]) * 1000.0 / khz);
        std::sort(d.begin(), d.end());
        std::printf("%zu MiB a request, gap %d us, host %s: kernel p50 %.1f us p90 %.1f us (%.0f GB/s), %.1f us a request wall\n",
                    mib, gap_us, mode ? "BLOCKING " : "spinning ", d[d.size() / 2], d[d.size() * 9 / 10],
                    mib * 1.048576e6 / (d[d.size() / 2] * 1e3), wall * 1e6 / reqs);
    }
    return 0;
}
