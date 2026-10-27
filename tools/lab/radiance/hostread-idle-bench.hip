// RAD20 decode diagnosis: does a GPU kernel that reads host-mapped memory (as the helper lanes read activations and
// routes from the mailbox) slow down when the host CPU is idle? Each request: the host sleeps (like a lane waiting), then
// launches one kernel that reads `kib` KiB of pinned, mapped host memory and is timed on the GPU clock. Run with
// spin=0 (CPU idle between requests) and spin=1 (one host thread busy-loops, as the polling lanes did).
// Build: hipcc -O3 --offload-arch=gfx1100 hostread-idle-bench.hip -o hostread-idle-bench
// Run:   HIP_VISIBLE_DEVICES=<xtx> ./hostread-idle-bench [kib=256] [requests=2000] [sleep_us=200]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <unistd.h>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

__global__ void host_read(const uint4* h, size_t n, unsigned long long* st, float* sink) {
    unsigned long long t0 = wall_clock64();
    float acc = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        acc += __uint_as_float(__builtin_nontemporal_load(&h[i].x));
    if (acc == 1234.5f) *sink = acc;
    __syncthreads();
    if (threadIdx.x == 0) { atomicMin(&st[0], t0); atomicMax(&st[1], wall_clock64()); }
}

int main(int argc, char** argv) {
    const size_t kib = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 256;
    const int reqs = argc > 2 ? std::atoi(argv[2]) : 2000;
    const int sleep_us = argc > 3 ? std::atoi(argv[3]) : 200;
    CK(hipSetDevice(0));
    int khz = 0; CK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0));
    uint4* h; float* sink; unsigned long long* st;
    CK(hipHostMalloc(&h, kib * 1024, hipHostMallocMapped | hipHostMallocPortable));
    std::fill((char*)h, (char*)h + kib * 1024, 1);
    uint4* hd; CK(hipHostGetDevicePointer((void**)&hd, h, 0));
    CK(hipMalloc(&sink, 4)); CK(hipMalloc(&st, 16 * reqs));
    hipStream_t s; CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    for (int spin : {0, 1, 0, 1}) {
        std::atomic<bool> run{spin != 0};
        std::thread spinner([&] { while (run.load(std::memory_order_relaxed)) {} });
        std::vector<unsigned long long> init(2 * reqs);
        for (int i = 0; i < reqs; ++i) { init[2 * i] = ~0ull; init[2 * i + 1] = 0; }
        CK(hipMemcpy(st, init.data(), 16 * reqs, hipMemcpyHostToDevice));
        for (int i = 0; i < reqs; ++i) {
            usleep(sleep_us);
            hipLaunchKernelGGL(host_read, dim3(64), dim3(256), 0, s, hd, kib * 1024 / sizeof(uint4), st + 2 * i, sink);
            CK(hipStreamSynchronize(s));
        }
        run = false; spinner.join();
        std::vector<unsigned long long> v(2 * reqs);
        CK(hipMemcpy(v.data(), st, 16 * reqs, hipMemcpyDeviceToHost));
        std::vector<double> d;
        for (int i = 50; i < reqs; ++i) d.push_back((v[2 * i + 1] - v[2 * i]) * 1000.0 / khz);
        std::sort(d.begin(), d.end());
        std::printf("%zu KiB host read, sleep %d us, CPU spinner %d: kernel p50 %.1f us p90 %.1f us\n", kib, sleep_us, spin,
                    d[d.size() / 2], d[d.size() * 9 / 10]);
    }
    return 0;
}
