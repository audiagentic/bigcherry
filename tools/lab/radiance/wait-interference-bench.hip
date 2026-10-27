// RAD20: does a wait armed on a second, idle stream slow kernels on the active stream? In the live helper, one idle
// lane stream holding an unsatisfied hipStreamWaitValue32 made the active lane's decode GEMV 2.2x slower (64 -> 134 us)
// with no overlapping kernel in the trace. Stream A runs a weight-streaming kernel back to back, timed on the GPU clock;
// stream B is idle, or holds an armed hipStreamWaitValue32 on a host-mapped word, or holds our own one-wave wait kernel
// (s_sleep loop on the same word). The word is released at the end so every stream drains.
// Build: hipcc -O3 --offload-arch=gfx1100 wait-interference-bench.hip -o wait-interference-bench
// Run:   HIP_VISIBLE_DEVICES=<xtx> GPU_MAX_HW_QUEUES=8 ./wait-interference-bench [MiB a kernel=16] [kernels=2000] [gated_us=0]
// gated_us > 0: like a live lane, stream A pre-queues a hipStreamWaitValue32 before every kernel (4 ahead) and a host
// thread releases one request every gated_us.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
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

/* One wave waits for the word to reach `want`, sleeping between system-scope loads. */
__global__ void sleep_wait(const uint32_t* word, uint32_t want) {
    if (threadIdx.x) return;
    while (__hip_atomic_load(word, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < want) __builtin_amdgcn_s_sleep(127);
}

int main(int argc, char** argv) {
    const size_t mib = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 16;
    const int kernels = argc > 2 ? std::atoi(argv[2]) : 2000;
    const int gated_us = argc > 3 ? std::atoi(argv[3]) : 0;
    CK(hipSetDevice(0));
    int khz = 0; CK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0));
    const size_t sets = 16, n = mib * 1024 * 1024 / sizeof(uint4);
    uint4* w; float* sink; unsigned long long* st;
    CK(hipMalloc(&w, n * sizeof(uint4) * sets)); CK(hipMemset(w, 1, n * sizeof(uint4) * sets));
    CK(hipMalloc(&sink, 4)); CK(hipMalloc(&st, 16 * kernels));
    uint32_t* word; uint32_t* word_dev;
    CK(hipHostMalloc(&word, 64, hipHostMallocMapped | hipHostMallocPortable));
    CK(hipHostGetDevicePointer((void**)&word_dev, word, 0));
    uint32_t* req = word + 8;            /* stream A's request word (gated mode), another cache line */
    uint32_t* req_dev = word_dev + 8;
    hipStream_t a, b;
    CK(hipStreamCreateWithFlags(&a, hipStreamNonBlocking));
    CK(hipStreamCreateWithFlags(&b, hipStreamNonBlocking));
    const char* names[] = {"stream B idle", "stream B: armed hipStreamWaitValue32", "stream B: armed s_sleep wait kernel",
                           "stream B: 4 armed hipStreamWaitValue32", "stream B: 4 armed s_sleep wait kernels"};
    /* argv[4]: the modes to run, in order (digits 0-4); the slowdown has been seen to persist once triggered. */
    const char* order = argc > 4 ? argv[4] : "01230123";
    for (const char* o = order; *o; ++o) {
            const int mode = *o - '0';
            if (mode < 0 || mode > 4) continue;
            *word = 0;
            if (mode == 1) CK(hipStreamWaitValue32(b, word_dev, 1, hipStreamWaitValueGte, 0xffffffffu));
            if (mode == 2) hipLaunchKernelGGL(sleep_wait, dim3(1), dim3(64), 0, b, word_dev, 1u);
            if (mode == 3) for (uint32_t k = 1; k <= 4; ++k) CK(hipStreamWaitValue32(b, word_dev, k, hipStreamWaitValueGte, 0xffffffffu));
            if (mode == 4) for (uint32_t k = 1; k <= 4; ++k) hipLaunchKernelGGL(sleep_wait, dim3(1), dim3(64), 0, b, word_dev, k);
            std::vector<unsigned long long> init(2 * kernels);
            for (int i = 0; i < kernels; ++i) { init[2 * i] = ~0ull; init[2 * i + 1] = 0; }
            CK(hipMemcpyAsync(st, init.data(), 16 * kernels, hipMemcpyHostToDevice, a));
            __atomic_store_n(req, 0u, __ATOMIC_RELEASE);
            std::thread releaser;
            if (gated_us > 0)
                releaser = std::thread([&] {   /* the engine: one request every gated_us */
                    for (int i = 1; i <= kernels; ++i) {
                        std::this_thread::sleep_for(std::chrono::microseconds(gated_us));
                        __atomic_store_n(req, (uint32_t)i, __ATOMIC_RELEASE);
                    }
                });
            for (int i = 0; i < kernels; ++i) {
                if (gated_us > 0) CK(hipStreamWaitValue32(a, req_dev, (uint32_t)(i + 1), hipStreamWaitValueGte, 0xffffffffu));
                hipLaunchKernelGGL(stream_read, dim3(192), dim3(256), 0, a, w + (i % sets) * n, n, st + 2 * i, sink);
            }
            CK(hipStreamSynchronize(a));
            if (releaser.joinable()) releaser.join();
            __atomic_store_n(word, 8u, __ATOMIC_RELEASE);   // release stream B's wait
            CK(hipStreamSynchronize(b));
            std::vector<unsigned long long> v(2 * kernels);
            CK(hipMemcpy(v.data(), st, 16 * kernels, hipMemcpyDeviceToHost));
            std::vector<double> d;
            for (int i = 100; i < kernels; ++i) d.push_back((v[2 * i + 1] - v[2 * i]) * 1000.0 / khz);
            std::sort(d.begin(), d.end());
            std::printf("%-44s kernel p50 %7.1f us p90 %7.1f us\n", names[mode], d[d.size() / 2], d[d.size() * 9 / 10]);
        }
    return 0;
}
