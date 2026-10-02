// CPU-root one-shot AllReduce feasibility microbenchmark (Flash-Next decode: 10 KB f32 per call).
// Each GPU kernel stores its slice into pinned, device-mapped host memory, fences system-wide and bumps
// its own 128-B epoch line; one CPU thread busy-polls the epochs, sums the ranks in fixed order (exact
// f32), writes the result and the result epoch; each GPU kernel spins on the result epoch and copies the
// result back. Compares against nothing here: RCCL numbers come from ar-latency.hip on the same host.
// Build: hipcc -O2 --offload-arch=gfx1100,gfx1201 cpu-root-ar.hip -o cpu-root-ar -pthread
#include <hip/hip_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include <immintrin.h>

struct alignas(128) Epoch { volatile unsigned int v; char pad[124]; };

__global__ void empty_kernel() {}

__global__ void produce(const float * __restrict__ src, float * slot, Epoch * ep, unsigned int epoch, int n) {
    const float4 * s4 = (const float4 *) src; float4 * d4 = (float4 *) slot;
    for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
        d4[i] = s4[i];
    }
    // Single block: every store is done after the barrier; publish once.
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        ep->v = epoch;
    }
}

__global__ void consume(float * __restrict__ dst, const float * result, const Epoch * rep, unsigned int epoch, int n) {
    if (threadIdx.x == 0) {
        while (((volatile Epoch *) rep)->v != epoch) { __builtin_amdgcn_s_sleep(1); }
    }
    __syncthreads();
    __threadfence_system();
    const float4 * r4 = (const float4 *) result; float4 * d4 = (float4 *) dst;
    for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
        d4[i] = r4[i];
    }
}

int main(int argc, char ** argv) {
    int nd = 0; hipGetDeviceCount(&nd);
    const int n = argc > 1 ? atoi(argv[1]) : 2560;
    const int iters = 5000;
    const int blocks = 1, threads = 1024;
    float * h_slots; Epoch * h_ep; float * h_res; Epoch * h_rep;
    (void) hipHostMalloc((void **) &h_slots, sizeof(float) * n * nd, hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent);
    (void) hipHostMalloc((void **) &h_ep, sizeof(Epoch) * nd, hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent);
    (void) hipHostMalloc((void **) &h_res, sizeof(float) * n, hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent);
    (void) hipHostMalloc((void **) &h_rep, sizeof(Epoch), hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent);
    memset(h_ep, 0, sizeof(Epoch) * nd); memset((void *) h_rep, 0, sizeof(Epoch));
    std::vector<float *> d_buf(nd); std::vector<hipStream_t> st(nd);
    for (int d = 0; d < nd; d++) {
        (void) hipSetDevice(d); (void) hipStreamCreate(&st[d]);
        (void) hipMalloc(&d_buf[d], sizeof(float) * n);
        std::vector<float> init(n, float(d + 1));
        (void) hipMemcpy(d_buf[d], init.data(), sizeof(float) * n, hipMemcpyHostToDevice);
    }
    std::atomic<bool> stop{false};
    std::atomic<unsigned int> served{0};
    std::thread cpu([&] {
        unsigned int want = 1;
        while (!stop.load(std::memory_order_relaxed)) {
            bool ready = true;
            for (int d = 0; d < nd; d++) { if (h_ep[d].v != want) { ready = false; break; } }
            if (!ready) { _mm_pause(); continue; }
            std::atomic_thread_fence(std::memory_order_acquire);
            for (int i = 0; i < n; i += 8) {
                __m256 acc = _mm256_loadu_ps(h_slots + i);
                for (int d = 1; d < nd; d++) acc = _mm256_add_ps(acc, _mm256_loadu_ps(h_slots + (size_t) d * n + i));
                _mm256_storeu_ps(h_res + i, acc);
            }
            std::atomic_thread_fence(std::memory_order_release);
            h_rep->v = want;
            served.store(want, std::memory_order_relaxed);
            want++;
        }
    });
    float * d_slot_base; Epoch * d_ep; float * d_res; Epoch * d_rep;
    (void) hipHostGetDevicePointer((void **) &d_slot_base, h_slots, 0);
    (void) hipHostGetDevicePointer((void **) &d_ep, h_ep, 0);
    (void) hipHostGetDevicePointer((void **) &d_res, h_res, 0);
    (void) hipHostGetDevicePointer((void **) &d_rep, (void *) h_rep, 0);
    // Floor: the same launch/sync pattern with empty kernels (no host memory traffic, no CPU work).
    {
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; it++) {
            for (int d = 0; d < nd; d++) { (void) hipSetDevice(d); empty_kernel<<<1, 64, 0, st[d]>>>(); empty_kernel<<<1, 64, 0, st[d]>>>(); }
            for (int d = 0; d < nd; d++) { (void) hipSetDevice(d); (void) hipStreamSynchronize(st[d]); }
        }
        printf("ranks=%d floor (2 empty kernels per rank + sync) %.1f us/call\n", nd,
               std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters);
    }
    double best = 1e9;
    for (int rep = 0; rep < 3; rep++) {
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; it++) {
            const unsigned int epoch = rep * iters + it + 1;
            for (int d = 0; d < nd; d++) {
                (void) hipSetDevice(d);
                produce<<<blocks, threads, 0, st[d]>>>(d_buf[d], d_slot_base + (size_t) d * n, d_ep + d, epoch, n);
                consume<<<blocks, threads, 0, st[d]>>>(d_buf[d], d_res, d_rep, epoch, n);
            }
            for (int d = 0; d < nd; d++) { (void) hipSetDevice(d); (void) hipStreamSynchronize(st[d]); }
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        if (us < best) best = us;
        printf("ranks=%d elems=%d bytes=%d  %.1f us/call (rep %d, synchronised each call)\n", nd, n, n * 4, us, rep);
    }
    // values double every call (x -> sum over ranks); just check rank 0 equals rank 1.
    std::vector<float> a(n), b(n);
    (void) hipSetDevice(0); (void) hipMemcpy(a.data(), d_buf[0], sizeof(float) * n, hipMemcpyDeviceToHost);
    (void) hipSetDevice(1); (void) hipMemcpy(b.data(), d_buf[1], sizeof(float) * n, hipMemcpyDeviceToHost);
    printf("ranks agree: %s  best %.1f us\n", memcmp(a.data(), b.data(), sizeof(float) * n) == 0 ? "yes" : "NO", best);
    stop = true; cpu.join();
    return 0;
}
