// Pipelined host-staged AllReduce for LARGE messages (Flash-Next prefill: ~10 MB f32 per call, 3 ranks).
// Chunked: each GPU copies chunk k device->pinned host on its copy engine (hipMemcpyAsync, own stream) and
// records an event; CPU workers wait for chunk k of every rank, sum in fixed rank order (exact f32, AVX2,
// several threads per chunk), then each GPU copies the reduced chunk host->device on a second stream.
// D2H of chunk k+1 overlaps the CPU sum of k and the H2D of k-1. Compares against the RCCL numbers from
// ar-latency.hip (10 MB 3 ranks: ~3.6 ms stream-ordered).
// Build: hipcc -O2 -mavx2 --offload-arch=gfx1100,gfx1201 large-host-ar.hip -o large-host-ar -pthread
#include <hip/hip_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include <immintrin.h>

static void sum_range(float * out, const std::vector<float *> & in, size_t lo, size_t hi) {
    for (size_t i = lo; i < hi; i += 8) {
        __m256 acc = _mm256_loadu_ps(in[0] + i);
        for (size_t r = 1; r < in.size(); r++) acc = _mm256_add_ps(acc, _mm256_loadu_ps(in[r] + i));
        _mm256_storeu_ps(out + i, acc);
    }
}

int main(int argc, char ** argv) {
    int nd = 0; (void) hipGetDeviceCount(&nd);
    const size_t elems = argc > 1 ? strtoull(argv[1], nullptr, 10) : 2621440;       // 10 MB
    const size_t chunk = argc > 2 ? strtoull(argv[2], nullptr, 10) : 262144;         // 1 MB chunks
    const int cpu_threads = argc > 3 ? atoi(argv[3]) : 4;
    const int iters = 30;
    const size_t nchunks = (elems + chunk - 1) / chunk;
    std::vector<float *> dev(nd), slot(nd);
    std::vector<hipStream_t> d2h(nd), h2d(nd);
    std::vector<std::vector<hipEvent_t>> ev(nd, std::vector<hipEvent_t>(nchunks));
    float * result;
    (void) hipHostMalloc((void **) &result, elems * sizeof(float), hipHostMallocPortable);
    for (int d = 0; d < nd; d++) {
        (void) hipSetDevice(d);
        (void) hipMalloc(&dev[d], elems * sizeof(float));
        (void) hipHostMalloc((void **) &slot[d], elems * sizeof(float), hipHostMallocPortable);
        (void) hipStreamCreateWithFlags(&d2h[d], hipStreamNonBlocking);
        (void) hipStreamCreateWithFlags(&h2d[d], hipStreamNonBlocking);
        for (auto & e : ev[d]) (void) hipEventCreateWithFlags(&e, hipEventDisableTiming);
        std::vector<float> init(elems, 1.0f);
        (void) hipMemcpy(dev[d], init.data(), elems * sizeof(float), hipMemcpyHostToDevice);
    }
    auto one = [&]() {
        for (size_t c = 0; c < nchunks; c++) {
            const size_t off = c * chunk, n = std::min(chunk, elems - off);
            for (int d = 0; d < nd; d++) {
                (void) hipSetDevice(d);
                (void) hipMemcpyAsync(slot[d] + off, dev[d] + off, n * sizeof(float), hipMemcpyDeviceToHost, d2h[d]);
                (void) hipEventRecord(ev[d][c], d2h[d]);
            }
        }
        for (size_t c = 0; c < nchunks; c++) {
            const size_t off = c * chunk, n = std::min(chunk, elems - off);
            for (int d = 0; d < nd; d++) (void) hipEventSynchronize(ev[d][c]);
            std::vector<float *> in(nd);
            for (int d = 0; d < nd; d++) in[d] = slot[d] + off;
            std::vector<std::thread> th;
            const size_t per = ((n / cpu_threads) + 7) & ~size_t(7);
            for (int t = 0; t < cpu_threads; t++) {
                const size_t lo = t * per, hi = std::min(n, lo + per);
                if (lo >= hi) break;
                std::vector<float *> sub(nd);
                for (int d = 0; d < nd; d++) sub[d] = in[d];
                th.emplace_back([=] { sum_range(result + off, sub, lo, hi); });
            }
            for (auto & t : th) t.join();
            for (int d = 0; d < nd; d++) {
                (void) hipSetDevice(d);
                (void) hipMemcpyAsync(dev[d] + off, result + off, n * sizeof(float), hipMemcpyHostToDevice, h2d[d]);
            }
        }
        for (int d = 0; d < nd; d++) { (void) hipSetDevice(d); (void) hipStreamSynchronize(h2d[d]); }
    };
    // Pool variant: T threads, each owns chunks c = t, t+T, ...; per chunk it waits for every rank's D2H
    // event, sums, and issues that chunk's H2D on its own per-(thread, rank) stream, so the sum and the
    // return copy of chunk k overlap the D2H of later chunks with no per-chunk thread creation.
    std::vector<std::vector<hipStream_t>> h2d_t(cpu_threads, std::vector<hipStream_t>(nd));
    for (int t = 0; t < cpu_threads; t++) for (int d = 0; d < nd; d++) {
        (void) hipSetDevice(d); (void) hipStreamCreateWithFlags(&h2d_t[t][d], hipStreamNonBlocking);
    }
    auto pool = [&]() {
        for (size_t c = 0; c < nchunks; c++) {
            const size_t off = c * chunk, n = std::min(chunk, elems - off);
            for (int d = 0; d < nd; d++) {
                (void) hipSetDevice(d);
                (void) hipMemcpyAsync(slot[d] + off, dev[d] + off, n * sizeof(float), hipMemcpyDeviceToHost, d2h[d]);
                (void) hipEventRecord(ev[d][c], d2h[d]);
            }
        }
        std::vector<std::thread> th;
        for (int t = 0; t < cpu_threads; t++) {
            th.emplace_back([&, t] {
                for (size_t c = t; c < nchunks; c += cpu_threads) {
                    const size_t off = c * chunk, n = std::min(chunk, elems - off);
                    for (int d = 0; d < nd; d++) (void) hipEventSynchronize(ev[d][c]);
                    std::vector<float *> in(nd);
                    for (int d = 0; d < nd; d++) in[d] = slot[d] + off;
                    sum_range(result + off, in, 0, n);
                    for (int d = 0; d < nd; d++) {
                        (void) hipSetDevice(d);
                        (void) hipMemcpyAsync(dev[d] + off, result + off, n * sizeof(float), hipMemcpyHostToDevice, h2d_t[t][d]);
                    }
                }
                for (int d = 0; d < nd; d++) { (void) hipSetDevice(d); (void) hipStreamSynchronize(h2d_t[t][d]); }
            });
        }
        for (auto & x : th) x.join();
    };
    // memop variant (integration shape): the MAIN thread enqueues everything without blocking:
    //   per rank, per chunk: D2H copy -> hipStreamWriteValue32(arrived[r][c] = gen)
    //   per rank, per chunk: hipStreamWaitValue32(reduced[c] >= gen) -> H2D copy   (second stream)
    // a CPU worker polls arrived flags, sums each chunk, publishes reduced[c] = gen.
    const bool use_memop = argc > 4 && strcmp(argv[4], "memop") == 0;
    if (use_memop) {
        volatile uint32_t * arrived; volatile uint32_t * reduced;
        (void) hipHostMalloc((void **) &arrived, sizeof(uint32_t) * nd * nchunks, hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent);
        (void) hipHostMalloc((void **) &reduced, sizeof(uint32_t) * nchunks, hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent);
        memset((void *) arrived, 0, sizeof(uint32_t) * nd * nchunks); memset((void *) reduced, 0, sizeof(uint32_t) * nchunks);
        std::atomic<uint32_t> want_gen{0}; std::atomic<bool> stop{false};
        std::thread worker([&] {
            uint32_t gen = 1;
            while (!stop.load()) {
                if (want_gen.load() < gen) { _mm_pause(); continue; }
                for (size_t c = 0; c < nchunks; c++) {
                    for (int d = 0; d < nd; d++) { while (arrived[d * nchunks + c] < gen) { _mm_pause(); } }
                    std::atomic_thread_fence(std::memory_order_acquire);
                    const size_t off = c * chunk, n = std::min(chunk, elems - off);
                    std::vector<float *> in(nd);
                    for (int d = 0; d < nd; d++) in[d] = slot[d] + off;
                    sum_range(result + off, in, 0, n);
                    std::atomic_thread_fence(std::memory_order_release);
                    reduced[c] = gen;
                }
                gen++;
            }
        });
        uint32_t gen = 0;
        auto memop_call = [&]() {
            gen++;
            for (int d = 0; d < nd; d++) {
                (void) hipSetDevice(d);
                for (size_t c = 0; c < nchunks; c++) {
                    const size_t off = c * chunk, n = std::min(chunk, elems - off);
                    (void) hipMemcpyAsync(slot[d] + off, dev[d] + off, n * sizeof(float), hipMemcpyDeviceToHost, d2h[d]);
                    (void) hipStreamWriteValue32(d2h[d], (void *) &arrived[d * nchunks + c], gen, 0);
                }
                for (size_t c = 0; c < nchunks; c++) {
                    const size_t off = c * chunk, n = std::min(chunk, elems - off);
                    (void) hipStreamWaitValue32(h2d[d], (void *) &reduced[c], gen, hipStreamWaitValueGte, 0xffffffff);
                    (void) hipMemcpyAsync(dev[d] + off, result + off, n * sizeof(float), hipMemcpyHostToDevice, h2d[d]);
                }
            }
            want_gen.store(gen);
            for (int d = 0; d < nd; d++) { (void) hipSetDevice(d); (void) hipStreamSynchronize(h2d[d]); }
        };
        memop_call();  // warmup
        for (int d = 0; d < nd; d++) {
            std::vector<float> init(elems, 1.0f);
            (void) hipSetDevice(d); (void) hipMemcpy(dev[d], init.data(), elems * sizeof(float), hipMemcpyHostToDevice);
        }
        auto m0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; it++) memop_call();
        const double mus = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - m0).count() / iters;
        stop = true; worker.join();
        std::vector<float> a(elems), b(elems);
        (void) hipSetDevice(0); (void) hipMemcpy(a.data(), dev[0], elems * sizeof(float), hipMemcpyDeviceToHost);
        (void) hipSetDevice(nd - 1); (void) hipMemcpy(b.data(), dev[nd - 1], elems * sizeof(float), hipMemcpyDeviceToHost);
        printf("MEMOP ranks=%d bytes=%zu chunk_bytes=%zu  %.1f us/call  %.2f GB/s algbw  ranks agree: %s\n",
               nd, elems * 4, chunk * 4, mus, elems * 4 / mus / 1e3, memcmp(a.data(), b.data(), elems * 4) == 0 ? "yes" : "NO");
        return 0;
    }
    const bool use_pool = argc > 4 && strcmp(argv[4], "pool") == 0;
    if (use_pool) {
        pool();  // warmup
        for (int d = 0; d < nd; d++) {
            std::vector<float> init(elems, 1.0f);
            (void) hipSetDevice(d); (void) hipMemcpy(dev[d], init.data(), elems * sizeof(float), hipMemcpyHostToDevice);
        }
        auto p0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; it++) pool();
        const double pus = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - p0).count() / iters;
        std::vector<float> a(elems), b(elems);
        (void) hipSetDevice(0); (void) hipMemcpy(a.data(), dev[0], elems * sizeof(float), hipMemcpyDeviceToHost);
        (void) hipSetDevice(nd - 1); (void) hipMemcpy(b.data(), dev[nd - 1], elems * sizeof(float), hipMemcpyDeviceToHost);
        printf("POOL ranks=%d bytes=%zu chunk_bytes=%zu cpu_threads=%d  %.1f us/call  %.2f GB/s algbw  ranks agree: %s\n",
               nd, elems * 4, chunk * 4, cpu_threads, pus, elems * 4 / pus / 1e3, memcmp(a.data(), b.data(), elems * 4) == 0 ? "yes" : "NO");
        return 0;
    }
    one();  // warmup
    for (int d = 0; d < nd; d++) {  // reset to 1.0 so values stay finite (3^k grows fast)
        std::vector<float> init(elems, 1.0f);
        (void) hipSetDevice(d); (void) hipMemcpy(dev[d], init.data(), elems * sizeof(float), hipMemcpyHostToDevice);
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; it++) one();
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    std::vector<float> a(elems), b(elems);
    (void) hipSetDevice(0); (void) hipMemcpy(a.data(), dev[0], elems * sizeof(float), hipMemcpyDeviceToHost);
    (void) hipSetDevice(nd - 1); (void) hipMemcpy(b.data(), dev[nd - 1], elems * sizeof(float), hipMemcpyDeviceToHost);
    printf("ranks=%d bytes=%zu chunk_bytes=%zu cpu_threads=%d  %.1f us/call  %.2f GB/s algbw  ranks agree: %s\n",
           nd, elems * 4, chunk * 4, cpu_threads, us, elems * 4 / us / 1e3, memcmp(a.data(), b.data(), elems * 4) == 0 ? "yes" : "NO");
    return 0;
}
