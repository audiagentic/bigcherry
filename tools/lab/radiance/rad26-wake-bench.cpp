// RAD26 harness: can a GPU-published request wake a sleeping CPU thread fast enough for the helper's large lane?
// Engine role: one stream on the engine-class GPU publishes request i (system-scope store to a host-registered word),
// then a lab-only GPU kernel waits for the helper's ack and stamps the round trip in GPU clock ticks.
// Helper role: a CPU thread that waits for the request word and acks it, either
//   spin      yield-spinning on the word (the large lane today), or
//   hostfunc  sleeping in FUTEX_WAIT on the word, woken by hipLaunchHostFunc queued after the publish (the callback
//             only does a futex wake; no HIP calls).
// stall mode measures what a host callback costs the engine stream itself: N x (work kernel) against N x (work kernel +
// a no-op host callback).
// Usage: rad26-wake-bench spin|hostfunc|stall [iterations=2000] [gap_us=500] [work_us=50]
#include <hip/hip_runtime.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

__global__ void work(uint64_t ticks) {
    const uint64_t end = wall_clock64() + ticks;
    while (wall_clock64() < end) {}
}
__global__ void publish(uint32_t* req, uint32_t i, uint64_t* t_pub) {
    *t_pub = wall_clock64();
    __hip_atomic_store(req, i, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
/* Lab only: a resident wait for the ack, to time the round trip on the GPU clock. */
__global__ void wait_ack(const uint32_t* ack, uint32_t i, uint64_t* t_ack) {
    while (__hip_atomic_load(ack, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != i) __builtin_amdgcn_s_sleep(1);
    *t_ack = wall_clock64();
}

static void futex_wake_all(uint32_t* p) { (void)::syscall(SYS_futex, p, FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0); }
static void on_published(void* word) { futex_wake_all(static_cast<uint32_t*>(word)); }
static void no_op(void*) {}
static double now_us() { timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static double self_cpu_ms() { timespec t{}; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "";
    if (std::strcmp(mode, "spin") && std::strcmp(mode, "hostfunc") && std::strcmp(mode, "stall")) {
        std::fprintf(stderr, "usage: %s spin|hostfunc|stall [iterations] [gap_us] [work_us]\n", argv[0]);
        return 2;
    }
    const uint32_t n = argc > 2 ? std::atoi(argv[2]) : 2000;
    const uint32_t gap_us = argc > 3 ? std::atoi(argv[3]) : 500;
    const uint32_t work_us = argc > 4 ? std::atoi(argv[4]) : 50;
    CHECK(hipSetDevice(0));
    int khz = 0;
    CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0));
    const uint64_t work_ticks = (uint64_t)work_us * khz / 1000;
    hipStream_t s;
    CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

    if (!std::strcmp(mode, "stall")) {   // engine-stream cost of a host callback
        auto run = [&](bool cb) {
            CHECK(hipStreamSynchronize(s));
            const double t0 = now_us();
            for (uint32_t i = 0; i < n; ++i) {
                work<<<1, 64, 0, s>>>(work_ticks);
                if (cb) CHECK(hipLaunchHostFunc(s, no_op, nullptr));
            }
            CHECK(hipStreamSynchronize(s));
            return now_us() - t0;
        };
        run(false);   // warm-up
        const double base = run(false), with = run(true);
        std::printf("RESULT mode=stall iterations=%u work_us=%u base_us_per_iter=%.2f hostfunc_us_per_iter=%.2f "
                    "stall_us_per_callback=%.2f\n", n, work_us, base / n, with / n, (with - base) / n);
        return 0;
    }

    auto* words = static_cast<uint32_t*>(aligned_alloc(4096, 4096));
    std::memset(words, 0, 4096);
    CHECK(hipHostRegister(words, 4096, hipHostRegisterMapped | hipHostRegisterPortable));
    uint32_t* req = words;
    uint32_t* ack = words + 64;
    uint32_t *d_req, *d_ack;
    CHECK(hipHostGetDevicePointer((void**)&d_req, req, 0));
    CHECK(hipHostGetDevicePointer((void**)&d_ack, ack, 0));
    uint64_t* d_t;
    CHECK(hipMalloc(&d_t, sizeof(uint64_t) * 2 * (n + 1)));
    const bool hostfunc = !std::strcmp(mode, "hostfunc");

    double helper_cpu = 0;
    std::thread helper([&] {
        const double c0 = self_cpu_ms();
        for (uint32_t i = 1; i <= n; ++i) {
            uint32_t seen;
            while ((seen = __atomic_load_n(req, __ATOMIC_ACQUIRE)) < i) {
                if (hostfunc) (void)::syscall(SYS_futex, req, FUTEX_WAIT, seen, nullptr, nullptr, 0);
                else std::this_thread::yield();
            }
            __atomic_store_n(ack, i, __ATOMIC_RELEASE);
        }
        helper_cpu = self_cpu_ms() - c0;
    });
    const double t0 = now_us();
    for (uint32_t i = 1; i <= n; ++i) {
        publish<<<1, 1, 0, s>>>(d_req, i, d_t + 2 * i);
        if (hostfunc) CHECK(hipLaunchHostFunc(s, on_published, req));
        wait_ack<<<1, 1, 0, s>>>(d_ack, i, d_t + 2 * i + 1);
        CHECK(hipStreamSynchronize(s));
        const double due = t0 + (double)i * gap_us;   // decode-like spacing between requests
        while (now_us() < due) std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    const double wall_ms = (now_us() - t0) / 1e3;
    helper.join();
    std::vector<uint64_t> t(2 * (n + 1));
    CHECK(hipMemcpy(t.data(), d_t, t.size() * sizeof(uint64_t), hipMemcpyDeviceToHost));
    std::vector<double> rt;
    for (uint32_t i = 2; i <= n; ++i) rt.push_back((double)(t[2 * i + 1] - t[2 * i]) * 1000.0 / khz);   // us
    std::sort(rt.begin(), rt.end());
    auto q = [&](double f) { return rt[std::min(rt.size() - 1, (size_t)(rt.size() * f))]; };
    std::printf("RESULT mode=%s iterations=%u gap_us=%u rt_p50_us=%.1f rt_p90_us=%.1f rt_p99_us=%.1f rt_max_us=%.1f "
                "helper_cpu_pct=%.1f\n", mode, n, gap_us, q(0.5), q(0.9), q(0.99), rt.back(), 100.0 * helper_cpu / wall_ms);
    (void)hipFree(d_t);
    (void)hipHostUnregister(words);
    std::free(words);
    return 0;
}
