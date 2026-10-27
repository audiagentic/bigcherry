// RAD25 harness: does a blocking completion event keep a small lane fed as well as the yield loop does?
// One GPU, one stream, the helper small lane's shape without a model: every request is a pre-queued
// hipStreamWaitValue32 on a host-registered request word, a fixed-length "GEMV" kernel and a system-scope done publish.
// An engine thread releases request i at t0 + i*period (decode cadence); the lane thread keeps `depth` requests queued
// and, on each completion, queues the next. The lane observes completion either by yield-spinning on the done word
// (production today) or by hipEventSynchronize on a per-slot hipEventBlockingSync|hipEventDisableTiming event (RAD25).
// A slow observation shows up as starvation: the engine releases a request that is not queued yet.
// blockdev also sets hipDeviceScheduleBlockingSync for the device. Runtime wait knobs (ROC_ACTIVE_WAIT_TIMEOUT,
// HSA_ENABLE_MWAITX, ...) come from the environment; the runner sweeps them.
// Usage: rad25-event-wake yield|block|blockdev [requests=20000] [period_us=150] [work_us=60] [depth=4]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
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
__global__ void publish(uint32_t* done, uint32_t seq) {
    __hip_atomic_store(done, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

static uint64_t now_ns() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}
static double self_cpu_ms() {
    timespec t{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
    if (argc < 2 || (std::strcmp(argv[1], "yield") && std::strcmp(argv[1], "block") && std::strcmp(argv[1], "blockdev"))) {
        std::fprintf(stderr, "usage: %s yield|block|blockdev [requests] [period_us] [work_us] [depth]\n", argv[0]);
        return 2;
    }
    const bool block = std::strcmp(argv[1], "yield") != 0;
    if (!std::strcmp(argv[1], "blockdev")) CHECK(hipSetDeviceFlags(hipDeviceScheduleBlockingSync));
    const uint32_t n = argc > 2 ? std::atoi(argv[2]) : 20000;
    const uint32_t period_us = argc > 3 ? std::atoi(argv[3]) : 150;
    const uint32_t work_us = argc > 4 ? std::atoi(argv[4]) : 60;
    const uint32_t depth = argc > 5 ? std::atoi(argv[5]) : 4;

    CHECK(hipSetDevice(0));
    int clock_khz = 0;
    CHECK(hipDeviceGetAttribute(&clock_khz, hipDeviceAttributeWallClockRate, 0));
    const uint64_t ticks = (uint64_t)work_us * clock_khz / 1000;
    hipStream_t s;
    CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    // The words live in ordinary host memory registered with HIP, like the helper's shared segment.
    auto* words = static_cast<uint32_t*>(aligned_alloc(4096, 4096));
    std::memset(words, 0, 4096);
    CHECK(hipHostRegister(words, 4096, hipHostRegisterMapped | hipHostRegisterPortable));
    uint32_t* req = words;          // engine -> lane: newest released request
    uint32_t* done = words + 64;    // lane GPU -> CPU: newest completed request
    uint32_t *d_req, *d_done;
    CHECK(hipHostGetDevicePointer((void**)&d_req, req, 0));
    CHECK(hipHostGetDevicePointer((void**)&d_done, done, 0));
    std::vector<hipEvent_t> ev(depth);
    for (auto& e : ev) CHECK(hipEventCreateWithFlags(&e, hipEventBlockingSync | hipEventDisableTiming));

    std::atomic<uint32_t> queued{0};
    auto enqueue = [&](uint32_t i) {
        CHECK(hipStreamWaitValue32(s, d_req, i, hipStreamWaitValueGte, 0xffffffffu));
        work<<<1, 64, 0, s>>>(ticks);
        publish<<<1, 1, 0, s>>>(d_done, i);
        if (block) CHECK(hipEventRecord(ev[i % depth], s));
        queued.store(i, std::memory_order_release);
    };
    // Warm-up outside the measurement: one released request through the whole chain.
    __atomic_store_n(req, 1u, __ATOMIC_RELEASE);
    enqueue(1);
    CHECK(hipStreamSynchronize(s));
    const uint32_t base = 1;   // requests base+1 .. base+n are measured

    std::vector<uint64_t> starve_ns;
    std::atomic<bool> lane_ready{false};
    uint64_t t0 = 0, t_end = 0;
    double lane_cpu = 0;
    std::thread lane([&] {
        const double cpu0 = self_cpu_ms();
        uint32_t next = base + 1, seen = base;
        auto fill = [&] { while (next <= base + n && next - seen - 1 < depth) enqueue(next++); };
        fill();
        lane_ready.store(true, std::memory_order_release);
        while (seen < base + n) {
            uint32_t d;
            if (block) {
                CHECK(hipEventSynchronize(ev[(seen + 1) % depth]));
                d = __atomic_load_n(done, __ATOMIC_ACQUIRE);
                if (d < seen + 1) { std::fprintf(stderr, "event fired before done (%u < %u)\n", d, seen + 1); std::exit(3); }
            } else {
                while ((d = __atomic_load_n(done, __ATOMIC_ACQUIRE)) == seen) std::this_thread::yield();
            }
            seen = d;
            fill();
        }
        lane_cpu = self_cpu_ms() - cpu0;
    });
    while (!lane_ready.load(std::memory_order_acquire)) std::this_thread::yield();
    t0 = now_ns();
    for (uint32_t k = 1; k <= n; ++k) {
        const uint32_t i = base + k;
        const uint64_t due = t0 + (uint64_t)k * period_us * 1000;
        timespec ts{(time_t)(due / 1000000000ull), (long)(due % 1000000000ull)};
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
        if (queued.load(std::memory_order_acquire) < i) {   // the lane has not queued this one yet: starved
            const uint64_t a = now_ns();
            while (queued.load(std::memory_order_acquire) < i) {}
            starve_ns.push_back(now_ns() - a);
        }
        __atomic_store_n(req, i, __ATOMIC_RELEASE);
    }
    lane.join();
    t_end = now_ns();
    CHECK(hipStreamSynchronize(s));

    std::sort(starve_ns.begin(), starve_ns.end());
    double starve_total = 0;
    for (auto v : starve_ns) starve_total += v / 1e3;
    const double wall_ms = (t_end - t0) / 1e6;
    std::printf("RESULT mode=%s requests=%u period_us=%u work_us=%u depth=%u wall_ms=%.1f ideal_ms=%.1f "
                "starved=%zu starve_total_us=%.0f starve_p99_us=%.1f starve_max_us=%.1f lane_cpu_ms=%.1f lane_cpu_pct=%.1f\n",
                argv[1], n, period_us, work_us, depth, wall_ms, n * period_us / 1e3,
                starve_ns.size(), starve_total,
                starve_ns.empty() ? 0.0 : starve_ns[std::min(starve_ns.size() - 1, starve_ns.size() * 99 / 100)] / 1e3,
                starve_ns.empty() ? 0.0 : starve_ns.back() / 1e3, lane_cpu, 100.0 * lane_cpu / wall_ms);
    for (auto& e : ev) (void)hipEventDestroy(e);
    (void)hipStreamDestroy(s);
    (void)hipHostUnregister(words);
    std::free(words);
    return 0;
}
