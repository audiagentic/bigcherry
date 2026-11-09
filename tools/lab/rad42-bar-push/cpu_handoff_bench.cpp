// RAD50: host-only lower-bound latency probe. THIS DOES NOT USE ROCm/GPU.
// It measures a CPU producer publishing a generation for one CPU consumer.
// A GPU cannot call std::condition_variable::notify_one or futex_wake.
// This does NOT model GPU -> CPU notification and is not a real serving trace.
// Avoid running even this CPU-heavy probe while another GPU benchmark lease
// holder is measuring; build only until exclusive access is scheduled.
//
// Build: g++ -O2 -std=c++17 -pthread -Wall -Wextra cpu_handoff_bench.cpp -o cpu_handoff_bench
// Run:   ./cpu_handoff_bench --mode spin|yield|sleep1|condvar --samples 3000
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <thread>
#include <vector>

namespace {
uint64_t now_ns(clockid_t id) {
    timespec ts{};
    if (clock_gettime(id, &ts)) std::abort();
    return uint64_t(ts.tv_sec)*1000000000ULL + uint64_t(ts.tv_nsec);
}
double percentile(std::vector<double> values, double pct) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const size_t idx = static_cast<size_t>((values.size()-1)*pct);
    return values[idx];
}
enum class Mode { Spin, Yield, Sleep1, Condvar };
} // namespace

int main(int argc, char** argv) {
    if (argc != 5 || std::strcmp(argv[1], "--mode") ||
        std::strcmp(argv[3], "--samples")) {
        std::fprintf(stderr, "usage: cpu_handoff_bench --mode spin|yield|sleep1|condvar --samples 100..100000\n");
        return 2;
    }
    Mode mode;
    if (!std::strcmp(argv[2], "spin")) mode = Mode::Spin;
    else if (!std::strcmp(argv[2], "yield")) mode = Mode::Yield;
    else if (!std::strcmp(argv[2], "sleep1")) mode = Mode::Sleep1;
    else if (!std::strcmp(argv[2], "condvar")) mode = Mode::Condvar;
    else return 2;
    char* end=nullptr;
    long samples = std::strtol(argv[4], &end, 10);
    if (!argv[4][0] || *end || samples<100 || samples>100000) return 2;
    const int warmup = 100;
    const uint64_t total = static_cast<uint64_t>(samples + warmup);
    std::atomic<uint64_t> published{0}, consumed{0};
    std::vector<uint64_t> sent(total+1, 0);
    std::vector<double> latency(total+1, 0);
    std::mutex lock;
    std::condition_variable cv;
    uint64_t cpu_thread_ns=0;

    std::thread worker([&] {
        const uint64_t begin_cpu = now_ns(CLOCK_THREAD_CPUTIME_ID);
        for (uint64_t next=1; next<=total; ++next) {
            while (published.load(std::memory_order_acquire) < next) {
                switch (mode) {
                    case Mode::Spin: break;  // busy polling, impractical at idle
                    case Mode::Yield: std::this_thread::yield(); break;
                    case Mode::Sleep1: std::this_thread::sleep_for(
                        std::chrono::microseconds(1)); break;
                    case Mode::Condvar: {
                        std::unique_lock<std::mutex> guard(lock);
                        cv.wait(guard, [&] {
                            return published.load(std::memory_order_acquire) >= next;
                        });
                        break;
                    }
                }
            }
            const uint64_t seen = now_ns(CLOCK_MONOTONIC_RAW);
            latency[next] = static_cast<double>(seen - sent[next]) / 1000.0;
            consumed.store(next, std::memory_order_release);
        }
        cpu_thread_ns = now_ns(CLOCK_THREAD_CPUTIME_ID) - begin_cpu;
    });

    for (uint64_t seq=1; seq<=total; ++seq) {
        sent[seq] = now_ns(CLOCK_MONOTONIC_RAW);
        published.store(seq, std::memory_order_release);
        if (mode == Mode::Condvar) cv.notify_one(); // CPU-only, NOT GPU possible
        while (consumed.load(std::memory_order_acquire) != seq)
            std::this_thread::yield();
    }
    worker.join();
    std::vector<double> measured(latency.begin() + warmup + 1, latency.end());
    std::printf("host-only lower-bound mode=%s samples=%ld p50_us=%.3f p95_us=%.3f "
                "p99_us=%.3f max_us=%.3f cpu_worker_us_per_sample=%.3f\n",
                argv[2], samples,
                percentile(measured, .50), percentile(measured, .95),
                percentile(measured, .99), *std::max_element(measured.begin(), measured.end()),
                static_cast<double>(cpu_thread_ns) / 1000.0 / total);
    std::puts("NOT GPU->CPU latency. Do not use as direct RAD50 go/no-go evidence.");
    return 0; // Required when this entrypoint is linked into rad50-tester.
}
