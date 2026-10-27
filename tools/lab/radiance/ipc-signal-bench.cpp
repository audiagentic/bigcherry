// RAD20 experiment B: can a helper process sleep in the driver until another process's GPU work completes?
//
// A producer process (standing in for the radiance engine, on the R9700) creates an IPC-capable HSA signal and, for each
// notification, submits a barrier-AND packet to its own HSA queue with that signal as the completion signal: the packet
// processor decrements it when the packet completes, as an engine's recorded publish could. A consumer process (standing
// in for a helper, on an XTX) attaches the signal and waits with HSA_WAIT_STATE_BLOCKED for the value to drop below the
// last value it saw. Measures wake latency (host stamp before the doorbell -> consumer wake), lost or merged wakes, and
// the consumer's CPU time while waiting.
//
// Build: hipcc -O2 ipc-signal-bench.cpp -o ipc-signal-bench -lhsa-runtime64
// Run:   ROCR_VISIBLE_DEVICES=<xtx> ./ipc-signal-bench consumer [count=5000] &
//        ROCR_VISIBLE_DEVICES=<r9700> ./ipc-signal-bench producer [count=5000] [interval_us=150] [burst=1]
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/amd_hsa_signal.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/mman.h>
#include <linux/futex.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define CK(x) do { hsa_status_t s_ = (x); if (s_ != HSA_STATUS_SUCCESS) { const char* m_ = nullptr; hsa_status_string(s_, &m_); \
    std::fprintf(stderr, "%s: %s\n", #x, m_ ? m_ : "?"); std::exit(1); } } while (0)

namespace {
/* Whether the runtime gave this signal a driver event (event mailbox and id): without one no wait can block on an interrupt. */
void describe(const char* who, hsa_signal_t s) {
    const auto* a = reinterpret_cast<const amd_signal_t*>(s.handle);
    std::fprintf(stderr, "%s: signal kind %lld, event_mailbox_ptr %#llx, event_id %u\n", who, (long long)a->kind,
                 (unsigned long long)a->event_mailbox_ptr, a->event_id);
}
constexpr const char* kShm = "/rad20-ipc-signal-bench";
constexpr int64_t kStart = 1ll << 40;
constexpr int kMax = 100000;

struct Shared {
    hsa_amd_ipc_signal_t handle;
    volatile int ready;       // producer: the handle is valid
    volatile int attached;    // consumer: attached and waiting
    volatile int done;        // producer: finished submitting
    int relay;                // 1: the producer relays (interrupt signal in its own process -> futex on `count`)
    uint32_t count;           // relay mode: completions seen so far, futex-woken by the producer's relay thread
    int64_t t_send[kMax];     // CLOCK_MONOTONIC ns, stamped just before each doorbell
};

int64_t now_ns() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (int64_t)t.tv_sec * 1000000000ll + t.tv_nsec; }

hsa_status_t pick_gpu(hsa_agent_t a, void* out) {
    hsa_device_type_t t;
    hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
    if (t == HSA_DEVICE_TYPE_GPU) { *static_cast<hsa_agent_t*>(out) = a; return HSA_STATUS_INFO_BREAK; }
    return HSA_STATUS_SUCCESS;
}

Shared* map_shared(bool create) {
    const int fd = shm_open(kShm, create ? (O_CREAT | O_RDWR) : O_RDWR, 0600);
    if (fd < 0) { std::perror("shm_open"); std::exit(1); }
    if (create && ftruncate(fd, sizeof(Shared)) != 0) { std::perror("ftruncate"); std::exit(1); }
    void* p = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
    return static_cast<Shared*>(p);
}

long futex(uint32_t* p, int op, uint32_t v, const timespec* t) { return syscall(SYS_futex, p, op, v, t, nullptr, 0); }

double thread_cpu_ms() { rusage r{}; getrusage(RUSAGE_THREAD, &r);
    return (r.ru_utime.tv_sec + r.ru_stime.tv_sec) * 1e3 + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1e3; }

int producer(int count, int interval_us, int burst, bool relay) {
    shm_unlink(kShm);
    Shared* sh = map_shared(true);
    std::memset(sh, 0, sizeof(Shared));
    CK(hsa_init());
    hsa_agent_t gpu{};
    hsa_iterate_agents(pick_gpu, &gpu);
    char name[64] = {};
    hsa_agent_get_info(gpu, HSA_AGENT_INFO_NAME, name);
    hsa_signal_t sig;
    sh->relay = relay ? 1 : 0;
    if (relay) {
        CK(hsa_signal_create(kStart, 0, nullptr, &sig));   // an ordinary signal: interrupt capable in this process
    } else {
        CK(hsa_amd_signal_create(kStart, 0, nullptr, HSA_AMD_SIGNAL_IPC, &sig));
        CK(hsa_amd_ipc_signal_create(sig, &sh->handle));
    }
    describe("producer", sig);
    // Relay mode: one thread here sleeps on the interrupt signal and futex-wakes the consumer with the completion count.
    double relay_cpu = 0;
    std::thread relay_thread;
    if (relay)
        relay_thread = std::thread([&] {
            int64_t last = kStart;
            const double c0 = thread_cpu_ms();
            while (last > kStart - count) {
                const hsa_signal_value_t v = hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_LT, last, 1000000000ull,
                                                                       HSA_WAIT_STATE_BLOCKED);
                if (v >= last) continue;
                last = v;
                __atomic_store_n(&sh->count, (uint32_t)(kStart - v), __ATOMIC_RELEASE);
                futex(&sh->count, FUTEX_WAKE, INT32_MAX, nullptr);
            }
            relay_cpu = thread_cpu_ms() - c0;
        });
    hsa_queue_t* q = nullptr;
    CK(hsa_queue_create(gpu, 256, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &q));
    __atomic_store_n(&sh->ready, 1, __ATOMIC_RELEASE);
    std::fprintf(stderr, "producer on %s: waiting for the consumer\n", name);
    while (!__atomic_load_n(&sh->attached, __ATOMIC_ACQUIRE)) usleep(1000);
    usleep(200000);
    for (int i = 0; i < count; i += burst) {
        for (int b = 0; b < burst && i + b < count; ++b) {
            const uint64_t idx = hsa_queue_add_write_index_relaxed(q, 1);
            while (idx - hsa_queue_load_read_index_scacquire(q) >= q->size) {}
            auto* p = static_cast<hsa_barrier_and_packet_t*>(q->base_address) + (idx % q->size);
            std::memset(reinterpret_cast<uint8_t*>(p) + 4, 0, sizeof(*p) - 4);
            p->completion_signal = sig;
            const uint16_t header = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                                    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                                    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
            sh->t_send[i + b] = now_ns();
            __atomic_store_n(reinterpret_cast<uint16_t*>(p), header, __ATOMIC_RELEASE);
            hsa_signal_store_screlease(q->doorbell_signal, (hsa_signal_value_t)idx);
        }
        const int64_t until = now_ns() + (int64_t)interval_us * 1000;
        timespec ts{(time_t)(until / 1000000000ll), (long)(until % 1000000000ll)};
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
    }
    while (hsa_signal_load_scacquire(sig) > kStart - count) usleep(1000);
    __atomic_store_n(&sh->done, 1, __ATOMIC_RELEASE);
    if (relay_thread.joinable()) relay_thread.join();
    if (relay) std::fprintf(stderr, "producer relay thread CPU %.1f ms\n", relay_cpu);
    usleep(500000);   // let the consumer finish before the signal goes away
    hsa_queue_destroy(q);
    hsa_signal_destroy(sig);
    hsa_shut_down();
    shm_unlink(kShm);
    std::fprintf(stderr, "producer: %d notifications sent\n", count);
    return 0;
}

int consumer(int count) {
    Shared* sh = nullptr;
    for (int i = 0; i < 600 && !sh; ++i) {
        const int fd = shm_open(kShm, O_RDWR, 0);
        if (fd >= 0) { close(fd); sh = map_shared(false); } else usleep(100000);
    }
    if (!sh) { std::fprintf(stderr, "consumer: no producer\n"); return 1; }
    while (!__atomic_load_n(&sh->ready, __ATOMIC_ACQUIRE)) usleep(1000);
    if (sh->relay) {   // no HSA on this side: sleep on the shared count word
        __atomic_store_n(&sh->attached, 1, __ATOMIC_RELEASE);
        std::fprintf(stderr, "consumer: relay mode, futex wait\n");
        rusage r0{}, r1{};
        getrusage(RUSAGE_SELF, &r0);
        const int64_t w0 = now_ns();
        std::vector<double> lat;
        uint32_t seen = 0;
        int wakes = 0, merged = 0;
        while ((int)seen < count) {
            const uint32_t c = __atomic_load_n(&sh->count, __ATOMIC_ACQUIRE);
            if (c == seen) {
                timespec t{1, 0};
                futex(&sh->count, FUTEX_WAIT, seen, &t);
                if (__atomic_load_n(&sh->done, __ATOMIC_ACQUIRE) && __atomic_load_n(&sh->count, __ATOMIC_ACQUIRE) == seen) break;
                continue;
            }
            const int64_t t = now_ns();
            ++wakes;
            merged += (int)(c - seen - 1);
            for (uint32_t k = seen; k < c && (int)k < count; ++k) lat.push_back((t - sh->t_send[k]) / 1000.0);
            seen = c;
        }
        const int64_t w1 = now_ns();
        getrusage(RUSAGE_SELF, &r1);
        auto us = [](const timeval& a, const timeval& b) { return (b.tv_sec - a.tv_sec) * 1e6 + (b.tv_usec - a.tv_usec); };
        const double cpu_us = us(r0.ru_utime, r1.ru_utime) + us(r0.ru_stime, r1.ru_stime);
        std::sort(lat.begin(), lat.end());
        auto pct = [&](double p) { return lat.empty() ? 0.0 : lat[(size_t)(p * (lat.size() - 1))]; };
        std::printf("consumer (relay): %zu notifications, %d wakes (%d merged)\n", lat.size(), wakes, merged);
        std::printf("  end-to-end latency us: p50 %.1f p90 %.1f p99 %.1f p99.9 %.1f max %.1f\n", pct(.5), pct(.9), pct(.99),
                    pct(.999), lat.empty() ? 0.0 : lat.back());
        std::printf("  consumer CPU %.1f ms over %.1f ms wall (%.2f%% of a core)\n", cpu_us / 1000.0, (w1 - w0) / 1e6,
                    100.0 * cpu_us * 1000.0 / (double)(w1 - w0));
        return 0;
    }
    CK(hsa_init());
    hsa_agent_t gpu{};
    hsa_iterate_agents(pick_gpu, &gpu);
    char name[64] = {};
    hsa_agent_get_info(gpu, HSA_AGENT_INFO_NAME, name);
    hsa_signal_t sig;
    CK(hsa_amd_ipc_signal_attach(&sh->handle, &sig));
    describe("consumer", sig);
    __atomic_store_n(&sh->attached, 1, __ATOMIC_RELEASE);
    std::fprintf(stderr, "consumer on %s: attached, waiting blocked\n", name);

    rusage r0{}, r1{};
    getrusage(RUSAGE_SELF, &r0);
    const int64_t w0 = now_ns();
    std::vector<double> lat;
    lat.reserve(count);
    int64_t last = kStart;
    int wakes = 0, merged = 0, timeouts = 0;
    while ((int)lat.size() < count) {
        // Timeout hint ~1 s (system timestamp ticks): bounds how late a dead producer is noticed.
        const hsa_signal_value_t v = hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_LT, last, 1000000000ull,
                                                               HSA_WAIT_STATE_BLOCKED);
        const int64_t t = now_ns();
        if (v >= last) {
            ++timeouts;
            if (__atomic_load_n(&sh->done, __ATOMIC_ACQUIRE)) break;
            continue;
        }
        ++wakes;
        const int64_t n = last - v;
        if (n > 1) merged += (int)(n - 1);
        for (int64_t k = kStart - last; k < kStart - v && k < count; ++k) lat.push_back((t - sh->t_send[k]) / 1000.0);
        last = v;
    }
    const int64_t w1 = now_ns();
    getrusage(RUSAGE_SELF, &r1);
    auto us = [](const timeval& a, const timeval& b) { return (b.tv_sec - a.tv_sec) * 1e6 + (b.tv_usec - a.tv_usec); };
    const double cpu_us = us(r0.ru_utime, r1.ru_utime) + us(r0.ru_stime, r1.ru_stime);
    std::sort(lat.begin(), lat.end());
    auto pct = [&](double p) { return lat.empty() ? 0.0 : lat[(size_t)(p * (lat.size() - 1))]; };
    std::printf("consumer on %s: %zu notifications, %d wakes (%d merged into an earlier wake), %d timeouts\n", name,
                lat.size(), wakes, merged, timeouts);
    std::printf("  wake latency us: p50 %.1f p90 %.1f p99 %.1f p99.9 %.1f max %.1f\n", pct(.5), pct(.9), pct(.99),
                pct(.999), lat.empty() ? 0.0 : lat.back());
    std::printf("  consumer CPU %.1f ms over %.1f ms wall (%.2f%% of a core)\n", cpu_us / 1000.0, (w1 - w0) / 1e6,
                100.0 * cpu_us * 1000.0 / (double)(w1 - w0));
    hsa_signal_destroy(sig);
    hsa_shut_down();
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s producer|consumer [count] [interval_us] [burst]\n", argv[0]); return 2; }
    const int count = std::min(argc > 2 ? std::atoi(argv[2]) : 5000, kMax);
    if (std::strcmp(argv[1], "producer") == 0)
        return producer(count, argc > 3 ? std::atoi(argv[3]) : 150, std::max(1, argc > 4 ? std::atoi(argv[4]) : 1),
                        argc > 5 && std::strcmp(argv[5], "relay") == 0);
    if (std::strcmp(argv[1], "consumer") == 0) return consumer(count);
    return 2;
}
