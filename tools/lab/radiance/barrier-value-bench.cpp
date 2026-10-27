// RAD20 stage A: packet-processor waits across processes and GPUs, no resident wait kernel, no host polling loop.
//
// latency  - a producer process (engine stand-in, R9700) owns an IPC HSA signal that starts at 2^40 and decrements it once
//            per notification (a BARRIER_AND whose completion signal it is). A consumer process (helper stand-in, XTX)
//            attaches it and keeps `depth` requests queued on its own HSA queue: for request N an AMD BARRIER_VALUE packet
//            (signal < 2^40 - N + 1) holds the queue in the packet processor, then a BARRIER_AND marker completes a local
//            signal. Checks every notification releases exactly one marker and none early; measures notify -> marker.
// idle     - consumer only: Q queues each hold one BARRIER_VALUE that is never satisfied, for `seconds`, while an outside
//            sampler reads board power / GFX activity; then releases them by storing the signal (the shutdown path) and
//            checks every marker completed.
//
// Build: hipcc -O2 barrier-value-bench.cpp -o barrier-value-bench -lhsa-runtime64
// Run:   ROCR_VISIBLE_DEVICES=<xtx> ./barrier-value-bench consumer-latency [count=2000] [depth=4] &
//        ROCR_VISIBLE_DEVICES=<r9700> ./barrier-value-bench producer [count=2000] [interval_us=150]
//        ROCR_VISIBLE_DEVICES=<xtx> ./barrier-value-bench idle [queues=1] [seconds=30]
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

#define CK(x) do { hsa_status_t s_ = (x); if (s_ != HSA_STATUS_SUCCESS) { const char* m_ = nullptr; hsa_status_string(s_, &m_); \
    std::fprintf(stderr, "%s: %s\n", #x, m_ ? m_ : "?"); std::exit(1); } } while (0)

namespace {
constexpr const char* kShm = "/rad20-barrier-value-bench";
constexpr int64_t kStart = 1ll << 40;
constexpr int kMax = 100000;

struct Shared {
    hsa_amd_ipc_signal_t handle;
    volatile int ready, armed, done;
    int64_t t_send[kMax];
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
    if (fd < 0) return nullptr;
    if (create && ftruncate(fd, sizeof(Shared)) != 0) { std::perror("ftruncate"); std::exit(1); }
    void* p = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    return p == MAP_FAILED ? nullptr : static_cast<Shared*>(p);
}

/* Reserve one slot, fill everything but the header, then publish the header and ring the doorbell. */
template <class Fill>
void submit(hsa_queue_t* q, uint16_t header, uint16_t amd_format, Fill fill) {
    const uint64_t idx = hsa_queue_add_write_index_relaxed(q, 1);
    while (idx - hsa_queue_load_read_index_scacquire(q) >= q->size) {}   // never happens at these depths
    auto* slot = reinterpret_cast<uint8_t*>(q->base_address) + (idx % q->size) * 64;
    std::memset(slot + 4, 0, 60);
    fill(slot);
    const uint32_t word = header | ((uint32_t)amd_format << 16);
    __atomic_store_n(reinterpret_cast<uint32_t*>(slot), word, __ATOMIC_RELEASE);
    hsa_signal_store_screlease(q->doorbell_signal, (hsa_signal_value_t)idx);
}

constexpr uint16_t kSysFences = (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                                (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);

void barrier_and(hsa_queue_t* q, hsa_signal_t completion, hsa_signal_t dep = hsa_signal_t{0}) {
    submit(q, (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) | kSysFences, 0,
           [&](uint8_t* p) {
               auto* b = reinterpret_cast<hsa_barrier_and_packet_t*>(p);
               b->dep_signal[0] = dep;
               b->completion_signal = completion;
           });
}

/* Hold the queue until (signal & mask) < value. */
void barrier_value_lt(hsa_queue_t* q, hsa_signal_t sig, int64_t value) {
    submit(q, (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) | kSysFences,
           HSA_AMD_PACKET_TYPE_BARRIER_VALUE, [&](uint8_t* p) {
               auto* b = reinterpret_cast<hsa_amd_barrier_value_packet_t*>(p);
               b->signal = sig;
               b->value = value;
               b->mask = (hsa_signal_value_t)-1;
               b->cond = HSA_SIGNAL_CONDITION_LT;
           });
}

int producer(int count, int interval_us) {
    shm_unlink(kShm);
    Shared* sh = map_shared(true);
    std::memset(sh, 0, sizeof(Shared));
    CK(hsa_init());
    hsa_agent_t gpu{};
    hsa_iterate_agents(pick_gpu, &gpu);
    hsa_signal_t sig;
    CK(hsa_amd_signal_create(kStart, 0, nullptr, HSA_AMD_SIGNAL_IPC, &sig));
    CK(hsa_amd_ipc_signal_create(sig, &sh->handle));
    hsa_queue_t* q = nullptr;
    CK(hsa_queue_create(gpu, 256, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &q));
    __atomic_store_n(&sh->ready, 1, __ATOMIC_RELEASE);
    while (!__atomic_load_n(&sh->armed, __ATOMIC_ACQUIRE)) usleep(1000);
    usleep(300000);   // the consumer's first barriers are now held: none may have released early
    for (int i = 0; i < count; ++i) {
        sh->t_send[i] = now_ns();
        barrier_and(q, sig);
        const int64_t until = now_ns() + (int64_t)interval_us * 1000;
        timespec ts{(time_t)(until / 1000000000ll), (long)(until % 1000000000ll)};
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
    }
    while (hsa_signal_load_scacquire(sig) > kStart - count) usleep(1000);
    while (!__atomic_load_n(&sh->done, __ATOMIC_ACQUIRE)) usleep(1000);
    hsa_queue_destroy(q);
    hsa_signal_destroy(sig);
    hsa_shut_down();
    shm_unlink(kShm);
    return 0;
}

int consumer_latency(int count, int depth) {
    Shared* sh = nullptr;
    for (int i = 0; i < 600 && !(sh = map_shared(false)); ++i) usleep(100000);
    if (!sh) { std::fprintf(stderr, "consumer: no producer\n"); return 1; }
    while (!__atomic_load_n(&sh->ready, __ATOMIC_ACQUIRE)) usleep(1000);
    CK(hsa_init());
    hsa_agent_t gpu{};
    hsa_iterate_agents(pick_gpu, &gpu);
    hsa_signal_t sig, marker;
    CK(hsa_amd_ipc_signal_attach(&sh->handle, &sig));
    CK(hsa_signal_create(kStart, 0, nullptr, &marker));   // local interrupt signal, one decrement per released request
    hsa_queue_t* q = nullptr;
    CK(hsa_queue_create(gpu, 256, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &q));
    int queued = 0;
    auto queue_one = [&] { ++queued; barrier_value_lt(q, sig, kStart - queued + 1); barrier_and(q, marker); };
    for (int i = 0; i < depth && queued < count; ++i) queue_one();
    usleep(100000);
    const bool early = hsa_signal_load_scacquire(marker) != kStart;   // nothing notified yet: nothing may have passed
    __atomic_store_n(&sh->armed, 1, __ATOMIC_RELEASE);
    rusage r0{}, r1{};
    getrusage(RUSAGE_SELF, &r0);
    const int64_t w0 = now_ns();
    std::vector<double> lat;
    int64_t last = kStart;
    int wakes = 0, ahead = 0;
    while ((int)lat.size() < count) {
        const hsa_signal_value_t v = hsa_signal_wait_scacquire(marker, HSA_SIGNAL_CONDITION_LT, last, 2000000000ull,
                                                               HSA_WAIT_STATE_BLOCKED);
        const int64_t t = now_ns();
        if (v >= last) { std::fprintf(stderr, "consumer: wait timed out at %zu\n", lat.size()); break; }
        ++wakes;
        const int64_t released = kStart - v;
        const int64_t notified = kStart - hsa_signal_load_scacquire(sig);
        if (released > notified) ++ahead;   // a marker passed a request not yet notified: must never happen
        for (int64_t k = kStart - last; k < released && k < count; ++k) lat.push_back((t - sh->t_send[k]) / 1000.0);
        last = v;
        while (queued < count && queued - (int)(kStart - last) < depth) queue_one();
    }
    const int64_t w1 = now_ns();
    getrusage(RUSAGE_SELF, &r1);
    __atomic_store_n(&sh->done, 1, __ATOMIC_RELEASE);
    auto us = [](const timeval& a, const timeval& b) { return (b.tv_sec - a.tv_sec) * 1e6 + (b.tv_usec - a.tv_usec); };
    const double cpu_us = us(r0.ru_utime, r1.ru_utime) + us(r0.ru_stime, r1.ru_stime);
    std::sort(lat.begin(), lat.end());
    auto pct = [&](double p) { return lat.empty() ? 0.0 : lat[(size_t)(p * (lat.size() - 1))]; };
    std::printf("latency: %zu of %d released, %d wakes, early release before any notify: %s, released ahead of notify: %d\n",
                lat.size(), count, wakes, early ? "YES" : "no", ahead);
    std::printf("  notify -> marker us: p50 %.1f p90 %.1f p99 %.1f p99.9 %.1f max %.1f; consumer CPU %.2f%% of a core\n",
                pct(.5), pct(.9), pct(.99), pct(.999), lat.empty() ? 0.0 : lat.back(),
                100.0 * cpu_us * 1000.0 / (double)(w1 - w0));
    hsa_queue_destroy(q);
    hsa_signal_destroy(marker);
    hsa_signal_destroy(sig);
    hsa_shut_down();
    return 0;
}

int idle(int queues, int seconds, bool use_and, char idle_kind) {   // idle_kind: b(arrier) e(mpty queue) d(rained queue);   // use_and: BARRIER_AND on a signal at 1 (gfx1100 rejects BARRIER_VALUE)
    CK(hsa_init());
    hsa_agent_t gpu{};
    hsa_iterate_agents(pick_gpu, &gpu);
    hsa_signal_t sig, marker;
    CK(hsa_amd_signal_create(use_and ? 1 : kStart, 0, nullptr, HSA_AMD_SIGNAL_IPC, &sig));
    CK(hsa_signal_create(queues, 0, nullptr, &marker));
    std::vector<hsa_queue_t*> qs((size_t)queues);
    for (auto& q : qs) {
        CK(hsa_queue_create(gpu, 64, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &q));
        if (idle_kind == 'e') continue;                         // empty: live queue, no packets
        if (idle_kind == 'd') { barrier_and(q, marker); continue; }   // drained: one packet ran to completion
        if (use_and) barrier_and(q, marker, sig);   // held until the signal reaches 0
        else { barrier_value_lt(q, sig, 1); barrier_and(q, marker); }
    }
    std::printf("idle: %d queue(s) holding a %s from %lld\n", queues, use_and ? "BARRIER_AND" : "BARRIER_VALUE", (long long)(now_ns() / 1000000000ll));
    std::fflush(stdout);
    sleep((unsigned)seconds);
    if (idle_kind != 'b') { std::printf("idle: %d %s queue(s) held %d s (no waits to release)\n", queues, idle_kind == 'e' ? "empty" : "drained", seconds); for (auto* q : qs) hsa_queue_destroy(q); hsa_shut_down(); return 0; }
    const bool held = hsa_signal_load_scacquire(marker) == queues;
    hsa_signal_store_screlease(sig, 0);   // the shutdown release: the holder satisfies every outstanding wait
    const hsa_signal_value_t v = hsa_signal_wait_scacquire(marker, HSA_SIGNAL_CONDITION_EQ, 0, 2000000000ull,
                                                           HSA_WAIT_STATE_BLOCKED);
    std::printf("idle: held for %d s: %s; released by signal store: %s (to %lld)\n", seconds, held ? "yes" : "NO",
                v == 0 ? "all markers completed" : "NOT ALL", (long long)(now_ns() / 1000000000ll));
    for (auto* q : qs) hsa_queue_destroy(q);
    hsa_signal_destroy(marker);
    hsa_signal_destroy(sig);
    hsa_shut_down();
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s producer|consumer-latency|idle ...\n", argv[0]); return 2; }
    const int a2 = argc > 2 ? std::atoi(argv[2]) : 0, a3 = argc > 3 ? std::atoi(argv[3]) : 0;
    if (!std::strcmp(argv[1], "producer")) return producer(std::min(a2 ? a2 : 2000, kMax), a3 ? a3 : 150);
    if (!std::strcmp(argv[1], "consumer-latency")) return consumer_latency(std::min(a2 ? a2 : 2000, kMax), a3 ? a3 : 4);
    if (!std::strcmp(argv[1], "idle")) { const char* k = argc > 4 ? argv[4] : "and"; return idle(a2 ? a2 : 1, a3 ? a3 : 30, true, !std::strcmp(k, "empty") ? 'e' : !std::strcmp(k, "drained") ? 'd' : 'b'); }
    return 2;
}
