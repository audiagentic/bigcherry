// Standalone reproducer: on gfx1100 (MES), one HSA queue holding a pending AQL BARRIER_AND keeps the GPU at 100% GFX
// activity and ~95 W instead of ~10 W idle. No GPU kernel runs: the packet processor holds the queue on a signal the
// host releases at the end. EMPTY and DRAINED are the controls (a live queue with no pending packet idles normally).
//
// Modes (one HSA queue on the first visible GPU agent; select the GPU with ROCR_VISIBLE_DEVICES):
//   empty    queue created, no packets
//   drained  one BARRIER_AND with no dependency, run to completion, queue kept alive
//   pending  BARRIER_AND depending on a signal at 1; checked still blocked after the hold; then released (signal -> 0)
//            and its completion awaited with a deadline
// Prints one line per phase change with CLOCK_MONOTONIC seconds, so an outside sampler can align its samples.
// Exits nonzero on any unexpected result. Build: hipcc -O2 rad20-mes-idle.cpp -o rad20-mes-idle -lhsa-runtime64
// Run:   ROCR_VISIBLE_DEVICES=<n> ./rad20-mes-idle {empty|drained|pending} <hold seconds>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>

namespace {
#define CK(x) do { hsa_status_t s_ = (x); if (s_ != HSA_STATUS_SUCCESS) { const char* m_ = nullptr; hsa_status_string(s_, &m_); \
    std::fprintf(stderr, "FAIL %s: %s\n", #x, m_ ? m_ : "?"); std::exit(2); } } while (0)

double now_s() { timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + t.tv_nsec * 1e-9; }
void phase(const char* what) { std::printf("PHASE %s t=%.3f\n", what, now_s()); std::fflush(stdout); }

hsa_status_t first_gpu(hsa_agent_t a, void* out) {
    hsa_device_type_t t{};
    if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t) == HSA_STATUS_SUCCESS && t == HSA_DEVICE_TYPE_GPU) {
        *static_cast<hsa_agent_t*>(out) = a;
        return HSA_STATUS_INFO_BREAK;   // stop: found; hsa_iterate_agents then returns INFO_BREAK, not SUCCESS
    }
    return HSA_STATUS_SUCCESS;
}

void queue_error(hsa_status_t s, hsa_queue_t*, void*) {
    const char* m = nullptr;
    hsa_status_string(s, &m);
    std::fprintf(stderr, "FAIL async queue error: %s\n", m ? m : "?");
    std::_Exit(3);
}

/* Packet variants for the pending wait (one variable each): the packet type and the fence scopes on its header. */
uint16_t g_type = HSA_PACKET_TYPE_BARRIER_AND;
uint16_t g_scope = HSA_FENCE_SCOPE_SYSTEM;

/* One barrier packet (BARRIER_AND / BARRIER_OR share the layout): the body is written first, the header last with
 * release order, then the doorbell. */
void barrier_and(hsa_queue_t* q, hsa_signal_t dep, hsa_signal_t completion) {
    const uint64_t idx = hsa_queue_add_write_index_relaxed(q, 1);
    if (idx - hsa_queue_load_read_index_scacquire(q) >= q->size) { std::fprintf(stderr, "FAIL queue full\n"); std::exit(2); }
    auto* p = static_cast<hsa_barrier_and_packet_t*>(q->base_address) + (idx % q->size);
    std::memset(reinterpret_cast<uint8_t*>(p) + sizeof(uint16_t) * 2, 0, sizeof(*p) - sizeof(uint16_t) * 2);
    p->dep_signal[0] = dep;
    p->completion_signal = completion;
    const uint16_t header = (uint16_t)((g_type << HSA_PACKET_HEADER_TYPE) | (1u << HSA_PACKET_HEADER_BARRIER) |
                                       (g_scope << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                                       (g_scope << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
    __atomic_store_n(&p->header, header, __ATOMIC_RELEASE);
    hsa_signal_store_screlease(q->doorbell_signal, (hsa_signal_value_t)idx);
}

bool wait_zero(hsa_signal_t s, double seconds) {
    uint64_t hz = 0;
    CK(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &hz));
    return hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_EQ, 0, (uint64_t)(seconds * (double)hz),
                                     HSA_WAIT_STATE_BLOCKED) == 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s {empty|drained|pending|pending-or|pending-agent|pending-nofence} <hold seconds>\n", argv[0]);
        return 2;
    }
    const char* arg = argv[1];
    const int hold = std::atoi(argv[2]);
    /* pending-or: BARRIER_OR on the same dependency; pending-agent / pending-nofence: the same BARRIER_AND with agent
     * scope or no fences (diagnostic only: cross-process production waits need system scope). */
    const char* mode = arg;
    if (!std::strcmp(arg, "pending-or")) { g_type = HSA_PACKET_TYPE_BARRIER_OR; mode = "pending"; }
    else if (!std::strcmp(arg, "pending-agent")) { g_scope = HSA_FENCE_SCOPE_AGENT; mode = "pending"; }
    else if (!std::strcmp(arg, "pending-nofence")) { g_scope = HSA_FENCE_SCOPE_NONE; mode = "pending"; }
    if (hold < 1 || (std::strcmp(mode, "empty") && std::strcmp(mode, "drained") && std::strcmp(mode, "pending"))) return 2;

    CK(hsa_init());
    hsa_agent_t gpu{};
    const hsa_status_t it = hsa_iterate_agents(first_gpu, &gpu);
    if ((it != HSA_STATUS_SUCCESS && it != HSA_STATUS_INFO_BREAK) || !gpu.handle) { std::fprintf(stderr, "FAIL no GPU agent\n"); return 2; }
    char name[64] = {};
    CK(hsa_agent_get_info(gpu, HSA_AGENT_INFO_NAME, name));
    std::printf("AGENT %s\n", name);

    hsa_queue_t* q = nullptr;
    CK(hsa_queue_create(gpu, 64, HSA_QUEUE_TYPE_SINGLE, queue_error, nullptr, UINT32_MAX, UINT32_MAX, &q));
    hsa_signal_t dep{}, done{};
    CK(hsa_signal_create(1, 0, nullptr, &dep));
    CK(hsa_signal_create(1, 0, nullptr, &done));

    if (!std::strcmp(mode, "drained")) {
        barrier_and(q, hsa_signal_t{0}, done);
        if (!wait_zero(done, 5.0)) { std::fprintf(stderr, "FAIL drained barrier did not complete\n"); return 3; }
    }
    if (!std::strcmp(mode, "pending")) barrier_and(q, dep, done);

    phase("hold-begin");
    sleep((unsigned)hold);
    phase("hold-end");

    if (!std::strcmp(mode, "pending")) {
        if (hsa_signal_load_scacquire(done) != 1) { std::fprintf(stderr, "FAIL barrier passed before release\n"); return 3; }
        hsa_signal_store_screlease(dep, 0);
        if (!wait_zero(done, 5.0)) { std::fprintf(stderr, "FAIL barrier not released within 5 s\n"); return 3; }
        phase("released");
        sleep(5);   // queue alive, nothing pending: should already be back near idle
        phase("released-end");
    }
    CK(hsa_queue_destroy(q));
    CK(hsa_signal_destroy(done));
    CK(hsa_signal_destroy(dep));
    CK(hsa_shut_down());
    std::printf("PASS %s\n", arg);
    return 0;
}
