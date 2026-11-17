// RAD20 lab-only: does HSA queue priority change idle power under an
// unsatisfied CP BARRIER_AND on gfx1100? No production/driver mutations.
// Build: /opt/rocm-7.2.4/bin/hipcc -O2 -std=c++17 rad20-queue-priority.cpp -lhsa-runtime64 -o rad20-queue-priority
// Run one XTX only: ROCR_VISIBLE_DEVICES=0 ./rad20-queue-priority low 15
// Compare low/normal/high and an "empty" control while sampling rocm-smi.
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <cstdint>
#include <sys/resource.h>

#define CK(call) do { hsa_status_t st = (call); if (st != HSA_STATUS_SUCCESS) { \
 const char* msg = nullptr; hsa_status_string(st, &msg); \
 fprintf(stderr, "%s failed: %s (%d)\\n", #call, msg ? msg : "unknown", (int)st); return 2; \
}} while(0)
static hsa_status_t find_gpu(hsa_agent_t a, void* data) {
    hsa_device_type_t type{};
    if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &type) == HSA_STATUS_SUCCESS
        && type == HSA_DEVICE_TYPE_GPU) {
        *static_cast<hsa_agent_t*>(data) = a;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}
static void error_cb(hsa_status_t s, hsa_queue_t*, void*) {
    fprintf(stderr, "HSA async queue error: %d\\n", (int)s);
    _Exit(3);
}
static void submit(hsa_queue_t* q, hsa_signal_t dep, hsa_signal_t completion) {
    uint64_t i = hsa_queue_add_write_index_relaxed(q, 1);
    while (i - hsa_queue_load_read_index_scacquire(q) >= q->size)
        std::this_thread::yield();
    auto* p = reinterpret_cast<hsa_barrier_and_packet_t*>(
        reinterpret_cast<uint8_t*>(q->base_address) + (i % q->size) * 64);
    std::memset(reinterpret_cast<uint8_t*>(p) + 4, 0, 60);
    p->dep_signal[0] = dep;
    p->completion_signal = completion;
    constexpr uint16_t hdr =
        (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
        (1u << HSA_PACKET_HEADER_BARRIER) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    __atomic_store_n(&p->header, hdr, __ATOMIC_RELEASE);
    hsa_signal_store_screlease(q->doorbell_signal, (hsa_signal_value_t)i);
}
int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s {low|normal|high|empty} [seconds=15]\\n", argv[0]);
        return 2;
    }
    int sec = argc == 3 ? std::atoi(argv[2]) : 15;
    if (sec < 1 || sec > 60) return 2;
    CK(hsa_init());
    hsa_agent_t gpu{};
    auto result = hsa_iterate_agents(find_gpu, &gpu);
    if ((result != HSA_STATUS_SUCCESS && result != HSA_STATUS_INFO_BREAK) || !gpu.handle) {
        fprintf(stderr, "No HSA GPU agent\\n"); return 2;
    }
    hsa_queue_t* q = nullptr;
    CK(hsa_queue_create(gpu, 64, HSA_QUEUE_TYPE_SINGLE,
                        error_cb, nullptr, UINT32_MAX, UINT32_MAX, &q));
    const bool empty = strcmp(argv[1], "empty") == 0;
    hsa_amd_queue_priority_t pri = HSA_AMD_QUEUE_PRIORITY_NORMAL;
    if (strcmp(argv[1], "low") == 0) pri = HSA_AMD_QUEUE_PRIORITY_LOW;
    else if (strcmp(argv[1], "high") == 0) pri = HSA_AMD_QUEUE_PRIORITY_HIGH;
    else if (strcmp(argv[1], "normal") && !empty) return 2;
    CK(hsa_amd_queue_set_priority(q, pri));

    hsa_signal_t blocker{}, marker{};
    CK(hsa_signal_create(1, 0, nullptr, &blocker));
    CK(hsa_signal_create(1, 0, nullptr, &marker));
    if (!empty) submit(q, blocker, marker);
    fprintf(stderr, "PHASE idle priority=%s seconds=%d, pending=%s\\n",
            argv[1], sec, empty ? "no" : "yes");
    fflush(stderr);
    std::this_thread::sleep_for(std::chrono::seconds(sec));
    if (!empty) {
        auto before = hsa_signal_load_scacquire(marker);
        if (before != 1) {
            fprintf(stderr, "FAIL: barrier passed prematurely: %lld\\n", (long long)before);
            return 3;
        }
        hsa_signal_store_screlease(blocker, 0);
        auto after = hsa_signal_wait_scacquire(marker, HSA_SIGNAL_CONDITION_EQ,
                    0, 2000000000ull, HSA_WAIT_STATE_BLOCKED);
        if (after != 0) { fprintf(stderr, "FAIL: barrier release timed out\\n"); return 3; }
    }
    CK(hsa_queue_destroy(q));
    CK(hsa_signal_destroy(marker));
    CK(hsa_signal_destroy(blocker));
    CK(hsa_shut_down());
    fprintf(stderr, "PASS: queue drained, all signals retired\\n");
    return 0;
}
