// RAD20 harness: can idle placeholder queues keep the gfx11 dispatch path warm only when we want it?
//
// Background (rad20-queue-probe.sh): with more active compute queues than the GPU's hardware slots (8 on an XTX), MES is
// oversubscribed and stays busy, which costs ~85 W at idle but dispatches kernels faster (p95 launch gap 3.6 vs 7.6 us).
// KFD counts a queue as active only while its queue percentage is > 0 (QUEUE_IS_ACTIVE); AMDKFD_IOC_UPDATE_QUEUE with 0
// removes it from MES (remove_queue_mes) and 100 adds it back -- what ROCr's own AqlQueue::Suspend/Resume do. So a
// process could hold P placeholder queues and switch oversubscription on for a burst and off between bursts.
//
// This measures, on one GPU, for each mode: the start-to-start gap of a chain of tiny kernels pre-queued behind a host
// released wait (the helper's decode pattern), the cost of switching the placeholders, and holds the GPU in the mode
// for a few seconds (PHASE lines) so the runner can sample power.
//   base:    no placeholders
//   active:  P placeholder queues, active (oversubscribes MES when the process total exceeds the slots)
//   parked:  P placeholder queues, set to 0% (inactive)
// The placeholders never receive work. Pass/fail is printed; exits nonzero on any error.
// Build: hipcc -O2 rad20-queue-warm.cpp -o rad20-queue-warm -lhsa-runtime64
// Run:   ROCR_VISIBLE_DEVICES=<n> ./rad20-queue-warm <mode> <placeholders> <rounds> <hold seconds>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <linux/kfd_ioctl.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
constexpr int kChain = 8;          // kernels per round (front, gemv, gemv, reduce, publish ... order of magnitude)
constexpr uint32_t kKfdNormalPriority = 7;   // libhsakmt priority_map[HSA_QUEUE_PRIORITY_NORMAL + 3]

#define HIPCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "FAIL %s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)
#define HSACK(x) do { hsa_status_t s_ = (x); if (s_ != HSA_STATUS_SUCCESS) { const char* m_ = nullptr; \
    hsa_status_string(s_, &m_); std::fprintf(stderr, "FAIL %s: %s\n", #x, m_ ? m_ : "?"); std::exit(2); } } while (0)

double now_s() { timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + t.tv_nsec * 1e-9; }
void phase(const char* what) { std::printf("PHASE %s t=%.3f\n", what, now_s()); std::fflush(stdout); }

__global__ void stamp(uint64_t* out) {
    if (threadIdx.x == 0) out[0] = wall_clock64();
}

hsa_status_t first_gpu(hsa_agent_t a, void* out) {
    hsa_device_type_t t{};
    if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t) == HSA_STATUS_SUCCESS && t == HSA_DEVICE_TYPE_GPU) {
        *static_cast<hsa_agent_t*>(out) = a;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

/* Set a queue's percentage through KFD on this process's own /dev/kfd handle (KFD keys queues by process). */
double set_percent(int kfd, hsa_queue_t* q, uint32_t pct) {
    kfd_ioctl_update_queue_args a{};
    a.ring_base_address = reinterpret_cast<uint64_t>(q->base_address);
    a.queue_id = static_cast<uint32_t>(q->id);
    a.ring_size = q->size * 64u;
    a.queue_percentage = pct;
    a.queue_priority = kKfdNormalPriority;
    const double t0 = now_s();
    if (ioctl(kfd, AMDKFD_IOC_UPDATE_QUEUE, &a) != 0) {
        std::fprintf(stderr, "FAIL AMDKFD_IOC_UPDATE_QUEUE queue %u pct %u: %s\n", a.queue_id, pct, std::strerror(errno));
        std::exit(3);
    }
    return (now_s() - t0) * 1e6;
}

double pct_of(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t)(p * (v.size() - 1)))];
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s {base|active|parked} <placeholders> <rounds> <hold seconds>\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const int placeholders = mode == "base" ? 0 : std::atoi(argv[2]);
    const int rounds = std::atoi(argv[3]);
    const int hold = std::atoi(argv[4]);
    if ((mode != "base" && mode != "active" && mode != "parked") || placeholders < 0 || rounds < 1 || hold < 0) return 2;

    HIPCK(hipSetDevice(0));
    hipStream_t s;
    HIPCK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

    // Placeholder queues on the same agent, through the public HSA API; they never receive packets.
    HSACK(hsa_init());
    hsa_agent_t gpu{};
    const hsa_status_t it = hsa_iterate_agents(first_gpu, &gpu);
    if ((it != HSA_STATUS_SUCCESS && it != HSA_STATUS_INFO_BREAK) || !gpu.handle) { std::fprintf(stderr, "FAIL no GPU\n"); return 2; }
    std::vector<hsa_queue_t*> ph(placeholders, nullptr);
    for (auto& q : ph) HSACK(hsa_queue_create(gpu, 64, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &q));
    const int kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if (kfd < 0) { std::perror("FAIL open /dev/kfd"); return 2; }

    std::vector<double> park_us, unpark_us;
    if (mode == "parked") for (auto* q : ph) park_us.push_back(set_percent(kfd, q, 0));

    // Toggle cost: switch every placeholder off and on a few times (only in a placeholder mode); end in the mode.
    if (mode != "base") {
        for (int r = 0; r < 5; ++r) {
            for (auto* q : ph) park_us.push_back(set_percent(kfd, q, 0));
            for (auto* q : ph) unpark_us.push_back(set_percent(kfd, q, 100));
        }
        if (mode == "parked") for (auto* q : ph) set_percent(kfd, q, 0);
    }

    // Dispatch chains: each round, K stamp kernels behind a wait the host releases after a short idle (decode-like).
    uint32_t* flag = nullptr;
    uint64_t* stamps = nullptr;
    HIPCK(hipHostMalloc(&flag, sizeof(uint32_t), hipHostMallocCoherent));
    HIPCK(hipHostMalloc(&stamps, sizeof(uint64_t) * kChain * rounds, hipHostMallocCoherent));
    *flag = 0;
    int rate = 0;
    HIPCK(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0));   // kHz
    const double us_per_tick = 1000.0 / rate;
    std::vector<double> gaps;
    phase("chains-begin");
    for (int r = 0; r < rounds; ++r) {
        HIPCK(hipStreamWaitValue32(s, flag, (uint32_t)(r + 1), hipStreamWaitValueGte, 0xffffffffu));
        for (int k = 0; k < kChain; ++k) hipLaunchKernelGGL(stamp, dim3(1), dim3(64), 0, s, stamps + r * kChain + k);
        HIPCK(hipGetLastError());
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        __atomic_store_n(flag, (uint32_t)(r + 1), __ATOMIC_RELEASE);
        HIPCK(hipStreamSynchronize(s));
        for (int k = 1; k < kChain; ++k)
            gaps.push_back((double)(int64_t)(stamps[r * kChain + k] - stamps[r * kChain + k - 1]) * us_per_tick);
    }
    phase("chains-end");
    std::printf("RESULT mode %s placeholders %d start-to-start us: p50 %.2f p95 %.2f p99 %.2f mean %.2f (n %zu)\n",
                mode.c_str(), placeholders, pct_of(gaps, .5), pct_of(gaps, .95), pct_of(gaps, .99),
                gaps.empty() ? 0 : [&] { double t = 0; for (double g : gaps) t += g; return t / gaps.size(); }(), gaps.size());
    if (!park_us.empty())
        std::printf("RESULT toggle us per queue: park p50 %.1f p95 %.1f, unpark p50 %.1f p95 %.1f\n",
                    pct_of(park_us, .5), pct_of(park_us, .95), pct_of(unpark_us, .5), pct_of(unpark_us, .95));

    phase("hold-begin");   // the GPU sits in this mode, idle, for power sampling
    sleep((unsigned)hold);
    phase("hold-end");

    for (auto* q : ph) { if (mode == "parked") set_percent(kfd, q, 100); HSACK(hsa_queue_destroy(q)); }
    close(kfd);
    HIPCK(hipHostFree(flag));
    HIPCK(hipHostFree(stamps));
    HIPCK(hipStreamDestroy(s));
    std::printf("PASS %s\n", mode.c_str());
    return 0;
}
