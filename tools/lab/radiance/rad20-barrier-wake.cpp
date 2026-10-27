// RAD20 option B lab harness: cross-GPU wake latency of a wait on an RX 7900 XTX queue that the R9700 releases.
//
// Question: a wait sits on an XTX queue; the R9700 (or a host thread) releases it by writing over PCIe.
// How long from the release write until the next kernel on the XTX starts?
//
// Arms (--arms, comma list):
//   A.gpu   AQL BARRIER_AND on an HSA signal (raw HSA queue on the XTX), then a timestamp kernel. Released by an
//           R9700 kernel that records its wall clock and then does a system-scope store of 0 to the signal value.
//   A.host  same wait; released by the engine role's host thread (hsa_signal_store_screlease).
//   A.self  same wait; released by the helper role's own host thread (control only).
//   B.gpu   baseline: hipStreamWaitValue32 on a uint32 in hipHostRegister'd shared memory, then the timestamp
//           kernel; released by an R9700 kernel's system-scope store of the sequence number.
//   B.host  same wait; engine host store.   B.self  same wait; helper host store (control).
// Latency = XTX timestamp-kernel start - release time (R9700 wall clock just before its store, or host
// CLOCK_MONOTONIC just before the host store).
//
// Roles mirror production: the helper owns the XTX queue and the waits (r3_moe_worker), the engine owns the R9700
// and releases them.
//   --mode single   both roles as threads in one process, plain HSA signal
//   --mode helper   helper process: creates the shm segment (--shm) and an IPC signal (HSA_AMD_SIGNAL_IPC)
//   --mode engine   engine process: opens the segment and attaches the IPC signal
// --dump FILE (single or helper) writes one CSV row per iteration: arm, iteration, latency, and the host-clock
// times of arming, release and XTX kernel start.
// Schedule: per iteration the helper idles --idle-us with nothing queued, arms the wait, and the release comes
// --gap-us after arming. Every wait has a deadline (--deadline-ms); a timed-out wait is cancelled and drained.
//
// Devices: R9700_BDF and XTX_BDF (PCI addresses such as 0000:09:00.0) map to ROCr agents by
// HSA_AMD_AGENT_INFO_BDFID/DOMAIN and to HIP devices by hipDeviceGetPCIBusId. No device index is hard-coded.
// Clocks: each GPU's constant wall clock (100 MHz) is fitted to host CLOCK_MONOTONIC from ping-pong samples taken
// before and after the arms (lowest-round-trip quarter of each set, least squares). The error bound per clock is
// max(|residual| + round trip / 2) over the samples used; a GPU-released arm's bound is the sum of both clocks'.
//
// The raw-queue timestamp kernel is the one HIP loads for the XTX: it is found through the ROCr loader extension
// (iterate executables, agent kernel symbol rad20_ts) so the harness stays one file.
// Build: hipcc -O2 -std=c++17 -o rad20-barrier-wake rad20-barrier-wake.cpp -lhsa-runtime64
#include <hip/hip_runtime.h>
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <ctime>
#include <fcntl.h>
#include <numeric>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define HIPCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
    std::exit(2); } } while (0)
#define HSACK(x) do { hsa_status_t s_ = (x); if (s_ != HSA_STATUS_SUCCESS) { const char* m_ = nullptr; \
    hsa_status_string(s_, &m_); std::fprintf(stderr, "FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #x, m_ ? m_ : "?"); \
    std::exit(2); } } while (0)

// ---------------------------------------------------------------------------------------------- device code

extern "C" __global__ void rad20_ts(unsigned long long* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        __hip_atomic_store(out, (unsigned long long)wall_clock64(), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

extern "C" __global__ void rad20_release_signal(long long* value, unsigned long long* ts) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const unsigned long long t = wall_clock64();
        __hip_atomic_store(value, 0LL, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(ts, t, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

extern "C" __global__ void rad20_release_word(unsigned* word, unsigned value, unsigned long long* ts) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const unsigned long long t = wall_clock64();
        __hip_atomic_store(word, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(ts, t, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

// Clock ping-pong: for k = 1..n wait for flag >= k, record the wall clock, ack k. Bounded by max_ticks.
extern "C" __global__ void rad20_calib(const unsigned* flag, unsigned* ack, unsigned long long* ts, unsigned n,
                                       unsigned long long max_ticks) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const unsigned long long begin = wall_clock64();
    for (unsigned k = 1; k <= n; ++k) {
        while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < k) {
            if ((unsigned long long)wall_clock64() - begin > max_ticks) {
                __hip_atomic_store(ack, 0xffffffffu, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
                return;
            }
            __builtin_amdgcn_s_sleep(1);
        }
        ts[k - 1] = wall_clock64();
        __hip_atomic_store(ack, k, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

// ---------------------------------------------------------------------------------------------- host utilities

static int64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

static void sleep_us(int64_t us) {
    if (us <= 0) return;
    timespec ts{time_t(us / 1000000), long((us % 1000000) * 1000)};
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {}
}

static void fail(const char* what) {
    std::fprintf(stderr, "FAIL %s\n", what);
    std::exit(2);
}

// hsa_iterate_* returns HSA_STATUS_INFO_BREAK when the callback stopped early; that is not an error.
static void check_iter(hsa_status_t s, const char* what) {
    if (s == HSA_STATUS_SUCCESS || s == HSA_STATUS_INFO_BREAK) return;
    const char* m = nullptr;
    hsa_status_string(s, &m);
    std::fprintf(stderr, "FAIL %s: %s\n", what, m ? m : "?");
    std::exit(2);
}

struct Options {
    std::string mode = "single";
    std::string arms = "A.gpu,A.host,B.gpu,B.host";
    std::string shm = "/rad20-barrier-wake";
    int n = 200;
    int64_t gap_us = 100;
    int64_t idle_us = 0;
    int64_t deadline_ms = 2000;
    int calib = 200;
    std::string dump;
};

static Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (i + 1 >= argc) { std::fprintf(stderr, "FAIL missing value for %s\n", k.c_str()); std::exit(2); }
        const char* v = argv[++i];
        if (k == "--mode") o.mode = v;
        else if (k == "--arms") o.arms = v;
        else if (k == "--shm") o.shm = v;
        else if (k == "--n") o.n = std::atoi(v);
        else if (k == "--gap-us") o.gap_us = std::atoll(v);
        else if (k == "--idle-us") o.idle_us = std::atoll(v);
        else if (k == "--deadline-ms") o.deadline_ms = std::atoll(v);
        else if (k == "--calib") o.calib = std::atoi(v);
        else if (k == "--dump") o.dump = v;
        else { std::fprintf(stderr, "FAIL unknown option %s\n", k.c_str()); std::exit(2); }
    }
    if (o.mode != "single" && o.mode != "helper" && o.mode != "engine") fail("--mode must be single, helper or engine");
    if (o.calib < 20) fail("--calib must be >= 20");
    if (o.deadline_ms <= 0) fail("--deadline-ms must be > 0");
    return o;
}

struct Bdf {
    uint32_t domain = 0, bus = 0, dev = 0, fn = 0;
    std::string text;
    uint32_t id() const { return (bus << 8) | (dev << 3) | fn; }
    bool same(const Bdf& o) const { return domain == o.domain && id() == o.id(); }
};

static bool parse_bdf(const char* s, Bdf* out) {
    unsigned d = 0, b = 0, v = 0, f = 0;
    if (std::sscanf(s, "%x:%x:%x.%x", &d, &b, &v, &f) != 4) {
        d = 0;
        if (std::sscanf(s, "%x:%x.%x", &b, &v, &f) != 3) return false;
    }
    if (b > 0xff || v > 0x1f || f > 7) return false;
    out->domain = d; out->bus = b; out->dev = v; out->fn = f; out->text = s;
    return true;
}

static Bdf env_bdf(const char* name) {
    const char* s = std::getenv(name);
    Bdf b;
    if (!s || !parse_bdf(s, &b)) {
        std::fprintf(stderr, "FAIL %s must be set to a PCI address such as 0000:09:00.0\n", name);
        std::exit(2);
    }
    return b;
}

struct AgentQuery {
    Bdf want;
    hsa_agent_t agent{};
    int rocr_index = -1;
    int seen = 0;
    char name[64] = {};
};

static hsa_status_t match_agent(hsa_agent_t a, void* p) {
    auto* q = static_cast<AgentQuery*>(p);
    hsa_device_type_t type;
    if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS || type != HSA_DEVICE_TYPE_GPU)
        return HSA_STATUS_SUCCESS;
    const int index = q->seen++;
    uint32_t bdfid = 0, domain = 0;
    if (hsa_agent_get_info(a, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdfid) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;
    hsa_agent_get_info(a, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN, &domain);
    if (bdfid != q->want.id() || domain != q->want.domain) return HSA_STATUS_SUCCESS;
    q->agent = a;
    q->rocr_index = index;
    hsa_agent_get_info(a, HSA_AGENT_INFO_NAME, q->name);
    return HSA_STATUS_INFO_BREAK;
}

static AgentQuery find_agent(const Bdf& want, const char* role) {
    AgentQuery q;
    q.want = want;
    check_iter(hsa_iterate_agents(match_agent, &q), "hsa_iterate_agents");
    if (q.rocr_index < 0) {
        std::fprintf(stderr, "FAIL no ROCr GPU agent at %s (%s)\n", want.text.c_str(), role);
        std::exit(2);
    }
    return q;
}

static int hip_device_for(const Bdf& want, const char* role) {
    int count = 0;
    HIPCK(hipGetDeviceCount(&count));
    for (int i = 0; i < count; ++i) {
        char buf[64] = {};
        HIPCK(hipDeviceGetPCIBusId(buf, sizeof buf, i));
        Bdf b;
        if (parse_bdf(buf, &b) && b.same(want)) return i;
    }
    std::fprintf(stderr, "FAIL no HIP device at %s (%s)\n", want.text.c_str(), role);
    std::exit(2);
}

static bool wait_hip_stream(hipStream_t s, int64_t deadline) {
    for (;;) {
        const hipError_t e = hipStreamQuery(s);
        if (e == hipSuccess) return true;
        if (e != hipErrorNotReady) HIPCK(e);
        if (now_ns() > deadline) return false;
        sleep_us(20);
    }
}

// ---------------------------------------------------------------------------------------------- clock alignment

struct ClockSample { uint64_t ticks; int64_t t0, t1; };

struct ClockFit {
    double a = 0, b = 0;      // host_ns = a + b * ticks
    double err_ns = 0, ppm = 0, min_rtt_ns = 0;
    int used = 0;
    double to_ns(uint64_t t) const { return a + b * double(t); }
};

static std::vector<ClockSample> calibrate(int hip_dev, hipStream_t stream, int n, double rate_hz,
                                          int64_t deadline_ms) {
    HIPCK(hipSetDevice(hip_dev));
    unsigned *flag = nullptr, *ack = nullptr;
    unsigned long long* ts = nullptr;
    HIPCK(hipHostMalloc(&flag, 64, hipHostMallocMapped | hipHostMallocCoherent));
    HIPCK(hipHostMalloc(&ack, 64, hipHostMallocMapped | hipHostMallocCoherent));
    HIPCK(hipHostMalloc(&ts, sizeof(unsigned long long) * n, hipHostMallocMapped | hipHostMallocCoherent));
    __atomic_store_n(flag, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(ack, 0u, __ATOMIC_RELEASE);
    unsigned *dflag = nullptr, *dack = nullptr;
    unsigned long long* dts = nullptr;
    HIPCK(hipHostGetDevicePointer((void**)&dflag, flag, 0));
    HIPCK(hipHostGetDevicePointer((void**)&dack, ack, 0));
    HIPCK(hipHostGetDevicePointer((void**)&dts, ts, 0));
    const double budget_s = n * 0.002 + deadline_ms * 1e-3;
    const unsigned long long max_ticks = (unsigned long long)(budget_s * rate_hz);
    rad20_calib<<<1, 32, 0, stream>>>(dflag, dack, dts, (unsigned)n, max_ticks);
    HIPCK(hipGetLastError());
    std::vector<ClockSample> s(n);
    const int64_t deadline = now_ns() + int64_t(budget_s * 1e9);
    for (int k = 1; k <= n; ++k) {
        sleep_us(300);
        const int64_t t0 = now_ns();
        __atomic_store_n(flag, (unsigned)k, __ATOMIC_RELEASE);
        unsigned seen;
        while ((seen = __atomic_load_n(ack, __ATOMIC_ACQUIRE)) < (unsigned)k) {
            if (now_ns() > deadline) fail("clock calibration: no ack before the deadline");
        }
        if (seen == 0xffffffffu) fail("clock calibration: device-side deadline expired");
        s[k - 1] = {0, t0, now_ns()};
    }
    if (!wait_hip_stream(stream, deadline + 1000000000)) fail("clock calibration kernel did not finish");
    for (int k = 0; k < n; ++k) s[k].ticks = ts[k];
    HIPCK(hipHostFree(flag));
    HIPCK(hipHostFree(ack));
    HIPCK(hipHostFree(ts));
    return s;
}

// Keep the lowest-round-trip quarter of one calibration set.
static void select_best(const std::vector<ClockSample>& s, std::vector<ClockSample>* out) {
    std::vector<ClockSample> v = s;
    std::sort(v.begin(), v.end(), [](const ClockSample& x, const ClockSample& y) { return x.t1 - x.t0 < y.t1 - y.t0; });
    const size_t keep = std::max<size_t>(std::min<size_t>(v.size(), 10), v.size() / 4);
    out->insert(out->end(), v.begin(), v.begin() + keep);
}

static ClockFit fit_clock(const std::vector<ClockSample>& pre, const std::vector<ClockSample>& post, double rate_hz) {
    std::vector<ClockSample> s;
    select_best(pre, &s);
    select_best(post, &s);
    double mx = 0, my = 0;
    for (const auto& c : s) { mx += double(c.ticks); my += 0.5 * double(c.t0 + c.t1); }
    mx /= s.size();
    my /= s.size();
    double sxx = 0, sxy = 0;
    for (const auto& c : s) {
        const double dx = double(c.ticks) - mx, dy = 0.5 * double(c.t0 + c.t1) - my;
        sxx += dx * dx;
        sxy += dx * dy;
    }
    ClockFit f;
    f.b = sxx > 0 ? sxy / sxx : 1e9 / rate_hz;
    f.a = my - f.b * mx;
    f.used = int(s.size());
    f.min_rtt_ns = 1e30;
    for (const auto& c : s) {
        const double rtt = double(c.t1 - c.t0);
        const double resid = std::fabs(0.5 * double(c.t0 + c.t1) - f.to_ns(c.ticks));
        f.err_ns = std::max(f.err_ns, resid + 0.5 * rtt);
        f.min_rtt_ns = std::min(f.min_rtt_ns, rtt);
    }
    f.ppm = (f.b * rate_hz / 1e9 - 1.0) * 1e6;
    return f;
}

// ---------------------------------------------------------------------------------------------- shared state

constexpr int kMaxArms = 8;
constexpr int kMaxIter = 4096;

enum class Wait { Barrier, StreamWait };
enum class Release { Gpu, Host, Self };

struct Arm {
    std::string name;
    Wait wait;
    Release release;
};

static std::vector<Arm> parse_arms(const std::string& list) {
    std::vector<Arm> arms;
    size_t pos = 0;
    while (pos <= list.size()) {
        const size_t end = std::min(list.find(',', pos), list.size());
        const std::string a = list.substr(pos, end - pos);
        pos = end + 1;
        if (a.empty()) continue;
        Arm arm{a, Wait::Barrier, Release::Gpu};
        if (a.size() < 3 || (a[0] != 'A' && a[0] != 'B') || a[1] != '.') fail("arm names are A|B . gpu|host|self");
        arm.wait = a[0] == 'A' ? Wait::Barrier : Wait::StreamWait;
        const std::string r = a.substr(2);
        if (r == "gpu") arm.release = Release::Gpu;
        else if (r == "host") arm.release = Release::Host;
        else if (r == "self") arm.release = Release::Self;
        else fail("arm names are A|B . gpu|host|self");
        arms.push_back(arm);
    }
    if (arms.empty() || arms.size() > size_t(kMaxArms)) fail("--arms must name 1..8 arms");
    return arms;
}

// Lives in a POSIX shm segment (two processes) or an anonymous shared mapping (single process). Registered with
// hipHostRegister in each process, so both GPUs can read and write it, as the production mailbox is.
struct alignas(4096) Ctl {
    std::atomic<uint32_t> helper_ready, engine_ready, sig_ready, abort;
    std::atomic<uint32_t> armed, done;
    std::atomic<uint32_t> engine_fit_ready, engine_detached;
    std::atomic<int32_t> helper_pid;  // a segment left by a dead helper must not be used
    hsa_amd_ipc_signal_t sig_handle;
    double engine_a, engine_b, engine_err, engine_ppm, engine_min_rtt;
    int32_t engine_used;
    alignas(64) uint32_t word;                       // arm B wait word, own cache line
    alignas(64) uint64_t start_ticks[kMaxArms][kMaxIter];
    uint64_t rel_ticks[kMaxArms][kMaxIter];
    int64_t rel_host_ns[kMaxArms][kMaxIter];
    int64_t armed_host_ns[kMaxArms][kMaxIter];
    uint8_t timed_out[kMaxArms][kMaxIter];
};
constexpr size_t kCtlBytes = (sizeof(Ctl) + 4095) & ~size_t(4095);

template <typename T>
static T* dev_ptr(char* ctl_dev, Ctl* c, T* host) {
    return reinterpret_cast<T*>(ctl_dev + (reinterpret_cast<char*>(host) - reinterpret_cast<char*>(c)));
}

static bool wait_at_least(const std::atomic<uint32_t>& v, uint32_t want, int64_t deadline) {
    for (;;) {
        if (v.load(std::memory_order_acquire) >= want) return true;
        if (now_ns() > deadline) return false;
        sleep_us(10);
    }
}

static bool wait_at_least(const std::atomic<uint32_t>& v, uint32_t want, const std::atomic<uint32_t>& abort,
                          int64_t deadline) {
    for (;;) {
        if (v.load(std::memory_order_acquire) >= want) return true;
        if (abort.load(std::memory_order_acquire)) return false;
        if (now_ns() > deadline) return false;
        sleep_us(10);
    }
}

// ---------------------------------------------------------------------------------------------- helper role (XTX)

struct SymbolQuery {
    hsa_agent_t agent;
    std::string name;
    hsa_executable_symbol_t symbol{};
    bool found = false;
};

static hsa_status_t match_symbol(hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t s, void* p) {
    auto* q = static_cast<SymbolQuery*>(p);
    hsa_symbol_kind_t kind;
    if (hsa_executable_symbol_get_info(s, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind) != HSA_STATUS_SUCCESS ||
        kind != HSA_SYMBOL_KIND_KERNEL)
        return HSA_STATUS_SUCCESS;
    uint32_t len = 0;
    hsa_executable_symbol_get_info(s, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &len);
    std::string name(len, '\0');
    hsa_executable_symbol_get_info(s, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data());
    if (name != q->name && name != q->name + ".kd") return HSA_STATUS_SUCCESS;
    q->symbol = s;
    q->found = true;
    return HSA_STATUS_INFO_BREAK;
}

static hsa_status_t visit_executable(hsa_executable_t e, void* p) {
    auto* q = static_cast<SymbolQuery*>(p);
    hsa_executable_iterate_agent_symbols(e, q->agent, match_symbol, p);  // executables for other agents: no match
    return q->found ? HSA_STATUS_INFO_BREAK : HSA_STATUS_SUCCESS;
}

static hsa_status_t match_kernarg_region(hsa_region_t r, void* p) {
    hsa_region_segment_t seg;
    if (hsa_region_get_info(r, HSA_REGION_INFO_SEGMENT, &seg) != HSA_STATUS_SUCCESS || seg != HSA_REGION_SEGMENT_GLOBAL)
        return HSA_STATUS_SUCCESS;
    uint32_t flags = 0;
    hsa_region_get_info(r, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
    if (!(flags & HSA_REGION_GLOBAL_FLAG_KERNARG)) return HSA_STATUS_SUCCESS;
    *static_cast<hsa_region_t*>(p) = r;
    return HSA_STATUS_INFO_BREAK;
}

static uint16_t packet_header(hsa_packet_type_t type) {
    return uint16_t((type << HSA_PACKET_HEADER_TYPE) | (1u << HSA_PACKET_HEADER_BARRIER) |
                    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
}

template <typename P>
static P* reserve_packet(hsa_queue_t* q, uint64_t* index, int64_t deadline) {
    const uint64_t i = hsa_queue_add_write_index_scacq_screl(q, 1);
    while (i - hsa_queue_load_read_index_scacquire(q) >= q->size) {
        if (now_ns() > deadline) fail("XTX HSA queue full past the deadline");
        std::this_thread::yield();
    }
    *index = i;
    P* p = reinterpret_cast<P*>(static_cast<char*>(q->base_address) + 64 * (i & (q->size - 1)));
    std::memset(reinterpret_cast<char*>(p) + 4, 0, sizeof(P) - 4);
    return p;
}

static void publish_packet(hsa_queue_t* q, void* p, uint16_t header, uint16_t setup, uint64_t index) {
    __atomic_store_n(static_cast<uint32_t*>(p), uint32_t(header) | (uint32_t(setup) << 16), __ATOMIC_RELEASE);
    hsa_signal_store_screlease(q->doorbell_signal, hsa_signal_value_t(index));
}

struct Helper {
    AgentQuery agent;
    int hip_dev = -1;
    double rate_hz = 1e8;
    hipStream_t stream{};
    hipEvent_t ev{};
    hsa_queue_t* queue = nullptr;
    uint64_t kernel_object = 0;
    uint32_t kernarg_size = 0, group_size = 0, private_size = 0;
    void* kernarg = nullptr;
    hsa_signal_t wait_sig{}, done_sig{};
    uint64_t wait_hint = 0;
    Ctl* c = nullptr;
    char* ctl_dev = nullptr;
    ClockFit fit;

    void setup(const Bdf& bdf, bool ipc, Ctl* ctl) {
        c = ctl;
        agent = find_agent(bdf, "XTX_BDF");
        hip_dev = hip_device_for(bdf, "XTX_BDF");
        HIPCK(hipSetDevice(hip_dev));
        int khz = 0;
        HIPCK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, hip_dev));
        rate_hz = khz * 1e3;
        HIPCK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        HIPCK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
        HIPCK(hipHostGetDevicePointer((void**)&ctl_dev, c, 0));

        // Make HIP load rad20_ts for this device, then take its kernel object from ROCr.
        rad20_ts<<<1, 32, 0, stream>>>(
            reinterpret_cast<unsigned long long*>(dev_ptr(ctl_dev, c, &c->start_ticks[0][0])));
        HIPCK(hipGetLastError());
        HIPCK(hipStreamSynchronize(stream));
        c->start_ticks[0][0] = 0;
        hsa_ven_amd_loader_1_03_pfn_t loader{};
        HSACK(hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(loader), &loader));
        SymbolQuery sq;
        sq.agent = agent.agent;
        sq.name = "rad20_ts";
        check_iter(loader.hsa_ven_amd_loader_iterate_executables(visit_executable, &sq), "iterate executables");
        if (!sq.found) fail("rad20_ts kernel symbol not found for the XTX agent");
        HSACK(hsa_executable_symbol_get_info(sq.symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object));
        HSACK(hsa_executable_symbol_get_info(sq.symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                                             &kernarg_size));
        HSACK(hsa_executable_symbol_get_info(sq.symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                             &group_size));
        HSACK(hsa_executable_symbol_get_info(sq.symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                             &private_size));
        hsa_region_t region{};
        check_iter(hsa_agent_iterate_regions(agent.agent, match_kernarg_region, &region), "iterate regions");
        if (!region.handle) fail("no kernarg region on the XTX agent");
        HSACK(hsa_memory_allocate(region, std::max<uint32_t>(kernarg_size, 64), &kernarg));
        HSACK(hsa_queue_create(agent.agent, 256, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX,
                               &queue));

        if (ipc) {
            HSACK(hsa_amd_signal_create(1, 0, nullptr, HSA_AMD_SIGNAL_IPC, &wait_sig));
            HSACK(hsa_amd_ipc_signal_create(wait_sig, &c->sig_handle));
        } else {
            HSACK(hsa_signal_create(1, 0, nullptr, &wait_sig));
        }
        c->sig_ready.store(1, std::memory_order_release);
        HSACK(hsa_signal_create(1, 0, nullptr, &done_sig));
        uint64_t freq = 0;
        HSACK(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq));
        wait_hint = std::max<uint64_t>(freq / 1000, 1);  // 1 ms per blocked wait, then the deadline is checked
        std::printf("helper: XTX %s %s rocr=%d hip=%d wallclock=%.0f Hz kernarg=%u bytes queue=%u slots\n",
                    bdf.text.c_str(), agent.name, agent.rocr_index, hip_dev, rate_hz, kernarg_size, queue->size);
    }

    void arm(const Arm& a, uint32_t seq, uint64_t* slot, int64_t deadline) {
        unsigned long long* out = reinterpret_cast<unsigned long long*>(dev_ptr(ctl_dev, c, slot));
        if (a.wait == Wait::StreamWait) {
            HIPCK(hipStreamWaitValue32(stream, dev_ptr(ctl_dev, c, &c->word), seq, hipStreamWaitValueGte, 0xffffffffu));
            rad20_ts<<<1, 32, 0, stream>>>(out);
            HIPCK(hipGetLastError());
            HIPCK(hipEventRecord(ev, stream));
            return;
        }
        hsa_signal_store_screlease(wait_sig, 1);
        hsa_signal_store_screlease(done_sig, 1);
        std::memset(kernarg, 0, kernarg_size);
        std::memcpy(kernarg, &out, sizeof out);
        uint64_t ib = 0;
        auto* b = reserve_packet<hsa_barrier_and_packet_t>(queue, &ib, deadline);
        b->dep_signal[0] = wait_sig;
        publish_packet(queue, b, packet_header(HSA_PACKET_TYPE_BARRIER_AND), 0, ib);
        uint64_t id = 0;
        auto* d = reserve_packet<hsa_kernel_dispatch_packet_t>(queue, &id, deadline);
        d->workgroup_size_x = 32;
        d->workgroup_size_y = 1;
        d->workgroup_size_z = 1;
        d->grid_size_x = 32;
        d->grid_size_y = 1;
        d->grid_size_z = 1;
        d->private_segment_size = private_size;
        d->group_segment_size = group_size;
        d->kernel_object = kernel_object;
        d->kernarg_address = kernarg;
        d->completion_signal = done_sig;
        publish_packet(queue, d, packet_header(HSA_PACKET_TYPE_KERNEL_DISPATCH),
                       uint16_t(1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS), id);
    }

    void release_from_host(const Arm& a, uint32_t value) {
        if (a.wait == Wait::Barrier) hsa_signal_store_screlease(wait_sig, 0);
        else __atomic_store_n(&c->word, value, __ATOMIC_RELEASE);
    }

    bool wait_done(const Arm& a, int64_t deadline) {
        if (a.wait == Wait::StreamWait) {
            for (;;) {
                const hipError_t e = hipEventQuery(ev);
                if (e == hipSuccess) return true;
                if (e != hipErrorNotReady) HIPCK(e);
                if (now_ns() > deadline) return false;
                sleep_us(20);
            }
        }
        for (;;) {
            if (hsa_signal_wait_scacquire(done_sig, HSA_SIGNAL_CONDITION_LT, 1, wait_hint, HSA_WAIT_STATE_BLOCKED) < 1)
                return true;
            if (now_ns() > deadline) return false;
        }
    }

    void run(const Options& o, const std::vector<Arm>& arms) {
        uint32_t seq = 0;
        for (size_t ai = 0; ai < arms.size(); ++ai) {
            const Arm& a = arms[ai];
            for (int i = 0; i < o.n; ++i) {
                ++seq;
                if (c->abort.load(std::memory_order_acquire)) return;
                sleep_us(o.idle_us);
                const int64_t deadline = now_ns() + (o.gap_us + o.deadline_ms * 1000) * 1000;
                arm(a, seq, &c->start_ticks[ai][i], deadline);
                c->armed_host_ns[ai][i] = now_ns();
                c->armed.store(seq, std::memory_order_release);
                if (a.release == Release::Self) {
                    sleep_us(o.gap_us);
                    c->rel_host_ns[ai][i] = now_ns();
                    release_from_host(a, seq);
                }
                if (!wait_done(a, deadline)) {
                    c->timed_out[ai][i] = 1;
                    c->abort.store(1, std::memory_order_release);
                    std::fprintf(stderr, "FAIL helper: arm %s iteration %d not released before the deadline; "
                                 "cancelling\n", a.name.c_str(), i);
                    // Cancel: satisfy the wait so the queue drains; arm B words only ever grow.
                    if (a.wait == Wait::Barrier) hsa_signal_store_screlease(wait_sig, 0);
                    else __atomic_store_n(&c->word, 0xffffffffu, __ATOMIC_RELEASE);
                    if (!wait_done(a, now_ns() + o.deadline_ms * 1000000))
                        fail("helper: cancelled wait still pending; queue stranded");
                    c->start_ticks[ai][i] = 0;
                    c->done.store(seq, std::memory_order_release);
                    return;
                }
                c->done.store(seq, std::memory_order_release);
            }
        }
    }

    void teardown() {
        HSACK(hsa_queue_destroy(queue));
        HSACK(hsa_signal_destroy(done_sig));
        HSACK(hsa_signal_destroy(wait_sig));
        HSACK(hsa_memory_free(kernarg));
        HIPCK(hipEventDestroy(ev));
        HIPCK(hipStreamDestroy(stream));
    }
};

// ---------------------------------------------------------------------------------------------- engine role (R9700)

// Address the R9700 uses for the signal's value word. ROCr signal memory is system memory; report which agents
// can reach it. A local signal is mapped to every agent. hsa_amd_ipc_signal_attach maps the shared signal for the
// host only (ROCr does not track the pointer), so for an attached signal the same IPC allocation (the signal
// handle is an IPC memory handle; the signal sits at the start of its page) is attached again for the R9700.
static long long* signal_value_for(hsa_signal_t sig, hsa_agent_t agent, const hsa_amd_ipc_signal_t* ipc_handle,
                                   void** gpu_mapping) {
    auto* value = const_cast<long long*>(
        reinterpret_cast<volatile long long*>(&reinterpret_cast<amd_signal_t*>(sig.handle)->value));
    hsa_amd_pointer_info_t info{};
    info.size = sizeof(info);
    uint32_t n = 0;
    hsa_agent_t* agents = nullptr;
    HSACK(hsa_amd_pointer_info(value, &info, std::malloc, &n, &agents));
    bool reachable = false;
    for (uint32_t k = 0; k < n; ++k) reachable |= agents[k].handle == agent.handle;
    std::free(agents);
    std::printf("engine: signal value %p pointer type=%d accessible agents=%u r9700=%s\n", (void*)value, int(info.type),
                n, reachable ? "yes" : "no");
    if (ipc_handle && !reachable) {
        void* mapped = nullptr;
        const hsa_status_t s = hsa_amd_ipc_memory_attach(ipc_handle, 4096, 1, &agent, &mapped);
        std::printf("engine: hsa_amd_ipc_memory_attach(signal page, R9700) -> %d at %p\n", int(s), mapped);
        if (s != HSA_STATUS_SUCCESS) fail("R9700 cannot map the IPC signal page");
        *gpu_mapping = mapped;
        return reinterpret_cast<long long*>(static_cast<char*>(mapped) + (reinterpret_cast<uintptr_t>(value) & 4095));
    }
    if (info.type == HSA_EXT_POINTER_TYPE_UNKNOWN) return value;  // not ROCr-tracked; host VA is the GPU VA
    if (!reachable) {
        const hsa_status_t s = hsa_amd_agents_allow_access(1, &agent, nullptr, info.hostBaseAddress);
        std::printf("engine: hsa_amd_agents_allow_access(R9700) -> %d\n", int(s));
        if (s != HSA_STATUS_SUCCESS) fail("R9700 cannot reach the signal memory");
    }
    if (!info.agentBaseAddress || !info.hostBaseAddress) return value;
    return reinterpret_cast<long long*>(static_cast<char*>(info.agentBaseAddress) +
                                        (reinterpret_cast<char*>(value) - static_cast<char*>(info.hostBaseAddress)));
}

struct Engine {
    AgentQuery agent;
    int hip_dev = -1;
    double rate_hz = 1e8;
    hipStream_t stream{};
    hsa_signal_t sig{};
    bool attached = false;
    void* gpu_mapping = nullptr;
    long long* sig_value_dev = nullptr;
    Ctl* c = nullptr;
    char* ctl_dev = nullptr;
    ClockFit fit;

    void setup(const Bdf& bdf, Ctl* ctl, hsa_signal_t local_sig, bool ipc, int64_t deadline_ms) {
        c = ctl;
        agent = find_agent(bdf, "R9700_BDF");
        hip_dev = hip_device_for(bdf, "R9700_BDF");
        HIPCK(hipSetDevice(hip_dev));
        int khz = 0;
        HIPCK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, hip_dev));
        rate_hz = khz * 1e3;
        HIPCK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        HIPCK(hipHostGetDevicePointer((void**)&ctl_dev, c, 0));
        if (ipc) {
            if (!wait_at_least(c->sig_ready, 1, c->abort, now_ns() + deadline_ms * 1000000))
                fail("engine: helper never published the IPC signal");
            HSACK(hsa_amd_ipc_signal_attach(&c->sig_handle, &sig));
            attached = true;
        } else {
            sig = local_sig;
        }
        sig_value_dev = signal_value_for(sig, agent.agent, ipc ? &c->sig_handle : nullptr, &gpu_mapping);
        std::printf("engine: R9700 %s %s rocr=%d hip=%d wallclock=%.0f Hz signal=%s\n", bdf.text.c_str(), agent.name,
                    agent.rocr_index, hip_dev, rate_hz, ipc ? "ipc-attached" : "local");
    }

    void run(const Options& o, const std::vector<Arm>& arms) {
        uint32_t seq = 0;
        for (size_t ai = 0; ai < arms.size(); ++ai) {
            const Arm& a = arms[ai];
            for (int i = 0; i < o.n; ++i) {
                ++seq;
                const int64_t deadline = now_ns() + (o.idle_us + o.gap_us + 2 * o.deadline_ms * 1000) * 1000;
                if (a.release == Release::Self) {
                    if (!wait_at_least(c->done, seq, c->abort, deadline)) return abort_run("helper self-release", a, i);
                    continue;
                }
                if (!wait_at_least(c->armed, seq, c->abort, deadline)) return abort_run("arming", a, i);
                sleep_us(o.gap_us);
                if (a.release == Release::Gpu) {
                    unsigned long long* ts =
                        reinterpret_cast<unsigned long long*>(dev_ptr(ctl_dev, c, &c->rel_ticks[ai][i]));
                    if (a.wait == Wait::Barrier) rad20_release_signal<<<1, 32, 0, stream>>>(sig_value_dev, ts);
                    else rad20_release_word<<<1, 32, 0, stream>>>(dev_ptr(ctl_dev, c, &c->word), seq, ts);
                    HIPCK(hipGetLastError());
                    if (!wait_hip_stream(stream, deadline)) return abort_run("R9700 release kernel", a, i);
                } else {
                    const int64_t t = now_ns();
                    if (a.wait == Wait::Barrier) hsa_signal_store_screlease(sig, 0);
                    else __atomic_store_n(&c->word, seq, __ATOMIC_RELEASE);
                    c->rel_host_ns[ai][i] = t;
                }
                if (!wait_at_least(c->done, seq, c->abort, deadline)) return abort_run("completion", a, i);
            }
        }
    }

    void abort_run(const char* stage, const Arm& a, int i) {
        if (!c->abort.load(std::memory_order_acquire))
            std::fprintf(stderr, "FAIL engine: arm %s iteration %d: deadline at %s\n", a.name.c_str(), i, stage);
        c->abort.store(1, std::memory_order_release);
    }

    void teardown() {
        if (gpu_mapping) HSACK(hsa_amd_ipc_memory_detach(gpu_mapping));
        if (attached) HSACK(hsa_signal_destroy(sig));
        HIPCK(hipStreamDestroy(stream));
    }
};

// ---------------------------------------------------------------------------------------------- report

static double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return NAN;
    size_t k = size_t(std::ceil(p * sorted.size()));
    k = std::min(std::max<size_t>(k, 1), sorted.size());
    return sorted[k - 1];
}

static const char* wait_name(Wait w) { return w == Wait::Barrier ? "aql-barrier-and" : "hipStreamWaitValue32"; }

static const char* release_name(Release r) {
    return r == Release::Gpu ? "r9700-kernel" : r == Release::Host ? "engine-host" : "helper-host";
}

static bool report(const Options& o, const std::vector<Arm>& arms, Ctl* c, const ClockFit& hx, const ClockFit& er) {
    std::printf("clock xtx:   err=+-%.2f us min_rtt=%.2f us drift=%+.1f ppm samples=%d\n", hx.err_ns / 1e3,
                hx.min_rtt_ns / 1e3, hx.ppm, hx.used);
    std::printf("clock r9700: err=+-%.2f us min_rtt=%.2f us drift=%+.1f ppm samples=%d\n", er.err_ns / 1e3,
                er.min_rtt_ns / 1e3, er.ppm, er.used);
    bool all = true;
    FILE* dump = nullptr;
    if (!o.dump.empty()) {
        dump = std::fopen(o.dump.c_str(), "w");
        if (!dump) { std::perror(o.dump.c_str()); std::exit(2); }
        std::fprintf(dump, "arm,iter,latency_us,armed_ns,release_ns,start_ns\n");
    }
    for (size_t ai = 0; ai < arms.size(); ++ai) {
        const Arm& a = arms[ai];
        std::vector<double> lat;
        int timeouts = 0;
        int max_iter = -1;
        double max_lat = -1e30;
        for (int i = 0; i < o.n; ++i) {
            timeouts += c->timed_out[ai][i];
            if (c->timed_out[ai][i] || !c->start_ticks[ai][i]) continue;
            double rel;
            if (a.release == Release::Gpu) {
                if (!c->rel_ticks[ai][i]) continue;
                rel = er.to_ns(c->rel_ticks[ai][i]);
            } else {
                rel = double(c->rel_host_ns[ai][i]);
            }
            const double start = hx.to_ns(c->start_ticks[ai][i]);
            lat.push_back((start - rel) / 1e3);
            if (lat.back() > max_lat) { max_lat = lat.back(); max_iter = i; }
            if (dump)
                std::fprintf(dump, "%s,%d,%.3f,%lld,%.0f,%.0f\n", a.name.c_str(), i, lat.back(),
                             (long long)c->armed_host_ns[ai][i], rel, start);
        }
        std::sort(lat.begin(), lat.end());
        const double err_us = (a.release == Release::Gpu ? hx.err_ns + er.err_ns : hx.err_ns) / 1e3;
        std::printf("arm=%-6s wait=%s release=%s n=%d ok=%zu timeouts=%d p50=%.2f p95=%.2f p99=%.2f min=%.2f "
                    "max=%.2f us max_iter=%d err=+-%.2f us\n",
                    a.name.c_str(), wait_name(a.wait), release_name(a.release), o.n, lat.size(), timeouts,
                    percentile(lat, 0.50), percentile(lat, 0.95), percentile(lat, 0.99),
                    lat.empty() ? NAN : lat.front(), lat.empty() ? NAN : lat.back(), max_iter, err_us);
        std::string why;
        if (int(lat.size()) != o.n)
            why = "only " + std::to_string(lat.size()) + "/" + std::to_string(o.n) + " completed";
        else if (lat.front() < -err_us) why = "negative latency beyond the clock error bound";
        const bool pass = why.empty();
        all &= pass;
        std::printf("%s rad20-barrier-wake mode=%s arm=%s gap_us=%lld idle_us=%lld n=%d%s%s\n", pass ? "PASS" : "FAIL",
                    o.mode.c_str(), a.name.c_str(), (long long)o.gap_us, (long long)o.idle_us, o.n,
                    pass ? "" : ": ", why.c_str());
    }
    if (dump) std::fclose(dump);
    return all;
}

// ---------------------------------------------------------------------------------------------- main

static std::string g_shm_to_unlink;

static void unlink_shm() {
    if (!g_shm_to_unlink.empty()) shm_unlink(g_shm_to_unlink.c_str());
}

static Ctl* map_ctl(const Options& o, int64_t deadline) {
    void* p = MAP_FAILED;
    if (o.mode == "single") {
        p = mmap(nullptr, kCtlBytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    } else {
        int fd = -1;
        if (o.mode == "helper") {
            shm_unlink(o.shm.c_str());
            fd = shm_open(o.shm.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
            if (fd < 0 || ftruncate(fd, kCtlBytes) != 0) { std::perror("shm"); std::exit(2); }
            g_shm_to_unlink = o.shm;
            std::atexit(unlink_shm);  // also on fail() paths
        } else {
            while ((fd = shm_open(o.shm.c_str(), O_RDWR, 0600)) < 0) {
                if (now_ns() > deadline) fail("engine: helper shm segment never appeared");
                sleep_us(1000);
            }
            struct stat st{};
            while (fstat(fd, &st) == 0 && size_t(st.st_size) < kCtlBytes) {
                if (now_ns() > deadline) fail("engine: helper shm segment never sized");
                sleep_us(1000);
            }
        }
        p = mmap(nullptr, kCtlBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
    }
    if (p == MAP_FAILED) { std::perror("mmap"); std::exit(2); }
    auto* c = static_cast<Ctl*>(p);
    if (o.mode == "helper") c->helper_pid.store(int32_t(getpid()), std::memory_order_release);
    return c;
}

int main(int argc, char** argv) {
    const Options o = parse_options(argc, argv);
    const std::vector<Arm> arms = parse_arms(o.arms);
    if (o.n < 1 || o.n > kMaxIter) fail("--n must be 1..4096");
    const Bdf xtx = env_bdf("XTX_BDF");
    const Bdf r9700 = env_bdf("R9700_BDF");
    const bool single = o.mode == "single";
    setvbuf(stdout, nullptr, _IOLBF, 0);

    HIPCK(hipInit(0));
    HSACK(hsa_init());
    const int64_t setup_deadline = now_ns() + o.deadline_ms * 5 * 1000000;
    Ctl* c = map_ctl(o, setup_deadline);
    HIPCK(hipHostRegister(c, kCtlBytes, hipHostRegisterPortable | hipHostRegisterMapped));
    std::printf("rad20-barrier-wake mode=%s arms=%s n=%d gap_us=%lld idle_us=%lld deadline_ms=%lld calib=%d\n",
                o.mode.c_str(), o.arms.c_str(), o.n, (long long)o.gap_us, (long long)o.idle_us,
                (long long)o.deadline_ms, o.calib);

    bool ok = true;
    if (single) {
        Helper h;
        Engine e;
        h.setup(xtx, false, c);
        e.setup(r9700, c, h.wait_sig, false, o.deadline_ms);
        std::vector<ClockSample> hpre, hpost, epre, epost;
        std::thread th([&] {
            HIPCK(hipSetDevice(h.hip_dev));
            hpre = calibrate(h.hip_dev, h.stream, o.calib, h.rate_hz, o.deadline_ms);
        });
        std::thread te([&] {
            HIPCK(hipSetDevice(e.hip_dev));
            epre = calibrate(e.hip_dev, e.stream, o.calib, e.rate_hz, o.deadline_ms);
        });
        th.join();
        te.join();
        th = std::thread([&] { HIPCK(hipSetDevice(h.hip_dev)); h.run(o, arms); });
        te = std::thread([&] { HIPCK(hipSetDevice(e.hip_dev)); e.run(o, arms); });
        th.join();
        te.join();
        th = std::thread([&] {
            HIPCK(hipSetDevice(h.hip_dev));
            hpost = calibrate(h.hip_dev, h.stream, o.calib, h.rate_hz, o.deadline_ms);
        });
        te = std::thread([&] {
            HIPCK(hipSetDevice(e.hip_dev));
            epost = calibrate(e.hip_dev, e.stream, o.calib, e.rate_hz, o.deadline_ms);
        });
        th.join();
        te.join();
        ok = report(o, arms, c, fit_clock(hpre, hpost, h.rate_hz), fit_clock(epre, epost, e.rate_hz));
        e.teardown();
        HIPCK(hipSetDevice(h.hip_dev));
        h.teardown();
    } else if (o.mode == "helper") {
        Helper h;
        h.setup(xtx, true, c);
        const auto pre = calibrate(h.hip_dev, h.stream, o.calib, h.rate_hz, o.deadline_ms);
        c->helper_ready.store(1, std::memory_order_release);
        std::printf("helper ready shm=%s\n", o.shm.c_str());
        if (!wait_at_least(c->engine_ready, 1, c->abort, now_ns() + o.deadline_ms * 10 * 1000000))
            fail("helper: engine process never became ready");
        h.run(o, arms);
        const auto post = calibrate(h.hip_dev, h.stream, o.calib, h.rate_hz, o.deadline_ms);
        // The engine detaches the IPC signal before it posts its fit, so the signal can be destroyed after this.
        if (!wait_at_least(c->engine_fit_ready, 1, now_ns() + o.deadline_ms * 10 * 1000000))
            fail("helper: engine never posted its clock fit");
        ClockFit er;
        er.a = c->engine_a;
        er.b = c->engine_b;
        er.err_ns = c->engine_err;
        er.ppm = c->engine_ppm;
        er.min_rtt_ns = c->engine_min_rtt;
        er.used = c->engine_used;
        ok = report(o, arms, c, fit_clock(pre, post, h.rate_hz), er);
        h.teardown();
    } else {
        // A stale segment from a dead helper can carry helper_ready; re-open until the live helper's segment is up.
        for (;;) {
            const bool ready = wait_at_least(c->helper_ready, 1, c->abort, now_ns() + 200 * 1000000LL);
            const int32_t pid = c->helper_pid.load(std::memory_order_acquire);
            if (ready && pid > 0 && kill(pid, 0) == 0) break;
            if (now_ns() > setup_deadline) fail("engine: no live helper segment");
            HIPCK(hipHostUnregister(c));
            munmap(c, kCtlBytes);
            sleep_us(10000);
            c = map_ctl(o, setup_deadline);
            HIPCK(hipHostRegister(c, kCtlBytes, hipHostRegisterPortable | hipHostRegisterMapped));
        }
        Engine e;
        e.setup(r9700, c, hsa_signal_t{}, true, o.deadline_ms);
        const auto pre = calibrate(e.hip_dev, e.stream, o.calib, e.rate_hz, o.deadline_ms);
        c->engine_ready.store(1, std::memory_order_release);
        e.run(o, arms);
        const auto post = calibrate(e.hip_dev, e.stream, o.calib, e.rate_hz, o.deadline_ms);
        const ClockFit f = fit_clock(pre, post, e.rate_hz);
        e.teardown();
        c->engine_a = f.a;
        c->engine_b = f.b;
        c->engine_err = f.err_ns;
        c->engine_ppm = f.ppm;
        c->engine_min_rtt = f.min_rtt_ns;
        c->engine_used = f.used;
        c->engine_fit_ready.store(1, std::memory_order_release);
        std::printf("engine done (results are reported by the helper)\n");
        ok = !c->abort.load();
    }

    HIPCK(hipHostUnregister(c));
    munmap(c, kCtlBytes);
    if (o.mode == "helper") shm_unlink(o.shm.c_str());
    hsa_shut_down();
    return ok ? 0 : 1;
}
