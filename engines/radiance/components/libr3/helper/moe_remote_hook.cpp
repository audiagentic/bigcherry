#include "moe_remote_hook.h"
#include "moe_remote_shm.h"
#include "rad_abi.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using namespace r3::moe_shm;

struct RankState {
    R3MoeMailboxDevice dev{};
    int device = -1;
    uint8_t* owner = nullptr;
    uint32_t* sequence = nullptr;
    uint32_t* published = nullptr;
    bool ready = false;
};

struct HookState {
    int fd = -1;
    void* map = MAP_FAILED;
    size_t bytes = 0;
    Header* hdr = nullptr;
    bool pinned = false;
    std::mutex mu;
    RankState rank[kMaxRanks];
};
HookState g;

bool enabled_env() {
    const char* e = std::getenv("R3_MOE_HELPER");
    return e && *e && std::strcmp(e, "0") != 0;
}

bool poison_env() {
    const char* e = std::getenv("R3_MOE_HELPER_POISON");
    return e && *e && std::strcmp(e, "0") != 0;
}

bool shadow_env() {
    const char* e = std::getenv("R3_MOE_HELPER_SHADOW");
    return e && *e && std::strcmp(e, "0") != 0;
}

bool cpu_ref_env() {
    const char* e = std::getenv("R3_MOE_HELPER_CPU");
    return e && *e && std::strcmp(e, "0") != 0;
}

const char* shm_name() {
    const char* e = std::getenv("R3_MOE_HELPER_SHM");
    return e && *e ? e : "/r3-moe-helper";
}

int64_t max_rows() {
    const char* e = std::getenv("R3_MOE_HELPER_MAX_ROWS");
    const long long n = e && *e ? std::atoll(e) : 64;
    return std::min<int64_t>(n > 0 ? n : 64, kMaxRows / kTopK);
}

bool hip_ok(hipError_t e, const char* what) {
    if (e == hipSuccess) return true;
    std::fprintf(stderr, "libr3: R3_MOE_HELPER %s: %s\n", what, hipGetErrorString(e));
    return false;
}

void register_mailbox() {
    if (!enabled_env()) return;
    g.fd = ::shm_open(shm_name(), O_RDWR | O_CLOEXEC, 0);
    if (g.fd < 0) {
        std::fprintf(stderr, "libr3: R3_MOE_HELPER shm %s is not ready\n", shm_name());
        return;
    }
    struct stat st{};
    if (::fstat(g.fd, &st) != 0 || st.st_size < (off_t)sizeof(Header)) return;
    g.bytes = (size_t)st.st_size;
    g.map = ::mmap(nullptr, g.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, g.fd, 0);
    if (g.map == MAP_FAILED || !valid(static_cast<Header*>(g.map), g.bytes)) {
        std::fprintf(stderr, "libr3: R3_MOE_HELPER shared-memory protocol mismatch\n");
        return;
    }
    g.hdr = static_cast<Header*>(g.map);
    if (!hip_ok(hipHostRegister(g.map, g.bytes, hipHostRegisterPortable | hipHostRegisterMapped),
                "could not pin/map shared segment")) {
        g.hdr = nullptr;
        return;
    }
    g.pinned = true;

    /* The engine policy only decides whether the graph contains/uses the two mailbox ops.
     * All route selection and synchronisation stays on the rank GPU. */
    /* Explicit engine bounds are policy. Supply defaults only when the caller did not set them;
     * in particular R3_MOE_REMOTE_MAX_ROWS=0 must disable hand-off without disabling the helper. */
    (void)::setenv("R3_MOE_REMOTE_MIN_ROWS", "0", 0);
    const std::string mr = std::to_string((long long)max_rows());
    (void)::setenv("R3_MOE_REMOTE_MAX_ROWS", mr.c_str(), 0);
    std::fprintf(stderr,
        "libr3: R3_MOE_HELPER GPU mailbox ready, %u owned experts, decode rows <= %lld\n",
        g.hdr->owned_count, (long long)max_rows());
    if (poison_env()) std::fprintf(stderr, "libr3: R3_MOE_HELPER poison\n");
    if (shadow_env()) std::fprintf(stderr, "libr3: R3_MOE_HELPER_SHADOW enabled\n");
}

void report_shadow(uint32_t rank, RankState& s) {
    if (!s.dev.shadow || !s.dev.shadow_stats || !s.dev.shadow_first) return;
    (void)hipDeviceSynchronize();
    R3MoeShadowLayerStats stats[kMaxLayers]{};
    R3MoeShadowFirst first{};
    if (!hip_ok(hipMemcpy(stats, s.dev.shadow_stats, sizeof(stats), hipMemcpyDeviceToHost),
                "read shadow layer stats") ||
        !hip_ok(hipMemcpy(&first, s.dev.shadow_first, sizeof(first), hipMemcpyDeviceToHost),
                "read shadow first mismatch"))
        return;
    for (uint32_t l = 0; l < kMaxLayers; ++l) {
        const auto& x = stats[l];
        if (!x.elems) continue;
        float max_abs = 0.0f;
        std::memcpy(&max_abs, &x.max_abs_bits, sizeof(max_abs));
        const double rel = std::sqrt((double)x.diff2 / std::max(1.0e-30, (double)x.ref2));
        std::fprintf(stderr,
            "libr3: R3_MOE_HELPER_SHADOW layer=%u rank=%u rows=%llu mean_abs=%.8g "
            "max_abs=%.8g rel_l2=%.8g bad_rows=%u\n",
            l, rank, (unsigned long long)x.rows,
            (double)x.sum_abs / (double)x.elems, (double)max_abs, rel, x.bad_rows);
    }
    if (first.claimed) {
        std::fprintf(stderr,
            "libr3: R3_MOE_HELPER_SHADOW first_bad layer=%u rank=%u r0=%u token=%d "
            "slot=%u expert=%d rel_l2=%.8g max_abs=%.8g "
            "local=[%.8g %.8g %.8g %.8g] helper=[%.8g %.8g %.8g %.8g]\n",
            first.layer, first.rank, first.row_base, first.token, (unsigned)first.slot,
            first.expert, (double)first.rel_l2, (double)first.max_abs,
            (double)first.local[0], (double)first.local[1],
            (double)first.local[2], (double)first.local[3],
            (double)first.helper[0], (double)first.helper[1],
            (double)first.helper[2], (double)first.helper[3]);
    }
}

void close_mailbox() {
    std::lock_guard<std::mutex> lock(g.mu);
    for (uint32_t r = 0; r < kMaxRanks; ++r) {
        RankState& s = g.rank[r];
        if (!s.ready) continue;
        if (s.device >= 0) (void)hipSetDevice(s.device);
        report_shadow(r, s);
        if (s.dev.shadow_first) (void)hipFree(s.dev.shadow_first);
        if (s.dev.shadow_stats) (void)hipFree(s.dev.shadow_stats);
        if (s.published) (void)hipFree(s.published);
        if (s.sequence) (void)hipFree(s.sequence);
        if (s.owner) (void)hipFree(s.owner);
        s = RankState{};
    }
    if (g.pinned) (void)hipHostUnregister(g.map);
    if (g.map != MAP_FAILED) ::munmap(g.map, g.bytes);
    if (g.fd >= 0) ::close(g.fd);
    g.hdr = nullptr;
    g.map = MAP_FAILED;
    g.fd = -1;
    g.pinned = false;
}

struct HookLifetime {
    HookLifetime() { register_mailbox(); }
    ~HookLifetime() { close_mailbox(); }
} g_lifetime;

}  // namespace

extern "C" int r3_moe_mailbox_enabled(void) {
    return g.hdr && g.pinned;
}

extern "C" int r3_moe_mailbox_device(int rank, int world_size, R3MoeMailboxDevice** out) {
    if (!out || !g.hdr || !g.pinned || world_size != (int)kMaxRanks ||
        rank < 0 || rank >= (int)kMaxRanks)
        return RAD_E_INVAL;

    std::lock_guard<std::mutex> lock(g.mu);
    RankState& s = g.rank[rank];
    int device = -1;
    if (!hip_ok(hipGetDevice(&device), "hipGetDevice")) return RAD_E_DEVICE;
    if (s.ready) {
        if (s.device != device) return RAD_E_STATE;
        *out = &s.dev;
        return RAD_OK;
    }

    void* dev_map = nullptr;
    if (!hip_ok(hipHostGetDevicePointer(&dev_map, g.map, 0), "hipHostGetDevicePointer"))
        return RAD_E_DEVICE;
    if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&s.owner), g.hdr->owner_slot_bytes),
                "hipMalloc ownership bitmap"))
        return RAD_E_NOMEM;
    if (!hip_ok(hipMemcpy(s.owner, owner_bits(g.map), g.hdr->owner_slot_bytes,
                          hipMemcpyHostToDevice), "upload ownership bitmap")) {
        (void)hipFree(s.owner);
        s.owner = nullptr;
        return RAD_E_DEVICE;
    }
    if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&s.sequence), sizeof(uint32_t)),
                "hipMalloc sequence counter") ||
        !hip_ok(hipMemset(s.sequence, 0, sizeof(uint32_t)), "zero sequence counter") ||
        !hip_ok(hipMalloc(reinterpret_cast<void**>(&s.published), sizeof(uint32_t)),
                "hipMalloc published flag") ||
        !hip_ok(hipMemset(s.published, 0, sizeof(uint32_t)), "zero published flag")) {
        if (s.published) (void)hipFree(s.published);
        if (s.sequence) (void)hipFree(s.sequence);
        (void)hipFree(s.owner);
        s.published = nullptr;
        s.sequence = nullptr;
        s.owner = nullptr;
        return RAD_E_DEVICE;
    }

    const bool shadow = shadow_env();
    R3MoeShadowLayerStats* shadow_stats = nullptr;
    R3MoeShadowFirst* shadow_first = nullptr;
    if (shadow &&
        (!hip_ok(hipMalloc(reinterpret_cast<void**>(&shadow_stats),
                           sizeof(R3MoeShadowLayerStats) * kMaxLayers),
                 "hipMalloc shadow stats") ||
         !hip_ok(hipMemset(shadow_stats, 0, sizeof(R3MoeShadowLayerStats) * kMaxLayers),
                 "zero shadow stats") ||
         !hip_ok(hipMalloc(reinterpret_cast<void**>(&shadow_first), sizeof(R3MoeShadowFirst)),
                 "hipMalloc shadow first") ||
         !hip_ok(hipMemset(shadow_first, 0, sizeof(R3MoeShadowFirst)),
                 "zero shadow first"))) {
        if (shadow_first) (void)hipFree(shadow_first);
        if (shadow_stats) (void)hipFree(shadow_stats);
        (void)hipFree(s.published);
        (void)hipFree(s.sequence);
        (void)hipFree(s.owner);
        s.published = nullptr;
        s.sequence = nullptr;
        s.owner = nullptr;
        return RAD_E_DEVICE;
    }

    s.device = device;
    s.dev.mailbox = dev_map;
    s.dev.owner = s.owner;
    s.dev.sequence = s.sequence;
    s.dev.published = s.published;
    s.dev.shadow_stats = shadow_stats;
    s.dev.shadow_first = shadow_first;
    s.dev.max_rows = (uint32_t)max_rows();
    s.dev.rank = (uint32_t)rank;
    s.dev.shadow = shadow ? 1u : 0u;
    s.dev.cpu_ref = cpu_ref_env() ? 1u : 0u;
    s.ready = true;
    *out = &s.dev;
    return RAD_OK;
}
