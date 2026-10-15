#include "moe_remote_hook.h"
#include "moe_remote_shm.h"
#include "rad_abi.h"

#include <hip/hip_runtime.h>

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <vector>
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
    uint16_t* mask_ref = nullptr;
    void* maps = nullptr;   /* inv + rmask, one allocation */
    bool ready = false;
};

using LayerMapHook = void (*)(int, int, const int32_t*);
using SetLayerMapHook = void (*)(LayerMapHook);

struct HookState {
    int fd = -1;
    void* map = MAP_FAILED;
    size_t bytes = 0;
    Header* hdr = nullptr;
    bool pinned = false;
    SetLayerMapHook set_layer_map = nullptr;
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

bool mask_only_env() {
    const char* e = std::getenv("R3_MOE_HELPER_MASK_ONLY");
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
    return std::min<int64_t>(n > 0 ? n : 64, kMaxRows);
}

bool hip_ok(hipError_t e, const char* what) {
    if (e == hipSuccess) return true;
    std::fprintf(stderr, "libr3: R3_MOE_HELPER %s: %s\n", what, hipGetErrorString(e));
    return false;
}

void publish_layer_map(int layer, int n_expert, const int32_t* slot_of_expert) {
    static std::atomic<int> seen{0};
    if (seen.fetch_add(1) == 0)
        std::fprintf(stderr, "libr3: R3_MOE_HELPER first layer map: layer %d n_expert %d\n", layer, n_expert);
    if (!g.hdr || !g.pinned || !slot_of_expert || layer < 0 ||
        layer >= (int)g.hdr->layer_count || n_expert <= 0 ||
        n_expert > (int)kExperts) {
        static std::atomic<int> said{0};
        if (said.fetch_add(1) == 0)
            std::fprintf(stderr, "libr3: R3_MOE_HELPER layer map dropped: hdr %d pinned %d map %d layer %d of %u "
                         "n_expert %d\n", g.hdr != nullptr, (int)g.pinned, slot_of_expert != nullptr, layer,
                         g.hdr ? g.hdr->layer_count : 0u, n_expert);
        return;
    }
    std::lock_guard<std::mutex> lock(g.mu);
    int32_t* dst = layer_slots(g.map, (uint32_t)layer);
    if (!dst) return;
    if (load(&g.hdr->layer_map_ready[layer])) {
        if (std::memcmp(dst, slot_of_expert, (size_t)n_expert * sizeof(int32_t)) != 0)
            std::fprintf(stderr, "libr3: R3_MOE_HELPER layer %d runtime slot map changed across ranks\n",
                         layer);
    } else {
        std::memcpy(dst, slot_of_expert, (size_t)n_expert * sizeof(int32_t));
        for (int e = n_expert; e < (int)kExperts; ++e) dst[e] = -1;
        /* Layers with protected experts stay fully local for now: shadow still disagrees on exactly those
         * layers (34/44/46/47 on Flash-Next) after the runtime map. An all-local map means the helper owns
         * nothing there and publish sends nothing. */
        bool protected_layer = false;
        for (int e = 0; e < n_expert; ++e) protected_layer |= slot_of_expert[e] < 0;
        if (protected_layer) {
            for (int e = 0; e < (int)kExperts; ++e) dst[e] = -1;
            std::fprintf(stderr, "libr3: R3_MOE_HELPER layer %d has protected experts; kept local\n", layer);
        }
        store(&g.hdr->layer_map_ready[layer], 1);
        (void)futex_wake(&g.hdr->layer_map_ready[layer]);
    }
    wait_until(&g.hdr->layer_map_consumed[layer], 1);
}

int owner_query(int layer, int expert) {
    if (!g.hdr || layer < 0 || expert < 0) return 0;
    for (uint32_t h = 0; h < g.hdr->helper_count && h < kMaxRanks; ++h)
        if (owns(g.map, (uint32_t)layer, (uint32_t)expert, h)) return 1;
    return 0;
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
    dlerror();
    g.set_layer_map = reinterpret_cast<SetLayerMapHook>(
        dlsym(RTLD_DEFAULT, "rad_moe_remote_set_layer_map_hook"));
    if (const char* e = dlerror(); e || !g.set_layer_map) {
        std::fprintf(stderr,
            "libr3: R3_MOE_HELPER engine does not export rad_moe_remote_set_layer_map_hook%s%s\n",
            e ? ": " : "", e ? e : "");
        (void)hipHostUnregister(g.map);
        g.pinned = false;
        g.hdr = nullptr;
        return;
    }
    g.set_layer_map(publish_layer_map);
    /* Ownership for placement (R3_MOE_HELPER_SKIP_OWNED): the helper's bitmap as published before the engine
     * starts. The engine applies the protected-layer and slot tests itself, the same ones the helper uses
     * when it narrows ownership to the runtime maps, so both sides agree on what is never placed here. */
    if (auto set_owner = reinterpret_cast<void (*)(int (*)(int, int))>(
            dlsym(RTLD_DEFAULT, "rad_moe_remote_set_owner_hook")))
        set_owner(owner_query);

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
    if (mask_only_env()) std::fprintf(stderr, "libr3: R3_MOE_HELPER_MASK_ONLY enabled\n");
}

void report_shadow(uint32_t rank, RankState& s) {
    if (!s.dev.shadow || !s.dev.shadow_stats || !s.dev.shadow_first) return;
    (void)hipDeviceSynchronize();
    R3MoeShadowLayerStats stats[kMaxLayers]{};
    R3MoeShadowFirst first[kMaxLayers]{};
    if (!hip_ok(hipMemcpy(stats, s.dev.shadow_stats, sizeof(stats), hipMemcpyDeviceToHost),
                "read shadow layer stats") ||
        !hip_ok(hipMemcpy(first, s.dev.shadow_first, sizeof(first), hipMemcpyDeviceToHost),
                "read shadow first mismatches"))
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
        const auto& f = first[l];
        if (f.claimed) {
            std::fprintf(stderr,
                "libr3: R3_MOE_HELPER_SHADOW first_bad layer=%u rank=%u r0=%u token=%d "
                "slot=%u expert=%d rel_l2=%.8g max_abs=%.8g "
                "local=[%.8g %.8g %.8g %.8g] helper=[%.8g %.8g %.8g %.8g]\n",
                f.layer, f.rank, f.row_base, f.token, (unsigned)f.slot,
                f.expert, (double)f.rel_l2, (double)f.max_abs,
                (double)f.local[0], (double)f.local[1],
                (double)f.local[2], (double)f.local[3],
                (double)f.helper[0], (double)f.helper[1],
                (double)f.helper[2], (double)f.helper[3]);
        }
    }
}

/* R3_MOE_HELPER_HISTOGRAM=<path>: this rank's route counts, uint32[kMaxLayers][kExperts], written at shutdown
 * to <path>.rank<r>. A helper given it as R3_MOE_HELPER_HEAT owns the coldest experts first. */
void write_heat(uint32_t rank, RankState& s) {
    const char* path = std::getenv("R3_MOE_HELPER_HISTOGRAM");
    if (!path || !*path || !s.dev.heat) return;
    std::vector<uint32_t> heat((size_t)kMaxLayers * kExperts);
    if (!hip_ok(hipDeviceSynchronize(), "sync before route histogram") ||
        !hip_ok(hipMemcpy(heat.data(), s.dev.heat, heat.size() * sizeof(uint32_t), hipMemcpyDeviceToHost),
                "read route histogram"))
        return;
    const std::string out = std::string(path) + ".rank" + std::to_string(rank);
    if (FILE* f = std::fopen(out.c_str(), "wb")) {
        const size_t n = std::fwrite(heat.data(), sizeof(uint32_t), heat.size(), f);
        std::fclose(f);
        std::fprintf(stderr, "libr3: R3_MOE_HELPER route histogram %s (%zu counts)\n", out.c_str(), n);
    }
}

void close_mailbox() {
    std::lock_guard<std::mutex> lock(g.mu);
    for (uint32_t r = 0; r < kMaxRanks; ++r) {
        RankState& s = g.rank[r];
        if (!s.ready) continue;
        if (s.device >= 0) (void)hipSetDevice(s.device);
        report_shadow(r, s);
        write_heat(r, s);
        if (s.dev.heat) (void)hipFree(s.dev.heat);
        if (s.dev.shadow_first) (void)hipFree(s.dev.shadow_first);
        if (s.dev.shadow_stats) (void)hipFree(s.dev.shadow_stats);
        if (s.mask_ref) (void)hipFree(s.mask_ref);
        if (s.maps) (void)hipFree(s.maps);
        if (s.published) (void)hipFree(s.published);
        if (s.sequence) (void)hipFree(s.sequence);
        if (s.owner) (void)hipFree(s.owner);
        s = RankState{};
    }
    if (g.set_layer_map) g.set_layer_map(nullptr);
    g.set_layer_map = nullptr;
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
    if (!out || !g.hdr || !g.pinned || world_size < 1 || world_size > (int)kMaxRanks ||
        rank < 0 || rank >= world_size)
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
    const uint32_t helpers = g.hdr->helper_count;
    if (helpers < 1 || helpers > kMaxRanks || (helpers > 1 && world_size != 1)) {
        std::fprintf(stderr, "libr3: R3_MOE_HELPER %u helpers need a one-rank engine (world %d)\n",
                     helpers, world_size);
        return RAD_E_STATE;
    }
    if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&s.owner), (size_t)kMaxRanks * g.hdr->owner_slot_bytes),
                "hipMalloc ownership bitmap"))
        return RAD_E_NOMEM;
    if (!hip_ok(hipMemcpy(s.owner, owner_bits(g.map, 0), (size_t)kMaxRanks * g.hdr->owner_slot_bytes,
                          hipMemcpyHostToDevice), "upload ownership bitmap")) {
        (void)hipFree(s.owner);
        s.owner = nullptr;
        return RAD_E_DEVICE;
    }
    if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&s.sequence), 2 * kMaxRanks * sizeof(uint32_t)),
                "hipMalloc lane sequence counters") ||
        !hip_ok(hipMemset(s.sequence, 0, 2 * kMaxRanks * sizeof(uint32_t)), "zero lane sequence counters") ||
        !hip_ok(hipMalloc(reinterpret_cast<void**>(&s.published), kMaxRanks * sizeof(uint32_t)),
                "hipMalloc published flags") ||
        !hip_ok(hipMemset(s.published, 0, kMaxRanks * sizeof(uint32_t)), "zero published flags") ||
        !hip_ok(hipMalloc(&s.maps, (size_t)max_rows() * kTopK * (sizeof(int32_t) + sizeof(uint32_t))),
                "hipMalloc merge slot maps")) {
        if (s.maps) (void)hipFree(s.maps);
        if (s.published) (void)hipFree(s.published);
        if (s.sequence) (void)hipFree(s.sequence);
        (void)hipFree(s.owner);
        s.maps = nullptr;
        s.published = nullptr;
        s.sequence = nullptr;
        s.owner = nullptr;
        return RAD_E_DEVICE;
    }

    const bool shadow = shadow_env();
    const bool mask_only = mask_only_env();
    const uint32_t row_cap = (uint32_t)max_rows();
    if (mask_only) {
        const size_t bytes = (size_t)row_cap * kTopK * kHidden * sizeof(uint16_t);
        if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&s.mask_ref), bytes),
                    "hipMalloc mask-only engine rows")) {
            (void)hipFree(s.maps);
            (void)hipFree(s.published);
            (void)hipFree(s.sequence);
            (void)hipFree(s.owner);
            s.maps = nullptr;
            s.published = nullptr;
            s.sequence = nullptr;
            s.owner = nullptr;
            return RAD_E_NOMEM;
        }
    }
    R3MoeShadowLayerStats* shadow_stats = nullptr;
    R3MoeShadowFirst* shadow_first = nullptr;
    if (shadow &&
        (!hip_ok(hipMalloc(reinterpret_cast<void**>(&shadow_stats),
                           sizeof(R3MoeShadowLayerStats) * kMaxLayers),
                 "hipMalloc shadow stats") ||
         !hip_ok(hipMemset(shadow_stats, 0, sizeof(R3MoeShadowLayerStats) * kMaxLayers),
                 "zero shadow stats") ||
         !hip_ok(hipMalloc(reinterpret_cast<void**>(&shadow_first),
                           sizeof(R3MoeShadowFirst) * kMaxLayers),
                 "hipMalloc shadow first") ||
         !hip_ok(hipMemset(shadow_first, 0, sizeof(R3MoeShadowFirst) * kMaxLayers),
                 "zero shadow first"))) {
        if (shadow_first) (void)hipFree(shadow_first);
        if (shadow_stats) (void)hipFree(shadow_stats);
        if (s.mask_ref) (void)hipFree(s.mask_ref);
        s.mask_ref = nullptr;
        (void)hipFree(s.maps);
        s.maps = nullptr;
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
    s.dev.mask_ref = s.mask_ref;
    s.dev.inv = static_cast<int32_t*>(s.maps);
    s.dev.rmask = reinterpret_cast<uint32_t*>(s.dev.inv + (size_t)row_cap * kTopK);
    s.dev.shadow_stats = shadow_stats;
    s.dev.shadow_first = shadow_first;
    s.dev.max_rows = row_cap;
    s.dev.rank = (uint32_t)rank;
    s.dev.world = (uint32_t)world_size;
    if (const char* hp = std::getenv("R3_MOE_HELPER_HISTOGRAM"); hp && *hp) {
        const size_t hb = (size_t)kMaxLayers * kExperts * sizeof(uint32_t);
        if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&s.dev.heat), hb), "hipMalloc route histogram") ||
            !hip_ok(hipMemset(s.dev.heat, 0, hb), "zero route histogram"))
            s.dev.heat = nullptr;
    }
    s.dev.helpers = helpers;
    s.dev.shadow = shadow ? 1u : 0u;
    /* 1: add the engine's captured rows; 2: same late publish/mask/rescatter, but add the helper's rows. */
    {
        const char* e = std::getenv("R3_MOE_HELPER_MASK_ONLY");
        s.dev.mask_only = !mask_only ? 0u : (e && std::strcmp(e, "2") == 0 ? 2u : 1u);
    }
    s.dev.cpu_ref = cpu_ref_env() ? 1u : 0u;
    s.ready = true;
    *out = &s.dev;
    return RAD_OK;
}
