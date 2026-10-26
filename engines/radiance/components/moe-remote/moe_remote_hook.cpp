#include "moe_remote_hook.h"
#include "moe_remote_shm.h"
#include "rad_abi.h"

#include <hip/hip_runtime.h>

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <utility>
#include <cstdlib>
#include <cstring>
#include <ctime>
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

/* The engine's remote-MoE extension point (patch rad0002, abi/bc_moe_remote.h), mirrored here so libr3moe builds
 * against an unpatched radiance tree. The engine accepts the registration only if struct_size matches its own. */
struct EnginePlugin {
    uint32_t struct_size;
    void (*layer_map)(int layer, int n_expert, const int32_t* slot_of_expert);
    int (*owns_expert)(int layer, int expert);
    void (*burst)(int active);
};
using RegisterFn = int (*)(const EnginePlugin*);

struct HookState {
    int fd = -1;
    void* map = MAP_FAILED;
    size_t bytes = 0;
    Header* hdr = nullptr;
    bool pinned = false;
    RegisterFn engine_register = nullptr;
    std::mutex mu;
    RankState rank[kMaxRanks];
};
HookState g;

bool enabled_env() {
    const char* e = std::getenv("R3_MOE_WORKER");
    return e && *e && std::strcmp(e, "0") != 0;
}

bool poison_env() {
    const char* e = std::getenv("R3_MOE_WORKER_POISON");
    return e && *e && std::strcmp(e, "0") != 0;
}

bool shadow_env() {
    const char* e = std::getenv("R3_MOE_WORKER_SHADOW");
    return e && *e && std::strcmp(e, "0") != 0;
}

bool mask_only_env() {
    const char* e = std::getenv("R3_MOE_WORKER_MASK_ONLY");
    return e && *e && std::strcmp(e, "0") != 0;
}

bool cpu_ref_env() {
    const char* e = std::getenv("R3_MOE_WORKER_CPU");
    return e && *e && std::strcmp(e, "0") != 0;
}

const char* shm_name() {
    const char* e = std::getenv("R3_MOE_WORKER_SHM");
    return e && *e ? e : "/r3-moe-helper";
}

int64_t max_rows() {
    const char* e = std::getenv("R3_MOE_WORKER_MAX_ROWS");
    const long long n = e && *e ? std::atoll(e) : 64;
    return std::min<int64_t>(n > 0 ? n : 64, kMaxRows);
}

bool hip_ok(hipError_t e, const char* what) {
    if (e == hipSuccess) return true;
    std::fprintf(stderr, "libr3: R3_MOE_WORKER %s: %s\n", what, hipGetErrorString(e));
    return false;
}

/* R3_MOE_WORKER_LAYERS=MIN-MAX (diagnostic, RAD13): only layers in the range are served by the workers; every other layer's map is
 * published all-local (the same state as a layer with protected experts), so the engine runs it itself. Bisects a quality or
 * correctness difference by layer range without touching the engine's own path. Unset: every layer. */
bool layer_enabled(int layer) {
    static const std::pair<int, int> range = [] {
        const char* v = std::getenv("R3_MOE_WORKER_LAYERS");
        int lo = 0, hi = 1 << 30;
        if (v && *v && std::sscanf(v, "%d-%d", &lo, &hi) != 2) {
            std::fprintf(stderr, "libr3: R3_MOE_WORKER_LAYERS=%s: want MIN-MAX, using every layer\n", v);
            lo = 0; hi = 1 << 30;
        } else if (v && *v) {
            std::fprintf(stderr, "libr3: R3_MOE_WORKER_LAYERS on - workers serve layers %d to %d only\n", lo, hi);
        }
        return std::make_pair(lo, hi);
    }();
    return layer >= range.first && layer <= range.second;
}

void publish_layer_map(int layer, int n_expert, const int32_t* slot_of_expert) {
    static std::atomic<int> seen{0};
    if (seen.fetch_add(1) == 0)
        std::fprintf(stderr, "libr3: R3_MOE_WORKER first layer map: layer %d n_expert %d\n", layer, n_expert);
    if (!g.hdr || !g.pinned || !slot_of_expert || layer < 0 ||
        layer >= (int)g.hdr->layer_count || n_expert <= 0 ||
        n_expert > (int)g.hdr->expert_count) {
        static std::atomic<int> said{0};
        if (said.fetch_add(1) == 0)
            std::fprintf(stderr, "libr3: R3_MOE_WORKER layer map dropped: hdr %d pinned %d map %d layer %d of %u "
                         "n_expert %d\n", g.hdr != nullptr, (int)g.pinned, slot_of_expert != nullptr, layer,
                         g.hdr ? g.hdr->layer_count : 0u, n_expert);
        return;
    }
    std::lock_guard<std::mutex> lock(g.mu);
    int32_t* dst = layer_slots(g.map, (uint32_t)layer);
    if (!dst) return;
    if (load(&g.hdr->layer_map_ready[layer])) {
        if (std::memcmp(dst, slot_of_expert, (size_t)n_expert * sizeof(int32_t)) != 0)
            std::fprintf(stderr, "libr3: R3_MOE_WORKER layer %d runtime slot map changed across ranks\n",
                         layer);
    } else {
        std::memcpy(dst, slot_of_expert, (size_t)n_expert * sizeof(int32_t));
        for (int e = n_expert; e < (int)g.hdr->expert_count; ++e) dst[e] = -1;
        /* Layers with protected experts stay fully local for now: shadow still disagrees on exactly those
         * layers (34/44/46/47 on Flash-Next) after the runtime map. An all-local map means the helper owns
         * nothing there and publish sends nothing. */
        bool protected_layer = !layer_enabled(layer);   // outside R3_MOE_WORKER_LAYERS: kept local like a protected layer
        for (int e = 0; e < n_expert; ++e) protected_layer |= slot_of_expert[e] < 0;
        if (protected_layer) {
            for (int e = 0; e < (int)g.hdr->expert_count; ++e) dst[e] = -1;
            std::fprintf(stderr, "libr3: R3_MOE_WORKER layer %d has protected experts; kept local\n", layer);
        }
        store(&g.hdr->layer_map_ready[layer], 1);
        (void)futex_wake(&g.hdr->layer_map_ready[layer]);
    }
    wait_until(&g.hdr->layer_map_consumed[layer], 1);
}

/* An inference burst begins or ends (RAD20: the helpers arm GPU waits only inside a burst, so their queues are
 * empty and the GPUs idle between bursts). R3_MOE_WORKER_BURST_TRACE=1 logs each transition with a monotonic
 * timestamp, for checking begin/end pairing across single-stream, multi-stream and MTP runs. */
void burst_transition(int active) {
    static const bool trace = [] {
        const char* e = std::getenv("R3_MOE_WORKER_BURST_TRACE");
        return e && *e && std::strcmp(e, "0") != 0;
    }();
    static std::atomic<uint64_t> begins{0}, ends{0};
    const uint64_t n = active ? begins.fetch_add(1) + 1 : ends.fetch_add(1) + 1;
    /* One engine process drives the segment's burst word. Begin: wait for every helper's ack that its released waits
     * have drained (normally already given, so no sleep), then open the burst; the helpers arm their waits. End:
     * close it; the helpers drain and ack. A helper that stops acks kBurstParkedStop and sets helper_fault. */
    if (g.hdr) {
        Header* h = g.hdr;
        const uint32_t b = load(&h->burst) & ~kBurstStop;
        if (active && !(b & 1u)) {
            for (uint32_t i = 0; i < h->helper_count && i < kMaxRanks; ++i)
                for (uint32_t v; (v = load(&h->burst_parked[i])) != b;) {
                    if (v == kBurstParkedStop || load(&h->helper_fault)) {
                        std::fprintf(stderr, "libr3moe: helper %u stopped; burst not opened
", i);
                        return;
                    }
                    (void)futex_wait(&h->burst_parked[i], v);
                }
        }
        if (active == (int)(b & 1u)) return;
        (void)__atomic_fetch_add(&h->burst, 1u, __ATOMIC_ACQ_REL);
        (void)futex_wake(&h->burst);
    }
    if (trace) {
        timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        std::fprintf(stderr, "libr3moe: burst %s #%llu t=%lld.%06ld\n", active ? "begin" : "end",
                     (unsigned long long)n, (long long)ts.tv_sec, ts.tv_nsec / 1000);
    }
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
        std::fprintf(stderr, "libr3: R3_MOE_WORKER shm %s is not ready\n", shm_name());
        return;
    }
    struct stat st{};
    if (::fstat(g.fd, &st) != 0 || st.st_size < (off_t)sizeof(Header)) return;
    g.bytes = (size_t)st.st_size;
    g.map = ::mmap(nullptr, g.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, g.fd, 0);
    if (g.map == MAP_FAILED || !valid_self(static_cast<Header*>(g.map), g.bytes)) {
        std::fprintf(stderr, "libr3: R3_MOE_WORKER shared-memory protocol mismatch\n");
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
    g.engine_register = reinterpret_cast<RegisterFn>(dlsym(RTLD_DEFAULT, "bc_moe_remote_register"));
    if (const char* e = dlerror(); e || !g.engine_register) {
        std::fprintf(stderr,
            "libr3: R3_MOE_WORKER engine does not export bc_moe_remote_register (patch rad0002)%s%s\n",
            e ? ": " : "", e ? e : "");
        (void)hipHostUnregister(g.map);
        g.pinned = false;
        g.hdr = nullptr;
        return;
    }
    /* layer_map: the runtime slot maps (replayed for layers declared before this registration). owns_expert:
     * ownership for placement (R3_MOE_WORKER_SKIP_OWNED), the helper's bitmap as published before the engine
     * starts. The engine applies the protected-layer and slot tests itself, the same ones the helper uses when
     * it narrows ownership to the runtime maps, so both sides agree on what is never placed here. */
    static const EnginePlugin plugin{sizeof(EnginePlugin), publish_layer_map, owner_query, burst_transition};
    if (g.engine_register(&plugin) != 0) {
        std::fprintf(stderr, "libr3: R3_MOE_WORKER engine rejected the plugin registration (struct size)\n");
        (void)hipHostUnregister(g.map);
        g.pinned = false;
        g.hdr = nullptr;
        g.engine_register = nullptr;
        return;
    }

    /* The engine policy only decides whether the graph contains/uses the two mailbox ops.
     * All route selection and synchronisation stays on the rank GPU. */
    /* Explicit engine bounds are policy. Supply defaults only when the caller did not set them;
     * in particular R3_MOE_REMOTE_MAX_ROWS=0 must disable hand-off without disabling the helper. */
    (void)::setenv("R3_MOE_REMOTE_MIN_ROWS", "0", 0);
    const std::string mr = std::to_string((long long)max_rows());
    (void)::setenv("R3_MOE_REMOTE_MAX_ROWS", mr.c_str(), 0);
    std::fprintf(stderr,
        "libr3: R3_MOE_WORKER GPU mailbox ready, %u owned experts, decode rows <= %lld\n",
        g.hdr->owned_count, (long long)max_rows());
    if (poison_env()) std::fprintf(stderr, "libr3: R3_MOE_WORKER poison\n");
    if (shadow_env()) std::fprintf(stderr, "libr3: R3_MOE_WORKER_SHADOW enabled\n");
    if (mask_only_env()) std::fprintf(stderr, "libr3: R3_MOE_WORKER_MASK_ONLY enabled\n");
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
            "libr3: R3_MOE_WORKER_SHADOW layer=%u rank=%u rows=%llu mean_abs=%.8g "
            "max_abs=%.8g rel_l2=%.8g bad_rows=%u\n",
            l, rank, (unsigned long long)x.rows,
            (double)x.sum_abs / (double)x.elems, (double)max_abs, rel, x.bad_rows);
        const auto& f = first[l];
        if (f.claimed) {
            std::fprintf(stderr,
                "libr3: R3_MOE_WORKER_SHADOW first_bad layer=%u rank=%u r0=%u token=%d "
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

/* R3_MOE_WORKER_HISTOGRAM=<path>: this rank's route counts, uint32[kMaxLayers][Header::expert_count], written at shutdown
 * to <path>.rank<r>. A helper given it as R3_MOE_WORKER_HEAT owns the coldest experts first. */
void write_heat(uint32_t rank, RankState& s) {
    const char* path = std::getenv("R3_MOE_WORKER_HISTOGRAM");
    if (!path || !*path || !s.dev.heat) return;
    std::vector<uint32_t> heat((size_t)kMaxLayers * g.hdr->expert_count);
    if (!hip_ok(hipDeviceSynchronize(), "sync before route histogram") ||
        !hip_ok(hipMemcpy(heat.data(), s.dev.heat, heat.size() * sizeof(uint32_t), hipMemcpyDeviceToHost),
                "read route histogram"))
        return;
    const std::string out = std::string(path) + ".rank" + std::to_string(rank);
    if (FILE* f = std::fopen(out.c_str(), "wb")) {
        const size_t n = std::fwrite(heat.data(), sizeof(uint32_t), heat.size(), f);
        std::fclose(f);
        std::fprintf(stderr, "libr3: R3_MOE_WORKER route histogram %s (%zu counts)\n", out.c_str(), n);
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
    if (g.engine_register) (void)g.engine_register(nullptr);
    g.engine_register = nullptr;
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
        std::fprintf(stderr, "libr3: R3_MOE_WORKER %u helpers need a one-rank engine (world %d)\n",
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
    if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&s.sequence), kLanes * sizeof(uint32_t)),
                "hipMalloc lane sequence counters") ||
        !hip_ok(hipMemset(s.sequence, 0, kLanes * sizeof(uint32_t)), "zero lane sequence counters") ||
        !hip_ok(hipMalloc(reinterpret_cast<void**>(&s.published), (kMaxRanks + 1) * sizeof(uint32_t)),
                "hipMalloc published flags") ||
        !hip_ok(hipMemset(s.published, 0, (kMaxRanks + 1) * sizeof(uint32_t)), "zero published flags") ||
        !hip_ok(hipMalloc(&s.maps, (size_t)max_rows() * g.hdr->top_k * (sizeof(int32_t) + sizeof(uint32_t))),
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
        const size_t bytes = (size_t)row_cap * g.hdr->top_k * g.hdr->hidden * sizeof(uint16_t);
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
    s.dev.rmask = reinterpret_cast<uint32_t*>(s.dev.inv + (size_t)row_cap * g.hdr->top_k);
    s.dev.shadow_stats = shadow_stats;
    s.dev.shadow_first = shadow_first;
    s.dev.max_rows = row_cap;
    s.dev.hidden = g.hdr->hidden;
    s.dev.top_k = g.hdr->top_k;
    s.dev.experts = g.hdr->expert_count;
    s.dev.rank = (uint32_t)rank;
    s.dev.world = (uint32_t)world_size;
    if (const char* hp = std::getenv("R3_MOE_WORKER_HISTOGRAM"); hp && *hp) {
        const size_t hb = (size_t)kMaxLayers * g.hdr->expert_count * sizeof(uint32_t);
        if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&s.dev.heat), hb), "hipMalloc route histogram") ||
            !hip_ok(hipMemset(s.dev.heat, 0, hb), "zero route histogram"))
            s.dev.heat = nullptr;
    }
    s.dev.helpers = helpers;
    s.dev.shadow = shadow ? 1u : 0u;
    /* 1: add the engine's captured rows; 2: same late publish/mask/rescatter, but add the helper's rows. */
    {
        const char* e = std::getenv("R3_MOE_WORKER_MASK_ONLY");
        s.dev.mask_only = !mask_only ? 0u : (e && std::strcmp(e, "2") == 0 ? 2u : 1u);
    }
    s.dev.cpu_ref = cpu_ref_env() ? 1u : 0u;
    /* R3_MOE_WORKER_COMBINE=1 (default off): two helpers, no shadow/mask-only, and helper 0 started its combiner
     * (Header::combine.enabled, checked again by the publish kernel each step). */
    {
        const char* e = std::getenv("R3_MOE_WORKER_COMBINE");
        s.dev.combine = (e && std::strcmp(e, "1") == 0 && helpers == 2 && !shadow && !mask_only) ? 1u : 0u;
        if (s.dev.combine)
            std::fprintf(stderr, "libr3moe: R3_MOE_WORKER_COMBINE on - helper 0 combines the two helpers' rows on the large lane\n");
    }
    s.ready = true;
    *out = &s.dev;
    return RAD_OK;
}
