#include "moe_remote_shm.h"
#include "rad_moe_remote.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using namespace r3::moe_shm;
using HipMemcpy2D = int (*)(void*, size_t, const void*, size_t, size_t, size_t, int);
constexpr int kHipMemcpyDeviceToHost = 2;

struct HookState {
    int fd = -1;
    void* map = MAP_FAILED;
    size_t bytes = 0;
    Header* hdr = nullptr;
    HipMemcpy2D copy2d = nullptr;
    std::mutex submit_mu[kMaxRanks];
};
HookState g;

bool enabled() {
    const char* e = std::getenv("R3_MOE_HELPER");
    return e && *e && std::strcmp(e, "0") != 0;
}
const char* shm_name() {
    const char* e = std::getenv("R3_MOE_HELPER_SHM");
    return e && *e ? e : "/r3-moe-helper";
}

int owns_remote(void* user, int layer, int expert) {
    auto* s = static_cast<HookState*>(user);
    return s && s->hdr && layer >= 0 && expert >= 0 &&
           owns(s->map, (uint32_t)layer, (uint32_t)expert);
}

RadMoeRemoteHandle submit(void* user, int layer, int rank, int64_t rows, int64_t n_embd,
                          const RadMoeRemoteRoute* route, int64_t n_route,
                          const void* activation, int64_t activation_stride_bytes) {
    auto* s = static_cast<HookState*>(user);
    if (!s || !s->hdr || !route || !activation || rank < 0 || rank >= (int)kMaxRanks ||
        layer < 0 || (uint32_t)layer >= s->hdr->layer_count || rows <= 0 || rows > kMaxRows ||
        n_embd != kHidden || n_route <= 0 || n_route > kMaxRoutes ||
        activation_stride_bytes < (int64_t)kHidden * 2) return nullptr;

    /* The engine cannot have two remote blocks in flight on one rank: it waits/merges before the
     * next layer. Serialising submit per rank makes that protocol invariant explicit. */
    std::lock_guard<std::mutex> lock(s->submit_mu[rank]);
    uint32_t seq = fetch_add(&s->hdr->next_sequence, 1);
    if (seq == 0) return nullptr; /* 32-bit wrap is intentionally a hard stop in this diagnostic path. */
    Request* q = request(s->map, seq);
    if (seq > kRingSlots) wait_until(&q->completion_sequence, seq - kRingSlots);

    auto* rr = ptr<Route>(s->map, s->hdr->route_offset[rank]);
    static_assert(sizeof(Route) == sizeof(RadMoeRemoteRoute));
    std::memcpy(rr, route, static_cast<size_t>(n_route) * sizeof(Route));

    void* act = ptr<void>(s->map, s->hdr->activation_offset[rank]);
    const size_t row_bytes = kHidden * sizeof(uint16_t);
    if (!s->copy2d || s->copy2d(act, row_bytes, activation, (size_t)activation_stride_bytes,
                                row_bytes, (size_t)rows, kHipMemcpyDeviceToHost) != 0) return nullptr;

    q->status = 0; q->layer = (uint32_t)layer; q->rank = (uint32_t)rank;
    q->rank_count = kMaxRanks; q->rows = (uint32_t)rows; q->hidden = kHidden;
    q->route_count = (uint32_t)n_route; q->route_offset = s->hdr->route_offset[rank];
    q->activation_offset = s->hdr->activation_offset[rank]; q->activation_stride = (uint32_t)row_bytes;
    for (uint32_t r = 0; r < kMaxRanks; ++r) {
        q->result_offset[r] = s->hdr->result_offset[(uint32_t)rank * kMaxRanks + r];
        q->result_stride[r] = kHidden * sizeof(uint16_t);
    }
    store(&q->sequence, seq);
    store(&s->hdr->publish_sequence, seq);
    futex_wake(&s->hdr->publish_sequence);
    return reinterpret_cast<RadMoeRemoteHandle>((uintptr_t)seq);
}

const void* wait(void* user, RadMoeRemoteHandle handle) {
    auto* s = static_cast<HookState*>(user);
    const uint32_t seq = (uint32_t)(uintptr_t)handle;
    if (!s || !s->hdr || !seq) return nullptr;
    Request* q = request(s->map, seq);
    wait_until(&q->completion_sequence, seq);
    if (q->status != 0 || q->sequence != seq || q->rank >= kMaxRanks) return nullptr;
    return ptr<void>(s->map, q->result_offset[q->rank]);
}

void register_hook() __attribute__((constructor));
void register_hook() {
    if (!enabled()) return; /* Default path: no registration, therefore byte-identical engine behaviour. */
    /* R3_MOE_HELPER=1 is the switch for this component. Preserve an explicit engine threshold,
     * otherwise make patch 0002 hand every eligible owned route to the helper. */
    if (!std::getenv("R3_MOE_REMOTE_MIN_ROWS"))
        (void)::setenv("R3_MOE_REMOTE_MIN_ROWS", "0", 0);
    g.fd = ::shm_open(shm_name(), O_RDWR | O_CLOEXEC, 0);
    if (g.fd < 0) { std::fprintf(stderr, "libr3: R3_MOE_HELPER shm %s is not ready\n", shm_name()); return; }
    struct stat st{};
    if (::fstat(g.fd, &st) != 0 || st.st_size < (off_t)sizeof(Header)) return;
    g.bytes = (size_t)st.st_size;
    g.map = ::mmap(nullptr, g.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, g.fd, 0);
    if (g.map == MAP_FAILED || !valid(static_cast<Header*>(g.map), g.bytes)) {
        std::fprintf(stderr, "libr3: R3_MOE_HELPER shared-memory protocol mismatch\n"); return;
    }
    g.hdr = static_cast<Header*>(g.map);
    g.copy2d = reinterpret_cast<HipMemcpy2D>(::dlsym(RTLD_DEFAULT, "hipMemcpy2D"));
    auto reg = reinterpret_cast<int (*)(const RadMoeRemoteHooks*)>(::dlsym(RTLD_DEFAULT, "rad_moe_remote_register"));
    if (!g.copy2d || !reg) {
        std::fprintf(stderr, "libr3: R3_MOE_HELPER requires hipMemcpy2D and patch 0002 ABI\n"); return;
    }
    static RadMoeRemoteHooks hooks{RAD_MOE_REMOTE_ABI, &g, owns_remote, submit, wait};
    if (reg(&hooks) < 0) std::fprintf(stderr, "libr3: R3_MOE_HELPER hook registration refused\n");
}
} // namespace
