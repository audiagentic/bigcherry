/* moe_remote.cpp -- residency policy and C hand-off for remote whole-expert compute. */
#include "rad_moe_remote.h"
#include "runtime/ctx.h"
#include "rad_device.h"
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <vector>
namespace {
RadMoeRemoteHooks g_hooks{};
std::atomic<bool> g_active{false};
std::once_flag g_load_once;
int64_t min_rows() {
    static const int64_t v = [] {
        const char* s = std::getenv("R3_MOE_REMOTE_MIN_ROWS");
        if (!s || !*s) return (int64_t)LLONG_MAX;
        char* end = nullptr;
        const long long n = std::strtoll(s, &end, 10);
        return end && *end == 0 && n >= 0 ? (int64_t)n : (int64_t)LLONG_MAX;
    }();
    return v;
}
void load_env_hook() {
    const char* so = std::getenv("R3_MOE_REMOTE_SO");
    if (!so || !*so) return;
    void* h = dlopen(so, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        std::fprintf(stderr, "radiance: R3_MOE_REMOTE_SO '%s' could not be loaded: %s\n",
                     so, dlerror());
        return;
    }
    using GetHooks = const RadMoeRemoteHooks* (*)(void);
    auto get = (GetHooks)dlsym(h, "rad_moe_remote_hooks");
    if (!get) {
        std::fprintf(stderr, "radiance: R3_MOE_REMOTE_SO '%s' has no rad_moe_remote_hooks\n", so);
        return;
    }
    if (rad_moe_remote_register(get()) < 0)
        std::fprintf(stderr, "radiance: R3_MOE_REMOTE_SO '%s' returned an invalid MoE remote hook\n",
                     so);
}
void ensure_loaded() { std::call_once(g_load_once, load_env_hook); }
}
extern "C" int rad_moe_remote_register(const RadMoeRemoteHooks* hooks) {
    if (!hooks || hooks->abi_version != RAD_MOE_REMOTE_ABI || !hooks->submit || !hooks->wait)
        return RAD_E_INVAL;
    if (g_active.load(std::memory_order_acquire)) return RAD_E_STATE;
    g_hooks = *hooks;
    g_active.store(true, std::memory_order_release);
    const int64_t m = min_rows();
    if (m == LLONG_MAX)
        std::fprintf(stderr, "radiance: MoE remote hook active (R3_MOE_REMOTE_MIN_ROWS=never)\n");
    else
        std::fprintf(stderr, "radiance: MoE remote hook active (R3_MOE_REMOTE_MIN_ROWS=%lld)\n", (long long)m);
    return RAD_OK;
}
extern "C" int rad_moe_remote_wanted(int64_t step_rows) {
    ensure_loaded();
    return g_active.load(std::memory_order_acquire) && step_rows >= min_rows();
}
extern "C" RadMoeRemoteHandle rad_moe_remote_submit(
    RadCtx* base, int layer, int64_t step_rows, int64_t rows, int64_t n_embd,
    int64_t top_k, int64_t expert_base, const rad_weight* gate_up, const rad_weight* down,
    int64_t n_expert, int32_t* ids_dev, const uint16_t* ew_dev,
    const void* activation, int64_t activation_stride_bytes) {
    if (!base || !rad_moe_remote_wanted(step_rows) || rows <= 0 || n_embd <= 0 || top_k <= 0 ||
        n_expert <= 0 || !gate_up || !down || !ids_dev || !ew_dev || !activation) return nullptr;
    auto* c = static_cast<rad::Ctx*>(base);
    bool any = false;
    for (int64_t x = 0; x < n_expert; ++x)
        if (gate_up[x] && down[x] && c->weight_in_host_pool(gate_up[x]) &&
            c->weight_in_host_pool(down[x])) { any = true; break; }
    if (!any) return nullptr;
    const int64_t n = rows * top_k;
    thread_local std::vector<int32_t> ids;
    thread_local std::vector<uint16_t> ew;
    thread_local std::vector<int64_t> slot;
    thread_local std::vector<RadMoeRemoteRoute> route;
    ids.resize((size_t)n); ew.resize((size_t)n);
    RadStream s = c->cur_stream();
    if (rad_memcpy_async(ids.data(), ids_dev, n * (int64_t)sizeof(int32_t), s) < 0 ||
        rad_memcpy_async(ew.data(), ew_dev, n * (int64_t)sizeof(uint16_t), s) < 0 ||
        rad_stream_sync(s) < 0) {
        rad_step_fail(base, "MoE remote routing copy failed");
        return nullptr;
    }
    slot.clear(); route.clear();
    for (int64_t i = 0; i < n; ++i) {
        const int64_t le = (int64_t)ids[(size_t)i] - expert_base;
        if (le < 0 || le >= n_expert || !gate_up[le] || !down[le] ||
            !c->weight_in_host_pool(gate_up[le]) || !c->weight_in_host_pool(down[le])) continue;
        slot.push_back(i);
        route.push_back(RadMoeRemoteRoute{(int32_t)(i / top_k), ids[(size_t)i], ew[(size_t)i], 0});
    }
    if (route.empty()) return nullptr;
    const RadMoeRemoteHooks hooks = g_hooks;
    RadMoeRemoteHandle h = hooks.submit(hooks.user, layer, c->rank(), rows, n_embd,
                                        route.data(), (int64_t)route.size(),
                                        activation, activation_stride_bytes);
    if (!h) return nullptr;
    for (int64_t i : slot) ids[(size_t)i] = -1;
    if (rad_memcpy_async(ids_dev, ids.data(), n * (int64_t)sizeof(int32_t), s) < 0) {
        (void)hooks.wait(hooks.user, h);
        rad_step_fail(base, "MoE remote route mask copy failed");
        return nullptr;
    }
    return h;
}
extern "C" int rad_moe_remote_wait_into(RadCtx* base, RadMoeRemoteHandle handle,
                                          void* dst, int64_t bytes) {
    if (!base || !handle || !dst || bytes <= 0 ||
        !g_active.load(std::memory_order_acquire)) return RAD_E_INVAL;
    const RadMoeRemoteHooks hooks = g_hooks;
    const void* src = hooks.wait(hooks.user, handle);
    if (!src) return RAD_E_STATE;
    auto* c = static_cast<rad::Ctx*>(base);
    RadStream s = c->cur_stream();
    const int rc = rad_memcpy_async(dst, src, bytes, s);
    if (rc < 0) return rc;
    /* wait() transfers a borrowed result pointer, with no release callback in the minimal ABI.
     * Drain this engine stream so the hook may immediately recycle that storage safely. */
    return rad_stream_sync(s);
}
