"""Radiance 1.3.0: optional remote whole-expert MoE compute hand-off."""

from bigcherry.patcher import Edit, FilePatch

_RAD_MOE_REMOTE_H = r"""/* rad_moe_remote.h -- optional whole-expert compute hand-off for routed MoE blocks. */
#pragma once
#include <stdint.h>
#include "rad_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
#define RAD_MOE_REMOTE_ABI 2u
typedef void* RadMoeRemoteHandle;
typedef struct RadMoeRemoteRoute {
    int32_t token;
    int32_t expert;
    uint16_t weight_bf16;
    uint16_t reserved;
} RadMoeRemoteRoute;
typedef struct RadMoeRemoteHooks {
    uint32_t abi_version;
    void* user;
    /* Nonzero only for experts this helper owns. The engine asks before it masks a local route. */
    int (*owns)(void* user, int layer, int expert);
    RadMoeRemoteHandle (*submit)(void* user, int layer, int rank,
                                 int64_t rows, int64_t n_embd,
                                 const RadMoeRemoteRoute* route, int64_t n_route,
                                 const void* activation, int64_t activation_stride_bytes);
    const void* (*wait)(void* user, RadMoeRemoteHandle handle);
} RadMoeRemoteHooks;
/* Register before architecture declaration. An already-loaded plugin may call this directly.
 * R3_MOE_REMOTE_SO may instead name a .so exporting
 *   const RadMoeRemoteHooks* rad_moe_remote_hooks(void)
 * which the core loads during architecture declaration. */
int rad_moe_remote_register(const RadMoeRemoteHooks* hooks);
int rad_moe_remote_wanted(int64_t step_rows);
RadMoeRemoteHandle rad_moe_remote_submit(RadCtx* c, int layer,
                                         int64_t step_rows, int64_t rows, int64_t n_embd,
                                         int64_t top_k, int64_t expert_base,
                                         const rad_weight* gate_up, const rad_weight* down,
                                         int64_t n_expert,
                                         int32_t* ids, const uint16_t* weight_bf16,
                                         const void* activation, int64_t activation_stride_bytes);
/* Waits through the registered hook, copies its BF16 partial into dst on the engine's current
 * stream, and drains that stream before returning so the hook may reuse its returned storage. */
int rad_moe_remote_wait_into(RadCtx* c, RadMoeRemoteHandle handle, void* dst, int64_t bytes);
#ifdef __cplusplus
}
#endif
"""

_MOE_REMOTE_CPP = r"""/* moe_remote.cpp -- residency policy and C hand-off for remote whole-expert compute. */
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
/* Steps above this keep the engine's own path untouched: no routing copy, gather fusion kept. */
int64_t max_rows() {
    static const int64_t v = [] {
        const char* s = std::getenv("R3_MOE_REMOTE_MAX_ROWS");
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
    if (!hooks || hooks->abi_version != RAD_MOE_REMOTE_ABI || !hooks->owns ||
        !hooks->submit || !hooks->wait) return RAD_E_INVAL;
    if (g_active.load(std::memory_order_acquire)) return RAD_E_STATE;
    g_hooks = *hooks;
    g_active.store(true, std::memory_order_release);
    const int64_t m = min_rows();
    if (m == LLONG_MAX)
        std::fprintf(stderr, "radiance: MoE remote hook active (R3_MOE_REMOTE_MIN_ROWS=never)\n");
    else
        std::fprintf(stderr, "radiance: MoE remote hook active (R3_MOE_REMOTE_MIN_ROWS=%lld MAX_ROWS=%lld)\n",
                     (long long)m, (long long)max_rows());
    return RAD_OK;
}
extern "C" int rad_moe_remote_wanted(int64_t step_rows) {
    ensure_loaded();
    return g_active.load(std::memory_order_acquire) && step_rows >= min_rows() && step_rows <= max_rows();
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
    const RadMoeRemoteHooks hooks = g_hooks;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t le = (int64_t)ids[(size_t)i] - expert_base;
        if (le < 0 || le >= n_expert || !gate_up[le] || !down[le] ||
            !c->weight_in_host_pool(gate_up[le]) || !c->weight_in_host_pool(down[le]) ||
            !hooks.owns(hooks.user, layer, (int)le)) continue;
        slot.push_back(i);
        route.push_back(RadMoeRemoteRoute{(int32_t)(i / top_k), (int32_t)le, ew[(size_t)i], 0});
    }
    if (route.empty()) return nullptr;
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
"""

PATCHES = (
    FilePatch(
        path="abi/rad_moe_remote.h",
        create=True,
        description="Create the optional remote-MoE C ABI.",
        edits=(Edit(
            id="create-remote-moe-abi",
            anchor=r"\A",
            text=_RAD_MOE_REMOTE_H,
            guard=r"^#define RAD_MOE_REMOTE_ABI 2u$",
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="core/place/moe_remote.cpp",
        create=True,
        description="Create remote-MoE hook loading, routing, and result hand-off.",
        edits=(Edit(
            id="create-remote-moe-core",
            anchor=r"\A",
            text=_MOE_REMOTE_CPP,
            guard=r'^extern "C" int rad_moe_remote_register',
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="arch/common/rad_block_hc.h",
        description="Disable fused MoE gather only while remote rows must join before the collective.",
        edits=(
            Edit(
                id="include-remote-moe-abi",
                anchor=r"^#define RAD_BLOCK_HC_H$",
                text='\n#include "rad_moe_remote.h"',
                guard=r'^#include "rad_moe_remote\.h"$',
                expect_matches=1,
                rationale="The header guard define is unique and precedes the existing architecture includes.",
            ),
            Edit(
                id="remote-aware-hc-gather",
                anchor=r"^[ \t]*return takes_ar\(T\) && gather_taken\(g, T, gather_rows, gather_rows6\);$",
                mode="replace",
                text="""        /* Remote rows join x.x before the collective; only the gather fusion stands down. */
        return !rad_moe_remote_wanted(T) && takes_ar(T) &&
               gather_taken(g, T, gather_rows, gather_rows6);""",
                guard=r"!rad_moe_remote_wanted\(T\)",
                expect_matches=1,
                rationale="The unique takes_gather return is the policy point controlling the gather/write fusion.",
            ),
        ),
    ),
    FilePatch(
        path="arch/common/rad_block_moe_fp8.h",
        description="Route host-pool experts through the remote hook and add their partial before all-reduce.",
        edits=(
            Edit(
                id="include-remote-moe-abi",
                anchor=r'^#include "rad_device\.h"$',
                text='\n#include "rad_moe_remote.h"',
                guard=r'^#include "rad_moe_remote\.h"$',
                expect_matches=1,
                rationale="The device include is unique in this header and sits with the architecture dependencies.",
            ),
            Edit(
                id="remote-weight-probes",
                anchor=r"^[ \t]*std::vector<rad_weight> w_gu, w_gus, w_dn, w_dns;$",
                text="""
    /* Expert-id -> movement-unit probes for the remote residency policy. */
    std::vector<rad_weight> remote_gu, remote_dn;
    rad_buf b_remote = 0;""",
                guard=r"remote_gu, remote_dn",
                expect_matches=1,
                rationale="The expert weight vectors are the unique declaration owning the per-expert movement handles.",
            ),
            Edit(
                id="remote-add-op",
                anchor=r"^[ \t]*rad_op op_ar = 0, op_add = 0;$",
                mode="replace",
                text="    rad_op op_ar = 0, op_add = 0, op_remote_add = 0;",
                guard=r"op_remote_add = 0",
                expect_matches=1,
                rationale="The output all-reduce/add declaration is the unique place for the pre-reduction remote merge op.",
            ),
            Edit(
                id="init-remote-weight-probes",
                anchor=r"^[ \t]*w_dn\.resize\(\(size_t\)ne\); w_dns\.resize\(\(size_t\)ne\);$",
                text="""
    remote_gu.assign((size_t)ne, 0);
    remote_dn.assign((size_t)ne, 0);""",
                guard=r"remote_gu\.assign\(\(size_t\)ne, 0\)",
                expect_matches=1,
                rationale="The existing expert vector sizing is the stable declaration-time initialization boundary.",
            ),
            Edit(
                id="remember-bf16-gate-up",
                anchor=r"^[ \t]*if \(!w_gu\[\(size_t\)e\]\) return RAD_E_INVAL;$",
                text="\n            remote_gu[(size_t)e] = w_gu[(size_t)e];",
                guard=r"remote_gu\[\(size_t\)e\] = w_gu\[\(size_t\)e\]",
                expect_matches=1,
                rationale="The unique successful BF16 gate/up declaration has the movement handle needed by the policy.",
            ),
            Edit(
                id="remember-bf16-down",
                anchor=r"^[ \t]*if \(!w_dn\[\(size_t\)e\]\) return RAD_E_INVAL;$",
                text="\n            remote_dn[(size_t)e] = w_dn[(size_t)e];",
                guard=r"remote_dn\[\(size_t\)e\] = w_dn\[\(size_t\)e\]",
                expect_matches=1,
                rationale="The unique successful BF16 down declaration has the movement handle needed by the policy.",
            ),
            Edit(
                id="remember-quant-gate-up",
                anchor=r"^[ \t]*if \(!h\) return RAD_E_INVAL;$",
                text="\n            remote_gu[(size_t)e] = h;",
                guard=r"remote_gu\[\(size_t\)e\] = h;",
                expect_matches=4,
                occurrence=0,
                rationale="Exactly four quantized view success checks exist here; the first is the gate/up code-plane declaration.",
            ),
            Edit(
                id="remember-quant-down",
                anchor=r"^[ \t]*if \(!h\) return RAD_E_INVAL;$",
                text="\n            remote_dn[(size_t)e] = h;",
                guard=r"remote_dn\[\(size_t\)e\] = h;",
                expect_matches=4,
                occurrence=2,
                rationale="Exactly four quantized view success checks exist here; the third is the down code-plane declaration.",
            ),
            Edit(
                id="remember-protected-gate-up",
                anchor=r"^[ \t]*if \(!w_pgu\[\(size_t\)j\]\) return RAD_E_INVAL;$",
                text="\n            remote_gu[(size_t)prot[(size_t)j]] = w_pgu[(size_t)j];",
                guard=r"remote_gu\[\(size_t\)prot\[\(size_t\)j\]\] = w_pgu",
                expect_matches=1,
                rationale="The protected gate/up success check uniquely follows its per-expert handle declaration.",
            ),
            Edit(
                id="remember-protected-down",
                anchor=r"^[ \t]*if \(!w_pdn\[\(size_t\)j\]\) return RAD_E_INVAL;$",
                text="\n            remote_dn[(size_t)prot[(size_t)j]] = w_pdn[(size_t)j];",
                guard=r"remote_dn\[\(size_t\)prot\[\(size_t\)j\]\] = w_pdn",
                expect_matches=1,
                rationale="The protected down success check uniquely follows its per-expert handle declaration.",
            ),
            Edit(
                id="declare-remote-partial-add",
                anchor=r"^[ \t]*\{w\.x, w\.h\.x\}, \{w\.x\}\);$",
                text="""
    if (rad_moe_remote_wanted(g.max_tok)) {
        b_remote = decl_b(b, nm.f("blk.%d.moe_remote_partial", l), RAD_BF16,
                          {c.rows, g.n_embd});
        if (!b_remote) return RAD_E_INVAL;
        op_remote_add = rw(b, RAD_OP(b, "add",
                               RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd),
                                          RAD_STR("dtype", g.dtype)),
                               RAD_NOWEIGHTS),
                           {w.h.x, b_remote}, {w.h.x});
        if (!op_remote_add) return RAD_E_INVAL;
    }""",
                guard=r"moe_remote_partial",
                expect_matches=1,
                rationale="The unique final residual add declaration is immediately before MoeFP8::declare returns.",
            ),
            Edit(
                id="remote-pass-state",
                anchor=r"^[ \t]*const bool shared = op_sgate != 0;$",
                text="""
    const bool remote_policy = op_remote_add && rad_moe_remote_wanted(T);
    RadMoeRemoteHandle remote = nullptr;""",
                guard=r"remote_policy = op_remote_add",
                expect_matches=1,
                rationale="The shared-arm state is the first per-pass policy declaration after row geometry is resolved.",
            ),
            Edit(
                id="disable-fused-router-for-remote",
                anchor=r"^[ \t]*const bool rfused = op_router_fused && rows <= router_fused_rows;$",
                mode="replace",
                text="    const bool rfused = !remote_policy && op_router_fused && rows <= router_fused_rows;",
                guard=r"rfused = !remote_policy",
                expect_matches=1,
                rationale="The unique rfused decision must stand down so the hook can inspect top-k output before scatter.",
            ),
            Edit(
                id="disable-fused-topk-scatter-for-remote",
                anchor=r"^[ \t]*\} else if \(op_topk_scatter && rows <= topk_scatter_rows\) \{$",
                mode="replace",
                text="    } else if (!remote_policy && op_topk_scatter && rows <= topk_scatter_rows) {",
                guard=r"else if \(!remote_policy && op_topk_scatter",
                expect_matches=1,
                rationale="The unique fused top-k/scatter branch must stand down so remote routes can be masked before scatter.",
            ),
            Edit(
                id="submit-remote-before-scatter",
                anchor=r"^[ \t]*brow_slice\(w\.ids, r0, rows, k\), brow_slice\(w\.ew, r0, rows, k\)\);$",
                text="""
        if (remote_policy)
            remote = rad_moe_remote_submit(
                c_, layer, T, rows, g.n_embd, k, c.expert_base,
                remote_gu.data(), remote_dn.data(), ne,
                (int32_t*)rad_buf_ptr(c_, w.ids) + r0 * k,
                (const uint16_t*)rad_buf_ptr(c_, w.ew) + r0 * k,
                (const uint16_t*)rad_buf_ptr(c_, w.h.x) + r0 * g.n_embd,
                g.n_embd * (int64_t)sizeof(uint16_t));""",
                guard=r"remote = rad_moe_remote_submit",
                expect_matches=1,
                rationale="This operand line uniquely closes the standalone top-k issue immediately before scatter consumes ids.",
            ),
            Edit(
                id="force-local-gather-before-remote-merge",
                anchor=r"^[ \t]*if \(!gather_taken\(g, T, gather_out_rows, gather_out_rows6\)\)$",
                mode="replace",
                text="    if (remote_policy || !gather_taken(g, T, gather_out_rows, gather_out_rows6))",
                guard=r"remote_policy \|\| !gather_taken",
                expect_matches=1,
                rationale="The unique gather decision must materialize the local routed partial before the remote partial is added.",
            ),
            Edit(
                id="merge-remote-before-all-reduce",
                anchor=r"^[ \t]*brow_slice\(w\.h\.x, r0, rows, g\.n_embd\)\);$",
                text="""
    if (remote) {
        if (rad_moe_remote_wait_into(c_, remote, rad_buf_ptr(c_, b_remote),
                                     rows * g.n_embd * (int64_t)sizeof(uint16_t)) < 0) {
            rad_step_fail(c_, "MoE remote wait/result copy failed");
            return;
        }
        RAD_ISSUE_N(c_, op_remote_add, rows,
                    brow_slice(w.h.x, r0, rows, g.n_embd),
                    brows(b_remote, rows),
                    brow_slice(w.h.x, r0, rows, g.n_embd));
    }""",
                guard=r"rad_moe_remote_wait_into",
                expect_matches=1,
                rationale="This line uniquely closes the MoE gather issue; the remote result must join immediately before the existing all-reduce.",
            ),
        ),
    ),
    FilePatch(
        path="core/place/mover.cpp",
        description="Expose host-pool ownership through residency publication.",
        edits=(
            Edit(
                id="publish-host-pool-bit",
                anchor=r"^void Residency::publish\(rad_weight w, void\* ptr\) \{\n[ \t]*if \(\(size_t\)w >= slot_\.size\(\)\) return;\n[ \t]*slot_\[w\]\.ptr = ptr;\n[ \t]*slot_\[w\]\.ready = nullptr;$",
                mode="replace",
                text="""void Residency::publish(rad_weight w, void* ptr, bool host_pool) {
    if ((size_t)w >= slot_.size()) return;
    slot_[w].ptr = ptr;
    slot_[w].ready = nullptr;
    slot_[w].host_pool = host_pool ? 1 : 0;""",
                guard=r"host_pool = host_pool \? 1 : 0",
                expect_matches=1,
                max_span_lines=4,
                rationale="The publish prologue uniquely settles a residency slot; ownership must be recorded with the settled pointer.",
            ),
            Edit(
                id="clear-host-pool-during-transfer",
                anchor=r"^void Residency::begin_transfer\(rad_weight w, void\* ptr, RadEvent ready\) \{\n[ \t]*if \(\(size_t\)w >= slot_\.size\(\)\) return;\n[ \t]*slot_\[w\]\.ptr = ptr;\n[ \t]*slot_\[w\]\.ready = ready;$",
                mode="replace",
                text="""void Residency::begin_transfer(rad_weight w, void* ptr, RadEvent ready) {
    if ((size_t)w >= slot_.size()) return;
    slot_[w].ptr = ptr;
    slot_[w].ready = ready;
    slot_[w].host_pool = 0;""",
                guard=r"slot_\[w\]\.host_pool = 0",
                expect_matches=1,
                max_span_lines=4,
                rationale="The begin_transfer prologue uniquely installs an in-flight device destination, which is not host-pool residency.",
            ),
            Edit(
                id="republish-host-pool-owner",
                anchor=r"^void Mover::republish\(int32_t unit\) \{\n[ \t]*const MoveUnit& mu = plan_->units\[\(size_t\)unit\];\n[ \t]*char\* base = \(char\*\)unit_base\(unit\);\n[ \t]*for \(size_t i = 0; i < mu\.weights\.size\(\); \+\+i\)\n[ \t]*res_\.publish\(mu\.weights\[i\], base \? base \+ mu\.offset\[i\] : nullptr\);\n\}$",
                mode="replace",
                text="""void Mover::republish(int32_t unit) {
    const MoveUnit& mu = plan_->units[(size_t)unit];
    const UnitState& us = unit_[(size_t)unit];
    char* base = (char*)unit_base(unit);
    const bool host_pool = us.tier == Tier::HostPinned && us.slab_slot < 0;
    for (size_t i = 0; i < mu.weights.size(); ++i)
        res_.publish(mu.weights[i], base ? base + mu.offset[i] : nullptr, host_pool);
}""",
                guard=r"const bool host_pool = us\.tier == Tier::HostPinned",
                expect_matches=1,
                max_span_lines=6,
                rationale="The complete six-line republish function is atomic because tier ownership and every published weight must change together.",
            ),
            Edit(
                id="staged-copy-retains-host-ownership",
                anchor=r"^[ \t]*res_\.publish\(mu\.weights\[i\], dst \+ at\[k\] \+ mu\.offset\[i\]\);$",
                mode="replace",
                text="""            /* A prefill-stage copy is temporary; keep host ownership visible to the remote policy. */
            res_.publish(mu.weights[i], dst + at[k] + mu.offset[i], true);""",
                guard=r"A prefill-stage copy is temporary",
                expect_matches=1,
                rationale="The unique stager publication records a temporary device copy whose source ownership remains the host pool.",
            ),
        ),
    ),
    FilePatch(
        path="core/place/mover.h",
        description="Allow residency publication to carry host-pool ownership.",
        edits=(Edit(
            id="publish-host-pool-parameter",
            anchor=r"^[ \t]*void publish\(rad_weight w, void\* ptr\);$",
            mode="replace",
            text="    void publish(rad_weight w, void* ptr, bool host_pool = false);",
            guard=r"bool host_pool = false",
            expect_matches=1,
            rationale="The unique Residency::publish declaration mirrors the implementation and preserves existing callers through the default.",
        ),),
    ),
    FilePatch(
        path="core/runtime/ctx.h",
        description="Expose whether a weight is still owned by the pinned host pool.",
        edits=(Edit(
            id="weight-in-host-pool-query",
            anchor=r"^[ \t]*void\* buf_ptr\(rad_buf b\);$",
            text="""
    bool  weight_in_host_pool(rad_weight w) const {
        if (!residency_) return false;
        const WeightSlot* s = residency_->lookup(w);
        return s && s->host_pool;
    }""",
            guard=r"weight_in_host_pool\(rad_weight w\)",
            expect_matches=1,
            rationale="The buffer accessor is the unique adjacent runtime pointer-query boundary used by architecture code.",
        ),),
    ),
    FilePatch(
        path="core/runtime/residency.h",
        description="Record policy-visible host-pool ownership in each residency slot.",
        edits=(Edit(
            id="host-pool-residency-bit",
            anchor=r"^[ \t]*uint32_t generation;$",
            text="""
    /* Policy-visible residency for routed experts. Temporary prefill staging keeps this set:
     * ownership is still the pinned host pool even while that pass has a device copy. */
    uint8_t  host_pool;""",
            guard=r"uint8_t\s+host_pool;",
            expect_matches=1,
            rationale="The generation field is unique in WeightSlot and ownership belongs to that same residency record.",
        ),),
    ),
)
