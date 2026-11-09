"""Radiance 1.7.0: GPU-lane remote whole-expert MoE hand-off, as thin hooks into one BigCherry-owned extension point.

All policy lives in files this patch creates (abi/bc_moe_remote.h, core/place/bc_moe_remote.cpp,
core/place/bc_moe_remote_place.h, arch/common/bc_moe_remote_block.h) and in the libr3moe plugin. The edits to upstream
files are one-line guarded call sites into them. The shadow compare and mask-only diagnostics are rad0004.
"""

from bigcherry.patcher import Edit, FilePatch

_BC_MOE_REMOTE_H = r"""/* bc_moe_remote.h -- BigCherry's extension point for GPU-lane remote whole-expert MoE compute.
 *
 * The engine asks four questions of it: is this step handed off (wanted), does the helper own this routed
 * expert so the planner may leave it unplaced (skip_unit), what is each layer's runtime expert-to-slot map
 * (layer_map), and is a diagnostic mode on (shadow, mask_only; diagnostic builds only). The libr3moe plugin
 * registers its two callbacks once through bc_moe_remote_register. Every function is exported by the engine
 * binary and bound by the arch plugins and libr3moe at dlopen. */
#ifndef BC_MOE_REMOTE_H
#define BC_MOE_REMOTE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Fixed wire layout shared with the plugin's StepTicket, 16 bytes. */
typedef struct BcMoeRemoteTicket {
    uint32_t placement_epoch;
    uint8_t generation_slot;
    uint8_t reserved[3];
    uint64_t ticket_id;
} BcMoeRemoteTicket;
/* Filled by libr3moe. struct_size must equal sizeof(BcMoeRemotePlugin) of the engine's header. */
typedef struct BcMoeRemotePlugin {
    uint32_t struct_size;
    /* A layer's runtime slot map (slot_of_expert[e] < 0: expert e is a protected, locally served expert). */
    void (*layer_map)(int layer, int n_expert, const int32_t* slot_of_expert);
    /* Whether a helper owns routed expert (layer, expert). */
    int (*owns_expert)(int layer, int expert);
    /* An inference burst begins (1: the engine loop has a step after being idle) or ends (0: the loop found no
     * step). Called from the engine's host scheduler loop, so it fires whether a pass is recorded or replayed;
     * a helper arms its GPU waits only inside a burst, and its queues drain to empty between bursts. */
    void (*burst)(int active);
    /* Per-pass stream-ordered ticket setter; returns a negative RAD_E_* on
     * failure. The engine calls this on lane 0 before capture/replay. */
    /* Immutable scheduler-parent admission value, copied into each rank
     * thread before its trunk/history/draft pass. */
    int (*before_pass)(int rank, struct BcMoeRemoteTicket ticket, void* stream);
    int (*admit)(struct BcMoeRemoteTicket* out, int try_only, int world_size);
    int (*bind)(struct BcMoeRemoteTicket ticket, int64_t scheduler_step);
    int (*cancel)(struct BcMoeRemoteTicket ticket);
    int (*host_done)(struct BcMoeRemoteTicket ticket);
    int (*rank_done)(int rank, struct BcMoeRemoteTicket ticket, void* stream);
    void (*poll)(void);
    void (*shutdown)(void);
    int (*capacity)(uint32_t parent_limit);
} BcMoeRemotePlugin;
/* Register the plugin (NULL unregisters). Layer maps published before registration are replayed to it.
 * Returns 0, or -1 when struct_size does not match. */
int bc_moe_remote_register(const BcMoeRemotePlugin* plugin);
/* Runtime step gate (R3_MOE_REMOTE_MIN_ROWS / R3_MOE_REMOTE_MAX_ROWS, set by libr3moe once its mapped segment
 * is pinned and ready; an unset minimum means remote MoE is off). */
int bc_moe_remote_wanted(int64_t step_rows);
int bc_moe_remote_wanted_upto(int64_t max_step_rows);
/* Diagnostics: always 0 unless the engine is built with BC_MOE_REMOTE_DIAG (patch rad0004). A production build
 * that finds R3_MOE_WORKER_SHADOW or R3_MOE_WORKER_MASK_ONLY set answers "no remote rows" and says so once. */
int bc_moe_remote_shadow(void);
int bc_moe_remote_mask_only(void);
int bc_moe_remote_diag_build(void);
/* The engine loop calls engine_step for every step it runs and engine_idle when it finds no step; the
 * transitions between the two are forwarded to the plugin's burst callback, once each. */
void bc_moe_remote_engine_step(void);
void bc_moe_remote_engine_idle(void);
/* Admission is invoked before sched_.step() / prepare_release; try_only is
 * nonblocking for issue_ahead (1: admission closed/full; skip the run-ahead).
 * With no remote plugin the returned ticket is zero and calls are no-ops. */
int bc_moe_remote_admit(BcMoeRemoteTicket* ticket, int try_only, int world_size);
int bc_moe_remote_bind(BcMoeRemoteTicket ticket, int64_t scheduler_step);
/* Returns an unused tail reservation; negative when the plugin rejects it (the ledger has faulted). */
int bc_moe_remote_cancel(BcMoeRemoteTicket ticket);
int bc_moe_remote_host_done(BcMoeRemoteTicket ticket);
void bc_moe_remote_poll(void);
void bc_moe_remote_shutdown(void);
/* Per-release value handed explicitly from scheduler thread through the rank
 * barrier. The rank thread stores it in TLS for Ctx::run_pass to consume. */
void bc_moe_remote_rank_ticket(BcMoeRemoteTicket ticket);
BcMoeRemoteTicket bc_moe_remote_current_ticket(void);
int bc_moe_remote_before_pass(int rank, BcMoeRemoteTicket ticket, int64_t rows, void* stream);
int bc_moe_remote_rank_done(int rank, BcMoeRemoteTicket ticket, void* stream);
/* Block declare publishes a layer's map here; it is forwarded to the plugin and kept for replay. */
void bc_moe_remote_layer_map(int layer, int n_expert, const int32_t* slot_of_expert);
/* Placement (R3_MOE_WORKER_SKIP_OWNED): whether the engine may leave a routed expert unplaced because the
 * helper computes it and nothing local ever reads it. */
int bc_moe_remote_skip_unit(int layer, int expert, int64_t max_step_rows);
#ifdef __cplusplus
}
#endif
#endif /* BC_MOE_REMOTE_H */
"""

_BC_MOE_REMOTE_CPP = r"""/* bc_moe_remote.cpp -- policy behind abi/bc_moe_remote.h: the decode-sized step gate, the plugin registry,
 * the layer-map store and the SKIP_OWNED placement rule. */
#include "bc_moe_remote.h"
#include "bc_moe_remote_engine.h"
#include "runtime/ctx.h"
#include <type_traits>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>

static_assert(sizeof(BcMoeRemoteTicket) == 16);

#ifdef BC_MOE_REMOTE_DIAG
static constexpr bool kDiagBuild = true;
#else
static constexpr bool kDiagBuild = false;
#endif

namespace {
bool env_on(const char* name) {
    const char* e = std::getenv(name);
    return e && *e && !(e[0] == '0' && e[1] == 0);
}

int64_t rows_env(const char* name) {
    const char* s = std::getenv(name);
    if (!s || !*s) return (int64_t)LLONG_MAX;
    char* end = nullptr;
    const long long n = std::strtoll(s, &end, 10);
    return end && *end == 0 && n >= 0 ? (int64_t)n : (int64_t)LLONG_MAX;
}

int64_t min_rows() {
    static const int64_t v = rows_env("R3_MOE_REMOTE_MIN_ROWS");
    return v;
}
int64_t max_rows() {
    static const int64_t v = rows_env("R3_MOE_REMOTE_MAX_ROWS");
    return v;
}

/* The diagnostic modes exist only in a diagnostic build. Asked for in a production build they must not be
 * silently ignored (that would measure the wrong thing): the extension point answers "no remote rows", so every
 * expert computes locally, and says so once. */
bool diag_refused() {
    if (kDiagBuild) return false;
    static const bool requested = env_on("R3_MOE_WORKER_SHADOW") || env_on("R3_MOE_WORKER_MASK_ONLY");
    if (!requested) return false;
    static std::atomic<bool> said{false};
    if (!said.exchange(true))
        std::fprintf(stderr, "radiance: R3_MOE_WORKER_SHADOW / R3_MOE_WORKER_MASK_ONLY need the diagnostic build "
                             "(BC_MOE_REMOTE_DIAG, patch rad0004); remote MoE hand-off is off and every expert "
                             "computes locally\n");
    return true;
}

std::mutex g_mu;
BcMoeRemotePlugin g_plugin{};   /* struct_size 0: none registered */
std::map<int, std::vector<int32_t>> g_layer_maps;
}

extern "C" int bc_moe_remote_wanted(int64_t step_rows) {
    if (diag_refused()) return 0;
    const int64_t lo = min_rows(), hi = max_rows();
    return lo != LLONG_MAX && step_rows >= lo && step_rows <= hi;
}

extern "C" int bc_moe_remote_wanted_upto(int64_t max_step_rows) {
    if (diag_refused()) return 0;
    const int64_t lo = min_rows(), hi = max_rows();
    return lo != LLONG_MAX && lo <= hi && lo <= max_step_rows;
}

extern "C" int bc_moe_remote_shadow(void) {
    static const bool v = kDiagBuild && env_on("R3_MOE_WORKER_SHADOW");
    return v;
}

extern "C" int bc_moe_remote_mask_only(void) {
    static const bool v = kDiagBuild && env_on("R3_MOE_WORKER_MASK_ONLY");
    return v;
}

extern "C" int bc_moe_remote_diag_build(void) {
    return kDiagBuild ? 1 : 0;
}

/* Layer maps are published at declare, which can run before the plugin registers (libr3moe registers when it
 * loads). Keep every published map and replay them on registration, so no layer's map is lost to that ordering. */
extern "C" int bc_moe_remote_register(const BcMoeRemotePlugin* plugin) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!plugin) {
        g_plugin = BcMoeRemotePlugin{};
        return 0;
    }
    if (plugin->struct_size != sizeof(BcMoeRemotePlugin)) return -1;
    /* No scheduler backlog bound exists for no-output prefill. Use the
     * engine's route ring as a resource ceiling plus its one ahead batch.
     * Do this before map replay or graph lowering allocates rank buffers. */
    constexpr uint32_t limit = std::extent_v<decltype(rad::RouteSlot::staging)> +
                               BcMoeRemoteAdmission::kIssueAheadDepth;
    if (!plugin->capacity || plugin->capacity(limit) < 0) return -1;
    g_plugin = *plugin;
    if (g_plugin.layer_map)
        for (const auto& [layer, slots] : g_layer_maps)
            g_plugin.layer_map(layer, (int)slots.size(), slots.data());
    return 0;
}

namespace {
std::atomic<bool> g_burst_active{false};
void forward_burst(int active) {
    void (*burst)(int) = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        burst = g_plugin.burst;
    }
    if (burst) burst(active);
}
}

/* Per-step cost is one atomic exchange; the plugin hears only the transitions. */
extern "C" void bc_moe_remote_engine_step(void) {
    if (!g_burst_active.exchange(true, std::memory_order_acq_rel)) forward_burst(1);
}

extern "C" void bc_moe_remote_engine_idle(void) {
    if (g_burst_active.exchange(false, std::memory_order_acq_rel)) forward_burst(0);
}

/* Scheduler-thread and rank-thread calls into the size-checked plugin.
 * The ticket originates only from admission; b->step is debug identity and
 * is never used to derive the actual epoch/slot/ticket id. */
extern "C" int bc_moe_remote_admit(BcMoeRemoteTicket* out, int try_only, int world_size) {
    if (!out) return -1;
    *out = BcMoeRemoteTicket{};
    int (*fn)(BcMoeRemoteTicket*, int, int) = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        fn = g_plugin.admit;
    }
    if (!fn) return bc_moe_remote_wanted_upto(LLONG_MAX) ? -1 : 0;
    return fn(out, try_only, world_size);
}
extern "C" int bc_moe_remote_bind(BcMoeRemoteTicket ticket, int64_t step) {
    if (!ticket.ticket_id) return 0;
    int (*fn)(BcMoeRemoteTicket, int64_t) = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        fn = g_plugin.bind;
    }
    return fn ? fn(ticket, step) : -1;
}
extern "C" int bc_moe_remote_cancel(BcMoeRemoteTicket ticket) {
    if (!ticket.ticket_id) return 0;
    int (*fn)(BcMoeRemoteTicket) = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        fn = g_plugin.cancel;
    }
    return fn ? fn(ticket) : -1;
}
extern "C" int bc_moe_remote_host_done(BcMoeRemoteTicket ticket) {
    if (!ticket.ticket_id) return 0;
    int (*fn)(BcMoeRemoteTicket) = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        fn = g_plugin.host_done;
    }
    return fn ? fn(ticket) : -1;
}
extern "C" void bc_moe_remote_poll(void) {
    void (*fn)() = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        fn = g_plugin.poll;
    }
    if (fn) fn();
}
extern "C" void bc_moe_remote_shutdown(void) {
    void (*fn)() = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        fn = g_plugin.shutdown;
    }
    if (fn) fn();
}
static thread_local BcMoeRemoteTicket g_rank_ticket{};
extern "C" void bc_moe_remote_rank_ticket(BcMoeRemoteTicket t) { g_rank_ticket = t; }
extern "C" BcMoeRemoteTicket bc_moe_remote_current_ticket(void) { return g_rank_ticket; }
extern "C" int bc_moe_remote_before_pass(int rank, BcMoeRemoteTicket ticket, int64_t rows, void* stream) {
    int (*fn)(int, BcMoeRemoteTicket, void*) = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        fn = g_plugin.before_pass;
    }
    /* rows only decides the no-plugin case: a pass that wanted the remote path cannot run without it. */
    return fn ? fn(rank, ticket, stream) : (bc_moe_remote_wanted(rows) ? -1 : 0);
}
extern "C" int bc_moe_remote_rank_done(int rank, BcMoeRemoteTicket ticket, void* stream) {
    if (!ticket.ticket_id) return 0;
    int (*fn)(int, BcMoeRemoteTicket, void*) = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        fn = g_plugin.rank_done;
    }
    return fn ? fn(rank, ticket, stream) : -1;
}

extern "C" void bc_moe_remote_layer_map(int layer, int n_expert, const int32_t* slot_of_expert) {
    if (n_expert <= 0 || !slot_of_expert) return;
    std::lock_guard<std::mutex> lock(g_mu);
    g_layer_maps[layer].assign(slot_of_expert, slot_of_expert + n_expert);
    if (g_plugin.layer_map) g_plugin.layer_map(layer, n_expert, slot_of_expert);
}

/* R3_MOE_WORKER_SKIP_OWNED=1: a helper-owned expert is not placed at all -- no VRAM, no host pool -- which is
 * what lets one engine rank (the R9700) hold a model whose experts it could not hold alone. Only when nothing
 * local will ever read the unit: every step is handed off (the hand-off cap covers the largest step), no
 * diagnostic recomputes owned experts locally, and the layer's runtime map gives the expert a regular slot with
 * no protected expert in the layer (those layers stay local; the helper drops ownership on exactly the same
 * test). */
extern "C" int bc_moe_remote_skip_unit(int layer, int expert, int64_t max_step_rows) {
    static const bool on = env_on("R3_MOE_WORKER_SKIP_OWNED");
    if (!on || diag_refused() || bc_moe_remote_shadow() || bc_moe_remote_mask_only()) return 0;
    if (min_rows() == (int64_t)LLONG_MAX || min_rows() > 1 || max_rows() < max_step_rows) return 0;
    int (*owns)(int, int) = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        owns = g_plugin.owns_expert;
        const auto it = g_layer_maps.find(layer);
        if (it == g_layer_maps.end() || expert < 0 || expert >= (int)it->second.size()) return 0;
        for (int32_t slot : it->second)
            if (slot < 0) return 0;
    }
    return owns && owns(layer, expert) ? 1 : 0;
}
"""

_BC_MOE_REMOTE_ENGINE_H = r"""/* Scheduler/rank admission bridge. Scheduler changes state while ranks
 * are parked; the existing barrier publishes issued_. current_ survives
 * issue-ahead until host completion. The coordinator owns admission. */
#pragma once
#include "bc_moe_remote.h"
#include "device/device.h"

class BcMoeRemoteAdmission {
    BcMoeRemoteTicket current_{}, ahead_{}, issued_{};
    int ahead_status_ = 0;
public:
    // Engine::step_loop holds one scalar ahead batch, not a queue.
    static constexpr uint32_t kIssueAheadDepth = 1;
    template<class Scheduler, class Batch>
    int step(bool issued, const Batch*& b, Scheduler& scheduler, int ranks) {
        if (issued) {
            scheduler.promote_next();
            current_ = ahead_;
            ahead_ = {};
        } else {
            if (bc_moe_remote_admit(&current_, 0, ranks) < 0) return RAD_E_STATE;
            b = scheduler.step();
            if (!b) {
                if (bc_moe_remote_cancel(current_) < 0) return RAD_E_STATE;
                current_ = {};
            } else if (bc_moe_remote_bind(current_, b->step) < 0) return RAD_E_STATE;
        }
        issued_ = current_;
        return RAD_OK;
    }
    bool try_ahead(int ranks) {
        const int rc = bc_moe_remote_admit(&ahead_, 1, ranks);
        ahead_status_ = rc < 0 ? RAD_E_STATE : RAD_OK;
        return rc == 0;
    }
    int ahead_status() const { return ahead_status_; }
    template<class Batch>
    int bind_ahead(const Batch* b) {
        if (!b) {
            if (bc_moe_remote_cancel(ahead_) < 0) return RAD_E_STATE;
            ahead_ = {};
            return RAD_OK;
        }
        if (bc_moe_remote_bind(ahead_, b->step) < 0) return RAD_E_STATE;
        issued_ = ahead_;
        return RAD_OK;
    }
    BcMoeRemoteTicket issued_ticket() const { return issued_; }
    /* After mark_step_end, before issue_ahead: covers scheduler-issued
     * resolve/checkpoint work and lane-joined model/draft passes.
     * Each parent owns distinct events; step_end_ is never copied as proof.
     * No current-device switch: rank_done launches its seal kernel and
     * records its event on the rank's own engine stream, and the AQL backend
     * dispatches both on that stream's device (hipLaunchKernel resolves the
     * kernel for the stream's ordinal; rad_event_record enqueues a barrier on
     * the stream). */
    template<class Ranks>
    int seal(const Ranks& ranks) {
        if (!current_.ticket_id) return RAD_OK;
        for (const auto& r : ranks)
            if (bc_moe_remote_rank_done(r->index, current_, r->ctx.stream()) < 0) return RAD_E_DEVICE;
        return RAD_OK;
    }
    int host_done() { return bc_moe_remote_host_done(current_); }
};
"""

_BC_MOE_REMOTE_PLACE_H = r"""/* bc_moe_remote_place.h -- the placement and loading side of the BigCherry remote-expert extension point
 * (abi/bc_moe_remote.h). Header-only templates, so the planner and the bring-up call them with their own types
 * and need no further includes. */
#ifndef BC_MOE_REMOTE_PLACE_H
#define BC_MOE_REMOTE_PLACE_H

#include "bc_moe_remote.h"
#include "rad_core.h"

#include <cstdint>
#include <vector>

namespace rad {

/* The expert-tiered fill must not rank any preplaced unit. */
template<class P, class O>
inline void bc_moe_remote_rank_units(P* p, O* order) {
    for (int32_t i = 0; i < (int32_t)p->units.size(); ++i)
        if (!p->units[(size_t)i].preplaced()) order->push_back(i);
}

/* A routed expert the remote helper computes, when the engine never reads it (bc_moe_remote_skip_unit): no
 * slot, no pool and no copy. It stays on the mapped tier as a placeholder the loader publishes without
 * reading; the zero-copy site keeps it out of the host-site and mixed-operand logic. Returns whether the unit
 * was taken (the planner then leaves it alone). */
template <class MU>
inline bool bc_moe_remote_pin_unit(MU& mu, int64_t max_tok) {
    if (mu.layer < 0 || mu.expert < 0 || !bc_moe_remote_skip_unit(mu.layer, mu.expert, max_tok)) return false;
    mu.tier = Tier::Mapped;
    mu.movable = false;
    mu.site = Site::DeviceZeroCopy;
    mu.reason = "computed by the remote MoE helper; never read on this rank";
    return true;
}

/* The pre-plan budget makes the same call pin_unit does: a helper-owned expert costs this rank nothing. The
 * step-cap condition is the planner's and bring-up's to enforce (0 skips it here); a cap below the largest
 * step only makes this budget generous, and the plan then reports what did not fit. */
template <class U>
inline void bc_moe_remote_mark_footprint(std::vector<U>& units) {
    for (U& mu : units)
        if (mu.layer >= 0 && mu.expert >= 0 && bc_moe_remote_skip_unit(mu.layer, mu.expert, 0)) mu.mapped = true;
}

/* Bring-up: publish a unit the planner left on the mapped tier as a never-read placeholder (the container
 * address, never dereferenced) instead of loading it. Its stored form never matters because this rank never
 * reads it, so the byte-count and direct-read checks do not apply. Returns whether it was published. */
template <class W, class E, class F, class M, class I>
inline bool bc_moe_remote_publish_placeholder(int64_t max_tok, W& w, const E* e, const F& file, M& mover, I index) {
    if (!e || w.tier != Tier::Mapped || w.decl.group.layer < 0 || w.decl.group.expert < 0 ||
        !bc_moe_remote_skip_unit(w.decl.group.layer, w.decl.group.expert, max_tok))
        return false;
    w.ptr = (void*)file.data(file.plane(*e, w.sel[0]));
    w.host_ptr = w.ptr;
    mover.publish_mapped(index, w.ptr);
    return true;
}

}  /* namespace rad */

#endif /* BC_MOE_REMOTE_PLACE_H */
"""

_BC_MOE_REMOTE_BLOCK_H = r"""/* bc_moe_remote_block.h -- the MoE block's side of the BigCherry remote-expert extension point
 * (abi/bc_moe_remote.h). MoeFP8 holds one BcMoeRemote and calls it at four places; HyperConn asks
 * bc_moe_remote_hc_open. Everything else (what is handed off, when, and how the helper's rows rejoin the block's
 * output) is decided here, in the engine's C ABI and in the libr3moe plugin.
 *
 * Hand-off of a step: the routed ids are published to the helper (the publish op masks the helper-owned routes
 * out of the sort in place), the scatter is re-run so the local grouped GEMMs skip them, and the helper rows
 * rejoin the block:
 *  - on a step that fuses (bc_moe_remote_fuses: the write behind the block takes the gather, a TP engine's fused
 *    MoE gather + reducing hc write, and the step is at most one sequence's decode or verify step),
 *    moe_mailbox_inject gives each helper row a spare sorted row of that gather, with weight 1, on the layer's
 *    consumer rank (RAD49; the other ranks add nothing for it), and the block gathers nothing;
 *  - otherwise moe_mailbox_add replaces moe_gather: it sums the local rows and the helper's rows in slot order
 *    into the block's output; a HyperConn write behind it then leaves the reduction to the block's own all-reduce
 *    and writes plainly (the v25 path), and a norm-fused consumer keeps its fold.
 * With the mailbox ops absent or the step excluded, no call changes anything. */
#ifndef BC_MOE_REMOTE_BLOCK_H
#define BC_MOE_REMOTE_BLOCK_H

#include "rad_fp8.h"
#include "bc_moe_remote.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace rad {
namespace arch {

/* RADIANCE_DEBUG_ROUTING's print (rad_block_moe_fp8.h, defined after this header is included): the inject calls it first
 * on a fused step, while the router weights are still the router's, which also spends the layer's one print. */
inline void moe_debug_weights(RadCtx* c_, int layer, int64_t top_k, rad_buf ids, rad_buf ew, bool f32);

/* RAD49 per-step choice between the two ways a handed-off step's helper rows rejoin a TP engine's MoE-fed HyperConn
 * write (DESIGN: runs/rad49-fuse-ab measured the fused form -4.3% ms/step at one stream's MTP step, 4 rows, and +4.9%
 * at six streams, 24 rows; doc prefill flat):
 *  - fused: moe_mailbox_inject + the write's fused MoE gather + reducing write (ar_gather_hc_write);
 *  - unfused: moe_mailbox_add, the block's own standalone all-reduce, then the plain hc write (the v25 path).
 * A step fuses up to one sequence's decode or verify step (1 + max_spec rows, the geometry's own bound, which is where
 * it measured faster) and where the write takes the gather (gather_taken). The block and HyperConn ask the same
 * function of the same T and bounds, so they agree, as every rank does. Mask-only (diagnostic) never fuses: it restores
 * the local rows for the add. */
inline int64_t bc_moe_remote_fuse_rows(const Geom& g) { return 1 + (g.max_spec > 0 ? g.max_spec : 0); }
inline bool bc_moe_remote_fuses(const Geom& g, int64_t T, int64_t gather_rows, int64_t gather_rows6) {
    return !bc_moe_remote_mask_only() && T <= bc_moe_remote_fuse_rows(g) && gather_taken(g, T, gather_rows, gather_rows6);
}
/* HyperConn::takes_ar: the connection fed by a MoE gather source (top_k > 0) leaves the reduction to the block on a
 * handed-off step that does not fuse; attention and delta-net connections keep their fused writes. */
inline bool bc_moe_remote_hc_open(int64_t moe_top_k, const Geom& g, int64_t T, int64_t gather_rows, int64_t gather_rows6) {
    return moe_top_k > 0 && bc_moe_remote_wanted(T) && !bc_moe_remote_shadow() &&
           !bc_moe_remote_fuses(g, T, gather_rows, gather_rows6);
}

struct BcMoeRemote {
    /* publish_fused: the same op for a step whose helper rows join the fused gather (param fused = 1: no combine) */
    rad_op publish = 0, publish_fused = 0, add = 0, inject = 0;

    /* The step `T` is handed off (shadow keeps the local path and only publishes). */
    bool policy(int64_t T) const { return publish && add && bc_moe_remote_wanted(T); }
    bool active(int64_t T) const { return policy(T) && !bc_moe_remote_shadow(); }
    /* A handed-off step whose helper rows join the fused MoE gather + reducing hc write: exactly the predicate
     * HyperConn asks (bc_moe_remote_fuses, through bc_moe_remote_hc_open), on the same T and bounds, so the block and
     * the write always agree. It depends on T and declared bounds only, so every rank and every pass of the step agree;
     * after_route and merge both ask it. */
    template <class Moe>
    bool fused(const Moe& m, int64_t T) const {
        return active(T) && bc_moe_remote_fuses(m.g, T, m.gather_out_rows, m.gather_out_rows6);
    }
    /* A fused step on a block with a separate FP32 combine: its routed rows go to `moe` for the combine, which the fused
     * write never reads, so the step cannot be served exactly once. Upstream never declares the fused gather behind such
     * a block; if one is, merge fails the step (fail closed) rather than let the write read un-added rows. */
    template <class Moe>
    bool refuses(const Moe& m, int64_t T) const { return m.op_combine && fused(m, T); }
    /* MoeFP8::issues_ar: a handed-off step that does not fuse, behind a HyperConn write (which then leaves the
     * reduction, bc_moe_remote_hc_open), runs the block's own all-reduce. A norm-fused consumer keeps its fold. */
    template <class Moe>
    bool forces_ar(const Moe& m, int64_t T) const {
        return m.ar_out_take != kArOutNorm && active(T) && !fused(m, T);
    }

    /* Block declare, after the quantised experts' order `reg` is final: publish the layer's runtime slot map,
     * and declare the two mailbox ops when the step gate admits any step. The gate comes from libr3moe, so an
     * absent op while the gate is open is an error, never a silent local fallback. */
    template <class Moe>
    int declare(RadBuilder* b, const Moe& m, const std::vector<int64_t>& reg, int layer) {
        const auto& c = m.c;
        const auto& g = m.g;
        const auto& w = m.w;
        const int64_t ne = c.n_expert;
        std::vector<int32_t> slot((size_t)ne, -1);
        for (size_t s = 0; s < reg.size(); ++s)
            if (reg[s] >= 0 && reg[s] < ne) slot[(size_t)reg[s]] = (int32_t)s;
        bc_moe_remote_layer_map(layer, (int)ne, slot.data());

        if (!bc_moe_remote_wanted_upto(g.max_tok)) return RAD_OK;
        const rad_buf q = c.expert_rot ? w.hr.q : w.h.q;
        const rad_buf s = c.expert_rot ? w.hr.s : w.h.s;
        for (int f = 0; f < 2; ++f)
            (f ? publish_fused : publish) =
                rw(b, RAD_OP(b, "moe_mailbox_publish",
                             RAD_PARAMS(RAD_INT("layer", layer), RAD_RANGE("M", 1, c.rows),
                                        RAD_INT("n", g.n_embd), RAD_INT("top_k", c.top_k),
                                        RAD_INT("expert_base", c.expert_base),
                                        RAD_INT("n_expert", c.n_expert), RAD_INT("fused", f)),
                             RAD_NOWEIGHTS),
                   {w.ids, w.ew, w.h.x, q, s, q}, {w.ids});
        add = RAD_OP(b, "moe_mailbox_add",
                     RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd), RAD_INT("top_k", c.top_k)),
                     RAD_NOWEIGHTS);
        if (add) {
            const rad_buf rd[5] = { w.sorted, w.edn, w.ew, w.sout, w.sgate };
            const rad_buf wr[1] = { w.h.x };
            rad_op_reads(b, add, rd, m.gfold ? 5 : 3);
            rad_op_writes(b, add, wr, 1);
        }
        inject = RAD_OP(b, "moe_mailbox_inject",
                        RAD_PARAMS(RAD_INT("layer", layer), RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd),
                                   RAD_INT("top_k", c.top_k), RAD_INT("n_expert", c.n_expert)),
                        RAD_NOWEIGHTS);
        if (inject) {
            const rad_buf rd[4] = { w.sorted, w.edn, w.ew, w.eoff };
            const rad_buf wr[3] = { w.sorted, w.edn, w.ew };
            rad_op_reads(b, inject, rd, 4);
            rad_op_writes(b, inject, wr, 3);
        }
        if (!publish || !publish_fused || !add || !inject) {
            std::fprintf(stderr, "radiance: MoE remote hand-off is on but its mailbox ops are not declared at layer "
                         "%d (publish %d/%d merge %d inject %d; q %d s %d rot %d); is libr3moe loaded?\n", layer,
                         publish != 0, publish_fused != 0, add != 0, inject != 0, q != 0, s != 0, (int)c.expert_rot);
            return RAD_E_INVAL;
        }
        return RAD_OK;
    }

    /* After the router and scatter of a pass, before the grouped GEMMs: publish the routes and re-scatter, so
     * the helper-owned routes are masked out of the local sort. In mask-only (diagnostic) the publish comes
     * later, from the diagnostic hook. */
    template <class Moe>
    void after_route(RadCtx* c_, const Moe& m, int64_t T, int64_t r0, int64_t rows, const RadOperand& ecnt) const {
        if (!policy(T) || bc_moe_remote_mask_only()) return;
        const auto& c = m.c;
        const auto& g = m.g;
        const auto& w = m.w;
        const int64_t k = c.top_k;
        const rad_buf q = c.expert_rot ? w.hr.q : w.h.q;
        const rad_buf s = c.expert_rot ? w.hr.s : w.h.s;
        RAD_ISSUE_N(c_, fused(m, T) ? publish_fused : publish, rows,
                    brow_slice(w.ids, r0, rows, k),
                    brow_slice(w.ew, r0, rows, k),
                    brow_slice(w.h.x, r0, rows, g.n_embd),
                    brow_slice(q, r0, rows, g.n_embd),
                    brow_slice(s, r0, rows, g.n_embd / RAD_FP8_BLOCK),
                    brows(q, T));
        if (active(T))
            RAD_ISSUE_N(c_, m.op_scatter, rows, brow_slice(w.ids, r0, rows, k),
                        brows(w.sorted, rows * k), brows(w.eoff, c.n_expert + 1), ecnt);
    }

    /* In place of the gather. When the write behind the block takes the gather this step (the fused MoE gather +
     * reducing hc write of a TP engine; the same gather_taken the block asks), the helper rows join that gather:
     * moe_mailbox_inject on the layer's consumer rank, nothing on the others, and the block gathers nothing.
     * The inject overwrites the injected slots' router weights with 1, so the routing print is taken before it.
     * Otherwise one slot-order merge of the local rows and the helper's rows (and the shared arm when folded) into
     * the gather's output (`moe` when a separate FP32 combine follows). Returns whether it ran; false leaves the
     * normal gather. Every rank takes the same branch (it depends on T and the declared bounds only). */
    template <class Moe>
    bool merge(RadCtx* c_, const Moe& m, int64_t T, int64_t r0, int64_t rows) const {
        if (!active(T)) return false;
        const auto& c = m.c;
        const auto& g = m.g;
        const auto& w = m.w;
        const int64_t k = c.top_k;
        if (refuses(m, T)) {
            rad_step_fail(c_, "radiance: MoE remote hand-off: a fused MoE gather behind a block with a separate FP32 "
                              "combine cannot take the helper rows exactly once");
            return true;
        }
        if (fused(m, T)) {
            moe_debug_weights(c_, m.layer, k, w.ids, w.ew, c.sigmoid_router != 0);   /* off: one static check */
            RAD_ISSUE_N(c_, inject, rows,
                        brows(w.sorted, rows * k), brows(w.edn, rows * k),
                        brow_slice(w.ew, r0, rows, k), brows(w.eoff, c.n_expert + 1));
            return true;
        }
        RAD_ISSUE_N(c_, add, rows,
                    brows(w.sorted, rows * k), brows(w.edn, rows * k),
                    brow_slice(w.ew, r0, rows, k),
                    m.gfold ? brow_slice(w.sout, r0, rows, g.n_embd) : RAD_NONE,
                    m.gfold ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE,
                    brow_slice(m.op_combine ? w.moe : w.h.x, r0, rows, g.n_embd));
        return true;
    }
};

}  /* namespace arch */
}  /* namespace rad */

#endif /* BC_MOE_REMOTE_BLOCK_H */
"""

PATCHES = (
    FilePatch(
        path="abi/bc_moe_remote.h",
        create=True,
        description="Create the BigCherry remote-MoE extension-point ABI.",
        edits=(Edit(
            id="create-bc-moe-remote-abi",
            anchor=r"\A",
            text=_BC_MOE_REMOTE_H,
            guard=r"^int bc_moe_remote_wanted\(int64_t step_rows\);$",
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="core/place/bc_moe_remote.cpp",
        create=True,
        description="Create the remote-MoE policy: step gate, plugin registry, layer maps, SKIP_OWNED rule.",
        edits=(Edit(
            id="create-bc-moe-remote-core",
            anchor=r"\A",
            text=_BC_MOE_REMOTE_CPP,
            guard=r'^extern "C" int bc_moe_remote_wanted\(',
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="core/place/bc_moe_remote_place.h",
        create=True,
        description="Create the planner / bring-up helpers of the remote-MoE extension point.",
        edits=(Edit(
            id="create-bc-moe-remote-place",
            anchor=r"\A",
            text=_BC_MOE_REMOTE_PLACE_H,
            guard=r"^inline bool bc_moe_remote_pin_unit\(",
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="arch/common/bc_moe_remote_block.h",
        create=True,
        description="Create the MoE block's side of the remote-MoE extension point.",
        edits=(Edit(
            id="create-bc-moe-remote-block",
            anchor=r"\A",
            text=_BC_MOE_REMOTE_BLOCK_H,
            guard=r"^struct BcMoeRemote \{$",
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="arch/common/rad_block_moe_fp8.h",
        description="Call the remote-MoE extension point from the MoE block: declare, publish, merge, all-reduce.",
        edits=(
            Edit(
                id="include-bc-moe-remote-block",
                anchor=r'^#include "rad_device\.h"$',
                text='\n#include "bc_moe_remote_block.h"',
                guard=r'^#include "bc_moe_remote_block\.h"$',
                expect_matches=1,
                rationale="The device include is unique in this header and sits with architecture dependencies.",
            ),
            Edit(
                id="bc-remote-member",
                anchor=r"^[ \t]*rad_op op_ar = 0, op_add = 0;$",
                text="\n    BcMoeRemote bc_remote;",
                guard=r"^[ \t]*BcMoeRemote bc_remote;$",
                expect_matches=1,
                rationale="The output op declaration is the unique home of the block's per-layer mailbox state.",
            ),
            Edit(
                id="bc-remote-declare",
                anchor=r"^[ \t]*\{w\.x, w\.h\.x\}, \{w\.x\}\);$",
                text="\n    RAD_ARCH_TRY(bc_remote.declare(b, *this, reg, l));",
                guard=r"RAD_ARCH_TRY\(bc_remote\.declare\(b, \*this, reg, l\)\);",
                expect_matches=1,
                rationale="The end of the residual add declaration: reg[] (the runtime slots) is final and the gathered output buffers exist.",
            ),
            Edit(
                id="bc-remote-after-route",
                anchor=r"^[ \t]*const int64_t n4\[4\] = \{",
                mode="insert_before",
                text="    bc_remote.after_route(c_, *this, T, r0, rows, ecnt);\n",
                guard=r"bc_remote\.after_route\(c_, \*this, T, r0, rows, ecnt\);",
                expect_matches=1,
                rationale="n4 is the first routed-GEMM code after the router, scatter and rotated quantisation, so routes and the exact q/s operand are ready.",
            ),
            Edit(
                id="bc-remote-merge",
                anchor=r"^[ \t]*if \(op_combine \|\| !gather_taken\(g, T, gather_out_rows, gather_out_rows6\)\)$",
                mode="replace",
                text="    if (!bc_remote.merge(c_, *this, T, r0, rows) && (op_combine || !gather_taken(g, T, gather_out_rows, gather_out_rows6)))",
                guard=r"if \(!bc_remote\.merge\(c_, \*this, T, r0, rows\) &&",
                expect_matches=1,
                rationale="The pass's single gather guard: a handed-off step merges instead of gathering.",
            ),
            Edit(
                id="bc-remote-issues-ar",
                anchor=r"^[ \t]*return op_ar && !ar_taken\(g, ar_out_take != kArOutNorm \? T : rows, ar_out, ar_out_take\);$",
                mode="replace",
                text="        return op_ar && (bc_remote.forces_ar(*this, T) || !ar_taken(g, ar_out_take != kArOutNorm ? T : rows, ar_out, ar_out_take));",
                guard=r"return op_ar && \(bc_remote\.forces_ar\(\*this, T\) \|\|",
                expect_matches=1,
                rationale="issues_ar is the one place every caller (pass and the pipelined prefill callers) asks whether the block runs its own all-reduce; a handed-off step that does not fuse must when a HyperConn write follows, since that write then leaves the reduction (RAD49).",
            ),
        ),
    ),
    FilePatch(
        path="arch/common/rad_block_hc.h",
        description="Keep HyperConn's fused gather / reducing write off on a handed-off step of a MoE-fed connection that does not fuse.",
        edits=(
            Edit(
                id="include-bc-moe-remote-block-hc",
                anchor=r'^#include "rad_fp8\.h"$',
                text='\n#include "bc_moe_remote_block.h"',
                guard=r'^#include "bc_moe_remote_block\.h"$',
                expect_matches=1,
                rationale="The fp8 include is the unique architecture dependency line in this header.",
            ),
            Edit(
                id="bc-remote-takes-ar",
                anchor=r"^[ \t]*bool takes_ar\(int64_t T\) const \{ return ar_write\(T\) != 0; \}$",
                mode="replace",
                text="    bool takes_ar(int64_t T) const { return ar_write(T) != 0 && !bc_moe_remote_hc_open(gsrc.top_k, g, T, gather_rows, gather_rows6); }",
                guard=r"!bc_moe_remote_hc_open\(gsrc\.top_k, g, T, gather_rows, gather_rows6\)",
                expect_matches=1,
                rationale="write() reaches the gather form through takes_gather() = takes_ar() && ..., and the reducing write through takes_ar(), so this one predicate closes both for a MoE-fed connection.",
            ),
        ),
    ),
    FilePatch(
        path="core/place/planner.cpp",
        description="Leave helper-owned expert units unplaced (mapped, never read) when nothing local reads them.",
        edits=(
            Edit(
                id="include-bc-moe-remote-place-planner",
                anchor=r'^#include "planner\.h"$',
                text='\n#include "bc_moe_remote_place.h"',
                guard=r'^#include "bc_moe_remote_place\.h"$',
                expect_matches=1,
                rationale="The planner's own header include is its unique first include line.",
            ),
            Edit(
                id="bc-remote-pin-unit",
                anchor=r"^[ \t]*bool host_only = false, stored = true;$",
                mode="insert_before",
                text="        if (bc_moe_remote_pin_unit(mu, in_.config->max_tok)) continue;\n",
                guard=r"if \(bc_moe_remote_pin_unit\(mu, in_\.config->max_tok\)\) continue;",
                expect_matches=1,
                rationale="This declaration opens pin_mapped's per-unit loop body; a helper-owned unit is decided before any host or n-gram logic and leaves the loop.",
            ),
            Edit(
                id="bc-remote-mark-footprint",
                anchor=r"^    for \(U& mu : units\) mu\.mapped = mapped_unit\(mu\.mapped, container && mu\.direct, host_backend\);$",
                text="\n    bc_moe_remote_mark_footprint(units);",
                guard=r"bc_moe_remote_mark_footprint\(units\);",
                expect_matches=1,
                rationale="weight_footprint's unit pass ends with this mapped resolution; the budgets read mu.mapped right after it.",
            ),
            Edit(
                id="rank-units-skip-preplaced",
                anchor=r"^    for \(int32_t i = 0; i < \(int32_t\)p->units\.size\(\); \+\+i\) order->push_back\(i\);$",
                mode="replace",
                text="    bc_moe_remote_rank_units(p, order);",
                guard=r"bc_moe_remote_rank_units\(p, order\);",
                expect_matches=1,
                rationale="rank_units' order is what the expert-tiered fill walks; leaving preplaced units out keeps a helper-owned expert on the mapped tier. The extension owns the preplaced filter.",
            ),
        ),
    ),
    FilePatch(
        path="core/engine_bringup.cpp",
        description="Publish helper-owned expert units as never-read placeholders instead of loading them.",
        edits=(
            Edit(
                id="include-bc-moe-remote-place-bringup",
                anchor=r'^#include "engine_priv\.h"$',
                text='\n#include "place/bc_moe_remote_place.h"',
                guard=r'^#include "place/bc_moe_remote_place\.h"$',
                expect_matches=1,
                rationale="engine_priv.h is the bring-up's unique first include line.",
            ),
            Edit(
                id="bc-remote-publish-placeholder",
                anchor=r"^[ \t]*if \(\(w\.tier == Tier::Mapped \|\| w\.tier == Tier::SSD\) && \(!direct \|\| !e\)\) \{$",
                mode="insert_before",
                text="            if (bc_moe_remote_publish_placeholder(cfg_.max_tok, w, e, *file_, r->mover, (rad_weight)i)) continue;\n",
                guard=r"if \(bc_moe_remote_publish_placeholder\(cfg_\.max_tok, w, e, \*file_, r->mover, \(rad_weight\)i\)\) continue;",
                expect_matches=1,
                rationale="The single tier / stored-form check in the weight publish loop; a helper-owned placeholder is published before it, since nothing reads its bytes and its relayouted form would fail the check.",
            ),
        ),
    ),
    FilePatch(
        path="core/runtime/ctx.cpp",
        description="Bind every trunk/draft/issued-ahead pass to its parent ticket before tape capture or replay.",
        edits=(
            Edit(
                id="include-bc-moe-remote-ctx",
                anchor=r'^#include "ctx\.h"$',
                text='\n#include "bc_moe_remote.h"',
                guard=r'^#include "bc_moe_remote\.h"$',
                expect_matches=1,
                rationale="The context's own unique include is the stable place for the ticket hook ABI.",
            ),
            Edit(
                id="bc-remote-ticket-before-pass",
                anchor=r"^        const int rc = run_pass\(b\);\n        if \(rc < 0 && step_status_ >= 0\) step_status_ = rc;$",
                mode="replace",
                text="""        /* The rank thread installed this parent's immutable admission
         * ticket before releasing this pass. Never derive it from b->step. */
        const int rc = bc_moe_remote_before_pass(rank_, bc_moe_remote_current_ticket(), b->n_tok, stream_) < 0 ? RAD_E_STATE : run_pass(b);
        if (rc < 0 && step_status_ >= 0) step_status_ = rc;""",
                guard=r"const int rc = bc_moe_remote_before_pass\(rank_, bc_moe_remote_current_ticket\(\), b->n_tok, stream_\) < 0 \? RAD_E_STATE : run_pass\(b\);",
                expect_matches=1,
                rationale="Only this call enters capture/replay; the setter goes on lane 0 before every pass, even the first live recording.",
            ),
        ),
    ),
    FilePatch(
        path="core/place/bc_moe_remote_engine.h",
        create=True,
        description="Own the scheduler/rank admission bridge.",
        edits=(Edit(
            id="create-bc-moe-remote-engine",
            anchor=r"\A",
            text=_BC_MOE_REMOTE_ENGINE_H,
            guard=r"^class BcMoeRemoteAdmission \{$",
            expect_matches=1,
            rationale="Owned extension file; reject any upstream collision.",
        ),),
    ),
    FilePatch(
        path="core/engine.h",
        description="Own the scheduler/rank admission bridge.",
        edits=(
            Edit(
                id="bc-remote-engine-ticket-include",
                anchor=r'^#include "rad_core\.h"$',
                text='\n#include "place/bc_moe_remote_engine.h"',
                guard=r'^#include "place/bc_moe_remote_engine\.h"$',
                expect_matches=1,
                rationale="Unique ABI include; includes only the extension state owner.",
            ),
            Edit(
                id="bc-remote-engine-ticket-state",
                anchor=r"^    std::atomic<const RadBatch\*>  step_batch_\{nullptr\};$",
                text="\n    BcMoeRemoteAdmission bc_remote_admission_;",
                guard=r"BcMoeRemoteAdmission bc_remote_admission_;",
                expect_matches=1,
                rationale="Extension owns state, published through the existing rank barrier.",
            ),
        ),
    ),
    FilePatch(
        path="core/engine.cpp",
        description="Thin scheduler admission, terminal completion and burst hooks.",
        edits=(
            Edit(
                id="bc-remote-rank-ticket",
                anchor=r"^            int s = what == kIssueSampler \? RAD_OK : r\.ctx\.run_step\(b\);$",
                mode="insert_before",
                text="            bc_moe_remote_rank_ticket(bc_remote_admission_.issued_ticket());\n",
                guard=r"bc_moe_remote_rank_ticket\(bc_remote_admission_\.issued_ticket\(\)\);",
                expect_matches=1,
                rationale="Explicit immutable ticket transfer before every rank model/draft release.",
            ),
            Edit(
                id="bc-remote-ahead-try-admit",
                anchor=r"^    if \(!pipelining_\) return RAD_OK;$",
                text="\n    if (!bc_remote_admission_.try_ahead((int)ranks_.size())) return bc_remote_admission_.ahead_status();",
                guard=r"bc_remote_admission_\.try_ahead\(",
                expect_matches=1,
                rationale="Nonblocking call before step_next, advance_all and prepare_release.",
            ),
            Edit(
                id="bc-remote-ahead-bind",
                anchor=r"^    if \(!nb\) return RAD_OK;$",
                mode="insert_before",
                text="    if (bc_remote_admission_.bind_ahead(nb) < 0) return RAD_E_STATE;\n",
                guard=r"bc_remote_admission_\.bind_ahead\(nb\)",
                expect_matches=1,
                rationale="Cancel an unused reservation or bind it before rank release.",
            ),
            Edit(
                id="bc-remote-scheduler-admit",
                anchor=r"^        if \(issued\) sched_\.promote_next\(\);\n        else        b = sched_\.step\(\);$",
                mode="replace",
                text="        if (bc_remote_admission_.step(issued, b, sched_, (int)ranks_.size()) < 0) return fail_loop(RAD_E_STATE);",
                guard=r"bc_remote_admission_\.step\(issued, b, sched_,",
                expect_matches=1,
                rationale="Admission precedes scheduler mutation, checkpoint restore and prepare_release.",
            ),
            Edit(
                id="bc-remote-terminal-events",
                anchor=r"^        st = mark_step_end\(b, so\);\n        if \(st < 0\) return fail_loop\(st\);$",
                text="\n        if (bc_remote_admission_.seal(ranks_) < 0) return fail_loop(RAD_E_DEVICE);",
                guard=r"bc_remote_admission_\.seal\(ranks_\)",
                expect_matches=1,
                rationale="Distinct parent events after all parent GPU work, before issue-ahead.",
            ),
            Edit(
                id="bc-remote-host-logical-completion",
                anchor=r"^        sched_\.step_finished\(\);$",
                mode="insert_before",
                text="        if (bc_remote_admission_.host_done() < 0) return fail_loop(RAD_E_STATE);\n",
                guard=r"bc_remote_admission_\.host_done\(\)",
                expect_matches=1,
                rationale="Host completion only after finish_block_drafter/finish_serial_drafter returns.",
            ),
            Edit(
                id="bc-remote-shutdown-interrupt",
                anchor=r"^    stopping_\.store\(true, std::memory_order_release\);$",
                mode="insert_before",
                text="    bc_moe_remote_shutdown();\n",
                guard=r"bc_moe_remote_shutdown\(\);",
                expect_matches=2,
                rationale="Both fault and normal shutdown wake admission/drain before joining stepper.",
            ),
            Edit(
                id="include-bc-moe-remote-engine",
                anchor=r'^#include "engine_priv\.h"$',
                text='\n#include "bc_moe_remote.h"',
                guard=r'^#include "bc_moe_remote\.h"$',
                expect_matches=1,
                rationale="engine_priv.h is engine.cpp's unique first include line.",
            ),
            Edit(
                id="bc-remote-engine-idle",
                anchor=r"^            step_batch_\.store\(nullptr, std::memory_order_release\);$",
                text="\n            bc_moe_remote_engine_idle();",
                guard=r"bc_moe_remote_engine_idle\(\);",
                expect_matches=1,
                rationale="The engine loop's idle branch clears the step batch here when the scheduler has no step: every earlier step has joined, so a burst ends.",
            ),
            Edit(
                id="bc-remote-engine-step",
                anchor=r"^        idle = 0;$",
                mode="insert_before",
                text="        bc_moe_remote_engine_step();\n",
                guard=r"bc_moe_remote_engine_step\(\);",
                expect_matches=1,
                rationale="The loop resets its idle backoff exactly when the scheduler returned a step, before the ranks are released or any recorded pass is replayed.",
            ),
        ),
    ),
)
