"""Radiance 1.7.0: GPU-lane remote whole-expert MoE hand-off, as thin hooks into one BigCherry-owned extension point.

All policy lives in files this patch creates (abi/bc_moe_remote.h, core/place/bc_moe_remote.cpp,
core/place/bc_moe_remote_place.h, arch/common/bc_moe_remote_block.h) and in the libr3moe plugin. The edits to upstream
files are one-line guarded call sites into them. The shadow compare and mask-only diagnostics are rad0004.
"""

from bigcherry.patcher import Edit, FilePatch

_BC_MOE_REMOTE_H = r"""/* bc_moe_remote.h -- BigCherry's extension point for GPU-lane remote whole-expert MoE compute.
 *
 * One engine owns one BcMoeRemoteGeneration (opaque here). The engine opens it through an explicit factory the
 * already-loaded libr3moe plugin exports (BC_MOE_REMOTE_OPEN_SYMBOL, found with dlsym on the plugin's own handle),
 * hands it to every consumer it builds, and closes it before the plugin is unloaded. There is no global plugin
 * registry, no registration by a loader constructor and no thread-local owner: a consumer holds the generation it was
 * given, and a closed or foreign generation refuses every call with BC_MOE_REMOTE_STALE.
 *
 * The one process-wide structure is a slot-keyed table of NON-OWNING generation pointers, written when a generation
 * opens and cleared when it closes. It exists only so code that is given nothing but a build context (an architecture
 * plugin's declare, the planner) can find its engine's generation at init time: bc_moe_remote_lookup(rad_build_slot(ctx)).
 * Nothing on a per-step path takes it.
 *
 * Publication order: open validates the slot span and finds every slot empty, then calls the plugin factory, then
 * publishes the generation into each slot with a compare-and-swap from empty (release); lookup is an acquire load.
 * Close clears the slots (compare-and-swap from this generation) before it calls the plugin's close, so a lookup never
 * returns a generation whose plugin state is being torn down. Every function is exported by the engine binary and bound by
 * the arch plugins at dlopen; a NULL generation means "no remote hand-off" and every query answers it. */
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

/* Typed refusals (always negative; zero is success). */
enum {
    BC_MOE_REMOTE_OK = 0,
    BC_MOE_REMOTE_REFUSED = -1, /* the plugin or the arguments refused */
    BC_MOE_REMOTE_STALE = -2,   /* the generation is closed: its plugin state is gone */
    BC_MOE_REMOTE_BUSY = -3,    /* a slot of the span is bound already, or the plugin's helper is live */
    BC_MOE_REMOTE_ABSENT = -4   /* the manifest asks for a hand-off and the plugin offers no factory */
};

typedef struct BcMoeRemoteGeneration BcMoeRemoteGeneration;
/* The lookup table covers the engine's instance slots (RAD_MAX_SLOTS); the engine bring-up static_asserts the two agree. */
#define BC_MOE_REMOTE_MAX_SLOTS 32

/* What the engine tells the plugin factory about the generation it is opening. */
typedef struct BcMoeRemoteOpenArgs {
    uint32_t struct_size;
    uint32_t capacity_limit; /* the admission ledger's ceiling: route-slot ring extent plus issue-ahead depth */
    int32_t slot_base;       /* RadBuildCtx::slot_base of the opening engine */
    int32_t ranks;           /* its tensor-parallel width */
} BcMoeRemoteOpenArgs;

/* Filled by the factory. struct_size must equal sizeof(BcMoeRemotePlugin) of the engine's header. `self` is the
 * plugin's state for this generation and the first argument of every callback. */
typedef struct BcMoeRemotePlugin {
    /* The head every table starts with, whatever its size: struct_size, self, close. The engine presets struct_size to
     * its own size before calling the factory; a factory refuses (and allocates nothing) when it differs. */
    uint32_t struct_size;
    void* self;
    /* The generation ends: release everything `self` holds. Called once, after the engine stopped every consumer. */
    void (*close)(void* self);
    /* A layer's runtime slot map (slot_of_expert[e] < 0: expert e is a protected, locally served expert). */
    void (*layer_map)(void* self, int layer, int n_expert, const int32_t* slot_of_expert);
    /* Whether a helper owns routed expert (layer, expert). */
    int (*owns_expert)(void* self, int layer, int expert);
    /* An inference burst begins (1: the engine loop has a step after being idle) or ends (0: the loop found no
     * step). Called from the engine's host scheduler loop, so it fires whether a pass is recorded or replayed;
     * a helper arms its GPU waits only inside a burst, and its queues drain to empty between bursts. */
    void (*burst)(void* self, int active);
    /* Per-pass stream-ordered ticket setter; returns a negative RAD_E_* on failure. The engine calls this on lane 0
     * before capture/replay with the immutable scheduler-parent admission value its rank thread handed to the Ctx. */
    int (*before_pass)(void* self, int rank, struct BcMoeRemoteTicket ticket, void* stream);
    int (*admit)(void* self, struct BcMoeRemoteTicket* out, int try_only, int world_size);
    int (*bind)(void* self, struct BcMoeRemoteTicket ticket, int64_t scheduler_step);
    int (*cancel)(void* self, struct BcMoeRemoteTicket ticket);
    int (*host_done)(void* self, struct BcMoeRemoteTicket ticket);
    int (*rank_done)(void* self, int rank, struct BcMoeRemoteTicket ticket, void* stream);
    /* A fault stop only: interrupt every blocked wait so each fails now. A clean stop never calls it, and the plugin
     * decides at `close` whether the ledger drained cleanly. */
    void (*fault)(void* self);
    void (*finish_admission)(void* self);
    int (*wait_finished)(void* self, int64_t timeout_ms);
    /* The longest the engine waits for its stepper and the bound parents to finish at stop: the plugin's protocol
     * bound for draining to quiet (its quiesce deadline), so the engine holds no budget of its own. */
    int64_t stop_budget_ms;
    /* The generation's never-reused serial: the engine puts it in every mailbox op it declares, and the op's init binds
     * to exactly this generation (not merely to a numeric group). */
    uint64_t owner;
} BcMoeRemotePlugin;
typedef int (*BcMoeRemoteOpenFn)(const BcMoeRemoteOpenArgs* args, BcMoeRemotePlugin* out);
#define BC_MOE_REMOTE_OPEN_SYMBOL "bc_moe_remote_plugin_open"

/* Open the engine's generation. With the manifest asking for no hand-off (or a diagnostic mode a production build
 * cannot honour) *out is NULL and the result is BC_MOE_REMOTE_OK: every query then answers "no remote". Otherwise the
 * factory is required (open_with takes it directly, open finds it on the plugin handle; a NULL handle has none).
 * A slot of [slot_base, slot_base+ranks) that is out of range or bound refuses before the factory runs. */
int bc_moe_remote_open(void* plugin_handle, int slot_base, int ranks, uint32_t capacity_limit,
                       BcMoeRemoteGeneration** out);
int bc_moe_remote_open_with(BcMoeRemoteOpenFn factory, int slot_base, int ranks, uint32_t capacity_limit,
                            BcMoeRemoteGeneration** out);
/* Explicit lifecycle: close unbinds the slots and ends the plugin state (idempotent); destroy closes and frees. A closed
 * generation stays allocated until destroy so a stale holder is refused, not dereferenced into freed memory. */
void bc_moe_remote_close(BcMoeRemoteGeneration* g);
void bc_moe_remote_destroy(BcMoeRemoteGeneration* g);
/* The generation bound to an instance slot, or NULL (unclaimed, cleared, out of range). Init-time only. */
BcMoeRemoteGeneration* bc_moe_remote_lookup(int slot);

/* Runtime step gate: remote transport's typed manifest max_rows (min is zero), captured when the generation opened. */
int bc_moe_remote_wanted(const BcMoeRemoteGeneration* g, int64_t step_rows);
int bc_moe_remote_wanted_upto(const BcMoeRemoteGeneration* g, int64_t max_step_rows);
/* Diagnostics: always 0 unless the engine is built with BC_MOE_REMOTE_DIAG (patch rad0004). A production build
 * that finds R3_MOE_WORKER_SHADOW or R3_MOE_WORKER_MASK_ONLY set opens no generation and says so. */
int bc_moe_remote_shadow(const BcMoeRemoteGeneration* g);
int bc_moe_remote_mask_only(const BcMoeRemoteGeneration* g);
int bc_moe_remote_diag_build(void);
/* The engine loop calls engine_step for every step it runs and engine_idle when it finds no step; the
 * transitions between the two are forwarded to the plugin's burst callback, once each. */
void bc_moe_remote_engine_step(BcMoeRemoteGeneration* g);
void bc_moe_remote_engine_idle(BcMoeRemoteGeneration* g);
/* Admission is invoked before sched_.step() / prepare_release; try_only is nonblocking for issue_ahead (1: admission
 * closed/full; skip the run-ahead). With no generation the returned ticket is zero and calls are no-ops. */
int bc_moe_remote_admit(BcMoeRemoteGeneration* g, BcMoeRemoteTicket* ticket, int try_only, int world_size);
int bc_moe_remote_bind(BcMoeRemoteGeneration* g, BcMoeRemoteTicket ticket, int64_t scheduler_step);
/* Returns an unused tail reservation; negative when the plugin rejects it (the ledger has faulted). */
int bc_moe_remote_cancel(BcMoeRemoteGeneration* g, BcMoeRemoteTicket ticket);
int bc_moe_remote_host_done(BcMoeRemoteGeneration* g, BcMoeRemoteTicket ticket);
void bc_moe_remote_fault(BcMoeRemoteGeneration* g);
int bc_moe_remote_finish_admission(BcMoeRemoteGeneration* g);
int bc_moe_remote_wait_finished(BcMoeRemoteGeneration* g, int64_t timeout_ms);
/* The stop budget the plugin declared (BcMoeRemotePlugin::stop_budget_ms); 0 for no generation. */
int64_t bc_moe_remote_stop_budget_ms(const BcMoeRemoteGeneration* g);
/* The owner serial for the generation's mailbox ops; 0 for no generation or a closed one. */
uint64_t bc_moe_remote_owner(const BcMoeRemoteGeneration* g);
int bc_moe_remote_before_pass(BcMoeRemoteGeneration* g, int rank, BcMoeRemoteTicket ticket, void* stream);
int bc_moe_remote_rank_done(BcMoeRemoteGeneration* g, int rank, BcMoeRemoteTicket ticket, void* stream);
/* Block declare publishes a layer's map here; it is forwarded to the plugin and kept for the placement rule. */
void bc_moe_remote_layer_map(BcMoeRemoteGeneration* g, int layer, int n_expert, const int32_t* slot_of_expert);
/* Placement (manifest handoff.transport.skip_owned): whether the engine may leave a routed expert unplaced because the
 * helper computes it and nothing local ever reads it. */
int bc_moe_remote_skip_unit(const BcMoeRemoteGeneration* g, int layer, int expert, int64_t max_step_rows);
#ifdef __cplusplus
}
#endif
#endif /* BC_MOE_REMOTE_H */
"""

_BC_MOE_REMOTE_CPP = r"""/* bc_moe_remote.cpp -- policy behind abi/bc_moe_remote.h: the engine-owned generation, its slot-keyed init-time lookup,
 * the decode-sized step gate, the layer-map store and the SKIP_OWNED placement rule. No process-wide plugin, map or
 * burst state: all of it lives in a generation, and the one table (slot -> non-owning pointer) is written on open and
 * close only. */
#include "bc_moe_remote.h"
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <map>
#include <condition_variable>
#include <mutex>
#include <vector>

static_assert(sizeof(BcMoeRemoteTicket) == 16);

#ifdef BC_MOE_REMOTE_DIAG
static constexpr bool kDiagBuild = true;
#else
static constexpr bool kDiagBuild = false;
#endif

// Provided by rad0070 (the managed native entry), a declared dependency of this package.
extern "C" int bc_serving_remote_settings(int64_t*, uint32_t*, uint32_t*, uint32_t*);

struct BcMoeRemoteGeneration {
    std::atomic<int> state{1}; /* 1 open, 2 closed */
    BcMoeRemotePlugin vt{};
    /* From rad0070's installed admission, read once when the generation opens: the generation's immutable policy.
     * No environment fallback. */
    int64_t max_rows = 0;
    uint32_t skip_owned = 0, shadow = 0, mask_only = 0;
    int slot_base = 0, ranks = 0;
    std::atomic<bool> burst_active{false};
    mutable std::mutex maps_mu; /* init time only: declare publishes, the planner reads */
    std::map<int, std::vector<int32_t>> layer_maps;

    bool open() const { return state.load(std::memory_order_acquire) == 1; }
};

namespace {
std::atomic<BcMoeRemoteGeneration*> g_slot_table[BC_MOE_REMOTE_MAX_SLOTS];

BcMoeRemoteGeneration* live(BcMoeRemoteGeneration* g) { return g && g->open() ? g : nullptr; }
}  // namespace

extern "C" BcMoeRemoteGeneration* bc_moe_remote_lookup(int slot) {
    if (slot < 0 || slot >= BC_MOE_REMOTE_MAX_SLOTS) return nullptr;
    return g_slot_table[slot].load(std::memory_order_acquire);
}

extern "C" int bc_moe_remote_open_with(BcMoeRemoteOpenFn factory, int slot_base, int ranks, uint32_t capacity_limit,
                                       BcMoeRemoteGeneration** out) {
    if (!out) return BC_MOE_REMOTE_REFUSED;
    *out = nullptr;
    if (ranks < 1 || slot_base < 0 || slot_base + ranks > BC_MOE_REMOTE_MAX_SLOTS) return BC_MOE_REMOTE_REFUSED;
    int64_t max_rows = 0;
    uint32_t skip_owned = 0, shadow = 0, mask_only = 0;
    (void)bc_serving_remote_settings(&max_rows, &skip_owned, &shadow, &mask_only);
    /* Asked for in a production build the diagnostic modes must not be silently ignored (that would measure the wrong
     * thing): no generation opens, so every expert computes locally, and it is said. */
    if (!kDiagBuild && (shadow || mask_only)) {
        std::fprintf(stderr, "radiance: R3_MOE_WORKER_SHADOW / R3_MOE_WORKER_MASK_ONLY need the diagnostic build "
                             "(BC_MOE_REMOTE_DIAG, patch rad0004); remote MoE hand-off is off and every expert "
                             "computes locally\n");
        return BC_MOE_REMOTE_OK;
    }
    if (max_rows <= 0) return BC_MOE_REMOTE_OK;
    if (!factory) return BC_MOE_REMOTE_ABSENT;
    /* Refuse a bound slot before the factory has side effects. */
    for (int i = 0; i < ranks; ++i)
        if (g_slot_table[slot_base + i].load(std::memory_order_acquire)) return BC_MOE_REMOTE_BUSY;

    auto* g = new BcMoeRemoteGeneration;
    g->max_rows = max_rows;
    g->skip_owned = skip_owned;
    g->shadow = shadow;
    g->mask_only = mask_only;
    g->slot_base = slot_base;
    g->ranks = ranks;
    const BcMoeRemoteOpenArgs args{sizeof(BcMoeRemoteOpenArgs), capacity_limit, slot_base, ranks};
    /* Failure-atomic negotiation. The engine presets the table size it expects. A factory that fails has released
     * everything it allocated itself (its close is never called). A factory that succeeds has handed over its state; if
     * the table it returned is the wrong size the engine closes that state, once, through the head every table starts
     * with. A success that does not even fill the head leaves nothing the engine can close and is refused. */
    g->vt.struct_size = sizeof(BcMoeRemotePlugin);
    const int rc = factory(&args, &g->vt);
    if (rc != 0 || g->vt.struct_size != sizeof(BcMoeRemotePlugin)) {
        if (rc == 0 && g->vt.struct_size >= offsetof(BcMoeRemotePlugin, close) + sizeof(g->vt.close) && g->vt.close)
            g->vt.close(g->vt.self);
        delete g;
        return rc < 0 ? rc : BC_MOE_REMOTE_REFUSED;
    }
    for (int i = 0; i < ranks; ++i) {
        BcMoeRemoteGeneration* none = nullptr;
        if (g_slot_table[slot_base + i].compare_exchange_strong(none, g, std::memory_order_release,
                                                                std::memory_order_relaxed))
            continue;
        for (int j = 0; j < i; ++j) {
            BcMoeRemoteGeneration* mine = g;
            g_slot_table[slot_base + j].compare_exchange_strong(mine, nullptr, std::memory_order_release,
                                                                std::memory_order_relaxed);
        }
        if (g->vt.close) g->vt.close(g->vt.self);
        delete g;
        return BC_MOE_REMOTE_BUSY;
    }
    *out = g;
    return BC_MOE_REMOTE_OK;
}

extern "C" int bc_moe_remote_open(void* plugin_handle, int slot_base, int ranks, uint32_t capacity_limit,
                                  BcMoeRemoteGeneration** out) {
    BcMoeRemoteOpenFn factory = nullptr;
    if (plugin_handle) {
        (void)dlerror();
        factory = reinterpret_cast<BcMoeRemoteOpenFn>(dlsym(plugin_handle, BC_MOE_REMOTE_OPEN_SYMBOL));
        if (dlerror()) factory = nullptr;
    }
    return bc_moe_remote_open_with(factory, slot_base, ranks, capacity_limit, out);
}

extern "C" void bc_moe_remote_close(BcMoeRemoteGeneration* g) {
    if (!g) return;
    int expected = 1;
    if (!g->state.compare_exchange_strong(expected, 2, std::memory_order_acq_rel)) return;
    for (int i = 0; i < g->ranks; ++i) {
        BcMoeRemoteGeneration* mine = g;
        g_slot_table[g->slot_base + i].compare_exchange_strong(mine, nullptr, std::memory_order_release,
                                                               std::memory_order_relaxed);
    }
    if (g->vt.close) g->vt.close(g->vt.self);
    g->vt = BcMoeRemotePlugin{};
}

extern "C" void bc_moe_remote_destroy(BcMoeRemoteGeneration* g) {
    if (!g) return;
    bc_moe_remote_close(g);
    delete g;
}

extern "C" int bc_moe_remote_wanted(const BcMoeRemoteGeneration* g, int64_t step_rows) {
    return g && g->open() && step_rows >= 0 && step_rows <= g->max_rows;
}

extern "C" int bc_moe_remote_wanted_upto(const BcMoeRemoteGeneration* g, int64_t max_step_rows) {
    return g && g->open() && max_step_rows >= 0;
}

extern "C" int bc_moe_remote_shadow(const BcMoeRemoteGeneration* g) {
    return kDiagBuild && g && g->open() && g->shadow;
}

extern "C" int bc_moe_remote_mask_only(const BcMoeRemoteGeneration* g) {
    return kDiagBuild && g && g->open() && g->mask_only;
}

extern "C" int bc_moe_remote_diag_build(void) {
    return kDiagBuild ? 1 : 0;
}

/* Per-step cost is one atomic exchange; the plugin hears only the transitions. */
extern "C" void bc_moe_remote_engine_step(BcMoeRemoteGeneration* g) {
    if (!g || !g->open()) return;
    if (!g->burst_active.exchange(true, std::memory_order_acq_rel) && g->vt.burst) g->vt.burst(g->vt.self, 1);
}

extern "C" void bc_moe_remote_engine_idle(BcMoeRemoteGeneration* g) {
    if (!g || !g->open()) return;
    if (g->burst_active.exchange(false, std::memory_order_acq_rel) && g->vt.burst) g->vt.burst(g->vt.self, 0);
}

/* Scheduler-thread and rank-thread calls into the generation's size-checked plugin table. The ticket originates only
 * from admission; b->step is debug identity and is never used to derive the actual epoch/slot/ticket id. A NULL
 * generation is "no remote" (zero ticket); a closed one is refused with BC_MOE_REMOTE_STALE before any dispatch. */
extern "C" int bc_moe_remote_admit(BcMoeRemoteGeneration* g, BcMoeRemoteTicket* out, int try_only, int world_size) {
    if (!out) return BC_MOE_REMOTE_REFUSED;
    *out = BcMoeRemoteTicket{};
    if (!g) return 0;
    if (!g->open()) return BC_MOE_REMOTE_STALE;
    return g->vt.admit ? g->vt.admit(g->vt.self, out, try_only, world_size) : BC_MOE_REMOTE_REFUSED;
}
extern "C" int bc_moe_remote_bind(BcMoeRemoteGeneration* g, BcMoeRemoteTicket ticket, int64_t step) {
    if (!ticket.ticket_id) return 0;
    if (!g) return BC_MOE_REMOTE_REFUSED;
    if (!g->open()) return BC_MOE_REMOTE_STALE;
    return g->vt.bind ? g->vt.bind(g->vt.self, ticket, step) : BC_MOE_REMOTE_REFUSED;
}
extern "C" int bc_moe_remote_cancel(BcMoeRemoteGeneration* g, BcMoeRemoteTicket ticket) {
    if (!ticket.ticket_id) return 0;
    if (!g) return BC_MOE_REMOTE_REFUSED;
    if (!g->open()) return BC_MOE_REMOTE_STALE;
    return g->vt.cancel ? g->vt.cancel(g->vt.self, ticket) : BC_MOE_REMOTE_REFUSED;
}
extern "C" int bc_moe_remote_host_done(BcMoeRemoteGeneration* g, BcMoeRemoteTicket ticket) {
    if (!ticket.ticket_id) return 0;
    if (!g) return BC_MOE_REMOTE_REFUSED;
    if (!g->open()) return BC_MOE_REMOTE_STALE;
    return g->vt.host_done ? g->vt.host_done(g->vt.self, ticket) : BC_MOE_REMOTE_REFUSED;
}
extern "C" void bc_moe_remote_fault(BcMoeRemoteGeneration* g) {
    if (g = live(g); g && g->vt.fault) g->vt.fault(g->vt.self);
}
extern "C" int bc_moe_remote_finish_admission(BcMoeRemoteGeneration* g) {
    g = live(g);
    if (!g || !g->vt.finish_admission) return 0;
    g->vt.finish_admission(g->vt.self);
    return 1;
}
extern "C" uint64_t bc_moe_remote_owner(const BcMoeRemoteGeneration* g) {
    return g && g->open() ? g->vt.owner : 0;
}
extern "C" int64_t bc_moe_remote_stop_budget_ms(const BcMoeRemoteGeneration* g) {
    return g && g->open() ? g->vt.stop_budget_ms : 0;
}
extern "C" int bc_moe_remote_wait_finished(BcMoeRemoteGeneration* g, int64_t timeout_ms) {
    if (!g) return BC_MOE_REMOTE_REFUSED;
    if (!g->open()) return BC_MOE_REMOTE_STALE;
    return g->vt.wait_finished ? g->vt.wait_finished(g->vt.self, timeout_ms) : BC_MOE_REMOTE_REFUSED;
}
extern "C" int bc_moe_remote_before_pass(BcMoeRemoteGeneration* g, int rank, BcMoeRemoteTicket ticket, void* stream) {
    if (!g) return 0;   /* no hand-off for this engine: nothing to arm */
    if (!g->open()) return BC_MOE_REMOTE_STALE;
    return g->vt.before_pass ? g->vt.before_pass(g->vt.self, rank, ticket, stream) : BC_MOE_REMOTE_REFUSED;
}
extern "C" int bc_moe_remote_rank_done(BcMoeRemoteGeneration* g, int rank, BcMoeRemoteTicket ticket, void* stream) {
    if (!ticket.ticket_id) return 0;
    if (!g) return BC_MOE_REMOTE_REFUSED;
    if (!g->open()) return BC_MOE_REMOTE_STALE;
    return g->vt.rank_done ? g->vt.rank_done(g->vt.self, rank, ticket, stream) : BC_MOE_REMOTE_REFUSED;
}

extern "C" void bc_moe_remote_layer_map(BcMoeRemoteGeneration* g, int layer, int n_expert, const int32_t* slot_of_expert) {
    g = live(g);
    if (!g || n_expert <= 0 || !slot_of_expert) return;
    {
        std::lock_guard<std::mutex> lock(g->maps_mu);
        g->layer_maps[layer].assign(slot_of_expert, slot_of_expert + n_expert);
    }
    /* Never under maps_mu: the plugin may wait on a helper, and the placement rule (skip_unit) reads these maps. */
    if (g->vt.layer_map) g->vt.layer_map(g->vt.self, layer, n_expert, slot_of_expert);
}

/* R3_MOE_WORKER_SKIP_OWNED=1: a helper-owned expert is not placed at all -- no VRAM, no host pool -- which is
 * what lets one engine rank (the R9700) hold a model whose experts it could not hold alone. Only when nothing
 * local will ever read the unit: every step is handed off (the hand-off cap covers the largest step), no
 * diagnostic recomputes owned experts locally, and the layer's runtime map gives the expert a regular slot with
 * no protected expert in the layer (those layers stay local; the helper drops ownership on exactly the same
 * test). */
extern "C" int bc_moe_remote_skip_unit(const BcMoeRemoteGeneration* g, int layer, int expert, int64_t max_step_rows) {
    if (!g || !g->open() || g->skip_owned == 0 || bc_moe_remote_shadow(g) || bc_moe_remote_mask_only(g)) return 0;
    if (g->max_rows < max_step_rows) return 0;
    {
        std::lock_guard<std::mutex> lock(g->maps_mu);
        const auto it = g->layer_maps.find(layer);
        if (it == g->layer_maps.end() || expert < 0 || expert >= (int)it->second.size()) return 0;
        for (int32_t slot : it->second)
            if (slot < 0) return 0;
    }
    return g->vt.owns_expert && g->vt.owns_expert(g->vt.self, layer, expert) ? 1 : 0;
}
"""

_BC_MOE_REMOTE_ENGINE_H = r"""/* Scheduler/rank admission bridge and the owner of the engine's remote-MoE generation (abi/bc_moe_remote.h). Scheduler
 * changes state while ranks are parked; the existing barrier publishes issued_. current_ survives issue-ahead until host
 * completion. The coordinator owns admission. The generation is opened once by the engine after it claimed its instance
 * slots, handed to every consumer the engine builds, and closed (then destroyed) before the plugin library unloads. */
#pragma once
#include "bc_moe_remote.h"
#include "device/device.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

class BcMoeRemoteAdmission {
    BcMoeRemoteGeneration* gen_ = nullptr;   /* owned: null when the manifest asks for no hand-off */
    bool clean_ = false;                     /* every bound ticket proved terminal: the stop was clean */
    BcMoeRemoteTicket current_{}, ahead_{}, issued_{};
    int ahead_status_ = 0;
    std::atomic<bool> finishing_{false}, stepper_finished_{false};
    std::mutex stepper_mu_;
    std::condition_variable stepper_cv_;
public:
    BcMoeRemoteAdmission() = default;
    BcMoeRemoteAdmission(const BcMoeRemoteAdmission&) = delete;
    BcMoeRemoteAdmission& operator=(const BcMoeRemoteAdmission&) = delete;
    ~BcMoeRemoteAdmission() { close(); }
    /* Open the generation on the loaded plugin's handle (null: none loaded) for the slots this engine claimed. */
    int open(void* plugin_handle, int slot_base, int ranks, uint32_t capacity_limit) {
        close();
        return bc_moe_remote_open(plugin_handle, slot_base, ranks, capacity_limit, &gen_) < 0 ? RAD_E_STATE : RAD_OK;
    }
    int open_with(BcMoeRemoteOpenFn factory, int slot_base, int ranks, uint32_t capacity_limit) {
        close();
        return bc_moe_remote_open_with(factory, slot_base, ranks, capacity_limit, &gen_) < 0 ? RAD_E_STATE : RAD_OK;
    }
    /* Idempotent. Every consumer has stopped by now; a holder of the pointer after this is refused, not served. */
    void close() { bc_moe_remote_destroy(gen_); gen_ = nullptr; clean_ = false; }
    BcMoeRemoteGeneration* generation() const { return gen_; }
    /* The fault stop (an error, or a drain that missed its deadline): interrupts every blocked wait. */
    void fault() { bc_moe_remote_fault(gen_); }
    /* True once every bound ticket proved terminal (wait_finished succeeded): a clean stop, which never interrupts. */
    bool clean() const { return clean_; }
    int wait_finished(int64_t timeout_ms) {
        const int r = bc_moe_remote_wait_finished(gen_, timeout_ms);
        if (r >= 0) clean_ = true;
        return r;
    }
    void engine_step() { bc_moe_remote_engine_step(gen_); }
    void engine_idle() { bc_moe_remote_engine_idle(gen_); }
    // Engine::step_loop holds one scalar ahead batch, not a queue.
    static constexpr uint32_t kIssueAheadDepth = 1;
    template<class Scheduler, class Batch>
    int step(bool issued, const Batch*& b, Scheduler& scheduler, int ranks) {
        if (issued) {
            scheduler.promote_next();
            current_ = ahead_;
            ahead_ = {};
        } else {
            const int admission = bc_moe_remote_admit(gen_, &current_, 0, ranks);
            if (admission != 0) return admission < 0 ? RAD_E_STATE : admission;
            b = scheduler.step();
            if (!b) {
                if (bc_moe_remote_cancel(gen_, current_) < 0) return RAD_E_STATE;
                current_ = {};
            } else if (bc_moe_remote_bind(gen_, current_, b->step) < 0) return RAD_E_STATE;
        }
        issued_ = current_;
        return RAD_OK;
    }
    bool try_ahead(int ranks) {
        const int rc = bc_moe_remote_admit(gen_, &ahead_, 1, ranks);
        ahead_status_ = rc < 0 ? RAD_E_STATE : RAD_OK;
        return rc == 0;
    }
    int ahead_status() const { return ahead_status_; }
    template<class Batch>
    int bind_ahead(const Batch* b) {
        if (!b) {
            if (bc_moe_remote_cancel(gen_, ahead_) < 0) return RAD_E_STATE;
            ahead_ = {};
            return RAD_OK;
        }
        if (bc_moe_remote_bind(gen_, ahead_, b->step) < 0) return RAD_E_STATE;
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
            if (bc_moe_remote_rank_done(gen_, r->index, current_, r->ctx.stream()) < 0) return RAD_E_DEVICE;
        return RAD_OK;
    }
    int host_done() { return bc_moe_remote_host_done(gen_, current_); }
    bool finish() {
        if (!bc_moe_remote_finish_admission(gen_)) return false;
        finishing_.store(true, std::memory_order_release);
        return true;
    }
    bool finishing() const { return finishing_.load(std::memory_order_acquire); }
    bool stepper_finished() const { return stepper_finished_.load(std::memory_order_acquire); }
    void note_stepper_finished() {
        { std::lock_guard<std::mutex> lock(stepper_mu_); stepper_finished_.store(true, std::memory_order_release); }
        stepper_cv_.notify_all();
    }
    /* The plugin's stop bound (0 with no generation). */
    int64_t stop_budget_ms() const { return bc_moe_remote_stop_budget_ms(gen_); }
    /* Wait for the stepper to finish, until `deadline`: a condition wait, not a poll. */
    bool wait_stepper_finished(std::chrono::steady_clock::time_point deadline) {
        std::unique_lock<std::mutex> lock(stepper_mu_);
        return stepper_cv_.wait_until(lock, deadline, [&] { return stepper_finished_.load(std::memory_order_acquire); });
    }
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

/* A routed expert the remote helper computes, when the engine never reads it (bc_moe_remote_skip_unit of the engine's
 * generation, found by the callers from their program's build slot): no
 * slot, no pool and no copy. It stays on the mapped tier as a placeholder the loader publishes without
 * reading; the zero-copy site keeps it out of the host-site and mixed-operand logic. Returns whether the unit
 * was taken (the planner then leaves it alone). */
template <class MU>
inline bool bc_moe_remote_pin_unit(const BcMoeRemoteGeneration* gen, MU& mu, int64_t max_tok) {
    if (mu.layer < 0 || mu.expert < 0 || !bc_moe_remote_skip_unit(gen, mu.layer, mu.expert, max_tok)) return false;
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
inline void bc_moe_remote_mark_footprint(const BcMoeRemoteGeneration* gen, std::vector<U>& units) {
    for (U& mu : units)
        if (mu.layer >= 0 && mu.expert >= 0 && bc_moe_remote_skip_unit(gen, mu.layer, mu.expert, 0)) mu.mapped = true;
}

/* Bring-up: publish a unit the planner left on the mapped tier as a never-read placeholder (the container
 * address, never dereferenced) instead of loading it. Its stored form never matters because this rank never
 * reads it, so the byte-count and direct-read checks do not apply. Returns whether it was published. */
template <class W, class E, class F, class M, class I>
inline bool bc_moe_remote_publish_placeholder(const BcMoeRemoteGeneration* gen, int64_t max_tok, W& w, const E* e,
                                              const F& file, M& mover, I index) {
    if (!e || w.tier != Tier::Mapped || w.decl.group.layer < 0 || w.decl.group.expert < 0 ||
        !bc_moe_remote_skip_unit(gen, w.decl.group.layer, w.decl.group.expert, max_tok))
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
    return !bc_moe_remote_mask_only(g.bc_remote) && T <= bc_moe_remote_fuse_rows(g) && gather_taken(g, T, gather_rows, gather_rows6);
}
/* HyperConn::takes_ar: the connection fed by a MoE gather source (top_k > 0) leaves the reduction to the block on a
 * handed-off step that does not fuse; attention and delta-net connections keep their fused writes. */
inline bool bc_moe_remote_hc_open(int64_t moe_top_k, const Geom& g, int64_t T, int64_t gather_rows, int64_t gather_rows6) {
    return moe_top_k > 0 && bc_moe_remote_wanted(g.bc_remote, T) && !bc_moe_remote_shadow(g.bc_remote) &&
           !bc_moe_remote_fuses(g, T, gather_rows, gather_rows6);
}

struct BcMoeRemote {
    /* The engine's generation (Geom::bc_remote, found by the build slot when geom_from ran); null: no hand-off. Non-owning:
     * the engine closes the generation only after its ranks (and so this block) are gone. */
    BcMoeRemoteGeneration* gen = nullptr;
    /* publish_fused: the same op for a step whose helper rows join the fused gather (param fused = 1: no combine) */
    rad_op publish = 0, publish_fused = 0, add = 0, inject = 0;
    rad_op quantize = 0;
    rad_buf aq = 0, as = 0;

    /* The step `T` is handed off (shadow keeps the local path and only publishes). */
    bool policy(int64_t T) const { return publish && add && bc_moe_remote_wanted(gen, T); }
    bool active(int64_t T) const { return policy(T) && !bc_moe_remote_shadow(gen); }
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

    /* Keep mailbox-only buffers beside their first use in the declared graph. The helpers
     * consume group-128 E4M3, even when the local grouped GEMM uses rotated int8. */
    template <class Moe>
    int declare_publish(RadBuilder* b, const Moe& m, int layer, Names* materialize = nullptr) {
        const auto& c = m.c;
        const auto& g = m.g;
        const auto& w = m.w;
        gen = g.bc_remote;
        if (!bc_moe_remote_wanted_upto(gen, g.max_tok)) return RAD_OK;
        aq = c.expert_rot ? w.hr.q : w.h.q;
        as = c.expert_rot ? w.hr.s : w.h.s;
        if (materialize) {
            aq = decl_b(b, materialize->f("blk.%d.moe_mailbox.q", layer), RAD_F8E4M3, {g.max_tok, g.n_embd});
            as = decl_b(b, materialize->f("blk.%d.moe_mailbox.s", layer), RAD_F32,
                        {g.max_tok, g.n_embd / RAD_FP8_BLOCK});
            if (aq && as && g.n_embd % RAD_FP8_BLOCK == 0)
                quantize = rw(b, RAD_OP(b, c.expert_rot ? "had_quant_act_fp8" : "quant_act_fp8",
                               RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd),
                                          RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                               RAD_NOWEIGHTS),
                              {w.h.x}, {aq, as});
        }
        if (!aq || !as || (materialize && !quantize)) {
            std::fprintf(stderr, "radiance: MoE remote hand-off is missing the required E4M3 activation "
                         "planes or quantizer at layer %d\n", layer);
            return RAD_E_INVAL;
        }
        for (int f = 0; f < 2; ++f)
            (f ? publish_fused : publish) =
                rw(b, RAD_OP(b, "moe_mailbox_publish",
                             RAD_PARAMS(RAD_INT("owner", bc_moe_remote_owner(gen)), RAD_INT("layer", layer), RAD_RANGE("M", 1, c.rows),
                                        RAD_INT("n", g.n_embd), RAD_INT("top_k", c.top_k),
                                        RAD_INT("expert_base", c.expert_base),
                                        RAD_INT("n_expert", c.n_expert), RAD_INT("fused", f)),
                             RAD_NOWEIGHTS),
                   {w.ids, w.ew, w.h.x, aq, as}, {w.ids});
        return publish && publish_fused ? RAD_OK : RAD_E_INVAL;
    }

    /* Publish the final runtime slot map and declare merge operators. Legacy graphs without
     * independent int8 input planes declare their publish operators here as well. */
    template <class Moe>
    int declare(RadBuilder* b, const Moe& m, const std::vector<int64_t>& reg, int layer) {
        const auto& c = m.c;
        const auto& g = m.g;
        const auto& w = m.w;
        const int64_t ne = c.n_expert;
        std::vector<int32_t> slot((size_t)ne, -1);
        for (size_t s = 0; s < reg.size(); ++s)
            if (reg[s] >= 0 && reg[s] < ne) slot[(size_t)reg[s]] = (int32_t)s;
        gen = g.bc_remote;
        bc_moe_remote_layer_map(gen, layer, (int)ne, slot.data());

        if (!bc_moe_remote_wanted_upto(gen, g.max_tok)) return RAD_OK;
        if (!publish && declare_publish(b, m, layer) < 0) return RAD_E_INVAL;
        add = RAD_OP(b, "moe_mailbox_add",
                     RAD_PARAMS(RAD_INT("owner", bc_moe_remote_owner(gen)), RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd), RAD_INT("top_k", c.top_k)),
                     RAD_NOWEIGHTS);
        if (add) {
            const rad_buf rd[5] = { w.sorted, w.edn, w.ew, w.sout, w.sgate };
            const rad_buf wr[1] = { w.h.x };
            rad_op_reads(b, add, rd, m.gfold ? 5 : 3);
            rad_op_writes(b, add, wr, 1);
        }
        inject = RAD_OP(b, "moe_mailbox_inject",
                        RAD_PARAMS(RAD_INT("owner", bc_moe_remote_owner(gen)), RAD_INT("layer", layer), RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd),
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
                         publish != 0, publish_fused != 0, add != 0, inject != 0, aq != 0, as != 0, (int)c.expert_rot);
            return RAD_E_INVAL;
        }
        return RAD_OK;
    }

    /* After the router and scatter of a pass, before the grouped GEMMs: publish the routes and re-scatter, so
     * the helper-owned routes are masked out of the local sort. In mask-only (diagnostic) the publish comes
     * later, from the diagnostic hook. */
    template <class Moe>
    void after_route(RadCtx* c_, const Moe& m, int64_t T, int64_t r0, int64_t rows, const RadOperand& ecnt,
                     const RadOperand& inv = RAD_NONE) const {
        if (!policy(T) || bc_moe_remote_mask_only(gen)) return;
        const auto& c = m.c;
        const auto& g = m.g;
        const auto& w = m.w;
        const int64_t k = c.top_k;
        if (quantize)
            RAD_ISSUE_N(c_, quantize, rows, brow_slice(w.h.x, r0, rows, g.n_embd),
                        brow_slice(aq, r0, rows, g.n_embd),
                        brow_slice(as, r0, rows, g.n_embd / RAD_FP8_BLOCK));
        RAD_ISSUE_N(c_, fused(m, T) ? publish_fused : publish, rows,
                    brow_slice(w.ids, r0, rows, k),
                    brow_slice(w.ew, r0, rows, k),
                    brow_slice(w.h.x, r0, rows, g.n_embd),
                    brow_slice(aq, r0, rows, g.n_embd),
                    brow_slice(as, r0, rows, g.n_embd / RAD_FP8_BLOCK),
                    brows(aq, T));
        if (!active(T)) return;
        if (inv.kind != RAD_OPK_NONE)
            RAD_ISSUE_N(c_, m.op_scatter, rows, brow_slice(w.ids, r0, rows, k),
                        brows(w.sorted, rows * k), brows(w.eoff, c.n_expert + 1), ecnt, inv);
        else
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
            guard=r"^int bc_moe_remote_wanted\(const BcMoeRemoteGeneration\* g, int64_t step_rows\);$",
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
                id="bc-remote-declare-publish",
                anchor=r"^    if \(!gin_q \|\| !gin_s \|\| g\.n_embd % gin_g\) return RAD_E_INVAL;$",
                text=(
                    "\n    RAD_ARCH_TRY(bc_remote.declare_publish(b, *this, l,"
                    "\n        (c.expert_rot ? gin_q != w.hr.q || gin_s != w.hr.s"
                    "\n                      : gin_q != w.h.q || gin_s != w.h.s) ? &nm : nullptr));"
                ),
                guard=r"RAD_ARCH_TRY\(bc_remote\.declare_publish\(b, \*this, l,",
                applies_if=r"rad_buf gin_q = 0, gin_s = 0;",
                expect_matches=1,
                rationale="Declare scratch and publish at their issue position, with a separate E4M3 pair when the local pair differs.",
            ),
            Edit(
                id="bc-remote-after-route-inverse",
                anchor=r"^[ \t]*const int64_t n4\[4\] = \{",
                mode="insert_before",
                text=(
                    "    bc_remote.after_route(c_, *this, T, r0, rows, ecnt,\n"
                    "                          scatter_inv ? brows(w.inv, rows * k) : RAD_NONE);\n"
                ),
                guard=r"bc_remote\.after_route\(c_, \*this, T, r0, rows, ecnt,",
                applies_if=r"bool[ \t]+scatter_inv = false",
                expect_matches=1,
                rationale="The remasked scatter must write the same inverse slot table as the original scatter.",
            ),
            Edit(
                id="bc-remote-after-route",
                applies_if=r"\A(?![\s\S]*bool[ \t]+scatter_inv = false)",
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
                text="        if (bc_moe_remote_pin_unit(bc_moe_remote_lookup(rad_build_slot(&prog.ctx)), mu, in_.config->max_tok)) continue;\n",
                guard=r"if \(bc_moe_remote_pin_unit\(bc_moe_remote_lookup\(rad_build_slot\(&prog\.ctx\)\), mu, in_\.config->max_tok\)\) continue;",
                expect_matches=1,
                rationale="This declaration opens pin_mapped's per-unit loop body; a helper-owned unit is decided before any host or n-gram logic and leaves the loop.",
            ),
            Edit(
                id="bc-remote-mark-footprint",
                anchor=r"^    for \(U& mu : units\) mu\.mapped = mapped_unit\(mu\.mapped, container && mu\.direct, host_backend\);$",
                text="\n    bc_moe_remote_mark_footprint(bc_moe_remote_lookup(rad_build_slot(&prog.ctx)), units);",
                guard=r"bc_moe_remote_mark_footprint\(bc_moe_remote_lookup\(rad_build_slot\(&prog\.ctx\)\), units\);",
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
                text="            if (bc_moe_remote_publish_placeholder(bc_moe_remote_lookup(rad_build_slot(&r->program.ctx)), cfg_.max_tok, w, e, *file_, r->mover, (rad_weight)i)) continue;\n",
                guard=r"if \(bc_moe_remote_publish_placeholder\(bc_moe_remote_lookup\(rad_build_slot\(&r->program\.ctx\)\), cfg_\.max_tok, w, e, \*file_, r->mover, \(rad_weight\)i\)\) continue;",
                expect_matches=1,
                rationale="The single tier / stored-form check in the weight publish loop; a helper-owned placeholder is published before it, since nothing reads its bytes and its relayouted form would fail the check.",
            ),
            Edit(
                id="bc-remote-open-generation",
                anchor=r"^        reg_->set_tp_group\(base\);\n    \}$",
                text=(
                    "\n    /* The remote-MoE generation of this engine: opened through the factory of the already-loaded libr3moe plugin for the"
                    "\n     * slots just claimed, before any declare, planner or bring-up step looks it up. Closed in the destructor. */"
                    "\n    static_assert(RAD_MAX_SLOTS == BC_MOE_REMOTE_MAX_SLOTS, \"the remote-MoE slot table must cover every engine slot\");"
                    "\n    {"
                    "\n        const Plugin* bc_moe_plugin = reg_->plugin(\"libr3moe\");"
                    "\n        RAD_TRY(bc_remote_admission_.open(bc_moe_plugin ? bc_moe_plugin->handle : nullptr, slot_base_, cfg_.tp,"
                    "\n            (uint32_t)std::extent_v<decltype(rad::RouteSlot::staging)> + BcMoeRemoteAdmission::kIssueAheadDepth));"
                    "\n    }"),
                guard=r"bc_remote_admission_\.open\(bc_moe_plugin \? bc_moe_plugin->handle : nullptr, slot_base_, cfg_\.tp,",
                expect_matches=1,
                rationale="The engine's slots are claimed here; the generation must exist, bound to them, before the first consumer looks it up.",
            ),
        ),
    ),
    FilePatch(
        path="core/runtime/ctx.h",
        description="Carry the engine's remote-MoE generation and the parent's ticket in the rank's Ctx, not in a thread-local.",
        edits=(
            Edit(
                id="include-bc-moe-remote-ctx-h",
                anchor=r'^#include "wtab\.h"$',
                text='\n#include "bc_moe_remote.h"',
                guard=r'^#include "bc_moe_remote\.h"$',
                expect_matches=1,
                rationale="wtab.h is the last of the context's unique local includes; the ticket value type lives in the extension ABI.",
            ),
            Edit(
                id="bc-remote-ctx-issue",
                anchor=r"^    void set_gen_lora_slot\(int32_t s\) \{ gen_lora_slot_ = s; \}$",
                text=(
                    "\n    /* The engine's remote-MoE generation (null: no hand-off) and this parent's immutable admission ticket; the rank thread"
                    "\n     * sets both before it releases the rank for a pass, and Ctx::run_pass reads them. */"
                    "\n    void bc_moe_remote_issue(BcMoeRemoteGeneration* g, BcMoeRemoteTicket t) { bc_remote_ = g; bc_remote_ticket_ = t; }"),
                guard=r"void bc_moe_remote_issue\(BcMoeRemoteGeneration\* g, BcMoeRemoteTicket t\)",
                expect_matches=1,
                rationale="The other per-release setter the pass loop calls on the rank thread; unique line.",
            ),
            Edit(
                id="bc-remote-ctx-state",
                anchor=r"^    int32_t gen_lora_slot_ = -1;$",
                text="\n    BcMoeRemoteGeneration* bc_remote_ = nullptr;\n    BcMoeRemoteTicket bc_remote_ticket_{};",
                guard=r"^    BcMoeRemoteGeneration\* bc_remote_ = nullptr;$",
                expect_matches=1,
                rationale="Beside the other per-release pass state the rank thread sets; unique line.",
            ),
        ),
    ),
    FilePatch(
        path="arch/common/rad_arch.h",
        description="Hand each declaring instance the engine's remote-MoE generation, found once by its build slot.",
        edits=(
            Edit(
                id="include-bc-moe-remote-arch",
                anchor=r'^#include "rad_abi_layout\.h"$',
                text='\n#include "bc_moe_remote.h"',
                guard=r'^#include "bc_moe_remote\.h"$',
                expect_matches=1,
                rationale="rad_abi_layout.h is the last of the architecture header's unique local includes.",
            ),
            Edit(
                id="bc-remote-geom-member",
                anchor=r"^    int64_t n_layer    = 0;$",
                mode="insert_before",
                text="    BcMoeRemoteGeneration* bc_remote = nullptr;   /* this engine's remote-MoE generation (non-owning), null: no hand-off */\n",
                guard=r"^    BcMoeRemoteGeneration\* bc_remote = nullptr;",
                expect_matches=1,
                rationale="Geom's first member line is unique; blocks reach the generation through the Geom they already hold.",
            ),
            Edit(
                id="bc-remote-geom-lookup",
                anchor=r"^    g\.n_layer   = m->n_layers;$",
                mode="insert_before",
                text="    g.bc_remote = bc_moe_remote_lookup(rad_build_slot(c));\n",
                guard=r"g\.bc_remote = bc_moe_remote_lookup\(rad_build_slot\(c\)\);",
                expect_matches=1,
                rationale="The one init-time lookup of an architecture's declare: c is checked and its rank range-checked just above.",
            ),
        ),
    ),
    FilePatch(
        path="core/runtime/ctx.cpp",
        description="Bind every trunk/draft/issued-ahead pass to its parent ticket before tape capture or replay.",
        edits=(
            Edit(
                id="bc-remote-ticket-before-pass",
                anchor=r"^        const int rc = run_pass\(b\);\n        if \(rc < 0 && step_status_ >= 0\) step_status_ = rc;$",
                mode="replace",
                text="""        /* The rank thread installed this parent's immutable admission
         * ticket before releasing this pass. Never derive it from b->step. */
        const int rc = bc_moe_remote_before_pass(bc_remote_, rank_, bc_remote_ticket_, stream_) < 0 ? RAD_E_STATE : run_pass(b);
        if (rc < 0 && step_status_ >= 0) step_status_ = rc;""",
                guard=r"const int rc = bc_moe_remote_before_pass\(bc_remote_, rank_, bc_remote_ticket_, stream_\) < 0 \? RAD_E_STATE : run_pass\(b\);",
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
                text="            r.ctx.bc_moe_remote_issue(bc_remote_admission_.generation(), bc_remote_admission_.issued_ticket());\n",
                guard=r"r\.ctx\.bc_moe_remote_issue\(bc_remote_admission_\.generation\(\), bc_remote_admission_\.issued_ticket\(\)\);",
                expect_matches=1,
                rationale="Explicit immutable transfer of the engine's generation and the parent's ticket into the rank's Ctx before every rank model/draft release; no thread-local owner.",
            ),
            Edit(
                id="bc-remote-ahead-try-admit",
                anchor=r"^    for \(int i = 0; i < serial_rows_ && serial_done_ > 0; \+\+i\)$",
                mode="insert_before",
                text=("    if (bc_remote_admission_.finishing()) return RAD_OK;\n"
                      "    if (!bc_remote_admission_.try_ahead((int)ranks_.size())) return bc_remote_admission_.ahead_status();\n"),
                guard=r"bc_remote_admission_\.try_ahead\(",
                expect_matches=1,
                rationale="Reserve only after upstream's flex/quiet early exits and before expect_draft or step_next mutates scheduler state.",
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
                text=("        const int bc_remote_status = bc_remote_admission_.step(issued, b, sched_, (int)ranks_.size());\n"
                      "        if (bc_remote_status < 0) return fail_loop(RAD_E_STATE);\n"
                      "        if (bc_remote_status > 0) break;"),
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
                id="bc-remote-fault-interrupt",
                anchor=r"^int Engine::fail_loop\(int st\) \{$",
                text="\n    bc_remote_admission_.fault();",
                guard=r"^int Engine::fail_loop\(int st\) \{\n    bc_remote_admission_\.fault\(\);$",
                expect_matches=1,
                rationale="The fault stop interrupts the admission and completion waiters immediately.",
            ),
            Edit(
                id="bc-remote-finish-issued-ahead",
                anchor=r"^    while \(!stopping_\.load\(std::memory_order_acquire\)\) \{\n        step_watch\.progress\(\);$",
                mode="replace",
                text=("    while (!stopping_.load(std::memory_order_acquire) && (ahead || !bc_remote_admission_.finishing())) {\n"
                      "        step_watch.progress();"),
                guard=r"ahead \|\| !bc_remote_admission_\.finishing\(\)",
                expect_matches=1,
                rationale="A clean stop finishes the already-issued batch without starting a new parent.",
            ),
            Edit(
                id="bc-remote-note-stepper-finished",
                anchor=r"^        if \(gen_\) gen_loop\(\); else step_loop\(\);$",
                text="\n        bc_remote_admission_.note_stepper_finished();",
                guard=r"bc_remote_admission_\.note_stepper_finished\(\);",
                applies_if=r"front_->detach\(cfg_\.served_model_name",
                expect_matches=1,
                rationale="Let the live Server frame wait for the stepper before rank teardown.",
            ),
            Edit(
                id="bc-remote-finish-before-server-destruction",
                anchor=r"^    srv\.retire\(\);\n    if \(live\) live->stop\(\);\n    const int st = loop_status_\.load\(std::memory_order_acquire\);$",
                mode="insert_before",
                text="""    if (!gen_ && loop_status_.load(std::memory_order_acquire) >= 0 &&
        bc_remote_admission_.finish()) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(bc_remote_admission_.stop_budget_ms());
        if (!bc_remote_admission_.wait_stepper_finished(deadline)) {
            bc_remote_admission_.fault();
            fatal("model %s: remote MoE stepper did not finish within its shutdown deadline",
                  cfg_.served_model_name.c_str());
        }
        const int stepper_status = loop_status_.load(std::memory_order_acquire);
        if (stepper_status < 0) return stepper_status;
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (bc_remote_admission_.wait_finished(std::max<int64_t>(0, remaining)) < 0)
            return fail_loop(RAD_E_STATE);
        bc_remote_admission_.engine_idle();
    }
""",
                guard=r"bc_remote_admission_\.wait_finished\(std::max<int64_t>",
                applies_if=r"front_->detach\(cfg_\.served_model_name",
                expect_matches=1,
                rationale="Use the bound parents' existing terminal proof before server and rank teardown.",
            ),
            Edit(
                id="bc-remote-after-server-interrupt",
                anchor=r"^    stopping_\.store\(true, std::memory_order_release\);\n(?:    wake_hold\(\);\n)?    barrier_->abort\(\);$",
                mode="insert_before",
                text="    if (!bc_remote_admission_.clean()) bc_remote_admission_.fault();\n",
                guard=r"if \(!bc_remote_admission_\.clean\(\)\) bc_remote_admission_\.fault\(\);\n    stopping_\.store\(true, std::memory_order_release\);\n(?:    wake_hold\(\);\n)?    barrier_->abort\(\);",
                expect_matches=1,
                rationale="A clean stop (every bound ticket proven by wait_finished) never interrupts the ledger; anything else is a fault stop and wakes every remaining wait.",
            ),
            Edit(
                id="bc-remote-close-generation",
                anchor=r"^    ranks_\.clear\(\);$",
                mode="insert_after",
                text="\n    bc_remote_admission_.close();",
                guard=r"bc_remote_admission_\.close\(\);",
                expect_matches=1,
                rationale="The ranks (and every consumer of the generation) are gone here and the plugin library is still loaded: the one place the generation can close.",
            ),
            Edit(
                id="bc-remote-engine-idle",
                anchor=r"^            step_batch_\.store\(nullptr, std::memory_order_release\);$",
                text="\n            bc_remote_admission_.engine_idle();",
                guard=r"^            bc_remote_admission_\.engine_idle\(\);$",
                expect_matches=1,
                rationale="The engine loop's idle branch clears the step batch here when the scheduler has no step: every earlier step has joined, so a burst ends.",
            ),
            Edit(
                id="bc-remote-engine-step",
                anchor=r"^        idle = 0;$",
                mode="insert_before",
                text="        bc_remote_admission_.engine_step();\n",
                guard=r"bc_remote_admission_\.engine_step\(\);",
                expect_matches=1,
                rationale="The loop resets its idle backoff exactly when the scheduler returned a step, before the ranks are released or any recorded pass is replayed.",
            ),
        ),
    ),
    FilePatch(
        path="tests/arch_fakes.h",
        edits=(
            Edit(
                id="bc-remote-arch-fakes",
                anchor=r'^#include "rad_runtime\.h"$',
                mode="insert_after",
                text='''

#include "bc_moe_remote.h"

/* Architecture fixtures run without a remote helper or the engine core. */
extern "C" BcMoeRemoteGeneration* bc_moe_remote_lookup(int) { return nullptr; }
extern "C" int bc_moe_remote_wanted(const BcMoeRemoteGeneration*, int64_t) { return 0; }
extern "C" int bc_moe_remote_wanted_upto(const BcMoeRemoteGeneration*, int64_t) { return 0; }
extern "C" uint64_t bc_moe_remote_owner(const BcMoeRemoteGeneration*) { return 0; }
extern "C" int bc_moe_remote_shadow(const BcMoeRemoteGeneration*) { return 0; }
extern "C" int bc_moe_remote_mask_only(const BcMoeRemoteGeneration*) { return 0; }
extern "C" void bc_moe_remote_layer_map(BcMoeRemoteGeneration*, int, int, const int32_t*) {}
''',
                guard=r'extern "C" BcMoeRemoteGeneration\* bc_moe_remote_lookup\(int\)',
                expect_matches=1,
                rationale="Standalone architecture tests fake core entry points and declare no remote handoff.",
            ),
        ),
    ),
)
