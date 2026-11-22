#include "moe_remote_hook.h"
#include "hook_identity.h"
#include "layer_map_wait.h"
#include "../serving/bc_engine_settings.h"
#include "admission_completion.h"
#include "admission_ledger.h"
#include "engine_admission_gate.h"
#include "engine_rank_coordinator.h"
#include "heat_publication.h"
#include "moe_helper_watch.h"
#include "moe_remote_shm.h"
#include "moe_v26_gpu_contract.h"
#include "owner_generation.h"
#include "rad_abi.h"
#include "rad_device.h" /* engine streams take only radiance events (rad_event_*), never raw HIP events */
#include "worker_epoch.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <utility>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <thread>
#include <string>
#include <vector>
#include <deque>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using namespace r3::moe_shm;

struct RankState {
    R3MoeMailboxDevice dev{};
    int device = -1;
    uint8_t* owner[2]{};
    R3MoeOwnerBinding* owner_binding = nullptr;
    hipStream_t owner_stream = nullptr; /* transition H2D, never the serving stream */
    hipEvent_t owner_event[2]{};
    uint8_t* owner_stage[2]{}; /* pinned until the corresponding event completes */
    size_t owner_bytes = 0;
    R3MoeOwnerBinding owner_expected[2]{};
    bool owner_queued[2]{};
    StepTicket* ticket = nullptr;
    ConsumptionStamp* consumed_host = nullptr;
    ConsumptionStamp* stamps = nullptr;          /* device-memory counting ring */
    ConsumedRecord* consumed_rec = nullptr;     /* device-memory consumed lane/combine records */
    uint32_t* sequence = nullptr;
    uint32_t* published = nullptr;
    uint16_t* mask_ref = nullptr;
    void* maps = nullptr;   /* inv + rmask, one allocation */
    uint32_t* heat_host = nullptr;   /* R3_MOE_WORKER_HISTOGRAM: the counts in pinned host memory (dev.heat maps it) */
    bool ready = false;
};

/* The engine's remote-MoE extension point (patch rad0002, abi/bc_moe_remote.h), mirrored here so libr3moe builds
 * against an unpatched radiance tree. The engine accepts the table only if struct_size matches its own; the tests
 * compare both field lists. The engine finds bc_moe_remote_plugin_open on this plugin's own handle and calls it once per
 * engine generation. */
constexpr int kOpenRefused = -1;   /* BC_MOE_REMOTE_REFUSED */
constexpr int kOpenBusy = -3;      /* BC_MOE_REMOTE_BUSY */
struct EngineOpenArgs {
    uint32_t struct_size;
    uint32_t capacity_limit;
    int32_t slot_base;
    int32_t ranks;
};
struct EnginePlugin {
    uint32_t struct_size;   /* the head (struct_size, self, close) leads every table; the engine presets struct_size */
    void* self;
    void (*close)(void* self);
    void (*layer_map)(void* self, int layer, int n_expert, const int32_t* slot_of_expert);
    int (*owns_expert)(void* self, int layer, int expert);
    void (*burst)(void* self, int active);
    int (*before_pass)(void* self, int rank, StepTicket ticket, void* stream);
    int (*admit)(void* self, StepTicket* out, int try_only, int world_size);
    int (*bind)(void* self, StepTicket ticket, int64_t scheduler_step);
    int (*cancel)(void* self, StepTicket ticket);
    int (*host_done)(void* self, StepTicket ticket);
    int (*rank_done)(void* self, int rank, StepTicket ticket, void* stream);
    void (*fault)(void* self);   /* a fault stop: wake every waiter; never called on a clean stop */
    void (*finish_admission)(void* self);
    int (*wait_finished)(void* self, int64_t timeout_ms);
};

/* Each admission owns its events; never reuse one global step-end event
 * as evidence of several queued parents. An event is recorded after every
 * parent GPU effect, before issue-ahead, and queried only after host_done.
 * Events come from a per-rank pool sized to the ledger ring (one per slot,
 * admission_completion.h); queries and records need no current device. */
struct RadEvents {
    using Event = RadEvent;
    static int create(RadEvent* e) { return rad_event_create(e); }
    static void destroy(RadEvent e) { rad_event_destroy(e); }
    static int record(RadEvent e, void* stream) { return rad_event_record(e, (RadStream)stream); }
    static int query(RadEvent e) { return rad_event_query(e); }
};
using RankCompletion = r3::admission::RankCompletion<RadEvents>;
}  // namespace
struct R3MoeHook;
namespace {
using HookState = R3MoeHook;
}  // namespace
struct R3MoeHook {
    int fd = -1;
    void* map = MAP_FAILED;
    size_t bytes = 0;
    Header* hdr = nullptr;
    bool pinned = false;
    uint64_t serial = 0;   /* this generation's identity; never reused, so a stale handle cannot alias a new hook */
    int slot_base = -1;    /* the engine's claimed slot span (its tp_group); the kernels' init finds this hook by it */
    bool opened = false;   /* a generation holds this hook: set by the factory, cleared by the generation's close */
    r3::admission::Ledger ledger{0}; /* configured by the engine before map replay */
    std::deque<RankCompletion> completion;
    R3MoeDrainSnapshot closed_snapshot{};
    uint64_t closure_id = 0;
    std::atomic<uint32_t> engine_ranks{0}; /* scheduler-owned; published once before controller starts */
    /* Rank setup proves bootstrap seal/upload once. Admission tests only this
     * cached mask, never rehashes the canonical map on the hot path. */
    std::atomic<uint32_t> bootstrap_ready_ranks{0};
    // Cached result of the engine's EXACT bc_moe_remote_wanted_upto policy.
    // No independent environment parser or second source of declaration policy.
    bool remote_possible = true;
    bool drain_verified = false; /* reopen requires a successful full drain */
    std::thread placement_thread;
    /* Header::burst ownership: burst_transition (scheduler) and the helper drain (placement thread). While a drain
     * is requested the word is held even, so the helpers' burst spins end without reading drain words; radiance's
     * wanted state is applied at reopen. */
    std::mutex burst_mu;
    bool burst_want = false;
    bool burst_draining = false;
    std::atomic<bool> placement_stop{false};
    std::mutex mu;
    std::thread snapshotter;          /* R3_MOE_WORKER_HISTOGRAM_PERIOD_S: writes the live route histogram */
    std::atomic<bool> snapshot_stop{false};
    std::deque<RankState> rank;
    HelperWatch watch;   /* RAD20: fails the engine's helper waits when a helper process dies */
};
namespace {

bool enabled_env() { return bc_engine_settings_snapshot().handoff != 0; }
bool poison_env() { return false; } // RSC12: no production poison diagnostic
bool shadow_env() { return bc_engine_settings_snapshot().shadow != 0; }
bool mask_only_env() { return bc_engine_settings_snapshot().mask_only != 0; }
bool cpu_ref_env() { return false; } // RSC12: helper-only CPU reference
const char* shm_name() { return bc_engine_settings_snapshot().shm; }
int64_t max_rows() {
    const int64_t n = bc_engine_settings_snapshot().max_rows;
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
bool layer_enabled(int) {
    return true;   // R3_MOE_WORKER_LAYERS (a MIN-MAX bisection aid, RAD13) is no longer read from the environment (RSC12): every layer
}

/* All parent tickets originate in this coordinator. Neither a scheduler
 * number nor an in-flight tape's arguments are admission authority. */
bool committed(HookState& g, uint32_t* epoch, uint8_t* slot) {
    if (!g.hdr || !g.pinned || load(&g.hdr->helper_fault)) return false;
    const PlacementControl* c = placement_control(g.map);
    const uint32_t a = load(&c->seq);
    const uint64_t e = __atomic_load_n(&c->active_epoch, __ATOMIC_ACQUIRE);
    const uint32_t sl = load(&c->active_slot);
    if ((a & 1u) || load(&c->seq) != a ||
        load(&c->state) != (uint32_t)PlacementState::ACTIVE ||
        !e || e > UINT32_MAX || sl > 1) return false;
    *epoch = (uint32_t)e;
    *slot = (uint8_t)sl;
    return true;
}

bool helper_faulted(HookState& g) { return g.hdr && load(&g.hdr->helper_fault); }

void poll_tickets(HookState& g);

int admit_ticket(HookState& g, StepTicket* out, int try_only, int world_size) {
    if (!out) return RAD_E_INVAL;
    *out = StepTicket{};
    if (world_size < 1 || !g.hdr || world_size != (int)g.hdr->rank_count) return RAD_E_INVAL;
    const uint32_t ranks = (uint32_t)world_size;
    uint32_t mask = 0, helper_mask = 0;
    if (!v26::role_masks(g.hdr->topology, &helper_mask, &mask)) return RAD_E_INVAL;
    // When engine policy declares no mailbox ops, admission is a no-op.
    // No ticket/seal is possible (and none is needed for local-only steps).
    const auto bootstrap = r3::epoch::admission_bootstrap(
        g.remote_possible,g.bootstrap_ready_ranks,mask);
    if (bootstrap == r3::epoch::AdmissionBootstrap::LocalOnly) return RAD_OK;
    if (bootstrap != r3::epoch::AdmissionBootstrap::Ready) return RAD_E_STATE;
    uint32_t expected = g.engine_ranks.load(std::memory_order_acquire);
    if (!expected) {
        uint32_t empty = 0;
        if (g.engine_ranks.compare_exchange_strong(empty,ranks,
                std::memory_order_acq_rel,std::memory_order_acquire)) {
            __atomic_store_n(&placement_control(g.map)->expected_ranks,
                             (1u << ranks) - 1u, __ATOMIC_RELEASE);
            expected = ranks;
        } else expected = empty;
    }
    if (expected != ranks) {
        g.ledger.fault();
        return RAD_E_STATE;
    }
    /* Reclaim with event queries only; no serving-path waits for capacity. */
    poll_tickets(g);
    using Result = r3::admission::Ledger::Result;
    const Result r = try_only ? g.ledger.try_admit(ranks, out)
        : g.ledger.admit(ranks, out,
                         [&] { return helper_faulted(g); },
                         [&] { poll_tickets(g); });
    if (r == Result::Closed || (try_only && r == Result::Full)) return 1;
    if (r != Result::Ok) {
        if (r == Result::Full)
            std::fprintf(stderr, "libr3moe: admission route-ring ceiling exhausted; high-water %zu, capacity %zu\n",
                         g.ledger.high_water(), g.ledger.capacity());
        g.ledger.fault();
        return RAD_E_STATE;
    }
    return RAD_OK;
}

int bind_ticket(HookState& g, StepTicket ticket, int64_t step) {
    return ticket.ticket_id && g.ledger.bind(ticket.ticket_id, step) ? RAD_OK : RAD_E_STATE;
}
int cancel_ticket(HookState& g, StepTicket ticket) {
    return !ticket.ticket_id || g.ledger.cancel(ticket.ticket_id) ? RAD_OK : RAD_E_STATE;
}
int host_done_ticket(HookState& g, StepTicket ticket) {
    if (!ticket.ticket_id) return RAD_OK;
    if (!g.ledger.host_complete(ticket.ticket_id)) return RAD_E_STATE;
    poll_tickets(g);
    return g.ledger.stopped() ? RAD_E_STATE : RAD_OK;
}

/* The setter takes the exact admitted ticket. Every pass belonging to one
 * scheduler parent (trunk, history and draft passes) receives the same value,
 * including the pass issued ahead before that parent's host commit. */
int before_pass(HookState& g, int rank, StepTicket value, void* stream) {
    if (!value.ticket_id && !g.remote_possible) return RAD_OK;
    if (rank < 0 || rank >= (int)g.rank.size() || !stream ||
        !g.ledger.valid(value, (uint32_t)rank)) return RAD_E_STATE;
    const R3MoeMailboxDevice* dev = nullptr;
    {
        std::lock_guard<std::mutex> lock(g.mu);
        if (!g.hdr || !g.pinned || !g.rank[rank].ready || !g.rank[rank].ticket ||
            load(&g.hdr->helper_fault)) return RAD_E_STATE;
        dev = &g.rank[rank].dev;
        if (!g.completion[rank].arm(value)) return RAD_E_STATE;
    }
    return r3_moe_ticket_launch(dev, value, stream);   /* also arms the device counting slot */
}

/* Distinct events per admitted parent AND per rank, recorded by the
 * scheduler after all parent GPU effects, before issue-ahead. g.mu only
 * orders against close_mailbox; the ranks are parked here. */
int rank_done_ticket(HookState& g, int rank, StepTicket ticket, void* stream) {
    if (!ticket.ticket_id) return RAD_OK;
    if (!stream || rank < 0 || rank >= (int)g.rank.size() ||
        !g.ledger.valid(ticket, (uint32_t)rank)) return RAD_E_STATE;
    std::lock_guard<std::mutex> lock(g.mu);
    /* The seal copies the parent's device counts to the host-mapped stamp; the event recorded behind it is the proof
     * that they (and every earlier effect of the parent) reached host-visible memory. */
    if (!g.rank[rank].ready || r3_moe_seal_launch(&g.rank[rank].dev, ticket, stream) != RAD_OK) {
        std::fprintf(stderr, "libr3moe: parent rank seal kernel launch failed\n");
        g.ledger.fault();
        return RAD_E_DEVICE;
    }
    if (!g.completion[rank].record(ticket.ticket_id, stream)) {
        std::fprintf(stderr, "libr3moe: record parent rank completion: rad_event_record failed or repeated\n");
        g.ledger.fault();
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}

bool rank_completion_ok(bool prepared) {
    if (!prepared)
        std::fprintf(stderr, "libr3moe: rank parent completion events: rad_event_create failed\n");
    return prepared;
}

/* One ledger pass over host-complete parents; per rank it stops at the
 * first pending event (Ledger::poll). No allocation, no device switch. */
void poll_tickets(HookState& g) {
    if (helper_faulted(g)) { g.ledger.fault(); return; }
    (void)g.ledger.poll([&](const StepTicket& t, uint32_t rank) {
        return g.completion[rank].prove(t);
    });
}

int wait_finished_tickets(HookState& g, int64_t timeout_ms) {
    if (timeout_ms < 0) return RAD_E_INVAL;
    const auto cutoff = g.ledger.finish_admission();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    const auto result = g.ledger.wait_drained(cutoff.old_epoch, cutoff.latest_ticket, deadline,
                                              [&] { poll_tickets(g); }, [&] { return helper_faulted(g); });
    return result == r3::admission::Ledger::Result::Ok ? RAD_OK : RAD_E_STATE;
}

bool served_lane(HookState& g, uint32_t lane) {
    return lane < g.hdr->counts.lanes &&
           (!lane_is_small(lane, g.hdr->slot_count) || lane_class(lane, g.hdr->slot_count) < g.hdr->small_class_count);
}
void hold_burst_for_drain(Header* h);

/* Admission is closed and every admitted parent has retired, so ending the burst cuts off no work. The drain words
 * are stored before the burst word moves: a helper that sees the move (burst spin ends, or its futex wait on the word
 * returns) then sees its drain request. An idle word moves by 2, so it stays even and still wakes sleepers. */
void request_helper_drain(HookState& g, uint64_t closure) {
    Header* h = g.hdr;
    for (uint32_t lane = 0; lane < h->counts.lanes; ++lane)
        if (served_lane(g, lane))
            __atomic_store_n(&h->lane[lane].drain_request, closure, __ATOMIC_RELEASE);
    {
        std::lock_guard<std::mutex> lock(g.burst_mu);
        g.burst_draining = true;
        hold_burst_for_drain(h);
    }
    notify(h);   /* acknowledge_drain's closure wait and the engine's park waits use the generation word */
}

bool helper_drained(HookState& g, uint64_t closure) {
    if (!g.hdr || load(&g.hdr->helper_fault)) return false;
    const Header* h = g.hdr;
    for (uint32_t lane = 0; lane < h->counts.lanes; ++lane)
        if (!lane_quiesced(h, lane, closure, served_lane(g, lane))) return false;
    for (uint32_t r = 0; r < h->rank_count; ++r)
        if (!combine_quiesced(h->combine[r])) return false;
    return true;
}

void publish_layer_map(HookState& g, int layer, int n_expert, const int32_t* slot_of_expert) {
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
    g.watch.notice();
    /* The plugin ABI gives this callback no error return. The wait is bounded by the helper protocol's readiness
     * deadline (layer_map_wait.h); a helper that is alive but never consumes raises the sticky helper fault, so the
     * engine's first remote MoE wait fails and teardown is not blocked. */
    const MapWait r = await_layer_consumed(g.hdr, layer, layer_map_bound());
    if (r == MapWait::Timeout)
        std::fprintf(stderr, "libr3: R3_MOE_WORKER layer %d: helper did not consume its map within the readiness "
                     "deadline; failing the helper path\n", layer);
    else if (r != MapWait::Consumed)
        std::fprintf(stderr, "libr3: R3_MOE_WORKER layer %d: helper failed before consuming its map\n", layer);
}

/* Begin: wait only for lanes a helper is still draining (Header::burst; normally none, so no sleep), so nothing is
 * published into released waits. A fault leaves the engine's next GPU wait to trap; nothing to wait for here. */
bool wait_parks(Header* h,const timespec* deadline = nullptr) {
    const WaitResult r=r3::epoch::wait_parks_checked(h,deadline);
    if (r != WaitResult::Ready) {
        std::fprintf(stderr,
            "libr3moe: park barrier failed fault=%u wait=%u\n",
            uint32_t(r==WaitResult::Timeout ? r3::epoch::FaultCode::QUIESCE_TIMEOUT :
                     r==WaitResult::Fault ? r3::epoch::FaultCode::HELPER_DEAD :
                                            r3::epoch::FaultCode::DEVICE),uint32_t(r));
        return false; // wait_parks_checked already raised the sticky fault
    }
    return true;
}

/* Under g.burst_mu: move the word to `active`'s parity. True when it opened a burst (the caller waits for parks). */
bool apply_burst(HookState& g, Header* h, bool active) {
    const uint32_t b = load(&h->burst) & ~kBurstStop;
    if (active == (bool)(b & 1u)) return false;
    if (active) g.watch.notice();
    (void)__atomic_fetch_add(&h->burst, 1u, __ATOMIC_SEQ_CST);
    (void)futex_wake(&h->burst);
    return active;
}

/* Under g.burst_mu, with burst_draining set: end a running burst, or move an idle word by 2 (parity unchanged), so
 * burst spins exit and futex sleepers on the word wake to read their drain request. */
void hold_burst_for_drain(Header* h) {
    const uint32_t b = load(&h->burst) & ~kBurstStop;
    (void)__atomic_fetch_add(&h->burst, (b & 1u) ? 1u : 2u, __ATOMIC_SEQ_CST);
    (void)futex_wake(&h->burst);
}

/* An inference burst begins or ends (RAD20: the helpers arm GPU waits only inside a burst, so their queues are
 * empty and the GPUs idle between bursts). R3_MOE_WORKER_BURST_TRACE=1 logs each transition with a monotonic
 * timestamp, for checking begin/end pairing across single-stream, multi-stream and MTP runs. */
void burst_transition(HookState& g, int active) {
    static const bool trace = false;   // R3_MOE_WORKER_BURST_TRACE (a RAD20 transition log) is no longer read from the environment (RSC12)
    static std::atomic<uint64_t> begins{0}, ends{0};
    const uint64_t n = active ? begins.fetch_add(1) + 1 : ends.fetch_add(1) + 1;
    /* One engine process drives the segment's burst word. During a helper drain only radiance's wanted state is
     * recorded; reopen applies it. A helper that stops clears its drain flags and sets helper_fault. */
    if (g.hdr) {
        Header* h = g.hdr;
        bool opened = false;
        {
            std::lock_guard<std::mutex> lock(g.burst_mu);
            g.burst_want = active != 0;
            if (!g.burst_draining) opened = apply_burst(g, h, g.burst_want);
        }
        if (opened) (void)wait_parks(h);   /* a failure raised helper_fault; the next pass fails closed on it */
    }
    if (trace) {
        timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        std::fprintf(stderr, "libr3moe: burst %s #%llu t=%lld.%06ld\n", active ? "begin" : "end",
                     (unsigned long long)n, (long long)ts.tv_sec, ts.tv_nsec / 1000);
    }
}

/* Reopen, after the drain was verified and the drain words cleared: radiance's wanted burst state takes the word
 * again, parks waited for as at a begin, before admission lets the scheduler publish. */
bool resume_burst_after_drain(HookState& g, Header* h) {
    bool opened = false;
    {
        std::lock_guard<std::mutex> lock(g.burst_mu);
        g.burst_draining = false;
        opened = apply_burst(g, h, g.burst_want);
    }
    if (!opened) return true;
    // The phase deadline is cleared when the controller enters ACTIVE.
    // Derive a fresh bound from the existing quiescence phase duration.
    timespec deadline{};
    if (clock_gettime(CLOCK_MONOTONIC,&deadline) != 0) {
        raise_fault(h);
        return false;
    }
    const std::chrono::nanoseconds bound(r3::epoch::Deadlines{}.quiesce_ns);
    const auto seconds=std::chrono::duration_cast<std::chrono::seconds>(bound);
    deadline.tv_sec += seconds.count();
    deadline.tv_nsec += (bound-seconds).count();
    if (deadline.tv_nsec >= std::nano::den) {
        ++deadline.tv_sec;
        deadline.tv_nsec -= std::nano::den;
    }
    return wait_parks(h,&deadline);
}

/* This callback is engine weight-residency setup, not a step route. The
 * local/remote split stays immutable in Stage 1; GPU routes select by ticket. */
int owner_query(HookState& g, int layer, int expert) {
    if (!g.hdr || layer < 0 || expert < 0 ||
        layer >= (int)g.hdr->layer_count || expert >= (int)g.hdr->expert_count) {
        if (g.hdr) { g.ledger.fault(); raise_fault(g.hdr); }
        return 0;
    }
    /* Helper 0 can seal the final owner map while engine residency is being
     * constructed. Read generation metadata within the control seqlock:
     * do not mistake an in-flight coherent bootstrap publication for a bad
     * digest and poison serving. Owner bytes are already final at the seal. */
    for (uint32_t attempt=0;attempt<64;++attempt) {
        PlacementControl c{};
        if (!placement_snapshot(g.map, &c)) continue;
        if (!c.active_epoch || c.active_slot > 1 || c.fault_code) break;
        const auto* generation = generation_header(g.map,c.active_slot);
        const uint64_t epoch = generation->epoch, digest = generation->digest;
        const uint8_t owner = owner_at(g.map,c.active_slot,(uint32_t)layer,(uint32_t)expert);
        if (__atomic_load_n(&placement_control(g.map)->seq, __ATOMIC_ACQUIRE) != c.seq)
            continue;
        if (epoch != c.active_epoch || digest != c.active_digest ||
            (owner != kOwnerLocal && owner >= g.hdr->helper_count)) break;
        return owner < g.hdr->helper_count;
    }
    g.ledger.fault();
    raise_fault(g.hdr);
    return 0;
}

void free_owner_generations(RankState& s) {
    if (s.owner_stream) (void)hipStreamSynchronize(s.owner_stream); /* shutdown/failed startup only */
    for (uint32_t slot = 0; slot < 2; ++slot) {
        if (s.owner_event[slot]) (void)hipEventDestroy(s.owner_event[slot]);
        if (s.owner_stage[slot]) (void)hipHostFree(s.owner_stage[slot]);
        if (s.owner[slot]) (void)hipFree(s.owner[slot]);
        s.owner_event[slot] = nullptr;
        s.owner_stage[slot] = nullptr;
        s.owner[slot] = nullptr;
        s.owner_expected[slot] = {};
        s.owner_queued[slot] = false;
    }
    if (s.owner_binding) (void)hipFree(s.owner_binding);
    if (s.owner_stream) (void)hipStreamDestroy(s.owner_stream);
    s.owner_binding = nullptr;
    s.owner_stream = nullptr;
    s.owner_bytes = 0;
}

/* With the rank's device current. Serving streams have drained on normal close;
 * a failed attach has only its private owner-upload stream to drain. */
void release_rank(RankState& s, RankCompletion& completion) {
    completion.release();
    free_owner_generations(s);
    if (s.heat_host) (void)hipHostFree(s.heat_host);
    if (s.dev.shadow_first) (void)hipFree(s.dev.shadow_first);
    if (s.dev.shadow_stats) (void)hipFree(s.dev.shadow_stats);
    if (s.mask_ref) (void)hipFree(s.mask_ref);
    if (s.maps) (void)hipFree(s.maps);
    if (s.published) (void)hipFree(s.published);
    if (s.sequence) (void)hipFree(s.sequence);
    if (s.ticket) (void)hipFree(s.ticket);
    if (s.consumed_host) (void)hipHostFree(s.consumed_host);
    if (s.stamps) (void)hipFree(s.stamps);
    if (s.consumed_rec) (void)hipFree(s.consumed_rec);
    s = RankState{};
}

/* Under g.mu, on the owning device. The host bitmap and binding live in
 * pinned per-slot staging until the rank-private event completes. No host
 * stream wait or system-scope GPU operation is added to inference. */
bool queue_owner_upload(RankState& s, uint32_t slot, uint64_t epoch, uint64_t digest,
                        const std::vector<uint8_t>& bitmap) {
    if (slot > 1 || !epoch || epoch > UINT32_MAX || !digest ||
        !s.owner_stream || !s.owner_event[slot] || !s.owner_stage[slot] ||
        !s.owner[slot] || !s.owner_binding || bitmap.size() != s.owner_bytes) return false;
    if (s.owner_queued[slot]) {
        const hipError_t q = hipEventQuery(s.owner_event[slot]);
        if (q != hipSuccess) return false; /* still copying or an error: fail closed */
    }
    /* A failed second enqueue or event record must not leave an earlier
     * completed event claiming a partially replaced generation is ready.
     * Synchronize only on error, so pinned staging cannot be overwritten
     * while a preceding copy is still in flight. */
    s.owner_queued[slot] = false;
    s.owner_expected[slot] = {};
    const R3MoeOwnerBinding binding{epoch, digest};
    std::memcpy(s.owner_stage[slot], &binding, sizeof(binding));
    std::memcpy(s.owner_stage[slot] + sizeof(binding), bitmap.data(), bitmap.size());
    const bool issued =
        hip_ok(hipMemcpyAsync(s.owner[slot], s.owner_stage[slot] + sizeof(binding),
                              bitmap.size(), hipMemcpyHostToDevice, s.owner_stream),
               "upload owner bitmap") &&
        hip_ok(hipMemcpyAsync(s.owner_binding + slot, s.owner_stage[slot],
                              sizeof(binding), hipMemcpyHostToDevice, s.owner_stream),
               "upload owner binding") &&
        hip_ok(hipEventRecord(s.owner_event[slot], s.owner_stream),
               "record owner upload event");
    if (!issued) {
        (void)hip_ok(hipStreamSynchronize(s.owner_stream), "drain failed owner upload");
        return false;
    }
    s.owner_expected[slot] = binding;
    s.owner_queued[slot] = true;
    return true;
}

bool allocate_owner_generations(RankState& s, size_t bitmap_bytes) {
    if (!bitmap_bytes ||
        !hip_ok(hipStreamCreateWithFlags(&s.owner_stream, hipStreamNonBlocking),
                "create owner upload stream") ||
        !hip_ok(hipMalloc(reinterpret_cast<void**>(&s.owner_binding),
                          2 * sizeof(R3MoeOwnerBinding)), "allocate owner bindings") ||
        !hip_ok(hipMemset(s.owner_binding, 0, 2 * sizeof(R3MoeOwnerBinding)),
                "zero owner bindings")) return false;
    s.owner_bytes = bitmap_bytes;
    for (uint32_t slot = 0; slot < 2; ++slot)
        if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&s.owner[slot]), bitmap_bytes),
                    "allocate generation bitmap") ||
            !hip_ok(hipHostMalloc(reinterpret_cast<void**>(&s.owner_stage[slot]),
                                  sizeof(R3MoeOwnerBinding) + bitmap_bytes, hipHostMallocDefault),
                    "allocate pinned owner upload staging") ||
            !hip_ok(hipEventCreateWithFlags(&s.owner_event[slot], hipEventDisableTiming),
                    "create generation upload event")) return false;
    return true;
}

void placement_loop(HookState& g);

/* Undo a half-finished open: nothing it mapped or pinned stays behind. */
void abandon_mailbox(HookState& g) {
    if (g.pinned) (void)hipHostUnregister(g.map);
    if (g.map != MAP_FAILED) ::munmap(g.map, g.bytes);
    if (g.fd >= 0) ::close(g.fd);
    g.hdr = nullptr;
    g.rank.clear();
    g.completion.clear();
    g.map = MAP_FAILED;
    g.fd = -1;
    g.pinned = false;
}

/* Open the helper hand-off for one engine generation: map and pin the shared segment, watch the helpers, wait for them
 * to register, initialise the admission ledger to the engine's ceiling and start the placement thread. Returns whether the
 * hook is live; on failure nothing is left mapped (the caller refuses the generation). */
bool open_mailbox(HookState& g, uint32_t capacity_limit) {
    if (!enabled_env()) return false;
    g.fd = ::shm_open(shm_name(), O_RDWR | O_CLOEXEC, 0);
    if (g.fd < 0) {
        // errno tells a segment that is not there (ENOENT: no helper has made it, or something removed it) from one that is
        // there but cannot be opened (EACCES, EMFILE); without the hand-off the engine plans every expert locally.
        std::fprintf(stderr, "libr3: R3_MOE_WORKER shm %s is not ready: %s\n", shm_name(), std::strerror(errno));
        return false;
    }
    struct stat st{};
    if (::fstat(g.fd, &st) != 0 || st.st_size < (off_t)sizeof(Header)) { abandon_mailbox(g); return false; }
    g.bytes = (size_t)st.st_size;
    g.map = ::mmap(nullptr, g.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, g.fd, 0);
    if (g.map == MAP_FAILED || !valid_self(static_cast<Header*>(g.map), g.bytes)) {
        std::fprintf(stderr, "libr3: R3_MOE_WORKER shared-memory protocol mismatch\n");
        abandon_mailbox(g);
        return false;
    }
    g.hdr = static_cast<Header*>(g.map);
    g.rank.resize(g.hdr->rank_count);
    g.completion.resize(g.hdr->rank_count);
    if (!hip_ok(hipHostRegister(g.map, g.bytes, hipHostRegisterPortable | hipHostRegisterMapped),
                "could not pin/map shared segment")) {
        g.hdr = nullptr;
        abandon_mailbox(g);
        return false;
    }
    g.pinned = true;
    /* layer_map: the runtime slot maps the engine publishes at declare. owns_expert:
     * ownership for placement (R3_MOE_WORKER_SKIP_OWNED), the helper's bitmap as published before the engine
     * starts. The engine applies the protected-layer and slot tests itself, the same ones the helper uses when
     * it narrows ownership to the runtime maps, so both sides agree on what is never placed here. */
    /* Watch the helpers before the engine publishes its layer maps: publishing waits for the helpers to consume them, and
     * a helper that dies there must fail that wait. No watch, no helpers (fail closed). */
    if (!g.watch.start(g.hdr)) {
        abandon_mailbox(g);
        return false;
    }
    /* Open barrier: every helper has published its identity and the watcher holds a validated pidfd for it,
     * before anything waits on the helpers. A helper that died before publishing its pid has no process to watch, so
     * this bounded startup wait (blocking, not polled) is what catches it. On timeout or fault the helpers are failed
     * (sticky helper_fault): the generation still opens, its map waits return at once and its first remote step
     * fails, rather than serving with experts it was told the helpers hold. */
    {
        constexpr int kRegisterTimeoutS = 120;   /* helpers attach before the engine starts; this only bounds a failure */
        timespec deadline{};
        clock_gettime(CLOCK_MONOTONIC, &deadline);
        deadline.tv_sec += kRegisterTimeoutS;
        const WaitResult r = wait_for(g.hdr, &deadline, [&] {
            g.watch.notice();
            const uint32_t n = load(&g.hdr->helper_count);
            for (uint32_t i = 0; i < n; ++i)
                if (!g.watch.watching(i)) return false;
            return n > 0;
        });
        if (r != WaitResult::Ready) {
            std::fprintf(stderr, "libr3: R3_MOE_WORKER helpers not all registered and watched within %d s (%s); "
                         "failing the helper path\n", kRegisterTimeoutS,
                         r == WaitResult::Timeout ? "timeout" : r == WaitResult::Fault ? "helper fault" : "wait error");
            raise_fault(g.hdr);
        }
    }
    uint32_t epoch = 0;
    uint8_t slot = 0;
    if (!committed(g, &epoch, &slot) || !g.ledger.initialize(epoch, slot)) {
        raise_fault(g.hdr);
        g.watch.stop();
        abandon_mailbox(g);
        return false;
    }
    /* The block's declaration and plugin admission share ONE typed manifest policy: a generation exists only when
     * the engine's manifest asks for a hand-off, so every step may use it. */
    g.remote_possible = true;
    if (!g.ledger.configure(capacity_limit)) {
        std::fprintf(stderr, "libr3: R3_MOE_WORKER the engine's admission ceiling %u is refused by the ledger\n",
                     capacity_limit);
        g.watch.stop();
        abandon_mailbox(g);
        return false;
    }

    /* The engine policy only decides whether the graph contains/uses the two mailbox ops.
     * All route selection and synchronisation stays on the rank GPU. */
    /* Explicit engine bounds are policy. Supply defaults only when the caller did not set them;
     * in particular R3_MOE_REMOTE_MAX_ROWS=0 must disable hand-off without disabling the helper. */
    uint32_t owned = 0;
    const uint32_t active = placement_control(g.map)->active_slot;
    for (uint32_t l = 0; l < g.hdr->layer_count; ++l)
        for (uint32_t e = 0; e < g.hdr->expert_count; ++e)
            owned += owner_at(g.map, active, l, e) < g.hdr->helper_count;
    std::fprintf(stderr,
        "libr3: R3_MOE_WORKER GPU mailbox ready, %u owned experts, decode rows <= %lld\n",
        owned, (long long)max_rows());
    if (poison_env()) std::fprintf(stderr, "libr3: R3_MOE_WORKER poison\n");
    if (shadow_env()) std::fprintf(stderr, "libr3: R3_MOE_WORKER_SHADOW enabled\n");
    if (mask_only_env()) std::fprintf(stderr, "libr3: R3_MOE_WORKER_MASK_ONLY enabled\n");
    g.placement_thread = std::thread(placement_loop, std::ref(g));
    return true;
}

/* Engine participant of the shared helper-zero placement coordinator.
 * Same-map diagnostic cycles use TWO real generations and the complete
 * RESERVED/COPYING/PREPARED/QUIESCING/COMMIT/RETIRING path. Actual owner
 * movement and helper weight-table changes remain later slices.
 *
 * All GPU uploads/events are on private rank upload streams. This thread
 * polls events away from serving, and reuses the slice-C admission ledger,
 * parent seal/events, helper drain and reopen; there is no second gate. */
// Implement the extracted per-phase engine-rank coordinator with real HIP,
// admission, park and identity operations. Only this adapter touches g.mu.
struct LiveRankOperations final : r3::epoch::RankOperations {
    HookState& g;
    explicit LiveRankOperations(HookState& hook) : g(hook) {}
    R3MoeDrainSnapshot snap{};
    uint32_t actual_ranks() override {
        return g.engine_ranks.load(std::memory_order_acquire);
    }
    bool registered(uint32_t rank) override {
        std::lock_guard<std::mutex> lock(g.mu);
        return g.rank[rank].ready;
    }
    r3::epoch::RankResult upload(uint32_t rank,uint32_t slot,
                                 uint64_t epoch,uint64_t digest) override {
        const int r=r3_moe_owner_generation_upload(g.serial, (int)rank,slot,epoch,digest);
        return r == RAD_OK ? r3::epoch::RankResult::Ready :
               r == RAD_E_DEVICE ? r3::epoch::RankResult::DeviceError :
                                   r3::epoch::RankResult::Pending;
    }
    r3::epoch::RankResult ready(uint32_t rank,uint32_t slot,
                                uint64_t epoch,uint64_t digest) override {
        const int r=r3_moe_owner_generation_ready(g.serial, (int)rank,slot,epoch,digest);
        return r == RAD_OK ? r3::epoch::RankResult::Ready :
               r == RAD_E_DEVICE ? r3::epoch::RankResult::DeviceError :
                                   r3::epoch::RankResult::Pending;
    }
    r3::epoch::RankResult abort_upload_done(uint32_t rank,uint32_t slot,
                                              uint64_t epoch,uint64_t digest) override {
        std::lock_guard<std::mutex> lock(g.mu);
        if (rank>=g.rank.size() || slot>1) return r3::epoch::RankResult::DeviceError;
        const RankState& s=g.rank[rank];
        if (!s.ready || !s.owner_queued[slot] ||
            s.owner_expected[slot].epoch!=epoch || s.owner_expected[slot].digest!=digest)
            return r3::epoch::RankResult::DeviceError;
        if (!hip_ok(hipSetDevice(s.device),"select aborted owner event device"))
            return r3::epoch::RankResult::DeviceError;
        const auto status=hipEventQuery(s.owner_event[slot]);
        if (status==hipSuccess) return r3::epoch::RankResult::Ready;
        if (status==hipErrorNotReady) return r3::epoch::RankResult::Pending;
        (void)hip_ok(status,"query aborted owner upload event");
        return r3::epoch::RankResult::DeviceError;
    }
    bool exact_binding(uint32_t rank,uint32_t slot,
                       uint64_t epoch,uint64_t digest) override {
        std::lock_guard<std::mutex> lock(g.mu);
        const RankState& s=g.rank[rank];
        return slot < 2 && s.ready && s.owner_queued[slot] &&
               s.owner_expected[slot].epoch == epoch &&
               s.owner_expected[slot].digest == digest;
    }
    bool close_and_wait(uint64_t deadline,uint64_t* cutoff,uint32_t* old_epoch) override {
        if (r3_moe_close_admission(g.serial, &snap) != RAD_OK ||
            r3_moe_wait_drained(g.serial, &snap, deadline) != RAD_OK ||
            r3_moe_all_ranks_consumed(g.serial, &snap) != 1) return false;
        *cutoff=snap.cutoff_ticket_id;
        *old_epoch=snap.old_epoch;
        return true;
    }
    bool consumed() override { return r3_moe_all_ranks_consumed(g.serial, &snap) == 1; }
    bool drain_verified() override {
        std::lock_guard<std::mutex> lock(g.mu);
        return g.drain_verified && snap.closure_id &&
               g.closed_snapshot.closure_id == snap.closure_id;
    }
    bool helper_quiesced() override { return helper_drained(g, snap.closure_id); }
    bool generation_referenced(uint32_t slot) override {
        return g.ledger.generation_referenced(slot);
    }
    bool can_replace(uint32_t slot) override {
        return g.ledger.can_replace_generation(slot);
    }
    bool reopen(uint32_t epoch,uint32_t slot) override {
        if (r3_moe_reopen_admission(g.serial, epoch, slot) != RAD_OK) return false;
        std::fprintf(stderr,
            "libr3moe: placement cycle %llu reopened epoch %u slot %u\n",
            (unsigned long long)snap.closure_id,epoch,slot);
        snap={};
        return true;
    }
    bool stopped() override { return g.ledger.stopped() || helper_faulted(g); }
    void fatal() override { g.ledger.fault(); raise_fault(g.hdr); }
};

void placement_loop(HookState& g) {
    r3::epoch::EngineRankCoordinator coordinator;
    LiveRankOperations ops(g);
    while (!g.placement_stop.load(std::memory_order_acquire)) {
        PlacementControl c{};
        if (ops.stopped()) break;
        if (!placement_snapshot(g.map,&c)) {
            std::this_thread::yield();
            continue;
        }
        if (c.fault_code) { g.ledger.fault(); break; }
        if (!coordinator.step(g.map,c,ops)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
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
 * to <path>.rank<r>. A helper given it as R3_MOE_WORKER_HEAT owns the coldest experts first. With R3_MOE_WORKER_HISTOGRAM_PERIOD_S
 * the same cumulative file is also rewritten while the engine runs (RHP02): a reader differences two snapshots to get the routes in
 * between. The file is replaced by rename, so a reader never sees a half-written one. `final`: the engine is shutting down
 * (sync first, report the file). */
void write_heat(HookState& g, uint32_t rank, RankState& s, bool final) {
    const auto settings = bc_engine_settings_snapshot();
    const char* path = settings.histogram_file;
    if (!path || !*path || !s.heat_host) return;
    /* The counts live in pinned host memory (RHP02): a snapshot is a CPU copy, with no HIP stream or queue of its own. */
    if (final) (void)hip_ok(hipDeviceSynchronize(), "sync before route histogram");
    std::vector<uint32_t> heat((size_t)kMaxLayers * g.hdr->expert_count);
    for (size_t i = 0; i < heat.size(); ++i) heat[i] = __atomic_load_n(&s.heat_host[i], __ATOMIC_RELAXED);
    // The engine launcher provides a fresh nonce only to its serving child.
    // Legacy standalone launches still publish the unchanged raw file.
    const char* nonce = settings.heat_launch_nonce;
    const bool ok = r3::heat_publication::publish(path, rank, g.hdr->expert_count, heat,
                                                  nonce ? std::string_view(nonce) : std::string_view{});
    if (final)
        std::fprintf(stderr, "libr3: R3_MOE_WORKER route histogram %s.rank%u (%zu counts)%s\\n", path, rank,
                     heat.size(), ok ? "" : " NOT written");
}

/* The live snapshot thread: every R3_MOE_WORKER_HISTOGRAM_PERIOD_S seconds, each ready rank's histogram. It takes g.mu per snapshot, so
 * it never reads a rank that close_mailbox is tearing down (close_mailbox stops and joins it first, without holding g.mu). */
void snapshot_loop(HookState& g, double period_s) {
    auto next = std::chrono::steady_clock::now() + std::chrono::duration<double>(period_s);
    while (!g.snapshot_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (std::chrono::steady_clock::now() < next) continue;
        next += std::chrono::duration<double>(period_s);
        std::lock_guard<std::mutex> lock(g.mu);
        for (uint32_t r = 0; r < g.rank.size(); ++r) {
            RankState& s = g.rank[r];
            if (!s.ready || !s.heat_host) continue;
            write_heat(g, r, s, false);
        }
    }
}

/* `clean`: the engine drained every ticket and nothing faulted (Ledger::clean_stop), so the ledger is healthy and is
 * never interrupted: the placement thread is woken and joined with no deadline of its own. It leaves on its own: its
 * loop checks `placement_stop` every iteration, and any wait inside a cycle ends at that wait's own protocol deadline
 * (Ledger::wait_drained turns a missed deadline into a ledger fault, and a helper fault ends the loop), so a stuck cycle
 * is observed as a fault state, never timed out by a literal here. Not clean (a fault, or tickets that never retired):
 * the ledger is interrupted, which wakes quiescence and blocked scheduler admission. */
void close_mailbox(HookState& g, bool clean) {
    g.placement_stop.store(true, std::memory_order_release);
    if (clean) g.ledger.wake();
    else g.ledger.interrupt();
    if (g.placement_thread.joinable()) g.placement_thread.join();
    g.snapshot_stop = true;
    if (g.snapshotter.joinable()) g.snapshotter.join();
    std::lock_guard<std::mutex> lock(g.mu);
    if (g.opened)
        std::fprintf(stderr, "libr3moe: admission high-water %zu, capacity %zu, outstanding %zu\n",
                     g.ledger.high_water(), g.ledger.capacity(), g.ledger.outstanding());
    g.watch.stop();   /* first: helpers may exit while the engine tears down, and that is not a fault */
    for (uint32_t r = 0; r < g.rank.size(); ++r) {
        RankState& s = g.rank[r];
        if (s.device >= 0) (void)hipSetDevice(s.device);
        if (s.ready) {
            report_shadow(r, s);
            write_heat(g, r, s, true);
        }
        release_rank(s, g.completion[r]);
    }
    if (g.pinned) (void)hipHostUnregister(g.map);
    if (g.map != MAP_FAILED) ::munmap(g.map, g.bytes);
    if (g.fd >= 0) ::close(g.fd);
    g.hdr = nullptr;
    g.rank.clear();
    g.completion.clear();
    g.map = MAP_FAILED;
    g.fd = -1;
    g.pinned = false;
}

/* The one process-wide object left, and it is inherent: the helper segment is a single named, process-wide resource,
 * so one engine holds it at a time and a second asking is refused with BUSY before anything is mapped. g_live is a
 * non-owning mutual-exclusion and lookup pointer only (no policy or state hangs off it): the generation owns the state
 * and its close frees it. Entry points take the generation's serial, never an address: a handle from a closed
 * generation (or another engine's) no longer matches the live serial and is refused before any side effect, even when a
 * later hook is allocated at the same address. The engine closes a generation only after its threads have stopped. */
LiveHook<R3MoeHook> g_live;
HookState& hook(void* self) { return *static_cast<HookState*>(self); }
HookState* live_hook(uint64_t id) { return g_live.resolve(id); }

/* Callbacks of the engine's generation: thin adapters from the engine's `self`-carrying table to this hook. The
 * hook is the generation's own R3MoeHook, which `self` names. */
void cb_layer_map(void* self, int layer, int n, const int32_t* m) { publish_layer_map(hook(self), layer, n, m); }
int cb_owns_expert(void* self, int layer, int expert) { return owner_query(hook(self), layer, expert); }
void cb_burst(void* self, int active) { burst_transition(hook(self), active); }
int cb_before_pass(void* self, int rank, StepTicket t, void* stream) { return before_pass(hook(self), rank, t, stream); }
int cb_admit(void* self, StepTicket* out, int try_only, int world) { return admit_ticket(hook(self), out, try_only, world); }
int cb_bind(void* self, StepTicket t, int64_t step) { return bind_ticket(hook(self), t, step); }
int cb_cancel(void* self, StepTicket t) { return cancel_ticket(hook(self), t); }
int cb_host_done(void* self, StepTicket t) { return host_done_ticket(hook(self), t); }
int cb_rank_done(void* self, int rank, StepTicket t, void* stream) { return rank_done_ticket(hook(self), rank, t, stream); }
/* The fault stop: interrupt so every blocked admission and completion wait fails now. A clean stop never calls this. */
void cb_fault(void* self) { hook(self).ledger.interrupt(); }
void cb_finish_admission(void* self) { (void)hook(self).ledger.finish_admission(); }
int cb_wait_finished(void* self, int64_t ms) { return wait_finished_tickets(hook(self), ms); }
void cb_close(void* self) {
    HookState& g = hook(self);
    close_mailbox(g, g.ledger.clean_stop());
    /* Unpublish before the state is freed: the kernels' init and the direct entry points then refuse. */
    g_live.unpublish(&g);
    g.opened = false;
    delete &g;
}

}  // namespace

/* The engine's explicit factory (abi/bc_moe_remote.h, BC_MOE_REMOTE_OPEN_SYMBOL), found on this plugin's own handle.
 * The state lives in the generation (this allocates it; cb_close frees it). Only one generation can hold the helper
 * segment at a time, because the segment is one named process-wide resource: a second engine asking while one holds
 * it is refused with BUSY before anything is mapped. */
extern "C" int bc_moe_remote_plugin_open(const EngineOpenArgs* args, EnginePlugin* out) {
    /* Failure-atomic: every check precedes the first allocation, and every later failure releases what it took; the
     * engine never calls close for a failed open. */
    if (!args || !out || args->struct_size != sizeof(EngineOpenArgs) || out->struct_size != sizeof(EnginePlugin) ||
        args->capacity_limit == 0 || args->ranks < 1 || args->slot_base < 0)
        return kOpenRefused;
    static std::mutex open_mu;
    std::lock_guard<std::mutex> lock(open_mu);
    if (g_live.busy()) return kOpenBusy;
    auto* h = new (std::nothrow) HookState;
    if (!h) return kOpenRefused;
    h->slot_base = args->slot_base;
    /* Publish first: the hook's own placement thread starts inside open_mailbox and calls the entry points by serial. */
    if (!g_live.publish(h)) { delete h; return kOpenBusy; }
    if (!open_mailbox(*h, args->capacity_limit)) { g_live.unpublish(h); delete h; return kOpenRefused; }
    h->opened = true;
    *out = EnginePlugin{sizeof(EnginePlugin), h, cb_close, cb_layer_map, cb_owns_expert, cb_burst, cb_before_pass, cb_admit,
                        cb_bind, cb_cancel, cb_host_done, cb_rank_done, cb_fault, cb_finish_admission,
                        cb_wait_finished};
    return 0;
}

/* The hook a kernel's init belongs to: the engine's claimed tp_group is the slot span's base the generation opened. */
extern "C" uint64_t r3_moe_hook_for_group(long long tp_group) {
    return g_live.serial_if([&](const R3MoeHook& h) { return h.opened && h.slot_base == tp_group; });
}

extern "C" int r3_moe_close_admission(uint64_t h, R3MoeDrainSnapshot* out) {
    HookState* hp = live_hook(h);
    if (!hp) return RAD_E_STATE;
    HookState& g = *hp;
    if (!out || !g.hdr || !g.pinned || helper_faulted(g)) return RAD_E_STATE;
    {
        std::lock_guard<std::mutex> lock(g.mu);
        if (g.closed_snapshot.closure_id || g.closure_id == UINT64_MAX || g.ledger.stopped()) return RAD_E_STATE;
        const auto snap = g.ledger.close_admission();
        *out = R3MoeDrainSnapshot{++g.closure_id, snap.latest_ticket, snap.old_epoch,
                                 (uint32_t)snap.outstanding, (uint32_t)snap.high_water,
                                 (uint32_t)snap.capacity};
        g.closed_snapshot = *out;
        g.drain_verified = false;
    }
    std::fprintf(stderr, "libr3moe: placement admission closed; outstanding %u, "
                         "high-water %u, capacity %u\n",
                 out->outstanding, out->high_water, out->capacity);
    return RAD_OK;
}

/* Read-only coordinator query. A unique closure ID and exact cutoff bind
 * this proof to the existing admission gate; it cannot claim success while
 * new epoch-N parents can still be admitted. poll_tickets checks every
 * participating rank's existing seal+terminal event, and retires in order.
 * No GPU synchronization, new host stamp or helper hot-path traffic. */
extern "C" int r3_moe_all_ranks_consumed(uint64_t h, const R3MoeDrainSnapshot* snap) {
    HookState* hp = live_hook(h);
    if (!hp) return RAD_E_STATE;
    HookState& g = *hp;
    if (!snap) return RAD_E_INVAL;
    std::lock_guard<std::mutex> lock(g.mu);
    const R3MoeDrainSnapshot& closed = g.closed_snapshot;
    if (!g.hdr || !g.pinned || !closed.closure_id ||
        snap->closure_id != closed.closure_id ||
        snap->cutoff_ticket_id != closed.cutoff_ticket_id ||
        snap->old_epoch != closed.old_epoch ||
        snap->outstanding != closed.outstanding ||
        snap->high_water != closed.high_water ||
        snap->capacity != closed.capacity ||
        g.ledger.stopped() || helper_faulted(g))
        return RAD_E_STATE;
    poll_tickets(g);
    if (g.ledger.stopped() || helper_faulted(g)) return RAD_E_STATE;
    return g.ledger.all_ranks_consumed(snap->old_epoch, snap->cutoff_ticket_id) ? 1 : 0;
}

extern "C" int r3_moe_wait_drained(uint64_t h, const R3MoeDrainSnapshot* snap,
                                    uint64_t deadline_monotonic_ns) {
    HookState* hp = live_hook(h);
    if (!hp) return RAD_E_STATE;
    HookState& g = *hp;
    if (!snap || !g.hdr || !g.pinned || !deadline_monotonic_ns) return RAD_E_INVAL;
    {
        std::lock_guard<std::mutex> lock(g.mu);
        if (!snap->closure_id || snap->closure_id != g.closed_snapshot.closure_id ||
            snap->cutoff_ticket_id != g.closed_snapshot.cutoff_ticket_id ||
            snap->old_epoch != g.closed_snapshot.old_epoch)
            return RAD_E_STATE;
    }
    timespec now{};
    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    const uint64_t now_ns = (uint64_t)now.tv_sec * 1000000000u + (uint64_t)now.tv_nsec;
    const auto now_clock = std::chrono::steady_clock::now();
    const auto maximum = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::time_point::max() - now_clock).count();
    if (deadline_monotonic_ns <= now_ns ||
        deadline_monotonic_ns - now_ns > (uint64_t)maximum) {
        g.ledger.fault();
        return RAD_E_INVAL;
    }
    const auto limit = now_clock + std::chrono::nanoseconds(deadline_monotonic_ns - now_ns);
    using Result = r3::admission::Ledger::Result;
    const Result drained = g.ledger.wait_drained(snap->old_epoch, snap->cutoff_ticket_id,
                                                 limit, [&] { poll_tickets(g); },
                                                 [&] { return helper_faulted(g); });
    if (drained != Result::Ok) { g.ledger.fault(); return RAD_E_STATE; }
    /* A successful event query guarantees helper wait and consumption before
     * the event, but explicitly check every lane before controller COMMIT. */
    {
        std::lock_guard<std::mutex> lock(g.mu);
        if (snap->closure_id != g.closed_snapshot.closure_id) return RAD_E_STATE;
        request_helper_drain(g, snap->closure_id);
    }
    while (!helper_drained(g, snap->closure_id)) {
        if (g.ledger.stopped() || helper_faulted(g) || std::chrono::steady_clock::now() >= limit) {
            g.ledger.fault();
            return RAD_E_STATE;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    {
        std::lock_guard<std::mutex> lock(g.mu);
        if (g.ledger.stopped() || helper_faulted(g) ||
            snap->closure_id != g.closed_snapshot.closure_id ||
            !g.ledger.drained(snap->old_epoch, snap->cutoff_ticket_id))
            return RAD_E_STATE;
        g.drain_verified = true;
    }
    return RAD_OK;
}

extern "C" int r3_moe_reopen_admission(uint64_t h, uint32_t epoch, uint32_t slot) {
    HookState* hp = live_hook(h);
    if (!hp) return RAD_E_STATE;
    HookState& g = *hp;
    uint32_t committed_epoch = 0;
    uint8_t committed_slot = 0;
    std::lock_guard<std::mutex> lock(g.mu);
    if (!g.drain_verified ||
        !committed(g, &committed_epoch, &committed_slot) ||
        committed_epoch != epoch || committed_slot != slot)
        return RAD_E_STATE;
    /* Helpers leave their closure wait, then the burst word returns to radiance's state (with its park wait), and
     * only then may the scheduler publish: the ledger reopens last. */
    for (uint32_t lane = 0; lane < g.hdr->counts.lanes; ++lane)
        __atomic_store_n(&g.hdr->lane[lane].drain_request, uint64_t(0), __ATOMIC_RELEASE);
    notify(g.hdr);
    /* Fail closed: a park wait that did not see every lane return, or a helper fault since the commit, keeps
     * admission shut. */
    if (!resume_burst_after_drain(g, g.hdr) || helper_faulted(g)) {
        g.ledger.fault();
        return RAD_E_STATE;
    }
    PlacementControl c{};
    if (!placement_snapshot(g.map,&c) ||
        c.state != uint32_t(PlacementState::ACTIVE) ||
        c.active_epoch != epoch || c.active_slot != slot ||
        !c.active_digest || !c.transaction_id) {
        g.ledger.fault();
        return RAD_E_STATE;
    }
    if (!g.ledger.reopen(epoch, (uint8_t)slot) ||
        !g.ledger.can_replace_generation(1u-slot)) {
        g.ledger.fault();
        return RAD_E_STATE;
    }
    g.drain_verified = false;
    g.closed_snapshot = {};
    /* Release-published only after successful parks AND ledger.reopen.
     * Helper zero cannot stage the old slot based on cleared drain words. */
    return RAD_OK;
}

extern "C" int r3_moe_mailbox_device(uint64_t h, int rank, int world_size, R3MoeMailboxDevice** out) {
    HookState* hp = live_hook(h);
    if (!hp) return RAD_E_STATE;
    HookState& g = *hp;
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
    if (!g.hdr || !g.pinned || world_size < 1 || world_size != (int)g.hdr->rank_count ||
        rank < 0 || rank >= world_size)
        return RAD_E_INVAL;

    // The shared production lock-order boundary waits for the sealed
    // generation BEFORE taking g.mu; layer-map publication needs that lock.
    timespec deadline{};
    if (clock_gettime(CLOCK_MONOTONIC, &deadline) != 0) return RAD_E_STATE;
    deadline.tv_sec += kStartupWaitSeconds;
    return r3::owner_generation::with_bootstrap_rank_lock(
        g.map, g.bootstrap_ready_ranks, (uint32_t)rank, g.mu, &deadline,
        [&]() -> int {
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
    /* RAD49: N helpers behind M ranks, one slot a (helper, rank). The segment is laid out for the engine width the
     * helpers were started for; a different engine is refused here, before any request. */
    if (!slots_fit(helpers, (uint32_t)world_size) || g.hdr->rank_count != (uint32_t)world_size ||
        g.hdr->slot_count != slots_for(helpers, (uint32_t)world_size)) {
        std::fprintf(stderr, "libr3: R3_MOE_WORKER segment is for %u helper(s) behind %u engine rank(s) (%u slots, at most %u), "
                     "this engine has %d rank(s)\n", helpers, g.hdr->rank_count, g.hdr->slot_count, g.hdr->counts.slots, world_size);
        return RAD_E_STATE;
    }
    /* The shadow compare and mask-only control compare a helper row with the engine's own rows for the same slots. Behind
     * TP an engine rank holds only its width slice of a local expert, while the helper row is full width and added once
     * (consumer_of), so neither comparison is defined there. */
    if (world_size > 1 && (shadow_env() || mask_only_env())) {
        std::fprintf(stderr, "libr3: R3_MOE_WORKER_SHADOW / R3_MOE_WORKER_MASK_ONLY need a one-rank engine (world %d): the "
                     "helper rows are full width and added once by the layer's consumer rank\n", world_size);
        return RAD_E_STATE;
    }
    /* Helper 0 already sealed the fully filtered map under the controller
     * seqlock. Do not repair its digest here: the rank upload must consume
     * exactly that immutable publication, or fail closed. */
    PlacementControl sealed{};
    if (!bootstrap_generation_sealed(g.map) ||
        !placement_snapshot(g.map, &sealed) ||
        g.ledger.outstanding()) return RAD_E_STATE;
    const uint32_t active_slot = sealed.active_slot;
    const uint64_t digest = sealed.active_digest;
    for (const RankState& other : g.rank)
        if (other.ready && (other.owner_expected[active_slot].epoch !=
                                sealed.active_epoch ||
                            other.owner_expected[active_slot].digest != digest))
            return RAD_E_STATE;
    /* Digests are controller publications, never rewritten by rank setup. */
    std::vector<uint8_t> owner_bitmap;
    if (!r3::owner_generation::build_bitmap(g.map, active_slot,
            sealed.active_epoch, digest, &owner_bitmap))
        return RAD_E_STATE;
    s.device = device;
    RankCompletion& completion = g.completion[rank];
    const auto rollback = [&completion](RankState* partial) { release_rank(*partial, completion); };
    std::unique_ptr<RankState, decltype(rollback)> attaching(&s, rollback);
    if (!allocate_owner_generations(s, owner_bitmap.size()) ||
        !queue_owner_upload(s, active_slot, sealed.active_epoch,
                            digest, owner_bitmap) ||
        !hip_ok(hipEventSynchronize(s.owner_event[active_slot]),
                "finish bootstrap owner upload"))
        return RAD_E_DEVICE;
    r3::moe_shm::v26::gpu::Scratch scratch{};
    r3::moe_shm::v26::gpu::FinishLaunch finish{};
    int max_threads = 0;
    if (!hip_ok(hipDeviceGetAttribute(&max_threads, hipDeviceAttributeMaxThreadsPerBlock, device),
                "GPU max threads per block") || max_threads <= 0 ||
        r3::moe_shm::v26::gpu::finish_launch({helpers, (uint32_t)world_size}, (uint32_t)max_threads, &finish) !=
            r3::moe_shm::v26::gpu::Refusal::None) return RAD_E_STATE;
    if (r3::moe_shm::v26::gpu::scratch_for({helpers, (uint32_t)world_size}, g.hdr->max_rows, &scratch) !=
        r3::moe_shm::v26::gpu::Refusal::None) return RAD_E_STATE;
    if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&s.sequence), scratch.sequence_bytes),
                "hipMalloc lane sequence counters") ||
        !hip_ok(hipMemset(s.sequence, 0, scratch.sequence_bytes), "zero lane sequence counters") ||
        !hip_ok(hipMalloc(reinterpret_cast<void**>(&s.published), scratch.published_bytes),
                "hipMalloc published flags") ||
        !hip_ok(hipMemset(s.published, 0, scratch.published_bytes), "zero published flags") ||
        !hip_ok(hipMalloc(&s.maps, (size_t)max_rows() * g.hdr->top_k * (sizeof(int32_t) + sizeof(uint32_t)) +
                                   sizeof(uint32_t)),
                "hipMalloc merge slot maps") ||
        !hip_ok(hipMalloc(reinterpret_cast<void**>(&s.ticket), sizeof(StepTicket)),
                "hipMalloc step ticket cell") ||
        !hip_ok(hipMemset(s.ticket, 0, sizeof(StepTicket)), "zero step ticket cell"))
        return RAD_E_DEVICE;

    const bool shadow = shadow_env();
    const bool mask_only = mask_only_env();
    const uint32_t row_cap = (uint32_t)max_rows();
    if (mask_only) {
        const size_t bytes = (size_t)row_cap * g.hdr->top_k * g.hdr->hidden * sizeof(uint16_t);
        if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&s.mask_ref), bytes),
                    "hipMalloc mask-only engine rows")) return RAD_E_NOMEM;
    }
    if (shadow &&
        (!hip_ok(hipMalloc(reinterpret_cast<void**>(&s.dev.shadow_stats),
                           sizeof(R3MoeShadowLayerStats) * kMaxLayers),
                 "hipMalloc shadow stats") ||
         !hip_ok(hipMemset(s.dev.shadow_stats, 0, sizeof(R3MoeShadowLayerStats) * kMaxLayers),
                 "zero shadow stats") ||
         !hip_ok(hipMalloc(reinterpret_cast<void**>(&s.dev.shadow_first),
                           sizeof(R3MoeShadowFirst) * kMaxLayers),
                 "hipMalloc shadow first") ||
         !hip_ok(hipMemset(s.dev.shadow_first, 0, sizeof(R3MoeShadowFirst) * kMaxLayers),
                 "zero shadow first")))
        return RAD_E_DEVICE;

    /* One stable mapped completion buffer per rank, with distinct slots for
     * all permitted concurrent parents. Must be allocated before publishing
     * s.ready so tape kernels capture its fixed device pointer. */
    const size_t consume_bytes = g.ledger.capacity() * sizeof(ConsumptionStamp);
    if (!hip_ok(hipHostMalloc(reinterpret_cast<void**>(&s.consumed_host), consume_bytes,
                              hipHostMallocMapped | hipHostMallocCoherent),
                "rank parent consumption ring") ||
        !hip_ok(hipHostGetDevicePointer(reinterpret_cast<void**>(&s.dev.consumed),
                                       s.consumed_host, 0), "map consumption ring") ||
        !hip_ok(hipMalloc(reinterpret_cast<void**>(&s.stamps), consume_bytes), "device counting ring") ||
        !hip_ok(hipMemset(s.stamps, 0, consume_bytes), "zero device counting ring") ||
        !hip_ok(hipMalloc(reinterpret_cast<void**>(&s.consumed_rec), scratch.consumed_bytes),
                "device consumed records") ||
        !hip_ok(hipMemset(s.consumed_rec, 0, scratch.consumed_bytes), "zero device consumed records") ||
        !rank_completion_ok(completion.prepare(g.ledger.capacity(), s.consumed_host)))
        return RAD_E_DEVICE;
    std::memset(s.consumed_host, 0, consume_bytes);
    s.dev.consumed_capacity = (uint32_t)g.ledger.capacity();
    s.dev.stamps = s.stamps;
    s.dev.consumed_rec = s.consumed_rec;
    s.dev.mailbox = dev_map;
    s.dev.owner[0] = s.owner[0];
    s.dev.owner[1] = s.owner[1];
    s.dev.owner_binding = s.owner_binding;
    s.dev.ticket = s.ticket;
    s.dev.sequence = s.sequence;
    s.dev.published = s.published;
    s.dev.mask_ref = s.mask_ref;
    s.dev.inv = static_cast<int32_t*>(s.maps);
    s.dev.rmask = reinterpret_cast<uint32_t*>(s.dev.inv + (size_t)row_cap * g.hdr->top_k);
    s.dev.inject_next = s.dev.rmask + (size_t)row_cap * g.hdr->top_k;   /* the word past the maps (allocated with them) */
    s.dev.max_rows = row_cap;
    s.dev.hidden = g.hdr->hidden;
    s.dev.top_k = g.hdr->top_k;
    s.dev.experts = g.hdr->expert_count;
    s.dev.rank = (uint32_t)rank;
    s.dev.world = (uint32_t)world_size;
    if (const char* hp = bc_engine_settings_snapshot().histogram_file; hp && *hp) {
        const size_t hb = (size_t)kMaxLayers * g.hdr->expert_count * sizeof(uint32_t);
        /* RHP02: pinned, mapped host memory -- no engine VRAM and no copy stream; the publish kernel adds a step's counts
         * once an expert (it is one block and the steps run in order, so the adds need no atomics over the link). */
        if (!hip_ok(hipHostMalloc(reinterpret_cast<void**>(&s.heat_host), hb, hipHostMallocMapped | hipHostMallocCoherent), "host route histogram") ||
            !hip_ok(hipHostGetDevicePointer(reinterpret_cast<void**>(&s.dev.heat), s.heat_host, 0), "map route histogram")) {
            if (s.heat_host) (void)hipHostFree(s.heat_host);
            s.heat_host = nullptr;
            s.dev.heat = nullptr;
        } else {
            std::memset(s.heat_host, 0, hb);
        }
        if (s.heat_host && !g.snapshotter.joinable()) {
            const double period = static_cast<double>(bc_engine_settings_snapshot().histogram_period_s);
            if (period >= 1.0) g.snapshotter = std::thread(snapshot_loop, std::ref(g), period);
        }
    }
    s.dev.helpers = helpers;
    s.dev.finish_threads = finish.threads;
    s.dev.trace_every = bc_engine_settings_snapshot().mailbox_trace_every;
    s.dev.layer_count = g.hdr->layer_count;
    s.dev.slots = g.hdr->slot_count;
    s.dev.lanes = g.hdr->counts.lanes;
    s.dev.ticket_begin = scratch.ticket_begin;
    s.dev.published_words = scratch.published_words;
    s.dev.shadow = shadow ? 1u : 0u;
    /* 1: add the engine's captured rows; 2: same late publish/mask/rescatter, but add the helper's rows. */
    s.dev.mask_only = !mask_only ? 0u : bc_engine_settings_snapshot().mask_only;
    s.dev.cpu_ref = cpu_ref_env() ? 1u : 0u;
    /* R3_MOE_WORKER_COMBINE=1 (default off): two to kCombineMaskBits helpers, no shadow/mask-only, and helper 0 started its combiner
     * (Header::combine.enabled, checked again by the publish kernel each step). */
    {
        s.dev.combine = (bc_engine_settings_snapshot().combine && helpers >= 2 && helpers <= kCombineMaskBits && !shadow && !mask_only) ? 1u : 0u;
        if (s.dev.combine)
            std::fprintf(stderr, "libr3moe: R3_MOE_WORKER_COMBINE on - helper 0 combines the helpers' rows on the large lane\n");
    }
    s.ready = true;
    attaching.release();
    g.bootstrap_ready_ranks.fetch_or(1u << rank, std::memory_order_release);
    *out = &s.dev;
    return RAD_OK;
        },
        [&](WaitResult waited) -> int {
            const auto* control = placement_control(g.map);
            const auto* generation = generation_header(g.map, 0);
            std::fprintf(stderr,
                "libr3moe: bootstrap rank %d seal wait refused (result=%u seq=%u filtered=0x%x ready=0x%x expected=0x%x fault=%u)\n",
                rank, uint32_t(waited), load(&control->seq),
                load(&g.hdr->bootstrap_filtered_helpers),
                generation ? load(&generation->ready_helpers) : 0u,
                control->expected_helpers, load(&g.hdr->helper_fault));
            return RAD_E_STATE;
        });
}

/* Inactive generation preparation only: no overwrite until the shared
 * admission/completion ledger has retired EVERY ticket for this slot. */
extern "C" int r3_moe_owner_generation_upload(uint64_t h, int rank, uint32_t slot,
                                               uint64_t epoch, uint64_t digest) {
    HookState* hp = live_hook(h);
    if (!hp) return RAD_E_STATE;
    HookState& g = *hp;
    if (rank < 0 || rank >= (int)g.rank.size()) return RAD_E_INVAL;
    std::lock_guard<std::mutex> lock(g.mu);
    if (!g.hdr || !g.pinned || !g.rank[rank].ready || load(&g.hdr->helper_fault) ||
        !r3::owner_generation::may_upload(g.ledger, slot)) return RAD_E_STATE;
    std::vector<uint8_t> bits;
    if (!r3::owner_generation::build_bitmap(g.map, slot, epoch, digest, &bits))
        return RAD_E_STATE;
    RankState& s = g.rank[rank];
    if (!hip_ok(hipSetDevice(s.device), "select owner upload device") ||
        !queue_owner_upload(s, slot, epoch, digest, bits)) return RAD_E_DEVICE;
    return RAD_OK;
}

extern "C" int r3_moe_owner_generation_ready(uint64_t h, int rank, uint32_t slot,
                                              uint64_t epoch, uint64_t digest) {
    HookState* hp = live_hook(h);
    if (!hp) return RAD_E_STATE;
    HookState& g = *hp;
    if (rank < 0 || rank >= (int)g.rank.size() || slot > 1 ||
        !epoch || !digest) return RAD_E_INVAL;
    std::lock_guard<std::mutex> lock(g.mu);
    const RankState& s = g.rank[rank];
    if (!s.ready || !s.owner_queued[slot] ||
        s.owner_expected[slot].epoch != epoch || s.owner_expected[slot].digest != digest)
        return RAD_E_STATE;
    if (!hip_ok(hipSetDevice(s.device), "select owner event device")) return RAD_E_DEVICE;
    const hipError_t status = hipEventQuery(s.owner_event[slot]);
    if (status == hipSuccess) {
        /* The event is not a digest proof by itself. The immutable source
         * generation must still be exactly the uploaded candidate. */
        std::vector<uint8_t> check_bits;
        if (!r3::owner_generation::build_bitmap(g.map, slot, epoch, digest, &check_bits) ||
            check_bits.size() != s.owner_bytes ||
            std::memcmp(s.owner_stage[slot] + sizeof(R3MoeOwnerBinding),
                        check_bits.data(), s.owner_bytes) != 0)
            return RAD_E_STATE;
        return RAD_OK;
    }
    if (status == hipErrorNotReady) return RAD_E_STATE;
    (void)hip_ok(status, "query owner upload event");
    return RAD_E_DEVICE;
}
