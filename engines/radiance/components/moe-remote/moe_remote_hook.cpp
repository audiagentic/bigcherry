#include "moe_remote_hook.h"
#include "admission_ledger.h"
#include "admission_completion.h"
#include "owner_generation.h"
#include "worker_epoch.h"
#include "moe_helper_watch.h"
#include "moe_remote_shm.h"
#include "rad_abi.h"
#include "rad_device.h"   /* engine streams take only radiance events (rad_event_*), never raw HIP events */

#include <hip/hip_runtime.h>

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <utility>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <mutex>
#include <thread>
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
    ConsumedRecords* consumed_rec = nullptr;     /* device-memory consumed lane/combine records */
    uint32_t* sequence = nullptr;
    uint32_t* published = nullptr;
    uint16_t* mask_ref = nullptr;
    void* maps = nullptr;   /* inv + rmask, one allocation */
    uint32_t* heat_host = nullptr;   /* R3_MOE_WORKER_HISTOGRAM: the counts in pinned host memory (dev.heat maps it) */
    bool ready = false;
};

/* The engine's remote-MoE extension point (patch rad0002, abi/bc_moe_remote.h), mirrored here so libr3moe builds
 * against an unpatched radiance tree. The engine accepts the registration only if struct_size matches its own. */
struct EnginePlugin {
    uint32_t struct_size;
    void (*layer_map)(int layer, int n_expert, const int32_t* slot_of_expert);
    int (*owns_expert)(int layer, int expert);
    void (*burst)(int active);
    int (*before_pass)(int rank, StepTicket ticket, void* stream);
    int (*admit)(StepTicket* out, int try_only, int world_size);
    int (*bind)(StepTicket ticket, int64_t scheduler_step);
    int (*cancel)(StepTicket ticket);
    int (*host_done)(StepTicket ticket);
    int (*rank_done)(int rank, StepTicket ticket, void* stream);
    void (*poll)(void);
    void (*shutdown)(void);
    int (*capacity)(uint32_t parent_limit);
};
using RegisterFn = int (*)(const EnginePlugin*);

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
struct HookState {
    int fd = -1;
    void* map = MAP_FAILED;
    size_t bytes = 0;
    Header* hdr = nullptr;
    bool pinned = false;
    RegisterFn engine_register = nullptr;
    r3::admission::Ledger ledger{0}; /* configured by the engine before map replay */
    RankCompletion completion[kMaxEngineRanks];
    R3MoeDrainSnapshot closed_snapshot{};
    uint64_t closure_id = 0;
    std::atomic<uint32_t> engine_ranks{0}; /* scheduler-owned; published once before controller starts */
    /* Rank setup proves bootstrap seal/upload once. Admission tests only this
     * cached mask, never rehashes the canonical map on the hot path. */
    std::atomic<uint32_t> bootstrap_ready_ranks{0};
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
    RankState rank[kMaxEngineRanks];
    HelperWatch watch;   /* RAD20: fails the engine's helper waits when a helper process dies */
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

/* All parent tickets originate in this coordinator. Neither a scheduler
 * number nor an in-flight tape's arguments are admission authority. */
bool committed(uint32_t* epoch, uint8_t* slot) {
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

bool helper_faulted() { return g.hdr && load(&g.hdr->helper_fault); }

void poll_tickets();

int admit_ticket(StepTicket* out, int try_only, int world_size) {
    if (!out) return RAD_E_INVAL;
    *out = StepTicket{};
    if (world_size < 1 || world_size > (int)kMaxEngineRanks) return RAD_E_INVAL;
    const uint32_t ranks = (uint32_t)world_size;
    const uint32_t mask = (1u << ranks) - 1u;
    if ((g.bootstrap_ready_ranks.load(std::memory_order_acquire) & mask) != mask)
        return RAD_E_STATE;
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
    poll_tickets();
    using Result = r3::admission::Ledger::Result;
    const Result r = try_only ? g.ledger.try_admit(ranks, out)
        : g.ledger.admit(ranks, out,
                         [] { return helper_faulted(); },
                         [] { poll_tickets(); });
    if (try_only && (r == Result::Closed || r == Result::Full)) return 1;
    if (r != Result::Ok) {
        if (r == Result::Full)
            std::fprintf(stderr, "libr3moe: admission route-ring ceiling exhausted; high-water %zu, capacity %zu\n",
                         g.ledger.high_water(), g.ledger.capacity());
        g.ledger.fault();
        return RAD_E_STATE;
    }
    return RAD_OK;
}

int bind_ticket(StepTicket ticket, int64_t step) {
    return ticket.ticket_id && g.ledger.bind(ticket.ticket_id, step) ? RAD_OK : RAD_E_STATE;
}
int cancel_ticket(StepTicket ticket) {
    return !ticket.ticket_id || g.ledger.cancel(ticket.ticket_id) ? RAD_OK : RAD_E_STATE;
}
int host_done_ticket(StepTicket ticket) {
    if (!ticket.ticket_id) return RAD_OK;
    if (!g.ledger.host_complete(ticket.ticket_id)) return RAD_E_STATE;
    poll_tickets();
    return g.ledger.stopped() ? RAD_E_STATE : RAD_OK;
}

/* The setter takes the exact admitted ticket. Every pass belonging to one
 * scheduler parent (trunk, history and draft passes) receives the same value,
 * including the pass issued ahead before that parent's host commit. */
int before_pass(int rank, StepTicket value, void* stream) {
    if (rank < 0 || rank >= (int)kMaxEngineRanks || !stream ||
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
int rank_done_ticket(int rank, StepTicket ticket, void* stream) {
    if (!ticket.ticket_id) return RAD_OK;
    if (!stream || rank < 0 || rank >= (int)kMaxEngineRanks ||
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
void poll_tickets() {
    if (helper_faulted()) { g.ledger.fault(); return; }
    (void)g.ledger.poll([](const StepTicket& t, uint32_t rank) {
        return g.completion[rank].prove(t);
    });
}

bool served_lane(uint32_t lane) {
    return (g.hdr->helper_count <= 1 || lane % kMaxSlots < g.hdr->helper_count) &&
           (!lane_is_small(lane) || lane_class(lane) < g.hdr->small_class_count);
}
void hold_burst_for_drain(Header* h);

/* Admission is closed and every admitted parent has retired, so ending the burst cuts off no work. The drain words
 * are stored before the burst word moves: a helper that sees the move (burst spin ends, or its futex wait on the word
 * returns) then sees its drain request. An idle word moves by 2, so it stays even and still wakes sleepers. */
void request_helper_drain(uint64_t closure) {
    Header* h = g.hdr;
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        if (served_lane(lane))
            __atomic_store_n(&h->lane[lane].drain_request, closure, __ATOMIC_RELEASE);
    {
        std::lock_guard<std::mutex> lock(g.burst_mu);
        g.burst_draining = true;
        hold_burst_for_drain(h);
    }
    notify(h);   /* acknowledge_drain's closure wait and the engine's park waits use the generation word */
}

bool helper_drained(uint64_t closure) {
    if (!g.hdr || load(&g.hdr->helper_fault)) return false;
    const Header* h = g.hdr;
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        if (!lane_quiesced(h, lane, closure, served_lane(lane))) return false;
    if (h->combine.request &&
        (load(&h->combine.done) != load(&h->combine.request) ||
         load(&h->combine.result_consumed_generation) != load(&h->combine.request)))
        return false;
    return true;
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
    g.watch.notice();
    /* The plugin ABI gives this callback no error return. A failed helper leaves helper_fault raised (sticky), so the
     * engine's first remote MoE wait traps; the map wait just must not hang. */
    uint32_t* consumed = &g.hdr->layer_map_consumed[layer];
    const WaitResult r = wait_for(g.hdr, nullptr, [&] { return load(consumed) == 1; });
    if (r == WaitResult::Error) raise_fault(g.hdr);   /* a wait that cannot complete is terminal too */
    if (r != WaitResult::Ready)
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
bool apply_burst(Header* h, bool active) {
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
void burst_transition(int active) {
    static const bool trace = [] {
        const char* e = std::getenv("R3_MOE_WORKER_BURST_TRACE");
        return e && *e && std::strcmp(e, "0") != 0;
    }();
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
            if (!g.burst_draining) opened = apply_burst(h, g.burst_want);
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
bool resume_burst_after_drain(Header* h) {
    bool opened = false;
    {
        std::lock_guard<std::mutex> lock(g.burst_mu);
        g.burst_draining = false;
        opened = apply_burst(h, g.burst_want);
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
int owner_query(int layer, int expert) {
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

void placement_loop();

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
    /* Watch the helpers before registering: registration replays the layer maps and waits for the helpers to consume
     * them, and a helper that dies there must fail that wait. No watch, no helpers (fail closed). */
    if (!g.watch.start(g.hdr)) {
        (void)hipHostUnregister(g.map);
        g.pinned = false;
        g.hdr = nullptr;
        g.engine_register = nullptr;
        return;
    }
    /* Registration barrier: every helper has published its identity and the watcher holds a validated pidfd for it,
     * before anything waits on the helpers. A helper that died before publishing its pid has no process to watch, so
     * this bounded startup wait (blocking, not polled) is what catches it. On timeout or fault the helpers are failed
     * (sticky helper_fault): the engine still registers, its map waits return at once and its first remote step
     * fails, rather than serving with experts it was told the helpers hold. */
    {
        constexpr int kRegisterTimeoutS = 120;   /* helpers attach before the engine starts; this only bounds a failure */
        timespec deadline{};
        clock_gettime(CLOCK_MONOTONIC, &deadline);
        deadline.tv_sec += kRegisterTimeoutS;
        const WaitResult r = wait_for(g.hdr, &deadline, [] {
            g.watch.notice();
            const uint32_t n = load(&g.hdr->helper_count);
            for (uint32_t i = 0; i < n && i < kMaxHelpers; ++i)
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
    if (!committed(&epoch, &slot) || !g.ledger.initialize(epoch, slot)) {
        raise_fault(g.hdr);
        return;
    }
    static const EnginePlugin plugin{sizeof(EnginePlugin), publish_layer_map, owner_query, burst_transition,
                                     before_pass, admit_ticket, bind_ticket, cancel_ticket,
                                     host_done_ticket, rank_done_ticket, poll_tickets,
                                     [] { g.ledger.interrupt(); },
                                     [](uint32_t limit) -> int { return g.ledger.configure(limit) ? RAD_OK : RAD_E_STATE; }};
    if (g.engine_register(&plugin) != 0) {
        std::fprintf(stderr, "libr3: R3_MOE_WORKER engine rejected the plugin registration (struct size)\n");
        g.watch.stop();
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
    g.placement_thread = std::thread(placement_loop);
}

/* Engine participant of the shared helper-zero placement coordinator.
 * Same-map diagnostic cycles use TWO real generations and the complete
 * RESERVED/COPYING/PREPARED/QUIESCING/COMMIT/RETIRING path. Actual owner
 * movement and helper weight-table changes remain later slices.
 *
 * All GPU uploads/events are on private rank upload streams. This thread
 * polls events away from serving, and reuses the slice-C admission ledger,
 * parent seal/events, helper drain and reopen; there is no second gate. */
void placement_loop() {
    R3MoeDrainSnapshot snap{};
    uint64_t uploaded_transaction = 0;
    uint64_t reserved_transaction = 0;
    uint64_t closed_transaction = 0;
    uint32_t reserved_mask = 0, issued_mask = 0, ready_mask = 0, committed_mask = 0;
    while (!g.placement_stop.load(std::memory_order_acquire)) {
        PlacementControl c{};
        if (helper_faulted() || g.ledger.stopped()) break;
        if (!placement_snapshot(g.map, &c)) { std::this_thread::yield(); continue; }
        if (c.fault_code) { g.ledger.fault(); break; }
        const auto state = PlacementState(c.state);
        auto fatal = [&] {
            g.ledger.fault();
            raise_fault(g.hdr);
        };
        const uint32_t ranks = c.expected_ranks;
        auto current_cycle = [&] {
            const uint32_t actual = g.engine_ranks.load(std::memory_order_acquire);
            return actual && actual <= kMaxEngineRanks &&
                   ranks == ((1u << actual) - 1u) &&
                   r3::epoch::same_map_cycle(g.map,c);
        };
        /* Fault one or more current engine-rank READY actors without poisoning
         * serving at epoch N before the controller commits. A retry is a new
         * transaction, never an in-place rebind of its generation. */
        auto failed_upload = [&](r3::epoch::FaultCode code) {
            for (uint32_t rank = 0; rank < kMaxEngineRanks; ++rank)
                if (ranks & (1u << rank))
                    r3::epoch::publish_fault_ack(g.map,kMaxHelpers+rank,c,AckPhase::READY,code);
        };
        if (state == PlacementState::RESERVING) {
            if (c.transaction_id != reserved_transaction) {
                reserved_transaction = c.transaction_id;
                reserved_mask = 0;
            }
            if (closed_transaction || !current_cycle()) {
                for (uint32_t rank=0;rank<kMaxEngineRanks;++rank)
                    if (ranks & (1u<<rank))
                        r3::epoch::publish_fault_ack(g.map,kMaxHelpers+rank,c,
                            AckPhase::RESERVED,r3::epoch::FaultCode::CHECKSUM);
            } else {
                const uint32_t actual=g.engine_ranks.load(std::memory_order_acquire);
                const uint32_t actual_mask=(1u<<actual)-1u;
                for (uint32_t rank=0;rank<kMaxEngineRanks;++rank) {
                    const uint32_t bit=1u<<rank;
                    if (!(ranks & bit) || (reserved_mask & bit)) continue;
                    bool registered=false;
                    {
                        std::lock_guard<std::mutex> lock(g.mu);
                        registered=g.rank[rank].ready;
                    }
                    if (!registered) continue; // controller enforces deadline
                    const int ready=r3_moe_owner_generation_ready((int)rank,
                        c.active_slot,c.active_epoch,c.active_digest);
                    if (ready == RAD_E_DEVICE) {
                        r3::epoch::publish_fault_ack(g.map,kMaxHelpers+rank,c,
                            AckPhase::RESERVED,r3::epoch::FaultCode::DEVICE);
                    } else if (r3::epoch::service_rank_reservation(g.map,c,rank,
                               actual_mask,registered,ready==RAD_OK)) {
                        reserved_mask |= bit;
                    } else if (ready == RAD_E_STATE) {
                        r3::epoch::publish_fault_ack(g.map,kMaxHelpers+rank,c,
                            AckPhase::RESERVED,r3::epoch::FaultCode::CHECKSUM);
                    }
                }
            }
        } else if (state == PlacementState::COPYING) {
            if (c.transaction_id != uploaded_transaction) {
                uploaded_transaction = c.transaction_id;
                issued_mask = ready_mask = committed_mask = 0;
            }
            if (closed_transaction || !current_cycle()) {
                failed_upload(r3::epoch::FaultCode::CHECKSUM);
            } else {
                for (uint32_t rank = 0; rank < kMaxEngineRanks; ++rank) {
                    const uint32_t bit = 1u << rank;
                    if (!(ranks & bit)) continue;
                    if (!(issued_mask & bit)) {
                        const int issued = r3_moe_owner_generation_upload(
                            (int)rank,c.pending_slot,c.pending_epoch,c.pending_digest);
                        if (issued != RAD_OK) {
                            failed_upload(issued == RAD_E_DEVICE ?
                                r3::epoch::FaultCode::DEVICE : r3::epoch::FaultCode::UPLOAD);
                            break;
                        }
                        issued_mask |= bit;
                    }
                    if (!(ready_mask & bit)) {
                        /* An event that is merely pending is retryable.
                         * An upload binding for the wrong epoch/digest is an
                         * immediate CHECKSUM fault, not a timeout that might
                         * accidentally be treated as ordinary GPU latency. */
                        bool exact_binding = false;
                        {
                            std::lock_guard<std::mutex> lock(g.mu);
                            const RankState& s = g.rank[rank];
                            exact_binding = c.pending_slot < 2 &&
                                s.ready && s.owner_queued[c.pending_slot] &&
                                s.owner_expected[c.pending_slot].epoch == c.pending_epoch &&
                                s.owner_expected[c.pending_slot].digest == c.pending_digest;
                        }
                        if (!exact_binding) {
                            failed_upload(r3::epoch::FaultCode::CHECKSUM);
                            break;
                        }
                        const int ready = r3_moe_owner_generation_ready(
                            (int)rank,c.pending_slot,c.pending_epoch,c.pending_digest);
                        if (ready == RAD_OK &&
                            r3::epoch::rank_upload_prepared(c,rank,true,true,
                                c.pending_epoch,c.pending_digest,c.pending_slot)) {
                            ready_mask |= bit;
                            r3::epoch::publish_ack(g.map,kMaxHelpers+rank,c,AckPhase::READY);
                        } else if (ready == RAD_E_DEVICE) {
                            failed_upload(r3::epoch::FaultCode::DEVICE);
                            break;
                        } /* RAD_E_STATE: event pending or invalid, until
                           * deadline; controller aborts with READY_TIMEOUT. */
                    }
                }
            }
        } else if (state == PlacementState::QUIESCING && c.transaction_id != closed_transaction) {
            /* READY ACKs were sent only after the real HIP event on EVERY
             * actual rank; the Controller cannot publish PREPARED without all
             * of them. Neither missing rank nor a stale/mismatched map can
             * pass this barrier. */
            if (uploaded_transaction != c.transaction_id ||
                ready_mask != ranks || !current_cycle() ||
                r3_moe_close_admission(&snap) < 0 ||
                r3_moe_wait_drained(&snap,c.deadline_ns) < 0 ||
                r3_moe_all_ranks_consumed(&snap) != 1) {
                fatal(); break;
            }
            closed_transaction = c.transaction_id;
            for (uint32_t rank=0; rank<kMaxEngineRanks; ++rank)
                if (ranks & (1u<<rank))
                    r3::epoch::publish_ack(g.map,kMaxHelpers+rank,c,
                                          AckPhase::QUIESCED,snap.cutoff_ticket_id);
        } else if (state == PlacementState::COMMIT && closed_transaction == c.transaction_id &&
                   closed_transaction) {
            if (!current_cycle() || snap.old_epoch + 1u != c.active_epoch ||
                c.active_slot != 1u - placement_journal(g.map)->entry[c.journal_index].old_slot ||
                ready_mask != ranks) {
                fatal(); break;
            }
            /* The controller's COMMIT seqlock is already published, but the
             * helper participant marks GenerationHeader ACTIVE on its next
             * poll. Never acknowledge until that exact header is active. */
            const auto* new_gen = generation_header(g.map,c.active_slot);
            if (new_gen->state == uint32_t(GenerationState::ACTIVE)) {
                for (uint32_t rank=0; rank<kMaxEngineRanks; ++rank)
                    if (ranks & (1u<<rank)) {
                        r3::epoch::publish_ack(g.map,kMaxHelpers+rank,c,
                                              AckPhase::COMMITTED,snap.cutoff_ticket_id);
                        committed_mask |= 1u<<rank;
                    }
            }
        } else if (state == PlacementState::RETIRING && closed_transaction == c.transaction_id &&
                   closed_transaction) {
            const uint32_t old_slot = 1u-c.active_slot;
            const bool all_consumed = r3_moe_all_ranks_consumed(&snap) == 1;
            bool drained = false;
            {
                std::lock_guard<std::mutex> lock(g.mu);
                drained = g.drain_verified && g.closed_snapshot.closure_id == snap.closure_id;
            }
            const bool quiescent = drained && helper_drained(snap.closure_id);
            const bool no_old_refs = !g.ledger.generation_referenced(old_slot);
            if (!current_cycle() ||
                !r3::epoch::rank_retirement_proved(c,snap.old_epoch,old_slot,
                       all_consumed,drained,quiescent,no_old_refs,
                       ready_mask==ranks,committed_mask==ranks)) {
                /* Never ACK a missing result-consumed proof or helper drain.
                 * If still pending, leave the controller to time out and
                 * fault without releasing or overwriting the old bitmap. */
                if (!current_cycle() || g.ledger.stopped() || helper_faulted()) {
                    fatal(); break;
                }
            } else {
                /* Before reopen the ledger's active slot is still old_slot:
                 * its Part-1 may_upload guard MUST reject overwrite here. */
                if (g.ledger.can_replace_generation(old_slot)) { fatal(); break; }
                for (uint32_t rank=0; rank<kMaxEngineRanks; ++rank)
                    if (ranks & (1u<<rank))
                        r3::epoch::publish_ack(g.map,kMaxHelpers+rank,c,
                                              AckPhase::RETIRED,snap.cutoff_ticket_id);
            }
        } else if (state == PlacementState::ACTIVE && closed_transaction && snap.closure_id) {
            if (c.active_slot > 1) { fatal(); break; }
            const uint32_t old_slot=1u-c.active_slot;
            const auto* retired=generation_header(g.map,old_slot);
            /* The helper controller marks the old header RETIRED only after
             * EVERY rank and helper has ACKed RETIRED. This also ensures it
             * cannot be overwritten until the full close/drain barrier. */
            if (retired->state == uint32_t(GenerationState::RETIRED)) {
                bool owner_events_ready = true;
                for (uint32_t rank=0; rank<kMaxEngineRanks; ++rank)
                    if ((ranks & (1u<<rank)) &&
                        r3_moe_owner_generation_ready((int)rank,c.active_slot,
                             c.active_epoch,c.active_digest) != RAD_OK)
                        owner_events_ready = false;
                if (c.transaction_id != closed_transaction || !c.active_epoch ||
                    c.active_epoch != (uint64_t)snap.old_epoch+1u ||
                    ready_mask != ranks || committed_mask != ranks ||
                    !owner_events_ready ||
                    r3_moe_all_ranks_consumed(&snap) != 1 ||
                    r3_moe_reopen_admission((uint32_t)c.active_epoch,c.active_slot) < 0 ||
                    !g.ledger.can_replace_generation(old_slot)) {
                    fatal(); break;
                }
                snap = {};
                std::fprintf(stderr,
                    "libr3moe: placement cycle %llu reopened epoch %llu slot %u\n",
                    (unsigned long long)closed_transaction,
                    (unsigned long long)c.active_epoch,c.active_slot);
                closed_transaction = 0;
            }
        } else if (state == PlacementState::ACTIVE && !closed_transaction &&
                   uploaded_transaction && uploaded_transaction == c.transaction_id) {
            /* A precommit abort leaves ACTIVE(N) and existing owner pointers
             * unchanged. The next diagnostic proposal receives a fresh
             * transaction ID and re-uploads the alternate slot. */
            issued_mask = ready_mask = committed_mask = 0;
            uploaded_transaction = 0;
        }
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
void write_heat(uint32_t rank, RankState& s, bool final) {
    const char* path = std::getenv("R3_MOE_WORKER_HISTOGRAM");
    if (!path || !*path || !s.heat_host) return;
    /* The counts live in pinned host memory (RHP02): a snapshot is a CPU copy, with no HIP stream or queue of its own. */
    if (final) (void)hip_ok(hipDeviceSynchronize(), "sync before route histogram");
    std::vector<uint32_t> heat((size_t)kMaxLayers * g.hdr->expert_count);
    for (size_t i = 0; i < heat.size(); ++i) heat[i] = __atomic_load_n(&s.heat_host[i], __ATOMIC_RELAXED);
    const bool stream = !final;
    const std::string out = std::string(path) + ".rank" + std::to_string(rank);
    const std::string tmp = out + ".tmp";
    if (FILE* f = std::fopen(tmp.c_str(), "wb")) {
        const size_t n = std::fwrite(heat.data(), sizeof(uint32_t), heat.size(), f);
        const bool ok = std::fclose(f) == 0 && n == heat.size() && std::rename(tmp.c_str(), out.c_str()) == 0;
        if (!stream) std::fprintf(stderr, "libr3: R3_MOE_WORKER route histogram %s (%zu counts)%s\n", out.c_str(), n, ok ? "" : " NOT written");
    }
}

/* The live snapshot thread: every R3_MOE_WORKER_HISTOGRAM_PERIOD_S seconds, each ready rank's histogram. It takes g.mu per snapshot, so
 * it never reads a rank that close_mailbox is tearing down (close_mailbox stops and joins it first, without holding g.mu). */
void snapshot_loop(double period_s) {
    auto next = std::chrono::steady_clock::now() + std::chrono::duration<double>(period_s);
    while (!g.snapshot_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (std::chrono::steady_clock::now() < next) continue;
        next += std::chrono::duration<double>(period_s);
        std::lock_guard<std::mutex> lock(g.mu);
        for (uint32_t r = 0; r < kMaxEngineRanks; ++r) {
            RankState& s = g.rank[r];
            if (!s.ready || !s.heat_host) continue;
            write_heat(r, s, false);
        }
    }
}

void close_mailbox() {
    g.placement_stop.store(true, std::memory_order_release);
    g.ledger.interrupt(); /* wakes quiescence and blocked scheduler admission */
    if (g.placement_thread.joinable()) g.placement_thread.join();
    g.snapshot_stop = true;
    if (g.snapshotter.joinable()) g.snapshotter.join();
    std::lock_guard<std::mutex> lock(g.mu);
    if (g.engine_register)
        std::fprintf(stderr, "libr3moe: admission high-water %zu, capacity %zu, outstanding %zu\n",
                     g.ledger.high_water(), g.ledger.capacity(), g.ledger.outstanding());
    g.watch.stop();   /* first: helpers may exit while the engine tears down, and that is not a fault */
    for (RankCompletion& c : g.completion) c.release();   /* after interrupt(): no poll can query them */
    for (uint32_t r = 0; r < kMaxEngineRanks; ++r) {
        RankState& s = g.rank[r];
        if (!s.ready) continue;
        if (s.device >= 0) (void)hipSetDevice(s.device);
        report_shadow(r, s);
        write_heat(r, s, true);
        if (s.heat_host) (void)hipHostFree(s.heat_host);
        s.heat_host = nullptr;
        if (s.dev.shadow_first) (void)hipFree(s.dev.shadow_first);
        if (s.dev.shadow_stats) (void)hipFree(s.dev.shadow_stats);
        if (s.mask_ref) (void)hipFree(s.mask_ref);
        if (s.maps) (void)hipFree(s.maps);
        if (s.published) (void)hipFree(s.published);
        if (s.sequence) (void)hipFree(s.sequence);
        free_owner_generations(s);
        if (s.ticket) (void)hipFree(s.ticket);
        if (s.consumed_host) (void)hipHostFree(s.consumed_host);
        if (s.stamps) (void)hipFree(s.stamps);
        if (s.consumed_rec) (void)hipFree(s.consumed_rec);
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

extern "C" int r3_moe_close_admission(R3MoeDrainSnapshot* out) {
    if (!out || !g.hdr || !g.pinned || helper_faulted()) return RAD_E_STATE;
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
extern "C" int r3_moe_all_ranks_consumed(const R3MoeDrainSnapshot* snap) {
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
        g.ledger.stopped() || helper_faulted())
        return RAD_E_STATE;
    poll_tickets();
    if (g.ledger.stopped() || helper_faulted()) return RAD_E_STATE;
    return g.ledger.all_ranks_consumed(snap->old_epoch, snap->cutoff_ticket_id) ? 1 : 0;
}

extern "C" int r3_moe_wait_drained(const R3MoeDrainSnapshot* snap,
                                    uint64_t deadline_monotonic_ns) {
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
                                                 limit, [] { poll_tickets(); },
                                                 [] { return helper_faulted(); });
    if (drained != Result::Ok) { g.ledger.fault(); return RAD_E_STATE; }
    /* A successful event query guarantees helper wait and consumption before
     * the event, but explicitly check every lane before controller COMMIT. */
    {
        std::lock_guard<std::mutex> lock(g.mu);
        if (snap->closure_id != g.closed_snapshot.closure_id) return RAD_E_STATE;
        request_helper_drain(snap->closure_id);
    }
    while (!helper_drained(snap->closure_id)) {
        if (g.ledger.stopped() || helper_faulted() || std::chrono::steady_clock::now() >= limit) {
            g.ledger.fault();
            return RAD_E_STATE;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    {
        std::lock_guard<std::mutex> lock(g.mu);
        if (g.ledger.stopped() || helper_faulted() ||
            snap->closure_id != g.closed_snapshot.closure_id ||
            !g.ledger.drained(snap->old_epoch, snap->cutoff_ticket_id))
            return RAD_E_STATE;
        g.drain_verified = true;
    }
    return RAD_OK;
}

extern "C" int r3_moe_reopen_admission(uint32_t epoch, uint32_t slot) {
    uint32_t committed_epoch = 0;
    uint8_t committed_slot = 0;
    std::lock_guard<std::mutex> lock(g.mu);
    if (!g.drain_verified ||
        !committed(&committed_epoch, &committed_slot) ||
        committed_epoch != epoch || committed_slot != slot)
        return RAD_E_STATE;
    /* Helpers leave their closure wait, then the burst word returns to radiance's state (with its park wait), and
     * only then may the scheduler publish: the ledger reopens last. */
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        __atomic_store_n(&g.hdr->lane[lane].drain_request, uint64_t(0), __ATOMIC_RELEASE);
    notify(g.hdr);
    /* Fail closed: a park wait that did not see every lane return, or a helper fault since the commit, keeps
     * admission shut. */
    if (!resume_burst_after_drain(g.hdr) || helper_faulted()) {
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
    r3::epoch::publish_reopen_ack(g.map,c);
    return RAD_OK;
}

extern "C" int r3_moe_mailbox_enabled(void) {
    return g.hdr && g.pinned;
}

extern "C" int r3_moe_mailbox_device(int rank, int world_size, R3MoeMailboxDevice** out) {
    if (!out || !g.hdr || !g.pinned || world_size < 1 || world_size > (int)kMaxEngineRanks ||
        rank < 0 || rank >= world_size)
        return RAD_E_INVAL;

    /* Radiance declares the kernel instances while the helpers can still
     * be loading/filtering runtime maps. This wait must be BEFORE g.mu:
     * publish_layer_map needs that mutex to unblock the helpers, which in
     * turn let helper 0 seal the canonical generation. */
    if (!(g.bootstrap_ready_ranks.load(std::memory_order_acquire) & (1u << rank))) {
        timespec deadline{};
        if (clock_gettime(CLOCK_MONOTONIC, &deadline) != 0) return RAD_E_STATE;
        deadline.tv_sec += kStartupWaitSeconds;
        const WaitResult waited = r3::owner_generation::wait_bootstrap_seal(g.map, &deadline);
        if (waited != WaitResult::Ready) {
            const auto* control = placement_control(g.map);
            const auto* generation = generation_header(g.map, 0);
            std::fprintf(stderr,
                "libr3moe: bootstrap rank %d seal wait refused (result=%u seq=%u filtered=0x%x ready=0x%x expected=0x%x fault=%u)\n",
                rank, uint32_t(waited), load(&control->seq),
                load(&g.hdr->bootstrap_filtered_helpers),
                generation ? load(&generation->ready_helpers) : 0u,
                control->expected_helpers, load(&g.hdr->helper_fault));
            return RAD_E_STATE;
        }
    }

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
    if (helpers < 1 || helpers > kMaxHelpers || (helpers > 1 && world_size != 1)) {
        std::fprintf(stderr, "libr3: R3_MOE_WORKER %u helpers need a one-rank engine (world %d)\n",
                     helpers, world_size);
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
    if (!allocate_owner_generations(s, owner_bitmap.size()) ||
        !queue_owner_upload(s, active_slot, sealed.active_epoch,
                            digest, owner_bitmap) ||
        !hip_ok(hipEventSynchronize(s.owner_event[active_slot]),
                "finish bootstrap owner upload")) {
        free_owner_generations(s);
        return RAD_E_DEVICE;
    }
    if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&s.sequence), kLanes * sizeof(uint32_t)),
                "hipMalloc lane sequence counters") ||
        !hip_ok(hipMemset(s.sequence, 0, kLanes * sizeof(uint32_t)), "zero lane sequence counters") ||
        !hip_ok(hipMalloc(reinterpret_cast<void**>(&s.published), (kMaxSlots + 1 + kCombineChunks + 4) * sizeof(uint32_t)),
                "hipMalloc published flags") ||
        !hip_ok(hipMemset(s.published, 0, (kMaxSlots + 1 + kCombineChunks + 4) * sizeof(uint32_t)), "zero published flags") ||
        !hip_ok(hipMalloc(&s.maps, (size_t)max_rows() * g.hdr->top_k * (sizeof(int32_t) + sizeof(uint32_t))),
                "hipMalloc merge slot maps") ||
        !hip_ok(hipMalloc(reinterpret_cast<void**>(&s.ticket), sizeof(StepTicket)),
                "hipMalloc step ticket cell") ||
        !hip_ok(hipMemset(s.ticket, 0, sizeof(StepTicket)), "zero step ticket cell")) {
        if (s.ticket) (void)hipFree(s.ticket);
        s.ticket = nullptr;
        if (s.maps) (void)hipFree(s.maps);
        if (s.published) (void)hipFree(s.published);
        if (s.sequence) (void)hipFree(s.sequence);
        free_owner_generations(s);
        s.maps = nullptr;
        s.published = nullptr;
        s.sequence = nullptr;
        return RAD_E_DEVICE;
    }

    const bool shadow = shadow_env();
    const bool mask_only = mask_only_env();
    const uint32_t row_cap = (uint32_t)max_rows();
    if (mask_only) {
        const size_t bytes = (size_t)row_cap * g.hdr->top_k * g.hdr->hidden * sizeof(uint16_t);
        if (!hip_ok(hipMalloc(reinterpret_cast<void**>(&s.mask_ref), bytes),
                    "hipMalloc mask-only engine rows")) {
            (void)hipFree(s.ticket);
            s.ticket = nullptr;
            (void)hipFree(s.maps);
            (void)hipFree(s.published);
            (void)hipFree(s.sequence);
            free_owner_generations(s);
            s.maps = nullptr;
            s.published = nullptr;
            s.sequence = nullptr;
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
        (void)hipFree(s.ticket);
        s.ticket = nullptr;
        (void)hipFree(s.published);
        (void)hipFree(s.sequence);
        free_owner_generations(s);
        s.published = nullptr;
        s.sequence = nullptr;
        return RAD_E_DEVICE;
    }

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
        !hip_ok(hipMalloc(reinterpret_cast<void**>(&s.consumed_rec), sizeof(ConsumedRecords)),
                "device consumed records") ||
        !hip_ok(hipMemset(s.consumed_rec, 0, sizeof(ConsumedRecords)), "zero device consumed records") ||
        !rank_completion_ok(g.completion[rank].prepare(g.ledger.capacity(), s.consumed_host))) {
        if (s.consumed_host) (void)hipHostFree(s.consumed_host);
        if (s.stamps) (void)hipFree(s.stamps);
        if (s.consumed_rec) (void)hipFree(s.consumed_rec);
        if (shadow_first) (void)hipFree(shadow_first);
        if (shadow_stats) (void)hipFree(shadow_stats);
        if (s.mask_ref) (void)hipFree(s.mask_ref);
        if (s.ticket) (void)hipFree(s.ticket);
        if (s.maps) (void)hipFree(s.maps);
        if (s.published) (void)hipFree(s.published);
        if (s.sequence) (void)hipFree(s.sequence);
        free_owner_generations(s);
        s = RankState{};
        return RAD_E_DEVICE;
    }
    std::memset(s.consumed_host, 0, consume_bytes);
    s.dev.consumed_capacity = (uint32_t)g.ledger.capacity();
    s.dev.stamps = s.stamps;
    s.dev.consumed_rec = s.consumed_rec;
    s.device = device;
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
            const char* ps = std::getenv("R3_MOE_WORKER_HISTOGRAM_PERIOD_S");
            const double period = ps && *ps ? std::atof(ps) : 0.0;
            if (period >= 1.0) g.snapshotter = std::thread(snapshot_loop, period);
        }
    }
    s.dev.helpers = helpers;
    s.dev.shadow = shadow ? 1u : 0u;
    /* 1: add the engine's captured rows; 2: same late publish/mask/rescatter, but add the helper's rows. */
    {
        const char* e = std::getenv("R3_MOE_WORKER_MASK_ONLY");
        s.dev.mask_only = !mask_only ? 0u : (e && std::strcmp(e, "2") == 0 ? 2u : 1u);
    }
    s.dev.cpu_ref = cpu_ref_env() ? 1u : 0u;
    /* R3_MOE_WORKER_COMBINE=1 (default off): two to kCombineHelpers helpers, no shadow/mask-only, and helper 0 started its combiner
     * (Header::combine.enabled, checked again by the publish kernel each step). */
    {
        const char* e = std::getenv("R3_MOE_WORKER_COMBINE");
        s.dev.combine = (e && std::strcmp(e, "1") == 0 && helpers >= 2 && helpers <= kCombineHelpers && !shadow && !mask_only) ? 1u : 0u;
        if (s.dev.combine)
            std::fprintf(stderr, "libr3moe: R3_MOE_WORKER_COMBINE on - helper 0 combines the helpers' rows on the large lane\n");
    }
    s.ready = true;
    g.bootstrap_ready_ranks.fetch_or(1u << rank, std::memory_order_release);
    *out = &s.dev;
    return RAD_OK;
}

/* Inactive generation preparation only: no overwrite until the shared
 * admission/completion ledger has retired EVERY ticket for this slot. */
extern "C" int r3_moe_owner_generation_upload(int rank, uint32_t slot,
                                               uint64_t epoch, uint64_t digest) {
    if (rank < 0 || rank >= (int)r3::moe_shm::kMaxEngineRanks) return RAD_E_INVAL;
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

extern "C" int r3_moe_owner_generation_ready(int rank, uint32_t slot,
                                              uint64_t epoch, uint64_t digest) {
    if (rank < 0 || rank >= (int)r3::moe_shm::kMaxEngineRanks || slot > 1 ||
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
