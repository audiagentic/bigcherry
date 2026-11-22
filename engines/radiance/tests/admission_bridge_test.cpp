// Card-free exercise of the generated scheduler/rank bridge.
#include "bc_moe_remote_engine.h"
#include <cassert>
#include <cstdint>
#include <vector>

static uint64_t sequence = 0, cancelled = 0, host = 0;
static uint32_t epoch = 7;
static bool closed = false, finishing = false, fault = false, reject_cancel = false;
static std::vector<uint64_t> terminal;

extern "C" int bc_serving_remote_settings(int64_t* rows, uint32_t* skip, uint32_t* shadow, uint32_t* mask) {
    *rows = 4; *skip = *shadow = *mask = 0;
    return 0;
}
static int plugin_admit(void* self, BcMoeRemoteTicket* out, int try_only, int ranks) {
    assert(self == &sequence && ranks == 2);
    *out = {};
    if (fault) return -1;
    if (finishing) return 1;
    if (closed) { assert(try_only); return 1; }
    *out = BcMoeRemoteTicket{epoch, uint8_t(epoch % 2), {}, ++sequence};
    return 0;
}
static void plugin_finish_admission(void*) { finishing = true; }
static int plugin_bind(void*, BcMoeRemoteTicket t, int64_t step) {
    assert(t.ticket_id && step >= 40); // scheduler IDs are unrelated to tickets
    return 0;
}
static int plugin_cancel(void*, BcMoeRemoteTicket t) {
    if (reject_cancel) return -1;
    cancelled = t.ticket_id;
    return 0;
}
static int plugin_host_done(void*, BcMoeRemoteTicket t) { host = t.ticket_id; return 0; }
static int plugin_rank_done(void*, int rank, BcMoeRemoteTicket t, void* stream) {
    assert(rank >= 0 && rank < 2 && stream);
    terminal.push_back(t.ticket_id);
    return 0;
}
static int plugin_closed = 0;
static void plugin_close(void*) { ++plugin_closed; }
static int plugin_open(const BcMoeRemoteOpenArgs* a, BcMoeRemotePlugin* out) {
    assert(a->ranks == 2 && a->slot_base == 6 && a->capacity_limit == 9);
    BcMoeRemotePlugin p{};
    p.struct_size = sizeof p;
    p.self = &sequence;
    p.admit = plugin_admit; p.bind = plugin_bind; p.cancel = plugin_cancel; p.host_done = plugin_host_done;
    p.rank_done = plugin_rank_done; p.finish_admission = plugin_finish_admission; p.close = plugin_close;
    *out = p;
    return 0;
}
struct Batch { int64_t step = 40; };
struct Scheduler {
    Batch batch;
    int steps = 0, promotions = 0;
    bool empty = false;
    const Batch* step() { assert(sequence); ++steps; return empty ? nullptr : &batch; }
    void promote_next() { ++promotions; }
};
struct Ctx { void* stream() const { return (void*)uintptr_t(1); } };
struct Rank { int device, index; Ctx ctx; };

int main() {
    BcMoeRemoteAdmission bridge;
    assert(bridge.open_with(plugin_open, 6, 2, 9) == 0 && bridge.generation() && bc_moe_remote_lookup(6) == bridge.generation());
    Scheduler scheduler;
    const Batch* b = nullptr;
    assert(bridge.step(false, b, scheduler, 2) == 0);
    assert(b && scheduler.steps == 1 && bridge.issued_ticket().ticket_id == 1);
    assert(bridge.issued_ticket().placement_epoch == 7);
    Rank r0{0,0,{}}, r1{1,1,{}};
    std::vector<Rank*> ranks{&r0,&r1};
    assert(bridge.seal(ranks) == 0);
    assert((terminal == std::vector<uint64_t>{1,1}));

    assert(bridge.try_ahead(2));
    Batch next{41};
    assert(bridge.bind_ahead(&next) == 0);
    assert(bridge.issued_ticket().ticket_id == 2);
    assert(bridge.host_done() == 0 && host == 1); // never complete ahead here
    b = &next;
    assert(bridge.step(true, b, scheduler, 2) == 0);
    assert(scheduler.promotions == 1 && scheduler.steps == 1);
    assert(bridge.issued_ticket().ticket_id == 2);

    closed = true;
    assert(!bridge.try_ahead(2) && bridge.ahead_status() == 0);
    assert(sequence == 2); // no scheduler mutation or reservation while closed
    assert(bridge.seal(ranks) == 0);
    assert(bridge.host_done() == 0 && host == 2); // current can still retire
    closed = false;
    epoch = 8;
    assert(bridge.try_ahead(2));
    assert(bridge.bind_ahead<Batch>(nullptr) == 0 && cancelled == 3);
    assert(bridge.step(false, b, scheduler, 2) == 0);
    assert(bridge.issued_ticket().placement_epoch == 8);
    assert(bridge.issued_ticket().ticket_id == 4);
    fault = true;
    assert(!bridge.try_ahead(2) && bridge.ahead_status() < 0);
    assert(bridge.step(false, b, scheduler, 2) < 0 && scheduler.steps == 2);

    // A rejected cancellation fails closed on both paths that return a reservation.
    fault = false;
    reject_cancel = true;
    assert(bridge.try_ahead(2));
    assert(bridge.bind_ahead<Batch>(nullptr) < 0);
    scheduler.empty = true;
    assert(bridge.step(false, b, scheduler, 2) < 0);
    reject_cancel = false;
    assert(bridge.step(false, b, scheduler, 2) == 0 && bridge.issued_ticket().ticket_id == 0); // accepted cancel

    scheduler.empty = false;
    assert(bridge.step(false, b, scheduler, 2) == 0 && bridge.try_ahead(2));
    assert(bridge.bind_ahead(&next) == 0);
    const auto issued = bridge.issued_ticket().ticket_id;
    const auto steps_before_stop = scheduler.steps;
    assert(bridge.finish() && bridge.finishing());
    assert(bridge.seal(ranks) == 0 && bridge.host_done() == 0 && host < issued);
    b = &next;
    assert(bridge.step(true, b, scheduler, 2) == 0 && bridge.issued_ticket().ticket_id == issued);
    assert(bridge.seal(ranks) == 0 && bridge.host_done() == 0 && host == issued);
    assert(bridge.step(false, b, scheduler, 2) > 0 && scheduler.steps == steps_before_stop);

    // Closing the bridge closes the plugin once, unbinds the engine's slots, and is safe to repeat.
    bridge.close();
    assert(plugin_closed == 1 && !bridge.generation() && !bc_moe_remote_lookup(6) && !bc_moe_remote_lookup(7));
    bridge.close();
    assert(plugin_closed == 1);
    // A bridge without a generation (no hand-off): admission hands out zero tickets and every step runs locally.
    BcMoeRemoteAdmission local;
    const Batch* lb = nullptr;
    Scheduler ls;
    assert(local.step(false, lb, ls, 2) == 0 && lb && local.issued_ticket().ticket_id == 0);
    assert(local.seal(ranks) == 0 && local.host_done() == 0 && !local.finish());
}
