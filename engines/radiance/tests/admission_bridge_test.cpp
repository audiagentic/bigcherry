// Card-free exercise of the generated scheduler/rank bridge.
#include "bc_moe_remote_engine.h"
#include <cassert>
#include <cstdint>
#include <vector>

static uint64_t sequence = 0, cancelled = 0, host = 0;
static uint32_t epoch = 7;
static bool closed = false, fault = false;
static std::vector<uint64_t> terminal;

extern "C" int bc_moe_remote_admit(BcMoeRemoteTicket* out, int try_only, int ranks) {
    assert(ranks == 2);
    *out = {};
    if (fault) return -1;
    if (closed) { assert(try_only); return 1; }
    *out = BcMoeRemoteTicket{epoch, uint8_t(epoch % 2), {}, ++sequence};
    return 0;
}
extern "C" int bc_moe_remote_bind(BcMoeRemoteTicket t, int64_t step) {
    assert(t.ticket_id && step >= 40); // scheduler IDs are unrelated to tickets
    return 0;
}
extern "C" void bc_moe_remote_cancel(BcMoeRemoteTicket t) { cancelled = t.ticket_id; }
extern "C" int bc_moe_remote_host_done(BcMoeRemoteTicket t) { host = t.ticket_id; return 0; }
extern "C" int bc_moe_remote_rank_done(int rank, BcMoeRemoteTicket t, void* stream) {
    assert(rank >= 0 && rank < 2 && stream);
    terminal.push_back(t.ticket_id);
    return 0;
}
struct Batch { int64_t step = 40; };
struct Scheduler {
    Batch batch;
    int steps = 0, promotions = 0;
    const Batch* step() { assert(sequence); ++steps; return &batch; }
    void promote_next() { ++promotions; }
};
struct Ctx { void* stream() const { return (void*)uintptr_t(1); } };
struct Rank { int device, index; Ctx ctx; };

int main() {
    BcMoeRemoteAdmission bridge;
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
}
