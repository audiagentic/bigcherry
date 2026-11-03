#include "admission_ledger.h"
#include "admission_completion.h"
#include "consumption_ring.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <thread>
using r3::admission::Ledger;
using R = Ledger::Result;
static void check(bool b, const char* why) {
    if (!b) { std::fprintf(stderr,"FAIL: %s\n",why); std::abort(); }
}
/* Stub event API: an event is an index into a status table. */
struct StubEvents {
    using Event = int*;
    static int status[64];
    static int created, destroyed, records, queries;
    static int create(Event* e) { if (created >= 64) return -1; *e = &status[created++]; **e = 1; return 0; }
    static void destroy(Event) { ++destroyed; }
    static Event last;
    static int record(Event e, void*) { ++records; *e = 0; last = e; return 0; }
    static int query(Event e) { ++queries; return *e; }
};
int StubEvents::status[64];
StubEvents::Event StubEvents::last = nullptr;
int StubEvents::created = 0, StubEvents::destroyed = 0, StubEvents::records = 0, StubEvents::queries = 0;
using Completion = r3::admission::RankCompletion<StubEvents>;

/* One rank's GPU, in stream order, with the kernels' counting steps:
 * set_ticket arms the device slot (consumption::arm), each publish counts into
 * the pass's own slot and each finish consumes one, and the seal copies the
 * device slot to the host-mapped ring (fields, then ticket_id). */
struct RankGpu {
    static constexpr uint32_t cap = 3;
    ConsumptionStamp dev[cap]{};
    ConsumptionStamp host[cap]{};
    StepTicket cell{};
    unsigned unconsumed = 0;                   /* published requests awaiting their finish */
    void set_ticket(const StepTicket& t) { cell = t; r3::consumption::arm(dev, cap, t); }
    void publish() {
        ConsumptionStamp& s = dev[cell.ticket_id % cap];
        check(r3::consumption::owns(s, cell.ticket_id, cell.placement_epoch, cell.generation_slot),
              "publish counts only into its own parent's slot");
        ++s.published_count; ++unconsumed;
    }
    void finish() {
        ConsumptionStamp& s = dev[cell.ticket_id % cap];
        check(unconsumed && r3::consumption::owns(s, cell.ticket_id, cell.placement_epoch, cell.generation_slot),
              "finish consumes only its own parent's request");
        ++s.consumed_count; --unconsumed;
    }
    /* A pass: one request a layer, each consumed by that layer's add. */
    void pass(const StepTicket& t, unsigned layers) {
        set_ticket(t);
        for (unsigned l = 0; l < layers; ++l) { publish(); finish(); }
    }
    void seal(uint64_t id) {
        const ConsumptionStamp s = dev[id % cap];
        check(s.consumed_count <= s.published_count, "seal: consumed never exceeds published");
        ConsumptionStamp& o = host[id % cap];
        o.placement_epoch = s.placement_epoch; o.generation_slot = s.generation_slot;
        o.published_count = s.published_count; o.consumed_count = s.consumed_count;
        o.ticket_id = s.ticket_id;
    }
};

/* Ledger::poll with per-rank pooled events and sealed consumption stamps:
 * two ranks, MTP passes sharing their parent's slot, a run-ahead parent in
 * flight beside an unretired one, and slot reuse across the ring. */
static void completion_ring() {
    using S = StubEvents;
    constexpr uint32_t cap = RankGpu::cap;
    constexpr unsigned layers = 4, mtp = 3;     /* trunk + 3 draft passes a parent */
    RankGpu gpu[2];
    Completion rank[2];
    Ledger q(cap);
    check(q.initialize(5,1), "completion fixture");
    check(rank[0].prepare(cap, gpu[0].host) && rank[1].prepare(cap, gpu[1].host) && S::created == 2*(int)cap,
          "one event per ring slot per rank, made once");
    check(!rank[0].prepare(cap, gpu[0].host), "pool is prepared once");
    void* stream = &gpu;
    std::map<uint64_t, int*> ev[2];
    auto prove = [&](const StepTicket& t, uint32_t r) { return rank[r].prove(t); };
    /* The engine's per-parent sequence on one rank: before_pass + pass per pass. */
    auto passes = [&](const StepTicket& t, uint32_t r, unsigned n) {
        for (unsigned p = 0; p < n; ++p) { check(rank[r].arm(t), "arm every pass"); gpu[r].pass(t, layers); }
    };
    /* seal(): the seal kernel then the terminal event on the same stream. */
    auto seal = [&](const StepTicket& t, uint32_t r) {
        gpu[r].seal(t.ticket_id);
        check(rank[r].record(t.ticket_id, stream), "terminal record");
        ev[r][t.ticket_id] = S::last;
        check(!rank[r].record(t.ticket_id, stream), "one terminal record per parent and rank");
    };
    auto reach = [&](const StepTicket& t, uint32_t r) { *ev[r][t.ticket_id] = 1; };   /* the GPU passes the event */
    StepTicket a{}, b{};
    check(q.admit(2,&a)==R::Ok && q.bind(a.ticket_id,1), "parent a");
    for (uint32_t r = 0; r < 2; ++r) { passes(a, r, 1 + mtp); seal(a, r); }
    for (uint32_t r = 0; r < 2; ++r)
        check(gpu[r].host[a.ticket_id%cap].published_count == (1 + mtp) * layers &&
              gpu[r].host[a.ticket_id%cap].consumed_count == (1 + mtp) * layers,
              "MTP passes count into their parent's one slot, exactly");
    /* Run-ahead: b's first pass is issued behind a's seal, before a retires. */
    check(q.try_admit(2,&b)==R::Ok && q.bind(b.ticket_id,2), "ahead parent b");
    for (uint32_t r = 0; r < 2; ++r) passes(b, r, 1);
    check(gpu[0].host[a.ticket_id%cap].published_count == (1 + mtp) * layers, "b never touches a's sealed stamp");
    const int before = S::queries;
    check(q.poll(prove) && S::queries==before, "nothing is queried before host completion");
    check(q.host_complete(a.ticket_id), "host complete a");
    check(q.poll(prove) && S::queries==before+2 && q.outstanding()==2, "pending events keep a");
    for (uint32_t r = 0; r < 2; ++r) { passes(b, r, mtp); seal(b, r); }
    check(q.host_complete(b.ticket_id), "host complete b");
    reach(b,0); reach(b,1);
    check(q.poll(prove) && q.outstanding()==2, "younger parent cannot retire first");
    reach(a,1);
    check(q.poll(prove) && q.outstanding()==2, "every rank's event is needed");
    reach(a,0);
    check(q.poll(prove) && q.outstanding()==0, "in-order retirement on all three predicates");
    /* Steady state: parents cycle through every slot, re-armed each time. */
    for (int i = 0; i < 50; ++i) {
        StepTicket t{};
        check(q.admit(2,&t)==R::Ok && q.bind(t.ticket_id,10+i), "steady admit");
        for (uint32_t r = 0; r < 2; ++r) { passes(t, r, 1 + (unsigned)(i % 3)); seal(t, r); }
        check(q.host_complete(t.ticket_id), "steady host");
        reach(t,0); reach(t,1);
        check(q.poll(prove) && q.outstanding()==0, "steady retire");
    }
    check(S::created == 2*(int)cap, "no event created in the steady state");
    StepTicket c{};
    check(q.admit(2,&c)==R::Ok && q.bind(c.ticket_id,99), "unarmed parent");
    check(rank[0].record(c.ticket_id, stream), "recorded");
    *S::last = 1;
    check(rank[1].record(c.ticket_id, stream), "recorded");
    check(q.host_complete(c.ticket_id) && !q.poll(prove) && q.stopped(), "complete event for an unarmed rank faults closed");

    /* A request still unconsumed at the seal: once its event completes, the
     * sealed counts differ and the ledger faults closed. */
    Ledger u(cap);
    RankGpu g1;
    Completion r1;
    check(u.initialize(7,0) && r1.prepare(cap, g1.host) && u.admit(1,&c)==R::Ok && u.bind(c.ticket_id,1), "unconsumed fixture");
    check(r1.arm(c), "arm");
    g1.set_ticket(c); g1.publish(); g1.publish(); g1.finish();
    g1.seal(c.ticket_id);
    check(r1.record(c.ticket_id, stream), "record");
    *S::last = 1;
    auto prove1 = [&](const StepTicket& t, uint32_t) { return r1.prove(t); };
    check(u.host_complete(c.ticket_id) && !u.poll(prove1) && u.stopped() && u.outstanding()==1,
          "a published request without its finish faults once the event completes");
    /* A stamp sealed for an older parent of the same slot does not prove this one. */
    Ledger o(cap);
    RankGpu g2;
    Completion r2;
    StepTicket d{};
    check(o.initialize(8,0) && r2.prepare(cap, g2.host), "stale fixture");
    for (int i = 0; i < 4; ++i) {   /* id 4 reuses id 1's slot */
        check(o.admit(1,&d)==R::Ok && o.bind(d.ticket_id,i), "stale admit");
        check(r2.arm(d), "stale arm");
        g2.pass(d, layers);
        if (i < 3) {
            g2.seal(d.ticket_id);
            check(r2.record(d.ticket_id, stream), "stale record");
            *S::last = 1;
            check(o.host_complete(d.ticket_id) && o.poll([&](const StepTicket& t, uint32_t) { return r2.prove(t); }) &&
                  o.outstanding()==0, "stale retire");
        }
    }
    check(r2.record(d.ticket_id, stream), "record without seal");   /* seal skipped: host slot still holds id 1 */
    *S::last = 1;
    check(o.host_complete(d.ticket_id) && !o.poll([&](const StepTicket& t, uint32_t) { return r2.prove(t); }) &&
          o.stopped() && o.outstanding()==1, "an older parent's sealed stamp faults, never proves a newer one");

    Ledger e(cap);
    check(e.initialize(6,0) && e.admit(1,&c)==R::Ok && e.bind(c.ticket_id,1), "error fixture");
    Completion r0;
    ConsumptionStamp st[cap]{};
    check(r0.prepare(cap, st) && r0.arm(c) && r0.record(c.ticket_id, stream), "error parent");
    *S::last = -1;
    check(e.host_complete(c.ticket_id) &&
          !e.poll([&](const StepTicket& t, uint32_t) { return r0.prove(t); }) && e.stopped(),
          "event query error faults closed");
    for (Completion& r : rank) r.release();
    r0.release(); r1.release(); r2.release();
    check(S::destroyed == S::created, "every pooled event destroyed once");
}

int main() {
    completion_ring();
    Ledger q(0);
    check(q.initialize(3,0) && q.try_admit(1, nullptr)==R::Invalid, "unconfigured fails closed");
    check(q.configure(2) && q.configure(2) && !q.configure(3), "engine ceiling configured once");
    check(q.capacity()==2,"fixture ceiling");
    StepTicket a{}, b{}, c{};
    check(q.admit(2,&a)==R::Ok,"admit");
    check(q.bind(a.ticket_id,50),"bind first parent");
    check(q.valid(a,0) && q.valid(a,1),"rank-local same admission");
    check(q.try_admit(2,&b)==R::Ok,"run ahead");
    check(q.bind(b.ticket_id,51),"bind ahead");
    check(q.high_water()==2 && q.outstanding()==2,"high-water recorded");
    check(q.try_admit(2,&c)==R::Full,"bounded outstanding");
    check(q.admit(2,&c)==R::Full,"full serving admission returns without a GPU wait");
    check(q.host_complete(b.ticket_id),"ahead host done");
    check(q.gpu_complete(b.ticket_id,1) && q.consumed(b.ticket_id,1),"ahead rank1");
    check(q.gpu_complete(b.ticket_id,0) && q.consumed(b.ticket_id,0),"ahead rank0");
    check(q.outstanding()==2,"out of order cannot retire");
    check(q.host_complete(a.ticket_id),"first host done");
    check(q.gpu_complete(a.ticket_id,0) && q.consumed(a.ticket_id,0),"first rank0");
    check(q.outstanding()==2,"first rank1 remains");
    check(q.gpu_complete(a.ticket_id,1) && q.consumed(a.ticket_id,1),"first rank1");
    check(q.outstanding()==0,"in-order cascading retirement");
    check(q.try_admit(1,&c)==R::Ok,"after retirement");
    check(q.cancel(c.ticket_id),"unused reservation cancelled");
    check(q.outstanding()==0,"cancel removes only unused tail");
    const uint64_t cancelled_id = c.ticket_id;
    check(q.try_admit(1,&c)==R::Ok,"new admission");
    check(c.ticket_id == cancelled_id,"unused reservations cannot create consumed-ring alias gaps");
    check(q.bind(c.ticket_id,53),"bind new");
    auto snap=q.close_admission();
    check(snap.outstanding==1 && snap.high_water==2 &&
          snap.latest_ticket==c.ticket_id && snap.old_epoch==3,"atomic close/snapshot");
    StepTicket n{};
    check(q.try_admit(1,&n)==R::Closed,"try-admit fails immediately during quiesce");
    check(!q.drained(3,snap.latest_ticket),"pending old epoch");
    check(q.host_complete(c.ticket_id),"last host complete");
    check(q.gpu_complete(c.ticket_id,0),"last gpu complete");
    check(!q.drained(3,snap.latest_ticket),"consumed is independent predicate");
    check(q.consumed(c.ticket_id,0),"last consumption");
    check(q.wait_drained(3,snap.latest_ticket,
          std::chrono::steady_clock::now()+std::chrono::milliseconds(50))==R::Ok,
          "drain returns only after retirement");
    check(q.reopen(4,1),"reopen new epoch");
    check(q.try_admit(1,&n)==R::Ok,"next epoch admitted");
    check(q.bind(n.ticket_id,54),"bind new epoch");
    snap=q.close_admission();
    std::atomic<R> outcome{R::Ok};
    std::thread waiter([&] {
        outcome=q.wait_drained(4,snap.latest_ticket,
            std::chrono::steady_clock::now()+std::chrono::seconds(10));
    });
    q.interrupt();
    waiter.join();
    check(outcome==R::Interrupted,"shutdown interrupts wait");
    Ledger f(2);
    check(f.initialize(8,0), "initialize fault fixture");
    StepTicket x{};
    check(f.try_admit(1,&x)==R::Ok,"fault fixture admission");
    check(f.bind(x.ticket_id,1),"fault bind");
    auto fs=f.close_admission();
    f.fault();
    check(f.wait_drained(8,fs.latest_ticket,
          std::chrono::steady_clock::now()+std::chrono::seconds(1))==R::Fault,
          "helper fault interrupts drain");
    check(!f.reopen(9,1),"fault never advances epoch");
    Ledger t(2);
    check(t.initialize(11,0), "initialize timeout fixture");
    StepTicket y{};
    check(t.try_admit(1,&y)==R::Ok && t.bind(y.ticket_id,10),"timeout fixture");
    auto ts=t.close_admission();
    check(t.wait_drained(11,ts.latest_ticket,
          std::chrono::steady_clock::now())==R::Timeout,"deadline enforced");
    check(!t.reopen(12,1),"nonempty ledger cannot reopen");
    // A scheduler already sleeping at the gate must allocate the reopened
    // epoch, rather than the epoch it observed before closing.
    Ledger gate(5);
    check(gate.capacity()==5 && gate.initialize(20,0), "explicit fixture ceiling");
    auto empty = gate.close_admission();
    check(empty.old_epoch==20 && empty.outstanding==0, "empty close retains epoch");
    std::atomic<bool> waiting{false};
    StepTicket after{};
    std::thread blocked([&] {
        outcome = gate.admit(1, &after, {}, [&] { waiting = true; });
    });
    while (!waiting.load()) std::this_thread::yield();
    check(gate.reopen(21,1), "reopen releases scheduler");
    blocked.join();
    check(outcome==R::Ok && after.placement_epoch==21 && after.generation_slot==1,
          "blocked scheduler receives new immutable epoch");
    check(gate.bind(after.ticket_id,1) && !gate.cancel(after.ticket_id) && gate.stopped(),
          "a rejected cancel (bound admission) faults the ledger");
    StepTicket forged=after; ++forged.placement_epoch;
    check(!gate.valid(forged,0) && !gate.valid(after,1), "forged ticket and invalid rank rejected");
    gate.close_admission();
    std::thread shutdown_wait([&] { outcome=gate.admit(1,&after); });
    gate.interrupt();
    shutdown_wait.join();
    check(outcome==R::Interrupted && !gate.reopen(22,0), "shutdown interrupts blocked admission");
    Ledger external(2);
    check(external.initialize(30,0), "initialize external fault fixture");
    external.close_admission();
    check(external.admit(1,&after,[]{return true;})==R::Fault &&
          !external.reopen(31,1), "helper fault interrupts closed admission and remains sticky");
    check(t.host_complete(y.ticket_id) && t.gpu_complete(y.ticket_id,0) && t.consumed(y.ticket_id,0),
          "complete timed-out parent");
    check(!t.reopen(12,1), "timeout stays failed closed even after retirement");
    // No-output prefill can leave many host-complete parents pending GPU
    // proof. Grow on demand to an engine-supplied ceiling, preserving the
    // fixed consumption-ring modulus across cancellations and retirement.
    Ledger backlog(33);
    check(backlog.initialize(40,0), "prefill fixture");
    for (unsigned i=0; i<33; ++i) {
        check(backlog.try_admit(2,&x)==R::Ok && backlog.bind(x.ticket_id,100+i) &&
              backlog.host_complete(x.ticket_id), "dynamic prefill backlog");
    }
    check(backlog.high_water()==33 && backlog.try_admit(2,&x)==R::Full,
          "derived ceiling exhausted without GPU wait");
    for (const auto& e : backlog.entries())
        for (unsigned rank=0;rank<2;++rank)
            check(backlog.gpu_complete(e.ticket.ticket_id,rank) &&
                  backlog.consumed(e.ticket.ticket_id,rank), "prefill completion");
    check(backlog.outstanding()==0 && backlog.high_water()==33, "prefill high-water retained");
    r3::moe_shm::LaneSignal lane{};
    unsigned drains=0;
    lane.drain_request=10;
    check(r3::moe_shm::acknowledge_lane_drain(lane,10,[&] {
        check(lane.drain_ack==0,"helper cannot ACK before its stream/combine drain");
        ++drains;
        return true;
    }) && lane.drain_ack==10, "helper publishes exact closure after drain");
    lane.drain_request=11;
    check(!r3::moe_shm::acknowledge_lane_drain(lane,11,[]{return false;}) &&
          lane.drain_ack==10, "helper drain fault cannot publish proof");
    check(!r3::moe_shm::acknowledge_lane_drain(lane,10,[&]{++drains;return true;}) &&
          drains==1, "stale closure rejected before touching queues");
    check(!r3::moe_shm::acknowledge_lane_drain(lane,11,[&]{
        lane.drain_request=12; return true;
    }) && lane.drain_ack==10, "replaced closure cannot receive a stale ACK");
    std::puts("r3_admission_ledger_test: OK (admission, high-water, close, ordered retirement, interruption, fault)");
}
