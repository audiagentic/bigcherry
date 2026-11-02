#include "admission_ledger.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
using r3::admission::Ledger;
using R = Ledger::Result;
static void check(bool b, const char* why) {
    if (!b) { std::fprintf(stderr,"FAIL: %s\n",why); std::abort(); }
}
int main() {
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
    check(gate.bind(after.ticket_id,1) && !gate.cancel(after.ticket_id), "bound admission cannot cancel");
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
