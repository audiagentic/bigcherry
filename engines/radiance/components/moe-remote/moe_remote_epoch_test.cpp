#include "worker_epoch.h"
#include "worker_bootstrap.h"
#include "admission_ledger.h"
#include "owner_generation.h"
#include <sys/mman.h>
#include "worker_residency.h"
#include "worker_heat.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace r3::epoch;
using namespace r3::moe_shm;
static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr,"r3_moe_epoch_test: FAILED %s\n",what); std::exit(1); }
}
struct FakeClock : Clock {
    uint64_t t=100;
    uint64_t now_ns() const override { return t; }
    void elapse(uint64_t ns) { t+=ns; }
};
struct FakeTransport : Transport {
    std::array<PlacementAck,kMaxHelpers+kMaxEngineRanks> a{};
    PlacementControl snapshot{};
    PlacementJournal log{};
    uint32_t publications=0;
    PlacementAck read_ack(uint32_t actor) const override { return a.at(actor); }
    void publish(const PlacementControl& c,const PlacementJournal& j) override {
        snapshot=c;log=j;++publications;
    }
    void ack(const PlacementControl& c,AckPhase p,uint32_t actor) {
        a.at(actor)={};
        auto& x=a[actor];
        x.transaction_id=c.transaction_id; x.epoch=c.pending_epoch;
        x.digest=c.pending_digest; x.generation_slot=c.pending_slot;
        x.phase=uint32_t(p);
    }
    void all(const PlacementControl& c,AckPhase p) {
        for (uint32_t i=0;i<kMaxHelpers;++i)
            if (c.expected_helpers & (1u<<i)) ack(c,p,i);
        for (uint32_t i=0;i<kMaxEngineRanks;++i)
            if (c.expected_ranks & (1u<<i)) ack(c,p,kMaxHelpers+i);
    }
};
static void complete(Controller& c,FakeTransport& t) {
    check(c.start_reserving(),"reserving");
    t.all(c.control(),AckPhase::RESERVED);
    check(c.poll()==Poll::ADVANCED && c.control().state==uint32_t(PlacementState::COPYING),"reserve ACK -> copying");
    t.all(c.control(),AckPhase::READY);
    check(c.poll()==Poll::ADVANCED && c.control().state==uint32_t(PlacementState::PREPARED),"copy READY -> prepared");
    check(c.begin_quiesce(),"explicit quiesce");
    check(!c.commit(),"no commit without drained actors");
    t.all(c.control(),AckPhase::QUIESCED);
    check(c.poll()==Poll::ADVANCED && c.commit(),"quiesced proof + explicit commit");
    t.all(c.control(),AckPhase::COMMITTED);
    check(c.poll()==Poll::ADVANCED && c.control().state==uint32_t(PlacementState::RETIRING),"committed -> retire");
    t.all(c.control(),AckPhase::RETIRED);
    check(c.poll()==Poll::ADVANCED && c.control().state==uint32_t(PlacementState::ACTIVE),"retired -> active");
}
static void V23Epoch_ForwardOnlyAbortRetry() {
    FakeClock clk; FakeTransport tr;
    Controller c(tr,clk,101,2,1);
    check(c.control().active_epoch==1 && c.control().active_slot==0 &&
          c.control().expected_helpers==3 && c.control().expected_ranks==1,"ACTIVE(1) actual participants");
    check(!c.propose(101) && c.propose(202),"no_change vs proposed");
    check(c.control().pending_epoch==2 && c.control().pending_slot==1 &&
          c.control().state==uint32_t(PlacementState::PROPOSED),"pending epoch2");
    check(!c.propose(303),"single transaction");
    check(c.start_reserving() && c.fail(FaultCode::CAPACITY,1),"precommit fault abort");
    check(c.control().active_epoch==1 && c.control().active_digest==101 &&
          c.control().pending_epoch==0 && !c.failed_closed(),"abort ACTIVE(1) unchanged");
    check(c.journal().entry[c.control().journal_index].fault==uint32_t(FaultCode::CAPACITY),
          "two-entry abort journal");
    check(c.propose(202) && c.control().pending_epoch==2 &&
          c.control().transaction_id==2,"retry uses new transaction id, not new epoch");
    complete(c,tr);
    check(c.control().active_epoch==2 && c.control().active_slot==1 &&
          c.control().active_digest==202,"committed ACTIVE(2)");
    check(c.propose_rollback() && c.control().pending_epoch==3 &&
          c.control().pending_digest==101,"rollback is new epoch");
    complete(c,tr);
    check(c.control().active_epoch==3 && c.control().active_digest==101 &&
          c.journal().entry[0].transaction_id==2 &&
          c.journal().entry[1].transaction_id==3,"two-entry journal rollover");
    check(c.propose(202) && c.fail(FaultCode::CAPACITY),"post-success aborted attempt");
    check(c.propose(202) && c.fail(FaultCode::CAPACITY),"second aborted attempt");
    check(c.journal().entry[1].transaction_id==3 &&
          c.journal().entry[1].commit_ns!=0,
          "last committed journal survives repeated aborts");
    FakeTransport bad; Controller invalid(bad,clk,77,0,1);
    check(invalid.failed_closed() && !invalid.propose(88),"invalid participant count");
    FakeTransport four_ack;Controller four(four_ack,clk,500,4,2);
    check(four.control().expected_helpers==15 && four.control().expected_ranks==3 &&
          four.propose(501),"all four helpers and actual two ranks");
    complete(four,four_ack);
    check(four.control().active_epoch==2,"four helper transition");
    FakeTransport emax;Controller exhausted(emax,clk,3,1,1,Deadlines{},
                                         Bootstrap{UINT64_MAX,0,0,0});
    check(!exhausted.propose(4),"64-bit epoch exhaustion refused");
    FakeTransport smax;Controller sequence(smax,clk,3,1,1,Deadlines{},
                                         Bootstrap{1,0,UINT32_MAX-3,0});
    check(!sequence.propose(4),"32-bit publication-sequence exhaustion refused");
    FakeTransport tmax;Controller transaction(tmax,clk,3,1,1,Deadlines{},
                                         Bootstrap{1,0,0,UINT64_MAX});
    check(!transaction.propose(4),"64-bit txn exhaustion refused");
    FakeTransport dt;Controller fixed(dt,clk,3,1,1,Deadlines{},Bootstrap{},true);
    check(!fixed.propose(4) && fixed.control().active_epoch==1,
          "deterministic controller never proposes");
}
static void V23Ack_OldTxnDelayedDuplicateLost() {
    FakeClock clk;FakeTransport tr; Controller c(tr,clk,11,1,1);
    check(c.propose(22) && c.start_reserving(),"propose -> reserve");
    auto a=c.control();
    tr.ack(a,AckPhase::RESERVED,0);
    tr.ack(a,AckPhase::RESERVED,kMaxHelpers);
    tr.a[0].transaction_id--;
    check(c.poll()==Poll::WAITING && c.control().ready_helpers==0,"old txn rejected");
    tr.ack(a,AckPhase::READY,0);
    check(c.poll()==Poll::WAITING,"delayed wrong phase ignored");
    tr.ack(a,AckPhase::RESERVED,0);
    check(c.poll()==Poll::ADVANCED,"exact ACK only");
    check(c.poll()==Poll::WAITING,"delayed RESERVED not READY");
    tr.all(c.control(),AckPhase::READY);
    check(c.poll()==Poll::ADVANCED && c.poll()==Poll::WAITING,"prepared, duplicate READY ignored");
    c.begin_quiesce();
    clk.elapse(31'000'000'000ULL);
    check(c.poll()==Poll::FAILED && c.failed_closed() &&
          c.control().active_epoch==1 && !c.commit(),"lost QUIESCED -> timeout fail-closed");
    FakeClock later;FakeTransport lost;Controller w(lost,later,5,1,1);
    check(w.propose(6) && w.start_reserving(),"second scenario");
    later.elapse(61'000'000'000ULL);
    check(w.poll()==Poll::FAILED && !w.failed_closed() && w.control().active_epoch==1,
          "lost RESERVED timeout abort without advancing epoch");
}
static void V23Ack_CorruptCurrentAndFault() {
    for (int mode=0;mode<4;++mode) {
        FakeClock clk;FakeTransport tr;Controller c(tr,clk,11,1,1);
        check(c.propose(22) && c.start_reserving(),"corrupt ACK scenario");
        tr.all(c.control(),AckPhase::RESERVED);
        if (mode==0) tr.a[0].epoch++;
        if (mode==1) tr.a[0].digest++;
        if (mode==2) tr.a[0].generation_slot^=1;
        if (mode==3) tr.a[0].fault_code=uint32_t(FaultCode::SOURCE);
        check(c.poll()==Poll::FAILED && c.control().active_epoch==1 &&
              c.control().pending_epoch==0,"matching txn bad ACK aborts");
        const uint32_t expected=mode==1?uint32_t(FaultCode::CHECKSUM):
            mode==3?uint32_t(FaultCode::SOURCE):uint32_t(FaultCode::EPOCH_MISMATCH);
        check(c.journal().entry[c.control().journal_index].fault==expected,
              "immediate exact fault classification");
    }
}
static void V23Fault_BeforeAndAfterCommit() {
    FakeClock clk;FakeTransport tr;Controller c(tr,clk,123,1,1);
    check(c.propose(234) && c.start_reserving(),"begin");
    check(c.fail(FaultCode::HELPER_DEAD,0) && !c.failed_closed(),"helper death before commit aborts");
    check(c.propose(234) && c.start_reserving(),"retry after helper death");
    tr.all(c.control(),AckPhase::RESERVED);
    check(c.poll()==Poll::ADVANCED && c.fail(FaultCode::UPLOAD,0),"mid-copy error precommit");
    check(c.control().active_epoch==1,"upload fault never advances active");
    check(c.propose(234) && c.start_reserving(),"second retry");
    tr.all(c.control(),AckPhase::RESERVED);
    check(c.poll()==Poll::ADVANCED,"copying");
    tr.all(c.control(),AckPhase::READY);
    check(c.poll()==Poll::ADVANCED && c.begin_quiesce(),"prepared");
    tr.all(c.control(),AckPhase::QUIESCED);
    check(c.poll()==Poll::ADVANCED && c.commit(),"commit");
    check(!c.fail(FaultCode::RANK_DEAD,actor_id(ActorKind::ENGINE_RANK,0)) && c.failed_closed() &&
          c.control().active_epoch==2 && c.control().state==uint32_t(PlacementState::COMMIT) &&
          !c.propose(123) && !c.propose_rollback(),"postcommit death stops service, never regresses epoch");
    check(c.control().fault_code==uint32_t(FaultCode::RANK_DEAD) &&
          c.control().fault_actor==actor_id(ActorKind::ENGINE_RANK,0) &&
          c.journal().entry[c.control().journal_index].fault==uint32_t(FaultCode::RANK_DEAD),"fault journal");
    check(!c.fail(FaultCode::CHECKSUM) && c.control().fault_code==uint32_t(FaultCode::RANK_DEAD),
          "first postcommit fault sticky");
    FakeTransport dead;Controller q(dead,clk,1,1,1);
    q.propose(2);q.start_reserving();dead.all(q.control(),AckPhase::RESERVED);q.poll();
    dead.all(q.control(),AckPhase::READY);q.poll();q.begin_quiesce();
    check(q.fail(FaultCode::RESULT_NOT_CONSUMED) && q.failed_closed() &&
          q.control().active_epoch==1,"quiescence output uncertainty fail-closed");
}
static void V23Slots_ReserveCopyRetireFaults() {
    using namespace r3::residency;
    Ledger l(120,10,20);
    uint32_t old=UINT32_MAX, incoming=UINT32_MAX, second=UINT32_MAX;
    check(l.bootstrap(1,60,1,&old)==Result::OK && l.charged()==60,"bootstrap and incoming headroom");
    check(l.reserve(2,30,2,false,&incoming)==Result::SOURCE_UNAVAILABLE,"no source no reservation");
    check(l.reserve(2,30,2,true,&incoming)==Result::OK && l.charged()==90,"shadow charged");
    check(l.start_copy(incoming)==Result::OK && l.ready(incoming)==Result::INVALID,"all four planes required");
    check(l.reserve(3,10,2,true,&second)==Result::OK &&
          l.start_copy(second)==Result::BUSY,"only one in-flight upload");
    check(l.cancel(incoming,false)==Result::INVALID,"inflight cannot be silently freed");
    for (uint32_t plane=0;plane<3;++plane)
        check(l.plane_complete(incoming,plane)==Result::OK,"plane upload proof");
    check(l.ready(incoming)==Result::INVALID,"partial fourth plane blocks READY");
    check(l.plane_complete(incoming,3)==Result::OK && l.ready(incoming)==Result::OK &&
          l.activate(incoming,2)==Result::OK,"four complete planes permit active new generation");
    check(l.reclaim(old,true)==Result::INVALID && l.retire(old)==Result::OK &&
          l.reclaim(old,false)==Result::INVALID,"old resident until explicit retirement proof");
    check(l.charged()==100 && l.valid(),"retiring still charged");
    check(l.reclaim(old,true)==Result::OK && l.charged()==40 &&
          l.cancel(second,true)==Result::OK && l.charged()==30,"no leak after retirement/cancel");
    Ledger a(100,0,0),b(100,0,0);uint32_t ia=0,ib=0;
    check(a.bootstrap(1,100,1,&ia)==Result::OK && b.bootstrap(2,100,1,&ib)==Result::OK,
          "two full helpers");
    check(a.reserve(2,100,2,true,&ia)==Result::DEFERRED_CAPACITY &&
          b.reserve(1,100,2,true,&ib)==Result::DEFERRED_CAPACITY,
          "two-way full swap deferred, never overwrites live");
    Ledger empty(100,10,20);uint32_t fresh=0;
    check(empty.reserve(9,30,2,true,&fresh)==Result::OK,"empty helper accepts first incoming");
    Ledger churn(80,10,20);
    uint32_t active=0, next=0;
    check(churn.bootstrap(1,30,1,&active)==Result::OK,"churn bootstrap");
    for (uint64_t epoch=2;epoch<=1001;++epoch) {
        check(churn.reserve(uint32_t(2+epoch%2),20,epoch,true,&next)==Result::OK,
              "1000 reserves");
        check(churn.start_copy(next)==Result::OK,"churn copy");
        for(uint32_t plane=0;plane<4;++plane)
            check(churn.plane_complete(next,plane)==Result::OK,"churn all planes");
        check(churn.ready(next)==Result::OK && churn.activate(next,epoch)==Result::OK,
              "churn activate");
        check(churn.retire(active)==Result::OK &&
              churn.reclaim(active,true)==Result::OK,"churn reclaim");
        active=next;
        check(churn.valid() && churn.slots().size()<=2,"no growth of charged/metadata slots");
    }
}
static double oracle(const r3::heat::Input& in) {
    const size_t n=in.current.size(), h=in.devices.size();
    std::vector<int32_t> a=in.current;
    double best=std::numeric_limits<double>::infinity();
    auto dfs=[&](auto&& self,size_t i,uint32_t moves,uint64_t bytes)->void {
        if (i==n) {
            std::vector<uint64_t> used(h,0);
            std::vector<std::vector<double>> load(h,std::vector<double>(2,0));
            for(size_t j=0;j<n;++j) if(a[j]>=0) {
                size_t d=size_t(a[j]);
                used[d]+=in.bytes[j];
                if(used[d]>in.devices[d].budget)return;
                load[d][in.layer[j]]+=in.heat[j];
            }
            double sum=0;
            for(size_t ly=0;ly<2;++ly) {
                double mx=0;
                for(size_t d=0;d<h;++d) {
                    if(in.devices[d].layer_cap && load[d][ly]>in.devices[d].layer_cap)return;
                    mx=std::max(mx,load[d][ly]*in.devices[d].cost);
                }
                sum+=mx;
            }
            best=std::min(best,sum);
            return;
        }
        for (int32_t d=0;d<int32_t(h);++d) {
            if (!in.allowed[i][size_t(d)])continue;
            const bool change=d!=in.current[i];
            if(change && (moves==in.max_moves || bytes+in.bytes[i]>in.max_moved_bytes))continue;
            a[i]=d;
            self(self,i+1,moves+change,bytes+(change?in.bytes[i]:0));
        }
        a[i]=in.current[i];
    };
    dfs(dfs,0,0,0);
    return best;
}
static void V23Policy_HeatEqualCostAndSwap() {
    using namespace r3::heat;
    // Independent exhaustive oracle over whole assignments and exact hard constraints.
    for (int seed=0;seed<30;++seed) {
        Input in;
        in.bytes={10,10,10,10,10,10};
        in.current={0,0,0,1,1,1};
        in.layer={0,1,0,1,0,1};
        in.heat.resize(6);
        for(int i=0;i<6;++i)in.heat[i]=double(1+(seed*7+i*11)%97);
        in.devices={{40,1.0,0},{50,(seed%3==0?1.0:1.4),0}};
        in.allowed.assign(6,{1,1});
        in.max_moves=2;in.max_moved_bytes=20;in.margin=0;in.now_ns=2'000'000'000'000ULL;
        if(seed%2)in.allowed[0][1]=0;
        const Plan p=propose(in);
        check(p.valid && delta_matches(p,in.current),"valid whole-map oracle proposal");
        check(std::abs(p.predicted-oracle(in))<1e-7,"small-instance exact global optimum");
        check(p.delta.size()<=2 && p.used[0]<=40 && p.used[1]<=50,"oracle budgets");
    }
}
/* Card-free fake HIP upload streams/events, using the exact same
 * build_bitmap/ledger and engine ACK predicates as the real coordinator. */
struct FakeCycleUpload {
    std::vector<uint8_t> bits[2];
    R3MoeOwnerBinding binding[2]{};
    bool issued[2]{}, event_ready[2]{};
    bool upload(const void* map,const r3::admission::Ledger& ledger,
                const PlacementControl& c) {
        const uint32_t slot=c.pending_slot;
        if (slot > 1 || !r3::owner_generation::may_upload(ledger,slot) ||
            !r3::owner_generation::build_bitmap(map,slot,c.pending_epoch,
                                                 c.pending_digest,&bits[slot]))
            return false;
        binding[slot]={c.pending_epoch,c.pending_digest};
        issued[slot]=true;
        event_ready[slot]=false;
        return true;
    }
    void complete(uint32_t slot) { event_ready[slot]=true; }
    bool prepared(const PlacementControl& c,uint32_t rank) const {
        const uint32_t slot=c.pending_slot;
        return slot < 2 && r3::epoch::rank_upload_prepared(c,rank,
            issued[slot],event_ready[slot],binding[slot].epoch,binding[slot].digest,slot);
    }
};

static void V23Cycle_UnreadyAndAbortPreserveOldGeneration() {
    FakeClock clk; FakeTransport tr;
    Controller c(tr,clk,97,2,2);
    check(c.begin_epoch_cycle() && c.control().state==uint32_t(PlacementState::PROPOSED) &&
          c.control().pending_slot==1 && c.control().pending_epoch==2 &&
          c.control().active_slot==0 && c.control().active_epoch==1,
          "real two-slot diagnostic proposal preserves active");
    check(c.start_reserving(), "diagnostic reserve begins");
    tr.all(c.control(),AckPhase::RESERVED);
    check(c.poll()==Poll::ADVANCED &&
          c.control().state==uint32_t(PlacementState::COPYING), "diagnostic COPY waits for events");
    FakeCycleUpload upload[2]{};
    const auto& pending=c.control();
    for (uint32_t r=0;r<2;++r)
        check(!r3::epoch::rank_upload_prepared(pending,r,true,false,2,97,1),
              "upload-not-ready blocks PREPARED per rank");
    tr.ack(c.control(),AckPhase::READY,0);
    tr.ack(c.control(),AckPhase::READY,1);
    tr.ack(c.control(),AckPhase::READY,kMaxHelpers);
    check(c.poll()==Poll::WAITING &&
          c.control().state==uint32_t(PlacementState::COPYING),
          "missing second rank READY prevents PREPARED");
    clk.elapse(181'000'000'000ULL);
    check(c.poll()==Poll::FAILED && !c.failed_closed() &&
          c.control().state==uint32_t(PlacementState::ACTIVE) &&
          c.control().active_epoch==1 && c.control().active_slot==0 &&
          c.control().active_digest==97, "abort before commit preserves epoch slot owner digest");
    const uint64_t failed_txn=c.control().transaction_id;
    check(c.begin_epoch_cycle() && c.control().transaction_id==failed_txn+1 &&
          c.control().active_epoch==1 && c.control().pending_epoch==2,
          "retry is a fresh transaction without phantom epoch advance");
    check(!r3::epoch::rank_retirement_proved(
          c.control(),1,0,true,true,true,true,true,true),
          "retirement refused before COMMIT and RETIRING");
    (void)upload;
}

/* A rejected diagnostic proposal cannot kill a serving helper, poison the
 * shared GPU waits, close admission, or make the next parent use epoch N+1.
 * Exercise all actual helper EpochCycle participants without GPUs. */
static void V23Cycle_PrecommitAbortDuringLiveServing() {
    const LayoutShape shape{128,3,5,8,128,64};
    Layout layout{};
    check(layout_for(shape,3,slots_for(3),&layout),"abort segment layout");
    void* map=mmap(nullptr,layout.bytes,PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    check(map!=MAP_FAILED && init(map,3,shape,slots_for(3)),
          "abort segment created");
    auto* h=static_cast<Header*>(map);
    h->helper_count=3;
    auto* c=placement_control(map);
    c->expected_ranks=1;
    struct Selected { std::vector<std::pair<uint32_t,uint32_t>> expert; };
    check(r3::epoch::publish_owned(map,Selected{{{0,0}}},0) &&
          r3::epoch::publish_owned(map,Selected{{{1,2}}},1) &&
          r3::epoch::publish_owned(map,Selected{{{2,4}}},2),
          "abort segment real owner publication");
    for (uint32_t helper=0;helper<3;++helper)
        check(r3::epoch::mark_bootstrap_filtered(map,helper),
              "abort segment real helper filtering completion");
    SharedTransport transport(map);
    check(transport.seal_bootstrap(),"abort segment bootstrap seal");
    const uint64_t digest=generation_digest(map,0);
    FakeClock clock;
    EpochCycle leader(map,0,clock,std::chrono::nanoseconds{100});
    EpochCycle peer1(map,1,clock,std::chrono::nanoseconds{100});
    EpochCycle peer2(map,2,clock,std::chrono::nanoseconds{100});
    r3::admission::Ledger ledger(5);
    check(ledger.initialize(1,0),"active serving ledger");
    check(leader.tick() && peer1.tick() && peer2.tick(),"live participants bootstrap");
    h->burst=1;
    clock.elapse(100);
    check(leader.tick() && c->state==uint32_t(PlacementState::PROPOSED),
          "live parent still admitted while candidate is proposed");
    StepTicket before{};
    check(ledger.admit(1,&before)==r3::admission::Ledger::Result::Ok &&
          ledger.bind(before.ticket_id,7) && before.placement_epoch==1 &&
          before.generation_slot==0,"live parent step before abort on N");
    check(peer1.tick() && peer2.tick(),"peers survive proposal");
    check(leader.tick() && c->state==uint32_t(PlacementState::RESERVING),
          "reservation begins without closing serving");
    PlacementControl snapshot{};
    check(placement_snapshot(map,&snapshot),"abort cycle snapshot");
    publish_fault_ack(map,kMaxHelpers,snapshot,AckPhase::RESERVED,FaultCode::SOURCE);
    check(leader.tick() && c->state==uint32_t(PlacementState::ACTIVE) &&
          c->active_epoch==1 && c->active_slot==0 &&
          c->active_digest==digest && c->fault_code==0,
          "precommit actor fault aborts to exact ACTIVE N");
    check(placement_journal(map)->entry[c->journal_index].fault==
          uint32_t(FaultCode::SOURCE),"precommit fault journal retained");
    check(peer1.tick() && peer2.tick() && leader.tick() &&
          h->helper_fault==0 && !ledger.stopped(),
          "all helpers stay alive; no sticky GPU trap; admission open");
    check(ledger.host_complete(before.ticket_id) &&
          ledger.poll([](const StepTicket&,uint32_t) {
              return r3::admission::Ledger::Proof::Proven;
          }),"in-flight parent N retires after abort");
    StepTicket after{};
    check(ledger.admit(1,&after)==r3::admission::Ledger::Result::Ok &&
          ledger.bind(after.ticket_id,8) &&
          after.placement_epoch==1 && after.generation_slot==0 &&
          h->helper_fault==0,"next parent served under N after abort");
    check(ledger.host_complete(after.ticket_id) &&
          ledger.poll([](const StepTicket&,uint32_t) {
              return r3::admission::Ledger::Proof::Proven;
          }),"next parent consumption and completion");
    // Candidate-only metadata corruption must not turn into a helper exit.
    // The active generation and the admission ledger remain valid.
    auto* inactive=generation_header(map,1);
    const uint32_t map_bytes=inactive->map_bytes;
    inactive->map_bytes=0;
    clock.elapse(100);
    check(leader.tick() && c->state==uint32_t(PlacementState::ACTIVE) &&
          c->active_epoch==1 && c->active_slot==0 &&
          c->transaction_id==snapshot.transaction_id+1 &&
          placement_journal(map)->entry[c->journal_index].fault==
              uint32_t(FaultCode::INVALID_MAP) &&
          peer1.tick() && peer2.tick() && h->helper_fault==0 &&
          !ledger.stopped(),
          "bad inactive candidate journals abort without helper exit or trap");
    inactive->map_bytes=map_bytes;
    clock.elapse(100);
    check(leader.tick() && c->transaction_id==snapshot.transaction_id+2 &&
          c->state==uint32_t(PlacementState::PROPOSED) &&
          c->active_epoch==1 && h->helper_fault==0,
          "retry after invalid candidate gets fresh transaction and N");
    munmap(map,layout.bytes);
}
static void V23Cycle_StalledParkFaultsAndKeepsAdmissionClosed() {
    const LayoutShape shape{128,1,3,8,128,64};
    Layout layout{};
    check(layout_for(shape,2,slots_for(1),&layout),"park layout");
    void* map=mmap(nullptr,layout.bytes,PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    check(map!=MAP_FAILED && init(map,2,shape,slots_for(1)),"park segment");
    auto* h=static_cast<Header*>(map);
    h->lane[0].park=kParkDraining;
    r3::admission::Ledger ledger(2);
    check(ledger.initialize(1,0),"park gate ledger");
    const auto closed=ledger.close_admission();
    check(ledger.drained(closed.old_epoch,closed.latest_ticket),
          "park gate initially drained");
    timespec deadline{};
    check(clock_gettime(CLOCK_MONOTONIC,&deadline)==0,"monotonic clock");
    // Expired absolute deadline: no arbitrary duration or GPU execution.
    deadline.tv_sec-=1;
    check(wait_parks_checked(h,&deadline)==WaitResult::Timeout &&
          load(&h->helper_fault)!=0,"stalled park times out and raises sticky fault");
    ledger.fault();
    check(ledger.stopped() && !ledger.reopen(2,1),
          "stalled park never reopens admission or releases old generation");
    munmap(map,layout.bytes);
}

/* The unsealed production path must reject the cycle with the exact
 * INVALID_MAP journal code, not silently repair the old all-local digest. */
static void V23Bootstrap_UnsealedTickReportsInvalidMap() {
    const LayoutShape shape{128,3,3,8,128,64};
    Layout layout{};
    check(layout_for(shape,2,slots_for(3),&layout),"bootstrap negative layout");
    void* base=mmap(nullptr,layout.bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    check(base!=MAP_FAILED && init(base,2,shape,3),"bootstrap negative init");
    auto* h=static_cast<Header*>(base);
    h->helper_count=3;
    placement_control(base)->expected_ranks=1;
    struct Selected { std::vector<std::pair<uint32_t,uint32_t>> expert; };
    Selected selected{{{0,0},{1,0}}};
    check(r3::epoch::publish_owned(base,selected,0) &&
          r3::epoch::drop_bootstrap_owner(base,1,0,0),
          "bootstrap negative real publish and filter");
    for (uint32_t helper=0;helper<3;++helper)
        check(r3::epoch::mark_bootstrap_filtered(base,helper),
              "bootstrap negative all peers filtered");
    check(!bootstrap_generation_sealed(base) &&
          generation_digest(base,0)!=placement_control(base)->active_digest,
          "unsealed canonical map cannot be used");
    FakeClock clk;
    EpochCycle helper0(base,0,clk,std::chrono::nanoseconds{100});
    check(helper0.tick(),"construct real controller for unsealed startup");
    h->burst=1;
    clk.elapse(100);
    check(helper0.tick(),"unsealed cycle rejected without terminating serving");
    const auto* c=placement_control(base);
    const auto& j=placement_journal(base)->entry[c->journal_index];
    check(c->state==uint32_t(PlacementState::ACTIVE) &&
          c->active_epoch==1 && c->active_slot==0 &&
          j.fault==uint32_t(FaultCode::INVALID_MAP) &&
          !load(&h->helper_fault),
          "unsealed real startup yields transaction-local INVALID_MAP");
    munmap(base,layout.bytes);
}

/* The actual SharedTransport/Helper EpochCycle state machine under a
 * two-slot same-map diagnostic, with fake rank GPU uploads and consumption
 * events. No GPU or background threads. */
static void V23Cycle_SamePlacementAndSharedProtocol(
    uint32_t helper_count,uint32_t rank_count) {
    check((helper_count==1 && rank_count==2) ||
          (helper_count==3 && rank_count==1),"only supported production shapes");
    const LayoutShape shape{128,helper_count,3,8,128,64};
    Layout layout{};
    check(layout_for(shape,2,slots_for(helper_count),&layout), "cycle layout");
    void* base=mmap(nullptr,layout.bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    check(base!=MAP_FAILED && init(base,2,shape,helper_count), "cycle segment");
    auto* h=static_cast<Header*>(base);
    h->helper_count=helper_count;
    auto* control=placement_control(base);
    control->expected_ranks=(1u<<rank_count)-1u;
    check(!stage_same_map_generation(base),
          "init all-local map with matching digest is not a sealed generation");
    /* Real worker bootstrap: init's digest is of the initial all-local map.
     * Helpers publish disjoint ownership and discard protected runtime
     * experts, then helper 0 seals exactly once after every peer is done. */
    struct Selected { std::vector<std::pair<uint32_t,uint32_t>> expert; };
    std::vector<Selected> selection(helper_count);
    for (uint32_t l=0;l<2;++l)
        for (uint32_t e=0;e<3;++e)
            selection[e%helper_count].expert.emplace_back(l,e);
    for (uint32_t helper=0;helper<helper_count;++helper)
        check(r3::epoch::publish_owned(base,selection[helper],helper),
              "real helper publish_owned");
    check(r3::epoch::drop_bootstrap_owner(base,1,0,0),
          "real helper protected-layer drop_bootstrap_owner");
    check(generation_digest(base,0) != control->active_digest &&
          !bootstrap_generation_sealed(base) && !stage_same_map_generation(base),
          "unsealed real bootstrap cannot stage same-map generation");
    SharedTransport bootstrap(base);
    check(!bootstrap.seal_bootstrap(), "incomplete helper completion mask cannot seal");
    for (uint32_t helper=0;helper<helper_count;++helper)
        check(r3::epoch::mark_bootstrap_filtered(base,helper),
              "real helper marks filtered owner bytes complete");
    check(bootstrap.seal_bootstrap() && !bootstrap.seal_bootstrap(),
          "helper zero publishes single coherent bootstrap seal");
    const uint64_t digest=generation_digest(base,0);
    check(bootstrap_generation_sealed(base) && control->active_digest==digest &&
          generation_header(base,0)->digest==digest &&
          generation_header(base,0)->ready_helpers==control->expected_helpers,
          "sealed owner digest and helper readiness agree under seqlock");
    Selected newly_unowned{{{1,0}}}; // explicitly dropped above: CAS would succeed without the seal guard
    check(!r3::epoch::publish_owned(base,newly_unowned,0) &&
          !r3::epoch::drop_bootstrap_owner(base,0,0,0) &&
          !r3::epoch::mark_bootstrap_filtered(base,0),
          "after seal every bootstrap map mutation and duplicate completion fails closed");
    uint8_t* active_owners=generation_map(base,0);
    const uint8_t original=active_owners[0];
    active_owners[0]=kOwnerLocal; // corrupt a sealed, actually-owned expert
    check(!stage_same_map_generation(base) &&
          generation_header(base,1)->state==uint32_t(GenerationState::EMPTY),
          "invalid active digest refuses staging before modifying inactive generation");
    active_owners[0]=original;
    const std::vector<uint8_t> owners(active_owners,active_owners+h->owner_map_bytes);
    FakeClock clock;
    EpochCycle disabled(base,0,clock,std::chrono::nanoseconds{0});
    clock.elapse(100);
    check(disabled.tick() && control->transaction_id==0 && control->seq==2,
          "zero period disables cycle");
    EpochCycle helper0(base,0,clock,std::chrono::nanoseconds{100});
    std::vector<std::unique_ptr<EpochCycle>> peers;
    for (uint32_t helper=1;helper<helper_count;++helper)
        peers.emplace_back(std::make_unique<EpochCycle>(
            base,helper,clock,std::chrono::nanoseconds{100}));
    auto peer_ticks=[&]() {
        for (auto& peer:peers) if (!peer->tick()) return false;
        return true;
    };
    check(helper0.tick() && peer_ticks(), "bootstrap controller with actual ranks");
    clock.elapse(100);
    check(helper0.tick() && control->state==uint32_t(PlacementState::ACTIVE),
          "idle burst does not start cycle");
    h->burst=1;
    r3::admission::Ledger ledger(5);
    check(ledger.initialize(1,0), "cycle ledger");
    FakeCycleUpload gpu[2];
    for (uint32_t rank=0;rank<rank_count;++rank) {
        check(r3::owner_generation::build_bitmap(base,0,1,digest,&gpu[rank].bits[0]),
              "bootstrap active bitmap");
        gpu[rank].binding[0]={1,digest};
        gpu[rank].issued[0]=gpu[rank].event_ready[0]=true;
    }
    uint32_t slot=0;
    const uint64_t cycles=1000;
    for (uint64_t cycle=1; cycle<=cycles; ++cycle) {
        const uint32_t old_slot=slot,new_slot=1u-slot;
        StepTicket ticket{};
        check(ledger.admit(2,&ticket)==r3::admission::Ledger::Result::Ok &&
              ledger.bind(ticket.ticket_id,cycle) &&
              ticket.generation_slot==old_slot &&
              ticket.placement_epoch==cycle, "serving parent belongs to currently active generation");
        check(helper0.tick() && control->state==uint32_t(PlacementState::PROPOSED) &&
              control->transaction_id==cycle &&
              control->active_slot==old_slot &&
              control->pending_slot==new_slot &&
              control->pending_epoch==cycle+1 &&
              same_map_cycle(base,*control), "cycle PROPOSED contains new exact same-map generation");
        check(generation_header(base,new_slot)->state==uint32_t(GenerationState::STAGING) &&
              std::equal(owners.begin(),owners.end(),generation_map(base,new_slot)),
              "genuinely separate inactive canonical map staged unchanged");
        check(peer_ticks() && helper0.tick() &&
              control->state==uint32_t(PlacementState::RESERVING),
              "proposal enters reservation");
        check(peer_ticks() && helper0.tick(), "helpers publish reserved ACK");
        PlacementControl snapshot{};
        check(placement_snapshot(base,&snapshot), "reserving snapshot");
        const uint32_t rank_mask=(1u<<rank_count)-1u;
        for (uint32_t rank=0;rank<rank_count;++rank) {
            auto& device=gpu[rank];
            check(!service_rank_reservation(base,snapshot,rank,rank_mask,false,true) &&
                  !service_rank_reservation(base,snapshot,rank,rank_mask,true,false),
                  "production rank participant refuses unregistered or unready device");
            if (rank==0)
                check(!service_rank_reservation(base,snapshot,rank,
                        rank_count==1 ? 3u : 1u,true,true),
                      "production rank refuses valid but mismatched actual rank mask");
            if (cycle==1 && rank==0) {
                GenerationHeader* pending=generation_header(base,new_slot);
                const uint64_t saved=pending->digest;
                pending->digest^=1u;
                check(!service_rank_reservation(base,snapshot,rank,rank_mask,true,true),
                      "reservation refuses wrong candidate digest");
                pending->digest=saved;
                uint8_t* const candidate=generation_map(base,new_slot);
                const uint8_t first=candidate[0];
                candidate[0]=uint8_t(first==0 ? 1 : 0);
                check(!service_rank_reservation(base,snapshot,rank,rank_mask,true,true),
                      "reservation refuses changed candidate map bytes");
                candidate[0]=first;
            }
            check(service_rank_reservation(base,snapshot,rank,rank_mask,true,
                device.issued[old_slot] && device.event_ready[old_slot] &&
                device.binding[old_slot].epoch==cycle &&
                device.binding[old_slot].digest==digest),
                "production rank publishes its own RESERVED ACK");
        }
        check(helper0.tick() && control->state==uint32_t(PlacementState::COPYING),
              "actual rank reservation reaches COPYING");
        for (uint32_t rank=0;rank<rank_count;++rank)
            check(gpu[rank].upload(base,ledger,*control),
                  "fake rank issues inactive upload without overwriting old ticket");
        for (uint32_t rank=0;rank<rank_count;++rank)
            check(!gpu[rank].prepared(*control,rank),
                  "incomplete upload event blocks PREPARED");
        check(peer_ticks() && helper0.tick() &&
              control->state==uint32_t(PlacementState::COPYING),
              "helpers READY alone cannot publish PREPARED");
        check(placement_snapshot(base,&snapshot), "copying snapshot");
        gpu[0].complete(new_slot);
        check(gpu[0].prepared(*control,0), "rank 0 exact event ready");
        publish_ack(base,kMaxHelpers,snapshot,AckPhase::READY);
        if (rank_count==2) {
            check(helper0.tick() && control->state==uint32_t(PlacementState::COPYING),
                  "missing second rank READY blocks PREPARED");
            gpu[1].complete(new_slot);
            check(gpu[1].prepared(*control,1), "rank 1 exact event ready");
            publish_ack(base,kMaxHelpers+1,snapshot,AckPhase::READY);
        }
        check(helper0.tick() && control->state==uint32_t(PlacementState::PREPARED),
              "all exact READY ACKs allow PREPARED");
        check(helper0.tick() && control->state==uint32_t(PlacementState::QUIESCING) &&
              generation_header(base,new_slot)->state==uint32_t(GenerationState::READY),
              "QUIESCING only after prepared second generation");
        const auto cutoff=ledger.close_admission();
        check(!ledger.all_ranks_consumed((uint32_t)cycle,cutoff.latest_ticket) &&
              helper0.tick() && peer_ticks() &&
              control->state==uint32_t(PlacementState::QUIESCING),
              "no QUIESCED ACK while parent not consumed");
        h->burst=0;
        check(ledger.host_complete(ticket.ticket_id), "parent host done");
        check(ledger.poll([&](const StepTicket&,uint32_t rank) {
              return rank==0?r3::admission::Ledger::Proof::Proven:
                             r3::admission::Ledger::Proof::Pending;
              }) && !ledger.all_ranks_consumed((uint32_t)cycle,cutoff.latest_ticket),
              "last rank's missing proof prevents early consumed ACK");
        check(ledger.poll([](const StepTicket&,uint32_t) {
              return r3::admission::Ledger::Proof::Proven;
              }) && ledger.all_ranks_consumed((uint32_t)cycle,cutoff.latest_ticket),
              "every rank and ticket in closed epoch consumed");
        for (uint32_t lane=0;lane<kLanes;++lane) {
            const uint32_t slot_id=lane%kMaxSlots;
            if ((helper_count>1 ? slot_id>=helper_count :
                        !(control->expected_ranks & (1u<<slot_id))) ||
                (lane_is_small(lane) && lane_class(lane)>=h->small_class_count)) continue;
            h->lane[lane].drain_request=cycle;
            h->lane[lane].drain_ack=cycle-1;
        }
        check(helper0.tick() && peer_ticks() &&
              control->state==uint32_t(PlacementState::QUIESCING),
              "stale helper drain ACK cannot commit");
        for (auto& lane:h->lane) lane.drain_ack=lane.drain_request;
        check(placement_snapshot(base,&snapshot), "quiesce snapshot");
        publish_ack(base,kMaxHelpers,snapshot,AckPhase::QUIESCED,cutoff.latest_ticket);
        if (rank_count==2)
            publish_ack(base,kMaxHelpers+1,snapshot,AckPhase::QUIESCED,cutoff.latest_ticket);
        check(peer_ticks() && helper0.tick() &&
              control->state==uint32_t(PlacementState::COMMIT) &&
              control->active_epoch==cycle+1 &&
              control->active_slot==new_slot, "all drain proofs commit a truly swapped generation");
        check(generation_header(base,new_slot)->state==uint32_t(GenerationState::READY),
              "new generation not advertised ACTIVE before helper marks it");
        check(helper0.tick() && generation_header(base,new_slot)->state==
              uint32_t(GenerationState::ACTIVE), "new generation marked ACTIVE during COMMIT");
        check(placement_snapshot(base,&snapshot), "committed snapshot");
        publish_ack(base,kMaxHelpers,snapshot,AckPhase::COMMITTED,cutoff.latest_ticket);
        check(peer_ticks(),"other helpers acknowledge commit");
        if (rank_count==2)
            check(helper0.tick() &&
                  control->state==uint32_t(PlacementState::COMMIT),
                  "missing rank COMMITTED ACK blocks RETIRING");
        if (rank_count==2)
            publish_ack(base,kMaxHelpers+1,snapshot,AckPhase::COMMITTED,cutoff.latest_ticket);
        check(helper0.tick() && control->state==uint32_t(PlacementState::RETIRING),
              "all rank COMMITTED ACKs enter RETIRING");
        check(rank_retirement_proved(*control,(uint32_t)cycle,old_slot,
                true,true,true,!ledger.generation_referenced(old_slot),true,true) &&
              !rank_retirement_proved(*control,(uint32_t)cycle,old_slot,
                false,true,true,true,true,true) &&
              !rank_retirement_proved(*control,(uint32_t)cycle,old_slot,
                true,false,true,true,true,true) &&
              !ledger.can_replace_generation(old_slot),
              "retirement requires consumed proof and drain; old slot upload still refused");
        check(peer_ticks() && helper0.tick(), "helpers acknowledge retirement");
        check(placement_snapshot(base,&snapshot), "retiring snapshot");
        publish_ack(base,kMaxHelpers,snapshot,AckPhase::RETIRED,cutoff.latest_ticket);
        if (rank_count==2)
            check(helper0.tick() && control->state==uint32_t(PlacementState::RETIRING),
                  "missing rank RETIRED ACK blocks retirement");
        if (rank_count==2)
            publish_ack(base,kMaxHelpers+1,snapshot,AckPhase::RETIRED,cutoff.latest_ticket);
        check(helper0.tick() && control->state==uint32_t(PlacementState::ACTIVE) &&
              generation_header(base,old_slot)->state==uint32_t(GenerationState::RETIRED),
              "all rank RETIRED ACKs finish cycle and retire old header");
        std::vector<uint8_t> active_bits;
        check(r3::owner_generation::build_bitmap(base,new_slot,cycle+1,digest,&active_bits) &&
              active_bits == gpu[0].bits[new_slot] &&
              (rank_count==1 || active_bits == gpu[1].bits[new_slot]),
              "committed ACTIVE bitmap remains exact-event verifiable on both ranks");
        check(control->active_epoch==cycle+1 && control->active_slot==new_slot &&
              control->active_digest==digest &&
              std::equal(owners.begin(),owners.end(),generation_map(base,new_slot)),
              "full no-op cycle returns ACTIVE, slots swapped, bytes unchanged");
        h->burst=1;
        for (auto& lane:h->lane) lane.drain_request=0;
        check(helper0.tick() && helper0.completed()==cycle-1,
              "cleared drain requests do NOT certify ledger reopen");
        check(ledger.reopen((uint32_t)cycle+1,new_slot) &&
              ledger.can_replace_generation(old_slot),
              "only drained ledger can reopen on next epoch");
        check(helper0.tick() && helper0.completed()==cycle-1,
              "gate-open without release-published transaction ACK is insufficient");
        check(placement_snapshot(base,&snapshot),"reopen snapshot");
        publish_reopen_ack(base,snapshot);
        if (cycle==1) {
            PlacementAck* ack=placement_ack(base,kMaxHelpers);
            const uint64_t correct_txn=ack->transaction_id;
            __atomic_store_n(&ack->transaction_id,correct_txn-1u,__ATOMIC_RELAXED);
            check(helper0.tick() && helper0.completed()==cycle-1,
                  "stale reopen transaction cannot stage a new generation");
            __atomic_store_n(&ack->transaction_id,correct_txn,__ATOMIC_RELAXED);
            const uint64_t correct_epoch=ack->epoch;
            __atomic_store_n(&ack->epoch,correct_epoch+1u,__ATOMIC_RELAXED);
            check(helper0.tick() && helper0.completed()==cycle-1,
                  "wrong reopen epoch cannot stage");
            __atomic_store_n(&ack->epoch,correct_epoch,__ATOMIC_RELAXED);
            const uint64_t correct_digest=ack->digest;
            __atomic_store_n(&ack->digest,correct_digest^1u,__ATOMIC_RELAXED);
            check(helper0.tick() && helper0.completed()==cycle-1,
                  "wrong reopen digest cannot stage");
            __atomic_store_n(&ack->digest,correct_digest,__ATOMIC_RELAXED);
            const uint32_t correct_slot=ack->generation_slot;
            __atomic_store_n(&ack->generation_slot,1u-correct_slot,__ATOMIC_RELAXED);
            check(helper0.tick() && helper0.completed()==cycle-1,
                  "wrong reopen generation cannot stage");
            __atomic_store_n(&ack->generation_slot,correct_slot,__ATOMIC_RELAXED);
        }
        check(helper0.tick() && helper0.completed()==cycle,
              "exact post-ledger ACK completes transaction and permits staging");
        slot=new_slot;
        clock.elapse(100);
    }
    check(helper0.tick() && control->state==uint32_t(PlacementState::PROPOSED),
          "next full cycle starts a distinct transaction");
    check(helper0.tick() && control->state==uint32_t(PlacementState::RESERVING),
          "timed-out transaction enters reservation");
    clock.elapse(61'000'000'000ULL);
    check(helper0.tick() && control->state==uint32_t(PlacementState::ACTIVE) &&
          !control->fault_code &&
          placement_journal(base)->entry[control->journal_index].fault==
              uint32_t(FaultCode::READY_TIMEOUT),
          "reservation timeout is recoverable and leaves all helpers serving");
    check(control->active_epoch==cycles+1 &&
          control->active_slot==slot,"precommit timeout never switches active epoch or slot");
    munmap(base,layout.bytes);
}
int main() {
    V23Cycle_UnreadyAndAbortPreserveOldGeneration();
    V23Cycle_PrecommitAbortDuringLiveServing();
    V23Cycle_StalledParkFaultsAndKeepsAdmissionClosed();
    V23Bootstrap_UnsealedTickReportsInvalidMap();
    V23Cycle_SamePlacementAndSharedProtocol(1,2);
    V23Cycle_SamePlacementAndSharedProtocol(3,1);
    V23Epoch_ForwardOnlyAbortRetry();
    V23Ack_OldTxnDelayedDuplicateLost();
    V23Ack_CorruptCurrentAndFault();
    V23Fault_BeforeAndAfterCommit();
    V23Slots_ReserveCopyRetireFaults();
    V23Policy_HeatEqualCostAndSwap();
    std::puts("r3_moe_epoch_test: ok");
}
