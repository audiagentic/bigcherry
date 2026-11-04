#include "worker_epoch.h"
#include "engine_rank_coordinator.h"
#include "worker_bootstrap.h"
#include "admission_ledger.h"
#include "owner_generation.h"
#include <sys/mman.h>
#include "worker_residency.h"
#include "worker_move_slots.h"
#include "worker_slot_pool.h"
#include "worker_generation_tables.h"
#include <map>
#include <new>
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
    void close_actor(const PlacementControl& c,uint32_t actor,AckPhase phase) {
        auto& x=a.at(actor);
        x={}; x.transaction_id=c.transaction_id; x.epoch=c.active_epoch;
        x.digest=c.active_digest; x.generation_slot=c.active_slot;
        x.phase=uint32_t(phase);
    }
    void close(const PlacementControl& c,bool committed) {
        for (uint32_t i=0;i<kMaxHelpers;++i)
            if (c.expected_helpers & (1u<<i)) close_actor(c,i,AckPhase::CLEANED);
        for (uint32_t i=0;i<kMaxEngineRanks;++i)
            if (c.expected_ranks & (1u<<i))
                close_actor(c,kMaxHelpers+i,
                            committed && i==0 ? AckPhase::REOPENED : AckPhase::CLEANED);
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
    check(!c.can_propose(),"retirement alone cannot authorize next proposal");
    t.close(c.control(),true);
    check(c.can_propose(),"reopen plus all helper cleanups authorize proposal");
}
static void V23Cleanup_DeadlineAndExactAcks() {
    for (bool committed : {false,true}) {
        for (bool missing_rank : {false,true}) {
            // Aborted transactions need all rank CLEANED ACKs; committed
            // transactions need REOPENED and every helper CLEANED ACK.
            if (committed && missing_rank) continue;
            FakeClock clock;
            FakeTransport transport;
            Deadlines d{};
            d.copy_ns=200; d.retire_ns=100;
            Controller c(transport,clock,101,3,2,d);
            if (committed) {
                check(c.begin_epoch_cycle(),"cleanup test committed begin");
                complete(c,transport); // genuine COMMIT and RETIRING through Controller
            } else {
                check(c.propose(202) && c.start_reserving() &&
                      c.fail(FaultCode::CAPACITY),
                      "cleanup test abort before committed generation");
                transport.close(c.control(),false);
            }
            const uint32_t actor=missing_rank ? kMaxHelpers+1 : 2;
            const uint32_t expected_actor=missing_rank ?
                actor_id(ActorKind::ENGINE_RANK,1) :
                actor_id(ActorKind::HELPER,2);
            const PlacementAck exact=transport.a[actor];
            check(c.poll_cleanup() && c.can_propose(),
                  "only all exact cleanup ACKs allow subsequent transaction");
            // Duplicating a valid CLEANED does not change its proof.
            transport.close_actor(c.control(),actor,AckPhase::CLEANED);
            check(c.poll_cleanup() && c.can_propose(),
                  "duplicate CLEANED remains exact and harmless");
            // The controller must ignore a delayed old transaction, wrong
            // phase, wrong active identity, and even stale fault publications.
            for (uint32_t bad=0;bad<6;++bad) {
                transport.a[actor]=exact;
                auto& a=transport.a[actor];
                if (bad==0) --a.transaction_id;
                if (bad==1) a.phase=uint32_t(AckPhase::RETIRED);
                if (bad==2) ++a.epoch;
                if (bad==3) a.digest^=1u;
                if (bad==4) a.generation_slot=1u-a.generation_slot;
                if (bad==5) { --a.transaction_id; a.fault_code=uint32_t(FaultCode::DEVICE); }
                check(!c.can_propose() && c.poll_cleanup() && !c.failed_closed(),
                      "stale, wrong-phase, mismatched CLEANED not counted as proof or fault");
            }
            transport.a[actor]=exact;
            check(c.can_propose(),"restored exact cleanup identity");
            transport.a[actor]={}; // lost (or dead participant)
            check(!c.can_propose(),"lost CLEANED blocks proposals");
            clock.elapse((committed ? d.retire_ns : d.copy_ns)-1u);
            check(c.poll_cleanup() && !c.failed_closed() &&
                  !c.can_propose(),"cleanup waits until absolute deadline");
            clock.elapse(1u);
            check(!c.poll_cleanup() && c.failed_closed() &&
                  c.control().state==uint32_t(PlacementState::ACTIVE) &&
                  c.control().active_epoch==(committed ? 2u : 1u) &&
                  c.control().active_slot==(committed ? 1u : 0u) &&
                  c.control().fault_code==0 &&
                  c.control().fault_actor==expected_actor &&
                  c.journal().entry[c.control().journal_index].cleanup_fault==
                      uint32_t(FaultCode::CLEANUP_TIMEOUT) &&
                  c.journal().entry[c.control().journal_index].cleanup_actor==expected_actor &&
                  (!committed ? c.journal().entry[c.control().journal_index].fault==
                       uint32_t(FaultCode::CAPACITY) : true),
                  "expiry records missing participant and retains committed/old ACTIVE generation");
            transport.a[actor]=exact;
            check(!c.can_propose() && !c.poll_cleanup(),
                  "late cleanup cannot undo terminal placement halt");
        }
    }
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
    check(!c.can_propose(),"abort publication is not participant cleanup");
    tr.close(c.control(),false);
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
    check(!c.can_propose(),"post-success abort must await cleanup");
    tr.close(c.control(),false);
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
    check(!c.can_propose(),"helper death waits for abort cleanup");
    tr.close(c.control(),false);
    check(c.propose(234) && c.start_reserving(),"retry after helper death");
    tr.all(c.control(),AckPhase::RESERVED);
    check(c.poll()==Poll::ADVANCED && c.fail(FaultCode::UPLOAD,0),"mid-copy error precommit");
    check(c.control().active_epoch==1,"upload fault never advances active");
    tr.close(c.control(),false);
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

// Injectable fake GPU ranks, running the production EngineRankCoordinator.
// Every rank phase is acknowledged BY that coordinator. The fake implements
// the same ledger closure, bitmap upload, event-query and lane-identity checks.
struct FakeRankOperations final : RankOperations {
    void* map;
    r3::admission::Ledger& ledger;
    FakeCycleUpload* gpu;
    uint32_t rank_count;
    bool closed=false, faulted=false;
    int abort_event_error_rank=-1; // simulate hipEventQuery returning a real device error
    uint64_t closure=0, closed_cutoff=0;
    uint32_t previous_epoch=0;
    FakeRankOperations(void* m, r3::admission::Ledger& l,
                       FakeCycleUpload* d, uint32_t n)
        : map(m),ledger(l),gpu(d),rank_count(n) {}
    r3::admission::Ledger::Snapshot prepare_close() {
        // The fake advances the clocked admission gate to CLOSED before
        // injecting per-rank event proofs, just like a real drain waiter.
        return ledger.close_admission();
    }
    uint32_t actual_ranks() override { return rank_count; }
    bool registered(uint32_t rank) override { return rank < rank_count; }
    RankResult upload(uint32_t rank,uint32_t slot,uint64_t epoch,uint64_t digest) override {
        PlacementControl c{};
        if (rank >= rank_count || !placement_snapshot(map,&c) ||
            c.pending_slot != slot || c.pending_epoch != epoch ||
            c.pending_digest != digest) return RankResult::Pending;
        return gpu[rank].upload(map,ledger,c) ? RankResult::Ready : RankResult::Pending;
    }
    RankResult ready(uint32_t rank,uint32_t slot,uint64_t epoch,uint64_t digest) override {
        if (rank >= rank_count || slot > 1) return RankResult::Pending;
        const auto& d=gpu[rank];
        return d.issued[slot] && d.event_ready[slot] &&
               d.binding[slot].epoch == epoch && d.binding[slot].digest == digest ?
               RankResult::Ready : RankResult::Pending;
    }
    RankResult abort_upload_done(uint32_t rank,uint32_t slot,uint64_t epoch,
                                 uint64_t digest) override {
        if (rank>=rank_count || slot>1) return RankResult::DeviceError;
        const auto& d=gpu[rank];
        if (!d.issued[slot] || d.binding[slot].epoch!=epoch ||
            d.binding[slot].digest!=digest) return RankResult::DeviceError;
        if (abort_event_error_rank==int(rank)) return RankResult::DeviceError;
        return d.event_ready[slot] ? RankResult::Ready : RankResult::Pending;
    }
    bool exact_binding(uint32_t rank,uint32_t slot,uint64_t epoch,uint64_t digest) override {
        return rank < rank_count && slot < 2 &&
               gpu[rank].issued[slot] &&
               gpu[rank].binding[slot].epoch == epoch &&
               gpu[rank].binding[slot].digest == digest;
    }
    bool lanes_quiesced() const {
        const auto* h=static_cast<Header*>(map);
        for (uint32_t lane=0; lane<kLanes; ++lane) {
            const uint32_t slot=lane%kMaxSlots;
            const bool served=(h->helper_count > 1 ? slot<h->helper_count :
                                  (1u<<slot & ((1u<<rank_count)-1u)) != 0) &&
                               (!lane_is_small(lane) || lane_class(lane)<h->small_class_count);
            if (!lane_quiesced(h,lane,closure,served)) return false;
        }
        return true;
    }
    bool close_and_wait(uint64_t deadline,uint64_t* cutoff,uint32_t* old_epoch) override {
        if (closed || !deadline || faulted) return false;
        const auto snapshot=ledger.close_admission();
        PlacementControl c{};
        if (!placement_snapshot(map,&c)) return false;
        closure=c.transaction_id;
        previous_epoch=snapshot.old_epoch;
        closed_cutoff=snapshot.latest_ticket;
        closed=true;
        if (!ledger.drained(snapshot.old_epoch,snapshot.latest_ticket) ||
            !ledger.all_ranks_consumed(snapshot.old_epoch,snapshot.latest_ticket) ||
            !lanes_quiesced()) return false;
        *cutoff=snapshot.latest_ticket;
        *old_epoch=snapshot.old_epoch;
        return true;
    }
    bool consumed() override {
        return closed && ledger.all_ranks_consumed(previous_epoch,closed_cutoff);
    }
    bool drain_verified() override { return closed && lanes_quiesced(); }
    bool helper_quiesced() override { return closed && lanes_quiesced(); }
    bool generation_referenced(uint32_t slot) override {
        return ledger.generation_referenced(slot);
    }
    bool can_replace(uint32_t slot) override { return ledger.can_replace_generation(slot); }
    bool reopen(uint32_t epoch,uint32_t slot) override {
        if (!closed || !lanes_quiesced() || !ledger.reopen(epoch,(uint8_t)slot))
            return false;
        // Simulated engine drain-word clearing after all ranks and helper lanes.
        auto* h=static_cast<Header*>(map);
        for (auto& lane:h->lane) lane.drain_request=0;
        closed=false;
        return true;
    }
    bool stopped() override { return faulted || ledger.stopped(); }
    void fatal() override { faulted=true; ledger.fault(); }
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
    check(!c.can_propose(),"aborted upload requires participant cleanup");
    tr.close(c.control(),false);
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
    struct StuckAbort : HelperMigration {
        uint64_t pending=0;
        uint32_t attempts=0;
        FaultCode reserve(const void*,const PlacementControl&) override {
            return FaultCode::NONE;
        }
        HelperCopy copy() override { return HelperCopy::READY; }
        bool commit(const PlacementControl&) override { return true; }
        bool reclaim(const void*,const PlacementControl&) override { return true; }
        bool abort() override { ++attempts; return false; } // pending upload drain
        uint64_t transaction() const override { return pending; }
    } stuck;
    EpochCycle leader(map,0,clock,std::chrono::nanoseconds{100});
    EpochCycle peer1(map,1,clock,std::chrono::nanoseconds{100});
    EpochCycle peer2(map,2,clock,std::chrono::nanoseconds{100},Deadlines{},&stuck);
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
    check(!leader.proposal_controller()->can_propose(),"manual rank fault needs cleanup");
    publish_cleanup_ack(map,kMaxHelpers,*c);
    check(leader.proposal_controller()->can_propose(),"rank abort cleanup gates retry");
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
    publish_cleanup_ack(map,kMaxHelpers,*c);
    clock.elapse(100);
    check(leader.tick() && c->transaction_id==snapshot.transaction_id+2 &&
          c->state==uint32_t(PlacementState::PROPOSED) &&
          c->active_epoch==1 && h->helper_fault==0,
          "retry after invalid candidate gets fresh transaction and N");
    // Third abort: one peer cannot complete a queued migration upload.
    // It must keep serving and withhold CLEANED until the deadline, rather
    // than exiting with helper_fault because abort() is still pending.
    const uint64_t stuck_transaction=c->transaction_id;
    check(leader.proposal_controller()->fail(FaultCode::UPLOAD),
          "stuck upload precommit forward abort");
    stuck.pending=stuck_transaction;
    check(peer1.tick() && peer2.tick() && leader.tick() &&
          stuck.attempts==1 && !peer2.placement_halted() &&
          !leader.proposal_controller()->can_propose() && !h->helper_fault,
          "incomplete abort retries without faking CLEANED or killing helper");
    publish_cleanup_ack(map,kMaxHelpers,*c);
    clock.elapse(Deadlines{}.copy_ns-1u);
    check(peer2.tick() && leader.tick() && stuck.attempts>=2 &&
          !peer2.placement_halted() && !h->helper_fault,
          "incomplete helper abort bounded by copy-derived cleanup deadline");
    clock.elapse(1u);
    check(peer2.tick() && leader.tick() && peer2.tick() &&
          leader.proposal_controller()->failed_closed() &&
          leader.placement_halted() && peer2.placement_halted() &&
          peer2.placement_fault()==FaultCode::CLEANUP_TIMEOUT &&
          c->active_epoch==1 && c->active_slot==0 && c->fault_code==0 &&
          placement_journal(map)->entry[c->journal_index].cleanup_fault==
              uint32_t(FaultCode::CLEANUP_TIMEOUT) &&
          placement_journal(map)->entry[c->journal_index].cleanup_actor==
              actor_id(ActorKind::HELPER,2) &&
          !h->helper_fault && !ledger.stopped(),
          "missing helper CLEANED yields safe placement-only terminal condition");
    StepTicket after_stuck{};
    check(ledger.admit(1,&after_stuck)==r3::admission::Ledger::Result::Ok &&
          after_stuck.placement_epoch==1 && after_stuck.generation_slot==0 &&
          ledger.bind(after_stuck.ticket_id,9) &&
          ledger.host_complete(after_stuck.ticket_id) &&
          ledger.poll([](const StepTicket&,uint32_t) {
              return r3::admission::Ledger::Proof::Proven;
          }),
          "live generation keeps serving after abort cleanup deadline");
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
    uint32_t helper_count,uint32_t rank_count,bool lose_reopen_ack=false) {
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
    EngineRankCoordinator coordinator;
    FakeRankOperations rank_ops(base,ledger,gpu,rank_count);
    uint32_t slot=0;
    const uint64_t cycles=lose_reopen_ack ? 1 : 1000;
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
        }
        check(coordinator.step(base,snapshot,rank_ops),
              "production coordinator dispatches RESERVED for every rank");
        for (uint32_t rank=0;rank<rank_count;++rank)
            check(read_shared_ack(base,kMaxHelpers+rank).phase==uint32_t(AckPhase::RESERVED),
                  "engine rank RESERVED publication comes from coordinator");
        check(helper0.tick() && control->state==uint32_t(PlacementState::COPYING),
              "actual rank reservation reaches COPYING");
        check(placement_snapshot(base,&snapshot) &&
              coordinator.step(base,snapshot,rank_ops),
              "production coordinator issues rank inactive uploads");
        for (uint32_t rank=0;rank<rank_count;++rank)
            check(!gpu[rank].prepared(*control,rank),
                  "incomplete upload event blocks PREPARED");
        check(peer_ticks() && helper0.tick() &&
              control->state==uint32_t(PlacementState::COPYING),
              "helpers READY alone cannot publish PREPARED");
        check(placement_snapshot(base,&snapshot), "copying snapshot");
        gpu[0].complete(new_slot);
        check(gpu[0].prepared(*control,0) &&
              coordinator.step(base,snapshot,rank_ops), "rank 0 exact event ready through coordinator");
        check(read_shared_ack(base,kMaxHelpers).phase==uint32_t(AckPhase::READY),
              "coordinator publishes rank 0 READY");
        if (rank_count==2) {
            check(helper0.tick() && control->state==uint32_t(PlacementState::COPYING),
                  "missing second rank READY blocks PREPARED");
            gpu[1].complete(new_slot);
            check(gpu[1].prepared(*control,1) &&
                  coordinator.step(base,snapshot,rank_ops), "rank 1 exact event ready through coordinator");
            check(read_shared_ack(base,kMaxHelpers+1).phase==uint32_t(AckPhase::READY),
                  "coordinator publishes rank 1 READY");
        }
        check(helper0.tick() && control->state==uint32_t(PlacementState::PREPARED),
              "all exact READY ACKs allow PREPARED");
        check(helper0.tick() && control->state==uint32_t(PlacementState::QUIESCING) &&
              generation_header(base,new_slot)->state==uint32_t(GenerationState::READY),
              "QUIESCING only after prepared second generation");
        // The fake closes the ledger before it can inject event proofs;
        // close_and_wait in the real coordinator verifies the same snapshot.
        const auto cutoff=rank_ops.prepare_close();
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
        check(coordinator.step(base,snapshot,rank_ops),
              "coordinator closes admission and waits for fake event/lane drain");
        for (uint32_t rank=0;rank<rank_count;++rank)
            check(read_shared_ack(base,kMaxHelpers+rank).phase==uint32_t(AckPhase::QUIESCED),
                  "production coordinator publishes QUIESCED after real proofs");
        check(peer_ticks() && helper0.tick() &&
              control->state==uint32_t(PlacementState::COMMIT) &&
              control->active_epoch==cycle+1 &&
              control->active_slot==new_slot, "all drain proofs commit a truly swapped generation");
        check(generation_header(base,new_slot)->state==uint32_t(GenerationState::READY),
              "new generation not advertised ACTIVE before helper marks it");
        check(helper0.tick() && generation_header(base,new_slot)->state==
              uint32_t(GenerationState::ACTIVE), "new generation marked ACTIVE during COMMIT");
        check(placement_snapshot(base,&snapshot), "committed snapshot");
        check(coordinator.step(base,snapshot,rank_ops),
              "production coordinator observes helper ACTIVE header then publishes COMMITTED");
        for (uint32_t rank=0;rank<rank_count;++rank)
            check(read_shared_ack(base,kMaxHelpers+rank).phase==uint32_t(AckPhase::COMMITTED),
                  "engine rank COMMITTED ACK originates in coordinator");
        check(peer_ticks(),"other helpers acknowledge commit");
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
        check(coordinator.step(base,snapshot,rank_ops),
              "production coordinator proves retired rank identities and consumed cutoff");
        for (uint32_t rank=0;rank<rank_count;++rank)
            check(read_shared_ack(base,kMaxHelpers+rank).phase==uint32_t(AckPhase::RETIRED),
                  "production coordinator publishes RETIRED");
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
        check(helper0.tick() && helper0.completed()==cycle-1,
              "engine has not reopened admission or published REOPENED");
        check(placement_snapshot(base,&snapshot) &&
              coordinator.step(base,snapshot,rank_ops) &&
              ledger.can_replace_generation(old_slot) &&
              read_shared_ack(base,kMaxHelpers).phase==uint32_t(AckPhase::REOPENED),
              "coordinator reopens guarded ledger and release-publishes REOPENED");
        if (lose_reopen_ack) {
            auto* ack=placement_ack(base,kMaxHelpers);
            __atomic_store_n(&ack->phase,uint32_t(AckPhase::NONE),__ATOMIC_RELEASE);
            clock.elapse(Deadlines{}.retire_ns-1u);
            check(helper0.tick() && peer_ticks() && helper0.completed()==cycle-1,
                  "all helpers remain retryable before REOPENED deadline");
            clock.elapse(1);
            // Drive NON-leaders through their production reopen-timeout branch
            // first: they own no Controller, and must not dereference one.
            check(peer_ticks() && helper0.tick() && helper0.tick() &&
                  std::all_of(peers.begin(),peers.end(),
                      [](const auto& p) {
                          return p->placement_halted() &&
                                 p->placement_fault()==FaultCode::REOPEN_TIMEOUT;
                      }) &&
                  helper0.placement_halted() &&
                  helper0.placement_fault()==FaultCode::REOPEN_TIMEOUT &&
                  control->state==uint32_t(PlacementState::ACTIVE) &&
                  control->active_epoch==cycle+1 && control->active_slot==new_slot &&
                  !control->fault_code && !h->helper_fault && !ledger.stopped() &&
                  placement_journal(base)->entry[control->journal_index].fault==
                      uint32_t(FaultCode::REOPEN_TIMEOUT),
                  "missing ACK publishes terminal placement-only fault after deadline");
            const uint64_t frozen_transaction=control->transaction_id;
            clock.elapse(Deadlines{}.retire_ns);
            check(peer_ticks() && helper0.tick() &&
                  control->transaction_id==frozen_transaction &&
                  helper0.completed()==cycle-1 && !load(&h->helper_fault) &&
                  !ledger.stopped(),
                  "placement stays disabled while worker and committed serving stay alive");
            StepTicket still_serving{};
            check(ledger.admit(1,&still_serving)==r3::admission::Ledger::Result::Ok &&
                  still_serving.placement_epoch==cycle+1 &&
                  still_serving.generation_slot==new_slot &&
                  ledger.bind(still_serving.ticket_id,cycle+1000) &&
                  ledger.host_complete(still_serving.ticket_id) &&
                  ledger.poll([](const StepTicket&,uint32_t) {
                      return r3::admission::Ledger::Proof::Proven;
                  }),
                  "lost REOPENED on three helpers still serves committed requests");
            munmap(base,layout.bytes);
            return;
        }
        if (cycle<=2) {
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
        check(peer_ticks() && helper0.proposal_controller()->can_propose(),
              "every helper cleanup completed before new proposal");
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

// Live-move tests use production Pool/Ledger/Participant/EpochCycle/Controller/
// EngineRankCoordinator. Only device HIP allocations, copy events and rank
// operations are faked; the controller proposal is the real entry point.
struct MoveAllocator {
    std::map<void*,uint64_t> live;
    uint64_t bytes=0, peak=0;
    bool allocate(uint64_t n,void** out) {
        if (!n || n>SIZE_MAX) return false;
        *out=new(std::nothrow) uint8_t[size_t(n)];
        if (!*out) return false;
        live.emplace(*out,n);
        bytes+=n; peak=std::max(peak,bytes);
        return true;
    }
    void release(void* p) {
        auto it=live.find(p);
        check(it!=live.end(),"fake allocator owns each released physical arena");
        bytes-=it->second;
        live.erase(it); delete[] static_cast<uint8_t*>(p);
    }
};
struct MoveOperations : r3::worker_moves::Operations {
    r3::residency::Ledger& ledger;
    uint32_t active=0, layers=0, experts=0;
    std::vector<int32_t> locals[2], physical[2];
    bool ready[2]{true,false};
    bool busy=false, event_ready=false;
    uint32_t pending_expert=UINT32_MAX, upload_count=0, events_seen=0, banks_built=0;
    MoveOperations(r3::residency::Ledger& l,uint32_t ls,uint32_t es,
                   const std::vector<int32_t>& initial)
        : ledger(l),layers(ls),experts(es) {
        locals[0]=initial;
        physical[0].assign(initial.size(),-1);
        for (const auto& s: ledger.slots())
            if (s.state==r3::residency::State::ACTIVE)
                physical[0][s.expert]=int32_t(s.id);
    }
    bool start_upload(uint32_t expert,uint32_t slot,uint32_t local) override {
        if (busy || slot>=ledger.slots().size() ||
            ledger.slots()[slot].state!=r3::residency::State::COPYING ||
            ledger.slots()[slot].expert!=expert || local>=experts)
            return false;
        busy=true;event_ready=false;
        pending_expert=expert;
        ++upload_count;
        return true;
    }
    r3::worker_moves::Event upload_event() override {
        if (!busy) return r3::worker_moves::Event::FAILED;
        if (!event_ready) return r3::worker_moves::Event::PENDING;
        busy=false;event_ready=false; pending_expert=UINT32_MAX;
        ++events_seen;
        return r3::worker_moves::Event::COMPLETE;
    }
    bool build_bank(uint32_t generation,const std::vector<int32_t>& values) override {
        if (generation==active || generation>1 || busy ||
            values.size()!=size_t(layers)*experts || upload_count!=events_seen)
            return false; // guard: no active-bank writes or early READY
        std::vector<int32_t> proposed(values.size(),-1);
        for (size_t i=0;i<values.size();++i) {
            if (values[i]<0) continue;
            for (const auto& slot:ledger.slots())
                if (slot.expert==i &&
                    (slot.state==r3::residency::State::ACTIVE ||
                     slot.state==r3::residency::State::READY)) {
                    proposed[i]=int32_t(slot.id);
                    break;
                }
            if (proposed[i]<0) return false;
        }
        locals[generation]=values;physical[generation]=proposed;
        ready[generation]=true;++banks_built;
        return true;
    }
    bool drain_upload() override {
        busy=false;event_ready=false;pending_expert=UINT32_MAX;
        return true;
    }
    void forget(uint32_t) override {}
    void activate_bank(uint32_t generation) override {
        check(generation!=active && generation<2 && ready[generation],
              "activate built inactive bank");
        active=generation;
    }
    int32_t preferred_local(uint32_t expert) const override {
        return expert<layers*experts ? int32_t(expert%experts) : -1;
    }
    void complete() { check(busy,"fake event is for outstanding expert");event_ready=true; }
    int32_t serving(uint32_t gen,uint32_t expert) const {
        if (gen>1 || !ready[gen] || expert>=physical[gen].size()) return -1;
        return physical[gen][expert];
    }
};

static void V23Move_PhysicalSlotLimitIndependentOfByteBudget() {
    MoveAllocator allocator;
    // The physical slot count is fixed even if budget leaves extra unused
    // bytes. A fourth ledger ID would have no physical GPU address.
    r3::worker_slots::Pool<MoveAllocator> pool(4*256,256,2,allocator);
    uint32_t a=UINT32_MAX,b=UINT32_MAX,c=UINT32_MAX,d=UINT32_MAX;
    check(pool.bootstrap(0,256,1,&a)==r3::residency::Result::OK &&
          pool.bootstrap(1,256,1,&b)==r3::residency::Result::OK &&
          pool.ledger().reserve(2,256,2,true,&c)==r3::residency::Result::OK &&
          c==2 && pool.address(c) &&
          pool.ledger().reserve(3,256,2,true,&d)==
              r3::residency::Result::DEFERRED_CAPACITY &&
          pool.address(3)==nullptr,
          "physical max slots rejects capacity despite unused byte budget");
}

static void V23Move_ChangedMapThroughProduction(uint32_t helpers,uint32_t ranks) {
    check((helpers==1 && ranks==2) ||
          ((helpers==2 || helpers==3) && ranks==1),
          "single-helper negative and multi-helper migration shapes");
    const LayoutShape shape{128,helpers,3,8,128,64};
    Layout layout{};
    check(layout_for(shape,2,slots_for(helpers),&layout),"move layout");
    void* map=mmap(nullptr,layout.bytes,PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    check(map!=MAP_FAILED && init(map,2,shape,helpers),"move segment");
    auto* h=static_cast<Header*>(map);
    h->helper_count=helpers;
    auto* control=placement_control(map);
    control->expected_ranks=(1u<<ranks)-1u;
    struct Selected {std::vector<std::pair<uint32_t,uint32_t>> expert;};
    std::vector<Selected> selected(helpers);
    // The tail has a genuinely smaller physical budget/arena. The second
    // layer's third expert stays local in both arrangements.
    selected[0].expert={{0,0},{1,0}};
    if (helpers>1) selected[1].expert={{0,1},{1,1}};
    // The tail begins with zero residents, but has one expert of budget:
    // its first incoming expert must use an arena allocated at startup.
    if (helpers==3) selected[2].expert={};
    for (uint32_t i=0;i<helpers;++i)
        check(publish_owned(map,selected[i],i),"move bootstrap owners");
    for (uint32_t i=0;i<helpers;++i)
        check(mark_bootstrap_filtered(map,i),"move bootstrap filtered");
    SharedTransport bootstrap(map);
    check(bootstrap.seal_bootstrap(),"move bootstrap seal");
    const uint64_t original_digest=generation_digest(map,0);
    check(original_digest==control->active_digest,"move sealed active digest");

    constexpr uint64_t stride=256; // fake payload, not production slot constant
    std::vector<std::unique_ptr<MoveAllocator>> alloc;
    using MovePool=r3::worker_slots::Pool<MoveAllocator>;
    std::vector<std::unique_ptr<MovePool>> pools;
    std::vector<std::unique_ptr<MoveOperations>> ops;
    std::vector<std::unique_ptr<r3::worker_moves::Participant>> migration;
    std::vector<std::unique_ptr<EpochCycle>> participants;
    FakeClock clock;
    for (uint32_t helper=0;helper<helpers;++helper) {
        alloc.emplace_back(std::make_unique<MoveAllocator>());
        const auto capacity=uint32_t(selected[helper].expert.size())+
            (helpers==3 && helper==2 ? 1u : 0u);
        pools.emplace_back(std::make_unique<MovePool>(capacity*stride,stride,
                                                     capacity,*alloc.back()));
        check(pools.back()->prepare() && pools.back()->address(0) &&
              pools.back()->address(pools.back()->staging_id()),
              "budget-wide arena and spare exist before bootstrap, including zero owners");
        std::vector<int32_t> locals(size_t(shape.experts)*2,-1);
        for (auto [l,e]:selected[helper].expert) {
            const uint32_t id=l*shape.experts+e;
            uint32_t physical=UINT32_MAX;
            locals[id]=int32_t(e);
            check(pools.back()->bootstrap(id,stride,1,&physical)==
                  r3::residency::Result::OK,"move bootstrap physical slot");
        }
        ops.emplace_back(std::make_unique<MoveOperations>(
            pools.back()->ledger(),2,shape.experts,locals));
        check(!ops.back()->build_bank(0,locals),
              "active pointer bank is never writable while serving N");
        migration.emplace_back(std::make_unique<r3::worker_moves::Participant>(
            pools.back()->ledger(),*ops.back(),helper,2,shape.experts,stride));
        check(migration.back()->initialize(map,0,locals),"move initialize ledger");
        participants.emplace_back(std::make_unique<EpochCycle>(
            map,helper,clock,std::chrono::nanoseconds{0},Deadlines{},migration.back().get()));
    }
    auto peers=[&]() {
        for (uint32_t i=1;i<helpers;++i)
            check(participants[i]->tick(),"move peer phase");
    };
    check(participants[0]->tick(),"move controller bootstrap");
    Controller* controller=participants[0]->proposal_controller();
    check(controller && controller->can_propose(),"move proposal uses existing controller");
    r3::admission::Ledger ledger(5);
    check(ledger.initialize(1,0),"move engine admission ledger");
    FakeCycleUpload gpu[2];
    for (uint32_t rank=0;rank<ranks;++rank) {
        check(r3::owner_generation::build_bitmap(map,0,1,original_digest,&gpu[rank].bits[0]),
              "move rank bootstrap bitmap");
        gpu[rank].binding[0]={1,original_digest};
        gpu[rank].issued[0]=gpu[rank].event_ready[0]=true;
    }
    FakeRankOperations rank_ops(map,ledger,gpu,ranks);
    EngineRankCoordinator coordinator;

    // Invalid bootstrap classification changes fail on both participants
    // before anyone can reserve or upload. Run both directions with the
    // production Controller, rank coordinator and migration participant.
    for (int direction=0; direction<2; ++direction) {
        auto* pending=generation_header(map,1);
        std::memcpy(generation_map(map,1),generation_map(map,0),h->owner_map_bytes);
        const uint32_t expert=direction ? 2u : 0u;
        const uint8_t original=generation_map(map,0)[expert];
        check((direction ? original==kOwnerLocal : original==0),
              "bootstrap local/helper negative fixture");
        generation_map(map,1)[expert]=direction ? uint8_t(0) : kOwnerLocal;
        pending->epoch=2;
        pending->digest=generation_digest(map,1);
        pending->state=uint32_t(GenerationState::STAGING);
        pending->ready_helpers=pending->ready_ranks=0;
        check(controller->propose(pending->digest),
              "Controller receives illegal-classification proposal");
        check(controller->start_reserving(),
              "invalid candidate reaches reservation for participant tests");
        check(migration[0]->reserve(map,*control)==FaultCode::CHECKSUM,
              "production HelperMigration rejects changed local classification");
        check(!placement_candidate_valid(map,*control),
              "shared validator rejects illegal local classification");
        PlacementControl snapshot{};
        check(placement_snapshot(map,&snapshot) &&
              coordinator.step(map,snapshot,rank_ops),
              "engine rank coordinator publishes invalid-map fault");
        for (uint32_t rank=0;rank<ranks;++rank)
            check(read_shared_ack(map,kMaxHelpers+rank).fault_code==
                      uint32_t(FaultCode::CHECKSUM),
                  "both engine rank reservations reject local classification");
        check(participants[0]->tick() &&
              control->state==uint32_t(PlacementState::ACTIVE) &&
              control->active_epoch==1 && control->active_digest==original_digest &&
              !controller->can_propose(),
              "epoch driver aborts invalid placement forward and blocks retry");
        peers();
        check(participants[0]->tick() && !controller->can_propose(),
              "helper abort cleaned is insufficient until rank is cleaned");
        check(placement_snapshot(map,&snapshot) &&
              coordinator.step(map,snapshot,rank_ops) &&
              controller->can_propose(),
              "rank abort cleanup closes rejected transaction");
    }
    // The single-helper shape can only run same-map cycles. Its impossible
    // local/helper moves were covered negatively above.
    if (helpers==1) {
        check(munmap(map,layout.bytes)==0,"one-helper negative unmap");
        return;
    }
    const uint64_t fixed_bytes[3] = {
        alloc[0]->bytes,
        helpers>1?alloc[1]->bytes:0,
        helpers==3?alloc[2]->bytes:0
    };

    // More incoming experts than physically free slots is a current-phase
    // CAPACITY fault. The initially selected physical slots are still held
    // by ACTIVE(N), including those scheduled to leave this very transaction.
    {
        auto* pending=generation_header(map,1);
        std::memcpy(generation_map(map,1),generation_map(map,0),h->owner_map_bytes);
        // One incoming expert at a full-budget destination, no departure:
        // the staging slot exists, but postcommit residency is illegal.
        generation_map(map,1)[0]=1;
        pending->epoch=2;
        pending->digest=generation_digest(map,1);
        pending->state=uint32_t(GenerationState::STAGING);
        pending->ready_helpers=pending->ready_ranks=0;
        const uint64_t saved_digest=control->active_digest;
        check(controller->propose(pending->digest),"capacity candidate proposed through controller");
        check(participants[0]->tick() &&
              control->state==uint32_t(PlacementState::RESERVING),
              "capacity proposal enters reservation");
        peers();
        check(read_shared_ack(map,1).fault_code==uint32_t(FaultCode::CAPACITY) &&
              migration[1]->reserve(map,*control)==FaultCode::CAPACITY,
              "full-budget gain without departure is sticky CAPACITY");
        check(participants[0]->tick(),"capacity detection by helper");
        check(participants[0]->tick() &&
              control->state==uint32_t(PlacementState::ACTIVE) &&
              control->active_epoch==1 && control->active_slot==0 &&
              control->active_digest==saved_digest &&
              placement_journal(map)->entry[control->journal_index].fault==
                  uint32_t(FaultCode::CAPACITY),
              "capacity ACK aborts forward without changing ACTIVE(N)");
        peers();
        check(participants[0]->tick(),"cancel precommit reservations after capacity abort");
        check(!controller->can_propose(),
              "abort blocks fresh proposal until rank abort cleanup");
        PlacementControl aborted{};
        check(placement_snapshot(map,&aborted) && coordinator.step(map,aborted,rank_ops),
              "production rank participant publishes abort-cleaned ACKs");
        check(controller->can_propose(),"all abort-cleaned ACKs release proposal gate");
        for (uint32_t helper=0;helper<helpers;++helper) {
            check(migration[helper]->transaction()==0 &&
                  pools[helper]->ledger().charged()==
                      selected[helper].expert.size()*stride &&
                  alloc[helper]->bytes==fixed_bytes[helper] &&
                  ops[helper]->active==0 &&
                  ops[helper]->banks_built==0,
                  "capacity rollback leaves every old slot/bank and physical allocation untouched");
        }
    }

    // Aborting after rank GPU upload is queued must wait for that rank's
    // independent DMA event, even though ACTIVE cleared pending map identity.
    // Normal READY cannot validate an aborted candidate; the coordinator's
    // production abort_upload_done callback proves event completion instead.
    {
        auto* pending=generation_header(map,1);
        std::memcpy(generation_map(map,1),generation_map(map,0),h->owner_map_bytes);
        generation_map(map,1)[0]=uint8_t(helpers==3 ? 2 : 1);
        generation_map(map,1)[1]=0;
        pending->epoch=2;
        pending->digest=generation_digest(map,1);
        pending->state=uint32_t(GenerationState::STAGING);
        pending->ready_helpers=pending->ready_ranks=0;
        check(controller->propose(pending->digest),
              "abort-inflight candidate proposed through controller");
        check(participants[0]->tick() &&
              control->state==uint32_t(PlacementState::RESERVING),
              "abort-inflight enters reservation");
        peers();
        check(participants[0]->tick(),"abort-inflight helper reserve");
        PlacementControl snapshot{};
        check(placement_snapshot(map,&snapshot) &&
              coordinator.step(map,snapshot,rank_ops),
              "abort-inflight engine reservation");
        check(participants[0]->tick() &&
              control->state==uint32_t(PlacementState::COPYING),
              "abort-inflight reaches COPYING");
        check(placement_snapshot(map,&snapshot) &&
              coordinator.step(map,snapshot,rank_ops) &&
              gpu[0].issued[1] && !gpu[0].event_ready[1],
              "engine rank enqueues bitmap upload with pending event");
        check(controller->fail(FaultCode::UPLOAD) &&
              control->state==uint32_t(PlacementState::ACTIVE) &&
              !controller->can_propose(),
              "forward abort blocks proposal before rank DMA completion");
        peers();
        check(participants[0]->tick(),
              "abort-inflight helper cancel before rank upload proof");
        check(placement_snapshot(map,&snapshot) &&
              coordinator.step(map,snapshot,rank_ops) &&
              !controller->can_propose() &&
              read_shared_ack(map,kMaxHelpers).phase!=uint32_t(AckPhase::CLEANED),
              "rank must not acknowledge abort while bitmap DMA is pending");
        gpu[0].complete(1);
        check(coordinator.step(map,snapshot,rank_ops) &&
              read_shared_ack(map,kMaxHelpers).phase==uint32_t(AckPhase::CLEANED) &&
              controller->can_propose(),
              "rank event completion publishes abort cleanup and unblocks proposal");
    }

    for (uint32_t cycle=1;cycle<=1000;++cycle) {
        const uint32_t old_slot=control->active_slot,new_slot=1u-old_slot;
        const uint64_t old_digest=control->active_digest;
        const uint8_t old_owner=owner_at(map,old_slot,0,0);
        const uint8_t new_owner=(cycle&1) ?
            uint8_t(helpers==3 ? 2 : 1) : uint8_t(0);
        const std::vector<uint8_t> old_map(
            generation_map(map,old_slot),
            generation_map(map,old_slot)+h->owner_map_bytes);
        StepTicket ticket{};
        check(ledger.admit(2,&ticket)==r3::admission::Ledger::Result::Ok &&
              ledger.bind(ticket.ticket_id,cycle) &&
              ticket.generation_slot==old_slot &&
              ticket.placement_epoch==cycle,
              "move actual request admitted on immutable old generation");
        auto* pending=generation_header(map,new_slot);
        std::memcpy(generation_map(map,new_slot),
                    generation_map(map,old_slot),h->owner_map_bytes);
        generation_map(map,new_slot)[0]=new_owner;
        // Concurrent legal transfers: helper 0 swaps at full budget.
        // In the three-helper shape the empty tail receives its first expert.
        generation_map(map,new_slot)[1]=(cycle&1) ? 0 : 1;
        check(!std::equal(old_map.begin(),old_map.end(),
                          generation_map(map,new_slot)),
              "candidate changes at least one canonical owner");
        pending->epoch=control->active_epoch+1;
        pending->digest=generation_digest(map,new_slot);
        pending->ready_helpers=pending->ready_ranks=0;
        pending->state=uint32_t(GenerationState::STAGING);
        for (uint32_t helper=0;helper<helpers;++helper)
            ops[helper]->ready[new_slot]=false;
        check(controller->propose(pending->digest),"move proposed via controller");
        const uint64_t transaction=control->transaction_id;
        check(control->active_slot==old_slot &&
              control->active_digest==old_digest &&
              std::equal(old_map.begin(),old_map.end(),
                         generation_map(map,old_slot)),
              "proposing never mutates active generation");
        check(participants[0]->tick() &&
              control->state==uint32_t(PlacementState::RESERVING),
              "move proposal enters RESERVING");
        peers();
        check(participants[0]->tick(),"move helper reserve");
        PlacementControl snapshot{};
        check(placement_snapshot(map,&snapshot) &&
              coordinator.step(map,snapshot,rank_ops),"move engine reserve");
        check(participants[0]->tick() &&
              control->state==uint32_t(PlacementState::COPYING),"move copying phase");
        for (uint32_t helper=0;helper<helpers;++helper) {
            const auto& physical=*pools[helper];
            const uint64_t physical_peak_budget=
                physical.ledger().hard(); // steady budget + incoming stride
            check(physical.ledger().charged()<=physical_peak_budget &&
                  alloc[helper]->bytes<=physical_peak_budget &&
                  physical.ledger().valid(),
                  "physical peak bounded by expert budget plus staging slot");
        }
        peers();
        check(participants[0]->tick(),"move initiate helper uploads");
        // The gained expert's fake event is deliberately incomplete while a
        // parent under N reads the exact old bank, and READY is still absent.
        const uint32_t destination=new_owner;
        check(ops[destination]->busy &&
              read_shared_ack(map,destination).phase!=uint32_t(AckPhase::READY),
              "READY cannot precede incoming expert's last upload event");
        check(ops[0]->busy && (helpers!=3 || cycle!=1 ||
              pools[2]->ledger().slots()[0].state==r3::residency::State::COPYING),
              "zero-bootstrap tail receives its first upload via physical slot");
        for (uint32_t helper=0;helper<helpers;++helper) {
            const auto& old_bindings=ops[helper]->physical[old_slot];
            for (size_t i=0;i<old_bindings.size();++i)
                check(ops[helper]->serving(old_slot,uint32_t(i))==old_bindings[i],
                      "old request uses old bank throughout COPYING");
        }
        peers();
        check(participants[0]->tick() &&
              control->state==uint32_t(PlacementState::COPYING),
              "incomplete helper events prevent global READY");
        check(read_shared_ack(map,destination).phase!=uint32_t(AckPhase::READY) &&
              !migration[destination]->bank_built() &&
              !ops[destination]->ready[new_slot],
              "unfinished expert event must not ACK READY or build its bank");
        // If two experts arrive, completion of the first event only
        // starts the next transfer; it does not unlock the READY ACK.
        bool all_built=false;
        for (uint32_t round=0;round<3;++round) {
            for (uint32_t helper=0;helper<helpers;++helper)
                if (ops[helper]->busy) ops[helper]->complete();
            peers();
            check(participants[0]->tick(),"move poll independent expert upload events");

            all_built=true;
            for (uint32_t helper=0;helper<helpers;++helper)
                all_built=all_built && migration[helper]->bank_built();
            if (all_built) break;
        }
        check(all_built,"all gained experts finished before candidate READY");

        for (uint32_t helper=0;helper<helpers;++helper)
            check(read_shared_ack(map,helper).phase==uint32_t(AckPhase::READY) &&
                  migration[helper]->bank_built() &&
                  ops[helper]->ready[new_slot],
                  "all helper READY ACKs require complete built banks");
        check(placement_snapshot(map,&snapshot) &&
              coordinator.step(map,snapshot,rank_ops),
              "move ranks issue and record pending bitmap events");
        for (uint32_t rank=0;rank<ranks;++rank)
            gpu[rank].complete(new_slot);
        check(coordinator.step(map,snapshot,rank_ops),
              "move rank READY after event");
        check(participants[0]->tick() &&
              control->state==uint32_t(PlacementState::PREPARED),
              "move all helper and rank ready");
        check(participants[0]->tick() &&
              control->state==uint32_t(PlacementState::QUIESCING),
              "move quiescing");
        const auto close=rank_ops.prepare_close();
        check(ledger.host_complete(ticket.ticket_id) &&
              ledger.poll([](const StepTicket&,uint32_t) {
                  return r3::admission::Ledger::Proof::Proven;
              }) &&
              ledger.all_ranks_consumed(uint32_t(cycle),close.latest_ticket),
              "move all ranks consumed");
        for (uint32_t lane=0;lane<kLanes;++lane) {
            const uint32_t lane_slot=lane%kMaxSlots;
            const bool served=(helpers>1 ? lane_slot<helpers :
                (control->expected_ranks & (1u<<lane_slot))!=0) &&
                (!lane_is_small(lane)||lane_class(lane)<h->small_class_count);
            if (!served) continue;
            h->lane[lane].drain_request=transaction;
            h->lane[lane].drain_ack=transaction;
        }
        check(placement_snapshot(map,&snapshot) &&
              coordinator.step(map,snapshot,rank_ops),"move drain and close");
        peers();
        check(participants[0]->tick() &&
              control->state==uint32_t(PlacementState::COMMIT),
              "move commit only after rank/helper quiescence");
        check(participants[0]->tick() &&
              generation_header(map,new_slot)->state==
                  uint32_t(GenerationState::ACTIVE),
              "move helper marks new generation ACTIVE");
        peers();
        check(placement_snapshot(map,&snapshot) &&
              coordinator.step(map,snapshot,rank_ops),
              "move all engine ranks commit");
        check(participants[0]->tick() &&
              control->state==uint32_t(PlacementState::RETIRING),
              "move retiring");
        peers();
        check(participants[0]->tick(),"move helper retirement ACKs");
        check(placement_snapshot(map,&snapshot) &&
              coordinator.step(map,snapshot,rank_ops),
              "move engine retirement proofs");
        check(participants[0]->tick() &&
              control->state==uint32_t(PlacementState::ACTIVE) &&
              generation_header(map,old_slot)->state==
                  uint32_t(GenerationState::RETIRED),
              "move publish committed generation");
        check(!controller->can_propose(),
              "ACTIVE before engine REOPENED refuses next proposal");
        for (uint32_t helper=0;helper<helpers;++helper)
            check(!migration[helper]->reclaim(map,*control),
                  "old physical slot unreclaimable before REOPENED");
        check(placement_snapshot(map,&snapshot) &&
              coordinator.step(map,snapshot,rank_ops) &&
              reopened_exactly(map,*control,transaction),
              "move engine ledger reopens and publishes exact REOPENED");
        check(!controller->can_propose(),
              "REOPENED alone cannot bypass last helper cleanup");
        peers();
        if (helpers>1) check(!controller->can_propose(),
              "peer cleanup cannot replace helper-zero reclamation");
        check(participants[0]->tick(),"move slot reclamation after REOPENED");
        check(controller->can_propose(),
              "only all post-REOPENED helper cleanups admit next proposal");
        for (uint32_t helper=0;helper<helpers;++helper) {
            check(migration[helper]->transaction()==0 &&
                  ops[helper]->active==new_slot &&
                  pools[helper]->ledger().valid() &&
                  alloc[helper]->bytes==fixed_bytes[helper] &&
                  pools[helper]->ledger().charged()<=
                      pools[helper]->ledger().hard()-pools[helper]->ledger().headroom(),
                  "move/return leaves no committed-budget excess or slot growth");
            for (const auto& slot:pools[helper]->ledger().slots())
                check(slot.state==r3::residency::State::FREE ||
                      slot.state==r3::residency::State::ACTIVE,
                      "no retained slot stuck in staging or retirement");
            check(migration[helper]->local(new_slot).size()==h->owner_map_bytes,
                  "new helper bank remains fully bound");
        }
        check(pools[0]->ledger().charged()==
                  selected[0].expert.size()*stride,
              "helper zero full-budget swap never exceeds committed expert budget");
        check(ops[0]->serving(new_slot,0)==
              (new_owner==0 ? migration[0]->physical(new_slot)[0] : -1),
              "moved expert bound in correct helper bank");
        check(ops[destination]->serving(new_slot,0)==
              migration[destination]->physical(new_slot)[0],
              "moved expert bound in receiving helper bank");
        check((old_owner==0 ? ops[0]->serving(old_slot,0)>=0 :
                               ops[0]->serving(old_slot,0)==-1),
              "old generation bank retains the prior owner, not the new one");
    }
    if (helpers==3) {
        // Production coordinator with the same event-query contract as the
        // HIP adapter: the queued inactive bitmap encounters an event ERROR,
        // not merely hipErrorNotReady. Keep serving the existing active map.
        const uint32_t active_slot=control->active_slot;
        const uint32_t inactive_slot=1u-active_slot;
        const uint64_t active_epoch=control->active_epoch;
        check(stage_same_map_generation(map) && controller->begin_epoch_cycle(),
              "device error candidate staged through real helper controller");
        check(participants[0]->tick() &&
              control->state==uint32_t(PlacementState::RESERVING),
              "device error enters reservation");
        peers();
        check(participants[0]->tick(),"device error helper reserve");
        PlacementControl aborted{};
        check(placement_snapshot(map,&aborted) &&
              coordinator.step(map,aborted,rank_ops),
              "device error rank reserve");
        check(participants[0]->tick() &&
              control->state==uint32_t(PlacementState::COPYING),
              "device error reaches COPYING");
        check(placement_snapshot(map,&aborted) &&
              coordinator.step(map,aborted,rank_ops) &&
              gpu[0].issued[inactive_slot] && !gpu[0].event_ready[inactive_slot],
              "device error rank enqueues actual inactive upload event");
        check(controller->fail(FaultCode::UPLOAD) &&
              control->state==uint32_t(PlacementState::ACTIVE) &&
              !controller->can_propose(),
              "device error forward abort keeps old generation");
        peers();
        check(participants[0]->tick(),"device error helper cancels reservations");
        rank_ops.abort_event_error_rank=0;
        check(placement_snapshot(map,&aborted) &&
              coordinator.step(map,aborted,rank_ops) &&
              read_shared_ack(map,kMaxHelpers).phase==uint32_t(AckPhase::CLEANED) &&
              read_shared_ack(map,kMaxHelpers).fault_code==uint32_t(FaultCode::DEVICE),
              "production rank coordinator publishes abort HIP event error");
        check(participants[0]->tick() && participants[0]->placement_halted() &&
              !controller->can_propose() && controller->failed_closed() &&
              control->state==uint32_t(PlacementState::ACTIVE) &&
              control->active_epoch==active_epoch &&
              control->active_slot==active_slot && !control->fault_code &&
              !h->helper_fault && !ledger.stopped() &&
              placement_journal(map)->entry[control->journal_index].cleanup_fault==
                  uint32_t(FaultCode::DEVICE) &&
              placement_journal(map)->entry[control->journal_index].cleanup_actor==
                  actor_id(ActorKind::ENGINE_RANK,0),
              "event query error stops placement immediately, not serving");
        // Later event completion cannot convert the sticky error into CLEANED.
        rank_ops.abort_event_error_rank=-1;
        gpu[0].complete(inactive_slot);
        check(coordinator.step(map,aborted,rank_ops) &&
              read_shared_ack(map,kMaxHelpers).fault_code==uint32_t(FaultCode::DEVICE) &&
              !controller->can_propose(),
              "aborted event failure remains sticky after eventual completion");
        StepTicket still_serving{};
        check(ledger.admit(1,&still_serving)==r3::admission::Ledger::Result::Ok &&
              still_serving.placement_epoch==active_epoch &&
              still_serving.generation_slot==active_slot &&
              ledger.bind(still_serving.ticket_id,2000) &&
              ledger.host_complete(still_serving.ticket_id) &&
              ledger.poll([](const StepTicket&,uint32_t) {
                  return r3::admission::Ledger::Proof::Proven;
              }),
              "rank abort device error retains committed serving generation");
    }
    for (uint32_t helper=0;helper<helpers;++helper)
        check(alloc[helper]->peak==alloc[helper]->bytes,
              "fake physical allocation is fixed across 1000 real-move cycles");
    check(munmap(map,layout.bytes)==0,"move unmap segment");
}

int main() {
    V23Cycle_UnreadyAndAbortPreserveOldGeneration();
    V23Cycle_PrecommitAbortDuringLiveServing();
    V23Cycle_StalledParkFaultsAndKeepsAdmissionClosed();
    V23Bootstrap_UnsealedTickReportsInvalidMap();
    V23Cycle_SamePlacementAndSharedProtocol(1,2);
    V23Cycle_SamePlacementAndSharedProtocol(3,1);
    V23Move_PhysicalSlotLimitIndependentOfByteBudget();
    V23Move_ChangedMapThroughProduction(1,2);
    V23Move_ChangedMapThroughProduction(2,1);
    V23Move_ChangedMapThroughProduction(3,1);
    V23Cycle_SamePlacementAndSharedProtocol(1,2,true);
    V23Cycle_SamePlacementAndSharedProtocol(3,1,true);
    V23Cleanup_DeadlineAndExactAcks();
    V23Epoch_ForwardOnlyAbortRetry();
    V23Ack_OldTxnDelayedDuplicateLost();
    V23Ack_CorruptCurrentAndFault();
    V23Fault_BeforeAndAfterCommit();
    V23Slots_ReserveCopyRetireFaults();
    V23Policy_HeatEqualCostAndSwap();
    std::puts("r3_moe_epoch_test: ok");
}
