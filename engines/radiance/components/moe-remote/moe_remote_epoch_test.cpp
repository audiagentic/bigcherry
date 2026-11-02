#include "worker_epoch.h"
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
int main() {
    V23Epoch_ForwardOnlyAbortRetry();
    V23Ack_OldTxnDelayedDuplicateLost();
    V23Ack_CorruptCurrentAndFault();
    V23Fault_BeforeAndAfterCommit();
    V23Slots_ReserveCopyRetireFaults();
    V23Policy_HeatEqualCostAndSwap();
    std::puts("r3_moe_epoch_test: ok");
}
