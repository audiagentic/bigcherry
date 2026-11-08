// RHP slice H: reconstruct the real fixed-stride Pool ledger from a v23
// ACTIVE generation; preserve commit/abort history without shared state.
// No device or GPU required.
#include "worker_policy_live.h"
#include "worker_bootstrap.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
using namespace r3::policy;
using namespace r3::moe_shm;
static void check(bool ok,const char* reason) {
    if (!ok) { std::fprintf(stderr,"r3_policy_live_test: FAIL %s\n",reason); std::exit(1); }
}
struct Fixture {
    const LayoutShape shape{128,1,5,8,128,64};
    Layout layout{};
    void* map=nullptr;
    OwnerSnapshot owner{};
    Fixture() {
        check(layout_for(shape,2,slots_for(2, 1),&layout),"layout");
        map=mmap(nullptr,layout.bytes,PROT_READ|PROT_WRITE,
                 MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        check(map!=MAP_FAILED && init(map,1,shape,2, 1),"map");
        auto* h=static_cast<Header*>(map);
        store(&h->helper_count,2u);
        for (uint32_t e=0;e<5;++e)
            layer_slots(map,0)[e]=e<4?int32_t(e):-1;
        store(&h->layer_map_ready[0],1u);
        struct Owned { std::vector<std::pair<uint32_t,uint32_t>> expert; };
        check(r3::epoch::publish_owned(map,Owned{{{0,0},{0,1},{0,2}}},0) &&
              r3::epoch::publish_owned(map,Owned{{{0,3}}},1),"publish");
        check(r3::epoch::mark_bootstrap_filtered(map,0) &&
              r3::epoch::mark_bootstrap_filtered(map,1),"filter");
        r3::epoch::SharedTransport transport(map);
        check(transport.seal_bootstrap(),"seal");
        check(capture_owners(map,&owner),"capture");
        check(owner.owner==std::vector<int32_t>({0,0,0,1,-1}),
              "real ACTIVE owner mapping");
    }
    ~Fixture(){ if(map && map!=MAP_FAILED)munmap(map,layout.bytes); }
};
int main() {
    Fixture f;
    const auto budgets=std::vector<uint64_t>{48,32};
    const auto bytes=std::vector<uint64_t>{16,16,16,16,0};
    std::vector<LedgerView> ledgers;
    std::string error;
    check(derive_active_ledgers(f.owner,bytes,budgets,16,
                                &ledgers,&error),"derive exact ledger charges");
    check(ledgers.size()==2 && ledgers[0].hard==64 &&
          ledgers[0].headroom==16 && ledgers[0].scratch==0 &&
          ledgers[0].charged==48 && ledgers[0].slots.size()==3 &&
          ledgers[1].hard==48 && ledgers[1].charged==16 &&
          ledgers[1].slots[0].expert==3,"same Pool hard/headroom/charged");
    for (auto& v:ledgers)
        for (auto& s:v.slots)
            check(s.state==r3::residency::State::ACTIVE && s.bytes==16 &&
                  s.completed_planes==15,"accounting witness");
    auto bad_bytes=bytes;bad_bytes[0]=15;
    check(!derive_active_ledgers(f.owner,bad_bytes,budgets,16,
                                 &ledgers),"unpadded byte size not a Ledger charge");
    auto bad_owner=f.owner;bad_owner.runtime[0]=0;
    check(!derive_active_ledgers(bad_owner,bytes,budgets,16,
                                 &ledgers),"inaccessible active owner refused");
    check(!derive_active_ledgers(f.owner,bytes,{32,32},16,
                                 &ledgers),"insufficient helper budget refused");
    check(!derive_active_ledgers(f.owner,bytes,{UINT64_MAX,32},16,
                                 &ledgers),"budget plus incoming reserve overflow");
    check(derive_active_ledgers(f.owner,bytes,budgets,16,
                                &ledgers),"recovery");

    // The derivation supplies exact capacity evidence, never eligibility:
    // missing destination format/source declarations still fail closed.
    std::vector<ExpertView> experts(5);
    for (auto& e:experts) e.bytes=16;
    experts[4].bytes=16;experts[4].protected_expert=true;
    r3::heat::Heat heat(1,5,1000);
    heat.counts={10,8,5,1,0};heat.samples={10,10,10,10,10};
    HeatDelta delta{};
    delta.layers=1;delta.experts=5;delta.counts={10,8,5,1,0};
    delta.complete_cells=delta.complete_observations=true;
    delta.interval_observations=24;delta.observation_count=24;delta.age_ns=1;
    PolicyControls controls{};
    controls.half_life_observations=1000;controls.margin=.1;
    controls.min_residency_ns=1;controls.max_heat_age_ns=10000;
    controls.max_moved_bytes=32;controls.min_samples=1;controls.max_moves=2;
    r3::heat::Input in;
    check(!build_input(f.owner,experts,ledgers,heat,delta,controls,
                       100000,0,false,&in),"no implicit eligibility for other helpers");
    for (auto& e:experts) e.target_eligible={1,1};
    experts[4].target_eligible={0,0};
    check(build_input(f.owner,experts,ledgers,heat,delta,controls,
                      100000,0,false,&in) && in.devices[0].budget==48 &&
          in.devices[1].budget==32,"reconstructed state accepted by real input builder");

    MoveHistory history;
    check(history.reset(f.owner),"seed coordinator history");
    check(history.times().size()==5 && history.times()[2]==0,
          "restart means all experts moved long ago");
    JournalRecord done{};
    done.transaction_id=1;done.before_epoch=f.owner.epoch;
    done.before_digest=f.owner.digest;
    done.after_epoch=f.owner.epoch+1;
    done.candidate_digest=f.owner.digest+1;
    done.start_ns=1;done.commit_ns=500;done.finish_ns=600;
    done.state=uint32_t(PlacementState::ACTIVE);
    auto moved=f.owner;
    moved.epoch=done.after_epoch;moved.digest=done.candidate_digest;
    moved.owner[2]=1;
    check(history.observe_closed(done,moved,1000)==HistoryObservation::COMMITTED,
          "closed committed transaction records change");
    check(history.times()[2]==1000 && history.times()[0]==0,
          "timestamp only moved expert");
    check(history.observe_closed(done,moved,1200)==HistoryObservation::UNCHANGED &&
          history.times()[2]==1000,"duplicate journal never advances timestamp");
    JournalRecord aborted{};
    aborted.transaction_id=2;aborted.before_epoch=moved.epoch;
    aborted.before_digest=moved.digest;aborted.start_ns=700;
    aborted.finish_ns=800;aborted.fault=3;
    aborted.state=uint32_t(PlacementState::ACTIVE);
    check(history.observe_closed(aborted,moved,1400)==HistoryObservation::ABORTED &&
          history.times()[2]==1000,"abort leaves N and last move untouched");
    JournalRecord forged=done;
    forged.transaction_id=3;forged.before_epoch=moved.epoch;
    forged.before_digest=moved.digest;forged.after_epoch=moved.epoch+1;
    forged.candidate_digest=moved.digest+1;
    forged.finish_ns=0;
    auto drift=moved;drift.epoch=forged.after_epoch;drift.digest=forged.candidate_digest;
    drift.owner[2]=0;
    check(history.observe_closed(forged,drift,1600)==HistoryObservation::INVALID &&
          history.times()[2]==1000,"unfinished journal cannot record a migration");
    forged.finish_ns=1700;
    check(history.observe_closed(forged,drift,1800)==HistoryObservation::COMMITTED &&
          history.times()[2]==1800,"closed drift back updates last move");
    check(history.reset(drift,3) && history.times()[2]==0,
          "restart drops ephemeral move history");
    std::puts("r3_policy_live_test: ok");
}
