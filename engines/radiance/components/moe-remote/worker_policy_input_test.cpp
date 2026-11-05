#include "worker_policy_input.h"
#include "worker_epoch.h"
#include "worker_bootstrap.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/mman.h>

using namespace r3::moe_shm;
using namespace r3::policy;
static void check(bool ok, const char* reason) {
    if (!ok) { std::fprintf(stderr,"r3_policy_input_test: FAIL %s\n",reason); std::exit(1); }
}

struct Fixture {
    const LayoutShape shape{128,1,5,8,128,64};
    Layout layout{};
    void* map=nullptr;
    OwnerSnapshot owner;
    std::vector<ExpertView> meta;
    std::vector<LedgerView> ledgers;
    r3::heat::Heat heat{1,5,1000};
    HeatDelta delta{};
    PolicyControls controls{};
    Fixture() {
        check(layout_for(shape,2,slots_for(2),&layout),"v23 layout");
        map=mmap(nullptr,layout.bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        check(map!=MAP_FAILED && init(map,1,shape,2),"v23 segment");
        store(&static_cast<Header*>(map)->helper_count,2);
        for (uint32_t e=0;e<5;++e) layer_slots(map,0)[e]=(e<4 ? int32_t(e) : -1);
        store(&static_cast<Header*>(map)->layer_map_ready[0],1);
        struct Owned { std::vector<std::pair<uint32_t,uint32_t>> expert; };
        check(r3::epoch::publish_owned(map,Owned{{{0,0},{0,1},{0,2}}},0) &&
              r3::epoch::publish_owned(map,Owned{{{0,3}}},1),"bootstrap owners");
        check(r3::epoch::mark_bootstrap_filtered(map,0) &&
              r3::epoch::mark_bootstrap_filtered(map,1),"bootstrap filter");
        r3::epoch::SharedTransport transport(map);
        check(transport.seal_bootstrap(),"bootstrap seal");
        check(capture_owners(map,&owner),"capture real active generation");
        check(owner.owner==std::vector<int32_t>({0,0,0,1,-1}) &&
              owner.runtime==std::vector<uint8_t>({1,1,1,1,0}),
              "exact model-global owner map and protected local slot");
        meta.resize(5);
        for (auto& e:meta) { e.bytes=10; e.target_eligible={1,1}; }
        meta[4].protected_expert=true; meta[4].target_eligible={0,0};
        r3::residency::Ledger a(50,0,10),b(50,0,10);
        for (uint32_t e=0;e<3;++e) {
            uint32_t slot=0;
            check(a.bootstrap(e,10,1,&slot)==r3::residency::Result::OK,"bootstrap ledger A");
        }
        uint32_t slot=0;
        check(b.bootstrap(3,10,1,&slot)==r3::residency::Result::OK,"bootstrap ledger B");
        ledgers={copy_ledger(a),copy_ledger(b)};
        heat.counts={90,75,45,10,10000};
        heat.samples.assign(5,10);
        delta.layers=1;delta.experts=5;delta.counts={90,75,45,10,0};
        delta.observation_count=220;delta.interval_observations=220;
        delta.complete_cells=true;delta.complete_observations=true;delta.age_ns=1;
        controls.half_life_observations=1000;
        controls.margin=0.1; controls.min_residency_ns=100;
        controls.max_heat_age_ns=10000; controls.max_moved_bytes=20;
        controls.min_samples=5; controls.max_moves=2;
    }
    ~Fixture(){munmap(map,layout.bytes);}
    bool build(r3::heat::Input* result,std::string* err=nullptr) const {
        return build_input(owner,meta,ledgers,heat,delta,controls,100000,0,false,result,err);
    }
};
int main() {
    Fixture f;
    r3::heat::Input input;
    std::string error;
    check(f.build(&input,&error),"valid live input");
    check(input.current[4]==-1 && input.allowed[4]==std::vector<uint8_t>({0,0}) &&
          input.devices[0].budget==40 && input.devices[1].budget==40,
          "protected slot fixed and helper residency budget derived");
    const auto plan=r3::heat::propose(input,{1,1});
    check(plan.valid && !plan.no_change &&
          r3::heat::delta_matches(plan,input.current) && plan.owner[4]==-1,
          "the live plan moves only helper-owned experts");
    r3::heat::Heat accumulated(1,5,1000);
    uint64_t consumed=0;
    check(consume_heat(f.delta,&accumulated,&consumed) && consumed==220 &&
          accumulated.counts[0]==90 && accumulated.samples[4]==1,
          "coherent interval atomically credits decayed heat");
    check(!consume_heat(f.delta,&accumulated,&consumed) && consumed==220 &&
          accumulated.counts[0]==90, "duplicate interval not double credited");
    auto invalid_window=f.delta;invalid_window.complete_observations=false;
    check(!consume_heat(invalid_window,&accumulated,&consumed),
          "incomplete diagnostic window cannot advance heat");
    invalid_window=f.delta;invalid_window.observation_count=221;
    check(!consume_heat(invalid_window,&accumulated,&consumed),
          "gapped observation sequence rejected");
    auto state=f.owner;
    state.digest=0;
    check(!build_input(state,f.meta,f.ledgers,f.heat,f.delta,f.controls,
                       100000,0,false,&input),"missing live generation rejected");
    f.meta[4].target_eligible={1,1};
    check(f.build(&input),"protected expert unchanged even when file exists");
    f.meta[4].protected_expert=false;
    check(f.build(&input) && input.allowed[4]==std::vector<uint8_t>({0,0}),
          "local expert never migrates");
    f.meta[0].target_eligible[1]=0;
    check(f.build(&input) && input.allowed[0][1]==0,
          "destination source/encoding eligibility honored");
    f.meta[0].target_eligible[1]=1;
    f.meta[0].target_eligible[0]=0;
    check(!f.build(&input,&error) && error.find("eligibility")!=std::string::npos,
          "invalid incumbent source rejected");
    f.meta[0].target_eligible[0]=1;
    f.meta[0].last_move_ns=100001;
    check(!f.build(&input),"future move timestamp rejected");
    f.meta[0].last_move_ns=99999;
    check(f.build(&input) && input.last_move_ns[0]==99999,
          "last move times copied exactly");
    f.meta[0].last_move_ns=0;
    f.ledgers[0].charged++;
    check(!f.build(&input),"ledger charge mismatch rejected");
    f.ledgers[0].charged--;
    f.ledgers[0].slots[0].state=r3::residency::State::COPYING;
    check(!f.build(&input),"in-flight residency snapshot rejected");
    f.ledgers[0].slots[0].state=r3::residency::State::ACTIVE;
    f.ledgers[0].headroom=25;
    check(!f.build(&input),"insufficient physical headroom rejected");
    f.ledgers[0].headroom=10;
    f.delta.complete_observations=false;
    check(!f.build(&input,&error) && error.find("coverage")!=std::string::npos,
          "diagnostic histogram cannot falsely authorize movement");
    f.delta.complete_observations=true; f.delta.age_ns=10001;
    check(!f.build(&input),"stale window rejected");
    f.delta.age_ns=1;f.controls.min_samples=0;
    check(!f.build(&input),"unset policy control rejected");
    f.controls.min_samples=5;
    check(f.build(&input),"recovery after invalid observations");

    // Read from the production snapshot converter, not test-local owner
    // decoding: tampered map fails before the solver sees it.
    generation_map(f.map,0)[0]=1;
    check(!capture_owners(f.map,&state),"bad active digest denied");
    generation_map(f.map,0)[0]=0;
    check(capture_owners(f.map,&state),"untampered real map accepted");
    store(&static_cast<Header*>(f.map)->layer_map_ready[0],0);
    check(!capture_owners(f.map,&state),"unready runtime slot denies helper owner");
    std::puts("r3_policy_input_test: ok");
}
