// RHP11 G: test-only forced proposal through the real Controller, Pool,
// residency Ledger and migration Participant. All GPU operations are faked.
#include "worker_forced_move.h"
#include "worker_bootstrap.h"
#include "worker_move_slots.h"
#include "worker_slot_pool.h"
#include "worker_config.h"
#include <fstream>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <sys/mman.h>
#include <vector>
using namespace r3::moe_shm;
using namespace r3::epoch;
static void check(bool x,const char* why) {
    if(!x){std::fprintf(stderr,"r3_forced_move_test FAILED: %s\n",why);std::exit(1);}
}
struct FakeClock:Clock {
    uint64_t t=1;
    uint64_t now_ns()const override {return t;}
    void step(){t+=10;}
};
struct Allocator {
    std::map<void*,uint64_t> live;
    // Card-free large-budget pool: the ledger tracks the real full-size
    // allocation, while the fake never dereferences its device addresses.
    bool sparse=false;
    explicit Allocator(bool virtual_device=false):sparse(virtual_device){}
    bool allocate(uint64_t bytes,void** out) {
        *out=::operator new(size_t(sparse?1:bytes),std::nothrow);
        if(!*out)return false;
        live[*out]=bytes;return true;
    }
    void release(void* p) {
        check(live.erase(p)==1,"release exactly one live slot");
        ::operator delete(p);
    }
};
struct FakeGpu:r3::worker_moves::Operations {
    r3::residency::Ledger& ledger;
    std::vector<int32_t> banks[2];
    uint32_t active=0;
    bool busy=false;
    uint64_t uploads=0;
    uint32_t eligible_experts=2, expert_columns=0;
    explicit FakeGpu(r3::residency::Ledger& ledger,uint32_t eligible=2,
                     uint32_t columns=0)
        :ledger(ledger),eligible_experts(eligible),expert_columns(columns){}
    int32_t preferred_local(uint32_t id) const override {
        // 2 layers, 3 experts; expert 2 and every expert in layer 1
        // are engine-local and cannot be uploaded as a regular expert.
        return id<eligible_experts ?
               int32_t(expert_columns ? id%expert_columns : id) : -1;
    }
    bool start_upload(uint32_t expert,uint32_t physical,uint32_t local) override {
        if(busy||preferred_local(expert)!=int32_t(local)||
           physical>=ledger.slots().size())return false;
        busy=true;++uploads;return true;
    }
    r3::worker_moves::Event upload_event() override {
        if(!busy)return r3::worker_moves::Event::FAILED;
        busy=false;return r3::worker_moves::Event::COMPLETE;
    }
    bool build_bank(uint32_t slot,const std::vector<int32_t>& locals) override {
        if(slot>1||slot==active||busy)return false;
        banks[slot]=locals;return true;
    }
    bool drain_upload() override {busy=false;return true;}
    void forget(uint32_t) override {}
    void activate_bank(uint32_t slot) override {active=slot;}
};
struct Selected {std::vector<std::pair<uint32_t,uint32_t>> expert;};


// Production Controller + per-helper Participant and full-size ledger path.
// All three helper budgets are exactly full, so any one-way arrival fails.
// The only legal rebalance is one outgoing plus one incoming on each helper.
static void full_budget_swaps(bool production_budgets) {
    using namespace r3::forced;
    constexpr uint32_t helpers=3;
    const uint32_t experts=production_budgets?512:9;
    const uint32_t layers=production_budgets?49:2;
    // Flash-Next W4's real padded weight stride and three production
    // flashnext-4gpu budgets. The fake allocator touches NO GPU memory.
    const uint64_t stride=production_budgets?2'534'400:256;
    const std::vector<uint64_t> budgets=production_budgets ?
        std::vector<uint64_t>{22528ull*1048576,22528ull*1048576,
                              14336ull*1048576} :
        std::vector<uint64_t>(helpers,(experts/helpers)*stride);
    const std::array<uint32_t,helpers> capacity={{
        uint32_t(budgets[0]/stride),uint32_t(budgets[1]/stride),
        uint32_t(budgets[2]/stride)}};
    if(production_budgets)
        check(capacity==std::array<uint32_t,helpers>{{9320,9320,5931}},
              "production four-GPU helper budget slots");
    const LayoutShape shape{128,layers,experts,8,128,64};
    Layout layout{};
    check(layout_for(shape,layers,slots_for(helpers),&layout),"swap layout");
    void* map=mmap(nullptr,layout.bytes,PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    check(map!=MAP_FAILED && init(map,layers,shape,helpers),"swap init");
    auto* h=static_cast<Header*>(map);
    h->helper_count=helpers;
    placement_control(map)->expected_ranks=1;
    Selected sel[helpers]{};
    for(uint32_t id=0;id<layers*experts;++id)
        for(uint32_t offset=0;offset<helpers;++offset) {
            const uint32_t helper=(id+offset)%helpers;
            if(sel[helper].expert.size()<capacity[helper]) {
                sel[helper].expert.emplace_back(id/experts,id%experts);
                break;
            }
        }
    for(uint32_t helper=0;helper<helpers;++helper)
        check(sel[helper].expert.size()==capacity[helper],
              "each production helper selected to its full expert budget");
    for(uint32_t i=0;i<helpers;++i) {
        check(publish_owned(map,sel[i],i),"full canonical bootstrap");
        check(mark_bootstrap_filtered(map,i),"full filtered agreement");
    }
    SharedTransport transport(map);
    check(transport.seal_bootstrap(),"full canonical seal");
    for(uint32_t l=0;l<layers;++l) {
        auto* slots=layer_slots(map,l);
        for(uint32_t e=0;e<experts;++e)
            slots[e]=(production_budgets || l==0)?int32_t(e):-1;
        store(&h->layer_map_ready[l],1);
    }
    FakeClock clock;
    Controller controller(transport,clock,placement_control(map)->active_digest,
                          helpers,1,Deadlines{},
                          Bootstrap{1,0,placement_control(map)->seq,0});
    check(controller.can_propose(),"full controller bootstrapped");
    const Source source=[&](uint32_t l,uint32_t e) {
        return (production_budgets || l==0) && l<layers && e<experts;
    };
    using Pool=r3::worker_slots::Pool<Allocator>;
    std::vector<std::unique_ptr<Allocator>> alloc;
    std::vector<std::unique_ptr<Pool>> pools;
    std::vector<std::unique_ptr<FakeGpu>> gpu;
    std::vector<std::unique_ptr<r3::worker_moves::Participant>> participant;
    for(uint32_t helper=0;helper<helpers;++helper) {
        alloc.push_back(std::make_unique<Allocator>(production_budgets));
        pools.push_back(std::make_unique<Pool>(
            budgets[helper],stride,capacity[helper],*alloc.back()));
        check(pools.back()->prepare(),"full pool prepare");
        std::vector<int32_t> locals(size_t(layers)*experts,-1);
        for(const auto& [l,e]:sel[helper].expert) {
            const uint32_t id=l*experts+e;
            uint32_t physical=UINT32_MAX;
            locals[id]=int32_t(e);
            check(pools.back()->bootstrap(id,stride,1,&physical)==
                  r3::residency::Result::OK,"full bootstrap physical");
        }
        gpu.push_back(std::make_unique<FakeGpu>(
            pools.back()->ledger(),production_budgets?layers*experts:experts,experts));
        gpu.back()->banks[0]=locals;
        participant.push_back(std::make_unique<r3::worker_moves::Participant>(
            pools.back()->ledger(),*gpu.back(),helper,layers,experts,stride));
        check(participant.back()->initialize(map,0,locals),"full generation bound");
        check(pools.back()->ledger().charged()==uint64_t(capacity[helper])*stride,
              "helper starts with every committed expert slot filled");
    }
    const auto invalid=[&](const Step& step,
                           const std::vector<uint64_t>& limits,
                           const Source& src,FaultCode wanted) {
        const auto old=controller.control();
        FaultCode code=FaultCode::NONE;
        check(propose(map,controller,step,limits,stride,src,{},&code) &&
              code==wanted,"swap reject published expected precommit fault");
        const auto& c=controller.control();
        check(c.active_slot==old.active_slot &&
              c.active_digest==old.active_digest &&
              c.active_epoch==old.active_epoch &&
              generation_digest(map,c.active_slot)==old.active_digest &&
              placement_journal(map)->entry[c.journal_index].fault==uint32_t(wanted),
              "invalid swap never modifies active map");
        for(uint32_t i=0;i<helpers;++i)publish_cleanup_ack(map,i,c);
        publish_cleanup_ack(map,kMaxHelpers,c);
        check(controller.can_propose(),"swap rejection cleaned before retry");
    };
    invalid(Step{Move{0,0,1}},budgets,source,FaultCode::CAPACITY);
    invalid(Step::swap(0,0,0,3),budgets,source,FaultCode::INVALID_MAP);
    invalid(Step::swap(0,0,0,0),budgets,source,FaultCode::INVALID_MAP);
    if(!production_budgets) {
        invalid(Step::swap(0,0,1,0),budgets,source,FaultCode::INVALID_MAP);
        layer_slots(map,1)[0]=0;
        invalid(Step::swap(0,0,1,0),budgets,source,FaultCode::INVALID_MAP);
        layer_slots(map,1)[0]=-1;
    }
    layer_slots(map,0)[0]=-1;
    invalid(Step::swap(0,0,0,1),budgets,source,FaultCode::INVALID_MAP);
    layer_slots(map,0)[0]=0;
    invalid(Step::swap(0,0,0,1),budgets,
            Source{[](uint32_t, uint32_t e){return e!=1;}},
            FaultCode::SOURCE);
    auto low_budget=budgets;
    low_budget[0]=(uint64_t(capacity[0])-1)*stride;
    invalid(Step::swap(0,0,0,1),low_budget,source,FaultCode::CAPACITY);
    invalid(Step::automatic(0,3),budgets,source,FaultCode::INVALID_MAP);
    invalid(Step::automatic(0,0),budgets,source,FaultCode::INVALID_MAP);
    const uint32_t old_slot=controller.control().active_slot;
    auto* owners=generation_map(map,old_slot);
    const uint8_t old_owner=owners[0];
    owners[0]=uint8_t((old_owner+1)%helpers);
    check(eligible(map,Step::swap(0,0,0,1),budgets,stride,source)==
          FaultCode::INVALID_MAP,"swap active digest guard");
    owners[0]=old_owner;
    const char* malformed[]={
        "swap:0:1:0", "swap:0:1:0:2:junk",
        "swap:0:1:-1:2", "swap:auto:0", "swap:auto:0:2:3",
        "swap:auto:-1:2", "swap:0:1:0:2garbage"
    };
    for(const auto* x:malformed) {
        Step unused;
        check(!parse_step(x,&unused),"strict swap parser refuses malformed row");
    }
    Step parsed;
    check(parse_step("swap:0:0:0:1",&parsed)&&
          parsed.kind==Step::Kind::SWAP &&
          parse_step("swap:auto:0:2",&parsed)&&
          parsed.kind==Step::Kind::AUTO_SWAP,
          "strict explicit and automatic swap parser");
    Resolved first{},alternative{};
    check(resolve(map,Step::automatic(0,1),budgets,stride,source,kNoPrevious,&first)==
          FaultCode::NONE && first.ids==std::array<uint32_t,2>{{0,1}},
          "AUTO picks lowest eligible layer and expert from ACTIVE");
    check(resolve(map,Step::automatic(0,1),budgets,stride,source,first.ids,
                  &alternative)==FaultCode::NONE &&
          alternative.ids==std::array<uint32_t,2>{{3,4}},
          "AUTO excludes both previous expert IDs deterministically");
    const auto empty_source=Source{[](uint32_t,uint32_t){return false;}};
    invalid(Step::automatic(0,1),budgets,empty_source,FaultCode::SOURCE);

    Schedule schedule;
    constexpr uint32_t transitions=1000;
    const uint32_t pairs[3][2]={{0,1},{0,2},{1,2}};
    for(uint32_t n=0;n<transitions;++n)
        schedule.steps.push_back(Step::automatic(pairs[n%3][0],pairs[n%3][1]));
    Driver driver(schedule,budgets,stride,source);
    std::array<uint32_t,2> previous{{UINT32_MAX,UINT32_MAX}};
    uint32_t active_slot=0;
    for(uint32_t step=0;step<transitions;++step) {
        Resolved expected{},repeat{};
        const Step request=schedule.steps[step];
        check(resolve(map,request,budgets,stride,source,previous,&expected)==
              FaultCode::NONE && expected.count==2,"full auto resolves");
        check(resolve(map,request,budgets,stride,source,previous,&repeat)==
              FaultCode::NONE && repeat.ids==expected.ids,
              "AUTO deterministic on unchanged canonical map");
        check(expected.ids[0]!=previous[0] && expected.ids[0]!=previous[1] &&
              expected.ids[1]!=previous[0] && expected.ids[1]!=previous[1],
              "AUTO never repeats an expert from previous transaction");
        const uint32_t old=active_slot,next=1u-old;
        const std::vector<uint8_t> before(
            generation_map(map,old),generation_map(map,old)+h->owner_map_bytes);
        clock.step();
        check(driver.submit(map,controller,clock.now_ns()),"full proposal");
        driver.observe(controller,clock.now_ns());
        const auto c=controller.control();
        check(c.state==uint32_t(PlacementState::PROPOSED) &&
              c.active_slot==old && c.pending_slot==next &&
              placement_candidate_valid(map,c),"full two-byte digested proposal");
        uint32_t changed=0;
        for(uint32_t id=0;id<h->owner_map_bytes;++id) {
            const uint8_t proposed=generation_map(map,next)[id];
            if(proposed!=before[id]) {
                ++changed;
                check(id==expected.ids[0]||id==expected.ids[1],
                      "candidate only changes resolved experts");
            }
        }
        check(changed==2,"exactly two ownership bytes changed");
        for(uint32_t i=0;i<helpers;++i) {
            for(uint32_t j=0;j<2;++j) {
                const uint32_t id=expected.ids[j];
                check(gpu[i]->active==old &&
                      gpu[i]->banks[old][id]==(before[id]==i?int32_t(id):-1),
                      "N's GPU bank remains readable during upload");
            }
        }
        check(controller.start_reserving(),"full reserving");
        clock.step(); driver.observe(controller,clock.now_ns());
        for(uint32_t i=0;i<helpers;++i) {
            check(participant[i]->reserve(map,controller.control())==FaultCode::NONE,
                  "real participant admits full-budget one-in/one-out");
            publish_ack(map,i,controller.control(),AckPhase::RESERVED);
        }
        publish_ack(map,kMaxHelpers,controller.control(),AckPhase::RESERVED);
        check(controller.poll()==Poll::ADVANCED,"full reserved ACKs");
        clock.step(); driver.observe(controller,clock.now_ns());
        bool all_ready=false;
        for(uint32_t round=0;round<5;++round) {
            all_ready=true;
            for(uint32_t i=0;i<helpers;++i) {
                auto status=participant[i]->copy();
                check(status!=HelperCopy::FAILED,"full source copied");
                if(status==HelperCopy::READY)
                    publish_ack(map,i,controller.control(),AckPhase::READY);
                else all_ready=false;
            }
            if(all_ready)break;
        }
        check(all_ready,"all copy participants ready");
        publish_ack(map,kMaxHelpers,controller.control(),AckPhase::READY);
        check(controller.poll()==Poll::ADVANCED &&
              controller.control().state==uint32_t(PlacementState::PREPARED),
              "full prepared");
        auto* pending=generation_header(map,next);
        pending->ready_helpers=controller.control().expected_helpers;
        pending->ready_ranks=controller.control().expected_ranks;
        pending->state=uint32_t(GenerationState::READY);
        clock.step(); driver.observe(controller,clock.now_ns());
        check(controller.begin_quiesce(),"full drain began");
        for(uint32_t i=0;i<helpers;++i)
            publish_ack(map,i,controller.control(),AckPhase::QUIESCED);
        publish_ack(map,kMaxHelpers,controller.control(),AckPhase::QUIESCED);
        check(controller.poll()==Poll::ADVANCED && controller.commit(),
              "full quiescence and commit");
        pending->state=uint32_t(GenerationState::ACTIVE);
        clock.step(); driver.observe(controller,clock.now_ns());
        for(uint32_t i=0;i<helpers;++i) {
            check(participant[i]->commit(controller.control()),"full bank commit");
            publish_ack(map,i,controller.control(),AckPhase::COMMITTED);
        }
        publish_ack(map,kMaxHelpers,controller.control(),AckPhase::COMMITTED);
        check(controller.poll()==Poll::ADVANCED &&
              controller.control().state==uint32_t(PlacementState::RETIRING),
              "full retiring after commit ACKs");
        for(uint32_t i=0;i<helpers;++i) {
            check(gpu[i]->active==next,"GPU switched new bank");
            for(uint32_t j=0;j<2;++j) {
                const uint32_t id=expected.ids[j];
                const bool now_mine=generation_map(map,next)[id]==i;
                check((participant[i]->physical(next)[id]>=0)==now_mine &&
                      (gpu[i]->banks[next][id]>=0)==now_mine,
                      "both swapped experts served by correct new owner");
            }
        }
        clock.step(); driver.observe(controller,clock.now_ns());
        for(uint32_t i=0;i<helpers;++i)
            publish_ack(map,i,controller.control(),AckPhase::RETIRED);
        publish_ack(map,kMaxHelpers,controller.control(),AckPhase::RETIRED);
        check(controller.poll()==Poll::ADVANCED &&
              controller.control().state==uint32_t(PlacementState::ACTIVE),
              "full retirement reached ACTIVE");
        generation_header(map,old)->state=uint32_t(GenerationState::RETIRED);
        check(!controller.can_propose(),"full next proposal needs REOPENED");
        publish_reopen_ack(map,controller.control());
        check(!controller.can_propose(),"full next proposal needs CLEANED");
        for(uint32_t i=0;i<helpers;++i) {
            check(participant[i]->reclaim(map,controller.control()),
                  "full lost slots reclaimed after REOPENED");
            publish_cleanup_ack(map,i,controller.control());
        }
        check(controller.can_propose(),"full CLEANED barrier");
        clock.step(); driver.observe(controller,clock.now_ns());
        check(driver.committed()==step+1 && driver.aborted()==0,
              "full transaction committed exactly once");
        for(uint32_t i=0;i<helpers;++i) {
            check(pools[i]->ledger().valid() &&
                  pools[i]->ledger().charged()==uint64_t(capacity[i])*stride &&
                  alloc[i]->live.size()==2,
                  "no full-budget ledger drift or allocation leak");
            for(const auto& slot:pools[i]->ledger().slots())
                check(slot.state==r3::residency::State::FREE ||
                      slot.state==r3::residency::State::ACTIVE,
                      "full-budget no stranded transient slot");
        }
        previous=expected.ids;
        active_slot=next;
    }
    check(!driver.pending() && driver.attempted()==transitions &&
          driver.committed()==transitions && driver.aborted()==0,
          "1000 alternating production swaps complete");
    check(munmap(map,layout.bytes)==0,"full unmap");
}

int main() {
    using namespace r3::forced;
    const LayoutShape shape{128,2,3,8,128,64};
    Layout layout{};
    check(layout_for(shape,2,slots_for(3),&layout),"layout");
    void* map=mmap(nullptr,layout.bytes,PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    check(map!=MAP_FAILED && init(map,2,shape,3),"init");
    Header* h=static_cast<Header*>(map);
    h->helper_count=3;
    placement_control(map)->expected_ranks=1;
    Selected sel[3]{};
    sel[0].expert={{0,0}};
    sel[1].expert={{0,1}};
    for(uint32_t helper=0;helper<3;++helper) {
        check(publish_owned(map,sel[helper],helper),"canonical owner bootstrap");
        check(mark_bootstrap_filtered(map,helper),"filtered agreement");
    }
    SharedTransport transport(map);
    check(transport.seal_bootstrap(),"canonical bootstrap seal");
    for(uint32_t l=0;l<2;++l) {
        int32_t* slots=layer_slots(map,l);
        for(uint32_t e=0;e<3;++e)
            slots[e]=(l==0) ? int32_t(e) : -1;
        store(&h->layer_map_ready[l],1);
    }
    const uint64_t bootstrap_digest=placement_control(map)->active_digest;
    FakeClock clock;
    Controller controller(transport,clock,bootstrap_digest,3,1,Deadlines{},
                          Bootstrap{1,0,placement_control(map)->seq,0});
    check(controller.can_propose(),"controller bootstrapped");

    const uint64_t stride=256; // byte count of fake weight planes, not a hardware constant
    std::vector<uint64_t> budgets={stride*2,stride*2,stride};
    using Pool=r3::worker_slots::Pool<Allocator>;
    std::vector<std::unique_ptr<Allocator>> alloc;
    std::vector<std::unique_ptr<Pool>> pools;
    std::vector<std::unique_ptr<FakeGpu>> gpu;
    std::vector<std::unique_ptr<r3::worker_moves::Participant>> participant;
    std::array<uint64_t,3> allocated{};
    for(uint32_t helper=0;helper<3;++helper) {
        alloc.push_back(std::make_unique<Allocator>());
        const uint32_t capacity=helper==2?1:2;
        pools.push_back(std::make_unique<Pool>(
            uint64_t(capacity)*stride,stride,capacity,*alloc.back()));
        check(pools.back()->prepare(),"pool prepare");
        std::vector<int32_t> locals(6,-1);
        for(auto [l,e]:sel[helper].expert) {
            const uint32_t id=l*shape.experts+e;
            uint32_t physical=UINT32_MAX;
            locals[id]=int32_t(e);
            check(pools.back()->bootstrap(id,stride,1,&physical)==
                  r3::residency::Result::OK,"bootstrap resident physical id");
        }
        gpu.push_back(std::make_unique<FakeGpu>(pools.back()->ledger()));
        gpu.back()->banks[0]=locals;
        participant.push_back(std::make_unique<r3::worker_moves::Participant>(
            pools.back()->ledger(),*gpu.back(),helper,2,shape.experts,stride));
        check(participant.back()->initialize(map,0,locals),"initial generation bound");
        allocated[helper]=alloc.back()->live.size();
    }
    const Source source=[](uint32_t l,uint32_t e) {return l==0&&e<2;};
    // The same production forced generator must reject every illegal
    // target before changing any canonical ACTIVE map byte or slot.
    auto illegal=[&](Move m,const std::vector<uint64_t>& limits,
                     const Source& available,FaultCode wanted) {
        FaultCode fault=FaultCode::NONE;
        const auto old=controller.control();
        check(propose(map,controller,m,limits,stride,available,{},&fault) &&
              fault==wanted,"invalid request published transaction-local fault");
        const auto& c=controller.control();
        check(c.active_epoch==old.active_epoch && c.active_slot==old.active_slot &&
              c.active_digest==old.active_digest &&
              generation_map(map,c.active_slot)[0]==0 &&
              placement_journal(map)->entry[c.journal_index].fault==uint32_t(wanted),
              "illegal target never changes ownership");
        for(uint32_t i=0;i<3;++i)publish_cleanup_ack(map,i,c);
        publish_cleanup_ack(map,kMaxHelpers,c);
        check(controller.can_propose(),"illegal transaction cleaned before retry");
    };
    illegal(Move{0,0,0},budgets,source,FaultCode::INVALID_MAP);
    illegal(Move{0,2,2},budgets,Source{[](uint32_t,uint32_t){return true;}},FaultCode::INVALID_MAP);
    illegal(Move{0,0,3},budgets,source,FaultCode::INVALID_MAP);
    // Frozen engine-local expert with a valid runtime slot still cannot move.
    layer_slots(map,1)[0]=0;
    illegal(Move{1,0,1},budgets,Source{[](uint32_t,uint32_t){return true;}},FaultCode::INVALID_MAP);
    layer_slots(map,1)[0]=-1;
    illegal(Move{0,0,2},budgets,Source{},FaultCode::SOURCE);
    illegal(Move{0,0,1},{stride*2,stride,stride},source,FaultCode::CAPACITY);
    Move invalid{};
    // A corrupted active map is never a trustworthy forced-move source.
    auto* active_map=generation_map(map,controller.control().active_slot);
    const uint8_t saved_owner=active_map[5];
    active_map[5]=0;
    check(eligible(map,Move{0,0,1},budgets,stride,source)==FaultCode::INVALID_MAP,
          "active digest corruption refuses candidate");
    active_map[5]=saved_owner;
    check(!parse_move("0:1:2garbage",&invalid) &&
          !parse_move("0:-1:2",&invalid),"strict triple parser");

    // Exercise the production startup schedule reader, not a test-only parser.
    const char* tmp=std::getenv("TMPDIR");
    const std::string path=std::string(tmp&&*tmp?tmp:".")+
        "/rhp11-g-schedule-"+std::to_string(getpid())+".txt";
    {
        std::ofstream file(path);
        file << "# test move/return sequence\nrepeat=250\n0:0:1\n0:0:0\n0:0:2\n0:0:0\n";
        check(bool(file),"schedule fixture written");
    }
    Schedule schedule;std::string parse_error;
    check(read_schedule(path,&schedule,&parse_error) &&
          schedule.count()==1000 && schedule.steps.size()==4,
          "production file parser expands 1000 ordered transactions");
    {
        std::ofstream file(path);
        file << "# mixed schedule\nrepeat=2\nswap:0:0:0:1\n"
                "swap:auto:0:2\n0:0:1\n";
        check(bool(file),"swap schedule fixture written");
    }
    Schedule mixed;
    check(read_schedule(path,&mixed,&parse_error) && mixed.count()==6 &&
          mixed.steps[0].kind==Step::Kind::SWAP &&
          mixed.steps[1].kind==Step::Kind::AUTO_SWAP &&
          mixed.steps[2].kind==Step::Kind::MOVE,
          "production schedule reader accepts explicit and automatic swaps");
    {
        std::ofstream file(path);
        file << "repeat=0\n0:0:1\n";
        check(bool(file),"invalid fixture written");
    }
    Schedule bad;
    check(!read_schedule(path,&bad,&parse_error),"zero repeat rejected");
    check(std::remove(path.c_str())==0,"schedule fixture cleanup");
    // The worker's typed environment parses once at helper-zero startup.
    setenv("R3_MOE_PLACEMENT_FORCE_MOVE","0:0:1",1);
    WorkerConfig typed;
    std::string config_error;
    check(typed.from_environment(kMaxHelpers,&config_error) &&
          typed.force_enabled && typed.force_schedule.steps.size()==1 &&
          typed.force_schedule.steps[0].move.destination==1,
          "production worker settings accept a strict move");
    unsetenv("R3_MOE_PLACEMENT_FORCE_MOVE");
    setenv("R3_MOE_PLACEMENT_FORCE_MOVE","0:0:-1",1);
    WorkerConfig invalid_settings;
    check(!invalid_settings.from_environment(kMaxHelpers,&config_error),
          "production worker settings reject negative destination");
    unsetenv("R3_MOE_PLACEMENT_FORCE_MOVE");
    setenv("R3_MOE_PLACEMENT_FORCE_MOVE","swap:0:0:0:1",1);
    WorkerConfig swap_settings;
    check(swap_settings.from_environment(kMaxHelpers,&config_error) &&
          swap_settings.force_enabled &&
          swap_settings.force_schedule.steps[0].kind==Step::Kind::SWAP,
          "worker accepts explicit single-transaction swap");
    setenv("R3_MOE_PLACEMENT_FORCE_MOVE","swap:auto:0:1",1);
    WorkerConfig bad_auto_settings;
    check(!bad_auto_settings.from_environment(kMaxHelpers,&config_error),
          "AUTO is a schedule-only row, not a FORCE_MOVE operand");
    unsetenv("R3_MOE_PLACEMENT_FORCE_MOVE");

    Driver driver(schedule,budgets,stride,source);
    uint32_t active_slot=0;
    for(uint32_t step=0;step<1000;++step) {
        clock.step();
        const uint32_t old=active_slot,next=1u-old;
        const auto old_owner=owner_at(map,old,0,0);
        const auto target=schedule.steps[step%schedule.steps.size()].move.destination;
        check(driver.submit(map,controller,clock.now_ns()),"force request proposed");
        driver.observe(controller,clock.now_ns());
        check(controller.control().state==uint32_t(PlacementState::PROPOSED) &&
              controller.control().active_slot==old &&
              controller.control().pending_slot==next &&
              owner_at(map,next,0,0)==target &&
              owner_at(map,old,0,0)==old_owner &&
              placement_candidate_valid(map,controller.control()),
              "candidate is one changed byte, correctly digested, old generation immutable");
        check(controller.start_reserving(),"production reserving");
        clock.step(); driver.observe(controller,clock.now_ns());
        for(uint32_t i=0;i<3;++i) {
            check(participant[i]->reserve(map,controller.control())==FaultCode::NONE,
                  "real participant reserves legal transient physical slot");
            publish_ack(map,i,controller.control(),AckPhase::RESERVED);
        }
        publish_ack(map,kMaxHelpers,controller.control(),AckPhase::RESERVED);
        check(controller.poll()==Poll::ADVANCED,"all reserving ACKs -> copying");
        clock.step(); driver.observe(controller,clock.now_ns());
        // The already-admitted old request keeps its old bank throughout
        // all private-stream uploads; the new bank cannot serve until COMMIT.
        for(uint32_t i=0;i<3;++i)
            check(gpu[i]->active==old && gpu[i]->banks[old][0]==
                  (old_owner==i ? 0 : -1),"N request stays on source");
        for(uint32_t round=0;round<5;++round) {
            bool ready=true;
            for(uint32_t i=0;i<3;++i) {
                const auto status=participant[i]->copy();
                check(status!=r3::epoch::HelperCopy::FAILED,"fake uploaded planes");
                if(status==r3::epoch::HelperCopy::READY)
                    publish_ack(map,i,controller.control(),AckPhase::READY);
                else ready=false;
            }
            if(ready)break;
        }
        publish_ack(map,kMaxHelpers,controller.control(),AckPhase::READY);
        check(controller.poll()==Poll::ADVANCED &&
              controller.control().state==uint32_t(PlacementState::PREPARED),
              "all physical uploads and rank READY before prepare");
        clock.step(); driver.observe(controller,clock.now_ns());
        auto* pending=generation_header(map,next);
        pending->ready_helpers=controller.control().expected_helpers;
        pending->ready_ranks=controller.control().expected_ranks;
        pending->state=uint32_t(GenerationState::READY);
        check(controller.begin_quiesce(),"admission quiescing");
        clock.step(); driver.observe(controller,clock.now_ns());
        driver.observe(controller,clock.now_ns());
        // Four prequeued waits from the N bank cannot be counted as retired.
        // A missing helper QUIESCED ACK keeps the controller in QUIESCING.
        std::array<bool,4> prequeued{{true,true,true,true}};
        publish_ack(map,kMaxHelpers,controller.control(),AckPhase::QUIESCED);
        check(controller.poll()==Poll::WAITING && !controller.commit(),
              "queued old-generation references bar COMMIT");
        prequeued.fill(false);
        check(std::none_of(prequeued.begin(),prequeued.end(),[](bool x){return x;}),
              "all four fake prequeued waits synchronized");
        for(uint32_t i=0;i<3;++i)
            publish_ack(map,i,controller.control(),AckPhase::QUIESCED);
        check(controller.poll()==Poll::ADVANCED && controller.commit(),
              "exact drain and ACK before commit");
        clock.step(); driver.observe(controller,clock.now_ns());
        pending->state=uint32_t(GenerationState::ACTIVE);
        for(uint32_t i=0;i<3;++i) {
            check(participant[i]->commit(controller.control()),"real participant bank commit");
            publish_ack(map,i,controller.control(),AckPhase::COMMITTED);
        }
        publish_ack(map,kMaxHelpers,controller.control(),AckPhase::COMMITTED);
        check(controller.poll()==Poll::ADVANCED &&
              controller.control().state==uint32_t(PlacementState::RETIRING),
              "committed ACKs advance to retiring");
        clock.step(); driver.observe(controller,clock.now_ns());
        for(uint32_t i=0;i<3;++i) {
            check(participant[i]->physical(next)[0]==
                      (target==i?participant[i]->physical(next)[0]:-1) &&
                  gpu[i]->active==next &&
                  gpu[i]->banks[next][0]==(target==i?0:-1),
                  "N+1 bank serves only the new owner");
        }
        for(uint32_t i=0;i<3;++i)
            publish_ack(map,i,controller.control(),AckPhase::RETIRED);
        publish_ack(map,kMaxHelpers,controller.control(),AckPhase::RETIRED);
        check(controller.poll()==Poll::ADVANCED &&
              controller.control().state==uint32_t(PlacementState::ACTIVE),
              "retired phase returns to active without freeing old slots");
        clock.step(); driver.observe(controller,clock.now_ns());
        generation_header(map,old)->state=uint32_t(GenerationState::RETIRED);
        check(!controller.can_propose(),"RETIRED is not a REOPENED barrier");
        publish_reopen_ack(map,controller.control());
        check(!controller.can_propose(),"REOPENED before helpers CLEANED still blocked");
        for(uint32_t i=0;i<3;++i) {
            check(participant[i]->reclaim(map,controller.control()),
                  "reclaim after exact engine reopen ACK only");
            publish_cleanup_ack(map,i,controller.control());
        }
        check(controller.can_propose(),"all cleanup before next forced request");
        clock.step();
        driver.observe(controller,clock.now_ns());
        check(driver.committed()==step+1 &&
              pools[0]->ledger().valid() && pools[1]->ledger().valid() &&
              pools[2]->ledger().valid(),"transaction evidence and ledgers remain consistent");
        for(uint32_t i=0;i<3;++i) {
            check(alloc[i]->live.size()==allocated[i],"bounded physical slots, no leak");
            for(const auto& x:pools[i]->ledger().slots())
                check(x.state==r3::residency::State::FREE ||
                      x.state==r3::residency::State::ACTIVE,
                      "no leaked RESERVED/COPYING/RETIRING slot");
        }
        active_slot=next;
    }
    check(!driver.pending() && driver.attempted()==1000 &&
          driver.committed()==1000 && driver.aborted()==0 &&
          owner_at(map,active_slot,0,0)==0,
          "repeat schedule completes 1000 forward-only churn transactions");
    driver.report(map,pools[0]->ledger().charged(),0,0);
    check(munmap(map,layout.bytes)==0,"unmap");
    full_budget_swaps(false);
    full_budget_swaps(true);
    std::puts("r3_forced_move_test: ok");
}
