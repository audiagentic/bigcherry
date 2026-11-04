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
    bool allocate(uint64_t bytes,void** out) {
        *out=::operator new(size_t(bytes),std::nothrow);
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
    explicit FakeGpu(r3::residency::Ledger& ledger):ledger(ledger){}
    int32_t preferred_local(uint32_t id) const override {
        // 2 layers, 3 experts; expert 2 and every expert in layer 1
        // are engine-local and cannot be uploaded as a regular expert.
        return id<2 ? int32_t(id) : -1;
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
        check(propose(map,controller,m,limits,stride,available,&fault) &&
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
    illegal(Move{1,0,1},budgets,source,FaultCode::INVALID_MAP);
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
          typed.force_schedule.steps[0].destination==1,
          "production worker settings accept a strict move");
    unsetenv("R3_MOE_PLACEMENT_FORCE_MOVE");
    setenv("R3_MOE_PLACEMENT_FORCE_MOVE","0:0:-1",1);
    WorkerConfig invalid_settings;
    check(!invalid_settings.from_environment(kMaxHelpers,&config_error),
          "production worker settings reject negative destination");
    unsetenv("R3_MOE_PLACEMENT_FORCE_MOVE");

    Driver driver(schedule,budgets,stride,source);
    uint32_t active_slot=0;
    for(uint32_t step=0;step<1000;++step) {
        clock.step();
        const uint32_t old=active_slot,next=1u-old;
        const auto old_owner=owner_at(map,old,0,0);
        const auto target=schedule.steps[step%schedule.steps.size()].destination;
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
    driver.report(map,pools[0]->ledger().charged(),0);
    check(munmap(map,layout.bytes)==0,"unmap");
    std::puts("r3_forced_move_test: ok");
}
