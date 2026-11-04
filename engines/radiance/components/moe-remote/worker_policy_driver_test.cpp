// Card-free production Controller + PolicyDriver fake-device integration.
#include "worker_policy_driver.h"
#include "worker_bootstrap.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#include <thread>
using namespace r3::policy;
using namespace r3::moe_shm;
using namespace r3::epoch;
static void check(bool b,const char* message) {
    if(!b) {std::fprintf(stderr,"policy driver FAIL: %s\n",message);std::exit(1);}
}
struct ClockTest : Clock {
    uint64_t t=1000000000ULL;
    uint64_t now_ns() const override {return t;}
    void advance(uint64_t ns) {t+=ns;}
};
struct Bridge : Transport {
    void* map;
    std::array<PlacementAck,kMaxHelpers+kMaxEngineRanks> ack{};
    explicit Bridge(void* p):map(p){}
    PlacementAck read_ack(uint32_t a) const override {return ack.at(a);}
    void publish(const PlacementControl& c,const PlacementJournal& j) override {
        SharedTransport(map).publish(c,j);
    }
    void all(const PlacementControl& c,AckPhase phase) {
        for(uint32_t a=0;a<kMaxHelpers+kMaxEngineRanks;++a) {
            const bool helper=a<kMaxHelpers;
            const uint32_t bit=1u<<(helper?a:a-kMaxHelpers);
            if(!(helper?c.expected_helpers:c.expected_ranks)&bit)continue;
            auto& v=ack[a];v={};
            v.transaction_id=c.transaction_id;
            v.epoch=c.pending_epoch;v.digest=c.pending_digest;
            v.generation_slot=c.pending_slot;v.phase=uint32_t(phase);
        }
    }
    void close(const PlacementControl& c,bool commit) {
        for(uint32_t a=0;a<kMaxHelpers+kMaxEngineRanks;++a) {
            const bool helper=a<kMaxHelpers;
            const uint32_t bit=1u<<(helper?a:a-kMaxHelpers);
            if(!(helper?c.expected_helpers:c.expected_ranks)&bit)continue;
            auto& v=ack[a];v={};
            v.transaction_id=c.transaction_id;
            v.epoch=c.active_epoch;v.digest=c.active_digest;
            v.generation_slot=c.active_slot;
            v.phase=uint32_t(!helper && commit && a==kMaxHelpers?
                              AckPhase::REOPENED:AckPhase::CLEANED);
        }
    }
};
struct HeatFake : HeatSource {
    std::atomic<bool> stale{false},flat{false},reverse{false},unwatermarked{false},unknown{false};
    std::atomic<int> drift_a{1},drift_b{2};
    uint64_t count=0;
    HeatPoll poll(uint64_t,uint64_t,HeatDelta* out,std::string*) override {
        std::vector<int32_t> v=flat.load()?std::vector<int32_t>{10,10,10,10}:
                          std::vector<int32_t>{500,450,1,1};
        if(reverse.load()) {
            v={1,1,1,1};
            v[size_t(drift_a.load())]=450;
            v[size_t(drift_b.load())]=900;
        }
        uint64_t sum=0;for(int32_t x:v)sum+=uint64_t(x);
        count+=sum;
        out->layers=1;out->experts=4;out->counts=v;
        out->interval_observations=sum;out->observation_count=count;
        out->complete_cells=true;
        out->complete_observations=!unwatermarked.load();
        out->provenance=unknown.load()?HeatProvenance::UNKNOWN:HeatProvenance::ENGINE_HISTOGRAM_FILE;
        out->age_ns=stale.load()?100000000ULL:1ULL;
        return HeatPoll::DELTA;
    }
};
struct Fixture {
    LayoutShape shape{128,1,4,8,128,64};
    Layout layout{};
    void* map=nullptr;
    ClockTest clock;
    Bridge* bridge=nullptr;
    std::unique_ptr<Controller> controller;
    Fixture() {
        check(layout_for(shape,2,slots_for(2),&layout),"layout");
        map=mmap(nullptr,layout.bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        check(map!=MAP_FAILED && init(map,1,shape,2),"init");
        auto* h=static_cast<Header*>(map);
        store(&h->helper_count,2u);
        for(uint32_t e=0;e<4;++e)layer_slots(map,0)[e]=int32_t(e);
        store(&h->layer_map_ready[0],1u);
        struct Owned {std::vector<std::pair<uint32_t,uint32_t>> expert;};
        check(publish_owned(map,Owned{{{0,0},{0,1}}},0)&&
              publish_owned(map,Owned{{{0,2},{0,3}}},1),"bootstrap owned");
        check(mark_bootstrap_filtered(map,0)&&mark_bootstrap_filtered(map,1),"filtered");
        SharedTransport temp(map);
        check(temp.seal_bootstrap(),"seal");
        PlacementControl c{};
        check(placement_snapshot(map,&c)&&c.active_digest,"bootstrap control");
        bridge=new Bridge(map);
        controller=std::make_unique<Controller>(*bridge,clock,c.active_digest,2,1,
                      Deadlines{},Bootstrap{c.active_epoch,c.active_slot,c.seq,c.transaction_id});
        check(controller->can_propose(),"initial controller");
    }
    ~Fixture() {controller.reset();delete bridge;munmap(map,layout.bytes);}
    void commit() {
        auto& c=*controller;auto& t=*bridge;
        check(c.start_reserving(),"reserve");
        t.all(c.control(),AckPhase::RESERVED);
        check(c.poll()==Poll::ADVANCED,"reserve acks");
        t.all(c.control(),AckPhase::READY);
        check(c.poll()==Poll::ADVANCED,"copy acks");
        auto* gh=generation_header(map,c.control().pending_slot);
        gh->state=uint32_t(GenerationState::READY);
        gh->ready_helpers=c.control().expected_helpers;
        gh->ready_ranks=c.control().expected_ranks;
        check(c.begin_quiesce(),"quiesce");
        t.all(c.control(),AckPhase::QUIESCED);
        check(c.poll()==Poll::ADVANCED&&c.commit(),"commit");
        t.all(c.control(),AckPhase::COMMITTED);
        check(c.poll()==Poll::ADVANCED,"commit acks");
        t.all(c.control(),AckPhase::RETIRED);
        check(c.poll()==Poll::ADVANCED,"retire acks");
        generation_header(map,c.control().active_slot)->state=
            uint32_t(GenerationState::ACTIVE);
        generation_header(map,1u-c.control().active_slot)->state=
            uint32_t(GenerationState::RETIRED);
        t.close(c.control(),true);
        check(c.poll_cleanup()&&c.can_propose(),"closed commit");
    }
    void abort() {
        auto& c=*controller;
        check(c.start_reserving()&&c.fail(FaultCode::CAPACITY),"fake device abort");
        bridge->close(c.control(),false);
        check(c.poll_cleanup()&&c.can_propose(),"closed abort");
    }
    OwnerSnapshot owners() {
        OwnerSnapshot o;
        check(capture_owners(map,&o),"capture owner generation");
        return o;
    }
};
static Settings settings() {
    Settings s;s.enabled=true;s.cadence_ns=1000;
    s.abort_cooldown_ns=20000;s.learned_eligibility_ttl_ns=30000;
    s.controls.half_life_observations=1;
    s.controls.margin=0.01;s.controls.min_residency_ns=1;
    s.controls.max_heat_age_ns=10000;s.controls.max_moved_bytes=20;
    s.controls.min_samples=1;s.controls.max_moves=2;
    s.costs={1,1};s.layer_caps={0,0};
    return s;
}
static bool pump(Fixture& f,Driver& d,bool want_move,int limit=200) {
    for(int i=0;i<limit;++i) {
        check(d.submit(f.map,*f.controller,f.clock.now_ns()),"submit callback");
        if(f.controller->control().state==uint32_t(PlacementState::PROPOSED))
            return true;
        f.clock.advance(1001);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if(want_move) {
        const auto& ev=d.evidence();
        std::fprintf(stderr,"pump diagnostic evaluated=%llu nochange=%llu proposed=%llu\n",
            (unsigned long long)ev.evaluated,(unsigned long long)ev.no_change,
            (unsigned long long)ev.proposed);
        for(auto& [key,value]:ev.reasons)
            std::fprintf(stderr,"reason %s %llu\n",key.c_str(),(unsigned long long)value);
    }
    return !want_move;
}
int main() {
    {
        Fixture f;
        auto source=std::make_unique<HeatFake>();auto* h=source.get();
        Driver d(settings(),std::move(source),{20,20},10,{0,0,0,0});
        check(d.start()&&d.running(),"background worker started only when enabled");
        h->stale=true;
        check(pump(f,d,false,12)&&d.evidence().proposed==0,
              "stale source cannot propose");
        h->stale=false;h->flat=true;
        check(pump(f,d,false,12)&&d.evidence().proposed==0,
              "flat balanced source cannot propose");
        h->flat=false;
        const auto before=f.owners();
        check(pump(f,d,true),"skewed heat proposes through Controller");
        check(before.owner==std::vector<int32_t>({0,0,1,1}) &&
              generation_map(f.map,f.controller->control().active_slot)[0]==0,
              "ACTIVE untouched before commit");
        const uint64_t digest=before.digest;
        f.commit();
        f.clock.advance(1001);
        d.observe(*f.controller,f.map,f.clock.now_ns());
        const auto after=f.owners();
        check(after.digest!=digest && after.epoch==before.epoch+1 &&
              after.owner!=before.owner,"commit changes ACTIVE");
        check(after.owner.size()==4 &&
              std::all_of(after.owner.begin(),after.owner.end(),
                          [](int32_t x){return x==0||x==1;}),
              "every expert has exactly one owner");
        check(d.evidence().committed==1 && d.evidence().bytes_moved==20,
              "committed bytes and journal history");
        uint32_t returning=0, buddy=0, stayed=0;
        for(uint32_t i=0;i<2;++i) {
            if(after.owner[i]!=before.owner[i])returning=i;
            else stayed=i;
        }
        for(uint32_t i=2;i<4;++i)
            if(after.owner[i]!=before.owner[i])buddy=i;
        check(after.owner[returning]!=before.owner[returning] &&
              after.owner[buddy]!=before.owner[buddy],
              "drift identifies both exchanged experts");
        h->drift_a=int(buddy);h->drift_b=int(stayed);
        h->reverse=true;
        f.clock.advance(10001); // permit previously committed experts to satisfy dwell
        check(pump(f,d,true),"heat drift triggers another proposal");
        f.commit();f.clock.advance(1001);
        d.observe(*f.controller,f.map,f.clock.now_ns());
        check(d.evidence().committed==2 && d.evidence().transition_samples==2,
              "second journal closed and transaction durations recorded");
        const auto back=f.owners();
        check(back.owner!=after.owner &&
              (back.owner[returning]==before.owner[returning] ||
               back.owner[buddy]==before.owner[buddy]),
              "new heat drives an expert back to its original owner");
        d.stop();check(!d.running(),"joined background thread");
    }
    {
        Fixture f;
        auto source=std::make_unique<HeatFake>();
        Driver d(settings(),std::move(source),{20,20},10,{1,0,0,1});
        check(d.start()&&pump(f,d,true),"abort proposal");
        const uint64_t epoch=f.controller->control().active_epoch;
        f.abort();f.clock.advance(1001);
        d.observe(*f.controller,f.map,f.clock.now_ns());
        check(d.evidence().aborted==1&&d.evidence().transition_samples==1&&
              d.evidence().learned_ineligible==2&&
              d.negative_count(f.clock.now_ns())==2,
              "closed abort learns negative cache");
        check(f.owners().epoch==epoch,"abort retains ACTIVE");
        const uint64_t proposals=d.evidence().proposed;
        check(pump(f,d,false,10)&&d.evidence().proposed==proposals,
              "abort cooldown blocks immediate retry");
        f.clock.advance(11000); // cooldown elapsed; cache has not
        check(d.negative_count(f.clock.now_ns())==2 &&
              pump(f,d,false,5)&&d.evidence().proposed==proposals,
              "learned eligibility excludes exact rejected destinations");
        f.clock.advance(30001);
        check(d.negative_count(f.clock.now_ns())==0,"negative cache expiry");
        check(pump(f,d,true),"reproposal only after TTL and cooldown expiry");
        d.stop();
    }
    {
        Fixture f;
        auto s=settings();
        auto source=std::make_unique<HeatFake>();source->unwatermarked=true;
        Driver d(s,std::move(source),{20,20},10,{0,0,0,0});
        check(d.start()&&pump(f,d,false,10),"unwatermarked refused by default");
        check(d.evidence().proposed==0,"refusal is fail closed");
        d.stop();
    }
    {
        Fixture f;
        auto s=settings();s.accept_unwatermarked_heat=true;
        auto source=std::make_unique<HeatFake>();source->unwatermarked=true;
        Driver d(s,std::move(source),{20,20},10,{0,0,0,0});
        check(d.start()&&pump(f,d,true),"explicit experimental file opt-in");
        check(d.evidence().unwatermarked_heat,"evidence flags unwatermarked use");
        d.stop();
    }
    {
        Fixture f;
        auto s=settings();s.accept_unwatermarked_heat=true;
        auto source=std::make_unique<HeatFake>();
        source->unwatermarked=true;source->unknown=true;
        Driver d(s,std::move(source),{20,20},10,{0,0,0,0});
        check(d.start()&&pump(f,d,false,10),"unknown unwatermarked source rejected");
        check(d.evidence().proposed==0,"opt-in never weakens other sources");
        d.stop();
    }
    puts("policy driver: commit, drift, abort/cache, age, opt-in passed");
}
