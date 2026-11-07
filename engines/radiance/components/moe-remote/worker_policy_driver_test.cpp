// Card-free production Controller + PolicyDriver fake-device integration.
#include "worker_policy_driver.h"
#include "worker_bootstrap.h"
#include "worker_move_slots.h"
#include <condition_variable>
#include <mutex>
#include <fcntl.h>
#include <filesystem>
#include <unistd.h>
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
    std::atomic<uint64_t> t{1000000000ULL};
    uint64_t now_ns() const override {return t.load();}
    void advance(uint64_t ns) {t.fetch_add(ns);}
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
            auto& v=ack[a];
            const auto previous=v;
            v={};
            if (helper && phase==AckPhase::RESERVED) {
                // One synthetic interval: service ns, wall ns and 20 routed rows.
                v.drained_step=20;
                v.reserved[0]=uint64_t(2000 + 1000*a);
                v.reserved[1]=10000;
            } else if (helper && phase==AckPhase::READY) {
                // Fake participants advertise their own exact ingress bytes
                // and distinct elapsed copy times in the ACK's reserved words.
                const auto* current=generation_map(map,c.active_slot);
                const auto* candidate=generation_map(map,c.pending_slot);
                uint64_t incoming=0;
                for (uint32_t e=0;e<4;++e)
                    if (current[e]!=candidate[e] && candidate[e]==int32_t(a)) ++incoming;
                v.reserved[0]=incoming*10;
                v.reserved[1]=incoming ? 1000+1000*a : 0;
            } else if (previous.transaction_id==c.transaction_id) {
                v.reserved[0]=previous.reserved[0];
                v.reserved[1]=previous.reserved[1];
            }
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
            auto& v=ack[a];const auto previous=v;v={};
            v.reserved[0]=2500; // sampled QUIESCED-to-REOPEN unavailable time
            v.reserved[1]=1;    // marked valid by the real cleanup publication
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
                          std::vector<int32_t>{1,1,500,450};   // hot on the tail helper: the static plan would hold them on the head
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

// Only test access: stage a large learned cache and synchronize with a ready
// result without adding test-only hooks to production policy behavior.
namespace r3::policy {
struct DriverTestAccess {
    static bool ready(Driver& d) {
        std::lock_guard<std::mutex> guard(d.lock_);
        return d.ready_;
    }
    static void seed_cache(Driver& d,uint64_t now) {
        for(uint32_t i=0;i<512;++i)
            d.negatives_[{i,i%2}]={now+i,now+1000000,3};
    }
    static size_t storage(Driver& d) {return d.negatives_.size();}
    static bool measured_worse(Driver& d,uint64_t now,uint64_t ttl) {
        bool any=false;
        for(const auto& [key,entry]:d.negatives_) {
            if(entry.reason!=Driver::kMeasuredWorseReason)return false;
            if(entry.observed_ns!=now || entry.until_ns!=now+ttl)return false;
            any=true;
        }
        return any;
    }
    static bool revert_blocked(Driver& d) {return d.revert_failed_;}
    static uint64_t post_wait(Driver& d) {return d.effect_post_wait_ns();}
    static uint64_t post_deadline(Driver& d) {return d.effect_post_deadline_ns_;}
    static bool post_pending(Driver& d) {return d.effect_.waiting();}
};
}
struct PausedHeat : HeatFake {
    std::mutex lock;
    std::condition_variable cv;
    int entered=0, released=0;
    HeatPoll poll(uint64_t mono,uint64_t real,HeatDelta* out,std::string* error) override {
        std::unique_lock<std::mutex> guard(lock);
        const int index=++entered;
        cv.notify_all();
        cv.wait(guard,[&] {return released>=index;});
        guard.unlock();
        return HeatFake::poll(mono,real,out,error);
    }
    void wait_for_call(int n) {
        std::unique_lock<std::mutex> guard(lock);
        check(cv.wait_for(guard,std::chrono::seconds(2),[&]{return entered>=n;}),
              "controlled source entered target poll");
    }
    void release(int n) {
        std::lock_guard<std::mutex> guard(lock);
        released=n;
        cv.notify_all();
    }
};
struct MissingSourceOps : r3::worker_moves::Operations {
    uint32_t owner;
    explicit MissingSourceOps(uint32_t h):owner(h){}
    int32_t preferred_local(uint32_t expert) const override {
        return expert/2==owner?int32_t(expert%2):-1;
    }
    bool start_upload(uint32_t,uint32_t,uint32_t) override {return false;}
    r3::worker_moves::Event upload_event() override {return r3::worker_moves::Event::FAILED;}
    bool build_bank(uint32_t,const std::vector<int32_t>&) override {return false;}
    bool drain_upload() override {return true;}
    void forget(uint32_t) override {}
    void activate_bank(uint32_t) override {}
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
    s.head_shares={1,1};
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
// Synthetic window totals drive the SAME EffectGate and Controller as runtime.
struct FakeService : ServiceSource {
    ServiceSnapshot now{};
    bool available=true;
    bool snapshot(ServiceSnapshot* out) const override {
        if(!available)return false;
        *out=now;return true;
    }
    void add(double ns_per_row,unsigned n=2) {
        auto& l=now.lane[1]; // live gate's synchronized large-lane class
        l.samples+=n;l.rows+=10*n;
        l.time_ns+=uint64_t(std::llround(ns_per_row*10*n));
    }
};
static Settings measured_settings() {
    Settings s=settings();s.effect_gate=true;
    s.effect_window_samples=2;
    s.effect_min_quiet_windows=2;
    s.effect_noise_multiplier=2.0;
    s.effect_revert_ttl_ns=30000;
    return s;
}
static bool pump_effect(Fixture& f,Driver& d,FakeService& service,int limit=200) {
    for(int i=0;i<limit;++i) {
        service.add(100+(i%3));
        check(d.submit(f.map,*f.controller,f.clock.now_ns()),"effect submit");
        if(f.controller->control().state==uint32_t(PlacementState::PROPOSED))
            return true;
        f.clock.advance(1001);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto& ev=d.evidence();
    std::fprintf(stderr,"effect pump evaluated=%llu proposed=%llu judged=%llu\n",
         (unsigned long long)ev.evaluated,
         (unsigned long long)ev.proposed,
         (unsigned long long)ev.effect_windows_judged);
    for(const auto& [name,count]:ev.reasons)
        std::fprintf(stderr,"effect reason %s=%llu\n",name.c_str(),(unsigned long long)count);
    return false;
}
static bool matches_rejection(const char* got,const char* expected) {
    return got && std::strcmp(got,expected)==0;
}
static void test_calibration_ack_roundtrip() {
    Fixture f;
    auto c=f.controller->control();
    c.transaction_id=42;
    c.pending_epoch=c.active_epoch+1;
    c.pending_slot=1u-c.active_slot;
    c.pending_digest=c.active_digest+1;
    publish_ack(f.map,0,c,AckPhase::RESERVED,4,2000,10000);
    auto a=read_shared_ack(f.map,0);
    check(a.reserved[0]==2000 && a.reserved[1]==10000 && a.drained_step==4,
          "RESERVED carries same-interval service ns, sampled routed rows and wall ns");
    publish_ack(f.map,0,c,AckPhase::READY,0,20,500);
    a=read_shared_ack(f.map,0);
    check(a.reserved[0]==20 && a.reserved[1]==500,
          "READY carries per-helper bytes and elapsed copy ns");
    publish_ack(f.map,0,c,AckPhase::QUIESCED);
    a=read_shared_ack(f.map,0);
    check(a.reserved[0]==20 && a.reserved[1]==500,
          "later phases do not discard READY copy evidence");
    c.state=uint32_t(PlacementState::ACTIVE);
    publish_cleanup_ack(f.map,0,c,125);
    a=read_shared_ack(f.map,0);
    check(a.reserved[0]==125 && a.reserved[1]==1 &&
          a.phase==uint32_t(AckPhase::CLEANED),
          "CLEANED carries actual lane unavailable duration and valid flag");
}

int main() {
    test_calibration_ack_roundtrip();
    {
        Fixture f;
        auto source=std::make_unique<HeatFake>();auto* h=source.get();
        Driver d(settings(),std::move(source),{20,20},10,{0,0,0,0},f.clock);
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
        // Drift: the experts now on the tail helper (the one that left the head and the one that never moved) turn hot.
        const uint32_t tail_other=buddy==2?3u:2u;
        (void)stayed;
        h->drift_a=int(tail_other);h->drift_b=int(returning);   // b is the hotter: the returning expert is the seed
        h->reverse=true;
        f.clock.advance(10001); // permit previously committed experts to satisfy dwell
        check(pump(f,d,true),"heat drift triggers another proposal");
        f.commit();f.clock.advance(1001);
        d.observe(*f.controller,f.map,f.clock.now_ns());
        check(d.evidence().committed==2 && d.evidence().transition_samples==2,
              "second journal closed and transaction durations recorded");
        {
            const Measured m=d.measured();
            check(m.transaction_ns.total()==2 && m.abort_ns.total()==0 && m.hot_set_k==2 &&
                  m.delta_gap_ns.total()>0 && m.accepted_heat_deltas>0 &&
                  !m.accepted_unwatermarked_heat && m.tail.gain_ppm.total()>0 &&
                  m.slowest_helper_copy_bps.total()==2 &&
                  m.slowest_helper_copy_bps.quantile(0.5)>0 &&
                  m.tail_lane_service_ns_per_s.count()==2 &&
                  m.tail_lane_service_ns_per_s.quantile(0.5)>0 &&
                  m.tail_lane_row_ns.count()==2 &&
                  m.tail_lane_row_ns.quantile(0.5)==150 &&
                  m.lane_lost_ns.count()==2 &&
                  m.lane_lost_ns.quantile(0.95)==2500,
                  "the driver measures committed transactions, update gaps and the tail's hot-set size");
        }
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
        Driver d(settings(),std::move(source),{20,20},10,{1,0,0,1},f.clock);
        check(d.start()&&pump(f,d,true),"abort proposal");
        const uint64_t epoch=f.controller->control().active_epoch;
        f.abort();f.clock.advance(1001);
        d.observe(*f.controller,f.map,f.clock.now_ns());
        check(d.evidence().aborted==1&&d.evidence().transition_samples==1&&
              d.evidence().learned_ineligible==2&&
              d.negative_count(f.clock.now_ns())==2,
              "closed abort learns negative cache");
        {
            const Measured m=d.measured();
            check(m.abort_ns.total()==1 && m.transaction_ns.total()==0 && m.copy.bytes==0,
                  "an aborted transaction is timed as an abort and moves no measured bytes");
        }
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
        Driver d(s,std::move(source),{20,20},10,{0,0,0,0},f.clock);
        check(d.start()&&pump(f,d,false,10),"unwatermarked refused by default");
        check(d.evidence().proposed==0,"refusal is fail closed");
        d.stop();
    }
    {
        Fixture f;
        auto s=settings();s.accept_unwatermarked_heat=true;
        auto source=std::make_unique<HeatFake>();source->unwatermarked=true;
        Driver d(s,std::move(source),{20,20},10,{0,0,0,0},f.clock);
        check(d.start()&&pump(f,d,false,10),"fake with forged file label rejected");
        check(d.evidence().proposed==0 && !d.evidence().unwatermarked_heat,
              "source identity cannot be forged through delta.provenance");
        d.stop();
    }
    {
        Fixture f;
        auto s=settings();s.accept_unwatermarked_heat=true;
        auto source=std::make_unique<HeatFake>();
        source->unwatermarked=true;source->unknown=true;
        Driver d(s,std::move(source),{20,20},10,{0,0,0,0},f.clock);
        check(d.start()&&pump(f,d,false,10),"unknown unwatermarked source rejected");
        check(d.evidence().proposed==0,"opt-in never weakens other sources");
        d.stop();
    }

    // Independent proposal guards: each branch must fail in isolation, so
    // removing any one conditional is caught even while others stay enabled.
    {
        r3::heat::Plan plan;
        plan.valid=true;plan.delta.push_back({1,0,1,1.0});
        const uint64_t now=100000;
        const std::map<std::pair<uint32_t,uint32_t>,Driver::Negative> none;
        check(!Driver::proposal_rejection(1,1,now+1000,now,0,plan,none),
              "fresh plan with no bans passes");
        check(matches_rejection(Driver::proposal_rejection(0,1,now+1000,now,0,plan,none),
              "stale_terminal_generation"),"terminal generation guard");
        check(matches_rejection(Driver::proposal_rejection(1,1,now+1000,now,now+10,plan,none),
              "abort_cooldown"),"proposal time cooldown guard");
        check(matches_rejection(Driver::proposal_rejection(1,1,now,now,0,plan,none),
              "heat_expired_before_proposal"),"proposal time heat-age guard");
        auto banned=none;
        banned[{1,1}]={now-10,now+100,3};
        check(matches_rejection(Driver::proposal_rejection(1,1,now+1000,now,0,plan,banned),
              "negative_destination"),"proposal time learned destination ban");
    }
    // B1: B is queued while A evaluates, then frozen after A is ready.
    // After A aborts the ACTIVE map is unchanged, but B is not admissible.
    {
        Fixture f;
        auto src=std::make_unique<PausedHeat>();auto* paused=src.get();
        Driver d(settings(),std::move(src),{20,20},10,{0,0,0,0},f.clock);
        check(d.start(),"paused policy start");
        check(d.submit(f.map,*f.controller,f.clock.now_ns()),"queue A");
        paused->wait_for_call(1);
        f.clock.advance(1001);
        check(d.submit(f.map,*f.controller,f.clock.now_ns()),"queue B while A runs");
        paused->release(1);
        paused->wait_for_call(2);
        check(d.submit(f.map,*f.controller,f.clock.now_ns()),"consume A");
        check(f.controller->control().state==uint32_t(PlacementState::PROPOSED),
              "A proposed while B frozen");
        f.abort();f.clock.advance(1);
        d.observe(*f.controller,f.map,f.clock.now_ns());
        check(d.evidence().aborted==1,"A aborted before B finishes");
        paused->release(2);
        for(int i=0;i<200&&!r3::policy::DriverTestAccess::ready(d);++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        check(r3::policy::DriverTestAccess::ready(d),"B ready after A abort");
        check(d.submit(f.map,*f.controller,f.clock.now_ns()),"consume stale B");
        check(d.evidence().proposed==1 &&
              d.evidence().reasons.count("stale_terminal_generation"),
              "B cannot bypass closed abort");
        d.stop();
    }
    // B2: evaluation completes with fresh heat; control thread then misses
    // the absolute deadline before it consumes the ready result.
    {
        Fixture f;
        auto src=std::make_unique<HeatFake>();
        Driver d(settings(),std::move(src),{20,20},10,{0,0,0,0},f.clock);
        check(d.start() && d.submit(f.map,*f.controller,f.clock.now_ns()),
              "queue delayed-ready result");
        for(int i=0;i<200&&!r3::policy::DriverTestAccess::ready(d);++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        check(r3::policy::DriverTestAccess::ready(d),"ready before delay");
        f.clock.advance(settings().controls.max_heat_age_ns+1);
        check(d.submit(f.map,*f.controller,f.clock.now_ns()),"submit after deadline");
        check(d.evidence().proposed==0 &&
              d.evidence().reasons.count("heat_expired_before_proposal"),
              "expired ready result never reaches Controller");
        d.stop();
    }
    // S1: actual helper Participant with no immutable source for incoming
    // expert refuses RESERVING and the existing Controller abort is learned.
    {
        Fixture f;
        auto src=std::make_unique<HeatFake>();
        Driver d(settings(),std::move(src),{20,20},10,{0,0,0,0},f.clock);
        check(d.start()&&pump(f,d,true),"optimistic policy proposal");
        r3::residency::Ledger l0(30,0,10),l1(30,0,10);
        uint32_t id=0;
        check(l0.bootstrap(0,10,1,&id)==r3::residency::Result::OK &&
              l0.bootstrap(1,10,1,&id)==r3::residency::Result::OK &&
              l1.bootstrap(2,10,1,&id)==r3::residency::Result::OK &&
              l1.bootstrap(3,10,1,&id)==r3::residency::Result::OK,"real resident slots");
        MissingSourceOps o0(0),o1(1);
        r3::worker_moves::Participant p0(l0,o0,0,1,4,10),p1(l1,o1,1,1,4,10);
        check(p0.initialize(f.map,f.controller->control().active_slot,{0,1,-1,-1}) &&
              p1.initialize(f.map,f.controller->control().active_slot,{-1,-1,0,1}),
              "real Participants accept frozen ACTIVE slots");
        check(f.controller->start_reserving(),"forward to real receivers");
        const auto a=p0.reserve(f.map,f.controller->control());
        const auto b=p1.reserve(f.map,f.controller->control());
        check(a==FaultCode::SOURCE || b==FaultCode::SOURCE,
              "missing target preferred_local refused at reserve");
        check(f.controller->fail(FaultCode::SOURCE),"forward abort");
        if(p0.transaction())check(p0.abort(),"participant 0 cleanup");
        if(p1.transaction())check(p1.abort(),"participant 1 cleanup");
        f.bridge->close(f.controller->control(),false);
        check(f.controller->poll_cleanup()&&f.controller->can_propose(),"forward abort fully closed");
        d.observe(*f.controller,f.map,f.clock.now_ns());
        check(d.evidence().aborted==1 && d.negative_count(f.clock.now_ns())>0,
              "real source refusal learned by policy");
        check(d.evidence().proposed==1,"speculative abort rate limited by cooldown");
        d.stop();
    }
    // Stopping an in-flight policy worker never publishes or overwrites the
    // Controller transaction; stop joins and closed abort observation remains safe.
    {
        Fixture f;
        auto src=std::make_unique<HeatFake>();
        Driver d(settings(),std::move(src),{20,20},10,{0,0,0,0},f.clock);
        check(d.start()&&pump(f,d,true),"inflight stop proposal");
        const uint64_t txn=f.controller->control().transaction_id;
        d.stop();
        check(!d.running() && f.controller->control().transaction_id==txn &&
              f.controller->control().state==uint32_t(PlacementState::PROPOSED),
              "stop does not mutate transaction");
        f.abort();
        d.observe(*f.controller,f.map,f.clock.now_ns());
        check(d.evidence().aborted==1,"post-stop journal observed");
    }
    // S4: an oversized historical cache prints at most a config-derived
    // number of newest entries, and expired entries are pruned on submit.
    {
        Fixture f;
        auto src=std::make_unique<HeatFake>();
        Driver d(settings(),std::move(src),{20,20},10,{0,0,0,0},f.clock);
        const uint64_t now=f.clock.now_ns();
        r3::policy::DriverTestAccess::seed_cache(d,now);
        FILE* output=std::tmpfile();
        check(output!=nullptr,"report capture temporary file");
        std::fflush(stderr);
        const int original=dup(fileno(stderr));
        check(original>=0 && dup2(fileno(output),fileno(stderr))>=0,"redirect report");
        d.report(f.map,0,now);
        std::fflush(stderr);
        check(dup2(original,fileno(stderr))>=0,"restore stderr");
        close(original);
        std::rewind(output);
        std::string captured;char line[512];
        while(std::fgets(line,sizeof(line),output))captured+=line;
        std::fclose(output);
        size_t details=0,pos=0;
        while((pos=captured.find("policy learned expert=",pos))!=std::string::npos)
            {++details;++pos;}
        check(details==settings().controls.max_moves &&
              captured.find("cache_active=512")!=std::string::npos &&
              captured.find("expert=511")!=std::string::npos,
              "large cache prints bounded newest entries only");
        f.clock.advance(1000001);
        check(d.start() && d.submit(f.map,*f.controller,f.clock.now_ns()),
              "cleanup after expiry");
        check(r3::policy::DriverTestAccess::storage(d)==0,
              "expired negative cache physically pruned");
        d.stop();
    }

    // S5: a configured policy with a silent producer emits a periodic, explicit
    // no-live-feed heartbeat and carries the state in its evidence report.
    {
        Fixture f;
        auto src=std::make_unique<HeatFake>();
        Driver d(settings(),std::move(src),{20,20},10,{0,0,0,0},f.clock);
        check(d.start(),"start for missing live feed");
        f.clock.advance(settings().controls.max_heat_age_ns+settings().cadence_ns);
        FILE* output=std::tmpfile();
        check(output!=nullptr,"feed heartbeat capture");
        std::fflush(stderr);const int original=dup(fileno(stderr));
        check(original>=0 && dup2(fileno(output),fileno(stderr))>=0,"redirect feed diagnostic");
        d.heartbeat(f.clock.now_ns());
        d.report(f.map,0,f.clock.now_ns());
        std::fflush(stderr);check(dup2(original,fileno(stderr))>=0,"restore feed stderr");
        close(original);
        std::rewind(output);std::string captured;char line[512];
        while(std::fgets(line,sizeof(line),output))captured+=line;
        std::fclose(output);
        check(captured.find("policy no_live_feed=1 window_ns=")!=std::string::npos &&
              captured.find("R3_MOE_WORKER_HISTOGRAM_PERIOD_S")!=std::string::npos &&
              captured.find("unwatermarked_heat=0 no_live_feed=1")!=std::string::npos,
              "silent producer heartbeat is explicit and bounded");
        d.stop();
    }
    // Authentic file source remains eligible for the explicit unwatermarked
    // experiment; the same provenance claimed by HeatFake is not eligible.
    {
        Fixture f;
        auto s=settings();s.accept_unwatermarked_heat=true;
        s.controls.max_heat_age_ns=5000000000ULL;
        const std::string temp=std::filesystem::temp_directory_path().string()+
                               "/policy_histogram_XXXXXX";
        std::vector<char> path(temp.begin(),temp.end());path.push_back(0);
        const int first=mkstemp(path.data());
        check(first>=0,"histogram temp file");
        close(first);
        const std::string name(path.data());
        auto write_snapshot=[&](const std::vector<uint32_t>& first_row) {
            std::string staged=name+".XXXXXX";
            std::vector<char> next(staged.begin(),staged.end());next.push_back(0);
            const int fd=mkstemp(next.data());
            check(fd>=0,"histogram atomic stage file");
            std::vector<uint32_t> cells(size_t(kMaxLayers)*4,0);
            for(size_t i=0;i<first_row.size();++i)cells[i]=first_row[i];
            const size_t bytes=cells.size()*sizeof(uint32_t);
            check(::write(fd,cells.data(),bytes)==ssize_t(bytes),"write full histogram");
            check(::close(fd)==0 && ::rename(next.data(),name.c_str())==0,
                  "publish atomic histogram");
        };
        write_snapshot({0,0,0,0});
        auto source=std::make_unique<HistogramFileSource>(name,1,4);
        Driver d(s,std::move(source),{20,20},10,{0,0,0,0},f.clock);
        check(d.start()&&pump(f,d,false,15),"actual histogram baseline");
        check(d.evidence().reasons.count("heat_baseline")>0,"file baseline captured");
        write_snapshot({1,1,500,450});
        check(pump(f,d,true,200),"true HistogramFileSource admits opted-in delta");
        check(d.evidence().unwatermarked_heat,
              "genuine source allowed and evidence stays unwatermarked");
        d.stop();
        check(::unlink(name.c_str())==0,"histogram temp cleanup");
    }
    // Real Controller integration and synthetic per-lane service counters.
    // With no completed quiet window a plan must NEVER reach PROPOSED.
    {
        Fixture f;FakeService service;
        auto source=std::make_unique<HeatFake>();
        Driver d(measured_settings(),std::move(source),{20,20},10,
                 {0,0,0,0},f.clock,&service);
        check(d.start(),"effect enabled with real service source");
        check(pump(f,d,false,14) && d.evidence().proposed==0,
              "no proposal without minimum consecutive quiet windows");
        const auto before=f.owners();
        check(pump_effect(f,d,service),"measured noise permits forward proposal");
        { const auto m=d.measured();
          check(m.effect.quiet_comparisons>=2 && m.effect.normalized_noise_ns.count()>=2 &&
                m.effect.quiet_delta_ns.max()>0 && m.effect.sampled_rows_per_s>0,
                "driver captures actual quiet service duration, counts and sampled rate");
        }
        f.commit();f.clock.advance(1001);
        d.observe(*f.controller,f.map,f.clock.now_ns());
        const auto after=f.owners();
        check(after.owner!=before.owner,"forward moved exactly through Controller");
        check(d.submit(f.map,*f.controller,f.clock.now_ns()) &&
              d.evidence().effect_windows_judged==0,
              "no verdict without enough postcommit routed rows before derived deadline");
        service.add(125,2);
        check(d.submit(f.map,*f.controller,f.clock.now_ns()),
              "postcommit regression drives proposal");
        check(f.controller->control().state==uint32_t(PlacementState::PROPOSED) &&
              d.evidence().effect_revert_attempts==1 &&
              d.evidence().effect_windows_judged==1,
              "worse result sends inverse through real Controller::propose");
        check(DriverTestAccess::measured_worse(
                  d,f.clock.now_ns(),measured_settings().effect_revert_ttl_ns),
              "cache stores distinct measured-worse reason and exact TTL");
        f.commit();f.clock.advance(1001);
        d.observe(*f.controller,f.map,f.clock.now_ns());
        const auto returned=f.owners();
        check(returned.owner==before.owner && returned.epoch==before.epoch+2 &&
              d.evidence().effect_reverted==1,
              "verified inverse restores ENTIRE previous ACTIVE ownership");
        check(d.negative_count(f.clock.now_ns())==2,
              "worse destinations are blacklisted with declared TTL");
        d.stop();
    }
    {
        Fixture f;FakeService service;
        auto source=std::make_unique<HeatFake>();
        Driver d(measured_settings(),std::move(source),{20,20},10,
                 {0,0,0,0},f.clock,&service);
        check(d.start()&&pump_effect(f,d,service),"good move proposed");
        f.commit();f.clock.advance(1001);d.observe(*f.controller,f.map,f.clock.now_ns());
        const auto after=f.owners();
        service.add(75,2);
        check(d.submit(f.map,*f.controller,f.clock.now_ns()),"good move judged");
        check(d.evidence().effect_windows_judged==1 &&
              d.evidence().effect_kept==1 &&
              d.evidence().effect_reverted==0 &&
              f.owners().owner==after.owner,
              "measured faster service keeps owner map");
        d.stop();
    }
    {
        Fixture f;FakeService service;
        auto source=std::make_unique<HeatFake>();
        Driver d(measured_settings(),std::move(source),{20,20},10,
                 {0,0,0,0},f.clock,&service);
        check(d.start()&&pump_effect(f,d,service),"noise move proposed");
        f.commit();f.clock.advance(1001);d.observe(*f.controller,f.map,f.clock.now_ns());
        const auto after=f.owners();
        service.add(101,2);
        check(d.submit(f.map,*f.controller,f.clock.now_ns()),"noise move judged");
        check(d.evidence().effect_undecided==1 &&
              d.evidence().effect_windows_judged==1 &&
              d.evidence().effect_reverted==0 &&
              f.owners().owner==after.owner,
              "noise-sized changes undecided and NOT reverted");
        d.stop();
    }
    {
        Fixture f;FakeService service;
        auto source=std::make_unique<HeatFake>();
        Driver d(measured_settings(),std::move(source),{20,20},10,
                 {0,0,0,0},f.clock,&service);
        check(d.start()&&pump_effect(f,d,service),"abort test forward proposal");
        f.commit();f.clock.advance(1001);d.observe(*f.controller,f.map,f.clock.now_ns());
        const auto degraded=f.owners();
        service.add(125,2);
        check(d.submit(f.map,*f.controller,f.clock.now_ns()) &&
              d.evidence().effect_revert_attempts==1,"revert proposed");
        f.abort();f.clock.advance(1001);d.observe(*f.controller,f.map,f.clock.now_ns());
        check(d.evidence().effect_revert_aborts==1 &&
              f.owners().owner==degraded.owner &&
              DriverTestAccess::revert_blocked(d),
              "revert abort retains ACTIVE and blocks another attempt");
        check(DriverTestAccess::measured_worse(
                  d,f.clock.now_ns()-1001,measured_settings().effect_revert_ttl_ns),
              "rollback abort cannot overwrite the measured-worse exclusion");
        const uint64_t proposals=d.evidence().proposed;
        f.clock.advance(60000);
        service.add(130,2);
        check(pump(f,d,false,12)&&d.evidence().proposed==proposals,
              "revert abort blocks further attempts; no retry loop");
        d.stop();
    }
    // Even with effect_gate=true, losing the post-move source must never
    // deadlock the planner. Heat deltas and drift continue DURING the wait
    // and AFTER an UNDECIDED timeout; no regression is invented.
    {
        Fixture f;FakeService service;
        auto source=std::make_unique<HeatFake>();
        auto* heat=source.get();
        auto gate_settings=measured_settings();
        gate_settings.effect_window_samples=40; // enough post time for several heat deltas
        Driver d(gate_settings,std::move(source),{20,20},10,
                 {0,0,0,0},f.clock,&service);
        check(DriverTestAccess::post_wait(d)==gate_settings.controls.max_heat_age_ns,
              "unmeasured post window falls back to heat-age limit");
        check(d.start()&&pump_effect(f,d,service),"missing post source forward proposal");
        service.available=false; // unavailable even at transaction close
        f.commit();f.clock.advance(1001);d.observe(*f.controller,f.map,f.clock.now_ns());
        const auto after=f.owners();
        const uint64_t start=f.clock.now_ns();
        const uint64_t wait=DriverTestAccess::post_wait(d);
        check(wait>0 && wait<UINT64_MAX/2 && DriverTestAccess::post_pending(d) &&
              DriverTestAccess::post_deadline(d)==start+wait,
              "closed forward transaction freezes finite derived post deadline");
        const auto m_before=d.measured();
        const uint64_t evaluated_before=d.evidence().evaluated;
        heat->reverse=true; // heat shifts again while verdict is unavailable
        // A few evaluations before expiry must still poll the source.
        for(int i=0;i<4 && f.clock.now_ns()+1001<start+wait;++i) {
            check(d.submit(f.map,*f.controller,f.clock.now_ns()),"waiting driver submit");
            f.clock.advance(1001);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const auto m_during=d.measured();
        check(d.evidence().effect_undecided==0 &&
              m_during.accepted_heat_deltas>m_before.accepted_heat_deltas &&
              m_during.drift_intervals>m_before.drift_intervals,
              "effect WAITING still consumes heat deltas and tracks drift");
        if(f.clock.now_ns()<start+wait)f.clock.advance(start+wait-f.clock.now_ns());
        check(d.submit(f.map,*f.controller,f.clock.now_ns()),"deadline submit");
        check(d.evidence().effect_undecided==1 &&
              d.evidence().reasons.count("effect_post_signal_timeout")==1 &&
              !DriverTestAccess::post_pending(d) &&
              !DriverTestAccess::revert_blocked(d) &&
              f.owners().owner==after.owner &&
              d.evidence().effect_reverted==0 &&
              d.negative_count(f.clock.now_ns())==0,
              "missing post source settles UNDECIDED, no revert, no block");
        for(int i=0;i<24;++i) {
            f.clock.advance(1001);
            check(d.submit(f.map,*f.controller,f.clock.now_ns()),"next cadence submit");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const auto m_after=d.measured();
        check(m_after.accepted_heat_deltas>m_before.accepted_heat_deltas+1 &&
              m_after.drift_intervals>m_before.drift_intervals &&
              d.evidence().evaluated>evaluated_before,
              "gate state never blocks subsequent heat consumption and drift tracking");
        d.stop();
    }
    // A tail lane without any peer reference is structurally unjudgeable:
    // resolve immediately, irrespective of apparent degradation on that lane.
    {
        Fixture f;FakeService service;
        auto source=std::make_unique<HeatFake>();
        Driver d(measured_settings(),std::move(source),{20,20},10,
                 {0,0,0,0},f.clock,&service);
        check(d.start()&&pump_effect(f,d,service),"no-peer forward proposal");
        f.commit();f.clock.advance(1001);d.observe(*f.controller,f.map,f.clock.now_ns());
        const auto after=f.owners();
        service.now.unjudgeable_tail_lane[1]=true;
        service.add(150,2);
        check(d.submit(f.map,*f.controller,f.clock.now_ns()),"no-peer submit");
        check(d.evidence().effect_undecided==1 &&
              d.evidence().reasons.count("effect_post_no_peer_reference")==1 &&
              d.evidence().effect_revert_attempts==0 &&
              !DriverTestAccess::post_pending(d) &&
              f.owners().owner==after.owner,
              "tail lane without peer is UNDECIDED not WORSE");
        d.stop();
    }
    // A regressed/reset post counter is INVALID evidence, not a reason to
    // permanently disable the driver or mark the move as WORSE.
    {
        Fixture f;FakeService service;
        auto source=std::make_unique<HeatFake>();
        Driver d(measured_settings(),std::move(source),{20,20},10,
                 {0,0,0,0},f.clock,&service);
        check(d.start()&&pump_effect(f,d,service),"invalid post forward proposal");
        f.commit();f.clock.advance(1001);d.observe(*f.controller,f.map,f.clock.now_ns());
        service.now.lane[1].rows=0; // below the saved post anchor
        check(d.submit(f.map,*f.controller,f.clock.now_ns()),"invalid post submit");
        check(d.evidence().effect_undecided==1 &&
              d.evidence().reasons.count("effect_post_signal_invalid")==1 &&
              !DriverTestAccess::revert_blocked(d) &&
              d.evidence().effect_revert_attempts==0,
              "invalid post signal resolves UNDECIDED, no permanent deadlock");
        d.stop();
    }
    {
        Fixture f;FakeService service;
        auto source=std::make_unique<HeatFake>();
        Driver old(measured_settings(),std::move(source),{20,20},10,
                   {0,0,0,0},f.clock,&service);
        check(old.start()&&pump_effect(f,old,service),"restart old driver proposal");
        f.commit();f.clock.advance(1001);old.observe(*f.controller,f.map,f.clock.now_ns());
        old.stop();
        // A new driver has no learned exclusions, last move timestamps or
        // calibrated noise, even though ACTIVE survived in shared memory.
        auto next_source=std::make_unique<HeatFake>();
        FakeService next_service;
        Driver restarted(measured_settings(),std::move(next_source),{20,20},10,
                         {0,0,0,0},f.clock,&next_service);
        check(restarted.start() && pump(f,restarted,false,10) &&
              restarted.evidence().proposed==0 &&
              restarted.negative_count(f.clock.now_ns())==0,
              "restart forgets process-local noise and history safely");
        restarted.stop();
    }
    {
        Fixture f;
        auto s=measured_settings();
        auto source=std::make_unique<HeatFake>();
        Driver no_source(s,std::move(source),{20,20},10,{0,0,0,0},f.clock);
        check(!no_source.start(),"gate enabled needs a source");
        s.effect_gate=false; // explicit opt-out, preserve old policy behavior
        auto next_source=std::make_unique<HeatFake>();
        Driver off(s,std::move(next_source),{20,20},10,{0,0,0,0},f.clock);
        check(off.start(),"gate off does not require a signal source");
        FILE* output=std::tmpfile();check(output!=nullptr,"report temp file");
        const int original=dup(fileno(stderr));check(original>=0,"stderr backup");
        check(dup2(fileno(output),fileno(stderr))>=0,"capture effect gate off");
        off.report(f.map,0,f.clock.now_ns());
        std::fflush(stderr);check(dup2(original,fileno(stderr))>=0,"restore stderr");
        close(original);std::rewind(output);
        std::string captured;char line[1024];
        while(std::fgets(line,sizeof(line),output))captured+=line;
        std::fclose(output);
        check(captured.find("policy effect_gate=0")!=std::string::npos,
              "explicitly off MUST appear in evidence");
        off.stop();
    }
    puts("policy driver: commit, drift, abort/cache, age, opt-in passed");
}
