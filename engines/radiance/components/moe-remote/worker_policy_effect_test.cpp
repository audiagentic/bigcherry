// Card-free measured service gate: production code exercised by synthetic samples.
#include "worker_policy_effect.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
using namespace r3::policy;
static void check(bool ok,const char* why) {
    if(!ok) {std::fprintf(stderr,"effect gate FAIL: %s\n",why);std::exit(1);}
}
struct Synthetic {
    ServiceSnapshot data{};
    uint64_t t=1000;
    void push(double ns_per_row,bool small=true,unsigned n=2) {
        auto& x=data.lane[small?0:1];
        x.samples+=n;x.rows+=n;x.time_ns+=uint64_t(std::llround(ns_per_row*n));
        t+=100;
    }
    void quiet(EffectGate& gate,double rate,bool small=true) {
        push(rate,small);gate.quiet(data,t);
    }
};
static void calibrate(EffectGate& gate,Synthetic& s) {
    gate.quiet(s.data,s.t); // anchor
    s.quiet(gate,100);
    check(!gate.ready(s.t,1000),"one quiet window is insufficient");
    s.quiet(gate,102);
    check(!gate.ready(s.t,1000),"one pair is insufficient");
    s.quiet(gate,101);
    check(gate.ready(s.t,1000),"two consecutive quiet comparisons");
}
// The hardware calib7 regression: helper0's own unchanged service is NOT
// evidence that shifting heat off the tail had no effect. Use the exact v24
// publication, reader and gate math for both before/after comparisons.
static void test_published_tail_signal() {
    using namespace r3::moe_shm;
    const LayoutShape shape{128,2,3,8,128,64};
    Layout layout{};
    check(layout_for(shape,2,2,&layout),"shared tail test layout");
    void* map=mmap(nullptr,layout.bytes,PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    check(map!=MAP_FAILED && init(map,2,shape,2),"shared tail test init");
    static_cast<Header*>(map)->helper_count=2;
    auto* coordinator_slot=published_service(map,0);
    auto* tail_slot=published_service(map,1);
    check(coordinator_slot && tail_slot && !published_service(map,2),
          "per-helper slots follow the active helper count");
    uint64_t now=1000;
    SharedTailServiceSource tail_source(map,1,1000,[&]{return now;});
    SharedTailServiceSource old_helper0(map,0,1000,[&]{return now;});
    ServiceSnapshot sampled{};
    check(!tail_source.snapshot(&sampled),"unpublished tail is unavailable");
    for(const double after : {91.0,110.0}) {
        *coordinator_slot={};
        *tail_slot={};
        ServiceSnapshot own{},tail{};
        EffectGate tail_gate({100,2,2.0});
        EffectGate old_gate({100,2,2.0});
        auto publish=[&] {
            check(SharedTailServiceSource::publish(coordinator_slot,own,now) &&
                  SharedTailServiceSource::publish(tail_slot,tail,now),
                  "publish coherent two-lane snapshots");
        };
        auto sample=[&](EffectGate& gate,SharedTailServiceSource& source) {
            ServiceSnapshot snap{};
            check(source.snapshot(&snap),"read selected helper's fresh slot");
            gate.quiet(snap,now);
        };
        publish();
        sample(tail_gate,tail_source);
        sample(old_gate,old_helper0);
        for(unsigned i=0;i<3;++i) {
            now+=100;
            const double tail_rate=(i==0?100.0:(i==1?101.0:100.0));
            own.lane[1].rows+=100;own.lane[1].samples++;
            own.lane[1].time_ns+=10000;
            tail.lane[1].rows+=100;tail.lane[1].samples++;
            tail.lane[1].time_ns+=uint64_t(tail_rate*100);
            publish();
            sample(tail_gate,tail_source);
            sample(old_gate,old_helper0);
        }
        check(tail_gate.ready(now,1000) && old_gate.ready(now,1000),
              "both sources reach quiet windows");
        check(tail_gate.observed().row_noise_ns()>0 &&
              tail_gate.observed().sampled_rows_per_s>0 &&
              old_gate.observed().row_noise_ns()==0,
              "calibration sigma and row rate come from the judged tail");
        check(tail_gate.begin(now,1000) && old_gate.begin(now,1000),
              "both gates accept a measured baseline");
        ServiceSnapshot tail_baseline{},own_baseline{};
        check(tail_source.snapshot(&tail_baseline) &&
              old_helper0.snapshot(&own_baseline),"captured committed baselines");
        tail_gate.committed(tail_baseline);
        old_gate.committed(own_baseline);
        now+=100;
        tail.lane[1].rows+=100;tail.lane[1].samples++;
        tail.lane[1].time_ns+=uint64_t(after*100);
        own.lane[1].rows+=100;own.lane[1].samples++;
        own.lane[1].time_ns+=10000;
        publish();
        ServiceSnapshot tail_after{},own_after{};
        check(tail_source.snapshot(&tail_after) &&
              old_helper0.snapshot(&own_after),"read committed trial service");
        check(tail_gate.judge(tail_after)==
                  (after<100.0?EffectVerdict::KEPT:EffectVerdict::WORSE),
              "tail's 9% gain is KEPT and 10% slowdown is WORSE");
        check(old_gate.judge(own_after)==EffectVerdict::UNDECIDED,
              "retired coordinator-local signal misses both tail effects");
    }
    // Publication may be interrupted; never use an odd sequence or a
    // timestamp from a stopped (or impossibly future) helper.
    check(SharedTailServiceSource::read(tail_slot,now,1000,&sampled),
          "fresh coherent service still readable");
    __atomic_add_fetch(&tail_slot->sequence,uint64_t{1},__ATOMIC_ACQ_REL);
    __atomic_store_n(&tail_slot->rows[1],UINT64_MAX,__ATOMIC_RELAXED);
    check(!tail_source.snapshot(&sampled),"torn odd snapshot rejected");
    __atomic_add_fetch(&tail_slot->sequence,uint64_t{1},__ATOMIC_RELEASE);
    check(!SharedTailServiceSource::read(tail_slot,now+1001,1000,&sampled),
          "stale publisher rejected");
    check(!SharedTailServiceSource::read(tail_slot,now-1,1000,&sampled),
          "future publisher rejected");
    munmap(map,layout.bytes);
}

int main() {
    test_published_tail_signal();
    LaneServiceSource source;
    ServiceSnapshot observed{};
    check(source.snapshot(&observed),"empty snapshot");
    source.record(0,25,100);
    source.record(r3::moe_shm::lane_of(0,r3::moe_shm::kSmallClasses),50,200);
    check(source.snapshot(&observed),"stable source snapshot");
    check(observed.lane[0].samples==1&&observed.lane[0].rows==25&&
          observed.lane[1].samples==1&&observed.lane[1].rows==50,
          "small and large lane samples never pooled");
    source.record(0,0,120);
    check(source.snapshot(&observed)&&observed.lane[0].samples==1,
          "zero routed work cannot create a misleading sample");

    for(unsigned scenario=0;scenario<3;++scenario) {
        EffectGate g({2,2,2.0});
        Synthetic s;
        check(g.valid(),"valid declared controls");
        check(!g.begin(s.t,1000),"cannot propose without measured noise");
        calibrate(g,s);
        check(g.noise_floor()>0,"quiet pair noise is visible before a trial");
        check(!g.ready(s.t+1001,1000),"stale quiet signal cannot qualify a future move");
        check(g.begin(s.t,1000),"can propose only after noise and pre-window");
        check(!g.begin(s.t,1000),"cannot overlap a trial");
        g.committed(s.data);
        check(g.judge(s.data)==EffectVerdict::WAITING,"no post observations no verdict");
        s.push(scenario==0?116.0:(scenario==1?84.0:102.0),true,1);
        check(g.judge(s.data)==EffectVerdict::WAITING,"partial post window never judged");
        s.push(scenario==0?116.0:(scenario==1?84.0:102.0),true,1);
        const EffectVerdict expect=scenario==0?EffectVerdict::WORSE:
                                   scenario==1?EffectVerdict::KEPT:EffectVerdict::UNDECIDED;
        check(g.judge(s.data)==expect,"regression, improvement and noise classification");
        check(g.last_ratio()>0&&g.noise_floor()>0,"record measured ratio and floor");
        check(!g.ready(s.t,1000),"old calibration cannot straddle a move");
    }
    {
        EffectGate g({2,1,2});
        Synthetic s;
        g.quiet(s.data,s.t);
        s.quiet(g,100);
        s.quiet(g,101);
        check(g.begin(s.t,1000),"minimum one pair calibrated");
        g.committed(s.data);
        s.data.lane[0].samples=0; // restart/reset of the counter
        check(g.judge(s.data)==EffectVerdict::INVALID,
              "counter reset produces no guessed verdict");
        g.abort();
        check(!g.ready(s.t,1000),"abort clears all pre-move noise");
    }
    {
        EffectGate bad({0,0,0});
        check(!bad.valid(),"missing controls fail closed");
    }
    {
        // Hardware regression: requests alternate 27K and 64 routed rows.
        // Their service time per request swings ~420x while per-row service
        // remains near 100 ns. The per-row quiet envelope must stay small.
        EffectGate g({64,2,2.576});
        ServiceSnapshot snap{};
        g.quiet(snap,1000);
        double request_rate[6]={};
        const uint64_t sizes[6]={27000,64,27000,64,27000,64};
        for(unsigned i=0;i<6;++i) {
            const uint64_t per_row=100+(i%2);
            snap.lane[1].samples++;
            snap.lane[1].rows+=sizes[i];
            snap.lane[1].time_ns+=sizes[i]*per_row;
            request_rate[i]=double(sizes[i]*per_row);
            g.quiet(snap,2000+i*1000);
        }
        check(g.ready(7000,10000),"quiet row series reaches READY");
        check(g.noise_floor()>0 && g.noise_floor()<0.02,
              "request-size swings do not inflate per-row noise floor");
        const auto stats=g.observed();
        check(stats.quiet_comparisons==5 && stats.row_noise_ns()>0 &&
              stats.sampled_rows_per_s>0,
              "quiet-window sigma and rate are measured in routed rows");
        check(g.begin(7000,10000),"row-normalised baseline available");
        g.committed(snap);
        snap.lane[1].samples++;
        snap.lane[1].rows+=64;
        snap.lane[1].time_ns+=64*91; // ~9% faster per routed row
        check(g.judge(snap)==EffectVerdict::KEPT,
              "9% per-row improvement must be kept despite request size mix");
        // Replay the very same series under the retired per-request metric:
        // the quiet envelope is dominated by alternating request sizes.
        // Its enormous threshold makes even a 9% real improvement UNDECIDED.
        const double retired_noise=std::abs(request_rate[4]/request_rate[5]-1.0);
        const double retired_ratio=double(64*91)/request_rate[5];
        check(retired_noise>0.70 && retired_ratio<1.0 &&
              retired_ratio>=1.0-2.576*retired_noise &&
              retired_ratio<=1.0+2.576*retired_noise,
              "same 9% gain is undecided with old per-request noise envelope");
    }
    puts("effect gate: published tail, class isolation, noise, good/bad/noise, partial, reset passed");
}
