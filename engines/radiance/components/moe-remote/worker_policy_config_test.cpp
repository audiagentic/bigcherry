#include "worker_policy_config.h"
#include "worker_policy_driver.h"
#include <cstdio>
#include "worker_epoch.h"
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

// The live policy is typed from the manifest (spec.handoff.placement.live_policy). The schema and the reader already
// refuse a missing or out-of-range member (v1 reader tests); this checks the conversion and the invariants the policy
// relies on, and that a manifest without a policy never starts one.
static void check(bool b,const char* reason) {
    if(!b) {std::fprintf(stderr,"policy config FAIL: %s\n",reason);std::exit(1);}
}
using bc::serving::v1::LivePolicy;
static LivePolicy valid() {
    LivePolicy p{};
    p.effect_gate=true;p.accept_unwatermarked_heat=false;
    p.cadence_ns=10000;p.max_moves=2;p.max_moved_bytes=20;p.min_residency_ns=1;p.max_heat_age_ns=100000000;
    p.min_samples=1;p.abort_cooldown_ns=20000;p.learned_eligibility_ttl_ns=30000;p.effect_window_samples=2;
    p.effect_min_quiet_windows=2;p.effect_revert_ttl_ns=30000;
    p.margin_ppm=50000;p.half_life_observations_milli=1000000;p.effect_noise_multiplier_milli=2000;
    return p;
}
int main() {
    std::string err;
    const std::vector<uint32_t> shares{1,1};
    r3::policy::Settings s;
    check(!s.enabled && !s.accept_unwatermarked_heat && !s.effect_gate,"a policy is off until the manifest has one");
    struct TestClock : r3::epoch::Clock {
        uint64_t now_ns() const override {return 1000000000ULL;}
    } clock;
    r3::policy::Driver off(s,nullptr,{},0,{},clock);
    check(off.start()&&!off.running(),"off never starts thread");

    check(s.from_manifest(valid(),shares,&err)&&s.enabled,"valid explicit controls");
    check(s.effect_gate && !s.accept_unwatermarked_heat && s.cadence_ns==10000 && s.controls.max_moves==2 &&
          s.controls.max_moved_bytes==20 && s.abort_cooldown_ns==20000 && s.learned_eligibility_ttl_ns==30000,
          "integers carried as written");
    check(s.controls.margin==0.05 && s.controls.half_life_observations==1000.0 && s.effect_noise_multiplier==2.0,
          "ppm and milli values become the controls' reals");
    check(s.head_shares==shares && s.protected_layers.empty(),"shares from the manifest, no protected layers");

    {   // every positive member refuses zero, by name
        const struct {const char* name;std::function<void(LivePolicy&)> zero;} cases[]={
            {"cadence_ns",[](LivePolicy& p){p.cadence_ns=0;}},
            {"max_moves",[](LivePolicy& p){p.max_moves=0;}},
            {"max_moved_bytes",[](LivePolicy& p){p.max_moved_bytes=0;}},
            {"min_residency_ns",[](LivePolicy& p){p.min_residency_ns=0;}},
            {"max_heat_age_ns",[](LivePolicy& p){p.max_heat_age_ns=0;}},
            {"min_samples",[](LivePolicy& p){p.min_samples=0;}},
            {"abort_cooldown_ns",[](LivePolicy& p){p.abort_cooldown_ns=0;}},
            {"learned_eligibility_ttl_ns",[](LivePolicy& p){p.learned_eligibility_ttl_ns=0;}},
            {"effect_window_samples",[](LivePolicy& p){p.effect_window_samples=0;}},
            {"effect_min_quiet_windows",[](LivePolicy& p){p.effect_min_quiet_windows=0;}},
            {"effect_revert_ttl_ns",[](LivePolicy& p){p.effect_revert_ttl_ns=0;}},
            {"half_life_observations_milli",[](LivePolicy& p){p.half_life_observations_milli=0;}},
            {"effect_noise_multiplier_milli",[](LivePolicy& p){p.effect_noise_multiplier_milli=0;}},
        };
        for(const auto& c:cases) {
            LivePolicy p=valid();c.zero(p);
            r3::policy::Settings probe;err.clear();
            check(!probe.from_manifest(p,shares,&err)&&err.find(c.name)!=std::string::npos&&!probe.enabled,
                  "a zero control fails closed by name");
        }
    }
    for(int64_t bad:{int64_t(0),int64_t(1000000),int64_t(-5)}) {
        LivePolicy p=valid();p.margin_ppm=bad;
        r3::policy::Settings probe;err.clear();
        check(!probe.from_manifest(p,shares,&err)&&err.find("margin_ppm")!=std::string::npos,"margin outside (0,1) refused");
    }
    {
        LivePolicy p=valid();p.max_moves=int64_t(UINT32_MAX)+1;
        r3::policy::Settings probe;
        check(!probe.from_manifest(p,shares,&err),"max_moves beyond 32 bits refused");
    }
    {
        LivePolicy p=valid();p.accept_unwatermarked_heat=true;p.effect_gate=false;
        r3::policy::Settings probe;
        check(probe.from_manifest(p,shares,&err)&&probe.accept_unwatermarked_heat&&!probe.effect_gate,
              "explicit file-source opt-in and a disabled gate are kept as declared");
    }
    // The live plan is the static placer's, so it reads the helper shares of the launch.
    check(s.from_manifest(valid(),{4,1},&err)&&s.head_shares==std::vector<uint32_t>({4,1}),"launch shares are read as set");
    for(const std::vector<uint32_t>& bad:{std::vector<uint32_t>{},std::vector<uint32_t>{0,0},std::vector<uint32_t>{1000001,1}}) {
        r3::policy::Settings probe;
        check(!probe.from_manifest(valid(),bad,&err),"shares must be one weight a helper, not all zero, not above 1e6");
    }
    {
        LivePolicy p=valid();p.protected_layers={1,2};
        r3::policy::Settings probe;
        check(probe.from_manifest(p,shares,&err)&&probe.protected_layers==std::vector<uint32_t>({1,2}),
              "explicit protected layers accepted");
        p.protected_layers={2,1};
        check(!probe.from_manifest(p,shares,&err),"unordered protected layers refused");
        p.protected_layers={1,1};
        check(!probe.from_manifest(p,shares,&err),"repeated protected layers refused");
        p.protected_layers={-1};
        check(!probe.from_manifest(p,shares,&err),"negative protected layer refused");
    }
    puts("policy config: typed controls and invariants passed");
}
