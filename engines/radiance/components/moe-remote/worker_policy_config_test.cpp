#include "worker_policy_config.h"
#include "worker_policy_driver.h"
#include "worker_config.h"
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
using bc::serving::v2::LivePolicy;
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
    // RHP20: even a lab explicit policy cannot bypass the resolver's
    // topology refusal and make the last XTX or an engine rank a tail.
    using r3::policy::live_policy_arrangement_supported;
    check(live_policy_arrangement_supported(1, "tail", {1,1,0}, &err),
          "gfx1100/gfx1030 head-head-tail arrangement is eligible to calibrate");
    check(live_policy_arrangement_supported(1, "tail", {1,0}, &err),
          "one head and one tail are eligible to calibrate");
    for (const auto& bad: std::vector<std::pair<std::string,std::vector<uint64_t>>>{
             {"roundrobin",{1,1}}, {"tail",{1,1}}, {"tail",{0,0}},
             {"tail",{1}}, {"tail",{0,0,0}}, {"head",{1,1,0}}}) {
        err.clear();
        check(!live_policy_arrangement_supported(1,bad.first,bad.second,&err) && !err.empty(),
              "explicit live policy must not make head-only/non-tail topology live");
    }
    err.clear();
    check(!live_policy_arrangement_supported(2,"tail",{1,0},&err) &&
          err.find("multi-rank")!=std::string::npos,
          "TP policy remains fail-closed until owner-generation acceptance");
    err.clear();
    check(!live_policy_arrangement_supported(1,"tail",{1000001,0},&err),
          "invalid high share refused");
    bc::serving::HelperSettings manifest;
    manifest.helper.count=2;
    manifest.helper.index=0;
    manifest.helper.engine_ranks=1;
    manifest.handoff.placement="tail";
    manifest.shares={1,0};
    manifest.live_policy=valid();
    WorkerConfig actual(manifest);
    check(actual.configure_placement(manifest,&err) && actual.policy.enabled,
          "production helper configuration accepts exactly a validated tail policy");
    manifest.handoff.placement="roundrobin";
    WorkerConfig head_only(manifest);
    err.clear();
    check(!head_only.configure_placement(manifest,&err) &&
          err.find("tail strategy")!=std::string::npos &&
          !head_only.policy.enabled,
          "helper startup with explicit policy refuses head-only pre-GPU");
    manifest.handoff.placement="tail";
    manifest.helper.engine_ranks=2;
    WorkerConfig tp(manifest);
    err.clear();
    check(!tp.configure_placement(manifest,&err) &&
          err.find("TP/multi-rank")!=std::string::npos &&
          !tp.policy.enabled,
          "helper startup with explicit policy refuses unverified TP pre-GPU");
    manifest.live_policy.reset();
    check(tp.configure_placement(manifest,&err) && !tp.policy.enabled,
          "unsupported topology still supports static/off policy");

    {   // RHP16: calibrating settings are structural, derived settings equal what the manifest conversion would give.
        r3::policy::Settings c;
        std::string e;
        check(!c.calibrating_from(0,shares,2,10000,&e) && !c.calibrating_from(1000,{1},2,10000,&e) &&
              !c.calibrating_from(1000,shares,2,1000000,&e),"calibrating needs its declared facts");
        check(c.calibrating_from(1000,shares,2,10000,&e) && c.calibrating && c.enabled && !c.effect_gate &&
              c.accept_unwatermarked_heat && c.cadence_ns==1000 && c.controls.max_heat_age_ns==2000 &&
              r3::policy::controls_valid(c.controls),"calibrating settings");
        std::vector<uint64_t> budgets{22ull<<30,22ull<<30};
        nlohmann::json m=nlohmann::json::object();
        m["txn_p50_ns"]=1593403198ull;m["txn_p95_ns"]=3794447513ull;m["copy_bytes_per_s"]=319377397ull;
        m["heat_gap_max_ns"]=6955375341ull;m["heat_watermarked"]=0;m["hot_set_k"]=5931;m["marginal_error_ppm"]=31905;
        m["marginal_observations"]=105169020ull;m["drift_half_life_observations"]=15849692ull;
        m["observation_rate_per_s"]=1282858;m["tail_share_gain_ppm"]=98958;
        m["tail_lane_service_ns_per_s"]=468160374ull;m["tail_lane_row_ns"]=62964;m["lane_lost_ns"]=14766582;
        const auto v=r3::policy::derive::derive_policy_values(m,nlohmann::json::object(),2,10000,2534544,budgets);
        r3::policy::Settings d;
        check(d.derived_from(v,c,&e) && d.enabled && !d.calibrating && d.head_shares==shares &&
              d.cadence_ns==v.cadence_ns && d.controls.max_moves==v.max_moves &&
              d.controls.margin==double(v.margin_ppm)/1e6 && d.effect_gate==v.effect_gate,
              "derived settings carry the derived values and the launch's shares");
        // The same values through the manifest conversion are the same settings (one mapping, two entrances).
        LivePolicy p{};
        p.effect_gate=v.effect_gate;p.accept_unwatermarked_heat=v.accept_unwatermarked_heat;
        p.cadence_ns=int64_t(v.cadence_ns);p.max_moves=int64_t(v.max_moves);p.max_moved_bytes=int64_t(v.max_moved_bytes);
        p.min_residency_ns=int64_t(v.min_residency_ns);p.max_heat_age_ns=int64_t(v.max_heat_age_ns);
        p.min_samples=int64_t(v.min_samples);p.abort_cooldown_ns=int64_t(v.abort_cooldown_ns);
        p.learned_eligibility_ttl_ns=int64_t(v.learned_eligibility_ttl_ns);
        p.effect_window_samples=int64_t(v.effect_window_samples);p.effect_min_quiet_windows=int64_t(v.effect_min_quiet_windows);
        p.effect_revert_ttl_ns=int64_t(v.effect_revert_ttl_ns);p.margin_ppm=int64_t(v.margin_ppm);
        p.half_life_observations_milli=int64_t(v.half_life_observations_milli);
        p.effect_noise_multiplier_milli=int64_t(v.effect_noise_multiplier_milli);
        r3::policy::Settings via_manifest;
        check(via_manifest.from_manifest(p,shares,&e) && via_manifest.cadence_ns==d.cadence_ns &&
              via_manifest.controls.margin==d.controls.margin &&
              via_manifest.controls.half_life_observations==d.controls.half_life_observations &&
              via_manifest.effect_noise_multiplier==d.effect_noise_multiplier &&
              via_manifest.controls.max_moved_bytes==d.controls.max_moved_bytes &&
              via_manifest.effect_gate==d.effect_gate,"derived_from and from_manifest map identically");
        check(r3::policy::derive::policy_digest(v).size()==16 &&
              r3::policy::derive::policy_digest(v)==r3::policy::derive::policy_digest(v),"stable digest");
        auto changed=v;++changed.max_moves;
        check(r3::policy::derive::policy_digest(changed)!=r3::policy::derive::policy_digest(v),"digest sees every value");
    }

    puts("policy config: typed controls and invariants passed");
}
