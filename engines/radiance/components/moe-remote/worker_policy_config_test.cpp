#include "worker_policy_config.h"
#include "worker_policy_driver.h"
#include <cstdio>
#include "worker_epoch.h"
#include <cstdlib>
#include <string>
#include <vector>
#include <utility>

static void check(bool b,const char* reason) {
    if(!b) {std::fprintf(stderr,"policy config FAIL: %s\n",reason);std::exit(1);}
}
static const std::vector<std::pair<const char*,const char*>> required={
    {"R3_MOE_POLICY_CADENCE_NS","10000"},
    {"R3_MOE_POLICY_MARGIN","0.05"},
    {"R3_MOE_POLICY_MAX_MOVES","2"},
    {"R3_MOE_POLICY_MAX_MOVED_BYTES","20"},
    {"R3_MOE_POLICY_MIN_RESIDENCY_NS","1"},
    {"R3_MOE_POLICY_HEAT_HALF_LIFE_OBSERVATIONS","1000"},
    {"R3_MOE_POLICY_MAX_HEAT_AGE_NS","100000000"},
    {"R3_MOE_POLICY_MIN_SAMPLES","1"},
    {"R3_MOE_POLICY_ABORT_COOLDOWN_NS","20000"},
    {"R3_MOE_POLICY_LEARNED_ELIGIBILITY_TTL_NS","30000"},
    {"R3_MOE_POLICY_PROTECTED_LAYERS","none"},
    {"R3_MOE_POLICY_EFFECT_WINDOW_SAMPLES","2"},
    {"R3_MOE_POLICY_EFFECT_MIN_QUIET_WINDOWS","2"},
    {"R3_MOE_POLICY_EFFECT_NOISE_MULTIPLIER","2.0"},
    {"R3_MOE_POLICY_EFFECT_REVERT_TTL_NS","30000"}
};
int main() {
    std::string err;
    r3::policy::Settings s;
    unsetenv("R3_MOE_POLICY_ENABLE");
    check(s.from_environment(2,false,&err)&&!s.enabled,
          "off by default and no required parameters");
    check(!s.enabled && !s.accept_unwatermarked_heat,
          "unwatermarked heat off by default");
    struct TestClock : r3::epoch::Clock {
        uint64_t now_ns() const override {return 1000000000ULL;}
    } clock;
    r3::policy::Driver off(s,nullptr,{},0,{},clock);
    check(off.start()&&!off.running(),"off never starts thread");
    setenv("R3_MOE_POLICY_ENABLE","1",1);
    unsetenv("R3_MOE_POLICY_EFFECT_GATE");
    check(!s.from_environment(2,false,&err)&&
          err.find("R3_MOE_POLICY_EFFECT_GATE")!=std::string::npos,
          "gate choice mandatory when policy enabled");
    setenv("R3_MOE_POLICY_EFFECT_GATE","1",1);
    for(const auto& [k,v]:required)unsetenv(k);
    for(const auto& [k,v]:required)setenv(k,v,1);
    check(s.from_environment(2,false,&err)&&s.enabled,"valid explicit controls");
    for(const auto& [key,value]:required) {
        unsetenv(key);
        err.clear();
        r3::policy::Settings probe;
        check(!probe.from_environment(2,false,&err)&&
              err.find(key)!=std::string::npos,
              "every missing setting must fail closed by name");
        setenv(key,value,1);
    }
    check(!s.from_environment(2,true,&err)&&err.find("conflicts")!=std::string::npos,
          "forced-move exclusion");
    setenv("R3_MOE_POLICY_ACCEPT_UNWATERMARKED_HEAT","1",1);
    check(s.from_environment(2,false,&err)&&s.accept_unwatermarked_heat,
          "explicit file-source opt-in");
    setenv("R3_MOE_POLICY_ACCEPT_UNWATERMARKED_HEAT","yes",1);
    check(!s.from_environment(2,false,&err),"invalid opt-in fails");
    setenv("R3_MOE_POLICY_ACCEPT_UNWATERMARKED_HEAT","0",1);
    setenv("R3_MOE_POLICY_MARGIN","1",1);
    check(!s.from_environment(2,false,&err),"invalid margin fails");
    setenv("R3_MOE_POLICY_MARGIN","0.05",1);
    // The live plan is the static placer's, so it reads the launch's helper shares: unset is every helper 1.
    unsetenv("R3_MOE_WORKER_SHARES");
    check(s.from_environment(2,false,&err) && s.head_shares==std::vector<uint32_t>({1,1}),
          "unset shares weight every helper 1");
    setenv("R3_MOE_WORKER_SHARES","4,1",1);
    check(s.from_environment(2,false,&err) && s.head_shares==std::vector<uint32_t>({4,1}),
          "launch shares are read as set");
    for(const char* bad:{"1","1,1,1","0,0","1,x","-1,1","1,,1"}) {
        setenv("R3_MOE_WORKER_SHARES",bad,1);
        check(!s.from_environment(2,false,&err),"shares must be one weight a helper, not all zero");
    }
    unsetenv("R3_MOE_WORKER_SHARES");
    setenv("R3_MOE_POLICY_PROTECTED_LAYERS","1,2",1);
    check(s.from_environment(2,false,&err) &&
          s.protected_layers==std::vector<uint32_t>({1,2}),
          "explicit protected layers accepted");
    setenv("R3_MOE_POLICY_PROTECTED_LAYERS","2,1",1);
    check(!s.from_environment(2,false,&err),"unordered protected layers refused");
    setenv("R3_MOE_POLICY_PROTECTED_LAYERS","none",1);

    // All 18 controls: policy activation and unwatermarked flag may be
    // absent/empty (=off); effect gate flag is MANDATORY and the other
    // fifteen numeric/list controls have no defaults.
    // Invalid inputs are table-driven to prevent permissive strtod fallback.
    for(const char* key:{"R3_MOE_POLICY_ENABLE",
                        "R3_MOE_POLICY_ACCEPT_UNWATERMARKED_HEAT",
                        "R3_MOE_POLICY_EFFECT_GATE"}) {
        for(const char* bad:{" 1","1 ","0x1","1e999","-1","nan","inf","1,0"}) {
            for(const auto& [k,v]:required)setenv(k,v,1);
            setenv("R3_MOE_POLICY_ENABLE","1",1);
            setenv("R3_MOE_POLICY_ACCEPT_UNWATERMARKED_HEAT","0",1);
            setenv("R3_MOE_POLICY_EFFECT_GATE","1",1);
            setenv(key,bad,1);
            r3::policy::Settings q;err.clear();
            check(!q.from_environment(2,false,&err) &&
                  err.find(key)!=std::string::npos,
                  "flag rejects whitespace/hex/overflow/negative/nan/inf/count");
        }
    }
    setenv("R3_MOE_POLICY_ENABLE","1",1);
    unsetenv("R3_MOE_POLICY_ACCEPT_UNWATERMARKED_HEAT");
    setenv("R3_MOE_POLICY_EFFECT_GATE","1",1);
    r3::policy::Settings flags;err.clear();
    for(const auto& [k,v]:required)setenv(k,v,1);
    check(flags.from_environment(2,false,&err) &&
          !flags.accept_unwatermarked_heat,"empty optional opt-in is disabled");
    const std::vector<std::pair<const char*,std::vector<std::string>>> bad_inputs={
      {"R3_MOE_POLICY_CADENCE_NS",{""," 10","10 ","0x10","18446744073709551616","-1","nan","inf","1,2"}},
      {"R3_MOE_POLICY_MARGIN",{""," 0.5","0.5 ","0x1p-2","1e999","-0.1","nan","inf","0.1,0.2"}},
      {"R3_MOE_POLICY_MAX_MOVES",{""," 2","2 ","0x2","4294967296","-1","nan","inf","1,2"}},
      {"R3_MOE_POLICY_MAX_MOVED_BYTES",{""," 20","20 ","0x20","18446744073709551616","-1","nan","inf","1,2"}},
      {"R3_MOE_POLICY_MIN_RESIDENCY_NS",{""," 1","1 ","0x1","18446744073709551616","-1","nan","inf","1,2"}},
      {"R3_MOE_POLICY_HEAT_HALF_LIFE_OBSERVATIONS",{""," 5","5 ","0x1p2","1e999","-1","nan","inf","1,2"}},
      {"R3_MOE_POLICY_MAX_HEAT_AGE_NS",{""," 10","10 ","0x10","18446744073709551616","-1","nan","inf","1,2"}},
      {"R3_MOE_POLICY_MIN_SAMPLES",{""," 1","1 ","0x1","18446744073709551616","-1","nan","inf","1,2"}},
      {"R3_MOE_POLICY_ABORT_COOLDOWN_NS",{""," 20","20 ","0x20","18446744073709551616","-1","nan","inf","1,2"}},
      {"R3_MOE_POLICY_LEARNED_ELIGIBILITY_TTL_NS",{""," 30","30 ","0x30","18446744073709551616","-1","nan","inf","1,2"}},
      {"R3_MOE_POLICY_PROTECTED_LAYERS",{""," none","none ","0x1","4294967296","-1","nan","inf","0,,1"}},
      {"R3_MOE_POLICY_EFFECT_WINDOW_SAMPLES",{""," 2","2 ","0x2","18446744073709551616","-1","nan","inf","1,2","0"}},
      {"R3_MOE_POLICY_EFFECT_MIN_QUIET_WINDOWS",{""," 2","2 ","0x2","18446744073709551616","-1","nan","inf","1,2","0"}},
      {"R3_MOE_POLICY_EFFECT_NOISE_MULTIPLIER",{""," 2","2 ","0x1p1","1e999","-0.1","nan","inf","1,2","0"}},
      {"R3_MOE_POLICY_EFFECT_REVERT_TTL_NS",{""," 2","2 ","0x2","18446744073709551616","-1","nan","inf","1,2","0"}}
    };
    check(bad_inputs.size()==required.size(),"all 15 required settings in test matrix");
    for(const auto& [key,invalid]:bad_inputs) {
        for(const auto& bad:invalid) {
            for(const auto& [k,v]:required)setenv(k,v,1);
            setenv("R3_MOE_POLICY_ENABLE","1",1);
            setenv("R3_MOE_POLICY_ACCEPT_UNWATERMARKED_HEAT","0",1);
            setenv("R3_MOE_POLICY_EFFECT_GATE","1",1);
            setenv(key,bad.c_str(),1);
            r3::policy::Settings probe;err.clear();
            if(probe.from_environment(2,false,&err) ||
               err.find(key)==std::string::npos) {
                std::fprintf(stderr,"failed reject key=%s input='%s' error='%s'\n",
                             key,bad.c_str(),err.c_str());
                check(false,"strict syntax or overflow must reject by setting name");
            }
        }
    }
    for(const auto& [k,v]:required)setenv(k,v,1);
    r3::policy::Settings verified;err.clear();
    setenv("R3_MOE_POLICY_EFFECT_GATE","0",1);
    check(verified.from_environment(2,false,&err)&&!verified.effect_gate,
          "explicitly disabled gate retains declared settings");
    setenv("R3_MOE_POLICY_EFFECT_GATE","1",1);
    check(verified.from_environment(2,false,&err),"all table entries restored valid");
    puts("policy config: settings matrix and mutual exclusion passed");
}
