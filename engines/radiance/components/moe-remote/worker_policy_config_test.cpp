#include "worker_policy_config.h"
#include "worker_policy_driver.h"
#include <cstdio>
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
    {"R3_MOE_POLICY_DEVICE_COSTS","1,1"},
    {"R3_MOE_POLICY_LAYER_CAPS","0,0"},
    {"R3_MOE_POLICY_PROTECTED_LAYERS","none"}
};
int main() {
    std::string err;
    r3::policy::Settings s;
    unsetenv("R3_MOE_POLICY_ENABLE");
    check(s.from_environment(2,false,&err)&&!s.enabled,
          "off by default and no required parameters");
    check(!s.enabled && !s.accept_unwatermarked_heat,
          "unwatermarked heat off by default");
    r3::policy::Driver off(s,nullptr,{},0,{});
    check(off.start()&&!off.running(),"off never starts thread");
    setenv("R3_MOE_POLICY_ENABLE","1",1);
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
    setenv("R3_MOE_POLICY_DEVICE_COSTS","1",1);
    check(!s.from_environment(2,false,&err),"missing helper costs fails");
    setenv("R3_MOE_POLICY_DEVICE_COSTS","1,1",1);
    setenv("R3_MOE_POLICY_PROTECTED_LAYERS","1,2",1);
    check(s.from_environment(2,false,&err) &&
          s.protected_layers==std::vector<uint32_t>({1,2}),
          "explicit protected layers accepted");
    setenv("R3_MOE_POLICY_PROTECTED_LAYERS","2,1",1);
    check(!s.from_environment(2,false,&err),"unordered protected layers refused");
    setenv("R3_MOE_POLICY_PROTECTED_LAYERS","none",1);
    setenv("R3_MOE_POLICY_LAYER_CAPS","-1,0",1);
    check(!s.from_environment(2,false,&err),"negative helper cap fails");
    puts("policy config: settings matrix and mutual exclusion passed");
}
