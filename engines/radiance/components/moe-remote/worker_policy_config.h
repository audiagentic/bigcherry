#pragma once
// Experimental policy controls: each trigger and size must be explicit.
#include "worker_policy_input.h"
#include <charconv>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace r3::policy {
struct Settings {
    bool enabled = false, accept_unwatermarked_heat = false;
    uint64_t cadence_ns = 0, abort_cooldown_ns = 0, learned_eligibility_ttl_ns = 0;
    PolicyControls controls{};
    std::vector<double> costs, layer_caps;
    std::vector<uint32_t> protected_layers;
    static bool read_flag(const char* key,const char* raw, bool* v, std::string* error) {
        if (!raw || !*raw) return true;
        if (std::strcmp(raw,"0") && std::strcmp(raw,"1")) {
            *error=std::string(key)+": expected 0 or 1";return false;
        }
        *v=*raw=='1';return true;
    }
    static bool u64(const char* key, uint64_t* out, std::string* error) {
        const char* s=std::getenv(key);
        if (!s || !*s) {*error=std::string(key)+": required for policy; no default";return false;}
        const auto [p,ec]=std::from_chars(s,s+std::strlen(s),*out);
        if(ec!=std::errc{}||p!=s+std::strlen(s)||!*out) {
            *error=std::string(key)+": expected positive integer";return false;
        }
        return true;
    }
    // ASCII decimal only: no whitespace, hex, inf/nan, separators or
    // locale-specific spellings. Conversion still checks overflow/underflow.
    static bool decimal_token(const char* s,size_t n) {
        size_t i=0;
        if(i<n && s[i]=='+')++i;
        size_t digits=0;
        while(i<n && s[i]>='0' && s[i]<='9') {++i;++digits;}
        if(i<n && s[i]=='.') {
            ++i;
            while(i<n && s[i]>='0' && s[i]<='9') {++i;++digits;}
        }
        if(!digits)return false;
        if(i<n && (s[i]=='e'||s[i]=='E')) {
            ++i;
            if(i<n && (s[i]=='+'||s[i]=='-'))++i;
            const size_t start=i;
            while(i<n && s[i]>='0' && s[i]<='9')++i;
            if(i==start)return false;
        }
        return i==n;
    }
    static bool real(const char* key, double* out, bool can_zero, std::string* error) {
        const char* s=std::getenv(key);
        if (!s || !*s) {*error=std::string(key)+": required for policy; no default";return false;}
        if(!decimal_token(s,std::strlen(s))) {
            *error=std::string(key)+": expected strict finite decimal";return false;
        }
        char* end=nullptr;errno=0;*out=std::strtod(s,&end);
        if(errno||end==s||*end||!std::isfinite(*out)||*out<0||
           (!can_zero && *out==0)) {
            *error=std::string(key)+": expected finite nonnegative decimal";return false;
        }
        return true;
    }
    static bool values(const char* key,size_t count,bool can_zero,
                       std::vector<double>* out,std::string* error) {
        const char* raw=std::getenv(key);
        if (!raw || !*raw) {*error=std::string(key)+": required for policy; no default";return false;}
        std::string s(raw);
        std::vector<double> result;
        for(size_t start=0;;) {
            const size_t end=s.find(',',start);
            const std::string token=s.substr(start,end==std::string::npos?end:end-start);
            if(token.empty()) {*error=std::string(key)+": empty entry";return false;}
            if(!decimal_token(token.c_str(),token.size())) {
                *error=std::string(key)+": expected strict finite decimal";return false;
            }
            char* tail=nullptr;errno=0;const double d=std::strtod(token.c_str(),&tail);
            if(errno||tail==token.c_str()||*tail||!std::isfinite(d)||
               d<0||(!can_zero && d==0)) {
                *error=std::string(key)+": invalid finite decimal";return false;
            }
            result.push_back(d);
            if(end==std::string::npos)break;
            start=end+1;
        }
        if(result.size()!=count) {
            *error=std::string(key)+": expected exactly "+std::to_string(count)+" helpers";return false;
        }
        *out=std::move(result);return true;
    }
    static bool layers(const char* raw,std::vector<uint32_t>* result,std::string* error) {
        const char* key="R3_MOE_POLICY_PROTECTED_LAYERS";
        if(!raw || !*raw) {*error=std::string(key)+": required; specify none or sorted model-global layer IDs";return false;}
        if(std::strcmp(raw,"none")==0) {result->clear();return true;}
        const char* start=raw;
        const char* end=raw+std::strlen(raw);
        std::vector<uint32_t> ids;
        while(start<end) {
            uint32_t id=0;
            const auto parsed=std::from_chars(start,end,id);
            if(parsed.ec!=std::errc{} || parsed.ptr==start ||
               (parsed.ptr!=end && *parsed.ptr!=',') ||
               (!ids.empty() && id<=ids.back())) {
                *error=std::string(key)+": expected sorted, distinct numeric layer IDs";return false;
            }
            ids.push_back(id);
            if(parsed.ptr==end)break;
            start=parsed.ptr+1;
            if(start==end) {
                *error=std::string(key)+": trailing comma";return false;
            }
        }
        *result=std::move(ids);
        return true;
    }
    bool from_environment(uint32_t helpers, bool forced_enabled,
                          std::string* error) {
        if(!error || !read_flag("R3_MOE_POLICY_ENABLE",std::getenv("R3_MOE_POLICY_ENABLE"),&enabled,error) ||
           !read_flag("R3_MOE_POLICY_ACCEPT_UNWATERMARKED_HEAT",
                      std::getenv("R3_MOE_POLICY_ACCEPT_UNWATERMARKED_HEAT"),
                      &accept_unwatermarked_heat,error)) return false;
        if(!enabled)return true; // no policy thread, no activation settings
        if(forced_enabled) {
            *error="R3_MOE_POLICY_ENABLE conflicts with R3_MOE_PLACEMENT_FORCE_MOVE / FORCE_SCHEDULE";
            return false;
        }
        // Read every declared trigger/size by its literal name so the
        // component gate can audit actual getenv accesses. Missing is fatal.
        const struct {const char* key;const char* raw;} required[]={
            {"R3_MOE_POLICY_CADENCE_NS",std::getenv("R3_MOE_POLICY_CADENCE_NS")},
            {"R3_MOE_POLICY_MARGIN",std::getenv("R3_MOE_POLICY_MARGIN")},
            {"R3_MOE_POLICY_MAX_MOVES",std::getenv("R3_MOE_POLICY_MAX_MOVES")},
            {"R3_MOE_POLICY_MAX_MOVED_BYTES",std::getenv("R3_MOE_POLICY_MAX_MOVED_BYTES")},
            {"R3_MOE_POLICY_MIN_RESIDENCY_NS",std::getenv("R3_MOE_POLICY_MIN_RESIDENCY_NS")},
            {"R3_MOE_POLICY_HEAT_HALF_LIFE_OBSERVATIONS",std::getenv("R3_MOE_POLICY_HEAT_HALF_LIFE_OBSERVATIONS")},
            {"R3_MOE_POLICY_MAX_HEAT_AGE_NS",std::getenv("R3_MOE_POLICY_MAX_HEAT_AGE_NS")},
            {"R3_MOE_POLICY_MIN_SAMPLES",std::getenv("R3_MOE_POLICY_MIN_SAMPLES")},
            {"R3_MOE_POLICY_ABORT_COOLDOWN_NS",std::getenv("R3_MOE_POLICY_ABORT_COOLDOWN_NS")},
            {"R3_MOE_POLICY_LEARNED_ELIGIBILITY_TTL_NS",std::getenv("R3_MOE_POLICY_LEARNED_ELIGIBILITY_TTL_NS")},
            {"R3_MOE_POLICY_DEVICE_COSTS",std::getenv("R3_MOE_POLICY_DEVICE_COSTS")},
            {"R3_MOE_POLICY_LAYER_CAPS",std::getenv("R3_MOE_POLICY_LAYER_CAPS")},
            {"R3_MOE_POLICY_PROTECTED_LAYERS",std::getenv("R3_MOE_POLICY_PROTECTED_LAYERS")},
        };
        for(const auto& item:required) if(!item.raw || !*item.raw) {
            *error=std::string(item.key)+": required for policy; no default";
            return false;
        }
        uint64_t moves=0;
        if(!u64("R3_MOE_POLICY_CADENCE_NS",&cadence_ns,error) ||
           !real("R3_MOE_POLICY_MARGIN",&controls.margin,false,error) ||
           !u64("R3_MOE_POLICY_MAX_MOVES",&moves,error) ||
           !u64("R3_MOE_POLICY_MAX_MOVED_BYTES",&controls.max_moved_bytes,error) ||
           !u64("R3_MOE_POLICY_MIN_RESIDENCY_NS",&controls.min_residency_ns,error) ||
           !real("R3_MOE_POLICY_HEAT_HALF_LIFE_OBSERVATIONS",
                 &controls.half_life_observations,false,error) ||
           !u64("R3_MOE_POLICY_MAX_HEAT_AGE_NS",&controls.max_heat_age_ns,error) ||
           !u64("R3_MOE_POLICY_MIN_SAMPLES",&controls.min_samples,error) ||
           !u64("R3_MOE_POLICY_ABORT_COOLDOWN_NS",&abort_cooldown_ns,error) ||
           !u64("R3_MOE_POLICY_LEARNED_ELIGIBILITY_TTL_NS",&learned_eligibility_ttl_ns,error) ||
           !values("R3_MOE_POLICY_DEVICE_COSTS",helpers,false,&costs,error) ||
           !values("R3_MOE_POLICY_LAYER_CAPS",helpers,true,&layer_caps,error) ||
           !layers(std::getenv("R3_MOE_POLICY_PROTECTED_LAYERS"),
                   &protected_layers,error))
            return false;
        if(moves>UINT32_MAX||controls.margin>=1.0||
           cadence_ns > uint64_t(std::chrono::nanoseconds::max().count())) {
            *error="R3_MOE_POLICY_MAX_MOVES / MARGIN / CADENCE_NS: out of range";
            return false;
        }
        controls.max_moves=uint32_t(moves);
        if(!controls_valid(controls)) {
            *error="R3_MOE_POLICY_*: invalid mandatory controls";return false;
        }
        return true;
    }
};
} // namespace r3::policy
