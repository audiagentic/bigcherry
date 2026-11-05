#pragma once
// Experimental policy controls: each trigger and size must be explicit.
#include "worker_forced_move.h"
#include "worker_policy_input.h"
#include "worker_share.h"
#include <algorithm>
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
    bool enabled = false, accept_unwatermarked_heat = false, effect_gate = false;
    uint64_t cadence_ns = 0, abort_cooldown_ns = 0, learned_eligibility_ttl_ns = 0;
    uint64_t effect_window_samples = 0, effect_min_quiet_windows = 0, effect_revert_ttl_ns = 0;
    double effect_noise_multiplier = 0;
    PolicyControls controls{};
    std::vector<uint32_t> head_shares;   // the launch's helper shares from the manifest, one weight a helper
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
    /* The RHP policy settings are the one remaining environment read in the helper: bigcherry/v1 does not model them
     * (v1 is closed), so they move into the manifest in v2 (plan item RSC10). The launch's helper shares come from the
     * manifest, not the environment. */
    bool from_environment(const std::vector<uint32_t>& shares, bool forced_enabled,
                          std::string* error) {
        const uint32_t helpers=uint32_t(shares.size());
        if(!error || !read_flag("R3_MOE_POLICY_ENABLE",std::getenv("R3_MOE_POLICY_ENABLE"),&enabled,error) ||
           !read_flag("R3_MOE_POLICY_ACCEPT_UNWATERMARKED_HEAT",
                      std::getenv("R3_MOE_POLICY_ACCEPT_UNWATERMARKED_HEAT"),
                      &accept_unwatermarked_heat,error)) return false;
        if(!enabled)return true; // no policy thread, no activation settings
        const char* gate_raw=std::getenv("R3_MOE_POLICY_EFFECT_GATE");
        if(!gate_raw || !*gate_raw) {
            *error="R3_MOE_POLICY_EFFECT_GATE: required for policy; no default";return false;
        }
        if(!read_flag("R3_MOE_POLICY_EFFECT_GATE",gate_raw,&effect_gate,error))return false;
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
            {"R3_MOE_POLICY_PROTECTED_LAYERS",std::getenv("R3_MOE_POLICY_PROTECTED_LAYERS")},
            {"R3_MOE_POLICY_EFFECT_WINDOW_SAMPLES",std::getenv("R3_MOE_POLICY_EFFECT_WINDOW_SAMPLES")},
            {"R3_MOE_POLICY_EFFECT_MIN_QUIET_WINDOWS",std::getenv("R3_MOE_POLICY_EFFECT_MIN_QUIET_WINDOWS")},
            {"R3_MOE_POLICY_EFFECT_NOISE_MULTIPLIER",std::getenv("R3_MOE_POLICY_EFFECT_NOISE_MULTIPLIER")},
            {"R3_MOE_POLICY_EFFECT_REVERT_TTL_NS",std::getenv("R3_MOE_POLICY_EFFECT_REVERT_TTL_NS")},
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
           !layers(std::getenv("R3_MOE_POLICY_PROTECTED_LAYERS"),
                   &protected_layers,error) ||
           !u64("R3_MOE_POLICY_EFFECT_WINDOW_SAMPLES",&effect_window_samples,error) ||
           !u64("R3_MOE_POLICY_EFFECT_MIN_QUIET_WINDOWS",&effect_min_quiet_windows,error) ||
           !real("R3_MOE_POLICY_EFFECT_NOISE_MULTIPLIER",&effect_noise_multiplier,false,error) ||
           !u64("R3_MOE_POLICY_EFFECT_REVERT_TTL_NS",&effect_revert_ttl_ns,error))
            return false;
        if(moves>UINT32_MAX||controls.margin>=1.0||
           cadence_ns > uint64_t(std::chrono::nanoseconds::max().count())) {
            *error="R3_MOE_POLICY_MAX_MOVES / MARGIN / CADENCE_NS: out of range";
            return false;
        }
        controls.max_moves=uint32_t(moves);
        // The live plan is the static placer's, so it reads the shares the launch placed with (unset means every helper 1).
        uint64_t share_total=0;
        for(uint32_t w:shares) {
            if(w>1000000u) {*error="helper share above 1000000";return false;}
            share_total+=w;
        }
        if(!helpers||helpers>worker_share::kMaxHelperShares||!share_total) {
            *error="helper shares: one weight for each of the "+std::to_string(helpers)+" helpers, not all zero";
            return false;
        }
        head_shares=shares;
        if(!controls_valid(controls)) {
            *error="R3_MOE_POLICY_*: invalid mandatory controls";return false;
        }
        return true;
    }
};

/* Everything the helper still reads from its environment: the RHP policy, forced-move and epoch-cycle controls that
 * bigcherry/v1 does not model. v1 is closed; these become manifest registry entries in v2 (plan item RSC10) and this
 * file is then free of getenv. It is the only file under the components allowed a helper R3_* getenv. */
struct UnmodelledEnvironment {
    static constexpr uint32_t kMaxEpochCycleS = (uint32_t)std::min<uint64_t>(
        UINT32_MAX, (uint64_t)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::nanoseconds::max()).count());
    uint32_t epoch_cycle_s = 0;   /* R3_MOE_WORKER_EPOCH_CYCLE_S (RHP11 slice C validation); 0 = off */
    bool force_enabled = false;
    Settings policy;
    forced::Schedule force_schedule;

    bool read(uint32_t helper_index, const std::vector<uint32_t>& shares, std::string* err) {
        if (!read_epoch_cycle(err) || !read_force(helper_index, err)) return false;
        return policy.from_environment(shares, force_enabled, err);
    }

private:
    bool read_epoch_cycle(std::string* err) {
        const char* s = std::getenv("R3_MOE_WORKER_EPOCH_CYCLE_S");
        if (!s || !*s) return true;
        uint32_t v = 0;
        const char* end = s + std::strlen(s);
        const auto [p, ec] = std::from_chars(s, end, v);
        if (ec != std::errc{} || p != end || v > kMaxEpochCycleS) {
            *err = std::string("R3_MOE_WORKER_EPOCH_CYCLE_S=") + s + ": want an integer in 0.." + std::to_string(kMaxEpochCycleS);
            return false;
        }
        epoch_cycle_s = v;
        return true;
    }
    bool read_force(uint32_t helper_index, std::string* err) {
        // Only the coordinator reads test controls; other helpers merely service canonical proposals.
        if (helper_index != 0) return true;
        const char* one = std::getenv("R3_MOE_PLACEMENT_FORCE_MOVE");
        const char* file = std::getenv("R3_MOE_PLACEMENT_FORCE_SCHEDULE");
        const bool has_one = one && *one, has_file = file && *file;
        if (has_one && has_file) {
            *err = "R3_MOE_PLACEMENT_FORCE_MOVE conflicts with FORCE_SCHEDULE";
            return false;
        }
        if (has_one) {
            forced::Step step;
            if (!forced::parse_step(one, &step) || step.kind == forced::Step::Kind::AUTO_SWAP) {
                *err = "R3_MOE_PLACEMENT_FORCE_MOVE: want layer:expert:helper or swap:layerA:expertA:layerB:expertB";
                return false;
            }
            force_schedule.steps.push_back(step);
        } else if (has_file && !forced::read_schedule(file, &force_schedule, err)) {
            return false;
        }
        force_enabled = has_one || has_file;
        return true;
    }
};
} // namespace r3::policy
