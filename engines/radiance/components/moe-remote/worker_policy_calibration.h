#pragma once
// Pure RHP16 lab evidence serialization. No GPU, clock, filesystem, or fabricated measurements.
// An absent field names the unavailable event; configured gate targets are NEVER serialized as samples.
#include "worker_policy_measure.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <map>
#include <vector>

namespace r3::policy {
struct CalibrationIdentity {
    std::string plan_cache_key, layout, objective;
};
// No relative service progress for a full measured heat horizon must not
// hold the calibration launcher hostage. A single slot read is not progress:
// both the tail and a peer must have added routed rows in a lane.
class EffectSignalTimeout {
    uint64_t missing_since_=0;
    bool publishers_seen_=false;
public:
    void observe(bool active_publishers,bool overlapping_rows,uint64_t now_ns,
                 bool heat_evidence_ready) {
        if(active_publishers)publishers_seen_=true;
        if(!heat_evidence_ready) {
            missing_since_=0;
            return; // baseline or one accepted delta does not establish a timed feed
        }
        if(!publishers_seen_)return; // do not time out during helper/workload warm-up
        if(overlapping_rows)missing_since_=0;
        else if(!missing_since_ && now_ns)missing_since_=now_ns;
    }
    bool expired(uint64_t now_ns,const Measured& m,uint64_t cadence_ns,
                 uint64_t max_heat_age_ns) const {
        if(!missing_since_ || now_ns<missing_since_ ||
           !m.accepted_heat_deltas || !m.delta_gap_ns.total() ||
           m.delta_gap_ns.max()<=0 || !m.drift_intervals ||
           !cadence_ns || !max_heat_age_ns)return false;
        // A measured accepted heat gap is mandatory. Cadence and freshness
        // remain minimum horizon scales, not substitutes for that gap.
        const long double gap=std::max({static_cast<long double>(m.delta_gap_ns.max()),
                                        static_cast<long double>(cadence_ns),
                                        static_cast<long double>(max_heat_age_ns)});
        const long double span=gap*std::max<size_t>(3,m.drift_capacity);
        return static_cast<long double>(now_ns-missing_since_)>=span;
    }
};
// One entry per unmet PLANNER requirement (effect-only evidence is optional).
// The progress token advances only when evidence for this specific unmet
// requirement advances, not when unrelated heat/transactions continue.
struct MissingPlannerCondition {
    std::string name, reason;
    uint64_t progress = 0;
};
inline std::vector<MissingPlannerCondition> calibration_missing_planner(const Measured& m) {
    std::vector<MissingPlannerCondition> out;
    const auto need = [&](bool ready, const char* name, const char* reason, uint64_t progress=0) {
        if (!ready) out.push_back({name,reason,progress});
    };
    need(m.accepted_heat_deltas>0, "heat_watermarked", "no accepted heat deltas");
    need(m.drift_capacity && m.drift_intervals==m.drift_capacity,
         "drift_window", "retained observation-spaced window not full", m.drift_intervals);
    need(m.delta_gap_ns.total() >= (m.drift_intervals ? m.drift_intervals-1 : 1) &&
         m.delta_gap_ns.max()>0, "heat_gap_max_ns",
         "not enough consecutive accepted heat delta gaps", m.delta_gap_ns.total());
    need(m.hot_set_k>0, "hot_set_k", "no tail-helper expert capacity");
    need(std::isfinite(m.marginal_error) && m.marginal_error>0,
         "marginal_error_ppm", "marginal relative error absent or zero",
         m.marginal_observations);
    need(m.marginal_observations>0, "marginal_observations",
         "no positive accepted source observation count",m.accepted_heat_deltas);
    need(std::isfinite(m.observation_rate_per_s) && m.observation_rate_per_s>0,
         "observation_rate_per_s", "no two accepted counter/time points establish a positive rate",
         m.accepted_heat_deltas);
    need(m.drift_half_life_observations>0, "drift_half_life_observations",
         "half-overlap crossing not observed in retained drift horizon",
         m.drift_horizon_observations);
    need(m.transaction_ns.quantile(0.5)>0, "txn_p50_ns",
         "no positive committed transaction wall time", m.transaction_ns.total());
    need(m.tail.gain_ppm.quantile(0.5)>0, "tail_share_gain_ppm",
         "no committed plan with positive predicted tail-share saving", m.tail.gain_ppm.total());
    need(m.slowest_helper_copy_bps.quantile(0.5)>0, "copy_bytes_per_s",
         "no committed transaction with complete per-receiver READY ACK timings",
         m.slowest_helper_copy_bps.total());
    need(m.tail_lane_service_ns_per_s.quantile(0.5)>0, "tail_lane_service_ns_per_s",
         "no positive sampled tail-lane service per second", m.tail_lane_service_ns_per_s.total());
    need(m.tail_lane_row_ns.quantile(0.5)>0, "tail_lane_row_ns",
         "no positive tail-lane service per routed row", m.tail_lane_row_ns.total());
    need(m.lane_lost_ns.count()>0, "lane_lost_ns",
         "no tail quiesce/reopen interval yet", m.lane_lost_ns.total());
    return out;
}
inline std::string calibration_waiting_message(const std::vector<MissingPlannerCondition>& missing,
                                               bool effect_waiting=false) {
    std::string message = "calibration waiting: ";
    for (const auto& c : missing) {
        if (message != "calibration waiting: ") message += ", ";
        message += c.name + " (" + c.reason + ")";
    }
    if (effect_waiting) {
        if (message != "calibration waiting: ") message += ", ";
        message += "effect signal (published relative-service windows not converged)";
    }
    return message == "calibration waiting: " ? std::string{} : message;
}
// One calibration report line as it reaches the helper's log: the text and a real newline. The launcher finds the record with
// `grep '^RHP16_CALIBRATION_JSON '`, so no line may end in anything else (a literal backslash-n here once glued the record onto the
// end of a waiting line and the launcher never saw it).
inline std::string calibration_report_line(const std::string& text) { return text + "\n"; }

// Evidence for the outer launcher deadline: measured by the helper from
// actual retained source deltas. No configured gate is forged as a sample.
inline nlohmann::json calibration_drift_progress(const Measured& m) {
    return {
        {"capacity",m.drift_capacity},
        {"intervals",m.drift_intervals},
        {"horizon_observations",m.drift_horizon_observations},
        {"observation_rate_per_s",uint64_t(std::ceil(std::max(0.0,m.observation_rate_per_s)))},
        {"accepted_gap_ns",uint64_t(std::ceil(std::max(0.0,m.delta_gap_ns.max())))},
        {"accepted_heat_deltas",m.accepted_heat_deltas}
    };
}

// An accepted heat delta opens the feed, but the first positive measured heat
// gap and a retained drift snapshot establish the no-progress timer basis.
// Configured age/cadence alone cannot be used as evidence of feed progress.
class CalibrationProgressTimeout {
    struct Pending {
        uint64_t since_ns=0, progress=0;
        long double frozen_span_ns=0;
    };
    std::map<std::string,Pending> pending_;
    uint64_t effect_since_ns_=0;
    long double effect_frozen_span_ns_=0;
    std::array<uint64_t,4> effect_progress_{};
    std::string last_message_;

    static bool has_basis(const Measured& m) {
        return m.accepted_heat_deltas && m.delta_gap_ns.total() &&
               m.delta_gap_ns.max()>0 && m.drift_intervals && m.drift_capacity;
    }
    static bool measurable(const std::string& name,const Measured& m) {
        if(!has_basis(m) || name=="heat_watermarked")return false;
        if(name=="txn_p50_ns" || name=="tail_share_gain_ppm" ||
           name=="copy_bytes_per_s")return m.committed_transactions>0;
        if(name=="tail_lane_service_ns_per_s" || name=="tail_lane_row_ns" ||
           name=="lane_lost_ns")return m.tail_committed_transactions>0;
        // Drift/observation requirements need the first retained snapshot.
        // Planner capacity is known, but may only time out after live heat.
        return true;
    }
public:
    bool observe(const Measured& m,const std::vector<MissingPlannerCondition>& missing,
                 uint64_t now_ns,bool effect_waiting,std::string* changed_line) {
        std::map<std::string,Pending> next;
        for(const auto& c:missing) {
            auto it=pending_.find(c.name);
            Pending p=it==pending_.end()?Pending{}:it->second;
            if(!measurable(c.name,m)) {
                p.since_ns=0;
                p.frozen_span_ns=0;
            } else if(!p.since_ns || c.progress!=p.progress) {
                p.since_ns=now_ns;
                p.frozen_span_ns=0;
            }
            p.progress=c.progress;
            next.emplace(c.name,p);
        }
        pending_.swap(next);
        // Only effect-specific quiet/ladder progress renews the effect timer.
        const std::array<uint64_t,4> progress={
            m.effect.quiet_comparisons,
            m.effect.normalized_noise_ns.total(),
            m.effect.quiet_delta_ns.total(),
            m.effect_window_noise_ladder.empty()?0:
                m.effect_window_noise_ladder.back().rows
        };
        if(effect_waiting && has_basis(m)) {
            if(!effect_since_ns_ || progress!=effect_progress_) {
                effect_since_ns_=now_ns;
                effect_frozen_span_ns_=0;
            }
        } else {
            effect_since_ns_=0;
            effect_frozen_span_ns_=0;
        }
        effect_progress_=progress;
        const std::string message=calibration_waiting_message(missing,effect_waiting);
        if(message==last_message_)return false;
        last_message_=message;
        if(changed_line)*changed_line=message;
        return !message.empty();
    }
    static long double wait_span(const Measured& m,uint64_t cadence_ns,
                                 uint64_t max_heat_age_ns) {
        if(!has_basis(m) || !cadence_ns || !max_heat_age_ns)return 0;
        const long double gap=std::max({static_cast<long double>(m.delta_gap_ns.max()),
                                        static_cast<long double>(cadence_ns),
                                        static_cast<long double>(max_heat_age_ns)});
        return gap*m.drift_capacity;
    }
    bool expired(uint64_t now_ns,const Measured& m,uint64_t cadence_ns,
                 uint64_t max_heat_age_ns) {
        const long double span=wait_span(m,cadence_ns,max_heat_age_ns);
        if(span<=0)return false;
        for(auto& [name,p]:pending_) {
            if(!p.since_ns || !measurable(name,m))continue;
            // Freeze per-condition once it becomes measurable: subsequent
            // larger heat gaps must not extend its deadline.
            if(p.frozen_span_ns<=0)p.frozen_span_ns=span;
            if(now_ns>=p.since_ns &&
               static_cast<long double>(now_ns-p.since_ns)>=p.frozen_span_ns)
                return true;
        }
        return false;
    }
    bool effect_expired(uint64_t now_ns,const Measured& m,uint64_t cadence_ns,
                        uint64_t max_heat_age_ns) {
        const long double span=wait_span(m,cadence_ns,max_heat_age_ns);
        if(!effect_since_ns_ || span<=0)return false;
        if(effect_frozen_span_ns_<=0)effect_frozen_span_ns_=span;
        return now_ns>=effect_since_ns_ &&
               static_cast<long double>(now_ns-effect_since_ns_)>=
                   effect_frozen_span_ns_;
    }
};
inline bool calibration_settled(const Measured& m,bool effect_absent=false,
                                bool structural_unjudgeable=false) {
    // A separate progress watchdog bounds planner evidence on EVERY path,
    // including the structural no-peer case. The effect timeout still
    // independently permits a partial record when the peer never publishes.
    if (effect_absent && !structural_unjudgeable) return true;
    const bool planner_ready=calibration_missing_planner(m).empty();
    if (!planner_ready || effect_absent) return planner_ready;
    return m.effect_required_quiet_windows &&
           m.effect.quiet_comparisons >= m.effect_required_quiet_windows &&
           m.effect.normalized_noise_ns.count() >= 2 &&
           m.effect.row_noise_ns() > 0 &&
           m.effect.quiet_delta_ns.max() > 0 &&
           m.effect.sampled_rows_per_s > 0 &&
           m.effect_window_noise_ladder.size() >= 3 &&
           double(m.effect_window_noise_ladder.back().rows) >=
               m.effect.sampled_rows_per_s * double(m.effect_cadence_ns) / 1000000000.0;
}
inline nlohmann::json calibration_record(const Measured& m,
                                          const CalibrationIdentity& identity,
                                          const std::string& utc_date,
                                          bool effect_absent=false,
                                          const std::string& effect_reason=
                                              "relative tail/peer service unavailable for a measured heat-gap horizon",
                                          bool planner_partial=false) {
    using Json = nlohmann::json;
    Json measured = Json::object(), absent = Json::object();
    constexpr uint64_t max_exact = (uint64_t(1) << 53) - 1;
    const auto sample = [&](const char* name, double value, const char* reason, bool zero_ok=false) {
        if (std::isfinite(value) && (value>0.0 || (zero_ok && value==0)) &&
            value <= double(max_exact))
            measured[name] = static_cast<uint64_t>(std::ceil(value));
        else absent[name] = reason;
    };
    const auto no_hook = [&](const char* name, const char* why) { absent[name] = why; };
    sample("txn_p50_ns", m.transaction_ns.quantile(0.5), "no positive committed transaction wall time");
    sample("txn_p95_ns", m.transaction_ns.quantile(0.95), "no positive committed transaction wall time");
    // The journal has only aggregate wall times; copy bytes/time is NOT a
    // measured per-helper DMA rate, and cannot be used as slowest-helper data.
    sample("copy_bytes_per_s", m.slowest_helper_copy_bps.quantile(0.5),
           "no committed transaction with complete per-receiving-helper copy ACK timings");
    sample("heat_gap_max_ns", m.delta_gap_ns.max(), "no consecutive accepted heat deltas");
    if (m.accepted_heat_deltas)
        sample("heat_watermarked", m.accepted_unwatermarked_heat ? 0.0 : 1.0,
               "no accepted heat deltas", true);
    else no_hook("heat_watermarked", "no accepted heat deltas");
    sample("hot_set_k", double(m.hot_set_k), "no tail-helper expert capacity");
    sample("marginal_error_ppm", m.marginal_error * 1000000.0,
           "marginal relative error absent, zero, or outside representable range");
    sample("marginal_observations", double(m.marginal_observations),
           "no positive accepted source observation count");
    sample("drift_half_life_observations", double(m.drift_half_life_observations),
           "half-overlap crossing not observed in retained drift horizon");
    sample("observation_rate_per_s", m.observation_rate_per_s,
           "no two accepted counter/time points establish a positive source rate");
    sample("tail_share_gain_ppm", m.tail.gain_ppm.quantile(0.5),
           "no committed plan with positive predicted relative tail-share saving");
    sample("tail_lane_service_ns_per_s", m.tail_lane_service_ns_per_s.quantile(0.5),
           "no positive tail-helper sampled lane service time over a measured wall interval");
    sample("tail_lane_row_ns", m.tail_lane_row_ns.quantile(0.5),
           "no positive tail-helper service nanoseconds per routed row in the same interval");
    if (m.lane_lost_ns.count())
        sample("lane_lost_ns", m.lane_lost_ns.quantile(0.95),
               "invalid measured quiesce-to-reopen interval", true);
    else no_hook("lane_lost_ns", "no verified tail-helper quiesce-to-reopen interval");
    if(effect_absent) {
        // An unjudgeable lane has no relative effect measurement. Retain
        // the actual cause in each absent field for serving provenance.
        for(const char* name : {"effect_rows_per_s","effect_noise_max_ns",
                                "effect_row_noise_ns","effect_window_noise_ladder"})
            absent[name]=effect_reason;
    } else {
        sample("effect_rows_per_s", m.effect.sampled_rows_per_s,
               "no positive measured sampled-lane routed-row rate over wall time");
        // No false-revert objective is available to the helper: this is the actual
        // observed maximum, not a mislabeled objective-specific quantile.
        sample("effect_noise_max_ns", m.effect.quiet_delta_ns.max(),
               "no positive observed maximum quiet-window service-time difference");
        sample("effect_row_noise_ns", m.effect.row_noise_ns(),
               "no variable signed quiet-window deltas for per-row dispersion");
        if(m.effect_window_noise_ladder.size() >= 3 &&
           double(m.effect_window_noise_ladder.back().rows) >=
               m.effect.sampled_rows_per_s * double(m.effect_cadence_ns) / 1000000000.0) {
            Json ladder=Json::array();
            for(const auto& point:m.effect_window_noise_ladder)
                ladder.push_back(Json::array({point.rows,point.noise_ppm}));
            measured["effect_window_noise_ladder"]=std::move(ladder);
        } else no_hook("effect_window_noise_ladder",
                      "published tail service too short to measure adjacent windows through one policy cadence");
    }
    sample("abort_quiescent_ns", m.abort_ns.quantile(0.95), "none observed");
    return Json{{"plan_cache_key", identity.plan_cache_key},
                {"layout", identity.layout}, {"objective", identity.objective},
                {"seeded", utc_date},
                {"evidence", planner_partial ?
                    "partial calibration: stalled planner conditions remain absent with explicit reasons; serving must independently validate completeness" :
                    effect_absent ?
                    "effect unjudgeable: " + effect_reason + "; planner measurements independently observed" :
                    "accepted heat, closed journals, per-helper READY/RESERVED/CLEANED ACK timing, sampled lane service and committed planner gains"},
                {"measurements", measured}, {"absent", absent}};
}
} // namespace r3::policy
