#pragma once
// Pure RHP16 lab evidence serialization. No GPU, clock, filesystem, or fabricated measurements.
// An absent field names the unavailable event; configured gate targets are NEVER serialized as samples.
#include "worker_policy_measure.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>

namespace r3::policy {
struct CalibrationIdentity {
    std::string plan_cache_key, layout, objective;
};
inline bool calibration_settled(const Measured& m) {
    // Require the complete observed drift horizon and a heat shift that actually
    // caused a committed move. Quiet-window statistics must converge separately.
    return m.accepted_heat_deltas && m.drift_capacity && m.drift_intervals == m.drift_capacity &&
           m.delta_gap_ns.total() >= m.drift_intervals - 1 &&
           m.delta_gap_ns.max() > 0 &&
           m.hot_set_k && std::isfinite(m.marginal_error) && m.marginal_error > 0 &&
           m.marginal_observations && std::isfinite(m.observation_rate_per_s) &&
           m.observation_rate_per_s > 0 &&
           m.drift_half_life_observations && m.transaction_ns.quantile(0.5) > 0 &&
           m.tail.gain_ppm.quantile(0.5) > 0 &&
           m.slowest_helper_copy_bps.quantile(0.5) > 0 &&
           m.tail_lane_service_ns_per_s.quantile(0.5) > 0 &&
           m.tail_lane_row_ns.quantile(0.5) > 0 &&
           m.lane_lost_ns.count() &&
           m.effect_required_quiet_windows &&
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
                                          const std::string& utc_date) {
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
    sample("abort_quiescent_ns", m.abort_ns.quantile(0.95), "none observed");
    return Json{{"plan_cache_key", identity.plan_cache_key},
                {"layout", identity.layout}, {"objective", identity.objective},
                {"seeded", utc_date},
                {"evidence", "accepted heat, closed journals, per-helper READY/RESERVED/CLEANED ACK timing, sampled lane service and committed planner gains"},
                {"measurements", measured}, {"absent", absent}};
}
} // namespace r3::policy
