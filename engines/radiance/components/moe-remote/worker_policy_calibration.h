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
    return m.drift_capacity && m.drift_intervals == m.drift_capacity &&
           m.delta_gap_ns.total() >= m.drift_intervals - 1 &&
           m.tail.noise_ppm.max() > 0 &&
           m.drift_half_life_observations && m.transaction_ns.count() &&
           m.tail.gain_ppm.count() && m.tail.noise_ppm.count() &&
           m.slowest_helper_copy_bps.count() && m.tail_lane_service_ns_per_s.count() &&
           m.lane_lost_ns.count() &&
           m.effect_required_quiet_windows &&
           m.effect.quiet_comparisons >= m.effect_required_quiet_windows &&
           m.effect.window_samples.count() &&
           m.effect.quiet_delta_ns.max() > 0 &&
           m.effect.sampled_requests_per_s > 0;
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
    sample("hot_set_k", double(m.hot_set_k), "no tail-helper expert capacity");
    sample("marginal_error_ppm", m.marginal_error * 1000000.0,
           "marginal relative error absent, zero, or outside representable range");
    sample("marginal_observations", double(m.marginal_observations),
           "no positive accepted source observation count");
    sample("drift_half_life_observations", double(m.drift_half_life_observations),
           "half-overlap crossing not observed in retained drift horizon");
    sample("observation_rate_per_s", m.observation_rate_per_s,
           "no two accepted counter/time points establish a positive source rate");
    sample("tail_share_noise_ppm", m.tail.noise_ppm.max(),
           "no positive no-move tail-share change on the same owner generation");
    sample("tail_share_gain_ppm", m.tail.gain_ppm.quantile(0.5),
           "no committed plan with positive predicted relative tail-share saving");
    sample("tail_lane_service_ns_per_s", m.tail_lane_service_ns_per_s.quantile(0.5),
           "no positive tail-helper sampled lane service time over a measured wall interval");
    if (m.lane_lost_ns.count())
        sample("lane_lost_ns", m.lane_lost_ns.quantile(0.95),
               "invalid measured quiesce-to-reopen interval", true);
    else no_hook("lane_lost_ns", "no verified tail-helper quiesce-to-reopen interval");
    // These come from observed completed quiet windows, never the configured
    // window_samples or min_quiet_windows targets.
    sample("effect_window_samples", m.effect.window_samples.quantile(0.5),
           "no observed full quiet-window sample series");
    sample("effect_requests_per_s", m.effect.sampled_requests_per_s,
           "no positive measured sampled-lane request rate over wall time");
    sample("effect_quiet_windows", double(m.effect.quiet_comparisons),
           "no measured consecutive quiet-window comparisons");
    sample("effect_noise_floor_ns", m.effect.quiet_delta_ns.quantile(0.5),
           "no positive observed quiet-window service-time difference in nanoseconds");
    sample("effect_noise_quantile_ns", m.effect.quiet_delta_ns.max(),
           "no positive measured quiet-window service-time quantile in nanoseconds");
    no_hook("noise_false_revert_rate_ppm",
            "no independently labelled false-revert outcomes; configured noise objective is not a measurement");
    sample("abort_quiescent_ns", m.abort_ns.quantile(0.95), "none observed");
    return Json{{"plan_cache_key", identity.plan_cache_key},
                {"layout", identity.layout}, {"objective", identity.objective},
                {"seeded", utc_date},
                {"evidence", "accepted heat, closed journals, per-helper READY/RESERVED/CLEANED ACK timing, sampled lane service and committed planner gains"},
                {"measurements", measured}, {"absent", absent}};
}
} // namespace r3::policy
