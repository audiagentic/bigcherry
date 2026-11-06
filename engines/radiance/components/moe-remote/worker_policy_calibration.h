#pragma once
// Pure RHP16 lab evidence serialization. No clock, GPU, filesystem or guessed calibration data.
// A settled record may be partial; every absent input names why it was not measured.
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
    // The drift tracker itself sets this observation horizon. A short or
    // reset source is not declared converged using an arbitrary timeout.
    return m.drift_capacity && m.drift_intervals == m.drift_capacity &&
           m.delta_gap_ns.total() >= m.drift_intervals - 1;
}
inline nlohmann::json calibration_record(const Measured& m,
                                          const CalibrationIdentity& identity,
                                          const std::string& utc_date) {
    using Json = nlohmann::json;
    Json measured = Json::object(), absent = Json::object();
    constexpr uint64_t max_exact = (uint64_t(1) << 53) - 1;
    const auto sample = [&](const char* name, double value, const char* reason) {
        if (std::isfinite(value) && value > 0.0 && value <= double(max_exact))
            measured[name] = static_cast<uint64_t>(std::ceil(value));
        else absent[name] = reason;
    };
    const auto no_hook = [&](const char* name, const char* why) { absent[name] = why; };
    sample("txn_p50_ns", m.transaction_ns.quantile(0.5), "no positive committed transaction wall time");
    sample("txn_p95_ns", m.transaction_ns.quantile(0.95), "no positive committed transaction wall time");
    // The driver knows aggregate bytes/time, not per-destination DMA timings.
    // An aggregate rate MUST NOT masquerade as the slowest helper's rate.
    no_hook("copy_bytes_per_s", "slowest-helper copy time is not measured by the transaction hook");
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
    no_hook("tail_share_noise_ppm", "tail-share noise series not exposed by helper hooks");
    no_hook("tail_share_gain_ppm", "marginal tail-share gain not measured by helper hooks");
    no_hook("tail_lane_service_ns_per_s", "per-tail lane service rate not measured");
    no_hook("lane_lost_ns", "lane service time lost to a move not measured");
    no_hook("effect_window_samples", "effect variance convergence count not measured independently of the configured gate");
    no_hook("effect_requests_per_s", "effect request arrival rate not measured");
    no_hook("effect_quiet_windows", "quiet window convergence count not measured independently of the configured gate");
    no_hook("effect_noise_floor_ns", "effect noise is a fractional lane ratio, not nanoseconds");
    no_hook("effect_noise_quantile_ns", "empirical nanosecond noise quantile not measured");
    no_hook("noise_false_revert_rate_ppm", "empirical false-revert rate not measured");
    sample("abort_quiescent_ns", m.abort_ns.quantile(0.95), "none observed");
    return Json{{"plan_cache_key", identity.plan_cache_key},
                {"layout", identity.layout}, {"objective", identity.objective},
                {"seeded", utc_date}, {"evidence", "worker_policy_measure: accepted heat deltas and closed transactions"},
                {"measurements", measured}, {"absent", absent}};
}
} // namespace r3::policy
