#include "worker_policy_calibration.h"
#include <cassert>
#ifdef NDEBUG
#error "assert-based test built with NDEBUG: its assertions would not run"
#endif
#include <string>
#include <vector>

using namespace r3::policy;

int main() {
    // Use the hook's actual ring capacity and accepted source observations.
    DriftTracker drift(1, 4);
    Measured m;
    m.drift_capacity = drift.capacity();
    for (uint64_t obs = 1; obs <= 3; ++obs) {
        drift.add({double(5 - obs), double(obs)}, obs * 10);
        m.drift_intervals = drift.intervals();
        if (obs > 1) m.delta_gap_ns.add(1000000000.0);
        assert(!calibration_settled(m));
    }
    drift.add({1, 4}, 40);
    m.drift_intervals = drift.intervals();
    m.drift_horizon_observations = drift.horizon_observations();
    m.drift_half_life_observations = drift.half_life_observations();
    m.delta_gap_ns.add(1000000000.0);
    // Heat horizon alone is not convergence: a move and independent quiet
    // timing series must also be observed.
    assert(!calibration_settled(m));
    m.transaction_ns.add(10.0);
    m.tail.gain_ppm.add(1000.0);
    m.slowest_helper_copy_bps.add(1000000.0);
    m.tail_lane_service_ns_per_s.add(200000000.0);
    m.tail_lane_row_ns.add(1250.0);
    m.lane_lost_ns.add(100.0);
    m.hot_set_k = 2;
    m.marginal_error = 0.025;
    m.marginal_observations = 40;
    m.observation_rate_per_s = 10.0;
    // No input accepted => no watermark measurement or settled record.
    assert(!calibration_settled(m));
    m.accepted_heat_delta(true);
    m.effect_required_quiet_windows=2;
    m.effect.normalized_noise_ns.add(10.0);
    m.effect.normalized_noise_ns.add(-10.0);
    m.effect.quiet_delta_ns.add(5.0);
    m.effect.quiet_comparisons=2;
    m.effect.sampled_rows_per_s=20.0;
    assert(calibration_settled(m));
    assert(m.drift_capacity == 4 && m.drift_horizon_observations == 30);
    // A counter discontinuity clears the horizon; the same cadence must not
    // accidentally certify a reset source.
    drift.add({2, 3}, 1);
    m.drift_intervals = drift.intervals();
    assert(!calibration_settled(m));

    m.drift_intervals = m.drift_capacity;
    m.transaction_ns.add(10.0);
    m.transaction_ns.add(30.0);
    m.hot_set_k = 2;
    m.marginal_error = 0.025;
    m.marginal_observations = 40;
    m.observation_rate_per_s = 10.0;
    const auto row = calibration_record(m, {std::string(64, 'a'), "flashnext-4gpu", "balanced"}, "2026-11-05");
    assert(row.at("measurements").at("txn_p50_ns") == 10);
    assert(row.at("measurements").at("txn_p95_ns") == 30);
    assert(row.at("measurements").at("hot_set_k") == 2);
    assert(row.at("measurements").at("heat_watermarked") == 1);
    Measured mixed = m;
    mixed.accepted_heat_delta(false);
    mixed.accepted_heat_delta(true); // once unwatermarked, always zero
    assert(calibration_record(mixed, {std::string(64, 'd'), "flashnext-4gpu", "balanced"},
                              "2026-11-06").at("measurements").at("heat_watermarked") == 0);
    Measured unobserved = m;
    unobserved.accepted_heat_deltas = 0;
    unobserved.accepted_unwatermarked_heat = false;
    assert(!calibration_settled(unobserved));
    const auto no_heat = calibration_record(unobserved,
        {std::string(64, 'e'), "flashnext-4gpu", "balanced"}, "2026-11-06");
    assert(no_heat.at("absent").at("heat_watermarked") == "no accepted heat deltas");
    assert(row.at("measurements").at("marginal_error_ppm") == 25000);
    assert(row.at("measurements").at("copy_bytes_per_s") == 1000000);
    assert(!row.at("absent").contains("copy_bytes_per_s"));
    assert(row.at("measurements").at("tail_lane_service_ns_per_s") == 200000000);
    assert(row.at("measurements").at("tail_lane_row_ns") == 1250);
    Measured no_requests=m;
    no_requests.tail_lane_row_ns=Series{64};
    assert(!calibration_settled(no_requests));
    const auto missing_request=calibration_record(no_requests,
        {std::string(64, 'c'), "flashnext-4gpu", "balanced"}, "2026-11-06");
    assert(missing_request.at("absent").at("tail_lane_row_ns").is_string());
    assert(!missing_request.at("measurements").contains("tail_lane_row_ns"));
    assert(row.at("measurements").at("lane_lost_ns") == 100);
    assert(row.at("measurements").at("effect_row_noise_ns") == 10);
    assert(row.at("measurements").at("effect_noise_max_ns") == 5);
    for (const char* removed : {"effect_window_samples", "effect_quiet_windows",
                                "noise_false_revert_rate_ppm", "effect_noise_quantile_ns",
                                "effect_noise_floor_ns"}) {
        assert(!row.at("measurements").contains(removed));
        assert(!row.at("absent").contains(removed));
    }
    Measured no_copy=m;
    no_copy.slowest_helper_copy_bps=Series{64};
    const auto absent_copy=calibration_record(no_copy,
        {std::string(64, 'b'), "flashnext-4gpu", "balanced"}, "2026-11-06");
    assert(absent_copy.at("absent").at("copy_bytes_per_s").is_string());
    assert(!absent_copy.at("measurements").contains("copy_bytes_per_s"));
    assert(row.at("absent").at("abort_quiescent_ns") == "none observed");
    assert(row.at("absent").size() + row.at("measurements").size() == 18);
}
