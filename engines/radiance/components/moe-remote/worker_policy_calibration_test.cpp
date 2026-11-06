#include "worker_policy_calibration.h"
#include <cassert>
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
    assert(row.at("measurements").at("marginal_error_ppm") == 25000);
    assert(row.at("absent").at("copy_bytes_per_s").is_string());
    assert(row.at("absent").at("abort_quiescent_ns") == "none observed");
    assert(!row.at("measurements").contains("copy_bytes_per_s"));
    assert(row.at("absent").size() + row.at("measurements").size() == 20);
}
