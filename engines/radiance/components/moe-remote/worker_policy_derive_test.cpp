// The shared live-policy derivation (worker_policy_derive.h), card-free. The numbers are the calib5 measurements the
// serving tests also derive from; serving's golden (live_policy_derive.golden) proves the owner's whole output is
// unchanged by the extraction, this proves the helper-callable function alone.
#include "worker_policy_derive.h"
#include <cassert>
#include <iostream>
#ifdef NDEBUG
#error "assert-based test built with NDEBUG: its assertions would not run"
#endif

namespace d = r3::policy::derive;
using Json = nlohmann::json;

static Json calib5() {
    Json m = Json::object();
    m["txn_p50_ns"]=1593403198ull; m["txn_p95_ns"]=3794447513ull; m["copy_bytes_per_s"]=319377397ull;
    m["heat_gap_max_ns"]=6955375341ull; m["heat_watermarked"]=0; m["hot_set_k"]=5931; m["marginal_error_ppm"]=31905;
    m["marginal_observations"]=105169020ull; m["drift_half_life_observations"]=15849692ull;
    m["observation_rate_per_s"]=1282858; m["tail_share_gain_ppm"]=98958; m["tail_lane_service_ns_per_s"]=468160374ull;
    m["tail_lane_row_ns"]=62964; m["lane_lost_ns"]=14766582; m["effect_rows_per_s"]=7100; m["effect_noise_max_ns"]=33585;
    m["effect_row_noise_ns"]=444685;
    m["effect_window_noise_ladder"]=Json::array({Json::array({uint64_t(33793),uint64_t(45000)}),
        Json::array({uint64_t(67586),uint64_t(35000)}), Json::array({uint64_t(135172),uint64_t(20000)})});
    return m;
}
template <class F> static bool throws_with(F f, const char* text) {
    try { f(); } catch (const std::runtime_error& e) { return std::string(e.what()).find(text) != std::string::npos; }
    return false;
}

int main() {
    const std::vector<uint64_t> budgets = {22ull<<30, 22ull<<30, 14ull<<30};
    const auto v = d::derive_policy_values(calib5(), Json::object(), 2, 10000, 2534544, budgets);
    // Identical to the owner's derivation of the same record (serving golden case "calib5").
    assert(v.effect_gate && v.effect_reason.empty() && v.accept_unwatermarked_heat && !v.abort_measured);
    assert(v.abort_cooldown_ns == 20268977784ull && v.cadence_ns == 13313602443ull);
    assert(v.effect_min_quiet_windows == 1 && v.effect_noise_multiplier_milli == 2576);
    assert(v.effect_revert_ttl_ns == 99330386712ull && v.effect_window_samples == 67586);
    assert(v.half_life_observations_milli == 15849692000ull && v.learned_eligibility_ttl_ns == 39307287644ull);
    assert(v.margin_ppm == 82185 && v.max_heat_age_ns == 20268977784ull && v.max_moved_bytes == 1211860770ull);
    assert(v.max_moves == 478 && v.min_residency_ns == 82222336756ull && v.min_samples == 1);

    // Effect-only absence disables only the effect verdict and names why.
    auto no_effect = calib5();
    no_effect.erase("effect_window_noise_ladder");
    const auto off = d::derive_policy_values(no_effect, Json{{"effect_window_noise_ladder", "no peer"}}, 2, 10000,
                                              2534544, budgets);
    assert(!off.effect_gate && off.effect_reason == "effect_window_noise_ladder absent: no peer");
    assert(off.cadence_ns == std::max<uint64_t>(3794447513ull, 6955375341ull));

    // Planner evidence is mandatory; a derivation the budgets cannot hold is refused.
    auto missing = calib5();
    missing.erase("copy_bytes_per_s");
    assert(throws_with([&] { d::derive_policy_values(missing, Json::object(), 2, 10000, 2534544, budgets); },
                       "copy_bytes_per_s"));
    assert(throws_with([&] { d::derive_policy_values(calib5(), Json::object(), 2, 10000, 2534544, {1000}); },
                       "cannot hold one expert stride"));
    assert(throws_with([&] { d::derive_policy_values(calib5(), Json::object(), 0, 10000, 2534544, budgets); },
                       "declared objectives"));
    assert(throws_with([&] { d::derive_policy_values(calib5(), Json::object(), 2, 10000, 0, budgets); },
                       "required"));
    // The z score is the two-sided normal quantile of the declared rate.
    assert(d::gaussian_z(0.01) > 2.575 && d::gaussian_z(0.01) < 2.577);
    std::cout << "worker_policy_derive_test: ok\n";
}
