#pragma once
// RHP16: the pure live-policy derivation, shared by the owner (serving's manifest writer) and the helper's policy driver.
// One function maps a record's measurements, the declared objectives, the expert stride and the helper budgets to every
// setting of the live policy. No file, manifest, TOML or provenance handling lives here: callers wrap the result.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

namespace r3::policy::derive {

using Json = nlohmann::json;
using Wide = unsigned __int128;
inline constexpr uint64_t kSecond = 1000000000;
inline constexpr uint64_t kMillion = 1000000;
inline constexpr uint64_t kMaxJsonInteger = (uint64_t(1) << 53) - 1;

[[noreturn]] inline void bad(const std::string& detail) {
    throw std::runtime_error("live policy calibration: " + detail);
}
// One measured integer of a record's measurements; refuses missing, negative, non-integer and out-of-range values.
inline uint64_t number(const Json& v, const char* name, bool zero_ok = false) {
    if (!v.contains(name) || (!v.at(name).is_number_unsigned() && !v.at(name).is_number_integer()))
        bad(std::string("missing or noninteger measurement ") + name);
    const auto& n = v.at(name);
    if (n.is_number_integer() && !n.is_number_unsigned() && n.get<int64_t>() < 0)
        bad(std::string("negative measurement ") + name);
    const uint64_t value = n.get<uint64_t>();
    if (value > kMaxJsonInteger || (!zero_ok && !value))
        bad(std::string("out of range measurement ") + name);
    return value;
}
inline uint64_t div_ceil(Wide n, Wide d, const std::string& name) {
    if (!d) bad(name + ": zero divisor");
    const Wide r = n / d + (n % d != 0);
    if (r == 0 || r > kMaxJsonInteger) bad(name + ": cannot represent the derived value in v1");
    return uint64_t(r);
}
// Two-sided normal-tail quantile from the declared false-revert objective.
inline double gaussian_z(double alpha) {
    const double root2 = std::sqrt(2.0);
    double low = 0.0, high = 1.0;
    while (std::erfc(high/root2) > alpha) high *= 2.0;
    for (unsigned i = 0; i < 128; ++i) {
        const double mid = low + (high-low)/2.0;
        if (mid == low || mid == high) break;
        if (std::erfc(mid/root2) > alpha) low = mid;
        else high = mid;
    }
    return high;
}
// Measured adjacent-window RMS fractional noise in r = tail ns/row divided
// by the same-window row-weighted mean ns/row of other helpers.
// Correlated requests need not obey sigma_row * sqrt(2/n).
// Only select lengths actually measured; there is no extrapolated i.i.d. cap.
inline std::optional<std::pair<uint64_t,uint64_t>> derive_effect_gate(const Json& ladder,
                                                uint64_t predicted_gain_ppm,
                                                uint64_t opportunities,double alpha) {
    const double z=gaussian_z(alpha);
    if(!ladder.is_array() || ladder.size()<3)
        bad("effect_window_noise_ladder: at least three observed lengths required");
    // For unchanged peer row-weighted service and tail ns/row proportional
    // to predicted tail share: r_after = r_before*(1 - gain_ppm/1e6).
    const double effect_fraction=double(predicted_gain_ppm)/double(kMillion);
    uint64_t previous=0,chosen=0,noise=0;
    for(const auto& pair:ladder) {
        if(!pair.is_array() || pair.size()!=2 ||
           !pair[0].is_number_unsigned() || !pair[1].is_number_unsigned())
            bad("effect_window_noise_ladder: expected [positive rows, nonnegative noise ppm]");
        const uint64_t rows=pair[0].get<uint64_t>(), ppm=pair[1].get<uint64_t>();
        if(!rows || rows<=previous || rows>kMaxJsonInteger || ppm>kMaxJsonInteger)
            bad("effect_window_noise_ladder: lengths must increase and fit v1");
        previous=rows;
        if(!chosen && double(ppm)*1e-6*z<=effect_fraction) {chosen=rows;noise=ppm;}
    }
    if(!chosen) return std::nullopt; // valid ladder, but no measured length resolves the gain
    // The observed adjacent-window difference supplies the null tail
    // probability. k uses the declared transaction opportunities; no
    // independence assumption is made about *rows* within the window.
    const double p=noise ? std::erfc(effect_fraction/(std::sqrt(2.0)*double(noise)*1e-6)) : 0.0;
    if(!std::isfinite(p) || p>=1.0)
        bad("effect_min_quiet_windows: invalid measured window probability");
    uint64_t quiet=1;
    long double all_quiet_p=p;
    while(static_cast<long double>(opportunities)*all_quiet_p>alpha) {
        if(quiet>=kMaxJsonInteger)
            bad("effect_min_quiet_windows: required quiet count exceeds v1 range");
        ++quiet;all_quiet_p*=p;
    }
    return std::make_pair(chosen,quiet);
}

// The planner-required measurements, each a positive integer (lane_lost_ns and heat_watermarked may be zero).
inline void require_planner_measurements(const Json& m) {
    for (const char* key : {
        "txn_p95_ns", "copy_bytes_per_s", "heat_gap_max_ns", "heat_watermarked", "marginal_error_ppm",
        "marginal_observations", "drift_half_life_observations", "observation_rate_per_s",
        "tail_share_gain_ppm", "tail_lane_service_ns_per_s", "tail_lane_row_ns", "lane_lost_ns"
    })
        (void)number(m, key, std::string(key) == "lane_lost_ns" || std::string(key) == "heat_watermarked");
}
// The declared objectives (not measurements): transactions opportunities per effect decision, false-revert rate in ppm.
inline void require_objectives(int64_t transaction_effect_windows, int64_t false_revert_rate_ppm) {
    if (transaction_effect_windows <= 0 || false_revert_rate_ppm <= 0 || false_revert_rate_ppm >= int64_t(kMillion))
        bad("declared objectives must be positive windows and a false-revert rate in (0,1000000)");
}

// Every live-policy setting, as typed values. effect_reason is empty exactly when effect_gate is true; otherwise it names
// why the effect verdict is off. margin_ppm etc. are scaled integers exactly as the manifest carries them.
struct PolicyValues {
    bool effect_gate = false;
    std::string effect_reason;
    bool accept_unwatermarked_heat = false;
    uint64_t cadence_ns = 0, max_moved_bytes = 0, max_moves = 0, min_residency_ns = 0, max_heat_age_ns = 0,
             min_samples = 0, abort_cooldown_ns = 0, learned_eligibility_ttl_ns = 0, effect_window_samples = 0,
             effect_min_quiet_windows = 0, effect_revert_ttl_ns = 0, margin_ppm = 0, half_life_observations_milli = 0,
             effect_noise_multiplier_milli = 0;
    bool abort_measured = false;   // abort_cooldown_ns used a measured abort_quiescent_ns
};

// `m` is a record's measurements, `absent` its named missing reasons. Throws std::runtime_error naming the defect.
inline PolicyValues derive_policy_values(const Json& m, const Json& absent, int64_t transaction_effect_windows,
                                         int64_t false_revert_rate_ppm, uint64_t stride,
                                         const std::vector<uint64_t>& budgets) {
    if (!stride || budgets.empty()) bad("expert stride and helper budgets are required");
    const auto n = [&](const char* key) { return number(m, key, std::string(key) == "lane_lost_ns" || std::string(key) == "heat_watermarked"); };
    require_planner_measurements(m);
    require_objectives(transaction_effect_windows, false_revert_rate_ppm);
    const int64_t windows = transaction_effect_windows, false_rate = false_revert_rate_ppm;
    // Missing effect-only observations or an unresolvable measured noise
    // ladder disable ONLY the effect verdict; planner evidence remains mandatory.
    std::string effect_reason;
    for (const char* key : {"effect_rows_per_s", "effect_noise_max_ns",
                            "effect_row_noise_ns", "effect_window_noise_ladder"}) {
        if (!m.contains(key)) {
            effect_reason = std::string(key) + " absent: " +
                absent.value(key, std::string("not observed"));
            break;
        }
    }
    uint64_t effect_samples = 1, effect_quiet = 1, window_ns = 0;
    if (effect_reason.empty()) {
        // Relative tail/peer service only, never an absolute fallback.
        const auto resolved = derive_effect_gate(
            m.at("effect_window_noise_ladder"), n("tail_share_gain_ppm"),
            uint64_t(windows), double(false_rate) / double(kMillion));
        if (resolved) {
            effect_samples = resolved->first;
            effect_quiet = resolved->second;
            window_ns = div_ceil(Wide(effect_samples) * kSecond,
                                 n("effect_rows_per_s"), "effect window");
        } else {
            effect_reason = "measured relative-service noise ladder cannot resolve predicted tail effect";
        }
    }
    const bool effect_enabled = effect_reason.empty();
    // Inert positive effect controls satisfy the v1 schema when disabled.
    // They are not inferred from nonexistent evidence and are never used.
    const uint64_t cadence = effect_enabled ?
        div_ceil(Wide(n("txn_p95_ns")) + Wide(effect_quiet)*window_ns, 1, "cadence_ns") :
        std::max(n("txn_p95_ns"), n("heat_gap_max_ns"));
    const uint64_t slow_budget = *std::min_element(budgets.begin(), budgets.end());
    if (!slow_budget || slow_budget < stride) bad("a helper budget cannot hold one expert stride");
    // The byte cap is per proposed transaction, not the sum of T later
    // effect windows. Otherwise the planner may stage thousands of copies
    // in ONE quiescence (calib5: 2,399 experts).
    const uint64_t move_bytes = std::min(
        slow_budget, div_ceil(Wide(n("copy_bytes_per_s")) * n("txn_p95_ns"),
                              kSecond, "max_moved_bytes"));
    const uint64_t max_moves = std::min<uint64_t>(move_bytes / stride, uint64_t(UINT32_MAX));
    if (!max_moves) bad("measured budget buys fewer than one complete expert; cannot activate policy");
    // The margin is the sampling error of the heat the planner can actually hold: the observations of one drift
    // half-life, after which the hot set has moved on. The calibration's own error was measured over however many
    // observations it ran (production calib: 3.98e9, about six drift half-lives); using that error as the margin made
    // the per-expert gate below demand the whole calibration's sample (522 windows, 43 min, against a 7.4 min drift
    // half-life), so no expert was ever movable. Never extrapolate below the measured error: horizon <= measured.
    const uint64_t horizon = std::min(n("marginal_observations"), n("drift_half_life_observations"));
    if (!horizon) bad("no observations bound the heat sampling horizon");
    const double margin_scaled = std::ceil(double(n("marginal_error_ppm")) *
        std::sqrt(double(n("marginal_observations")) / double(horizon)));
    if (!(margin_scaled >= 1.0) || margin_scaled >= double(kMillion))
        bad("relative sampling error over one drift half-life is outside v1 margin_ppm range");
    const uint64_t margin = uint64_t(margin_scaled);
    // The planner's per-expert gate counts heat WINDOWS (one per accepted delta, i.e. per cadence), not routed
    // observations: convert the observations the margin needs into windows of measured rate x cadence. Comparing an
    // observation count against the window counter (rhp16-calib4: 105,925,840 against ~50 windows) blocks every move.
    const uint64_t needed_observations = div_ceil(
        Wide(n("marginal_observations")) * n("marginal_error_ppm") * n("marginal_error_ppm"),
        Wide(margin) * margin, "observations for the margin");
    const uint64_t observations_per_window = div_ceil(
        Wide(n("observation_rate_per_s")) * cadence, kSecond, "observations per heat window");
    const uint64_t min_samples = div_ceil(needed_observations, observations_per_window, "min_samples");
    const uint64_t savings_ns_per_s = div_ceil(
        Wide(n("tail_share_gain_ppm")) * n("tail_lane_service_ns_per_s"), kMillion,
        "predicted tail savings");
    // The measured committed-transaction p95 is the cost of the typical
    // gain, not the theoretical maximum byte allowance of a single plan.
    const uint64_t residency = div_ceil(
        (Wide(n("txn_p95_ns")) + n("lane_lost_ns")) * kSecond,
        savings_ns_per_s, "min_residency_ns");
    const uint64_t drift_ns = div_ceil(
        Wide(n("drift_half_life_observations")) * kSecond, n("observation_rate_per_s"),
        "effect_revert_ttl_ns");
    // The effect gate compares FRACTIONAL before/after service rates with
    // a FRACTIONAL quiet-window envelope. The old max-ns / E multiplier
    // applied noise twice and made the calib5 improvement threshold >100%.
    const uint64_t noise_multiplier = uint64_t(std::ceil(
        gaussian_z(double(false_rate) / double(kMillion)) * 1000.0));
    // A ban must outlast at least one new policy poll and the k+1 service
    // windows needed to reestablish the effect gate after a transaction.
    const uint64_t max_heat_age = div_ceil(Wide(cadence) + n("heat_gap_max_ns"),
                                           1, "max_heat_age_ns");
    const uint64_t recovery = effect_enabled ?
        div_ceil(Wide(window_ns) * (effect_quiet + 1), 1, "effect gate recovery") : 0;
    const uint64_t cache_ttl = div_ceil(Wide(max_heat_age) + recovery,
                                         1, "learned eligibility TTL");
    // A WORSE verdict starts a rollback that refreshes per-expert move times.
    // Keep the destination ban through rollback, the ensuing residency and
    // one more policy poll; otherwise it expires before retry is even legal.
    const uint64_t revert_ttl = div_ceil(
        Wide(residency) + cadence + n("txn_p95_ns"), 1,
        "effect_revert_ttl_ns");
    const uint64_t effective_revert_ttl = std::max({drift_ns, cache_ttl, revert_ttl});
    if (n("heat_watermarked") > 1) bad("heat_watermarked must be 0 or 1");

    PolicyValues out;
    out.effect_gate = effect_enabled;
    out.effect_reason = effect_reason;
    out.accept_unwatermarked_heat = n("heat_watermarked") == 0;
    out.cadence_ns = cadence;
    out.max_moved_bytes = move_bytes;
    out.max_moves = max_moves;
    out.min_residency_ns = residency;
    out.max_heat_age_ns = max_heat_age;
    out.min_samples = min_samples;
    out.abort_measured = m.contains("abort_quiescent_ns");
    out.abort_cooldown_ns = std::max(max_heat_age, out.abort_measured ? n("abort_quiescent_ns") : cadence);
    out.learned_eligibility_ttl_ns = cache_ttl;
    out.effect_window_samples = effect_samples;
    out.effect_min_quiet_windows = effect_quiet;
    out.effect_revert_ttl_ns = effective_revert_ttl;
    out.margin_ppm = margin;
    out.half_life_observations_milli = div_ceil(Wide(n("drift_half_life_observations")) * 1000, 1,
                                                "half_life_observations_milli");
    out.effect_noise_multiplier_milli = noise_multiplier;
    return out;
}

// A stable digest of every derived setting (FNV-1a 64 over fixed-order decimal fields): the helper logs it when it adopts,
// and the owner derives the same values from the stored record and compares.
inline std::string policy_digest(const PolicyValues& v) {
    std::string text;
    for (uint64_t x : {uint64_t(v.effect_gate), uint64_t(v.accept_unwatermarked_heat), v.cadence_ns, v.max_moved_bytes,
                       v.max_moves, v.min_residency_ns, v.max_heat_age_ns, v.min_samples, v.abort_cooldown_ns,
                       v.learned_eligibility_ttl_ns, v.effect_window_samples, v.effect_min_quiet_windows,
                       v.effect_revert_ttl_ns, v.margin_ppm, v.half_life_observations_milli,
                       v.effect_noise_multiplier_milli})
        text += std::to_string(x) + ';';
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : text) { h ^= c; h *= 1099511628211ull; }
    char out[17];
    std::snprintf(out, sizeof(out), "%016llx", (unsigned long long)h);
    return out;
}

} // namespace r3::policy::derive
