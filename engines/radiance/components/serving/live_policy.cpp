#include "live_policy.h"
#include "third_party/toml.hpp"

#include <algorithm>
#include <utility>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace bc::serving {
namespace {
using Json = nlohmann::json;
using Wide = unsigned __int128;
constexpr uint64_t kSecond = 1000000000;
constexpr uint64_t kMillion = 1000000;
constexpr uint64_t kMaxJsonInteger = (uint64_t(1) << 53) - 1;

[[noreturn]] void bad(const std::string& detail) {
    throw std::runtime_error("live policy calibration: " + detail);
}
uint64_t positive(const toml::table& t, const char* name, const std::string& where, bool zero_ok = false) {
    const auto* v = t[name].as_integer();
    if (!v || v->get() < (zero_ok ? 0 : 1) || uint64_t(v->get()) > kMaxJsonInteger)
        bad(where + "." + name + " must be a " + (zero_ok ? "nonnegative" : "positive") + " integer");
    return uint64_t(v->get());
}
std::string required(const toml::table& t, const char* name, const std::string& where) {
    const auto* v = t[name].as_string();
    if (!v || v->get().empty()) bad(where + "." + name + " must be a nonempty string");
    return v->get();
}
uint64_t number(const Json& v, const char* name, bool zero_ok = false) {
    if (!v.contains(name) || !v.at(name).is_number_unsigned() && !v.at(name).is_number_integer())
        bad(std::string("missing or noninteger measurement ") + name);
    const auto& n = v.at(name);
    if (n.is_number_integer() && !n.is_number_unsigned() && n.get<int64_t>() < 0)
        bad(std::string("negative measurement ") + name);
    const uint64_t value = n.get<uint64_t>();
    if (value > kMaxJsonInteger || (!zero_ok && !value))
        bad(std::string("out of range measurement ") + name);
    return value;
}
uint64_t div_ceil(Wide n, Wide d, const std::string& name) {
    if (!d) bad(name + ": zero divisor");
    const Wide r = n / d + (n % d != 0);
    if (r == 0 || r > kMaxJsonInteger) bad(name + ": cannot represent the derived value in v1");
    return uint64_t(r);
}
Json provenance(const std::string& description, const Json& inputs) {
    return {{"origin", "derived"}, {"source", description}, {"class", "policy"},
            {"lifecycle", "startup_only"}, {"confidence", "measured"}, {"inputs", inputs}};
}
// Two-sided normal-tail quantile from the declared false-revert objective.
double gaussian_z(double alpha) {
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
// Gaussian null model: sd(delta of two n-request means) = sigma*sqrt(2/n).
// Choose the smallest n satisfying a two-sided noise-crossing objective;
// across transaction-effect opportunities, require opportunities*p^k <= alpha.
// Consecutive quiet crossings are modelled as independent, not verified trials.
std::pair<uint64_t,uint64_t> derive_effect_gate(uint64_t sigma_ns, uint64_t delta_effect_ns,
                                                uint64_t opportunities, double alpha) {
    const double z = gaussian_z(alpha);
    const double sigma = double(sigma_ns), effect = double(delta_effect_ns);
    const double needed = 2.0 * std::pow(sigma*z/effect, 2.0);
    if (!std::isfinite(needed) || needed > double(kMaxJsonInteger))
        bad("effect_window_samples: cannot represent the effect-size-resolving window");
    const uint64_t n = std::max<uint64_t>(1, uint64_t(std::ceil(needed)));
    const double sd_delta = sigma * std::sqrt(2.0/double(n));
    const double p = std::erfc(effect/(std::sqrt(2.0)*sd_delta));
    if (!std::isfinite(p) || p >= 1.0)
        bad("effect_min_quiet_windows: invalid Gaussian noise probability");
    uint64_t quiet = 1;
    long double all_quiet_p = p;
    while (static_cast<long double>(opportunities) * all_quiet_p > alpha) {
        if (quiet >= kMaxJsonInteger)
            bad("effect_min_quiet_windows: required quiet count exceeds v1 range");
        ++quiet;
        all_quiet_p *= p;
    }
    return {n, quiet};
}
} // namespace

std::optional<LivePolicyRecord> read_live_policy_record(const std::filesystem::path& cache,
                                                       const std::string& plan_key) {
    if (cache.empty()) return std::nullopt;
    std::ifstream in(cache);
    if (!in) bad("cannot read " + cache.string());
    std::ostringstream contents;
    contents << in.rdbuf();
    toml::table root;
    try { root = toml::parse(contents.str(), cache.string()); }
    catch (const toml::parse_error& e) { bad(cache.string() + ": TOML parse error: " + e.what()); }
    const auto* rows = root["live_policy_record"].as_array();
    if (!rows) {
        if (root.contains("live_policy_record")) bad("live_policy_record must be an array of tables");
        return std::nullopt;
    }
    const std::set<std::string> fields = {
        "plan_cache_key", "layout", "objective", "seeded",
        "evidence", "txn_p50_ns", "txn_p95_ns", "copy_bytes_per_s",
        "heat_gap_max_ns", "heat_watermarked", "hot_set_k", "marginal_error_ppm", "marginal_observations",
        "drift_half_life_observations", "observation_rate_per_s", "tail_share_gain_ppm",
        "tail_lane_service_ns_per_s", "tail_lane_request_ns", "lane_lost_ns", "effect_requests_per_s",
        "effect_noise_max_ns", "effect_sample_noise_ns", "abort_quiescent_ns", "absent"
    };
    const std::set<std::string> names = {
        "txn_p50_ns", "txn_p95_ns", "copy_bytes_per_s", "heat_gap_max_ns", "heat_watermarked",
        "hot_set_k", "marginal_error_ppm", "marginal_observations", "drift_half_life_observations",
        "observation_rate_per_s", "tail_share_gain_ppm", "tail_lane_service_ns_per_s",
        "tail_lane_request_ns", "lane_lost_ns", "effect_requests_per_s", "effect_noise_max_ns",
        "effect_sample_noise_ns"
    };
    std::set<std::string> seen;
    std::optional<LivePolicyRecord> found;
    for (size_t i = 0; i < rows->size(); ++i) {
        const std::string where = cache.string() + ": live_policy_record[" + std::to_string(i) + "]";
        const auto* t = (*rows)[i].as_table();
        if (!t) bad(where + " must be a table");
        for (const auto& [key, val] : *t)
            if (!fields.count(std::string(key.str()))) bad(where + ": unknown key " + std::string(key.str()));
        LivePolicyRecord row;
        row.key = required(*t, "plan_cache_key", where);
        if (row.key.size() != 64 || row.key.find_first_not_of("0123456789abcdef") != std::string::npos)
            bad(where + ": plan_cache_key must be lowercase SHA-256 hex");
        if (!seen.insert(row.key).second) bad(where + ": duplicate plan_cache_key " + row.key);
        row.layout = required(*t, "layout", where);
        row.objective = required(*t, "objective", where);
        row.seeded = required(*t, "seeded", where);
        row.evidence = required(*t, "evidence", where);
        row.source = cache.filename().string() + " live_policy_record[" + std::to_string(i) +
                     "] (" + row.seeded + "; " + row.evidence + ")";
        row.measurements = Json::object();
        row.absent = Json::object();
        if (const auto* a = t->get("absent")) {
            const toml::table* missing = a->as_table();
            if (!missing) bad(where + ": absent must be a table of named reasons");
            for (const auto& [name, reason] : *missing) {
                const std::string field(name.str());
                if (!names.count(field) && field != "abort_quiescent_ns")
                    bad(where + ": unknown absent field " + field);
                if (!reason.as_string() || reason.as_string()->get().empty())
                    bad(where + ": absent." + field + " must have a reason");
                if (t->contains(field)) bad(where + ": both measured and absent " + field);
                row.absent[field] = reason.as_string()->get();
            }
        }
        for (const auto& n : names) {
            if (t->contains(n)) row.measurements[n] = positive(*t, n.c_str(), where, n == "lane_lost_ns" || n == "heat_watermarked");
            else if (!row.absent.contains(n)) bad(where + ": " + n + " missing without absent reason");
        }
        if (t->contains("abort_quiescent_ns"))
            row.measurements["abort_quiescent_ns"] = positive(*t, "abort_quiescent_ns", where);
        // Legacy complete records legitimately omit the optional abort duration.
        if (row.measurements.contains("heat_watermarked") &&
            number(row.measurements, "heat_watermarked", true) > 1)
            bad(where + ": heat_watermarked must be 0 or 1");
        if (row.measurements.contains("txn_p50_ns") && row.measurements.contains("txn_p95_ns") &&
            number(row.measurements, "txn_p50_ns") > number(row.measurements, "txn_p95_ns"))
            bad(where + ": txn_p50_ns > txn_p95_ns");
        const auto too_big = [&](const char* name) {
            return row.measurements.contains(name) && number(row.measurements, name) >= kMillion;
        };
        if (too_big("marginal_error_ppm") ||
            too_big("tail_share_gain_ppm"))
            bad(where + ": invalid measured fraction");
        if (row.key == plan_key) found = std::move(row);
    }
    return found;
}

bool live_policy_record_complete(const LivePolicyRecord& record) {
    for (const char* name : {
        "txn_p50_ns", "txn_p95_ns", "copy_bytes_per_s", "heat_gap_max_ns", "heat_watermarked",
        "hot_set_k", "marginal_error_ppm", "marginal_observations", "drift_half_life_observations",
        "observation_rate_per_s", "tail_share_gain_ppm", "tail_lane_service_ns_per_s",
        "tail_lane_request_ns", "lane_lost_ns", "effect_requests_per_s", "effect_noise_max_ns",
        "effect_sample_noise_ns"
    })
        if (!record.measurements.contains(name)) return false;
    return true; // no abort seen is allowed, with the existing measured-cadence fallback
}


DerivedLivePolicy derive_live_policy(const LivePolicyRecord& record, const Calibration& calibration,
                                     const CalibrationMatch& match, uint64_t stride,
                                     const std::vector<uint64_t>& budgets) {
    if (!stride || budgets.empty()) bad("expert stride and helper budgets are required");
    if (record.layout != match.layout || record.objective != match.objective)
        bad("record layout/objective does not match the resolved serving intent");
    const Json& m = record.measurements;
    const auto n = [&](const char* key) { return number(m, key, std::string(key) == "lane_lost_ns" || std::string(key) == "heat_watermarked"); };
    for (const char* key : {
        "txn_p95_ns", "copy_bytes_per_s", "heat_gap_max_ns", "heat_watermarked", "marginal_error_ppm",
        "marginal_observations", "drift_half_life_observations", "observation_rate_per_s",
        "tail_share_gain_ppm", "tail_lane_service_ns_per_s", "tail_lane_request_ns", "lane_lost_ns", "effect_requests_per_s",
        "effect_noise_max_ns", "effect_sample_noise_ns"
    })
        (void)n(key);
    const int64_t windows = calibration.integer("calibration.live_policy.transaction_effect_windows", match);
    const int64_t false_rate = calibration.integer("calibration.live_policy.false_revert_rate_ppm", match);
    if (windows <= 0 || false_rate <= 0 || false_rate >= int64_t(kMillion))
        bad("declared objectives must be positive windows and a false-revert rate in (0,1000000)");
    // A committed plan's predicted relative tail-share reduction scales the
    // tail lane's measured service per completed request, not a quiet-window
    // statistic whose magnitude depends on the calibration gate window.
    const uint64_t delta_effect_ns = div_ceil(
        Wide(n("tail_lane_request_ns")) * n("tail_share_gain_ppm"), kMillion,
        "delta_effect_ns");
    const auto [effect_samples, effect_quiet] = derive_effect_gate(
        n("effect_sample_noise_ns"), delta_effect_ns, uint64_t(windows),
        double(false_rate) / double(kMillion));
    const uint64_t window_ns = div_ceil(Wide(effect_samples) * kSecond,
                                        n("effect_requests_per_s"), "effect window");
    const uint64_t cadence = div_ceil(Wide(n("txn_p95_ns")) + window_ns, 1, "cadence_ns");
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
    const uint64_t margin = n("marginal_error_ppm");
    if (!margin || margin >= kMillion) bad("relative sampling error is outside v1 margin_ppm range");
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
    const uint64_t recovery = div_ceil(Wide(window_ns) * (effect_quiet + 1),
                                       1, "effect gate recovery");
    const uint64_t cache_ttl = div_ceil(Wide(max_heat_age) + recovery,
                                         1, "learned eligibility TTL");
    // A WORSE verdict starts a rollback that refreshes per-expert move times.
    // Keep the destination ban through rollback, the ensuing residency and
    // one more policy poll; otherwise it expires before retry is even legal.
    const uint64_t revert_ttl = div_ceil(
        Wide(residency) + cadence + n("txn_p95_ns"), 1,
        "effect_revert_ttl_ns");
    const uint64_t effective_revert_ttl = std::max({drift_ns, cache_ttl, revert_ttl});
    DerivedLivePolicy out{Json::object(), Json::object()};
    const std::string base = "/spec/handoff/placement/live_policy/";
    const Json layout = Json::array({"/spec/intent/layout", "/spec/intent/service/objective",
                                      "/spec/plan/engine_memory/cache_key"});
    const Json budgets_input = Json::array({"/spec/processes/helpers", "/spec/plan/owner_map"});
    auto set = [&](const char* field, const Json& value, const std::string& how, const Json& inputs) {
        out.policy[field] = value;
        out.provenance[base + field] = provenance(
            "RHP16 " + how + "; " + record.source, inputs);
    };
    set("effect_gate", true, "effect gate required to check measured service noise", layout);
    if (n("heat_watermarked") > 1) bad("heat_watermarked must be 0 or 1");
    set("accept_unwatermarked_heat", n("heat_watermarked") == 0,
        n("heat_watermarked") == 0 ?
            "calibration measured the production heat feed: no step watermark; RHP02 replaces it" :
            "watermarked feed",
        Json::array({"/spec/plan/engine_memory/cache_key"}));
    set("cadence_ns", cadence, "txn_p95_ns + measured effect-window duration (requests/rate)", layout);
    set("max_moved_bytes", move_bytes, "slowest copy bytes/s x measured transaction p95, capped by helper budgets",
        Json::array({"/spec/processes/helpers", "/spec/calibration", "/spec/plan/engine_memory/cache_key"}));
    set("max_moves", max_moves, "max_moved_bytes / device-layout expert stride (bounded by uint32)", budgets_input);
    set("min_residency_ns", residency, "(committed transaction p95 + observed lane time lost) / predicted tail service saving per second", layout);
    set("max_heat_age_ns", max_heat_age, "one policy cadence plus largest accepted heat-delta gap", layout);
    set("min_samples", min_samples, "heat windows: marginal samples x (measured relative error / margin)^2 / (observation rate x cadence)", layout);
    set("abort_cooldown_ns", std::max(max_heat_age, m.contains("abort_quiescent_ns") ? n("abort_quiescent_ns") : cadence),
        m.contains("abort_quiescent_ns") ? "max(measured abort-to-quiescent, one cadence plus measured gap)" :
                                           "no abort seen: one cadence plus observed feed gap (conservative fallback; not abort-calibrated)", layout);
    set("learned_eligibility_ttl_ns", cache_ttl,
        "one cadence plus observed gap and quiet-gate recovery; event invalidation requires v2", layout);
    set("effect_window_samples", effect_samples,
        "smallest n with measured per-request sigma * sqrt(2/n) * Gaussian z(false-revert objective) <= predicted committed-move effect per request",
        Json::array({"/spec/calibration", "/spec/plan/engine_memory/cache_key"}));
    set("effect_min_quiet_windows", effect_quiet,
        "smallest consecutive quiet count with transaction_effect_windows * Gaussian noise-crossing probability^count <= declared false-revert objective",
        Json::array({"/spec/calibration", "/spec/plan/engine_memory/cache_key"}));
    set("effect_revert_ttl_ns", effective_revert_ttl, "max(drift horizon, quiet-gate recovery, rollback p95 plus residency plus cadence)", layout);
    set("margin_ppm", margin, "multinomial sampling error of the k-th hottest expert share", layout);
    set("half_life_observations_milli", div_ceil(Wide(n("drift_half_life_observations")) * 1000, 1,
                                                  "half_life_observations_milli"),
        "hot-set overlap drift half-life in observations, scaled by 1000", layout);
    set("effect_noise_multiplier_milli", noise_multiplier,
        "Gaussian two-sided false-revert z score multiplying the fractional quiet-window noise envelope", layout);
    // No guessed protected layer ids. The helper excludes experts with missing planes itself.
    // The v1 model container exposes no protected-layer list; an explicit guard needs RCS12's model metadata.
    set("protected_layers", Json::array(), "no model-global protected-layer metadata in the container (missing weights guarded by helper)",
        layout);
    return out;
}
} // namespace bc::serving
