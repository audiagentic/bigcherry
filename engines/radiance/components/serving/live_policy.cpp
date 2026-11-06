#include "live_policy.h"
#include "third_party/toml.hpp"

#include <algorithm>
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
    catch (const toml::parse_error& e) { bad(cache.string() + ": " + e.what()); }
    const auto* rows = root["live_policy_record"].as_array();
    if (!rows) {
        if (root.contains("live_policy_record")) bad("live_policy_record must be an array of tables");
        return std::nullopt;
    }
    const std::set<std::string> fields = {
        "plan_cache_key", "layout", "objective", "seeded", "evidence",
        "txn_p50_ns", "txn_p95_ns", "copy_bytes_per_s", "heat_gap_max_ns",
        "hot_set_k", "marginal_error_ppm", "marginal_observations",
        "drift_half_life_observations", "observation_rate_per_s",
        "tail_share_noise_ppm", "tail_share_gain_ppm", "tail_lane_service_ns_per_s",
        "lane_lost_ns", "effect_window_samples", "effect_requests_per_s",
        "effect_quiet_windows", "effect_noise_floor_ns", "effect_noise_quantile_ns",
        "noise_false_revert_rate_ppm", "abort_quiescent_ns", "absent"

    };
    const std::set<std::string> names = {
        "txn_p50_ns", "txn_p95_ns", "copy_bytes_per_s", "heat_gap_max_ns",
        "hot_set_k", "marginal_error_ppm", "marginal_observations",
        "drift_half_life_observations", "observation_rate_per_s",
        "tail_share_noise_ppm", "tail_share_gain_ppm", "tail_lane_service_ns_per_s",
        "lane_lost_ns", "effect_window_samples", "effect_requests_per_s",
        "effect_quiet_windows", "effect_noise_floor_ns", "effect_noise_quantile_ns",
        "noise_false_revert_rate_ppm"
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
            if (t->contains(n)) row.measurements[n] = positive(*t, n.c_str(), where, n == "lane_lost_ns");
            else if (!row.absent.contains(n)) bad(where + ": " + n + " missing without absent reason");
        }
        if (t->contains("abort_quiescent_ns"))
            row.measurements["abort_quiescent_ns"] = positive(*t, "abort_quiescent_ns", where);
        // Legacy complete records legitimately omit the optional abort duration.
        if (row.measurements.contains("txn_p50_ns") && row.measurements.contains("txn_p95_ns") &&
            number(row.measurements, "txn_p50_ns") > number(row.measurements, "txn_p95_ns"))
            bad(where + ": txn_p50_ns > txn_p95_ns");
        const auto too_big = [&](const char* name) {
            return row.measurements.contains(name) && number(row.measurements, name) >= kMillion;
        };
        if (too_big("tail_share_noise_ppm") || too_big("marginal_error_ppm") ||
            too_big("tail_share_gain_ppm") || too_big("noise_false_revert_rate_ppm") ||
            (row.measurements.contains("effect_noise_quantile_ns") &&
             row.measurements.contains("effect_noise_floor_ns") &&
             number(row.measurements, "effect_noise_quantile_ns") <
             number(row.measurements, "effect_noise_floor_ns")))

            bad(where + ": invalid measured fraction or noise quantile");
        if (row.key == plan_key) found = std::move(row);
    }
    return found;
}

bool live_policy_record_complete(const LivePolicyRecord& record) {
    for (const char* name : {"txn_p50_ns", "txn_p95_ns", "copy_bytes_per_s", "heat_gap_max_ns",
        "hot_set_k", "marginal_error_ppm", "marginal_observations",
        "drift_half_life_observations", "observation_rate_per_s", "tail_share_noise_ppm",
        "tail_share_gain_ppm", "tail_lane_service_ns_per_s", "lane_lost_ns",
        "effect_window_samples", "effect_requests_per_s", "effect_quiet_windows",
        "effect_noise_floor_ns", "effect_noise_quantile_ns", "noise_false_revert_rate_ppm"})
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
    const auto n = [&](const char* key) { return number(m, key, std::string(key) == "lane_lost_ns"); };
    for (const char* key : {"txn_p95_ns", "copy_bytes_per_s", "heat_gap_max_ns",
        "marginal_error_ppm", "marginal_observations", "drift_half_life_observations",
        "observation_rate_per_s", "tail_share_noise_ppm", "tail_share_gain_ppm",
        "tail_lane_service_ns_per_s", "lane_lost_ns", "effect_window_samples",
        "effect_requests_per_s", "effect_quiet_windows", "effect_noise_floor_ns",
        "effect_noise_quantile_ns", "noise_false_revert_rate_ppm"})
        (void)n(key);
    const int64_t windows = calibration.integer("calibration.live_policy.transaction_effect_windows", match);
    const int64_t false_rate = calibration.integer("calibration.live_policy.false_revert_rate_ppm", match);
    if (windows <= 0 || false_rate <= 0 || false_rate >= int64_t(kMillion))
        bad("declared objectives must be positive windows and a false-revert rate in (0,1000000)");
    if (n("noise_false_revert_rate_ppm") != uint64_t(false_rate))
        bad("measured noise quantile does not match the declared false-revert objective");
    const uint64_t window_ns = div_ceil(Wide(n("effect_window_samples")) * kSecond,
                                        n("effect_requests_per_s"), "effect window");
    const uint64_t cadence = div_ceil(Wide(n("txn_p95_ns")) + window_ns, 1, "cadence_ns");
    const uint64_t slow_budget = *std::min_element(budgets.begin(), budgets.end());
    if (!slow_budget || slow_budget < stride) bad("a helper budget cannot hold one expert stride");
    const uint64_t move_bytes = std::min(
        slow_budget, div_ceil(Wide(n("copy_bytes_per_s")) * window_ns * uint64_t(windows),
                              kSecond, "max_moved_bytes"));
    const uint64_t max_moves = std::min<uint64_t>(move_bytes / stride, uint64_t(UINT32_MAX));
    if (!max_moves) bad("measured budget buys fewer than one complete expert; cannot activate policy");
    const uint64_t margin = std::max(n("tail_share_noise_ppm"), n("marginal_error_ppm"));
    if (!margin || margin >= kMillion) bad("relative noise is outside v1 margin_ppm range");
    const uint64_t min_samples = div_ceil(
        Wide(n("marginal_observations")) * n("marginal_error_ppm") * n("marginal_error_ppm"),
        Wide(margin) * margin, "min_samples");
    const uint64_t move_cost_ns = div_ceil(Wide(move_bytes) * kSecond,
                                           n("copy_bytes_per_s"), "move cost");
    const uint64_t savings_ns_per_s = div_ceil(
        Wide(n("tail_share_gain_ppm")) * n("tail_lane_service_ns_per_s"), kMillion,
        "predicted tail savings");
    const uint64_t residency = div_ceil(
        (Wide(move_cost_ns) + n("lane_lost_ns")) * kSecond,
        savings_ns_per_s, "min_residency_ns");
    const uint64_t drift_ns = div_ceil(
        Wide(n("drift_half_life_observations")) * kSecond, n("observation_rate_per_s"),
        "effect_revert_ttl_ns");
    const uint64_t noise_multiplier = div_ceil(
        Wide(n("effect_noise_quantile_ns")) * 1000, n("effect_noise_floor_ns"),
        "effect_noise_multiplier_milli");

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
    set("accept_unwatermarked_heat", false, "complete heat watermark required; not an operator opt-in", layout);
    set("cadence_ns", cadence, "txn_p95_ns + measured effect-window duration (requests/rate)", layout);
    set("max_moved_bytes", move_bytes, "slowest copy bytes/s x effect window x declared transaction windows, capped by helper budgets",
        Json::array({"/spec/processes/helpers", "/spec/calibration", "/spec/plan/engine_memory/cache_key"}));
    set("max_moves", max_moves, "max_moved_bytes / device-layout expert stride (bounded by uint32)", budgets_input);
    set("min_residency_ns", residency, "(bytes / measured copy rate + lane time lost) / predicted tail service saving per second", layout);
    set("max_heat_age_ns", n("heat_gap_max_ns"), "largest measured heat-delta gap", layout);
    set("min_samples", min_samples, "marginal samples x (measured relative error / noise margin)^2", layout);
    set("abort_cooldown_ns", m.contains("abort_quiescent_ns") ? n("abort_quiescent_ns") : cadence,
        m.contains("abort_quiescent_ns") ? "measured abort-to-quiescent" :
                                           "no abort seen: transaction settling cadence (conservative fallback; not abort-calibrated)", layout);
    set("learned_eligibility_ttl_ns", n("heat_gap_max_ns"),
        "v1 mandatory time TTL: one measured heat source gap (event invalidation requires v2)", layout);
    set("effect_window_samples", n("effect_window_samples"), "measured lane-noise variance convergence", layout);
    set("effect_min_quiet_windows", n("effect_quiet_windows"), "measured quiet-window confidence convergence", layout);
    set("effect_revert_ttl_ns", drift_ns, "hot-set drift half-life / observation rate", layout);
    set("margin_ppm", margin, "maximum of unchanged-tail noise and marginal sampling error", layout);
    set("half_life_observations_milli", div_ceil(Wide(n("drift_half_life_observations")) * 1000, 1,
                                                  "half_life_observations_milli"),
        "hot-set overlap drift half-life in observations, scaled by 1000", layout);
    set("effect_noise_multiplier_milli", noise_multiplier,
        "empirical noise quantile at declared false-revert rate / measured noise floor", layout);
    // No guessed protected layer ids. The helper excludes experts with missing planes itself.
    // The v1 model container exposes no protected-layer list; an explicit guard needs RCS12's model metadata.
    set("protected_layers", Json::array(), "no model-global protected-layer metadata in the container (missing weights guarded by helper)",
        layout);
    return out;
}
} // namespace bc::serving
