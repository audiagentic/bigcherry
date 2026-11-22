#include "live_policy.h"
#include "plan_cache.h"
#include "third_party/toml.hpp"
#include "../moe-remote/worker_policy_derive.h"

#include <algorithm>
#include <utility>
#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

namespace bc::serving {
namespace {
using Json = nlohmann::json;
using r3::policy::derive::kMillion;
using r3::policy::derive::kMaxJsonInteger;
using r3::policy::derive::number;

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
    catch (const toml::parse_error& e) { bad(cache.string() + ": TOML parse error: " + e.what()); }
    const auto* rows = root["live_policy_record"].as_array();
    if (!rows) {
        if (root.contains("live_policy_record")) bad("live_policy_record must be an array of tables");
        return std::nullopt;
    }
    const std::set<std::string> fields = {
        "plan_cache_key", "calibration_inputs", "source_plan_key", "source_engine_build",
        "record_schema_version", "calibration_kind", "calibration_key_fields",
        "source_revision", "run_id", "host_facts_digest",
        "layout", "objective", "seeded",
        "evidence", "txn_p50_ns", "txn_p95_ns", "copy_bytes_per_s",
        "heat_gap_max_ns", "heat_watermarked", "hot_set_k", "marginal_error_ppm", "marginal_observations",
        "drift_half_life_observations", "observation_rate_per_s", "tail_share_gain_ppm",
        "tail_lane_service_ns_per_s", "tail_lane_row_ns", "lane_lost_ns", "effect_rows_per_s",
        "effect_noise_max_ns", "effect_row_noise_ns", "effect_window_noise_ladder", "abort_quiescent_ns", "absent"
    };
    const std::set<std::string> names = {
        "txn_p50_ns", "txn_p95_ns", "copy_bytes_per_s", "heat_gap_max_ns", "heat_watermarked",
        "hot_set_k", "marginal_error_ppm", "marginal_observations", "drift_half_life_observations",
        "observation_rate_per_s", "tail_share_gain_ppm", "tail_lane_service_ns_per_s",
        "tail_lane_row_ns", "lane_lost_ns", "effect_rows_per_s", "effect_noise_max_ns",
        "effect_row_noise_ns", "effect_window_noise_ladder"
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
        if (!t->contains("calibration_inputs") || !t->contains("record_schema_version"))
            bad(where + ": unsupported record format (needs calibration_inputs and record_schema_version 1); "
                        "remove it with bc-serve calibration-prune");
        {
            const std::string json = required(*t, "calibration_inputs", where);
            Json parsed = Json::parse(json, nullptr, false);
            if (parsed.is_discarded()) bad(where + ": calibration_inputs is not strict JSON");
            try {
                require_declared_policy_calibration_inputs(parsed);
                if (policy_calibration_key(parsed) != row.key)
                    bad(where + ": calibration_inputs digest disagrees with plan_cache_key");
            } catch (const std::exception& e) {
                bad(where + ": incompatible policy calibration inputs: " + e.what());
            }
            row.calibration_inputs = std::move(parsed);
            row.source_plan_key = required(*t, "source_plan_key", where);
            row.source_engine_build = required(*t, "source_engine_build", where);
            for (const std::string& hex : {row.source_plan_key, row.source_engine_build})
                if (hex.size() != 64 || hex.find_first_not_of("0123456789abcdef") != std::string::npos)
                    bad(where + ": source plan/build identities must be lowercase SHA-256");
        }
        {
            const auto* version = (*t)["record_schema_version"].as_integer();
            if (!version || version->get() != 1)
                bad(where + ": unsupported record_schema_version (expected 1)");
            row.record_schema_version = 1;
            row.calibration_kind = required(*t, "calibration_kind", where);
            if (row.calibration_kind != "radiance.live_policy")
                bad(where + ": unknown calibration_kind");
            const auto* fields = (*t)["calibration_key_fields"].as_array();
            if (!fields) bad(where + ": calibration_key_fields must be an array");
            Json names = Json::array();
            for (const auto& field : *fields) {
                const auto* value = field.as_string();
                if (!value) bad(where + ": calibration_key_fields must contain strings");
                names.push_back(value->get());
            }
            if (names != policy_calibration_key_fields())
                bad(where + ": calibration_key_fields differs from platform declaration");
            row.calibration_key_fields = std::move(names);
            if (!row.calibration_inputs)
                bad(where + ": version 1 requires declared calibration inputs");
            try { require_declared_policy_calibration_inputs(*row.calibration_inputs); }
            catch (const std::exception& e) { bad(where + ": " + e.what()); }
            row.source_revision = required(*t, "source_revision", where);
            row.run_id = required(*t, "run_id", where);
            row.host_facts_digest = required(*t, "host_facts_digest", where);
            if (row.host_facts_digest.size() != 64 ||
                row.host_facts_digest.find_first_not_of("0123456789abcdef") != std::string::npos)
                bad(where + ": host_facts_digest must be lowercase SHA-256");
        }
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
            if(n=="effect_window_noise_ladder") continue;
            if (t->contains(n)) row.measurements[n] = positive(*t, n.c_str(), where, n == "lane_lost_ns" || n == "heat_watermarked");
            else if (!row.absent.contains(n)) bad(where + ": " + n + " missing without absent reason");
        }
        if(const auto* node=t->get("effect_window_noise_ladder")) {
            const auto* ladder=node->as_array();
            if(!ladder || ladder->size()<3)
                bad(where + ": effect_window_noise_ladder requires at least three lengths");
            Json pairs=Json::array();
            uint64_t last=0;
            for(const auto& item:*ladder) {
                const auto* pair=item.as_array();
                if(!pair || pair->size()!=2 || !(*pair)[0].as_integer() || !(*pair)[1].as_integer())
                    bad(where + ": invalid effect_window_noise_ladder pair");
                const auto rows=(*pair)[0].as_integer()->get();
                const auto noise=(*pair)[1].as_integer()->get();
                if(rows<=0 || uint64_t(rows)<=last || uint64_t(rows)>kMaxJsonInteger ||
                   noise<0 || uint64_t(noise)>kMaxJsonInteger)
                    bad(where + ": invalid effect_window_noise_ladder rows/noise");
                last=uint64_t(rows);
                pairs.push_back(Json::array({last,uint64_t(noise)}));
            }
            row.measurements["effect_window_noise_ladder"]=std::move(pairs);
        } else if(!row.absent.contains("effect_window_noise_ladder"))
            bad(where + ": effect_window_noise_ladder missing without absent reason");
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
        "tail_lane_row_ns", "lane_lost_ns"
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
    r3::policy::derive::require_planner_measurements(record.measurements);
    // The arithmetic is the shared derivation (moe-remote/worker_policy_derive.h); this keeps the record, the declared
    // objectives and the manifest provenance.
    const auto v = r3::policy::derive::derive_policy_values(
        record.measurements, record.absent,
        calibration.integer("calibration.live_policy.transaction_effect_windows", match),
        calibration.integer("calibration.live_policy.false_revert_rate_ppm", match), stride, budgets);
    const bool effect_enabled = v.effect_gate;
    const Json layout = Json::array({"/spec/intent/layout", "/spec/intent/service/objective",
                                      "/spec/plan/engine_memory/cache_key"});
    const Json budgets_input = Json::array({"/spec/processes/helpers", "/spec/plan/owner_map"});
    DerivedLivePolicy out{Json::object(), Json::object()};
    const std::string base = "/spec/handoff/placement/live_policy/";
    auto set = [&](const char* field, const Json& value, const std::string& how, const Json& inputs) {
        out.policy[field] = value;
        out.provenance[base + field] = provenance(
            "RHP16 " + how + "; " + record.source, inputs);
    };
    set("effect_gate", effect_enabled, effect_enabled ?
        "relative tail/peer service judgeable with a measured resolvable noise ladder" :
        v.effect_reason + "; churn bounded by derived margin, residency and move caps", layout);
    set("accept_unwatermarked_heat", v.accept_unwatermarked_heat,
        v.accept_unwatermarked_heat ?
            "calibration measured the production heat feed: no step watermark; RHP02 replaces it" :
            "watermarked feed",
        Json::array({"/spec/plan/engine_memory/cache_key"}));
    set("cadence_ns", v.cadence_ns, effect_enabled ?
        "cadence increased as necessary: txn_p95_ns + k x measured tail window duration; no window truncation" :
        "effect gate off: max(measured transaction p95, observed heat gap); no unjudgeable quiet-window wait", layout);
    set("max_moved_bytes", v.max_moved_bytes, "slowest copy bytes/s x measured transaction p95, capped by helper budgets",
        Json::array({"/spec/processes/helpers", "/spec/calibration", "/spec/plan/engine_memory/cache_key"}));
    set("max_moves", v.max_moves, "max_moved_bytes / device-layout expert stride (bounded by uint32)", budgets_input);
    set("min_residency_ns", v.min_residency_ns, "(committed transaction p95 + observed lane time lost) / predicted tail service saving per second", layout);
    set("max_heat_age_ns", v.max_heat_age_ns, "one policy cadence plus largest accepted heat-delta gap", layout);
    set("min_samples", v.min_samples, "heat windows: min(measured, one drift half-life) observations / (observation rate x cadence)", layout);
    set("abort_cooldown_ns", v.abort_cooldown_ns,
        v.abort_measured ? "max(measured abort-to-quiescent, one cadence plus measured gap)" :
                                           "no abort seen: one cadence plus observed feed gap (conservative fallback; not abort-calibrated)", layout);
    set("learned_eligibility_ttl_ns", v.learned_eligibility_ttl_ns,
        "one cadence plus observed gap and quiet-gate recovery; event invalidation requires v2", layout);
    set("effect_window_samples", v.effect_window_samples,
        effect_enabled ? "smallest observed relative-service (tail ns/row / row-weighted other-helpers ns/row) window with RMS fractional noise x Gaussian z <= predicted tail-share reduction g = gain_ppm/1e6; r_after = r_before*(1-g), assuming peer per-row service unchanged and tail service proportional to share; no i.i.d. extrapolation" :
        "effect gate off: inert schema value, no effect sample length inferred",
        Json::array({"/spec/calibration", "/spec/plan/engine_memory/cache_key"}));
    set("effect_min_quiet_windows", v.effect_min_quiet_windows,
        effect_enabled ? "smallest consecutive quiet count with transaction_effect_windows * Gaussian noise-crossing probability^count <= declared false-revert objective" :
        "effect gate off: inert schema value, no quiet count inferred",
        Json::array({"/spec/calibration", "/spec/plan/engine_memory/cache_key"}));
    set("effect_revert_ttl_ns", v.effect_revert_ttl_ns, "max(drift horizon, quiet-gate recovery, rollback p95 plus residency plus cadence)", layout);
    set("margin_ppm", v.margin_ppm, "multinomial sampling error of the k-th hottest expert share over min(measured, one drift half-life) observations", layout);
    set("half_life_observations_milli", v.half_life_observations_milli,
        "hot-set overlap drift half-life in observations, scaled by 1000", layout);
    set("effect_noise_multiplier_milli", v.effect_noise_multiplier_milli,
        "Gaussian two-sided false-revert z score multiplying same-window relative tail/peer service quiet noise", layout);
    // No guessed protected layer ids. The helper excludes experts with missing planes itself.
    // The v1 model container exposes no protected-layer list; an explicit guard needs RCS12's model metadata.
    set("protected_layers", Json::array(), "no model-global protected-layer metadata in the container (missing weights guarded by helper)",
        layout);
    return out;
}
} // namespace bc::serving
