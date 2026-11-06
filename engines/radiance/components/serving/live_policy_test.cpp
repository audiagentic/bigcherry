#include "live_policy.h"

#include <cassert>
#ifdef NDEBUG
#error "assert-based test built with NDEBUG: its assertions would not run"
#endif
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
namespace sv = bc::serving;
using Json = nlohmann::json;

// Fixture writer for the future helper's step-4 calibration store. The transaction,
// copy, heat gap, marginal error and hot-set readings are from policy-live-9;
// the effect, lane and drift quantities NOT measured there are synthetic test data.
static std::string fixture(const std::string& key, bool malformed = false) {
    return "schema_version = 1\n"
           "[[live_policy_record]]\n"
           "plan_cache_key = \"" + key + "\"\n"
           "layout = \"flashnext-4gpu\"\n"
           "objective = \"balanced\"\n"
           "seeded = \"2026-11-05\"\n"
           "evidence = \"policy-live-9 measured timings, copy, gap, marginal error; supplemental synthetic effect/lanes/drift\"\n"
           "txn_p50_ns = 1770000000\n"
           "txn_p95_ns = 2720000000\n"
           "copy_bytes_per_s = 160000000\n"
           "heat_gap_max_ns = 5040000000\n"
           "hot_set_k = 5931\n"
           "marginal_error_ppm = 23000\n"
           "marginal_observations = 3\n"
           "drift_half_life_observations = 59000000\n"
           "observation_rate_per_s = 10000000\n"
           "tail_share_noise_ppm = 20000\n"
           "tail_share_gain_ppm = 45000\n"
           "tail_lane_service_ns_per_s = 750000000\n"
           "lane_lost_ns = 100000000\n"
           "effect_window_samples = 32\n"
           "effect_requests_per_s = 64\n"
           "effect_quiet_windows = 3\n"
           "effect_noise_floor_ns = 1000000\n"
           "effect_noise_quantile_ns = 2500000\n"
           "noise_false_revert_rate_ppm = 10000\n"
           "abort_quiescent_ns = 1500000000\n" +
           (malformed ? "unexpected_guess = 42\n" : "");
}
static fs::path write_fixture(const std::string& data) {
    const fs::path file = fs::temp_directory_path() / "bc-live-policy-calibration-test.toml";
    std::ofstream out(file);
    if (!out) throw std::runtime_error("cannot write test fixture");
    out << data;
    out.close();
    return file;
}
static bool throws_with(const fs::path& file, const std::string& key, const std::string& part) {
    try { (void)sv::read_live_policy_record(file, key); }
    catch (const std::runtime_error& e) { return std::string(e.what()).find(part) != std::string::npos; }
    return false;
}
int main() {
    const std::string key(64, 'a');
    const fs::path file = write_fixture("schema_version = 1\n");
    assert(!sv::read_live_policy_record(file, key));
    write_fixture(fixture(std::string(64, 'b')));
    assert(!sv::read_live_policy_record(file, key)); // different plan: remeasure, no reuse
    write_fixture(fixture(key));
    auto rec = sv::read_live_policy_record(file, key);
    assert(rec && rec->measurements.at("hot_set_k") == 5931);
    assert(sv::live_policy_record_complete(*rec));

    const auto cal = sv::Calibration::standard(file);
    const sv::CalibrationMatch at{"", "flashnext-4gpu", "balanced"};
    const auto d = sv::derive_live_policy(*rec, cal, at, 2500000, {1000000000, 1000000000});
    const Json& p = d.policy;
    assert(p.size() == 17);
    assert(p.at("cadence_ns") == 3220000000ull);
    assert(p.at("max_moved_bytes") == 160000000ull);
    assert(p.at("max_moves") == 64);
    assert(p.at("min_residency_ns") == 32592592593ull);
    assert(p.at("max_heat_age_ns") == 5040000000ull);
    assert(p.at("min_samples") == 3 && p.at("margin_ppm") == 23000);
    assert(p.at("abort_cooldown_ns") == 1500000000ull);
    assert(p.at("learned_eligibility_ttl_ns") == 5040000000ull);
    assert(p.at("effect_window_samples") == 32 && p.at("effect_min_quiet_windows") == 3);
    assert(p.at("half_life_observations_milli") == 59000000000ull);
    assert(p.at("effect_revert_ttl_ns") == 5900000000ull);
    assert(p.at("effect_noise_multiplier_milli") == 2500);
    assert(p.at("effect_gate") == true && p.at("accept_unwatermarked_heat") == false);
    assert(p.at("protected_layers").empty());
    assert(d.provenance.size() == p.size());
    for (auto it = d.provenance.begin(); it != d.provenance.end(); ++it)
        assert(it.value().at("origin") == "derived" && it.value().at("class") == "policy" &&
               it.value().at("source").get<std::string>().find("policy-live-9") != std::string::npos);
    // The budgets and stride are inputs, never fixed policy constants.
    const auto budget = sv::derive_live_policy(*rec, cal, at, 2500000, {100000000, 1000000000});
    assert(budget.policy.at("max_moved_bytes") == 100000000 && budget.policy.at("max_moves") == 40);
    const auto stride = sv::derive_live_policy(*rec, cal, at, 4000000, {1000000000, 1000000000});
    assert(stride.policy.at("max_moved_bytes") == 160000000 && stride.policy.at("max_moves") == 40);
    assert(stride.policy.at("min_residency_ns") == p.at("min_residency_ns"));
    // Partial evidence is accepted only with a named absence and cannot
    // silently become a policy with guessed numbers.
    std::string partial = fixture(key);
    const std::string copy = "copy_bytes_per_s = 160000000\n";
    assert(partial.find(copy) != std::string::npos);
    partial.erase(partial.find(copy), copy.size());
    partial += "absent = { copy_bytes_per_s = \"per-helper copy hook absent\" }\n";
    write_fixture(partial);
    const auto incomplete = sv::read_live_policy_record(file, key);
    assert(incomplete && !sv::live_policy_record_complete(*incomplete));
    assert(incomplete->absent.at("copy_bytes_per_s") == "per-helper copy hook absent");
    partial = fixture(key);
    partial.erase(partial.find(copy), copy.size());
    write_fixture(partial);
    assert(throws_with(file, key, "copy_bytes_per_s"));

    write_fixture(fixture(key, true));
    assert(throws_with(file, key, "unknown key"));
    write_fixture(fixture(key).substr(0, fixture(key).find("copy_bytes_per_s = ")));
    assert(throws_with(file, key, "copy_bytes_per_s"));
    // Two records for one plan key under a single header: refused as a duplicate, never first-one-wins.
    const std::string header = "schema_version = 1\n";
    write_fixture(fixture(key) + fixture(key).substr(header.size()));
    assert(throws_with(file, key, "duplicate"));
    // A second header inside a record table is malformed TOML for this schema.
    write_fixture(fixture(key) + fixture(key));
    assert(throws_with(file, key, "unknown key"));
    fs::remove(file);
}
