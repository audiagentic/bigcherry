#include "live_policy.h"

#include <cassert>
#include <cmath>
#include "../moe-remote/worker_policy_effect.h"
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
           "heat_watermarked = 0\n"
           "hot_set_k = 5931\n"
           "marginal_error_ppm = 23000\n"
           "marginal_observations = 3\n"
           "drift_half_life_observations = 59000000\n"
           "observation_rate_per_s = 10000000\n"
           "tail_share_noise_ppm = 20000\n"
           "tail_share_gain_ppm = 45000\n"
           "tail_lane_service_ns_per_s = 750000000\n"
           "tail_lane_request_ns = 22222222\n"
           "lane_lost_ns = 100000000\n"
           "effect_requests_per_s = 64\n"
           "effect_noise_max_ns = 2500000\n"
           "effect_sample_noise_ns = 1550000\n"
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
// Feed actual quiet-window means at two different configured sizes. Their
// mean differences scale by sqrt(1/n), while normalized per-request sigma
// stays fixed; neither the effect threshold nor the derived n/k may echo n_cfg.
static r3::policy::EffectSamples synthetic_quiet(uint64_t n_cfg) {
    r3::policy::EffectGate gate({n_cfg,2,2.0});
    r3::policy::ServiceSnapshot snapshot{};
    gate.quiet(snapshot,1000);
    for (uint64_t i=0;i<9;++i) {
        auto& tail=snapshot.lane[1];
        tail.samples+=n_cfg;
        tail.rows+=n_cfg*10;
        const uint64_t mean_ns=5000000+(i%2)*(n_cfg==2?1550000:775000);
        tail.time_ns+=n_cfg*mean_ns;
        gate.quiet(snapshot,2000+1000*i);
    }
    return gate.observed();
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
    // min_samples counts heat windows: 3 observations needed, 10M/s x 3.22 s per window -> one window.
    assert(p.at("min_samples") == 1 && p.at("margin_ppm") == 23000);
    {   // rhp16-calib4's measured sampling: 105,925,840 observations at 1,300,138/s. In observations the gate never
        // opens (the planner counts windows); in windows it is ceil(needed / (rate x cadence)).
        auto real = *rec;
        real.measurements["marginal_observations"] = 105925840;
        real.measurements["observation_rate_per_s"] = 1300138;
        const Json& q = sv::derive_live_policy(real, cal, at, 2500000, {1000000000, 1000000000}).policy;
        const uint64_t per_window = (uint64_t(1300138) * q.at("cadence_ns").get<uint64_t>() + 999999999) / 1000000000;
        assert(q.at("min_samples") == (uint64_t(105925840) + per_window - 1) / per_window);
        assert(q.at("min_samples").get<uint64_t>() < 100);
    }
    assert(p.at("abort_cooldown_ns") == 1500000000ull);
    assert(p.at("learned_eligibility_ttl_ns") == 5040000000ull);
    assert(p.at("effect_window_samples") == 32 && p.at("effect_min_quiet_windows") == 2);
    assert(p.at("half_life_observations_milli") == 59000000000ull);
    assert(p.at("effect_revert_ttl_ns") == 5900000000ull);
    assert(p.at("effect_noise_multiplier_milli") == 2500);
    const auto small_series=synthetic_quiet(2);
    const auto wide_series=synthetic_quiet(8);
    assert(small_series.quiet_comparisons==wide_series.quiet_comparisons &&
           small_series.quiet_comparisons==8);
    assert(small_series.quiet_delta_ns.max()==1550000 &&
           wide_series.quiet_delta_ns.max()==775000);
    assert(std::abs(small_series.sample_noise_ns()-1550000.0)<1e-6 &&
           std::abs(wide_series.sample_noise_ns()-1550000.0)<1e-6);
    auto small_window=*rec, wide_window=*rec;
    small_window.measurements["effect_sample_noise_ns"]=uint64_t(std::ceil(small_series.sample_noise_ns()));
    wide_window.measurements["effect_sample_noise_ns"]=uint64_t(std::ceil(wide_series.sample_noise_ns()));
    small_window.measurements["effect_noise_max_ns"]=uint64_t(small_series.quiet_delta_ns.max());
    wide_window.measurements["effect_noise_max_ns"]=uint64_t(wide_series.quiet_delta_ns.max());
    const auto derived_small=sv::derive_live_policy(small_window,cal,at,2500000,{1000000000,1000000000});
    const auto derived_wide=sv::derive_live_policy(wide_window,cal,at,2500000,{1000000000,1000000000});
    assert(derived_small.policy.at("effect_window_samples")==derived_wide.policy.at("effect_window_samples") &&
           derived_small.policy.at("effect_window_samples")==32);
    assert(derived_small.policy.at("effect_min_quiet_windows")==derived_wide.policy.at("effect_min_quiet_windows") &&
           derived_small.policy.at("effect_min_quiet_windows")==2);
    assert(derived_small.policy.at("effect_noise_multiplier_milli") !=
           derived_wide.policy.at("effect_noise_multiplier_milli"));
    // Neither gate setting is copied from the lab policy: changing measured
    // dispersion changes the derived window and can change the quiet count.
    auto quieter = *rec;
    quieter.measurements["effect_sample_noise_ns"] = 500000;
    const auto q = sv::derive_live_policy(quieter, cal, at, 2500000, {1000000000, 1000000000});
    assert(q.policy.at("effect_window_samples") == 4);
    assert(q.policy.at("effect_min_quiet_windows") == 1);
    auto noisier = *rec;
    noisier.measurements["effect_sample_noise_ns"] = 3100000;
    const auto h = sv::derive_live_policy(noisier, cal, at, 2500000, {1000000000, 1000000000});
    assert(h.policy.at("effect_window_samples") == 128);
    assert(q.provenance.at("/spec/handoff/placement/live_policy/effect_window_samples").at("origin") == "derived");
    assert(q.provenance.at("/spec/handoff/placement/live_policy/effect_min_quiet_windows").at("origin") == "derived");
    assert(p.at("effect_gate") == true && p.at("accept_unwatermarked_heat") == true);
    assert(d.provenance.at("/spec/handoff/placement/live_policy/accept_unwatermarked_heat")
               .at("origin") == "derived");
    assert(d.provenance.at("/spec/handoff/placement/live_policy/accept_unwatermarked_heat")
               .at("source").get<std::string>().find("no step watermark; RHP02 replaces it") != std::string::npos);
    auto watermarked = *rec;
    watermarked.measurements["heat_watermarked"] = 1;
    const auto with_watermark = sv::derive_live_policy(watermarked,cal,at,2500000,{1000000000,1000000000});
    assert(with_watermark.policy.at("accept_unwatermarked_heat") == false);
    assert(with_watermark.provenance.at("/spec/handoff/placement/live_policy/accept_unwatermarked_heat")
               .at("source").get<std::string>().find("watermarked feed") != std::string::npos);

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

    // The zero-valued reading is valid TOML, but a missing or nonbinary reading is not.
    std::string no_watermark = fixture(key);
    const std::string zero_field = "heat_watermarked = 0\n";
    no_watermark.erase(no_watermark.find(zero_field),zero_field.size());
    write_fixture(no_watermark);
    assert(throws_with(file,key,"heat_watermarked"));
    write_fixture(fixture(key) + "heat_watermarked = 2\n");
    assert(throws_with(file,key,"TOML") || throws_with(file,key,"duplicate"));
    auto bad_mark = fixture(key);
    bad_mark.replace(bad_mark.find(zero_field),zero_field.size(),"heat_watermarked = 2\n");
    write_fixture(bad_mark);
    assert(throws_with(file,key,"heat_watermarked must be 0 or 1"));
    write_fixture(fixture(key, true));
    assert(throws_with(file, key, "unknown key"));
    // Old measured keys must not have a compatibility path.
    for (const char* obsolete : {"noise_false_revert_rate_ppm", "effect_window_samples",
                                 "effect_quiet_windows", "effect_noise_quantile_ns",
                                 "effect_noise_floor_ns"}) {
        write_fixture(fixture(key) + std::string(obsolete) + " = 1\n");
        assert(throws_with(file, key, "unknown key"));
    }
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
