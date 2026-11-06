#include "live_policy.h"

#include <cassert>
#include <cmath>
#include "../moe-remote/worker_policy_effect.h"
#include "../moe-remote/worker_policy_config.h"
#include "../moe-remote/worker_policy_input.h"
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

// calib5 hardware measurements, not lab policy knobs. Run the derived
// settings through the typed helper adapter and the real CPU heat planner.
static void calib5_regression(const sv::LivePolicyRecord& base,
                              const sv::Calibration& cal,
                              const sv::CalibrationMatch& at) {
    auto rec = base;
    auto& m = rec.measurements;
    m["txn_p50_ns"]=1593403198ull; m["txn_p95_ns"]=3794447513ull;
    m["copy_bytes_per_s"]=319377397ull;
    m["heat_gap_max_ns"]=6955375341ull; m["heat_watermarked"]=0;
    m["hot_set_k"]=5931; m["marginal_error_ppm"]=31905;
    m["marginal_observations"]=105169020ull;
    m["drift_half_life_observations"]=15849692ull;
    m["observation_rate_per_s"]=1282858;
    m["tail_share_gain_ppm"]=98958;
    m["tail_lane_service_ns_per_s"]=468160374ull;
    m["tail_lane_request_ns"]=6296371;
    m["lane_lost_ns"]=14766582; m["effect_requests_per_s"]=71;
    m["effect_noise_max_ns"]=3358515;
    m["effect_sample_noise_ns"]=4446853;
    constexpr uint64_t stride=2534544; // calib5: 14 GiB / 5,931 tail experts
    const auto derived = sv::derive_live_policy(
        rec,cal,at,stride,{22ull<<30,22ull<<30,14ull<<30});
    const Json& p=derived.policy;
    assert(p.size()==17);
    assert(p.at("protected_layers").empty());
    assert(p.at("abort_cooldown_ns")==20270949615ull);
    assert(p.at("margin_ppm")==31905);
    assert(p.at("min_samples")==7);
    assert(p.at("cadence_ns")==13315574274ull);
    assert(p.at("max_moved_bytes")==1211860770ull);
    assert(p.at("max_moves")==478);
    assert(p.at("min_residency_ns")==82222336756ull);
    assert(p.at("max_heat_age_ns")==20270949615ull);
    assert(p.at("learned_eligibility_ttl_ns")==48834329898ull);
    assert(p.at("effect_revert_ttl_ns")==99332358543ull);
    assert(p.at("effect_revert_ttl_ns").get<uint64_t>() >
           p.at("min_residency_ns").get<uint64_t>() +
           p.at("cadence_ns").get<uint64_t>());
    assert(p.at("effect_window_samples")==676);
    assert(p.at("effect_min_quiet_windows")==2);
    assert(p.at("effect_noise_multiplier_milli")==2576);
    assert(p.at("half_life_observations_milli")==15849692000ull);
    bc::serving::v1::LivePolicy typed{};
    typed.effect_gate=p.at("effect_gate").get<bool>();
    typed.accept_unwatermarked_heat=p.at("accept_unwatermarked_heat").get<bool>();
#define LOAD_POLICY(x) typed.x=p.at(#x).get<int64_t>();
    LOAD_POLICY(cadence_ns) LOAD_POLICY(max_moves)
    LOAD_POLICY(max_moved_bytes) LOAD_POLICY(min_residency_ns)
    LOAD_POLICY(max_heat_age_ns) LOAD_POLICY(min_samples)
    LOAD_POLICY(abort_cooldown_ns) LOAD_POLICY(learned_eligibility_ttl_ns)
    LOAD_POLICY(effect_window_samples) LOAD_POLICY(effect_min_quiet_windows)
    LOAD_POLICY(effect_revert_ttl_ns) LOAD_POLICY(margin_ppm)
    LOAD_POLICY(half_life_observations_milli)
    LOAD_POLICY(effect_noise_multiplier_milli)
#undef LOAD_POLICY
    typed.protected_layers=p.at("protected_layers").get<std::vector<int64_t>>();
    r3::policy::Settings cfg;
    std::string error;
    assert(cfg.from_manifest(typed,{1,1},&error));
    assert(cfg.accept_unwatermarked_heat && cfg.effect_gate);
    assert(cfg.controls.margin==0.031905);
    assert(cfg.controls.half_life_observations==15849692.0);
    assert(cfg.protected_layers.empty());
    assert(cfg.effect_noise_multiplier==2.576);
    r3::heat::Heat heat(1,4,cfg.controls.half_life_observations);
    r3::policy::HeatDelta delta{};
    delta.layers=1; delta.experts=4; delta.complete_cells=true;
    delta.complete_observations=false;
    delta.provenance=r3::policy::HeatProvenance::ENGINE_HISTOGRAM_FILE;
    delta.age_ns=1000000000ull;
    uint64_t consumed=0;
    auto feed=[&](std::vector<int32_t> counts) {
        delta.counts=std::move(counts); delta.interval_observations=0;
        for (int32_t v:delta.counts) delta.interval_observations+=v;
        delta.observation_count+=delta.interval_observations;
        assert(r3::policy::consume_heat(
            delta,&heat,&consumed,cfg.accept_unwatermarked_heat));
    };
    for (int i=0;i<8;++i) feed({6000000,5500000,1200000,1100000});
    r3::heat::Input in;
    in.bytes.assign(4,stride); in.current={0,0,1,1};
    in.devices={{2*stride},{2*stride}};
    in.allowed.assign(4,{1,1});
    in.now_ns=1000000000000ull;
    in.observed_at_ns=in.now_ns-delta.age_ns;
    in.max_heat_age_ns=cfg.controls.max_heat_age_ns;
    in.min_residency_ns=cfg.controls.min_residency_ns;
    in.max_moved_bytes=cfg.controls.max_moved_bytes;
    in.min_samples=cfg.controls.min_samples;
    in.margin=cfg.controls.margin; in.max_moves=cfg.controls.max_moves;
    auto plan=[&]() {
        in.heat=heat.counts; in.samples=heat.samples;
        return r3::heat::propose(in,cfg.head_shares);
    };
    const auto stable=plan();
    assert(stable.valid && stable.no_change && stable.delta.empty());
    for (int i=0;i<4;++i) feed({900000,1000000,6200000,5700000});
    const auto shifted=plan();
    assert(shifted.valid && !shifted.no_change && !shifted.delta.empty());
    assert(r3::heat::delta_matches(shifted,in.current));
    // Every planner guard is checked at its actual unit/scale.
    assert(in.max_moves >= 2 && in.max_moved_bytes >= 2*stride);
    in.samples.assign(4,cfg.controls.min_samples-1);
    assert(r3::heat::propose(in,cfg.head_shares).no_change);
    in.samples=heat.samples;
    in.observed_at_ns=in.now_ns-cfg.controls.max_heat_age_ns-1;
    assert(plan().no_change);
    in.observed_at_ns=in.now_ns-delta.age_ns;
    in.cooldown_until_ns=in.now_ns+cfg.abort_cooldown_ns;
    assert(plan().no_change);
    in.cooldown_until_ns=0;
    in.last_move_ns.assign(4,in.now_ns-cfg.controls.min_residency_ns+1);
    assert(plan().no_change);
    in.last_move_ns.assign(4,0);
    in.max_moves=1; assert(plan().no_change);
    in.max_moves=cfg.controls.max_moves;
    in.max_moved_bytes=stride; assert(plan().no_change);
    in.max_moved_bytes=cfg.controls.max_moved_bytes;
    assert(!plan().no_change);
    // The same derived effect gate must reach READY and recognize a real
    // service improvement; former ns/E multiplier made the fractional
    // threshold greater than one and all improvements UNDECIDED.
    r3::policy::EffectGate gate({cfg.effect_window_samples,
                                 cfg.effect_min_quiet_windows,
                                 cfg.effect_noise_multiplier});
    r3::policy::ServiceSnapshot service{};
    const auto window=p.at("effect_window_samples").get<uint64_t>();
    const auto step=9521126761ull;
    gate.quiet(service,1);
    for (uint64_t i=1;i<=cfg.effect_min_quiet_windows+1;++i) {
        auto& lane=service.lane[1];
        lane.samples+=window; lane.rows+=window*10;
        lane.time_ns+=window*(6296371ull+(i%2)*10000);
        gate.quiet(service,1+i*step);
    }
    const uint64_t now=1+(cfg.effect_min_quiet_windows+1)*step;
    assert(gate.ready(now,cfg.controls.max_heat_age_ns));
    assert(gate.begin(now,cfg.controls.max_heat_age_ns));
    gate.committed(service);
    service.lane[1].samples+=window;
    service.lane[1].rows+=window*3; // post route mix changes: per-row rate would falsely report WORSE
    service.lane[1].time_ns+=window*5000000ull;
    assert(gate.judge(service)==r3::policy::EffectVerdict::KEPT);
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
    calib5_regression(*rec,cal,at);
    assert(p.size() == 17);
    assert(p.at("cadence_ns") == 3220000000ull);
    assert(p.at("max_moved_bytes") == 435200000ull);
    assert(p.at("max_moves") == 174);
    assert(p.at("min_residency_ns") == 83555555556ull);
    assert(p.at("max_heat_age_ns") == 8260000000ull);
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
    assert(p.at("abort_cooldown_ns") == 8260000000ull);
    assert(p.at("learned_eligibility_ttl_ns") == 9760000000ull);
    assert(p.at("effect_window_samples") == 32 && p.at("effect_min_quiet_windows") == 2);
    assert(p.at("half_life_observations_milli") == 59000000000ull);
    assert(p.at("effect_revert_ttl_ns") == 89495555556ull);
    assert(p.at("effect_noise_multiplier_milli") == 2576);
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
    assert(derived_small.policy.at("effect_noise_multiplier_milli") ==
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
    assert(stride.policy.at("max_moved_bytes") == 435200000 && stride.policy.at("max_moves") == 108);
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
                                 "effect_noise_floor_ns", "tail_share_noise_ppm"}) {
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
