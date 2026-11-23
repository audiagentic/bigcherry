// The owner's calibration watcher against a fake helper log: no helper, no GPU, no manifest. The record rows are the
// same strict shape the helper prints; provenance is supplied as the owner's running manifest would.
#include "calibration_watch.h"
#include "live_policy.h"
#include "plan_cache.h"
#include <cassert>
#ifdef NDEBUG
#error "assert-based test built with NDEBUG: its assertions would not run"
#endif
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;
namespace sv = bc::serving;
using Json = nlohmann::json;

static const Json kScope = {{"policy_key_version", 2}, {"model", "test-model"},
                            {"layout", "flashnext-4gpu"}, {"objective", "balanced"},
                            {"power_profile", "stock"}, {"engine_cards", Json::array()},
                            {"helper_cards", Json::array()}, {"histogram_period_s", 1},
                            {"helper_runtime", Json::object()}, {"remote_service", Json::object()}};

static sv::LivePolicyTarget target() {
    sv::LivePolicyTarget t;
    t.plan_key = std::string(64, 'a');
    t.layout = "flashnext-4gpu";
    t.objective = "balanced";
    t.store_key = sv::policy_calibration_key(kScope);
    t.provenance = sv::LivePolicyProvenance{kScope, std::string(64, '1'), std::string(64, '2'),
                                            "revision-test", "run-abc", std::string(64, '3')};
    return t;
}
static Json measurements() {
    return {{"txn_p50_ns",1770000000ull}, {"txn_p95_ns",2720000000ull}, {"copy_bytes_per_s",160000000ull},
            {"heat_gap_max_ns",5040000000ull}, {"heat_watermarked",0}, {"hot_set_k",5931},
            {"marginal_error_ppm",23000}, {"marginal_observations",3}, {"drift_half_life_observations",59000000},
            {"observation_rate_per_s",10000000}, {"tail_share_gain_ppm",45000},
            {"tail_lane_service_ns_per_s",750000000}, {"tail_lane_row_ns",22222222}, {"lane_lost_ns",100000000},
            {"effect_rows_per_s",64}, {"effect_noise_max_ns",2500000}, {"effect_row_noise_ns",1550000},
            {"effect_window_noise_ladder", Json::array({Json::array({uint64_t(8),uint64_t(80000)}),
                Json::array({uint64_t(16),uint64_t(40000)}), Json::array({uint64_t(32),uint64_t(16000)})})},
            {"abort_quiescent_ns",1500000000ull}};
}
static Json row(bool partial) {
    Json m = measurements(), absent = Json::object();
    if (partial) {
        m.erase("copy_bytes_per_s");
        absent["copy_bytes_per_s"] = "no throughput hook";
    }
    return {{"plan_cache_key", std::string(64, 'a')}, {"layout", "flashnext-4gpu"}, {"objective", "balanced"},
            {"seeded", "2026-11-22"}, {"evidence", "fake helper log"}, {"measurements", m}, {"absent", absent}};
}
static void append(const fs::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::app | std::ios::binary);
    out << text;
}
static std::string read(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}
static std::string marker(const Json& r) { return "RHP16_CALIBRATION_JSON " + r.dump() + "\n"; }

int main() {
    const fs::path dir = fs::temp_directory_path() / ("bc-calibration-watch-" + std::to_string(::getpid()));
    fs::create_directories(dir);
    const std::string key = target().store_key;
    {   // Waits for the log, ignores noise, and decides only on a complete (newline-terminated) marker line.
        const fs::path log = dir / "a.log", store = dir / "a.toml";
        sv::CalibrationWatch w(log, store, target());
        assert(!w.poll() && w.state() == sv::CalibrationWatch::State::waiting);
        append(log, "helper: started\ncalibration waiting: tail_lane_row_ns\n");
        assert(!w.poll());
        const std::string line = marker(row(false));
        append(log, line.substr(0, line.size() / 2));
        assert(!w.poll() && !fs::exists(store));
        append(log, line.substr(line.size() / 2));
        assert(w.poll() && w.state() == sv::CalibrationWatch::State::stored);
        const auto rec = sv::read_live_policy_record(store, key);
        assert(rec && sv::live_policy_record_complete(*rec));
        assert(rec->run_id == "run-abc" && rec->source_plan_key == std::string(64, '1'));
        assert(rec->calibration_inputs && *rec->calibration_inputs == kScope);
        // Decided once: later polls, and a second marker, change nothing.
        const std::string before = read(store);
        append(log, marker(row(false)));
        assert(!w.poll() && read(store) == before);
    }
    {   // A partial record is kept as diagnostic evidence and never replaces a complete one; a complete one replaces it.
        const fs::path log1 = dir / "b1.log", log2 = dir / "b2.log", log3 = dir / "b3.log", store = dir / "b.toml";
        append(log1, marker(row(true)));
        sv::CalibrationWatch w1(log1, store, target());
        assert(w1.poll() && w1.state() == sv::CalibrationWatch::State::stored);
        auto rec = sv::read_live_policy_record(store, key);
        assert(rec && !sv::live_policy_record_complete(*rec));
        append(log2, marker(row(true)));   // partial against partial: settled, store untouched
        const std::string partial_store = read(store);
        sv::CalibrationWatch w2(log2, store, target());
        assert(w2.poll() && w2.state() == sv::CalibrationWatch::State::settled_elsewhere && read(store) == partial_store);
        append(log3, marker(row(false)));  // complete replaces the partial row, exactly one row remains
        sv::CalibrationWatch w3(log3, store, target());
        assert(w3.poll() && w3.state() == sv::CalibrationWatch::State::stored);
        rec = sv::read_live_policy_record(store, key);
        assert(rec && sv::live_policy_record_complete(*rec));
        const std::string complete_store = read(store);
        assert(complete_store.find("[[live_policy_record]]") == complete_store.rfind("[[live_policy_record]]"));
        const fs::path log4 = dir / "b4.log";
        append(log4, marker(row(true)));   // partial never replaces complete
        sv::CalibrationWatch w4(log4, store, target());
        assert(w4.poll() && w4.state() == sv::CalibrationWatch::State::settled_elsewhere && read(store) == complete_store);
    }
    {   // A store that cannot be written is a reported refusal, never an exception out of poll().
        const fs::path log = dir / "c.log";
        append(log, marker(row(false)));
        sv::CalibrationWatch w(log, dir / "missing-parent" / "c.toml", target());
        assert(w.poll() && w.state() == sv::CalibrationWatch::State::refused && !w.detail().empty());
        assert(!w.poll());
    }
    {   // A malformed or foreign record is refused and reported; the store is not created.
        const fs::path log = dir / "d.log", store = dir / "d.toml";
        append(log, "RHP16_CALIBRATION_JSON not-json\n");
        sv::CalibrationWatch w(log, store, target());
        assert(w.poll() && w.state() == sv::CalibrationWatch::State::refused && !fs::exists(store));
        const fs::path log2 = dir / "d2.log";
        Json other = row(false);
        other["layout"] = "other-layout";
        append(log2, marker(other));
        sv::CalibrationWatch w2(log2, store, target());
        assert(w2.poll() && w2.state() == sv::CalibrationWatch::State::refused && !fs::exists(store));
    }
    {   // The target comes from one strict manifest with no lab-run requirement: the contract's production example
        // (a production profile, no experiment) yields the same identity the owner stores under.
        const sv::LivePolicyTarget t = sv::live_policy_target_from_manifest(BC_EXAMPLE_MANIFEST);
        assert(t.plan_key.size() == 64 && t.layout == "flashnext-4gpu" && !t.objective.empty());
        assert(t.store_key == sv::policy_calibration_key(t.provenance.inputs));
        assert(t.provenance.source_plan_key == t.plan_key && !t.provenance.run_id.empty());
        const fs::path log = dir / "e.log", store = dir / "e.toml";
        Json r = row(false);
        r["plan_cache_key"] = t.plan_key;
        append(log, marker(r));
        sv::CalibrationWatch w(log, store, t);
        assert(w.poll() && w.state() == sv::CalibrationWatch::State::stored);
        const auto rec = sv::read_live_policy_record(store, t.store_key);
        assert(rec && sv::live_policy_record_complete(*rec) && rec->run_id == t.provenance.run_id);
    }
    fs::remove_all(dir);
}
