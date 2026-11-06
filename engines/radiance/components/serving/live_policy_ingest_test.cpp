#include "live_policy_ingest.h"
#include "live_policy.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;
namespace sv = bc::serving;
using Json = nlohmann::json;

static void log_line(const fs::path& path, const Json& row) {
    std::ofstream out(path);
    assert(out);
    out << "helper: started\nRHP16_CALIBRATION_JSON " << row.dump() << "\n";
}
static bool refused(const fs::path& log, const fs::path& cache,
                    const std::string& key, const std::string& layout,
                    const std::string& objective) {
    try { sv::ingest_live_policy_record(log, cache, key, layout, objective); }
    catch (const std::runtime_error&) { return true; }
    return false;
}
int main() {
    const fs::path dir = fs::temp_directory_path() / ("bc-rhp16-ingest-" + std::to_string(::getpid()));
    fs::create_directories(dir);
    const fs::path log = dir / "worker.log", cache = dir / "calibration.toml";
    const std::string key(64, 'a'), store_key(64, 'd');
    // The exact policy-live-9 golden inputs: run them through the serializer,
    // reader and original derivation, not a second implementation.
    Json m = {
        {"txn_p50_ns",1770000000ull}, {"txn_p95_ns",2720000000ull},
        {"copy_bytes_per_s",160000000ull}, {"heat_gap_max_ns",5040000000ull},
        {"hot_set_k",5931}, {"marginal_error_ppm",23000},
        {"marginal_observations",3}, {"drift_half_life_observations",59000000},
        {"observation_rate_per_s",10000000}, {"tail_share_noise_ppm",20000},
        {"tail_share_gain_ppm",45000}, {"tail_lane_service_ns_per_s",750000000},
        {"lane_lost_ns",100000000}, {"effect_window_samples",32},
        {"effect_requests_per_s",64}, {"effect_quiet_windows",3},
        {"effect_noise_floor_ns",1000000}, {"effect_noise_quantile_ns",2500000},
        {"noise_false_revert_rate_ppm",10000}, {"abort_quiescent_ns",1500000000ull}
    };
    Json row = {{"plan_cache_key",key}, {"layout","flashnext-4gpu"},
                {"objective","balanced"}, {"seeded","2026-11-05"},
                {"evidence","policy-live-9 measured plus synthetic effect/lanes/drift"},
                {"measurements",m}, {"absent",Json::object()}};
    log_line(log, row);
    // The helper sees the typed v1 key, while production derives from the
    // RSC03 plan-store key. Ingestion maps only after source validation.
    sv::ingest_live_policy_record(log, cache, key, "flashnext-4gpu", "balanced", store_key);
    const auto rec = sv::read_live_policy_record(cache, store_key);
    assert(!sv::read_live_policy_record(cache, key));
    assert(rec && sv::live_policy_record_complete(*rec));
    const auto d = sv::derive_live_policy(*rec, sv::Calibration::standard(cache),
                                          {"", "flashnext-4gpu", "balanced"},
                                          2500000, {1000000000,1000000000});
    assert(d.policy.at("cadence_ns") == 3220000000ull);
    assert(d.policy.at("max_moves") == 64);
    assert(d.policy.at("min_residency_ns") == 32592592593ull);
    const std::string original = [&] { std::ifstream in(cache); return std::string(
        std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()); }();
    assert(refused(log,cache,key,"flashnext-4gpu","balanced")); // different key namespace
    try {
        sv::ingest_live_policy_record(log,cache,key,"flashnext-4gpu","balanced",store_key);
        assert(false); // no duplicates at the production plan-store key
    } catch (const std::runtime_error&) { }
    assert(refused(log,cache,std::string(64,'b'),"flashnext-4gpu","balanced"));
    assert(refused(log,cache,key,"other-layout","balanced"));
    assert(refused(log,cache,key,"flashnext-4gpu","latency"));
    assert([&] { std::ifstream in(cache); return std::string(
        std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()); }() == original);
    { std::ofstream out(log); out << "RHP16_CALIBRATION_JSON not-json\n"; }
    assert(refused(log,dir/"bad.toml",key,"flashnext-4gpu","balanced"));
    // Missing a quantity is explicit, accepted and remains uncalibrated.
    row["plan_cache_key"] = std::string(64, 'c');
    row["measurements"].erase("copy_bytes_per_s");
    row["absent"]["copy_bytes_per_s"] = "no per-helper throughput hook";
    log_line(log, row);
    const fs::path partial_cache = dir / "partial.toml";
    sv::ingest_live_policy_record(log, partial_cache, std::string(64,'c'), "flashnext-4gpu", "balanced");
    const auto partial = sv::read_live_policy_record(partial_cache, std::string(64,'c'));
    assert(partial && !sv::live_policy_record_complete(*partial));
    assert(partial->absent.at("copy_bytes_per_s") == "no per-helper throughput hook");
    row["absent"].erase("copy_bytes_per_s");
    log_line(log, row);
    assert(refused(log,dir/"missing-reason.toml",std::string(64,'c'),"flashnext-4gpu","balanced"));
    fs::remove_all(dir);
}
