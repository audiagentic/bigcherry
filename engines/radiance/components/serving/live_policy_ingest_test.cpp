#include "live_policy_ingest.h"
#include "live_policy.h"
#include <cassert>
#ifdef NDEBUG
#error "assert-based test built with NDEBUG: its assertions would not run"
#endif
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
                    const std::string& objective, const std::string& target) {
    try { sv::ingest_live_policy_record(log, cache, key, layout, objective, target); }
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
        {"heat_watermarked",0},
        {"hot_set_k",5931}, {"marginal_error_ppm",23000},
        {"marginal_observations",3}, {"drift_half_life_observations",59000000},
        {"observation_rate_per_s",10000000}, {"tail_share_noise_ppm",20000},
        {"tail_share_gain_ppm",45000}, {"tail_lane_service_ns_per_s",750000000},
        {"tail_lane_request_ns",22222222},
        {"lane_lost_ns",100000000}, {"effect_requests_per_s",64},
        {"effect_noise_max_ns",2500000},
        {"effect_sample_noise_ns",1550000}, {"abort_quiescent_ns",1500000000ull}
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
    assert(rec->measurements.at("heat_watermarked") == 0);
    const auto d = sv::derive_live_policy(*rec, sv::Calibration::standard(cache),
                                          {"", "flashnext-4gpu", "balanced"},
                                          2500000, {1000000000,1000000000});
    assert(d.policy.at("cadence_ns") == 3220000000ull);
    assert(d.policy.at("max_moves") == 64);
    assert(d.policy.at("min_residency_ns") == 32592592593ull);
    assert(d.policy.at("accept_unwatermarked_heat") == true);
    row["measurements"]["heat_watermarked"] = 1;
    log_line(log,row);
    const fs::path watermarked_store = dir / "watermarked.toml";
    sv::ingest_live_policy_record(log,watermarked_store,key,"flashnext-4gpu","balanced");
    auto with_mark = sv::read_live_policy_record(watermarked_store,key);
    assert(with_mark && with_mark->measurements.at("heat_watermarked") == 1);
    assert(!sv::derive_live_policy(*with_mark,sv::Calibration::standard(watermarked_store),
            {"","flashnext-4gpu","balanced"},2500000,{1000000000,1000000000})
            .policy.at("accept_unwatermarked_heat").get<bool>());
    row["measurements"]["heat_watermarked"] = 2;
    log_line(log,row);
    assert(refused(log,dir/"bad-mark.toml",key,"flashnext-4gpu","balanced"));
    row["measurements"]["heat_watermarked"] = 0;
    log_line(log,row);
    const std::string original = [&] { std::ifstream in(cache); return std::string(
        std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()); }();
    assert(refused(log,cache,key,"flashnext-4gpu","balanced","")); // no plan-store key: the helper key is never stored
    try {
        sv::ingest_live_policy_record(log,cache,key,"flashnext-4gpu","balanced",store_key);
        assert(false); // no duplicates at the production plan-store key
    } catch (const std::runtime_error&) { }
    assert(refused(log,cache,std::string(64,'b'),"flashnext-4gpu","balanced",store_key));
    assert(refused(log,cache,key,"other-layout","balanced",store_key));
    assert(refused(log,cache,key,"flashnext-4gpu","latency",store_key));
    assert([&] { std::ifstream in(cache); return std::string(
        std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()); }() == original);
    { std::ofstream out(log); out << "RHP16_CALIBRATION_JSON not-json\n"; }
    assert(refused(log,dir/"bad.toml",key,"flashnext-4gpu","balanced",store_key));
    // Missing a quantity is explicit, accepted and remains uncalibrated.
    row["plan_cache_key"] = std::string(64, 'c');
    row["measurements"].erase("copy_bytes_per_s");
    row["absent"]["copy_bytes_per_s"] = "no per-helper throughput hook";
    log_line(log, row);
    const fs::path partial_cache = dir / "partial.toml";
    sv::ingest_live_policy_record(log, partial_cache, std::string(64,'c'), "flashnext-4gpu", "balanced", std::string(64,'e'));
    const auto partial = sv::read_live_policy_record(partial_cache, std::string(64,'e'));
    assert(partial && !sv::live_policy_record_complete(*partial));
    assert(partial->absent.at("copy_bytes_per_s") == "no per-helper throughput hook");
    row["absent"].erase("copy_bytes_per_s");
    log_line(log, row);
    assert(refused(log,dir/"missing-reason.toml",std::string(64,'c'),"flashnext-4gpu","balanced",store_key));
    fs::remove_all(dir);
}
