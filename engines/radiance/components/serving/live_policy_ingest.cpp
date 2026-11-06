#include "live_policy_ingest.h"
#include "live_policy.h"
#include "v1/manifest.h"

#include <nlohmann/json.hpp>
#include <cerrno>
#include <fcntl.h>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/file.h>
#include <unistd.h>

namespace bc::serving {
namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
constexpr const char* kMarker = "RHP16_CALIBRATION_JSON ";
constexpr uint64_t kMaxExact = (uint64_t(1) << 53) - 1;
const std::set<std::string> kMeasurements = {
    "txn_p50_ns", "txn_p95_ns", "copy_bytes_per_s", "heat_gap_max_ns",
    "hot_set_k", "marginal_error_ppm", "marginal_observations",
    "drift_half_life_observations", "observation_rate_per_s",
    "tail_share_noise_ppm", "tail_share_gain_ppm", "tail_lane_service_ns_per_s",
    "lane_lost_ns", "effect_window_samples", "effect_requests_per_s",
    "effect_quiet_windows", "effect_noise_floor_ns", "effect_noise_quantile_ns",
    "noise_false_revert_rate_ppm", "abort_quiescent_ns"
};
[[noreturn]] void invalid(const std::string& message) {
    throw std::runtime_error("calibration-ingest: " + message);
}
std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) invalid("cannot read " + path.string());
    std::ostringstream out;
    out << in.rdbuf();
    if (!in.eof() && in.fail()) invalid("read failed: " + path.string());
    return out.str();
}
void check_record(const Json& row, const std::string& key,
                  const std::string& layout, const std::string& objective) {
    if (!row.is_object() || row.size() != 7)
        invalid("record must contain exactly plan_cache_key, layout, objective, seeded, evidence, measurements, absent");
    for (const auto& name : {"plan_cache_key", "layout", "objective", "seeded", "evidence"})
        if (!row.contains(name) || !row.at(name).is_string() ||
            row.at(name).get<std::string>().empty())
            invalid(std::string("missing or invalid ") + name);
    if (row.at("plan_cache_key") != key || row.at("layout") != layout ||
        row.at("objective") != objective)
        invalid("plan key/layout/objective mismatch with the verified manifest");
    const std::string date = row.at("seeded").get<std::string>();
    if (!std::regex_match(date, std::regex("[0-9]{4}-[0-9]{2}-[0-9]{2}")))
        invalid("seeded must be a YYYY-MM-DD UTC date");
    if (!row.at("measurements").is_object() || !row.at("absent").is_object())
        invalid("measurements and absent must be objects");
    for (auto it = row.at("measurements").begin(); it != row.at("measurements").end(); ++it) {
        if (!kMeasurements.count(it.key()) || !it.value().is_number_integer() ||
            (it.value().is_number_integer() && !it.value().is_number_unsigned() &&
             it.value().get<int64_t>() < 0) ||
            it.value().get<uint64_t>() > kMaxExact ||
            (it.value().get<uint64_t>() == 0 && it.key() != "lane_lost_ns"))
            invalid("invalid measured quantity: " + it.key());
    }
    for (auto it = row.at("absent").begin(); it != row.at("absent").end(); ++it) {
        if (!kMeasurements.count(it.key()) || !it.value().is_string() ||
            it.value().get<std::string>().empty() ||
            row.at("measurements").contains(it.key()))
            invalid("invalid or contradictory absent reason: " + it.key());
    }
    for (const auto& field : kMeasurements)
        if (!row.at("measurements").contains(field) && !row.at("absent").contains(field))
            invalid("quantity omitted without an absent reason: " + field);
}
std::string toml_string(const Json& value) {
    // JSON string escapes are accepted by TOML basic strings.
    return value.dump();
}
struct Lock {
    int fd = -1;
    explicit Lock(const fs::path& path) {
        fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (fd < 0) invalid("cannot create cache lock " + path.string());
        if (::flock(fd, LOCK_EX) != 0) {
            ::close(fd);
            invalid("cannot lock calibration cache");
        }
    }
    ~Lock() { if (fd >= 0) { ::flock(fd, LOCK_UN); ::close(fd); } }
};
} // namespace

void ingest_live_policy_log(const fs::path& log, const fs::path& cache,
                            const fs::path& manifest,
                             const std::string& target_store_key) {
    if (log.empty() || cache.empty() || manifest.empty())
        invalid("--from-log, --calibration and --manifest are all required");
    const auto loaded = v1::load_manifest(manifest, v1::PlanUse::serving);
    if (!loaded)
        invalid("strict v1 serving manifest rejected: " + v1::render_human("calibration-ingest", loaded.error()));
    const auto& m = loaded.value();
    // A calibration run (bc-serve --calibration-run) measures the production plan: lab profile, an explicit live
    // policy, and no lab experiment, so nothing but the policy separates it from the production start.
    // Amendment 12: an engine-only manifest has no handoff, so it has no live policy to calibrate.
    if (m.spec.intent.service.profile != v1::Profile::kLab || m.spec.experiment ||
        !m.spec.handoff || !m.spec.handoff->placement.live_policy)
        invalid("a calibration-run manifest (lab profile, explicit live policy, no experiment) is required");
    const std::string key = v1::plan_cache_key(m);
    const std::string layout = m.spec.intent.layout;
    const std::string objective(v1::kObjectiveNames[static_cast<size_t>(m.spec.intent.service.objective)]);
    if (key.size() != 64) invalid("manifest has no measured plan-cache key");
    if (target_store_key.size() != 64 || target_store_key.find_first_not_of("0123456789abcdef") != std::string::npos)
        invalid("missing or invalid validated plan-store key");
    ingest_live_policy_record(log, cache, key, layout, objective, target_store_key);
}

void ingest_live_policy_record(const fs::path& log, const fs::path& cache,
                               const std::string& key, const std::string& layout,
                                const std::string& objective,
                               const std::string& target_store_key) {
    if (log.empty() || cache.empty()) invalid("log and calibration cache are required");
    std::istringstream lines(read_text(log));
    std::string line;
    Json record;
    bool found = false;
    while (std::getline(lines, line)) {
        if (line.rfind(kMarker, 0) != 0) continue;
        if (found) invalid("more than one calibration record in helper log");
        found = true;
        record = Json::parse(line.substr(std::char_traits<char>::length(kMarker)), nullptr, false);
        if (record.is_discarded()) invalid("malformed JSON record");
    }
    if (!found) invalid("no settled calibration record in helper log");
    check_record(record, key, layout, objective);
    // Production reads records only at the RSC03 plan-store key; the helper's v1 key alone is never a store key.
    const std::string& store_key = target_store_key;
    if (store_key.size() != 64 || store_key.find_first_not_of("0123456789abcdef") != std::string::npos)
        invalid("invalid target plan-store key");
    // Only after validating the helper's authentic, manifest-backed key is
    // the record mapped to the distinct RSC03 plan-cache identity that the
    // serving manifest writer uses for calibration lookup.
    record["plan_cache_key"] = store_key;
    if (!cache.parent_path().empty() && !fs::exists(cache.parent_path()))
        invalid("calibration cache parent directory does not exist");
    Lock lock(cache.string() + ".lock");
    std::string current;
    if (fs::exists(cache)) {
        current = read_text(cache);
        // Fail closed on any malformed or duplicate existing record.
        if (read_live_policy_record(cache, store_key))
            invalid("plan key already calibrated (refuse duplicate)");
    }
    fs::path temp = cache.string() + ".tmp." + std::to_string(::getpid());
    try {
        {
            std::ofstream out(temp, std::ios::binary | std::ios::trunc);
            if (!out) invalid("cannot create temporary calibration file");
            if (current.empty()) out << "schema_version = 1\n";
            out << current;
            if (!current.empty() && current.back() != '\n') out << '\n';
            out << "\n[[live_policy_record]]\n";
            for (const auto& field : {"plan_cache_key", "layout", "objective", "seeded", "evidence"})
                out << field << " = " << toml_string(record.at(field)) << '\n';
            for (auto it = record.at("measurements").begin();
                 it != record.at("measurements").end(); ++it)
                out << it.key() << " = " << it.value().get<uint64_t>() << '\n';
            out << "absent = {";
            bool first = true;
            for (auto it = record.at("absent").begin(); it != record.at("absent").end(); ++it) {
                if (!first) out << ", ";
                first = false;
                out << it.key() << " = " << toml_string(it.value());
            }
            out << "}\n";
            out.flush();
            if (!out) invalid("cannot flush temporary calibration file");
        }
        // Validate the *whole* file using the same reader the next serving start uses.
        const auto ingested = read_live_policy_record(temp, store_key);
        if (!ingested || ingested->layout != layout || ingested->objective != objective)
            invalid("read-back of ingested record failed");
        const int fd = ::open(temp.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0 || ::fsync(fd) != 0) {
            if (fd >= 0) ::close(fd);
            invalid("cannot sync calibration file");
        }
        ::close(fd);
        fs::rename(temp, cache); // single atomic replacement, never a partial TOML
    } catch (...) {
        std::error_code ec;
        fs::remove(temp, ec);
        throw;
    }
}
} // namespace bc::serving
