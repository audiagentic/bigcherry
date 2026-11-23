#include "live_policy_ingest.h"
#include "live_policy.h"
#include "plan_cache.h"
#include "sha256.h"
#include "third_party/toml.hpp"
#include "v2/manifest.h"

#include <nlohmann/json.hpp>
#include <cerrno>
#include <fcntl.h>
#include <fstream>
#include <optional>
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
    "txn_p50_ns", "txn_p95_ns", "copy_bytes_per_s", "heat_gap_max_ns", "heat_watermarked",
    "hot_set_k", "marginal_error_ppm", "marginal_observations",
    "drift_half_life_observations", "observation_rate_per_s",
    "tail_share_gain_ppm", "tail_lane_service_ns_per_s",
    "tail_lane_row_ns", "lane_lost_ns", "effect_rows_per_s",
    "effect_noise_max_ns", "effect_row_noise_ns", "effect_window_noise_ladder", "abort_quiescent_ns"
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
        if(it.key()=="effect_window_noise_ladder") {
            const Json& ladder=it.value();
            if(!ladder.is_array() || ladder.size()<3)
                invalid("invalid measured window noise ladder");
            uint64_t previous=0;
            for(const auto& pair:ladder) {
                if(!pair.is_array() || pair.size()!=2 ||
                   !pair[0].is_number_unsigned() || !pair[1].is_number_unsigned() ||
                   pair[0].get<uint64_t>()<=previous ||
                   pair[0].get<uint64_t>()>kMaxExact || pair[1].get<uint64_t>()>kMaxExact)
                    invalid("invalid measured window noise ladder pair");
                previous=pair[0].get<uint64_t>();
            }
            continue;
        }
        if (!kMeasurements.count(it.key()) || !it.value().is_number_integer() ||
            (it.value().is_number_integer() && !it.value().is_number_unsigned() &&
             it.value().get<int64_t>() < 0) ||
            it.value().get<uint64_t>() > kMaxExact ||
            (it.value().get<uint64_t>() == 0 && it.key() != "lane_lost_ns" && it.key() != "heat_watermarked") ||
            (it.key() == "heat_watermarked" && it.value().get<uint64_t>() > 1))
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
// The store is written by this file, one [[live_policy_record]] block each: the text before the first block and the blocks.
struct StoreBlocks {
    std::string header;
    std::vector<std::string> blocks;
};
StoreBlocks split_store(const std::string& text) {
    static const std::string kBlock = "[[live_policy_record]]";
    std::vector<size_t> starts;
    for (size_t at = text.find(kBlock); at != std::string::npos; at = text.find(kBlock, at + 1))
        if (at == 0 || text[at - 1] == '\n') starts.push_back(at);
    StoreBlocks out;
    out.header = text.substr(0, starts.empty() ? text.size() : starts.front());
    for (size_t i = 0; i < starts.size(); ++i) {
        const size_t end = i + 1 < starts.size() ? starts[i + 1] : text.size();
        out.blocks.push_back(text.substr(starts[i], end - starts[i]));
    }
    return out;
}
const toml::table* block_row(const toml::table& parsed) {
    const auto* rows = parsed["live_policy_record"].as_array();
    return rows && rows->size() == 1 ? (*rows)[0].as_table() : nullptr;
}
toml::table parse_block(const std::string& block) {
    try { return toml::parse(block); }
    catch (const toml::parse_error& e) { invalid(std::string("unparseable live_policy_record block: ") + e.what()); }
}
// `text` without the block stored under `store_key` (the partial row a complete one replaces).
std::string without_record(const std::string& text, const std::string& store_key) {
    const auto split = split_store(text);
    std::string kept = split.header;
    for (const auto& block : split.blocks) {
        const toml::table parsed = parse_block(block);
        const auto* row = block_row(parsed);
        if (!row) invalid("unparseable live_policy_record block");
        if ((*row)["plan_cache_key"].value_or(std::string_view()) != store_key) kept += block;
    }
    return kept;
}
bool all_planner_measured(const Json& measurements) {
    LivePolicyRecord probe;
    probe.measurements = measurements;
    return live_policy_record_complete(probe);
}
} // namespace

LivePolicyTarget live_policy_target_from_manifest(const fs::path& manifest) {
    const auto loaded = v2::load_manifest(manifest, v2::PlanUse::serving);
    if (!loaded)
        invalid("strict v1 serving manifest rejected: " + v2::render_human("calibration-ingest", loaded.error()));
    const auto& m = loaded.value();
    // An engine-only manifest has no handoff, so it has no live policy to calibrate.
    if (!m.spec.handoff) invalid("manifest has no helper hand-off to calibrate");
    LivePolicyTarget target;
    target.plan_key = v2::plan_cache_key(m);
    target.layout = m.spec.intent.layout;
    target.objective = std::string(v2::kObjectiveNames[static_cast<size_t>(m.spec.intent.service.objective)]);
    if (target.plan_key.size() != 64) invalid("manifest has no measured plan-cache key");
    const Json raw = Json::parse(read_text(manifest));
    const Json policy_inputs = policy_calibration_inputs(raw.at("spec"));
    target.store_key = policy_calibration_key(policy_inputs);
    if (!m.spec.plan.engine_memory)
        invalid("manifest has no measured engine plan provenance");
    // Prefer the build's immutable source-slice identity when available.
    // Native release metadata is read-only; absence is recorded explicitly,
    // never replaced by a fabricated Git commit or an activation refusal.
    std::string revision = "unavailable";
    const fs::path home = raw.at("spec").at("processes").at("engine")
                             .at("runtime").at("radiance_home").get<std::string>();
    const fs::path build_metadata = home.parent_path() / "bigcherry-build-metadata-radiance.json";
    std::error_code ec;
    // Provenance enrichment must never turn a valid calibration observation
    // into an ingest failure just because optional build metadata is absent
    // or corrupt; the explicit "unavailable" value remains auditable.
    try {
        if (fs::is_regular_file(build_metadata, ec) &&
            fs::file_size(build_metadata, ec) < (1u << 20)) {
            const Json metadata = Json::parse(read_text(build_metadata), nullptr, false);
            if (metadata.is_object() && metadata.contains("source_slice_id") &&
                metadata.at("source_slice_id").is_string()) {
                const std::string slice = metadata.at("source_slice_id").get<std::string>();
                if (std::regex_match(slice, std::regex("[0-9a-f]{32}")))
                    revision = "source_slice:" + slice;
            }
        }
    } catch (const std::exception&) {
        revision = "unavailable";
    }
    require_declared_policy_calibration_inputs(policy_inputs);
    target.provenance = LivePolicyProvenance{policy_inputs, target.plan_key,
                                             m.spec.plan.engine_memory->engine_identity,
                                             revision,
                                             raw.at("metadata").at("instance_id").get<std::string>(),
                                             sha256_hex(raw.at("spec").at("facts").dump())};
    return target;
}

void ingest_live_policy_log(const fs::path& log, const fs::path& cache,
                            const fs::path& manifest,
                             const std::string& target_store_key) {
    if (log.empty() || cache.empty() || manifest.empty())
        invalid("--from-log, --calibration and --manifest are all required");
    // The lab calibration run: lab profile, an explicit live policy, and no lab experiment, so nothing but the policy
    // separates it from the production start. These checks belong to this lab entry only; the owner's own running
    // manifest is ingested through live_policy_target_from_manifest without them.
    {
        const auto loaded = v2::load_manifest(manifest, v2::PlanUse::serving);
        if (!loaded)
            invalid("strict v1 serving manifest rejected: " + v2::render_human("calibration-ingest", loaded.error()));
        const auto& m = loaded.value();
        if (m.spec.intent.service.profile != v2::Profile::kLab || m.spec.experiment ||
            !m.spec.handoff || !m.spec.handoff->placement.live_policy)
            invalid("a calibration-run manifest (lab profile, explicit live policy, no experiment) is required");
    }
    const LivePolicyTarget target = live_policy_target_from_manifest(manifest);
    if (target_store_key.size() != 64 || target_store_key.find_first_not_of("0123456789abcdef") != std::string::npos)
        invalid("missing or invalid validated plan-store key");
    if (target_store_key != target.store_key)
        invalid("target policy key differs from the verified measured manifest scope");
    ingest_live_policy_record(log, cache, target.plan_key, target.layout, target.objective, target_store_key,
                              target.provenance);
}

void ingest_live_policy_record(const fs::path& log, const fs::path& cache,
                               const std::string& key, const std::string& layout,
                                const std::string& objective,
                               const std::string& target_store_key,
                               const LivePolicyProvenance& provenance) {
    if (log.empty() || cache.empty()) invalid("log and calibration cache are required");
    std::istringstream lines(read_text(log));
    std::string line;
    std::optional<std::string> payload;
    while (std::getline(lines, line)) {
        if (line.rfind(kMarker, 0) != 0) continue;
        if (payload) invalid("more than one calibration record in helper log");
        payload = line.substr(std::char_traits<char>::length(kMarker));
    }
    if (!payload) invalid("no settled calibration record in helper log");
    ingest_live_policy_marker(*payload, cache, key, layout, objective, target_store_key, provenance);
}

void ingest_live_policy_marker(const std::string& payload, const fs::path& cache,
                               const std::string& key, const std::string& layout,
                               const std::string& objective,
                               const std::string& target_store_key,
                               const LivePolicyProvenance& provenance) {
    if (cache.empty()) invalid("calibration cache is required");
    Json record = Json::parse(payload, nullptr, false);
    if (record.is_discarded()) invalid("malformed JSON record");
    check_record(record, key, layout, objective);
    // Store by the independently scoped policy key, never the helper's legacy plan key.
    const std::string& store_key = target_store_key;
    if (store_key.size() != 64 || store_key.find_first_not_of("0123456789abcdef") != std::string::npos)
        invalid("invalid target policy-calibration key");
    // Validate the helper's measured plan key first, then store the observation
    // under the policy scope. The original full plan/build identities survive
    // separately in provenance and never affect lookup.
    record["plan_cache_key"] = store_key;
    require_declared_policy_calibration_inputs(provenance.inputs);
    if (policy_calibration_key(provenance.inputs) != store_key)
        invalid("policy scope digest differs from requested store key");
    if (provenance.source_revision.empty() || provenance.run_id.empty() ||
        provenance.host_facts_digest.size() != 64 ||
        provenance.host_facts_digest.find_first_not_of("0123456789abcdef") != std::string::npos)
        invalid("missing or invalid source revision, run or host-facts provenance");
    for (const std::string& hex : {provenance.source_plan_key, provenance.source_engine_build})
        if (hex.size() != 64 || hex.find_first_not_of("0123456789abcdef") != std::string::npos)
            invalid("missing or invalid source plan/build provenance");
    record["calibration_inputs"] = provenance.inputs.dump();
    record["source_plan_key"] = provenance.source_plan_key;
    record["source_engine_build"] = provenance.source_engine_build;
    record["record_schema_version"] = 1;
    record["calibration_kind"] = "radiance.live_policy";
    record["calibration_key_fields"] = policy_calibration_key_fields();
    record["source_revision"] = provenance.source_revision;
    record["run_id"] = provenance.run_id;
    record["host_facts_digest"] = provenance.host_facts_digest;
    if (!cache.parent_path().empty() && !fs::exists(cache.parent_path()))
        invalid("calibration cache parent directory does not exist");
    Lock lock(cache.string() + ".lock");
    std::string current;
    if (fs::exists(cache)) {
        current = read_text(cache);
        // Fail closed on any malformed or duplicate existing record. Only a partial row is ever replaced, and only by
        // a complete one: a complete row is never replaced, and a partial row never displaces anything.
        if (const auto existing = read_live_policy_record(cache, store_key)) {
            if (live_policy_record_complete(*existing))
                invalid("plan key already calibrated (refuse duplicate)");
            if (!all_planner_measured(record.at("measurements")))
                invalid("plan key already holds a partial record (refuse duplicate)");
            current = without_record(current, store_key);
        }
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
            {
                out << "record_schema_version = 1\n";
                out << "calibration_kind = " << toml_string(record.at("calibration_kind")) << '\n';
                out << "calibration_key_fields = " << record.at("calibration_key_fields").dump() << '\n';
                for (const auto& field : {"calibration_inputs", "source_plan_key", "source_engine_build",
                                          "source_revision", "run_id", "host_facts_digest"})
                    out << field << " = " << toml_string(record.at(field)) << '\n';
            }
            for (auto it = record.at("measurements").begin();
                 it != record.at("measurements").end(); ++it)
                out << it.key() << " = " << it.value().dump() << '\n';
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
        // Make directory entry durable as well as the synced TOML content.
        const fs::path parent = cache.parent_path().empty() ? fs::path(".") : cache.parent_path();
        const int directory = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (directory < 0 || ::fsync(directory) != 0) {
            if (directory >= 0) ::close(directory);
            invalid("cannot sync calibration cache directory");
        }
        ::close(directory);
    } catch (...) {
        std::error_code ec;
        fs::remove(temp, ec);
        throw;
    }
}
std::vector<std::string> prune_live_policy_store(const fs::path& cache, bool apply) {
    if (cache.empty() || !fs::is_regular_file(cache) || fs::is_symlink(cache))
        invalid("missing or unsafe calibration store " + cache.string());
    Lock lock(cache.string() + ".lock");
    const std::string text = read_text(cache);
    const auto split = split_store(text);
    std::string kept = split.header;
    std::vector<std::string> removed;
    for (const auto& block : split.blocks) {
        const toml::table parsed = parse_block(block);
        const auto* row = block_row(parsed);
        if (!row) invalid("unparseable live_policy_record block");
        if (row->contains("calibration_inputs") && row->contains("record_schema_version")) kept += block;
        else removed.push_back(std::string((*row)["plan_cache_key"].value_or(std::string_view("?"))));
    }
    if (!apply || removed.empty()) return removed;
    const fs::path temp = cache.string() + ".tmp." + std::to_string(::getpid());
    try {
        {
            std::ofstream out(temp, std::ios::binary | std::ios::trunc);
            if (!out) invalid("cannot create temporary calibration file");
            out << kept;
            out.flush();
            if (!out) invalid("cannot flush temporary calibration file");
        }
        // The pruned file must read back through the strict reader (any key, the read validates every row).
        (void)read_live_policy_record(temp, std::string(64, '0'));
        const int fd = ::open(temp.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0 || ::fsync(fd) != 0) {
            if (fd >= 0) ::close(fd);
            invalid("cannot sync calibration file");
        }
        ::close(fd);
        fs::permissions(temp, fs::status(cache).permissions());
        fs::rename(temp, cache);
        const fs::path parent = cache.parent_path().empty() ? fs::path(".") : cache.parent_path();
        const int directory = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (directory < 0 || ::fsync(directory) != 0) {
            if (directory >= 0) ::close(directory);
            invalid("cannot sync calibration cache directory");
        }
        ::close(directory);
    } catch (...) {
        std::error_code ec;
        fs::remove(temp, ec);
        throw;
    }
    return removed;
}
} // namespace bc::serving
