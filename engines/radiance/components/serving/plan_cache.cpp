#include "admission.h"
#include "calibration.h"
#include "plan_cache.h"
#include "file_stamp.h"
#include "rank_demand.h"
#include "serving_codec.h"
#include "sha256.h"
#include "strict_json.h"
#include "serving.h"

#include "rad_format.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <spawn.h>
#include <sys/wait.h>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <atomic>
#include <fstream>
#include <map>
#include <mutex>
#include <sys/stat.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <unistd.h>

extern "C" char** environ;

namespace bc::serving {
nlohmann::json parse_plan_json(const std::string& text, const std::string& source) {
    v2::Result<nlohmann::json> parsed = v2::parse_strict_json(text, source);
    if (!parsed.ok()) {
        const v2::ConfigError& e = parsed.error();
        throw std::runtime_error(source + ": " + (e.path.empty() ? std::string() : e.path + ": ") + e.message);
    }
    return std::move(parsed.value());
}

v2::EnginePlan decode_engine_plan(const nlohmann::json& plan, const std::string& where) {
    v2::EnginePlan p;
    if (const v2::Status e = v2::decode(plan, "", &p))
        throw std::runtime_error(where + ": engine plan" + (e->path.empty() ? std::string() : " " + e->path) + ": " +
                                 e->message);
    std::set<std::string> names;
    for (const v2::PlanHostConsumer& c : p.host)
        if (!names.insert(c.name).second) throw std::runtime_error(where + ": engine plan: duplicate consumer " + c.name);
    if (p.checkpoints.floor_slots > p.checkpoints.want_slots)
        throw std::runtime_error(where + ": engine plan.checkpoints: floor_slots exceeds want_slots");
    return p;
}

namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;

std::string read_range(std::ifstream& in, uint64_t off, uint64_t n, const fs::path& where) {
    std::string s(n, '\0');
    in.seekg(std::streamoff(off));
    if (n && !in.read(s.data(), std::streamsize(n)))
        throw std::runtime_error(where.string() + ": cannot read " + std::to_string(n) + " bytes at " + std::to_string(off));
    return s;
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + p.string());
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

// Hash runtime binaries without materialising them in the serving owner's
// heap. Repeated full-file string copies compete with live host admission and
// an ostream can silently return a truncated string if an allocation fails;
// a short/error read must refuse identity rather than invent a new digest.
std::string file_sha256_checked(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path.string());
    const uintmax_t expected = fs::file_size(path);
    Sha256 digest;
    std::array<char, 64 * 1024> chunk{};
    uintmax_t received = 0;
    while (in) {
        in.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const std::streamsize n = in.gcount();
        if (n > 0) {
            digest.update(std::string_view(chunk.data(), static_cast<size_t>(n)));
            received += static_cast<uintmax_t>(n);
        }
    }
    if (!in.eof() || received != expected)
        throw std::runtime_error("runtime identity: incomplete or changed file " + path.string());
    return digest.hex_digest();
}

// What is not part of what decides whether a measurement applies (spec section 5.1): the host KV tier and headroom (memory
// outputs the plan feeds, absent from the plan), and where the engine listens and lives (the port, the runtime-home and
// model paths: the engine build and model identities already cover their contents). The host pool stays in the key: the
// plan echoes it (EnginePlan.host_pool_mib) and the check demands it equal, so a plan measured under one pool must never be
// found under another.
const std::set<std::string> kNonPlanEngineKeys = {"prefix_cache_mib", "host_headroom_mib", "port", "radiance_home", "model"};

int plan_cache_version(const Json& key_inputs) {
    if (!key_inputs.is_object() || !key_inputs.contains("cache_version") ||
        !key_inputs.at("cache_version").is_number_integer())
        throw std::runtime_error("plan cache key: missing integer cache_version");
    const int version = key_inputs.at("cache_version").get<int>();
    if (version == 2) {
        if (key_inputs.contains("native_config"))
            throw std::runtime_error("plan cache key: native snapshot requires cache version 3");
        return version;
    }
    if (version == 3) {
        if (!key_inputs.contains("native_config") || !key_inputs.at("native_config").is_object() ||
            !key_inputs.at("native_config").contains("sha256") ||
            !key_inputs.at("native_config").at("sha256").is_string() ||
            key_inputs.at("native_config").at("sha256").get<std::string>().size() != 64 ||
            !key_inputs.at("native_config").contains("resolution_root") ||
            !key_inputs.at("native_config").at("resolution_root").is_string() ||
            !key_inputs.at("native_config").contains("rank_devices") ||
            !key_inputs.at("native_config").at("rank_devices").is_array() ||
            key_inputs.at("native_config").at("rank_devices").empty())
            throw std::runtime_error("plan cache key: native version 3 requires snapshot and ordered device identity");
        return version;
    }
    throw std::runtime_error("plan cache key: unsupported cache version " + std::to_string(version));
}

bool is_sha256(const Json& value) {
    if (!value.is_string()) return false;
    const std::string digest = value.get<std::string>();
    return digest.size() == 64 && std::all_of(digest.begin(), digest.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

std::string owner_measurement_digest(const Json& owner, const std::string& key) {
    const Json receipt = {{"attempt_id", owner.at("attempt_id")},
                          {"key", key},
                          {"measuring_manifest_sha256", owner.at("measuring_manifest_sha256")},
                          {"plan_sha256", owner.at("plan_sha256")},
                          {"rank_demand_sha256", owner.at("rank_demand_sha256")},
                          {"helpers_resident_mib", owner.at("helpers_resident_mib")},
                          {"engine_resident_mib", owner.at("engine_resident_mib")},
                          {"native_config_sha256", owner.at("native_config_sha256")},
                          {"resolution_root", owner.at("resolution_root")}};
    return sha256_hex(receipt.dump());
}

// Publish `bytes` as `file` all at once: a temporary beside it, written and synced, then renamed over it, so a reader sees
// the old file or the whole new one, never part of either.
void publish_file_atomic(const fs::path& file, const std::string& bytes) {
    fs::create_directories(file.parent_path());
    std::string pattern = file.string() + ".tmp.XXXXXX";
    std::vector<char> tmp_name(pattern.begin(), pattern.end());
    tmp_name.push_back('\0');
    const int fd = ::mkstemp(tmp_name.data());
    if (fd < 0) throw std::runtime_error("cannot create temporary file for " + file.string());
    const fs::path tmp(tmp_name.data());
    bool open = true;
    try {
        size_t offset = 0;
        while (offset < bytes.size()) {
            const ssize_t n = ::write(fd, bytes.data() + offset, bytes.size() - offset);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) throw std::runtime_error("cannot write " + tmp.string());
            offset += static_cast<size_t>(n);
        }
        if (::fsync(fd) != 0) throw std::runtime_error("cannot sync " + tmp.string());
        if (::close(fd) != 0) throw std::runtime_error("cannot close " + tmp.string());
        open = false;
        if (::rename(tmp.c_str(), file.c_str()) != 0)
            throw std::runtime_error("cannot publish " + file.string());
    } catch (...) {
        if (open) ::close(fd);
        std::error_code ignored;
        fs::remove(tmp, ignored);
        throw;
    }
}

void write_cache_entry(const fs::path& dir, const Json& key_inputs, const CachedPlan& cached, int version,
                       const Json* owner = nullptr) {
    Json entry = {{"cache_version", version}, {"key", cached.key}, {"key_inputs", key_inputs},
                  {"measured_at", cached.measured_at}, {"plan", cached.plan_json},
                  {"rank_demand", cached.rank_demand},
                  {"resident", {{"helpers_mib", cached.resident.helpers_mib},
                                 {"engine_mib", cached.resident.engine_mib}}}};
    if (owner) entry["owner"] = *owner;
    publish_file_atomic(dir / (cached.key + ".json"), entry.dump(2) + "\n");
}
}  // namespace

std::string container_identity(const fs::path& model) {
    std::ifstream in(model, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read the model container " + model.string());
    const uint64_t size = fs::file_size(model);
    if (size < sizeof(RadFileHeader)) throw std::runtime_error(model.string() + ": too small to be a .rad container");
    RadFileHeader h{};
    const std::string head = read_range(in, 0, sizeof h, model);
    std::memcpy(&h, head.data(), sizeof h);
    if (h.magic != RAD_MAGIC || h.version != RAD_FORMAT_VER)
        throw std::runtime_error(model.string() + ": not a .rad container of format " + std::to_string(RAD_FORMAT_VER));
    auto table = [&](uint64_t off, uint64_t count, size_t stride) {
        if (count > (size / std::max<size_t>(stride, 1)) || off > size || count * stride > size - off)
            throw std::runtime_error(model.string() + ": a table runs past the end of the file");
        return read_range(in, off, count * stride, model);
    };
    std::string text = std::to_string(size) + "\n" + head;
    text += table(h.dir_off, h.dir_count, sizeof(RadFileEntry));
    text += table(h.plane_off, h.plane_count, sizeof(RadFilePlane));
    text += table(h.enc_off, h.enc_count, sizeof(RadEncoding));
    return sha256_hex(text);
}

namespace {
struct IdentityMemo {
    std::vector<std::pair<fs::path, FileStamp>> stamps;
    std::string identity;
};
std::mutex identity_mutex;
std::map<std::string, IdentityMemo> identity_memo;
fs::path identity_store;  // the work root whose cache/engine-identity folder persists the digests; empty: none

// One small file per runtime home identity (binary + home), in the work root's cache beside the named-config store.
fs::path identity_file(const fs::path& work_root, const std::string& memo_key) {
    return work_root / "cache" / "engine-identity" / (sha256_hex(memo_key) + ".json");
}

Json stamps_json(const std::vector<std::pair<fs::path, FileStamp>>& stamps, const fs::path& home) {
    Json files = Json::array();
    for (size_t i = 0; i < stamps.size(); ++i) {
        const auto& [path, s] = stamps[i];
        // The binary (first) by its path as given, every runtime file by its path relative to the runtime home.
        files.push_back({i == 0 ? path.generic_string() : fs::relative(path, home).generic_string(), s.size, s.inode,
                         s.mtime_ns, s.ctime_ns});
    }
    return files;
}

// The persisted digest when the file is well formed, names this runtime home, and every stamp in it equals the stamp just
// taken; otherwise empty. Anything unreadable or unexpected is ignored, never an error: the runtime is hashed instead.
std::string persisted_identity(const fs::path& file, const std::string& memo_key,
                               const std::vector<std::pair<fs::path, FileStamp>>& stamps, const fs::path& home) {
    try {
        std::ifstream in(file, std::ios::binary);
        if (!in) return {};
        std::ostringstream bytes;
        bytes << in.rdbuf();
        const Json stored = Json::parse(bytes.str());
        if (stored.at("key").get<std::string>() != memo_key || stored.at("files") != stamps_json(stamps, home)) return {};
        const std::string identity = stored.at("identity").get<std::string>();
        return identity.size() == 64 && identity.find_first_not_of("0123456789abcdef") == std::string::npos ? identity
                                                                                                           : std::string();
    } catch (const std::exception&) {
        return {};
    }
}
std::atomic<uint64_t> identity_hash_runs{0};
}  // namespace

uint64_t engine_identity_hash_runs() { return identity_hash_runs.load(); }

void set_engine_identity_store(const fs::path& work_root) {
    const std::lock_guard<std::mutex> lock(identity_mutex);
    identity_store = work_root;
}

std::string engine_identity(const fs::path& binary, const fs::path& radiance_home) {
    if (!fs::is_regular_file(binary)) throw std::runtime_error("engine binary " + binary.string() + " is not a file");
    if (!fs::is_directory(radiance_home))
        throw std::runtime_error("engine runtime home " + radiance_home.string() + " is not a directory");
    std::vector<fs::path> files;
    for (const auto& e : fs::recursive_directory_iterator(radiance_home))
        if (e.is_regular_file()) files.push_back(fs::relative(e.path(), radiance_home));
    std::sort(files.begin(), files.end());
    // The stamps are taken before the bytes are read: a file that changes while it is hashed no longer matches its
    // stamp, so the next call hashes again rather than trusting a digest of a moving file.
    std::vector<std::pair<fs::path, FileStamp>> stamps;
    stamps.emplace_back(binary, stamp_file(binary));
    for (const auto& f : files) stamps.emplace_back(radiance_home / f, stamp_file(radiance_home / f));
    const std::string memo_key = binary.string() + "\n" + radiance_home.string();
    fs::path store;
    {
        const std::lock_guard<std::mutex> lock(identity_mutex);
        const auto found = identity_memo.find(memo_key);
        if (found != identity_memo.end() && found->second.stamps == stamps) return found->second.identity;
        store = identity_store;
    }
    const fs::path file = store.empty() ? fs::path{} : identity_file(store, memo_key);
    if (!file.empty()) {
        std::string stored = persisted_identity(file, memo_key, stamps, radiance_home);
        if (!stored.empty()) {
            const std::lock_guard<std::mutex> lock(identity_mutex);
            identity_memo[memo_key] = IdentityMemo{std::move(stamps), stored};
            return stored;
        }
    }
    ++identity_hash_runs;
    std::string text = "binary\n" + file_sha256_checked(binary) + "\n";
    for (const auto& f : files) text += f.generic_string() + "\n" + file_sha256_checked(radiance_home / f) + "\n";
    std::string identity = sha256_hex(text);
    if (!file.empty()) {
        // A digest that cannot be stored costs the next start one more hash; it never fails this one.
        try {
            publish_file_atomic(file, Json{{"key", memo_key}, {"identity", identity}, {"files", stamps_json(stamps, radiance_home)}}.dump() + "\n");
        } catch (const std::exception&) {
        }
    }
    const std::lock_guard<std::mutex> lock(identity_mutex);
    identity_memo[memo_key] = IdentityMemo{std::move(stamps), identity};
    return identity;
}

Json plan_key_inputs(const Json& values, const Json& facts, const std::string& model_id, const std::string& engine_id,
                     const std::string& engine_args, const NativeConfigSnapshot* native_config) {
    // When every rank is the same card type the first stands for them, and the rank count is part of the key through
    // engine.tp. A world of more than one card type (a layout with several engine groups) keys every rank's card.
    Json rank_cards = Json::array();
    const Json* gpu = nullptr;
    for (const Json& bdf : values.at("layout").at("engine_cards")) {
        const Json* found = nullptr;
        for (const Json& g : facts.at("gpus"))
            if (g.at("bdf") == bdf) found = &g;
        if (!found) throw std::runtime_error("plan key: the engine card " + bdf.get<std::string>() + " is not among the detected GPUs");
        if (!gpu) gpu = found;
        rank_cards.push_back({{"arch", found->at("arch")}, {"vram_total_mib", found->at("vram_total_mib")}});
    }
    if (!gpu) throw std::runtime_error("plan key: no engine card");
    const bool uniform = std::all_of(rank_cards.begin(), rank_cards.end(), [&](const Json& c) { return c == rank_cards.at(0); });
    Json engine = Json::object();
    for (auto it = values.at("engine").begin(); it != values.at("engine").end(); ++it)
        if (!kNonPlanEngineKeys.count(it.key()) &&
            // v3 YAML digest pins the authored policy. An omitted native auto
            // placement resolves from the engine's measured rank receipt,
            // not a new cache identity or an authored BigCherry option.
            !(native_config && it.key() == "ngram_placement"))
            engine[it.key()] = it.value();
    Json key = {{"cache_version", native_config ? 3 : 2},
            {"model", model_id},
            {"engine_build", engine_id},
            {"card", {{"arch", gpu->at("arch")}, {"vram_total_mib", gpu->at("vram_total_mib")}}},
            {"layout", values.at("layout").at("name")},
            // The helpers' budgets decide which experts the engine itself holds, and so the VRAM floor the plan measures
            // (a plan measured with a 14 GiB tail must never be found for a 7 GiB one).
            {"helper_budgets_mib", values.contains("memory") && values.at("memory").contains("helper_budgets_mib")
                                       ? values.at("memory").at("helper_budgets_mib") : Json::array()},
            {"engine_args", engine_args},
            {"engine", engine}};
    if (native_config) {
        Json devices = Json::array();
        for (const char* group : {"engine_cards", "helper_cards"}) {
            const Json& selected = values.at("layout").at(group);
            for (size_t rank = 0; rank < selected.size(); ++rank) {
                const Json& bdf = selected.at(rank);
                const Json* found = nullptr;
                for (const Json& candidate : facts.at("gpus"))
                    if (candidate.at("bdf") == bdf) found = &candidate;
                if (!found)
                    throw std::runtime_error("plan key: selected " + std::string(group) + " device " +
                                             bdf.get<std::string>() + " is not in the inventory");
                devices.push_back({{"role_group", group}, {"rank", rank}, {"bdf", bdf},
                                   {"hip_ordinal", found->at("hip_ordinal")},
                                   {"arch", found->at("arch")},
                                   {"vram_total_mib", found->at("vram_total_mib")}});
            }
        }
        key["native_config"] = {{"sha256", native_config->sha256()},
                                 {"resolution_root", native_config->resolution_root()},
                                 {"rank_devices", std::move(devices)}};
        if (!values.at("layout").at("helper_cards").empty()) {
            const fs::path home = values.at("engine").at("radiance_home").get<std::string>();
            key["native_config"]["helper_build"] = file_sha256_checked(home / "helpers" / "r3_moe_worker");
        }
    }
    if (!uniform) key["rank_cards"] = rank_cards;
    // The tuning settings bind to engine flags, and several change what the plan measures (tp_ownership moves each rank's
    // KV and weight shares; state_cache_dtype, embedding_placement, draft_model, ... move memory too), so all of them are
    // inputs. Added only when present, so a configuration without tuning keeps its key. A native start binds the
    // YAML digest (above), which already covers every tuning option the YAML states.
    if (!native_config && values.contains("tuning") && values.at("tuning").is_object() && !values.at("tuning").empty())
        key["tuning"] = values.at("tuning");
    return key;
}

void check_plan_for_configuration(const Json& values, const v2::EnginePlan& plan) {
    const Json& engine = values.at("engine");
    if (plan.host_pool_mib != engine.at("host_pool_mib").get<int64_t>())
        throw std::runtime_error("the plan was measured with a " + std::to_string(plan.host_pool_mib) +
                                 " MiB host pool, this configuration's is " +
                                 std::to_string(engine.at("host_pool_mib").get<int64_t>()));
    // The expected host consumer list: the n-gram table is a pinned consumer exactly when the table lives in RAM.
    const bool has_table = std::any_of(plan.host.begin(), plan.host.end(),
                                       [](const v2::PlanHostConsumer& c) { return c.name == "ngram_table"; });
    const std::string placement = engine.at("ngram_placement").get<std::string>();
    if (placement == "auto") {
        // Unmeasured native YAML requests the engine's real tier selection.
        // The strict rank receipt supplies its resolved choice after P1.
    } else if (placement == "ram" && !has_table)
        throw std::runtime_error("this configuration keeps the n-gram table in RAM but the plan has no ngram_table consumer");
    if (placement != "ram" && placement != "auto" && has_table)
        throw std::runtime_error("this configuration places the n-gram table on " + placement +
                                 " but the plan holds an ngram_table consumer");
    for (const v2::PlanHostConsumer& c : plan.host)
        if (c.name != "ngram_table" && c.name != "embedding" && c.name != "host_arenas")
            throw std::runtime_error("the plan holds an unknown host consumer " + c.name);
}

std::string gpu_inventory_digest(const Json& gpus) {
    static const char* const kIdentity[] = {"arch",        "bdf",           "drm_render_minor", "gfx_target_version",
                                            "hip_ordinal", "kfd_node",      "max_link_speed",   "max_link_width",
                                            "numa_node",   "simd_count",    "vram_total_mib"};
    Json identity = Json::array();
    for (const Json& g : gpus) {
        Json one = Json::object();
        for (const char* key : kIdentity)
            if (g.contains(key)) one[key] = g.at(key);
        identity.push_back(std::move(one));
    }
    return sha256_hex(identity.dump());
}

std::string plan_key(const Json& key_inputs) { return sha256_hex(key_inputs.dump()); }
// A policy measurement describes this hardware/service scope, not the engine's VRAM plan. Its
// transaction, lane-service, heat-drift and noise readings can be reused with a different engine
// binary, helper VRAM budget, host pool, checkpoint geometry or engine arguments. The policy
// derivation recomputes its byte cap against the *current* expert stride and helper budgets.
Json policy_calibration_inputs(const Json& spec) {
    const Json& gpufacts = spec.at("facts").at("gpus");
    const auto card = [&](const Json& bdf) {
        for (const Json& gpu : gpufacts)
            if (gpu.at("bdf") == bdf)
                return Json{{"bdf", bdf}, {"arch", gpu.at("arch")},
                            {"vram_total_mib", gpu.at("vram").at("total_mib")}};
        throw std::runtime_error("policy calibration key: GPU " + bdf.get<std::string>() +
                                 " is missing from detected facts");
    };
    Json engines = Json::array(), helpers = Json::array();
    for (const Json& bdf : spec.at("processes").at("engine").at("device").at("gpus"))
        engines.push_back(card(bdf));
    for (const Json& h : spec.at("processes").at("helpers")) {
        Json selected = card(h.at("gpu"));
        selected["tier"] = h.at("tier");
        selected["formats"] = h.at("formats");
        selected["rows_kernel"] = h.at("rows_kernel");
        helpers.push_back(std::move(selected));
    }
    // Transport shared-memory name, telemetry file, static heat file, helper budget, CLI
    // paths, measured memory allocations and engine build are *not* calibrated physics.
    const Json& transport = spec.at("handoff").at("transport");
    Json service = {{"max_rows", transport.at("max_rows")},
                    {"small_rows", transport.at("small_rows")},
                    {"gemv_large_rows", transport.at("gemv_large_rows")},
                    {"skip_owned", transport.at("skip_owned")},
                    {"moe_down_tile", spec.at("handoff").at("kernels").at("moe_down_tile")}};
    return {{"policy_key_version", 2},
            {"model", spec.at("plan").at("engine_memory").at("model_identity")},
            {"layout", spec.at("intent").at("layout")},
            {"objective", spec.at("intent").at("service").at("objective")},
            {"power_profile", spec.at("intent").at("power").at("profile")},
            {"engine_cards", engines}, {"helper_cards", helpers},
            {"histogram_period_s", spec.at("handoff").at("telemetry").at("period_s")},
            {"helper_runtime", spec.at("processes").at("helper_common")},
            {"remote_service", service}};
}
Json policy_calibration_key_fields() {
    return Json::array({"engine_cards", "helper_cards", "helper_runtime", "histogram_period_s",
                        "layout", "model", "objective", "policy_key_version",
                        "power_profile", "remote_service"});
}
void require_declared_policy_calibration_inputs(const Json& inputs) {
    const Json fields = policy_calibration_key_fields();
    if (!inputs.is_object() || inputs.size() != fields.size())
        throw std::runtime_error("policy calibration declaration: undeclared or absent key inputs");
    for (const Json& field : fields)
        if (!inputs.contains(field.get<std::string>()))
            throw std::runtime_error("policy calibration declaration: missing " + field.get<std::string>());
    if (inputs.at("policy_key_version") != 2)
        throw std::runtime_error("policy calibration declaration: key version must be 2");
}
std::string policy_calibration_key(const Json& inputs) {
    if (!inputs.is_object() || inputs.value("policy_key_version", 0) != 2)
        throw std::runtime_error("policy calibration key: not version 2 inputs");
    return sha256_hex(inputs.dump());
}

std::optional<CachedPlan> plan_cache_lookup(const fs::path& dir, const Json& key_inputs) {
    const int expected_version = plan_cache_version(key_inputs);
    const std::string key = plan_key(key_inputs);
    const fs::path file = dir / (key + ".json");
    if (!fs::exists(file)) return std::nullopt;
    const std::string where = "plan cache entry " + file.string();
    const Json e = parse_plan_json(read_file(file), where);
    if (!e.is_object()) throw std::runtime_error(where + ": not a JSON object");
    // An entry from before the resident host memory was measured (RSC08 S3) has no "resident": it is a miss, so the
    // configuration is measured again and the entry rewritten, never planned with a guessed figure.
    if (!e.contains("resident")) {
        if (expected_version == 3)
            throw std::runtime_error(where + ": native cache entry is missing measured resident memory");
        return std::nullopt;
    }
    for (const char* k : {"cache_version", "key", "key_inputs", "measured_at", "plan", "rank_demand", "resident"})
        if (!e.contains(k)) throw std::runtime_error(where + ": missing " + std::string(k));
    if (e.at("cache_version") != expected_version)
        throw std::runtime_error(where + ": cache_version differs from key inputs");
    const bool has_owner = e.contains("owner");
    if (e.size() != (has_owner ? 8U : 7U) || (expected_version == 3 && !has_owner))
        throw std::runtime_error(where + ": unknown keys");
    if (has_owner) {
        if (!e.at("owner").is_object() || e.at("owner").size() != 11 ||
            e.at("owner").value("kind", std::string()) != "bc_serve_owned_measurement" ||
            e.at("owner").value("version", 0) != 1 ||
            !e.at("owner").contains("attempt_id") || !e.at("owner").at("attempt_id").is_string() ||
            e.at("owner").at("attempt_id").get<std::string>().empty() ||
            !is_sha256(e.at("owner").value("measuring_manifest_sha256", Json())) ||
            !is_sha256(e.at("owner").value("plan_sha256", Json())) ||
            !is_sha256(e.at("owner").value("rank_demand_sha256", Json())) ||
            !e.at("owner").contains("helpers_resident_mib") ||
            !e.at("owner").at("helpers_resident_mib").is_number_unsigned() ||
            !e.at("owner").contains("engine_resident_mib") ||
            !e.at("owner").at("engine_resident_mib").is_number_unsigned() ||
            !e.at("owner").contains("native_config_sha256") ||
            !e.at("owner").contains("resolution_root"))
            throw std::runtime_error(where + ": invalid owner measurement provenance");
        const bool native = expected_version == 3;
        if ((native && (!is_sha256(e.at("owner").at("native_config_sha256")) ||
                        e.at("owner").at("native_config_sha256") != key_inputs.at("native_config").at("sha256") ||
                        e.at("owner").at("resolution_root") != key_inputs.at("native_config").at("resolution_root"))) ||
            (!native && (!e.at("owner").at("native_config_sha256").is_null() ||
                         !e.at("owner").at("resolution_root").is_null())) ||
            e.at("owner").at("plan_sha256") != sha256_hex(e.at("plan").dump()) ||
            e.at("owner").at("rank_demand_sha256") != sha256_hex(e.at("rank_demand").dump()) ||
            e.at("owner").at("helpers_resident_mib") != e.at("resident").at("helpers_mib") ||
            e.at("owner").at("engine_resident_mib") != e.at("resident").at("engine_mib") ||
            !is_sha256(e.at("owner").value("measurement_sha256", Json())) ||
            e.at("owner").at("measurement_sha256") != owner_measurement_digest(e.at("owner"), key))
            throw std::runtime_error(where + ": owner measurement digest mismatch");
    }
    if (!e.at("rank_demand").is_object() || e.at("rank_demand").value("kind", "") != "radiance_p1_rank_demand")
        throw std::runtime_error(where + ": rank_demand must be the engine's --plan-memory-ranks record");
    const Json& res = e.at("resident");
    if (!res.is_object() || res.size() != 2 || !res.contains("helpers_mib") || !res.contains("engine_mib") ||
        !res.at("helpers_mib").is_number_unsigned() || !res.at("engine_mib").is_number_unsigned() ||
        res.at("engine_mib").get<uint64_t>() == 0)
        throw std::runtime_error(where + ": resident must be {helpers_mib, engine_mib}: the engine's positive, the helpers' zero only for an engine-only layout");
    if (e.at("key") != key || e.at("key_inputs") != key_inputs)
        throw std::runtime_error(where + ": recorded key or key inputs differ from what it is filed under");
    CachedPlan c{decode_engine_plan(e.at("plan"), where), e.at("plan"), e.at("rank_demand"), key,
                 e.at("measured_at").get<std::string>(),
                 {res.at("helpers_mib").get<uint64_t>(), res.at("engine_mib").get<uint64_t>()}};
    if (has_owner) {
        c.owner = OwnerMeasurementEvidence{e.at("owner").at("attempt_id").get<std::string>(),
                                           e.at("owner").at("measuring_manifest_sha256").get<std::string>()};
        c.owner_sha256 = e.at("owner").at("measurement_sha256").get<std::string>();
    }
    return c;
}

CachedPlan plan_cache_store(const fs::path& dir, const Json& key_inputs, const Json& plan_json, const Json& rank_demand,
                            const HostResident& resident) {
    const int version = plan_cache_version(key_inputs);
    if (version == 3)
        throw std::runtime_error("native version-3 plans require owner measurement provenance");
    if (resident.engine_mib == 0)
        throw std::runtime_error("the resident host memory of the helpers and of the engine must both be measured (positive MiB)");
    if (!rank_demand.is_object() || rank_demand.value("kind", "") != "radiance_p1_rank_demand")
        throw std::runtime_error("the measured plan needs the engine's --plan-memory-ranks record with it");
    CachedPlan c{decode_engine_plan(plan_json, "measured plan"), plan_json, rank_demand, plan_key(key_inputs), {}, resident};
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char at[32];
    std::strftime(at, sizeof at, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    c.measured_at = at;
    write_cache_entry(dir, key_inputs, c, version);
    return c;
}

CachedPlan plan_cache_store_owned(const fs::path& dir, const Json& key_inputs, const Json& plan_json,
                                 const Json& rank_demand, const Json& values, const HostResident& resident,
                                 const OwnerMeasurementEvidence& owner) {
    const int version = plan_cache_version(key_inputs);
    if (resident.engine_mib == 0 || owner.attempt_id.empty() || !is_sha256(owner.measuring_manifest_sha256))
        throw std::runtime_error("owner measurement publication requires an attempt, manifest digest, and resident memory");
    const std::vector<std::string> selected_bdfs =
        values.at("layout").at("engine_cards").get<std::vector<std::string>>();
    const EngineRankDemand decoded_ranks =
        decode_engine_rank_demand(rank_demand, selected_bdfs, "owner rank demand");
    std::vector<std::string> engine_bdfs;
    for (const EngineRankGpuDemand& gpu : decoded_ranks.gpu) engine_bdfs.push_back(gpu.bdf);
    if (engine_bdfs.empty() || rank_demand.at("gpu").size() != engine_bdfs.size())
        throw std::runtime_error("owner rank demand does not match the engine-device count");
    if (version == 3) {
        std::vector<std::string> native_engine_bdfs;
        for (const Json& device : key_inputs.at("native_config").at("rank_devices"))
            if (device.is_object() && device.value("role_group", std::string()) == "engine_cards")
                native_engine_bdfs.push_back(device.at("bdf").get<std::string>());
        if (native_engine_bdfs != engine_bdfs)
            throw std::runtime_error("owner rank demand differs from the native ordered BDFs");
    }
    const v2::EnginePlan decoded = decode_engine_plan(plan_json, "owned measured plan");
    check_plan_for_configuration(values, decoded);
    if (decoded.vram.cards.size() != engine_bdfs.size())
        throw std::runtime_error("owner engine plan rank count differs from native ordered BDFs");
    for (size_t i = 0; i < decoded_ranks.gpu.size(); ++i)
        if (decoded.vram.cards[i].free != decoded_ranks.gpu[i].claimable_bytes)
            throw std::runtime_error(
                "owner rank claimable VRAM differs from the engine plan's same-rank free VRAM facts");
    if (owner.measuring_manifest_sha256.empty())
        throw std::runtime_error("owner measurement has no measuring-manifest digest");
    CachedPlan c{decoded, plan_json, rank_demand, plan_key(key_inputs), {}, resident};
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char at[32];
    std::strftime(at, sizeof at, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    c.measured_at = at;
    Json provenance = {
        {"kind", "bc_serve_owned_measurement"}, {"version", 1}, {"attempt_id", owner.attempt_id},
        {"measuring_manifest_sha256", owner.measuring_manifest_sha256},
        {"plan_sha256", sha256_hex(plan_json.dump())},
        {"rank_demand_sha256", sha256_hex(rank_demand.dump())},
        {"helpers_resident_mib", resident.helpers_mib}, {"engine_resident_mib", resident.engine_mib},
        {"native_config_sha256", version == 3 ? key_inputs.at("native_config").at("sha256") : Json(nullptr)},
        {"resolution_root", version == 3 ? key_inputs.at("native_config").at("resolution_root") : Json(nullptr)}};
    c.owner_sha256 = owner_measurement_digest(provenance, c.key);
    provenance["measurement_sha256"] = c.owner_sha256;
    c.owner = owner;
    write_cache_entry(dir, key_inputs, c, version, &provenance);
    return c;
}

}  // namespace bc::serving
