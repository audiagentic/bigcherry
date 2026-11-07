#include "admission.h"
#include "calibration.h"
#include "plan_cache.h"
#include "serving_codec.h"
#include "sha256.h"
#include "strict_json.h"
#include "serving.h"

#include "rad_format.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <spawn.h>
#include <sys/wait.h>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

extern "C" char** environ;

namespace bc::serving {
nlohmann::json parse_plan_json(const std::string& text, const std::string& source) {
    v1::Result<nlohmann::json> parsed = v1::parse_strict_json(text, source);
    if (!parsed.ok()) {
        const v1::ConfigError& e = parsed.error();
        throw std::runtime_error(source + ": " + (e.path.empty() ? std::string() : e.path + ": ") + e.message);
    }
    return std::move(parsed.value());
}

v1::EnginePlan decode_engine_plan(const nlohmann::json& plan, const std::string& where) {
    v1::EnginePlan p;
    if (const v1::Status e = v1::decode(plan, "", &p))
        throw std::runtime_error(where + ": engine plan" + (e->path.empty() ? std::string() : " " + e->path) + ": " +
                                 e->message);
    std::set<std::string> names;
    for (const v1::PlanHostConsumer& c : p.host)
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

// What is not part of what decides whether a measurement applies (spec section 5.1): the host KV tier and headroom (memory
// outputs the plan feeds, absent from the plan), and where the engine listens and lives (the port, the runtime-home and
// model paths: the engine build and model identities already cover their contents). The host pool stays in the key: the
// plan echoes it (EnginePlan.host_pool_mib) and the check demands it equal, so a plan measured under one pool must never be
// found under another.
const std::set<std::string> kNonPlanEngineKeys = {"prefix_cache_mib", "host_headroom_mib", "port", "radiance_home", "model"};
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

std::string engine_identity(const fs::path& binary, const fs::path& radiance_home) {
    if (!fs::is_regular_file(binary)) throw std::runtime_error("engine binary " + binary.string() + " is not a file");
    if (!fs::is_directory(radiance_home))
        throw std::runtime_error("engine runtime home " + radiance_home.string() + " is not a directory");
    std::vector<fs::path> files;
    for (const auto& e : fs::recursive_directory_iterator(radiance_home))
        if (e.is_regular_file()) files.push_back(fs::relative(e.path(), radiance_home));
    std::sort(files.begin(), files.end());
    std::string text = "binary\n" + sha256_hex(read_file(binary)) + "\n";
    for (const auto& f : files) text += f.generic_string() + "\n" + sha256_hex(read_file(radiance_home / f)) + "\n";
    return sha256_hex(text);
}

Json plan_key_inputs(const Json& values, const Json& facts, const std::string& model_id, const std::string& engine_id,
                     const std::string& engine_args) {
    // The ranks of one engine are one architecture (layouts.toml names an arch and a count); the first stands for them, and the
    // rank count is part of the key through engine.tp.
    const std::string card = values.at("layout").at("engine_cards").at(0).get<std::string>();
    const Json* gpu = nullptr;
    for (const Json& g : facts.at("gpus"))
        if (g.at("bdf") == card) gpu = &g;
    if (!gpu) throw std::runtime_error("plan key: the engine card " + card + " is not among the detected GPUs");
    Json engine = Json::object();
    for (auto it = values.at("engine").begin(); it != values.at("engine").end(); ++it)
        if (!kNonPlanEngineKeys.count(it.key())) engine[it.key()] = it.value();
    return {{"cache_version", 1},
            {"model", model_id},
            {"engine_build", engine_id},
            {"card", {{"arch", gpu->at("arch")}, {"vram_total_mib", gpu->at("vram_total_mib")}}},
            {"layout", values.at("layout").at("name")},
            {"engine_args", engine_args},
            {"engine", engine}};
}

void check_plan_for_configuration(const Json& values, const v1::EnginePlan& plan) {
    const Json& engine = values.at("engine");
    if (plan.host_pool_mib != engine.at("host_pool_mib").get<int64_t>())
        throw std::runtime_error("the plan was measured with a " + std::to_string(plan.host_pool_mib) +
                                 " MiB host pool, this configuration's is " +
                                 std::to_string(engine.at("host_pool_mib").get<int64_t>()));
    // The expected host consumer list: the n-gram table is a pinned consumer exactly when the table lives in RAM.
    const bool has_table = std::any_of(plan.host.begin(), plan.host.end(),
                                       [](const v1::PlanHostConsumer& c) { return c.name == "ngram_table"; });
    const std::string placement = engine.at("ngram_placement").get<std::string>();
    if (placement == "ram" && !has_table)
        throw std::runtime_error("this configuration keeps the n-gram table in RAM but the plan has no ngram_table consumer");
    if (placement != "ram" && has_table)
        throw std::runtime_error("this configuration places the n-gram table on " + placement +
                                 " but the plan holds an ngram_table consumer");
    for (const v1::PlanHostConsumer& c : plan.host)
        if (c.name != "ngram_table" && c.name != "embedding" && c.name != "host_arenas")
            throw std::runtime_error("the plan holds an unknown host consumer " + c.name);
}

std::string plan_key(const Json& key_inputs) { return sha256_hex(key_inputs.dump()); }
std::string calibration_key(const Json& key_inputs) {
    if (!key_inputs.contains("engine_build")) throw std::runtime_error("calibration key: plan key inputs without engine_build");
    Json inputs = key_inputs;
    inputs.erase("engine_build");
    inputs.erase("cache_version");
    inputs["calibration_key_version"] = 1;
    return sha256_hex(inputs.dump());
}

std::optional<CachedPlan> plan_cache_lookup(const fs::path& dir, const Json& key_inputs) {
    const std::string key = plan_key(key_inputs);
    const fs::path file = dir / (key + ".json");
    if (!fs::exists(file)) return std::nullopt;
    const std::string where = "plan cache entry " + file.string();
    const Json e = parse_plan_json(read_file(file), where);
    if (!e.is_object()) throw std::runtime_error(where + ": not a JSON object");
    // An entry from before the resident host memory was measured (RSC08 S3) has no "resident": it is a miss, so the
    // configuration is measured again and the entry rewritten, never planned with a guessed figure.
    if (!e.contains("resident")) return std::nullopt;
    for (const char* k : {"cache_version", "key", "key_inputs", "measured_at", "plan", "resident"})
        if (!e.contains(k)) throw std::runtime_error(where + ": missing " + std::string(k));
    if (e.size() != 6) throw std::runtime_error(where + ": unknown keys");
    const Json& res = e.at("resident");
    if (!res.is_object() || res.size() != 2 || !res.contains("helpers_mib") || !res.contains("engine_mib") ||
        !res.at("helpers_mib").is_number_unsigned() || !res.at("engine_mib").is_number_unsigned() ||
        res.at("engine_mib").get<uint64_t>() == 0)
        throw std::runtime_error(where + ": resident must be {helpers_mib, engine_mib}: the engine's positive, the helpers' zero only for an engine-only layout");
    if (e.at("cache_version") != 1) throw std::runtime_error(where + ": cache_version must be 1");
    if (e.at("key") != key || e.at("key_inputs") != key_inputs)
        throw std::runtime_error(where + ": recorded key or key inputs differ from what it is filed under");
    CachedPlan c{decode_engine_plan(e.at("plan"), where), e.at("plan"), key, e.at("measured_at").get<std::string>(),
                 {res.at("helpers_mib").get<uint64_t>(), res.at("engine_mib").get<uint64_t>()}};
    return c;
}

CachedPlan plan_cache_store(const fs::path& dir, const Json& key_inputs, const Json& plan_json, const HostResident& resident) {
    if (resident.engine_mib == 0)
        throw std::runtime_error("the resident host memory of the helpers and of the engine must both be measured (positive MiB)");
    CachedPlan c{decode_engine_plan(plan_json, "measured plan"), plan_json, plan_key(key_inputs), {}, resident};
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char at[32];
    std::strftime(at, sizeof at, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    c.measured_at = at;
    fs::create_directories(dir);
    const Json e = {{"cache_version", 1}, {"key", c.key}, {"key_inputs", key_inputs}, {"measured_at", c.measured_at},
                    {"plan", plan_json}, {"resident", {{"helpers_mib", resident.helpers_mib}, {"engine_mib", resident.engine_mib}}}};
    const fs::path file = dir / (c.key + ".json"), tmp = dir / (c.key + ".json.tmp");
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << e.dump(2) << '\n';
        if (!out) throw std::runtime_error("cannot write " + tmp.string());
    }
    fs::rename(tmp, file);
    return c;
}

}  // namespace bc::serving
