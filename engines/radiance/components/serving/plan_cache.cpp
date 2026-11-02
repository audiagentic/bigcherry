#include "plan_cache.h"
#include "serving.h"

#include "rad_format.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace bc::serving {
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

// The memory outputs the plan itself feeds: never part of what decides whether a measurement applies.
const std::set<std::string> kDerivedEngineKeys = {"host_pool_mib", "prefix_cache_mib", "host_headroom_mib"};
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

Json plan_key_inputs(const Json& values, const Json& facts, const std::string& model_id, const std::string& engine_id) {
    const std::string card = values.at("layout").at("engine_card").get<std::string>();
    const Json* gpu = nullptr;
    for (const Json& g : facts.at("gpus"))
        if (g.at("bdf") == card) gpu = &g;
    if (!gpu) throw std::runtime_error("plan key: the engine card " + card + " is not among the detected GPUs");
    Json engine = Json::object();
    for (auto it = values.at("engine").begin(); it != values.at("engine").end(); ++it)
        if (!kDerivedEngineKeys.count(it.key())) engine[it.key()] = it.value();
    return {{"cache_version", 1},
            {"model", model_id},
            {"engine_build", engine_id},
            {"card", {{"arch", gpu->at("arch")}, {"vram_total_mib", gpu->at("vram_total_mib")}}},
            {"layout", values.at("layout").at("name")},
            {"engine", engine}};
}

std::string plan_key(const Json& key_inputs) { return sha256_hex(key_inputs.dump()); }

std::optional<CachedPlan> plan_cache_lookup(const fs::path& dir, const Json& key_inputs) {
    const std::string key = plan_key(key_inputs);
    const fs::path file = dir / (key + ".json");
    if (!fs::exists(file)) return std::nullopt;
    const Json e = Json::parse(read_file(file), nullptr, false);
    const std::string where = "plan cache entry " + file.string();
    if (e.is_discarded() || !e.is_object()) throw std::runtime_error(where + ": not a JSON object");
    for (const char* k : {"cache_version", "key", "key_inputs", "measured_at", "plan"})
        if (!e.contains(k)) throw std::runtime_error(where + ": missing " + std::string(k));
    if (e.size() != 5) throw std::runtime_error(where + ": unknown keys");
    if (e.at("cache_version") != 1) throw std::runtime_error(where + ": cache_version must be 1");
    if (e.at("key") != key || e.at("key_inputs") != key_inputs)
        throw std::runtime_error(where + ": recorded key or key inputs differ from what it is filed under");
    CachedPlan c{parse_engine_plan(e.at("plan")), e.at("plan"), key, e.at("measured_at").get<std::string>()};
    return c;
}

CachedPlan plan_cache_store(const fs::path& dir, const Json& key_inputs, const Json& plan_json) {
    CachedPlan c{parse_engine_plan(plan_json), plan_json, plan_key(key_inputs), {}};
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char at[32];
    std::strftime(at, sizeof at, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    c.measured_at = at;
    fs::create_directories(dir);
    const Json e = {{"cache_version", 1}, {"key", c.key}, {"key_inputs", key_inputs}, {"measured_at", c.measured_at},
                    {"plan", plan_json}};
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
