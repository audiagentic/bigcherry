#include "plan_cache.h"
#include "serving_codec.h"
#include "sha256.h"
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
    CachedPlan c{decode_engine_plan(e.at("plan"), where), e.at("plan"), key, e.at("measured_at").get<std::string>()};
    return c;
}

CachedPlan plan_cache_store(const fs::path& dir, const Json& key_inputs, const Json& plan_json) {
    CachedPlan c{decode_engine_plan(plan_json, "measured plan"), plan_json, plan_key(key_inputs), {}};
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

Json measure_engine_plan(const fs::path& binary, const fs::path& manifest, uint32_t engine_card_ordinal,
                         const fs::path& out, double* wall_seconds) {
    if (!fs::is_regular_file(binary)) throw std::runtime_error("engine binary " + binary.string() + " is not a file");
    std::error_code ec;
    fs::remove(out, ec);
    // The child's environment: this one, less any device selection, plus the engine card alone.
    std::vector<std::string> env;
    for (char** e = environ; e && *e; ++e) {
        const std::string kv = *e;
        if (kv.rfind("ROCR_VISIBLE_DEVICES=", 0) == 0 || kv.rfind("HIP_VISIBLE_DEVICES=", 0) == 0 ||
            kv.rfind("CUDA_VISIBLE_DEVICES=", 0) == 0 || kv.rfind("GPU_DEVICE_ORDINAL=", 0) == 0)
            continue;
        env.push_back(kv);
    }
    env.push_back("ROCR_VISIBLE_DEVICES=" + std::to_string(engine_card_ordinal));
    env.push_back("HIP_VISIBLE_DEVICES=0");
    std::vector<std::string> args = {binary.string(), "--manifest", manifest.string(), "--role", "engine",
                                     "--plan-memory", out.string()};
    std::vector<char*> argv, envp;
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    for (auto& e : env) envp.push_back(e.data());
    envp.push_back(nullptr);
    const auto t0 = std::chrono::steady_clock::now();
    pid_t pid = 0;
    if (const int rc = ::posix_spawn(&pid, argv[0], nullptr, nullptr, argv.data(), envp.data()); rc != 0)
        throw std::runtime_error("cannot start " + binary.string() + ": " + std::strerror(rc));
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0)
        if (errno != EINTR) throw std::runtime_error("waiting for " + binary.string() + ": " + std::strerror(errno));
    if (wall_seconds)
        *wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error(binary.string() + " --plan-memory " +
                                 (WIFEXITED(status) ? "exited " + std::to_string(WEXITSTATUS(status))
                                                    : "was killed by signal " + std::to_string(WTERMSIG(status))) +
                                 "; its log says why");
    const Json plan = Json::parse(read_file(out), nullptr, false);
    if (plan.is_discarded()) throw std::runtime_error(out.string() + ": the engine's plan is not JSON");
    (void)decode_engine_plan(plan, out.string());
    return plan;
}

}  // namespace bc::serving
