#include "serving.h"
#include "vendor/toml.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <sys/statvfs.h>
#include <thread>
#include <utility>

namespace bc::serving {
namespace {
namespace fs = std::filesystem;

struct Field {
    std::string key, kind, origin, category, lifecycle;
    Json fallback;
    uint64_t lo = 0, hi = std::numeric_limits<uint32_t>::max();
    std::vector<std::string> choices;
    bool derived = false;
};

// Defaults are values from the 2026-11-02 launcher or WorkerConfig, not guesses
// about hardware. RSC03 will replace interim memory and step-size baselines
// with resource/geometry derivations. The explicit source of every constant is
// exported as x-origin in the JSON Schema.
const std::vector<Field>& schema_fields() {
    static const std::vector<Field> all = [] {
        std::vector<Field> f;
        auto add = [&](const char* k, const char* kind, Json v, const char* source,
                       const char* category = "policy", const char* life = "startup-only",
                       uint64_t lo = 0, uint64_t hi = UINT32_MAX,
                       std::vector<std::string> choices = {}, bool derived = false) {
            f.push_back({k, kind, source, category, life, std::move(v), lo, hi,
                         std::move(choices), derived});
        };
        add("schema_version", "integer", 1, "RSC01: first versioned serving contract",
            "policy", "startup-only", 1, 1);
        add("service.objective", "string", "balanced",
            "RSC01: neutral policy for existing mixed prefill/decode launcher", "policy",
            "startup-only", 0, 0, {"throughput", "latency", "balanced"});
        add("service.max_sessions", "integer", 8,
            "launcher SEQS=8; model and session policy", "policy", "startup-only", 1);
        add("service.max_context", "integer", 262144,
            "launcher MAX_LEN=262144; trained model context", "policy", "startup-only", 1);
        add("service.retention", "string", "keep",
            "launcher: finished-session prefix cache retention", "policy", "live",
            0, 0, {"keep", "evict"});
        add("service.quality", "string", "baseline",
            "launcher: baseline fp8 KV policy", "policy", "startup-only",
            0, 0, {"baseline", "lossless"});
        add("layout.name", "string", "flashnext-4gpu",
            "launcher LAYOUT default; named layouts in engines/radiance/layouts.toml");
        add("layout.engine_card", "string", nullptr,
            "RSC05: selected PCI BDF from KFD GPU nodes", "discovered");
        add("layout.helper_cards", "string_array", nullptr,
            "RSC05: selected PCI BDFs from KFD GPU nodes", "discovered", "step-boundary");
        add("memory.host_pool_mib", "integer", 9216,
            "launcher HOST_POOL_MIB=9216 measured interim; derive RSC03", "derived",
            "startup-only", 0, UINT32_MAX);
        add("memory.host_kv_mib", "integer", 12288,
            "launcher HOST_KV_MIB=12288 measured interim; derive RSC03", "derived",
            "startup-only", 0, UINT32_MAX);
        add("memory.tail_budget_mib", "integer", nullptr,
            "layouts.toml tail budget; derive RSC03", "derived", "step-boundary",
            0, UINT32_MAX, {}, true);
        add("memory.helper_budgets_mib", "integer_array", nullptr,
            "layouts.toml per-helper budgets; derive RSC03", "derived", "step-boundary",
            0, UINT32_MAX);
        add("handoff.shm", "string", "/r3-moe-helper",
            "launcher COMMON R3_MOE_WORKER_SHM; per-instance identity RSC02");
        add("handoff.max_rows", "integer", 4096,
            "launcher COMMON R3_MOE_WORKER_MAX_ROWS=4096", "derived",
            "startup-only", 1);
        add("handoff.small_rows", "integer", 8,
            "launcher COMMON R3_MOE_WORKER_SMALL_ROWS=8", "tuned",
            "startup-only", 1);
        add("handoff.gemv_large_rows", "integer", 32,
            "launcher COMMON R3_MOE_WORKER_GEMV_LARGE_ROWS=32", "tuned",
            "startup-only", 1);
        add("handoff.moe_down_tile", "boolean", true,
            "launcher COMMON R3_MOE_WORKER_MOE_DOWN_TILE=1");
        add("handoff.skip_owned", "boolean", true,
            "launcher COMMON R3_MOE_WORKER_SKIP_OWNED=1");
        add("handoff.heat", "string", nullptr,
            "required: workload heat file has no portable path", "policy");
        add("handoff.histogram", "string", nullptr,
            "required: launcher RECORD path or off, no portable path", "policy");
        add("handoff.histogram_period_s", "integer", 60,
            "launcher RECORD_PERIOD=60; RHP02", "policy", "live", 0);
        add("handoff.placement", "string", nullptr,
            "layouts.toml helper placement; tail or roundrobin", "derived",
            "step-boundary", 0, 0, {"tail", "roundrobin"}, true);
        add("helper.count", "integer", nullptr,
            "WorkerConfig helper_count; RSC05 selected helper count", "discovered",
            "startup-only", 1, 4);
        add("helper.index", "integer", 0,
            "WorkerConfig helper_index=0; RSC06 derives process ordinal", "derived",
            "startup-only", 0, 3);
        add("helper.engine_ranks", "integer", nullptr,
            "WorkerConfig default 0 (unset), launcher single engine rank=1", "derived",
            "startup-only", 1, 4);
        add("helper.cpu", "boolean", false, "WorkerConfig cpu=false");
        add("helper.power_save", "boolean", true, "WorkerConfig power_save=true");
        add("helper.queue_warm", "boolean", false, "WorkerConfig queue_warm=false");
        add("helper.combine", "boolean", false,
            "WorkerConfig combine=false, layouts.toml may enable combine");
        add("helper.combine_min_rows", "integer", 256,
            "WorkerConfig combine_min_rows=256", "tuned", "startup-only", 1);
        add("helper.combine_threads", "integer", 8,
            "WorkerConfig combine_threads=8 (also capped by discovered CPUs)",
            "tuned", "startup-only", 1, 1024);
        add("helper.cpu_set_auto", "boolean", true,
            "WorkerConfig cpu_set_auto=true (auto vs all)");
        add("engine.model", "string", nullptr,
            "required: no portable model-file default", "policy");
        add("engine.radiance_home", "string", nullptr,
            "required: release installation location", "discovered");
        add("engine.port", "integer", 36080, "launcher PORT=36080", "policy",
            "startup-only", 1, 65535);
        add("engine.kernels", "string", nullptr,
            "launcher engine plugin + libr3moe + libref; layout engine card", "discovered",
            "startup-only", 0, 0, {}, true);
        add("engine.placement", "string", "expert_tiered",
            "launcher --placement expert_tiered", "policy", "startup-only",
            0, 0, {"expert_tiered"});
        add("engine.ngram_placement", "string", "ram",
            "launcher NGRAM=ram", "policy", "startup-only", 0, 0,
            {"ram", "disk", "vram"});
        add("engine.host_pool_mib", "integer", nullptr,
            "derived from memory.host_pool_mib (launcher --host-pool-mib)", "derived",
            "startup-only", 0, UINT32_MAX, {}, true);
        add("engine.prefix_cache_mib", "integer", nullptr,
            "derived from memory.host_kv_mib (launcher --prefix-cache-mib)", "derived",
            "startup-only", 0, UINT32_MAX, {}, true);
        add("engine.tp", "integer", 1, "launcher --tp 1 (one engine card)",
            "derived", "startup-only", 1, 64);
        add("engine.max_num_seqs", "integer", nullptr,
            "derived from service.max_sessions (launcher --max-num-seqs)", "derived",
            "startup-only", 1, UINT32_MAX, {}, true);
        add("engine.max_num_batched_tokens", "integer", 2048,
            "launcher CHUNK=2048; derive geometry RSC03", "derived",
            "startup-only", 1);
        add("engine.checkpoint_interval", "integer", 4096,
            "launcher --checkpoint-interval 4096", "derived", "startup-only", 1);
        add("engine.kv_cache_dtype", "string", "fp8",
            "launcher KV_DTYPE=fp8", "policy", "startup-only", 0, 0,
            {"fp8", "fp8_v4", "k6_v4"});
        add("engine.num_speculative_tokens", "integer", 3,
            "launcher --num-speculative-tokens 3; model MTP depth",
            "derived", "startup-only", 0, 64);
        add("engine.max_model_len", "integer", nullptr,
            "derived from service.max_context (launcher --max-model-len)", "derived",
            "startup-only", 1, UINT32_MAX, {}, true);
        add("engine.rope_scaling", "string", "none",
            "launcher ROPE=none; model trained context", "policy",
            "startup-only", 0, 0, {"none", "yarn:4"});
        add("engine.prefix_resume", "string", "nearest",
            "launcher PREFIX_RESUME=nearest", "policy", "startup-only", 0, 0,
            {"nearest", "off"});
        return f;
    }();
    return all;
}

const Field* field_for(const std::string& key) {
    for (const auto& f : schema_fields())
        if (f.key == key) return &f;
    return nullptr;
}

std::string closest(const std::string& key) {
    size_t best = std::numeric_limits<size_t>::max();
    std::string winner;
    for (const auto& f : schema_fields()) {
        const auto& s = f.key;
        std::vector<size_t> prev(s.size() + 1), curr(s.size() + 1);
        for (size_t j = 0; j <= s.size(); ++j) prev[j] = j;
        for (size_t i = 1; i <= key.size(); ++i) {
            curr[0] = i;
            for (size_t j = 1; j <= s.size(); ++j)
                curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1,
                                    prev[j - 1] + (key[i - 1] != s[j - 1])});
            prev.swap(curr);
        }
        if (prev.back() < best) { best = prev.back(); winner = s; }
    }
    return winner;
}

std::string read_text(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path.string());
    std::ostringstream text;
    text << in.rdbuf();
    if (!in.good() && !in.eof()) throw std::runtime_error("read error: " + path.string());
    return text.str();
}

uint64_t parse_uint_text(std::string s, const std::string& what) {
    std::istringstream in(s);
    std::string first;
    in >> first;
    uint64_t number = 0;
    const auto [end, err] = std::from_chars(first.data(), first.data() + first.size(), number);
    if (err != std::errc{} || end != first.data() + first.size())
        throw std::runtime_error(what + ": invalid unsigned integer");
    return number;
}

uint64_t read_uint(const fs::path& path) {
    return parse_uint_text(read_text(path), path.string());
}

Json toml_to_json(const toml::node& node) {
    if (auto t = node.as_table()) {
        Json j = Json::object();
        for (const auto& [k, v] : *t) j[std::string(k.str())] = toml_to_json(v);
        return j;
    }
    if (auto a = node.as_array()) {
        Json j = Json::array();
        for (const auto& item : *a) j.push_back(toml_to_json(item));
        return j;
    }
    if (auto s = node.value<std::string>()) return *s;
    if (auto n = node.value<int64_t>()) return *n;
    if (auto b = node.value<bool>()) return *b;
    if (auto d = node.value<double>()) return *d;
    throw std::runtime_error("unsupported TOML value type (date/time is not a serving field)");
}

Json read_toml(const fs::path& file) {
    if (file.empty()) return Json::object();
    try { return toml_to_json(toml::parse_file(file.string())); }
    catch (const toml::parse_error& e) {
        throw std::runtime_error(file.string() + ": TOML parse error: " + e.what());
    }
}

void flatten(const Json& node, const std::string& prefix,
             std::vector<std::pair<std::string, Json>>& out) {
    if (node.is_object()) {
        if (node.empty() && !prefix.empty()) out.emplace_back(prefix, node);
        for (auto it = node.begin(); it != node.end(); ++it)
            flatten(it.value(), prefix.empty() ? it.key() : prefix + "." + it.key(), out);
    } else {
        out.emplace_back(prefix, node);
    }
}

void validate_value(const Field& field, const Json& v, const std::string& source) {
    const std::string where = source + ": " + field.key;
    if (field.kind == "boolean") {
        if (!v.is_boolean()) throw std::runtime_error(where + ": expected boolean");
    } else if (field.kind == "string") {
        if (!v.is_string()) throw std::runtime_error(where + ": expected string");
        if (v.get<std::string>().empty())
            throw std::runtime_error(where + ": string must not be empty");
        if (!field.choices.empty() &&
            std::find(field.choices.begin(), field.choices.end(),
                      v.get<std::string>()) == field.choices.end())
            throw std::runtime_error(where + ": invalid enum value " + v.dump());
    } else if (field.kind == "integer" || field.kind == "integer_array") {
        auto check = [&](const Json& x) {
            if (!x.is_number_integer() ||
                (x.is_number_unsigned() && x.get<uint64_t>() > INT64_MAX) ||
                (!x.is_number_unsigned() && x.get<int64_t>() < 0))
                throw std::runtime_error(where + ": expected non-negative integer");
            uint64_t n = x.get<uint64_t>();
            if (n < field.lo || n > field.hi)
                throw std::runtime_error(where + ": range " + std::to_string(field.lo) +
                                         ".." + std::to_string(field.hi));
        };
        if (field.kind == "integer") check(v);
        else {
            if (!v.is_array()) throw std::runtime_error(where + ": expected integer array");
            for (const auto& x : v) check(x);
        }
    } else if (field.kind == "string_array") {
        if (!v.is_array()) throw std::runtime_error(where + ": expected string array");
        for (const auto& x : v)
            if (!x.is_string() || x.get<std::string>().empty())
                throw std::runtime_error(where + ": array entries must be nonempty strings");
    } else {
        throw std::logic_error("unknown schema field kind");
    }
}

void put(Json& values, Json& provenance, const Field& field, const Json& value,
         const std::string& origin, const std::string& source,
         const Json& inputs = Json::array()) {
    validate_value(field, value, source);
    auto point = field.key.find('.');
    if (point == std::string::npos) values[field.key] = value;
    else values[field.key.substr(0, point)][field.key.substr(point + 1)] = value;
    provenance[field.key] = {{"origin", origin}, {"source", source},
                             {"inputs", inputs}, {"lifecycle", field.lifecycle},
                             {"class", field.category}, {"confidence", "exact"}};
}

Json value_at(const Json& values, const std::string& name) {
    const auto dot = name.find('.');
    if (dot == std::string::npos) return values.at(name);
    return values.at(name.substr(0, dot)).at(name.substr(dot + 1));
}

bool exists(const Json& values, const std::string& name) {
    const auto dot = name.find('.');
    if (dot == std::string::npos) return values.contains(name);
    return values.contains(name.substr(0, dot)) &&
           values.at(name.substr(0, dot)).contains(name.substr(dot + 1));
}

void layer(Json& values, Json& provenance, Json input,
           const std::string& origin, const std::string& path) {
    if (!input.is_object()) throw std::runtime_error(path + ": expected TOML table");
    input.erase("hard_caps");
    std::vector<std::pair<std::string, Json>> entries;
    flatten(input, "", entries);
    for (const auto& [key, v] : entries) {
        const Field* f = field_for(key);
        if (!f) throw std::runtime_error(path + ": unknown key " + key +
                                         " (nearest valid key: " + closest(key) + ")");
        if (f->derived) throw std::runtime_error(path + ": " + key +
                                                  " is derived and cannot be overridden");
        put(values, provenance, *f, v, origin, path, Json::array({path + ":" + key}));
    }
}

std::string quote_str(std::string s) {
    // TOML string literals for --set; values are parsed as TOML, not as CLI flags.
    return Json(s).dump();
}

Json parse_set(const std::string& key, const std::string& raw) {
    const Field* f = field_for(key);
    if (!f) throw std::runtime_error("--set: unknown key " + key +
                                     " (nearest valid key: " + closest(key) + ")");
    // Unquoted strings are convenient for paths and simple enums; arrays
    // and numeric/bool values must still be valid TOML of the correct type.
    const bool string_field = f->kind == "string";
    try {
        auto parsed = toml::parse("v = " + (string_field && !raw.empty() &&
                raw.front() != '"' && raw.front() != '\'' ? quote_str(raw) : raw));
        return toml_to_json(*parsed.get("v"));
    } catch (const toml::parse_error& e) {
        throw std::runtime_error("--set " + key + ": invalid TOML value: " + e.what());
    }
}

// Snapshot of the named recipes in engines/radiance/layouts.toml. Device choice
// comes only from discovery. The numeric limits are interim RSC03 inputs.
struct LayoutRecipe {
    bool includes_rdna2, tail, combine;
    uint64_t head_mib, tail_mib;
};
LayoutRecipe recipe_for(const std::string& name) {
    if (name == "flashnext-4gpu") return {true, true, true, 22528, 14336};
    if (name == "flashnext-2xtx-rdna2-tail") return {true, true, false, 22528, 2048};
    if (name == "flashnext-2xtx-rdna2-tail-combine")
        return {true, true, true, 22528, 2048};
    if (name == "flashnext-2xtx-rdna2-head") return {true, false, false, 22528, 4096};
    if (name == "flashnext-2xtx-combine") return {false, false, true, 22528, 0};
    if (name == "flashnext-2xtx") return {false, false, false, 22528, 0};
    throw std::runtime_error("layout.name: unknown named layout " + name +
                             " (no fallback)");
}

std::string selection_name(const Json& layer_data, const std::string& original) {
    if (layer_data.contains("layout") && layer_data.at("layout").is_object() &&
        layer_data.at("layout").contains("name")) {
        auto n = layer_data.at("layout").at("name");
        if (!n.is_string()) throw std::runtime_error("layout.name: expected string");
        return n.get<std::string>();
    }
    return original;
}

Json choose(const Json& facts, const std::string& arch,
            std::set<std::string>& used, size_t count) {
    Json chosen = Json::array();
    if (!facts.contains("gpus") || !facts.at("gpus").is_array())
        throw std::runtime_error("discovered facts missing gpus array");
    for (const auto& gpu : facts.at("gpus")) {
        if (gpu.at("arch") == arch && !used.count(gpu.at("bdf").get<std::string>())) {
            chosen.push_back(gpu);
            used.insert(gpu.at("bdf").get<std::string>());
            if (chosen.size() == count) break;
        }
    }
    if (chosen.size() != count)
        throw std::runtime_error("layout: missing " + arch + " card(s): need " +
                                 std::to_string(count) + ", found " +
                                 std::to_string(chosen.size()));
    return chosen;
}

std::string pci_bdf(uint64_t domain, uint64_t loc) {
    std::ostringstream s;
    s << std::hex << std::setfill('0') << std::setw(4) << domain << ':'
      << std::setw(2) << ((loc >> 8) & 0xff) << ':'
      << std::setw(2) << ((loc >> 3) & 0x1f) << '.'
      << (loc & 7);
    return s.str();
}

Json properties(const fs::path& path) {
    Json j = Json::object();
    std::istringstream input(read_text(path));
    std::string key, value;
    while (input >> key >> value) {
        uint64_t n = parse_uint_text(value, path.string() + ":" + key);
        j[key] = n;
    }
    return j;
}

uint64_t pval(const Json& j, const std::string& k, const fs::path& file) {
    if (!j.contains(k)) throw std::runtime_error(file.string() + ": missing " + k);
    return j.at(k).get<uint64_t>();
}

Json parse_meminfo(const fs::path& file) {
    Json data = Json::object();
    std::istringstream input(read_text(file));
    std::string key, unit;
    uint64_t val;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        if (!(row >> key >> val)) continue;
        if (!key.empty() && key.back() == ':') key.pop_back();
        if (key == "MemTotal" || key == "MemAvailable" || key == "SwapTotal") {
            if (!(row >> unit) || unit != "kB")
                throw std::runtime_error(file.string() + ": unexpected unit for " + key);
            data[key] = val / 1024;
        }
    }
    for (const char* name : {"MemTotal", "MemAvailable", "SwapTotal"})
        if (!data.contains(name)) throw std::runtime_error(file.string() +
                                                            ": missing " + name);
    return data;
}

void check_caps(Json& values, const Json& baseline, const Json& caps, const Json& facts,
                const std::string& source) {
    if (!caps.is_object()) throw std::runtime_error(source + ": hard_caps must be a table");
    std::vector<std::pair<std::string, Json>> entries;
    flatten(caps, "hard_caps", entries);
    for (const auto& [key, cap] : entries) {
        const std::string target = key.substr(std::string("hard_caps.").size());
        if (target != "memory.host_pool_mib" && target != "memory.host_kv_mib" &&
            target != "memory.tail_budget_mib" &&
            target != "memory.helper_budgets_mib")
            throw std::runtime_error(source + ": unknown cap " + key +
                                     " (allowed: hard_caps.memory.*)");
        const Field* field = field_for(target);
        validate_value(*field, cap, source + ":hard cap");
        auto actual = value_at(values, target);
        if (cap.is_array()) {
            if (cap.size() != actual.size())
                throw std::runtime_error(source + ": " + key + " length mismatch");
            for (size_t i = 0; i < cap.size(); ++i)
                if (actual[i].get<uint64_t>() > cap[i].get<uint64_t>())
                    throw std::runtime_error(source + ": hard cap refusal: " + target +
                                             "[" + std::to_string(i) + "]");
        } else if (actual.get<uint64_t>() > cap.get<uint64_t>()) {
            throw std::runtime_error(source + ": hard cap refusal: " + target);
        }
        // A site's cap cannot be looser than detected physical capacity.
        uint64_t physical = facts.at("host").at("mem_total_mib").get<uint64_t>();
        if ((target == "memory.host_pool_mib" || target == "memory.host_kv_mib") &&
            cap.get<uint64_t>() > physical)
            throw std::runtime_error(source + ": " + key + " exceeds detected RAM");
        const Json previous = value_at(baseline, target);
        if (cap.is_array()) {
            if (cap.size() != previous.size())
                throw std::runtime_error(source + ": hard cap baseline length mismatch: " + target);
            for (size_t i = 0; i < cap.size(); ++i)
                if (cap[i].get<uint64_t>() > previous[i].get<uint64_t>())
                    throw std::runtime_error(source + ": hard caps may only tighten: " + target);
        } else if (cap.get<uint64_t>() > previous.get<uint64_t>()) {
            throw std::runtime_error(source + ": hard caps may only tighten: " + target);
        }
    }
}

} // namespace

/* FIPS 180-4 SHA-256, self-contained so the release package carries no crypto library for one content hash. */
std::string sha256_hex(const std::string& text) {
    static constexpr uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    std::string m = text;
    const uint64_t bits = uint64_t(text.size()) * 8u;
    m.push_back(char(0x80));
    while (m.size() % 64 != 56) m.push_back(char(0));
    for (int i = 7; i >= 0; --i) m.push_back(char((bits >> (i * 8)) & 0xffu));
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = uint32_t(uint8_t(m[off + 4 * i])) << 24 | uint32_t(uint8_t(m[off + 4 * i + 1])) << 16 |
                   uint32_t(uint8_t(m[off + 4 * i + 2])) << 8 | uint32_t(uint8_t(m[off + 4 * i + 3]));
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    std::ostringstream hex;
    for (uint32_t v : h) hex << std::hex << std::setfill('0') << std::setw(8) << v;
    return hex.str();
}

Json detect(const fs::path& root) {
    const auto kfd = root / "sys/class/kfd/kfd/topology/nodes";
    if (!fs::is_directory(kfd))
        throw std::runtime_error("KFD topology missing: " + kfd.string());
    std::vector<std::pair<uint64_t, fs::path>> nodes;
    for (const auto& entry : fs::directory_iterator(kfd)) {
        if (!entry.is_directory()) continue;
        auto name = entry.path().filename().string();
        if (name.empty() || name.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("non-numeric KFD node directory: " + name);
        nodes.emplace_back(parse_uint_text(name, "KFD node"), entry.path());
    }
    std::sort(nodes.begin(), nodes.end());
    Json gpus = Json::array();
    std::set<std::string> bdfs;
    for (const auto& [node_number, path] : nodes) {
        const auto props = properties(path / "properties");
        const uint64_t version = pval(props, "gfx_target_version", path);
        if (version == 0) continue;  // KFD CPU node; not a HIP device.
        const uint64_t major = version / 10000;
        const uint64_t minor = (version / 100) % 100;
        const uint64_t stepping = version % 100;
        const std::string arch = "gfx" + std::to_string(major) +
                                 std::to_string(minor) + std::to_string(stepping);
        const std::string bdf =
            pci_bdf(pval(props, "domain", path), pval(props, "location_id", path));
        if (!bdfs.insert(bdf).second)
            throw std::runtime_error("duplicate KFD PCI locator " + bdf);
        auto pci = root / "sys/bus/pci/devices" / bdf;
        Json gpu = {
            {"kfd_node", node_number}, {"hip_ordinal", gpus.size()},
            {"arch", arch}, {"gfx_target_version", version},
            {"simd_count", pval(props, "simd_count", path)},
            {"bdf", bdf},
            {"drm_render_minor", pval(props, "drm_render_minor", path)},
            {"vram_total_mib", read_uint(pci / "mem_info_vram_total") / 1048576},
            {"vram_used_mib", read_uint(pci / "mem_info_vram_used") / 1048576},
            {"current_link_speed", read_text(pci / "current_link_speed")},
            {"current_link_width", read_uint(pci / "current_link_width")},
            {"max_link_speed", read_text(pci / "max_link_speed")},
            {"max_link_width", read_uint(pci / "max_link_width")}
        };
        const auto numa_s = read_text(pci / "numa_node");
        int numa = -1;
        try { numa = std::stoi(numa_s); }
        catch (...) { throw std::runtime_error(pci.string() + ": invalid numa_node"); }
        gpu["numa_node"] = numa;
        const auto total = gpu.at("vram_total_mib").get<uint64_t>();
        const auto used = gpu.at("vram_used_mib").get<uint64_t>();
        if (used > total) throw std::runtime_error(bdf + ": used VRAM exceeds total");
        gpu["vram_available_mib"] = total - used;
        // Trim the newline in the sysfs speed string.
        for (const char* k : {"current_link_speed", "max_link_speed"}) {
            auto s = gpu[k].get<std::string>();
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            gpu[k] = s;
        }
        gpus.push_back(std::move(gpu));
    }
    auto mem = parse_meminfo(root / "proc/meminfo");
    struct statvfs st {};
    const auto shm = root / "dev/shm";
    if (::statvfs(shm.c_str(), &st))
        throw std::runtime_error("cannot statvfs " + shm.string());
    return {
        {"source", "kfd+sysfs+proc+statvfs"},
        {"gpus", gpus},
        {"host", {
            {"mem_total_mib", mem.at("MemTotal")},
            {"mem_available_mib", mem.at("MemAvailable")},
            {"swap_total_mib", mem.at("SwapTotal")},
            {"shm_total_mib", (uint64_t(st.f_blocks) * st.f_frsize) / 1048576},
            {"shm_available_mib", (uint64_t(st.f_bavail) * st.f_frsize) / 1048576},
            {"cpu_count", std::max(1u, std::thread::hardware_concurrency())}
        }}
    };
}

Json json_schema() {
    Json root = {
        {"$schema", "https://json-schema.org/draft/2020-12/schema"},
        {"title", "BigCherry serving manifest input v1"}, {"type", "object"},
        {"additionalProperties", false}, {"properties", Json::object()}
    };
    for (const auto& f : schema_fields()) {
        Json prop = Json::object();
        if (f.kind == "integer" || f.kind == "integer_array") {
            prop["type"] = f.kind == "integer" ? "integer" : "array";
            Json item = {{"type", "integer"}, {"minimum", f.lo}, {"maximum", f.hi}};
            if (f.kind == "integer") prop.update(item);
            else prop["items"] = item;
        } else if (f.kind == "string_array") {
            prop["type"] = "array";
            prop["items"] = {{"type", "string"}, {"minLength", 1}};
        } else {
            prop["type"] = f.kind;
            if (f.kind == "string") prop["minLength"] = 1;
        }
        if (!f.choices.empty()) prop["enum"] = f.choices;
        if (!f.fallback.is_null()) prop["default"] = f.fallback;
        prop["x-origin"] = f.origin;
        prop["x-class"] = f.category;
        prop["x-lifecycle"] = f.lifecycle;
        if (f.derived) prop["readOnly"] = true;
        const auto dot = f.key.find('.');
        if (dot == std::string::npos) {
            root["properties"][f.key] = prop;
            if (f.fallback.is_null() && !f.derived)
                root["required"].push_back(f.key);
        } else {
            const std::string section = f.key.substr(0, dot);
            const std::string key = f.key.substr(dot + 1);
            if (!root["properties"].contains(section))
                root["properties"][section] = {
                    {"type", "object"}, {"additionalProperties", false},
                    {"properties", Json::object()}
                };
            root["properties"][section]["properties"][key] = prop;
            if (f.fallback.is_null() && !f.derived &&
                (section == "engine" || section == "handoff")) {
                auto& group = root["properties"][section];
                if (!group.contains("required")) group["required"] = Json::array();
                group["required"].push_back(key);
            }
        }
    }
    // Site-only constraint namespace; it never becomes a resolved value.
    Json caps = {{"type", "object"}, {"additionalProperties", false},
                 {"properties", Json::object()}};
    Json mem = {{"type", "object"}, {"additionalProperties", false},
                {"properties", Json::object()}};
    for (const auto& key : {"host_pool_mib", "host_kv_mib", "tail_budget_mib",
                            "helper_budgets_mib"}) {
        const auto* f = field_for("memory." + std::string(key));
        Json prop = {{"type", f->kind == "integer_array" ? "array" : "integer"}};
        Json range = {{"type", "integer"}, {"minimum", f->lo}, {"maximum", f->hi}};
        if (f->kind == "integer_array") prop["items"] = range;
        else { prop["minimum"] = f->lo; prop["maximum"] = f->hi; }
        mem["properties"][key] = prop;
    }
    caps["properties"]["memory"] = mem;
    caps["description"] = "site file only; maxima can only tighten";
    root["properties"]["hard_caps"] = caps;
    return root;
}

Resolved resolve(const Options& options, const Json& facts) {
    Json values = Json::object(), provenance = Json::object();
    for (const auto& f : schema_fields())
        if (!f.fallback.is_null())
            put(values, provenance, f, f.fallback, "schema-default", f.origin);
    const Json site = read_toml(options.site), profile = read_toml(options.profile);
    if (profile.contains("hard_caps"))
        throw std::runtime_error(options.profile.string() + ": hard_caps: site-only; profile cannot change hard caps");
    std::string name = value_at(values, "layout.name").get<std::string>();
    name = selection_name(site, name);
    name = selection_name(profile, name);
    for (const auto& set : options.sets) {
        auto eq = set.find('=');
        if (eq == std::string::npos || eq == 0)
            throw std::runtime_error("--set expects dotted.key=value");
        if (set.substr(0, eq) == "layout.name")
            name = parse_set("layout.name", set.substr(eq + 1)).get<std::string>();
    }
    const auto layout = recipe_for(name);
    std::set<std::string> used;
    Json engine = choose(facts, "gfx1201", used, 1);
    Json head = choose(facts, "gfx1100", used, 2);
    Json tail = layout.includes_rdna2 ? choose(facts, "gfx1030", used, 1) : Json::array();
    Json cards = Json::array(), budgets = Json::array(), selected = Json::array();
    for (const auto& g : head) {
        cards.push_back(g.at("bdf")); budgets.push_back(layout.head_mib);
        selected.push_back(g);
    }
    for (const auto& g : tail) {
        cards.push_back(g.at("bdf")); budgets.push_back(layout.tail_mib);
        selected.push_back(g);
    }
    auto derived = [&](const std::string& key, const Json& v,
                       const std::string& detail, Json inputs = Json::array()) {
        put(values, provenance, *field_for(key), v, "discovered", detail, inputs);
    };
    derived("layout.engine_card", engine[0].at("bdf"),
            "KFD and PCI inventory, " + name, Json::array({"layout.name", "facts.gpus"}));
    derived("layout.helper_cards", cards, "KFD and PCI inventory, " + name,
            Json::array({"layout.name", "facts.gpus"}));
    derived("memory.helper_budgets_mib", budgets,
            "engines/radiance/layouts.toml " + name + " (interim; RSC03)",
            Json::array({"layout.name", "facts.gpus"}));
    derived("memory.tail_budget_mib", layout.tail ? layout.tail_mib : 0,
            "engines/radiance/layouts.toml " + name + " (interim; RSC03)",
            Json::array({"layout.name"}));
    derived("handoff.placement", layout.tail ? "tail" : "roundrobin",
            "engines/radiance/layouts.toml " + name,
            Json::array({"layout.name"}));
    derived("helper.count", cards.size(), "RSC05 KFD selection",
            Json::array({"layout.helper_cards"}));
    derived("helper.engine_ranks", 1, "launcher --tp 1 and layout env",
            Json::array({"layout.engine_card"}));
    derived("helper.combine", layout.combine,
            "engines/radiance/layouts.toml " + name,
            Json::array({"layout.name"}));
    const Json baseline_values = values;
    layer(values, provenance, site, "site", options.site.empty() ? "site" : options.site.string());
    layer(values, provenance, profile, "profile",
          options.profile.empty() ? "profile" : options.profile.string());
    for (const auto& set : options.sets) {
        const auto eq = set.find('=');
        if (eq == std::string::npos || eq == 0)
            throw std::runtime_error("--set expects dotted.key=value");
        const std::string key = set.substr(0, eq);
        if (key.rfind("hard_caps.", 0) == 0)
            throw std::runtime_error("--set " + key + ": hard caps are site-only");
        const Field* f = field_for(key);
        if (!f) throw std::runtime_error("--set: unknown key " + key +
                                         " (nearest valid key: " + closest(key) + ")");
        if (f->derived)
            throw std::runtime_error("--set " + key + ": derived, cannot override");
        put(values, provenance, *f, parse_set(key, set.substr(eq + 1)),
            "cli", "--set", Json::array({set}));
    }
    if (value_at(values, "layout.name") != name)
        throw std::runtime_error("layout.name: resolution inconsistency");
    if (layout.tail) {
        const Json b = value_at(values, "memory.helper_budgets_mib");
        if (b.empty())
            throw std::runtime_error("memory.helper_budgets_mib: missing tail helper");
        put(values, provenance, *field_for("memory.tail_budget_mib"), b.back(),
            "derived", "final helper tail budget (RSC03 interim)",
            Json::array({"memory.helper_budgets_mib"}));
    }


    // Canonical derived engine option names. There is no argv/env conversion.
    auto alias = [&](const std::string& key, const Json& v,
                     const std::string& source, const std::string& input) {
        put(values, provenance, *field_for(key), v, "derived", source,
            Json::array({input}));
    };
    alias("engine.max_num_seqs", value_at(values, "service.max_sessions"),
          "RSC01 service/session projection", "service.max_sessions");
    alias("engine.max_model_len", value_at(values, "service.max_context"),
          "RSC01 service/context projection", "service.max_context");
    alias("engine.host_pool_mib", value_at(values, "memory.host_pool_mib"),
          "RSC01 memory/pool projection", "memory.host_pool_mib");
    alias("engine.prefix_cache_mib", value_at(values, "memory.host_kv_mib"),
          "RSC01 memory/host KV projection", "memory.host_kv_mib");
    // All current supported layouts use the engine architecture's plugin
    // declared as libr4d.so in archs.toml, followed by the mailbox and ref.
    alias("engine.kernels", "libr4d,libr3moe,libref",
          "engines/radiance/archs.toml gfx1201 plugin; launcher ENGINE_KERNELS",
          "layout.engine_card");

    for (const char* key : {"engine.model", "engine.radiance_home"})
        if (!exists(values, key))
            throw std::runtime_error(std::string("missing required key: ") + key);
    for (const auto& field : schema_fields())
        if (!exists(values, field.key))
            throw std::runtime_error("missing required key: " + field.key +
                                     " (" + field.origin + ")");
    const auto max_rows = value_at(values, "handoff.max_rows").get<uint64_t>();
    const auto small = value_at(values, "handoff.small_rows").get<uint64_t>();
    const auto large = value_at(values, "handoff.gemv_large_rows").get<uint64_t>();
    if (small > max_rows || large < small || large > max_rows)
        throw std::runtime_error("handoff: require small_rows <= gemv_large_rows <= max_rows");
    const auto shm = value_at(values, "handoff.shm").get<std::string>();
    if (shm[0] != '/' || shm.size() <= 1 || shm.find('/', 1) != std::string::npos)
        throw std::runtime_error("handoff.shm must be a POSIX shm name /name");
    const auto hcount = value_at(values, "helper.count").get<size_t>();
    const auto hindex = value_at(values, "helper.index").get<size_t>();
    if (hindex >= hcount)
        throw std::runtime_error("helper.index must be below helper.count");
    if (hcount != cards.size())
        throw std::runtime_error("helper.count must match selected layout helper cards");
    if (value_at(values, "helper.engine_ranks").get<size_t>() !=
        value_at(values, "engine.tp").get<size_t>())
        throw std::runtime_error("helper.engine_ranks must equal engine.tp");
    if (value_at(values, "helper.combine_threads").get<uint64_t>() >
        facts.at("host").at("cpu_count").get<uint64_t>())
        throw std::runtime_error("helper.combine_threads exceeds discovered CPU count");
    if (value_at(values, "layout.engine_card") != engine[0].at("bdf") ||
        value_at(values, "layout.helper_cards") != cards)
        throw std::runtime_error("layout: selectors must match the discovered named layout; no fallback");

    const Json effective_budgets = value_at(values, "memory.helper_budgets_mib");
    if (effective_budgets.size() != cards.size())
        throw std::runtime_error("memory.helper_budgets_mib must match helper card count");
    for (size_t i = 0; i < selected.size(); ++i) {
        const uint64_t budget = effective_budgets[i].get<uint64_t>();
        const uint64_t total = selected[i].at("vram_total_mib").get<uint64_t>();
        // archs.toml reserve-mib=1024 for both helper card families.
        const uint64_t reserve = 1024;
        if (budget == 0 || total < reserve || budget > total - reserve)
            throw std::runtime_error("memory.helper_budgets_mib[" +
                                     std::to_string(i) + "] exceeds detected VRAM minus arch reserve");
    }
    if (layout.tail &&
        value_at(values, "memory.tail_budget_mib") != effective_budgets.back())
        throw std::runtime_error("memory.tail_budget_mib must equal final helper budget");
    if (!layout.tail && value_at(values, "memory.tail_budget_mib") != 0)
        throw std::runtime_error("memory.tail_budget_mib must be zero without a tail");
    const uint64_t host_total = facts.at("host").at("mem_total_mib").get<uint64_t>();
    const uint64_t pool = value_at(values, "memory.host_pool_mib").get<uint64_t>();
    const uint64_t kv = value_at(values, "memory.host_kv_mib").get<uint64_t>();
    if (pool > host_total || kv > host_total - pool)
        throw std::runtime_error("memory host tiers exceed discovered MemTotal");
    if (site.contains("hard_caps"))
        check_caps(values, baseline_values, site.at("hard_caps"), facts, options.site.string());

    Resolved r;
    r.values = values; r.provenance = provenance; r.facts = facts;
    r.typed.schema_version = values.at("schema_version").get<uint32_t>();
    auto& c = r.typed;
    const auto& s = values.at("service");
    c.service = {s.at("objective"), s.at("retention"), s.at("quality"),
                 s.at("max_sessions"), s.at("max_context")};
    const auto& l = values.at("layout");
    c.layout = {l.at("name"), l.at("engine_card"),
                l.at("helper_cards").get<std::vector<std::string>>()};
    const auto& m = values.at("memory");
    c.memory = {m.at("host_pool_mib"), m.at("host_kv_mib"),
                m.at("tail_budget_mib"),
                m.at("helper_budgets_mib").get<std::vector<uint64_t>>()};
    const auto& a = values.at("handoff");
    c.handoff = {a.at("shm"), a.at("heat"), a.at("histogram"),
                 a.at("placement"), a.at("max_rows"), a.at("small_rows"),
                 a.at("gemv_large_rows"), a.at("histogram_period_s"),
                 a.at("moe_down_tile"), a.at("skip_owned")};
    const auto& h = values.at("helper");
    c.helper = {h.at("count"), h.at("index"), h.at("engine_ranks"),
                h.at("cpu"), h.at("power_save"), h.at("queue_warm"),
                h.at("combine"), h.at("combine_min_rows"),
                h.at("combine_threads"), h.at("cpu_set_auto")};
    const auto& e = values.at("engine");
    c.engine = {e.at("model"), e.at("radiance_home"), e.at("kernels"),
                e.at("placement"), e.at("ngram_placement"), e.at("kv_cache_dtype"),
                e.at("rope_scaling"), e.at("prefix_resume"), e.at("port"),
                e.at("tp"), e.at("max_num_seqs"), e.at("max_num_batched_tokens"),
                e.at("checkpoint_interval"), e.at("num_speculative_tokens"),
                e.at("max_model_len"), e.at("host_pool_mib"), e.at("prefix_cache_mib")};
    Json process_helpers = Json::array();
    for (size_t i = 0; i < cards.size(); ++i) {
        Json worker = h;
        worker["index"] = i;
        const bool rdna2 = selected[i].at("arch") == "gfx1030";
        const bool tail_card = layout.tail && rdna2;
        const uint64_t share = tail_card ? 0 :
            (name == "flashnext-2xtx-rdna2-head" && !rdna2 ? 5 : 1);
        const Json formats = rdna2 ? Json::array({"w4nl64"}) :
            Json::array({"w4nl64", "fp8_e4m3_block"});
        process_helpers.push_back({
            {"config", worker}, {"bdf", cards[i]},
            {"arch", selected[i].at("arch")}, {"budget_mib", effective_budgets[i]},
            {"hip_ordinal", selected[i].at("hip_ordinal")},
            {"plugin_file", rdna2 ? "libr2.so" : "libr3.so"},
            {"formats", formats}, {"rows_kernel", rdna2},
            {"placement", tail_card ? "tail" : "head"}, {"share", share},
            {"provenance", {{"origin", "archs.toml + layouts.toml + KFD"},
                            {"inputs", Json::array({"layout.name", "facts.gpus"})},
                            {"lifecycle", "startup-only"}}}
        });
    }
    r.manifest = {{"schema_version", 1}, {"values", values},
                  {"provenance", provenance}, {"facts", facts},
                  {"process_helpers", process_helpers}, {"hash_algorithm", "sha256"}};
    r.hash = sha256_hex(r.manifest.dump());
    r.manifest["content_sha256"] = r.hash;
    return r;
}
} // namespace bc::serving
