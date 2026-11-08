#include "serving.h"
#include "calibration.h"
#include "moe_remote_shm.h"
#include "experts.h"
#include "worker_footprint.h"
#include "sha256.h"
#include "generated/serving_types.h"
#include "third_party/toml.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
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
    // Absent unless set: no default, no "missing required key", not in the schema's required list (RSC11 A1).
    bool optional = false;
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
        auto add_optional = [&](const char* k, const char* kind, const char* source, uint64_t lo, uint64_t hi,
                                std::vector<std::string> choices = {}) {
            f.push_back({k, kind, source, "policy", "startup-only", Json(), lo, hi, std::move(choices), false, true});
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
            0, 0, {"keep", "discard"});
        add("service.quality", "string", "baseline",
            "launcher: baseline fp8 KV policy", "policy", "startup-only",
            0, 0, {"baseline", "strict"});
        add("layout.name", "string", "flashnext-4gpu",
            "launcher LAYOUT default; named layouts in engines/radiance/layouts.toml");
        add("layout.engine_cards", "string_array", nullptr,
            "RSC05: selected PCI BDFs from KFD GPU nodes, in rank order", "discovered");
        add("layout.helper_cards", "string_array", nullptr,
            "RSC05: selected PCI BDFs from KFD GPU nodes; empty for an engine-only layout", "discovered", "step-boundary");
        add("memory.host_pool_mib", "integer", 9216,
            "launcher HOST_POOL_MIB=9216 measured interim; derive RSC03", "derived",
            "startup-only", 0, UINT32_MAX);
        add("memory.host_kv_mib", "integer", 12288,
            "launcher HOST_KV_MIB=12288 measured interim; derive RSC03", "derived",
            "startup-only", 0, UINT32_MAX);
        add("memory.host_reserve_mib", "integer", nullptr,
            "calibrated: max(calibration.host_reserve.floor_mib, ceil(calibration.host_reserve.total_fraction_ppm x "
            "MemTotal / 10^6)); a site hard_floors entry may only raise it", "derived", "startup-only",
            0, UINT32_MAX, {}, true);
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
        add_optional("handoff.heat", "string", "required with helpers: workload heat file has no portable path", 0, 0);
        add_optional("handoff.histogram", "string", "required with helpers: launcher RECORD path or off, no portable path", 0, 0);
        add("handoff.histogram_period_s", "integer", 60,
            "launcher RECORD_PERIOD=60; RHP02", "policy", "live", 0);
        add("handoff.placement", "string", nullptr,
            "layouts.toml helper placement; tail or roundrobin", "derived",
            "step-boundary", 0, 0, {"tail", "roundrobin"}, true);
        add("helper.count", "integer", nullptr,
            "WorkerConfig helper_count; RSC05 selected helper count", "discovered",
            "startup-only", 0, 4);
        add("helper.index", "integer", 0,
            "WorkerConfig helper_index=0; RSC06 derives process ordinal", "derived",
            "startup-only", 0, 3);
        add("helper.engine_ranks", "integer", nullptr,
            "layouts.toml engine count: the engine's TP ranks the helpers serve (WorkerConfig default 0 = unset)", "derived",
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
        add("engine.host_headroom_mib", "integer", nullptr,
            "derived from memory.host_reserve_mib: the engine keeps the plan's one reserve free when it "
            "sizes its checkpoint store and pins tables", "derived", "startup-only", 0, UINT32_MAX, {}, true);
        add("engine.tp", "integer", nullptr, "layouts.toml engine count: one tensor-parallel rank per engine card",
            "derived", "startup-only", 1, 64, {}, true);
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
            {"nearest", "exact"});
        // Production engine options with no default: the manifest carries them only when an operator sets them, and an
        // absent one leaves radiance to its own default (amendment 11 fields; the contract registry binds each to its flag).
        add_optional("engine.tp_wire", "string", "radiance --tp-wire (cross-rank all-reduce payload)", 0, 0, {"exact", "wht6"});
        add_optional("engine.no_prefix_cache", "string", "radiance --no-prefix-cache: the value \"on\" disables prefix caching", 0, 0, {"on"});
        add_optional("engine.gpu_headroom_mib", "integer", "radiance --gpu-headroom-mib (VRAM left unclaimed on every engine card)", 0, UINT32_MAX);
        // The engine options the registry binds beyond those (amendment 13). The engine checks ranges and value text itself.
        for (const auto& o : engine_option_fields()) {
            if (!o.env.empty()) f.push_back({o.key, o.kind, "environment variable " + o.env + ", set from the manifest", "policy", "startup-only", Json(), 0, UINT32_MAX, {}, false, true});
            else if (o.is_switch) f.push_back({o.key, "string", "radiance flag (present means given)", "policy", "startup-only", Json(), 0, 0, {"on"}, false, true});
            else if (o.kind == "ref") f.push_back({o.key, "integer", "radiance flag", "policy", "startup-only", Json(), 0, UINT32_MAX, {}, false, true});
            else f.push_back({o.key, "string", "radiance flag (value text as the flag takes it)", "policy", "startup-only", Json(), 0, 0, {}, false, true});
        }
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
    // A TOML boolean reads as an integer through value<int64_t>(), so the boolean is asked first.
    if (auto b = node.as_boolean()) return b->get();
    if (auto s = node.value<std::string>()) return *s;
    if (auto n = node.value<int64_t>()) return *n;
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
    input.erase("hard_floors");
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

// engines/radiance/layouts.toml and archs.toml, embedded at build time (CMakeLists.txt writes layouts_seed.inc and
// archs_seed.inc): the layout names which cards play which part, archs.toml what a card type is. Device choice comes
// only from discovery. The numeric limits are interim RSC03 inputs.
constexpr const char kLayoutsSeed[] =
#include "layouts_seed.inc"
    ;
constexpr const char kArchsSeed[] =
#include "archs_seed.inc"
    ;

struct ArchInfo {
    std::string plugin;                   // kernels/<plugin>, e.g. libr4d.so
    std::vector<std::string> formats;
    bool rows_kernel = false;
    uint64_t reserve_mib = 0;
    std::vector<std::string> roles;
};
struct HelperGroup {
    std::string arch;
    int64_t count = 0;                    // -1: every unused card of the arch
    uint64_t budget_mib = 0, share = 1;
    bool tail = false;
};
struct LayoutRecipe {
    std::string engine_arch;
    size_t engine_count = 1;
    std::optional<std::string> tp_wire;
    std::vector<HelperGroup> groups;
    bool combine = false, tail = false;
};

Json embedded_toml(const char* text, const char* what) {
    try { return toml_to_json(toml::parse(std::string(text))); }
    catch (const toml::parse_error& e) { throw std::runtime_error(std::string(what) + ": TOML parse error: " + e.what()); }
}

ArchInfo arch_info(const std::string& arch) {
    static const Json archs = embedded_toml(kArchsSeed, "engines/radiance/archs.toml");
    if (!archs.at("arch").contains(arch))
        throw std::runtime_error("archs.toml: no card type " + arch);
    const Json& a = archs.at("arch").at(arch);
    ArchInfo info;
    info.plugin = a.at("plugin").get<std::string>();
    info.formats = a.at("formats").get<std::vector<std::string>>();
    info.rows_kernel = a.at("rows-kernel").get<bool>();
    info.reserve_mib = a.at("reserve-mib").get<uint64_t>();
    info.roles = a.at("roles").get<std::vector<std::string>>();
    return info;
}

LayoutRecipe recipe_for(const std::string& name) {
    static const Json layouts = embedded_toml(kLayoutsSeed, "engines/radiance/layouts.toml");
    if (!layouts.at("layout").contains(name))
        throw std::runtime_error("layout.name: unknown named layout " + name + " (no fallback)");
    const Json& l = layouts.at("layout").at(name);
    const std::string where = "layouts.toml [layout." + name + "]: ";
    LayoutRecipe r;
    const Json& e = l.at("engine");
    if (e.contains("devices"))
        throw std::runtime_error(where + "engine.devices pins host device indices; the serving resolver selects by architecture only");
    r.engine_arch = e.at("arch");
    r.engine_count = e.value("count", 1);
    const auto roles = arch_info(r.engine_arch).roles;
    if (std::find(roles.begin(), roles.end(), "engine") == roles.end())
        throw std::runtime_error(where + "architecture " + r.engine_arch + " cannot run the engine (archs.toml roles)");
    if (l.contains("tp-wire")) r.tp_wire = l.at("tp-wire").get<std::string>();
    if (l.contains("env"))
        for (auto it = l.at("env").begin(); it != l.at("env").end(); ++it) {
            if (it.key() != "R3_MOE_WORKER_COMBINE" || (it.value() != "1" && it.value() != "0"))
                throw std::runtime_error(where + "env " + it.key() + ": only R3_MOE_WORKER_COMBINE = \"0\" or \"1\" is a serving setting");
            r.combine = it.value() == "1";
        }
    if (l.contains("helper"))
        for (const Json& h : l.at("helper")) {
            if (h.contains("devices"))
                throw std::runtime_error(where + "helper.devices pins host device indices; the serving resolver selects by architecture only");
            HelperGroup g;
            g.arch = h.at("arch");
            const auto hroles = arch_info(g.arch).roles;
            if (std::find(hroles.begin(), hroles.end(), "helper") == hroles.end())
                throw std::runtime_error(where + "architecture " + g.arch + " cannot be a helper (archs.toml roles)");
            if (h.at("count").is_string()) {
                if (h.at("count") != "all") throw std::runtime_error(where + "helper count is a number or \"all\"");
                g.count = -1;
            } else {
                g.count = h.at("count");
                if (g.count < 1) throw std::runtime_error(where + "helper count must be at least 1");
            }
            // absent: the card's own VRAM less its archs.toml reserve-mib, per selected card (derived, not a literal)
            g.budget_mib = h.value("budget-mib", uint64_t{0});
            g.share = h.value("share", 1);
            g.tail = h.at("placement") == "tail";
            r.groups.push_back(g);
        }
    for (size_t i = 0; i < r.groups.size(); ++i)
        if (r.groups[i].tail && (i + 1 != r.groups.size() || i == 0))
            throw std::runtime_error(where + "only the last helper group may be a tail group, and only after a head group");
    r.tail = !r.groups.empty() && r.groups.back().tail;
    if (r.engine_count > 1 && !r.tp_wire)
        throw std::runtime_error(where + "an engine of " + std::to_string(r.engine_count) + " cards needs tp-wire");
    // RAD49: one request slot a (helper, engine rank), at most r3::moe_shm::kMaxSlots of them. A fixed helper count that
    // cannot fit is refused here, before any card; a group of count "all" is bounded once the cards are known (resolve).
    size_t fixed = 0;
    for (const auto& g : r.groups) fixed += g.count < 0 ? 1 : size_t(g.count);
    if (!r.groups.empty() && !r3::moe_shm::slots_fit(uint32_t(fixed), uint32_t(r.engine_count)))
        throw std::runtime_error(where + std::to_string(fixed) + " remote-MoE helper(s) behind an engine of " +
                                 std::to_string(r.engine_count) + " rank(s) need more than the protocol's " +
                                 std::to_string(r3::moe_shm::kMaxSlots) + " request slots (one a helper and rank)");
    return r;
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

constexpr size_t kEveryCard = std::numeric_limits<size_t>::max();
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
    if (count == kEveryCard ? chosen.empty() : chosen.size() != count)
        throw std::runtime_error("layout: missing " + arch + " card(s): need " +
                                 (count == kEveryCard ? std::string("at least 1") : std::to_string(count)) + ", found " +
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
            if (key == "MemTotal") data["MemTotal_kB"] = val;
        }
    }
    for (const char* name : {"MemTotal", "MemAvailable", "SwapTotal"})
        if (!data.contains(name)) throw std::runtime_error(file.string() +
                                                            ": missing " + name);
    return data;
}

/* A memory cgroup level's limit less the usage reclaim cannot give back (current less file pages
 * on v2, usage less cache on v1): radiance's rad::cgroup_room (core/engine_hostmem.cpp), read
 * under `root` so detection is testable. Null when the level has no limit. */
Json cgroup_room_mib(const fs::path& dir, bool v2) {
    auto read_ll = [](const fs::path& f, long long* v) {
        std::ifstream in(f);
        std::string t;
        if (!(in >> t)) return false;
        const auto r = std::from_chars(t.data(), t.data() + t.size(), *v);
        return r.ec == std::errc() && r.ptr == t.data() + t.size();
    };
    auto stat_ll = [](const fs::path& f, const char* key, long long* v) {
        std::ifstream in(f);
        std::string k;
        long long x = 0;
        while (in >> k >> x) if (k == key) { *v = x; return true; }
        return false;
    };
    long long lim = 0, used = 0, file = 0;
    if (v2) {
        if (!read_ll(dir / "memory.max", &lim)) return nullptr;   /* "max": no limit */
        if (!read_ll(dir / "memory.current", &used))
            throw std::runtime_error(dir.string() + ": memory.max without memory.current");
        (void)stat_ll(dir / "memory.stat", "file", &file);
    } else {
        if (!read_ll(dir / "memory.limit_in_bytes", &lim) || lim >= (1LL << 62)) return nullptr;
        if (!read_ll(dir / "memory.usage_in_bytes", &used))
            throw std::runtime_error(dir.string() + ": limit without memory.usage_in_bytes");
        if (!stat_ll(dir / "memory.stat", "total_cache", &file))
            (void)stat_ll(dir / "memory.stat", "cache", &file);
    }
    const long long hard = used > file ? used - file : 0;
    return uint64_t(lim > hard ? lim - hard : 0) / 1048576;
}

/* Every limited level of every memory cgroup this process is in, leaf first, as radiance's
 * memory_cgroups/cgroup_levels walk them; each level is a limit of its own. */
Json memory_cgroups(const fs::path& root) {
    Json levels = Json::array();
    std::istringstream input(read_text(root / "proc/self/cgroup"));
    std::string line;
    while (std::getline(input, line)) {
        const size_t a = line.find(':'), b = a == std::string::npos ? a : line.find(':', a + 1);
        if (b == std::string::npos) continue;
        const std::string ctrl = line.substr(a + 1, b - a - 1);
        std::string path = line.substr(b + 1);
        const bool v2 = ctrl.empty();
        bool memory = v2;
        for (size_t i = 0; !memory && i < ctrl.size();) {
            const size_t j = std::min(ctrl.find(',', i), ctrl.size());
            memory = ctrl.compare(i, j - i, "memory") == 0;
            i = j + 1;
        }
        if (!memory) continue;
        const fs::path base = root / (v2 ? "sys/fs/cgroup" : "sys/fs/cgroup/memory");
        for (;;) {
            const fs::path dir = path == "/" ? base : base / path.substr(1);
            if (Json room = cgroup_room_mib(dir, v2); !room.is_null())
                levels.push_back({{"dir", "/" + fs::relative(dir, root).generic_string()},
                                  {"version", v2 ? 2 : 1}, {"room_mib", room}});
            if (path.empty() || path == "/") break;
            const size_t up = path.find_last_of('/');
            path = up == 0 || up == std::string::npos ? "/" : path.substr(0, up);
        }
    }
    return levels;
}

/* RLIMIT_MEMLOCK's soft limit (null: unlimited) and CAP_IPC_LOCK, which lifts it. It bounds only
 * mlock()ed copies: pinned memory a card reads is a driver userptr mapping it does not count. */
Json memlock(const fs::path& root) {
    Json out = {{"limit_mib", nullptr}, {"cap_ipc_lock", false}};
    bool limit_seen = false, caps_seen = false;
    {
        std::istringstream input(read_text(root / "proc/self/limits"));
        std::string line;
        while (std::getline(input, line)) {
            if (line.rfind("Max locked memory", 0) != 0) continue;
            std::istringstream row(line.substr(std::string("Max locked memory").size()));
            std::string soft;
            if (!(row >> soft)) throw std::runtime_error("proc/self/limits: unreadable Max locked memory");
            if (soft != "unlimited") out["limit_mib"] = parse_uint_text(soft, "Max locked memory") / 1048576;
            limit_seen = true;
        }
    }
    {
        std::istringstream input(read_text(root / "proc/self/status"));
        std::string line;
        while (std::getline(input, line)) {
            if (line.rfind("CapEff:", 0) != 0) continue;
            const std::string hex = line.substr(7 + line.substr(7).find_first_not_of(" \t"));
            uint64_t caps = 0;
            const auto r = std::from_chars(hex.data(), hex.data() + hex.size(), caps, 16);
            if (r.ec != std::errc()) throw std::runtime_error("proc/self/status: unreadable CapEff");
            out["cap_ipc_lock"] = ((caps >> 14) & 1) != 0;
            caps_seen = true;
        }
    }
    if (!limit_seen) throw std::runtime_error("proc/self/limits: missing Max locked memory");
    if (!caps_seen) throw std::runtime_error("proc/self/status: missing CapEff");
    return out;
}

/* The amdgpu driver's cap on what every process together pins (radiance's amdgpu_pin_cap): RAM
 * less 1/64 and 1.5 GiB, half of RAM less 1/64 below 3 GiB, unless amdgpu.no_system_mem_limit
 * lifts it. Null cap when lifted, or when the amdgpu module is not loaded. */
Json amdgpu_pin_cap(const fs::path& root, uint64_t mem_total_kb) {
    const fs::path module = root / "sys/module/amdgpu";
    if (!fs::is_directory(module)) return {{"loaded", false}, {"lifted", false}, {"cap_mib", nullptr}};
    std::string v;
    { std::ifstream f(module / "parameters/no_system_mem_limit"); f >> v; }
    const bool lifted = v == "Y" || v == "1";
    Json out = {{"loaded", true}, {"lifted", lifted}, {"cap_mib", nullptr}};
    if (lifted) return out;
    const uint64_t total = mem_total_kb * 1024, reserve = uint64_t(3) << 29;
    const uint64_t cap = total - (total >> 6);
    out["cap_mib"] = (cap < 2 * reserve ? cap >> 1 : cap - reserve) / 1048576;
    return out;
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

const std::vector<OptionField>& engine_option_fields() {
    static const std::vector<OptionField> all = [] {
        static constexpr std::string_view kPrefix = "/spec/processes/engine/";
        static const std::set<std::string> kSections = {"sampling", "limits", "chat", "lora", "vision", "tuning", "diagnostics"};
        std::vector<OptionField> out;
        // Amendment 14: a field that stands in for an environment variable is a resolver key like the flags are. It has no
        // flag; the engine sets the variable from it (rad0009).
        for (const auto& b : v1::kEnvBindings) {
            if (b.path.substr(0, kPrefix.size()) != kPrefix) continue;
            const std::string_view rest = b.path.substr(kPrefix.size());
            const auto slash = rest.find('/');
            if (slash == std::string_view::npos) continue;
            OptionField o;
            o.section = std::string(rest.substr(0, slash));
            o.name = std::string(rest.substr(slash + 1));
            o.key = o.section + "." + o.name;
            o.path = std::string(b.path);
            o.kind = std::string(b.kind);
            o.env = std::string(b.env);
            for (const auto& f : v1::kFields) if (f.path == b.path) o.doc = std::string(f.doc);
            out.push_back(std::move(o));
        }
        for (const auto& f : v1::kFields) {
            if (f.radiance_option.empty() || f.path.substr(0, kPrefix.size()) != kPrefix) continue;
            const std::string_view rest = f.path.substr(kPrefix.size());
            const auto slash = rest.find('/');
            if (slash == std::string_view::npos || !kSections.count(std::string(rest.substr(0, slash)))) continue;
            OptionField o;
            o.section = std::string(rest.substr(0, slash));
            o.name = std::string(rest.substr(slash + 1));
            o.key = o.section + "." + o.name;
            o.path = std::string(f.path);
            o.kind = std::string(f.kind);
            o.doc = std::string(f.doc);
            o.is_switch = f.radiance_switch;
            out.push_back(std::move(o));
        }
        return out;
    }();
    return all;
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
        {"source", "kfd+sysfs+proc+statvfs+cgroup"},
        {"gpus", gpus},
        {"host", {
            {"mem_total_mib", mem.at("MemTotal")},
            {"mem_available_mib", mem.at("MemAvailable")},
            {"swap_total_mib", mem.at("SwapTotal")},
            {"shm_total_mib", (uint64_t(st.f_blocks) * st.f_frsize) / 1048576},
            {"shm_available_mib", (uint64_t(st.f_bavail) * st.f_frsize) / 1048576},
            {"cpu_count", std::max(1u, std::thread::hardware_concurrency())},
            {"memory_cgroups", memory_cgroups(root)},
            {"memlock", memlock(root)},
            {"amdgpu_pin", amdgpu_pin_cap(root, mem.at("MemTotal_kB").get<uint64_t>())}
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
            if (f.fallback.is_null() && !f.derived && !f.optional)
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
            if (f.fallback.is_null() && !f.derived && !f.optional &&
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
    const auto* reserve = field_for("memory.host_reserve_mib");
    root["properties"]["hard_floors"] = {
        {"type", "object"}, {"additionalProperties", false},
        {"description", "site file only; minima can only raise"},
        {"properties", {{"memory", {{"type", "object"}, {"additionalProperties", false},
            {"properties", {{"host_reserve_mib", {{"type", "integer"}, {"minimum", reserve->lo},
                                                  {"maximum", reserve->hi}}}}}}}}}};
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
    if (profile.contains("hard_floors"))
        throw std::runtime_error(options.profile.string() + ": hard_floors: site-only; profile cannot change hard floors");
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
    const ArchInfo engine_arch = arch_info(layout.engine_arch);
    Json engine = choose(facts, layout.engine_arch, used, layout.engine_count);
    Json engine_cards = Json::array();
    for (const auto& g : engine) engine_cards.push_back(g.at("bdf"));
    Json cards = Json::array(), budgets = Json::array(), selected = Json::array(), group_of = Json::array();
    std::vector<bool> budget_derived;   /* no budget-mib: re-derived below from the card and the helper's own footprint */
    for (size_t gi = 0; gi < layout.groups.size(); ++gi) {
        const HelperGroup& group = layout.groups[gi];
        for (const auto& g : choose(facts, group.arch, used, group.count < 0 ? kEveryCard : size_t(group.count))) {
            const uint64_t limit = g.at("vram_total_mib").get<uint64_t>() - std::min<uint64_t>(
                g.at("vram_total_mib").get<uint64_t>(), arch_info(g.at("arch")).reserve_mib);
            cards.push_back(g.at("bdf")); budgets.push_back(group.budget_mib ? group.budget_mib : limit);
            selected.push_back(g); group_of.push_back(gi); budget_derived.push_back(group.budget_mib == 0);
        }
    }
    const bool engine_only = cards.empty();
    auto derived = [&](const std::string& key, const Json& v,
                       const std::string& detail, Json inputs = Json::array()) {
        put(values, provenance, *field_for(key), v, "discovered", detail, inputs);
    };
    derived("layout.engine_cards", engine_cards,
            "KFD and PCI inventory, " + name, Json::array({"layout.name", "facts.gpus"}));
    derived("layout.helper_cards", cards, "KFD and PCI inventory, " + name,
            Json::array({"layout.name", "facts.gpus"}));
    derived("memory.helper_budgets_mib", budgets,
            "engines/radiance/layouts.toml " + name + " (interim; RSC03)",
            Json::array({"layout.name", "facts.gpus"}));
    derived("memory.tail_budget_mib", layout.tail ? layout.groups.back().budget_mib : 0,
            "engines/radiance/layouts.toml " + name + " (interim; RSC03)",
            Json::array({"layout.name"}));
    derived("handoff.placement", layout.tail ? "tail" : "roundrobin",
            "engines/radiance/layouts.toml " + name,
            Json::array({"layout.name"}));
    derived("helper.count", cards.size(), "RSC05 KFD selection",
            Json::array({"layout.helper_cards"}));
    derived("helper.engine_ranks", engine_cards.size(), "layouts.toml engine count",
            Json::array({"layout.engine_cards"}));
    put(values, provenance, *field_for("engine.tp"), engine_cards.size(), "derived",
        "layouts.toml engine count: one rank per engine card", Json::array({"layout.engine_cards"}));
    if (layout.tp_wire)
        put(values, provenance, *field_for("engine.tp_wire"), *layout.tp_wire, "discovered",
            "engines/radiance/layouts.toml " + name + " tp-wire", Json::array({"layout.name"}));
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
    // An omitted budget-mib (RAD49): the EXPERT budget is the card's VRAM less its archs.toml reserve-mib less the helper's own
    // device memory besides its arena (its lanes, pointer banks, slot map and one incoming slot, worker_footprint.h, the
    // definition the worker checks its allocations against), so the owner planner (plan_owners) and the helper's arena agree.
    // Only while no site, profile or --set gave the budgets; the lanes follow the resolved hand-off and power-save settings.
    if (std::find(budget_derived.begin(), budget_derived.end(), true) != budget_derived.end() &&
        provenance.contains("memory.helper_budgets_mib") &&
        provenance.at("memory.helper_budgets_mib").at("origin") == "discovered" && exists(values, "engine.model")) {
        const std::string model = value_at(values, "engine.model").get<std::string>();
        bc::experts::Container container;
        MoeShape cs{};
        if (!container.open(model.c_str()) || !bc::experts::container_shape(container, &cs))
            throw std::runtime_error("memory.helper_budgets_mib: cannot read the model container " + model +
                                     " to derive the helpers' expert budgets");
        namespace fp = r3::worker_footprint;
        const fp::Shape shape{container.layers, cs.experts, cs.hidden, cs.intermediate,
                              cs.top_k ? cs.top_k : r3::moe_shm::kTopK, r3::moe_shm::kActGroup};
        uint64_t largest = 0;
        for (uint32_t l = 0; l < container.layers; ++l)
            for (uint32_t e = 0; e < container.experts; ++e)
                if (container.weight(l, e, 0) && container.weight(l, e, 1)) {
                    const int64_t b = bc::experts::laid_out_bytes(container, true, l, e);
                    if (b <= 0) throw std::runtime_error("memory.helper_budgets_mib: an expert has no device layout size");
                    largest = std::max(largest, uint64_t(b));
                }
        const uint64_t max_rows = value_at(values, "handoff.max_rows").get<uint64_t>();
        fp::Lanes lanes{};
        lanes.small_rows[0] = uint32_t(std::min(value_at(values, "handoff.small_rows").get<uint64_t>(), max_rows));
        lanes.small_count = 1;   /* helper_profile.toml lanes/small_lanes is empty: the one small class */
        lanes.large_rows = uint32_t(max_rows);
        const bool power_save = value_at(values, "helper.power_save").get<bool>();
        Json derived_budgets = value_at(values, "memory.helper_budgets_mib");
        Json inputs = Json::array({"layout.name", "facts.gpus", "engine.model", "handoff.max_rows", "handoff.small_rows",
                                   "helper.power_save"});
        for (size_t i = 0; i < selected.size(); ++i) {
            if (!budget_derived[i]) continue;
            lanes.open = fp::served_lanes(uint32_t(cards.size()), uint32_t(i), uint32_t(engine_cards.size()), 1u, power_save);
            const uint64_t fixed = fp::fixed_bytes(shape, lanes, fp::stride_bound(largest));
            const uint64_t mib = 1ull << 20;
            const uint64_t card = selected[i].at("vram_total_mib").get<uint64_t>() * mib;
            const uint64_t reserve = arch_info(selected[i].at("arch")).reserve_mib * mib;
            const uint64_t budget = fp::expert_budget_bytes(card, reserve, fixed) / mib;
            if (!budget)
                throw std::runtime_error("memory.helper_budgets_mib[" + std::to_string(i) + "]: the " +
                                         selected[i].at("arch").get<std::string>() + " card's VRAM less its reserve does not "
                                         "hold the helper's own lanes, pointer banks and slot map");
            derived_budgets[i] = budget;
        }
        put(values, provenance, *field_for("memory.helper_budgets_mib"), derived_budgets, "derived",
            "card VRAM less archs.toml reserve-mib less the helper's lanes, pointer banks, slot map and incoming slot "
            "(worker_footprint.h)", inputs);
    }
    {
        // The host RAM reserve: calibrated rule parameters for this layout and objective, then the site's floor.
        const Calibration calibration = Calibration::standard(options.calibration_cache);
        const CalibrationMatch at{"", name, value_at(values, "service.objective").get<std::string>()};
        const Calibrated floor = calibration.get("calibration.host_reserve.floor_mib", at);
        const Calibrated fraction = calibration.get("calibration.host_reserve.total_fraction_ppm", at);
        if (!floor.value.is_number_integer() || floor.value.get<int64_t>() < 0 || !fraction.value.is_number_integer() ||
            fraction.value.get<int64_t>() < 0 || fraction.value.get<int64_t>() >= 1000000)
            throw std::runtime_error("calibration.host_reserve: floor_mib must be a whole MiB count and "
                                     "total_fraction_ppm an integer in [0, 1000000)");
        const uint64_t total = facts.at("host").at("mem_total_mib").get<uint64_t>();
        const uint64_t ppm = fraction.value.get<uint64_t>();
        const uint64_t scaled = total / 1000000 * ppm + (total % 1000000 * ppm + 999999) / 1000000;   // ceil, no overflow
        const uint64_t reserve = std::max(floor.value.get<uint64_t>(), scaled);
        const Field& field = *field_for("memory.host_reserve_mib");
        put(values, provenance, field, reserve, "calibrated",
            "max(" + std::to_string(floor.value.get<uint64_t>()) + " MiB, " + std::to_string(ppm) + " ppm x " +
            std::to_string(total) + " MiB MemTotal); " + floor.describe("calibration.host_reserve.floor_mib") +
            "; " + fraction.describe("calibration.host_reserve.total_fraction_ppm"),
            Json::array({"facts.host.mem_total_mib", "calibration.host_reserve.floor_mib",
                         "calibration.host_reserve.total_fraction_ppm"}));
        if (site.contains("hard_floors")) {
            const Json& floors = site.at("hard_floors");
            std::vector<std::pair<std::string, Json>> entries;
            if (!floors.is_object()) throw std::runtime_error(options.site.string() + ": hard_floors must be a table");
            flatten(floors, "hard_floors", entries);
            for (const auto& [key, v] : entries) {
                if (key != "hard_floors.memory.host_reserve_mib")
                    throw std::runtime_error(options.site.string() + ": unknown floor " + key +
                                             " (allowed: hard_floors.memory.host_reserve_mib)");
                validate_value(field, v, options.site.string() + ":hard floor");
                if (v.get<uint64_t>() > total)
                    throw std::runtime_error(options.site.string() + ": " + key + " exceeds detected RAM");
                if (v.get<uint64_t>() > reserve)
                    put(values, provenance, field, v, "site",
                        options.site.string() + ": hard floor raises the calibrated reserve of " +
                        std::to_string(reserve) + " MiB",
                        Json::array({options.site.string() + ":" + key, "calibration.host_reserve.floor_mib",
                                     "calibration.host_reserve.total_fraction_ppm"}));
            }
        }
    }
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
    alias("engine.host_headroom_mib", value_at(values, "memory.host_reserve_mib"),
          "RSC03 memory/reserve projection: one reserve for the plan and the engine", "memory.host_reserve_mib");
    // The engine architecture's plugin (archs.toml), the mailbox plugin when helpers exist, then ref.
    {
        const std::string stem = engine_arch.plugin.substr(0, engine_arch.plugin.rfind(".so"));
        alias("engine.kernels", stem + (engine_only ? "" : ",libr3moe") + ",libref",
              "engines/radiance/archs.toml " + layout.engine_arch + " plugin; launcher ENGINE_KERNELS",
              "layout.engine_cards");
    }

    for (const char* key : {"engine.model", "engine.radiance_home"})
        if (!exists(values, key))
            throw std::runtime_error(std::string("missing required key: ") + key);
    for (const auto& field : schema_fields())
        if (!field.optional && !exists(values, field.key))
            throw std::runtime_error("missing required key: " + field.key +
                                     " (" + field.origin + ")");
    // The handoff files belong to a configuration with helpers; an engine-only one has no use for them.
    for (const char* key : {"handoff.heat", "handoff.histogram"})
        if (!engine_only && !exists(values, key))
            throw std::runtime_error(std::string("missing required key: ") + key + " (" + field_for(key)->origin + ")");
    if (value_at(values, "engine.tp").get<size_t>() > 1 && !exists(values, "engine.tp_wire"))
        throw std::runtime_error("missing required key: engine.tp_wire (a multi-card engine needs its all-reduce transport)");
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
    if (hindex >= std::max<size_t>(hcount, 1))
        throw std::runtime_error("helper.index must be below helper.count");
    if (hcount != cards.size())
        throw std::runtime_error("helper.count must match selected layout helper cards");
    if (!engine_only && !r3::moe_shm::slots_fit(uint32_t(hcount), uint32_t(engine_cards.size())))
        throw std::runtime_error("layout " + name + ": " + std::to_string(hcount) + " helper(s) behind " +
                                 std::to_string(engine_cards.size()) + " engine rank(s) exceed the protocol's " +
                                 std::to_string(r3::moe_shm::kMaxSlots) + " request slots");
    if (value_at(values, "helper.engine_ranks").get<size_t>() !=
        value_at(values, "engine.tp").get<size_t>())
        throw std::runtime_error("helper.engine_ranks must equal engine.tp");
    if (value_at(values, "helper.combine_threads").get<uint64_t>() >
        facts.at("host").at("cpu_count").get<uint64_t>())
        throw std::runtime_error("helper.combine_threads exceeds discovered CPU count");
    if (value_at(values, "layout.engine_cards") != engine_cards ||
        value_at(values, "layout.helper_cards") != cards)
        throw std::runtime_error("layout: selectors must match the discovered named layout; no fallback");

    const Json effective_budgets = value_at(values, "memory.helper_budgets_mib");
    if (effective_budgets.size() != cards.size())
        throw std::runtime_error("memory.helper_budgets_mib must match helper card count");
    for (size_t i = 0; i < selected.size(); ++i) {
        const uint64_t budget = effective_budgets[i].get<uint64_t>();
        const uint64_t total = selected[i].at("vram_total_mib").get<uint64_t>();
        const uint64_t reserve = arch_info(selected[i].at("arch")).reserve_mib;
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
    c.layout = {l.at("name"), l.at("engine_cards").get<std::vector<std::string>>(),
                l.at("helper_cards").get<std::vector<std::string>>()};
    const auto& m = values.at("memory");
    c.memory = {m.at("host_pool_mib"), m.at("host_kv_mib"), m.at("host_reserve_mib"),
                m.at("tail_budget_mib"),
                m.at("helper_budgets_mib").get<std::vector<uint64_t>>()};
    const auto& a = values.at("handoff");
    c.handoff = {a.at("shm"), a.value("heat", std::string()), a.value("histogram", std::string()),   // empty: an engine-only layout has none
                
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
                e.at("max_model_len"), e.at("host_pool_mib"), e.at("prefix_cache_mib"),
                e.at("host_headroom_mib")};
    if (e.contains("tp_wire")) c.engine.tp_wire = e.at("tp_wire").get<std::string>();
    c.engine.no_prefix_cache = e.contains("no_prefix_cache");
    if (e.contains("gpu_headroom_mib")) c.engine.gpu_headroom_mib = e.at("gpu_headroom_mib").get<uint32_t>();
    Json process_helpers = Json::array();
    for (size_t i = 0; i < cards.size(); ++i) {
        Json worker = h;
        worker["index"] = i;
        const HelperGroup& group = layout.groups[group_of[i].get<size_t>()];
        const ArchInfo arch = arch_info(selected[i].at("arch"));
        const bool tail_card = group.tail;
        const uint64_t share = tail_card ? 0 : group.share;
        process_helpers.push_back({
            {"config", worker}, {"bdf", cards[i]},
            {"arch", selected[i].at("arch")}, {"budget_mib", effective_budgets[i]},
            {"hip_ordinal", selected[i].at("hip_ordinal")},
            {"plugin_file", arch.plugin},
            {"formats", arch.formats}, {"rows_kernel", arch.rows_kernel},
            {"placement", tail_card ? "tail" : "head"}, {"share", share},
            {"provenance", {{"origin", "archs.toml + layouts.toml + KFD"},
                            {"inputs", Json::array({"layout.name", "facts.gpus"})},
                            {"lifecycle", "startup-only"}}}
        });
    }
    // The resolver's own record (--explain --json, --validate's hash, the manifest writer's helper list): not a file any
    // process reads. The serving manifest v1 (manifest_writer.cpp) is the one document processes load.
    r.manifest = {{"values", values}, {"provenance", provenance}, {"facts", facts}, {"process_helpers", process_helpers}};
    r.hash = sha256_hex(r.manifest.dump());
    return r;
}
} // namespace bc::serving
