#include "config_loader.h"

#include "third_party/toml.hpp"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <functional>
#include <set>
#include <stdexcept>

namespace bc::serving {
namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

// The settings tables: the resolver's key prefixes a config may set. Everything else in a config is its recipe.
const std::set<std::string> kSettingsTables = {"service", "engine", "tuning", "memory", "helper", "handoff", "kernels"};
const std::set<std::string> kRecipeKeys = {"description", "base", "model", "ranks", "tp_wire", "helpers"};
const std::set<std::string> kRankKeys = {"arch", "count"};
const std::set<std::string> kHelperKeys = {"arch", "count", "placement", "budget_mib", "share"};

// Exact TOML types: an integer stays an integer and a float a float (never converted), so a count of 2.5 or a schema of
// 1.0 is refused by the checks below instead of being truncated. `where` names the key for errors.
Json to_json(const toml::node& node, const std::string& where) {
    if (auto t = node.as_table()) {
        Json j = Json::object();
        for (const auto& [k, v] : *t) j[std::string(k.str())] = to_json(v, where + "." + std::string(k.str()));
        return j;
    }
    if (auto a = node.as_array()) {
        Json j = Json::array();
        size_t i = 0;
        for (const auto& item : *a) j.push_back(to_json(item, where + "[" + std::to_string(i++) + "]"));
        return j;
    }
    if (auto b = node.as_boolean()) return b->get();
    if (auto s = node.as_string()) return s->get();
    if (auto n = node.as_integer()) return n->get();
    if (auto d = node.as_floating_point()) return d->get();
    throw std::runtime_error(where + ": unsupported TOML value (dates and times are not settings)");
}

bool is_count(const Json& v) { return v.is_number_integer() && v.get<int64_t>() >= 1; }

void check_list_of_tables(const Json& v, const std::set<std::string>& keys, const std::string& where) {
    if (!v.is_array() || v.empty()) throw std::runtime_error(where + " is a non-empty list of tables");
    for (const Json& t : v) {
        if (!t.is_object()) throw std::runtime_error(where + " is a list of tables");
        for (auto it = t.begin(); it != t.end(); ++it)
            if (!keys.count(it.key())) {
                std::string allowed;
                for (const auto& k : keys) allowed += (allowed.empty() ? "" : ", ") + k;
                throw std::runtime_error(where + ": unknown key " + it.key() + " (" + allowed + ")");
            }
        if (!t.contains("arch") || !t.at("arch").is_string()) throw std::runtime_error(where + ": every group names its arch");
    }
}

void check_ranks(const Json& ranks, const std::string& where) {
    for (const Json& r : ranks)
        if (r.contains("count") && !is_count(r.at("count")))
            throw std::runtime_error(where + ": count is a whole number of at least 1 (" + r.at("count").dump() + ")");
}

void check_helpers(const Json& helpers, const std::string& where) {
    for (const Json& h : helpers) {
        if (!h.contains("count") || !(is_count(h.at("count")) || h.at("count") == "all"))
            throw std::runtime_error(where + ": count is required, a whole number of at least 1 or \"all\"");
        if (!h.contains("placement") || !(h.at("placement") == "head" || h.at("placement") == "tail"))
            throw std::runtime_error(where + ": placement is required, \"head\" or \"tail\"");
        if (h.contains("budget_mib") && !is_count(h.at("budget_mib")))
            throw std::runtime_error(where + ": budget_mib is a whole number of MiB (leave it out to derive it)");
        if (h.contains("share") && !(h.at("share").is_number_integer() && h.at("share").get<int64_t>() >= 0))
            throw std::runtime_error(where + ": share is a whole number of at least 0");
    }
}

// One config as written (before variants): its keys are the recipe keys and the settings tables, nothing else.
void check_written(const Json& c, const std::string& where) {
    if (!c.is_object()) throw std::runtime_error(where + "a config is a table");
    for (auto it = c.begin(); it != c.end(); ++it) {
        const std::string& k = it.key();
        if (kSettingsTables.count(k)) {
            if (!it.value().is_object())
                throw std::runtime_error(where + k + " is a table of " + k + ".* settings");
            continue;
        }
        if (!kRecipeKeys.count(k))
            throw std::runtime_error(where + "unknown key " + k +
                                     " (description, base, model, ranks, tp_wire, helpers, or a settings table: service, "
                                     "engine, tuning, memory, helper, handoff)");
    }
    for (const char* s : {"description", "base", "model", "tp_wire"})
        if (c.contains(s) && !c.at(s).is_string()) throw std::runtime_error(where + s + " is a string");
    if (c.contains("model")) {   // a relative model stays under the model root: no `..` component
        const fs::path m = c.at("model").get<std::string>();
        if (m.empty()) throw std::runtime_error(where + "model is empty");
        for (const auto& part : m)
            if (part == "..") throw std::runtime_error(where + "model " + m.string() + " leaves the model root (`..`)");
    }
    if (!c.contains("description")) throw std::runtime_error(where + "description is required (what this config is for)");
    if (c.contains("ranks")) {
        check_list_of_tables(c.at("ranks"), kRankKeys, where + "ranks");
        check_ranks(c.at("ranks"), where + "ranks");
    }
    if (c.contains("helpers")) {
        check_list_of_tables(c.at("helpers"), kHelperKeys, where + "helpers");
        check_helpers(c.at("helpers"), where + "helpers");
    }
    if (c.contains("engine") && c.at("engine").contains("model"))
        throw std::runtime_error(where + "engine.model: give the model as `model`, the config's own key");
}

// Tables merge key by key; anything else (a scalar, a list) replaces. `who` records the config that set each leaf.
void merge(Json& into, const Json& from, Json& origin, const std::string& prefix, const std::string& who) {
    for (auto it = from.begin(); it != from.end(); ++it) {
        const std::string key = prefix.empty() ? it.key() : prefix + "." + it.key();
        if (it.value().is_object() && into.contains(it.key()) && into.at(it.key()).is_object()) {
            merge(into[it.key()], it.value(), origin, key, who);
        } else {
            into[it.key()] = it.value();
            // a replaced table or list: drop the origins recorded under it, then record its leaves
            for (auto o = origin.begin(); o != origin.end();)
                o = o.key() == key || o.key().rfind(key + ".", 0) == 0 ? origin.erase(o) : std::next(o);
            if (it.value().is_object()) {
                Json sub = Json::object();
                merge(sub, it.value(), origin, key, who);
            } else {
                origin[key] = who;
            }
        }
    }
}

} // namespace

std::vector<fs::path> config_sources(const std::vector<fs::path>& given_in) {
    std::vector<fs::path> given = given_in;
    if (given.empty())
        if (const char* env = std::getenv("BIGCHERRY_CONFIGS"); env && *env) {
            const std::string s(env);
            for (size_t at = 0; at <= s.size();) {
                const size_t colon = std::min(s.find(':', at), s.size());
                if (colon > at) given.emplace_back(s.substr(at, colon - at));
                at = colon + 1;
            }
        }
    std::vector<fs::path> files;
    for (const auto& g : given) {
        if (fs::is_directory(g)) {
            std::vector<fs::path> in_dir;
            for (const auto& e : fs::directory_iterator(g))
                if (e.is_regular_file() && e.path().extension() == ".toml") in_dir.push_back(e.path());
            std::sort(in_dir.begin(), in_dir.end());
            if (in_dir.empty()) throw std::runtime_error("config: " + g.string() + " holds no *.toml config file");
            files.insert(files.end(), in_dir.begin(), in_dir.end());
        } else if (fs::is_regular_file(g)) {
            files.push_back(g);
        } else {
            throw std::runtime_error("config: " + g.string() + " is neither a config file nor a directory");
        }
    }
    return files;
}

std::filesystem::path config_yaml_for(const std::string& name, const std::vector<std::filesystem::path>& given) {
    std::vector<fs::path> dirs = given;
    if (dirs.empty())
        if (const char* env = std::getenv("BIGCHERRY_CONFIGS"))
            for (std::stringstream parts(env); parts.good();) {
                std::string part;
                std::getline(parts, part, ':');
                if (!part.empty()) dirs.emplace_back(part);
            }
    if (name.empty() || name.find('/') != std::string::npos) return {};
    for (const fs::path& dir : dirs) {
        const fs::path candidate = fs::is_directory(dir) ? dir / (name + ".yaml") : fs::path{};
        if (!candidate.empty() && fs::is_regular_file(candidate)) return candidate;
    }
    return {};
}

ConfigSet load_configs(const std::vector<fs::path>& files) {
    Json written = Json::object(), source = Json::object();
    for (const auto& f : files) {
        Json doc;
        try { doc = to_json(toml::parse_file(f.string()), f.string()); }
        catch (const toml::parse_error& e) { throw std::runtime_error(f.string() + ": TOML parse error: " + e.what()); }
        for (auto it = doc.begin(); it != doc.end(); ++it)
            if (it.key() != "schema" && it.key() != "config")
                throw std::runtime_error(f.string() + ": unknown top-level key " + it.key() + " (schema, config)");
        if (!doc.contains("schema") || !doc.at("schema").is_number_integer() || doc.at("schema") != 1)
            throw std::runtime_error(f.string() + ": schema = 1 is required");
        if (!doc.contains("config")) continue;
        if (!doc.at("config").is_object()) throw std::runtime_error(f.string() + ": config is a table of named configs");
        for (auto it = doc.at("config").begin(); it != doc.at("config").end(); ++it) {
            if (written.contains(it.key()))
                throw std::runtime_error("config " + it.key() + " is defined in both " + source[it.key()].get<std::string>() +
                                         " and " + f.string());
            check_written(it.value(), f.string() + " [config." + it.key() + "]: ");
            written[it.key()] = it.value();
            source[it.key()] = f.string();
        }
    }
    ConfigSet out;
    out.source = source;
    std::set<std::string> visiting;
    std::function<void(const std::string&)> effective = [&](const std::string& name) {
        if (out.config.contains(name)) return;
        const Json& own = written.at(name);
        const std::string where = source.at(name).get<std::string>() + " [config." + name + "]: ";
        Json value = Json::object(), origin = Json::object(), chain = Json::array({name});
        if (own.contains("base")) {
            const std::string base = own.at("base");
            if (!written.contains(base)) throw std::runtime_error(where + "base " + base + " is not a defined config");
            if (visiting.count(base)) throw std::runtime_error(where + "base " + base + " leads back to " + name + " (a cycle)");
            visiting.insert(name);
            effective(base);
            visiting.erase(name);
            value = out.config.at(base);
            origin = out.origin.at(base);
            for (const Json& c : out.chain.at(base)) chain.push_back(c);
        }
        Json mine = own;
        mine.erase("base");
        merge(value, mine, origin, "", name);
        for (const char* required : {"model", "ranks"})
            if (!value.contains(required)) throw std::runtime_error(where + required + " is required (here or in its base)");
        out.config[name] = value;
        out.origin[name] = origin;
        out.chain[name] = chain;
    };
    for (auto it = written.begin(); it != written.end(); ++it) effective(it.key());
    return out;
}

Json config_as_layout(const Json& c, const fs::path& model_root, const fs::path& heat_source) {
    Json l = Json::object();
    l["description"] = c.at("description");
    l["model"] = c.at("model");
    l["engine"] = c.at("ranks");
    if (c.contains("tp_wire")) l["tp-wire"] = c.at("tp_wire");
    if (c.contains("helpers")) {
        Json groups = Json::array();
        for (const Json& h : c.at("helpers")) {
            Json g = h;
            if (g.contains("budget_mib")) {
                g["budget-mib"] = g.at("budget_mib");
                g.erase("budget_mib");
            }
            groups.push_back(g);
        }
        l["helper"] = groups;
    }
    Json settings = Json::object();
    for (const auto& t : kSettingsTables)
        if (c.contains(t) && !c.at(t).empty()) settings[t] = c.at(t);
    // @bundle/heat/... is a config-only reference to the runtime data
    // sibling of prefix/configs. Resolve it from the defining config's file,
    // not the caller's cwd; a site/profile/--set may still override the seed.
    if (settings.contains("handoff") && settings.at("handoff").contains("heat") &&
        settings.at("handoff").at("heat").is_string()) {
        const std::string seed = settings.at("handoff").at("heat").get<std::string>();
        if (seed.starts_with("@bundle/")) {
            const fs::path rel(seed.substr(std::string("@bundle/").size()));
            if (heat_source.empty() || rel.is_absolute() || rel.empty() ||
                rel.begin() == rel.end() || *rel.begin() != "heat" ||
                std::any_of(rel.begin(), rel.end(), [](const auto& part) {
                    return part == ".." || part == ".";
                }) || rel.filename().empty() || rel.filename() == "heat")
                throw std::runtime_error("config handoff.heat: unsafe @bundle/heat path");
            settings["handoff"]["heat"] =
                (fs::absolute(heat_source).parent_path().parent_path() / rel).lexically_normal().string();
        }
    }
    // The model is the config's lowest setting (engine.model; a site, profile or --set still wins). A relative path is
    // under the model root; without one the config cannot be served, which the resolver says when it is selected.
    const fs::path model = c.at("model").get<std::string>();
    if (model.is_absolute()) settings["engine"]["model"] = model.string();
    else if (!model_root.empty()) settings["engine"]["model"] = (model_root / model).string();
    else l["model_unresolved"] = model.string();
    if (!settings.empty()) l["settings"] = settings;
    return l;
}

// An inspected config (its native YAML, parsed by the engine) as the same effective table a catalogue entry gave, so one
// builder (config_as_layout) and the resolver's own checks serve both. `@bundle/heat/...` resolves against the YAML's
// file, whose directory is the bundle's configs/.
Json inspected_as_layout(const InspectedConfig& c, const fs::path& model_root) {
    const XBigcherry& x = c.x;
    Json effective = Json::object();
    effective["description"] = x.description;
    effective["model"] = c.model;
    Json ranks = Json::array();
    for (const XRank& r : x.ranks) ranks.push_back({{"arch", r.arch}, {"count", r.count}});
    effective["ranks"] = ranks;
    if (c.tp_wire) effective["tp_wire"] = *c.tp_wire;
    if (!x.helpers.empty()) {
        Json helpers = Json::array();
        for (const XHelperGroup& h : x.helpers) {
            Json g = {{"arch", h.arch}, {"placement", h.placement},
                      {"count", h.count < 0 ? Json("all") : Json(h.count)}};
            if (h.budget_mib) g["budget_mib"] = *h.budget_mib;
            if (h.share) g["share"] = *h.share;
            helpers.push_back(g);
        }
        effective["helpers"] = helpers;
    }
    const auto underscored = [](std::string key) {
        std::replace(key.begin(), key.end(), '-', '_');
        return key;
    };
    const std::pair<const char*, const Json*> tables[] = {{"service", &x.service}, {"memory", &x.memory},
                                                           {"handoff", &x.handoff}, {"helper", &x.helper},
                                                           {"kernels", &x.kernels}};
    for (const auto& [name, table] : tables) {
        if (table->empty()) continue;
        Json out = Json::object();
        for (const auto& [key, value] : table->items()) {
            if (key == "hard-caps" || key == "hard-floors")
                throw std::runtime_error(c.yaml_path.string() + ": x-bigcherry.memory." + key +
                                         " is an admission policy of a site or profile, not of a config file");
            out[underscored(key)] = value;
        }
        effective[name] = out;
    }
    return config_as_layout(effective, model_root, c.yaml_path);
}

} // namespace bc::serving
