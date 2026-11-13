#include "config_loader.h"

#include "third_party/toml.hpp"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <set>
#include <stdexcept>

namespace bc::serving {
namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

// The settings tables: the resolver's key prefixes a config may set. Everything else in a config is its recipe.
const std::set<std::string> kSettingsTables = {"service", "engine", "tuning", "memory", "helper", "handoff"};
const std::set<std::string> kRecipeKeys = {"description", "base", "model", "ranks", "tp_wire", "helpers"};
const std::set<std::string> kRankKeys = {"arch", "count"};
const std::set<std::string> kHelperKeys = {"arch", "count", "placement", "budget_mib", "share"};

Json to_json(const toml::node& node) {
    if (auto t = node.as_table()) {
        Json j = Json::object();
        for (const auto& [k, v] : *t) j[std::string(k.str())] = to_json(v);
        return j;
    }
    if (auto a = node.as_array()) {
        Json j = Json::array();
        for (const auto& item : *a) j.push_back(to_json(item));
        return j;
    }
    if (auto b = node.as_boolean()) return b->get();   // asked first: a boolean also reads as an integer
    if (auto s = node.value<std::string>()) return *s;
    if (auto n = node.value<int64_t>()) return *n;
    if (auto d = node.value<double>()) return *d;
    throw std::runtime_error("unsupported TOML value");
}

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
    if (!c.contains("description")) throw std::runtime_error(where + "description is required (what this config is for)");
    if (c.contains("ranks")) check_list_of_tables(c.at("ranks"), kRankKeys, where + "ranks");
    if (c.contains("helpers")) check_list_of_tables(c.at("helpers"), kHelperKeys, where + "helpers");
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
                o = o.key().rfind(key + ".", 0) == 0 ? origin.erase(o) : std::next(o);
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

ConfigSet load_configs(const std::vector<fs::path>& files) {
    Json written = Json::object(), source = Json::object();
    for (const auto& f : files) {
        Json doc;
        try { doc = to_json(toml::parse_file(f.string())); }
        catch (const toml::parse_error& e) { throw std::runtime_error(f.string() + ": TOML parse error: " + e.what()); }
        for (auto it = doc.begin(); it != doc.end(); ++it)
            if (it.key() != "schema" && it.key() != "config")
                throw std::runtime_error(f.string() + ": unknown top-level key " + it.key() + " (schema, config)");
        if (doc.value("schema", 0) != 1) throw std::runtime_error(f.string() + ": schema = 1 is required");
        if (!doc.contains("config")) continue;
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

Json config_as_layout(const Json& c) {
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
    if (!settings.empty()) l["settings"] = settings;
    return l;
}

} // namespace bc::serving
