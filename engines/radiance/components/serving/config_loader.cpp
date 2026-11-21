#include "config_loader.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <set>
#include <stdexcept>

namespace bc::serving {
namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {
// The settings tables: the resolver's key prefixes a config may set.
const std::set<std::string> kSettingsTables = {"service", "engine", "tuning", "memory", "helper", "handoff", "kernels"};
}  // namespace

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
            out[underscored(key)] = value;
        }
        effective[name] = out;
    }
    return config_as_layout(effective, model_root, c.yaml_path);
}

} // namespace bc::serving
