#pragma once
// Test-only fixtures for the engine-free resolver tests. Production reads a config from its native YAML through the engine's
// inspector; a card-free test has no engine, and bc-serve has no YAML parser of its own, so a test hands the resolver the
// InspectedConfig directly. A Preset is named after the shipped engines/radiance/configs/NAME.yaml it copies and is GENERATED
// from it (gen_fixture_presets.py; a test fails when the generated file drifts). A test that needs another shape builds a
// Preset by hand.
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "serving.h"

namespace bc::serving::fixture {

struct Preset {
    std::string name, description, model;
    std::optional<std::string> tp_wire;
    std::vector<XRank> ranks;
    std::vector<XHelperGroup> helpers;
    nlohmann::json service = nlohmann::json::object(), memory = nlohmann::json::object(),
                   handoff = nlohmann::json::object(), helper = nlohmann::json::object(),
                   kernels = nlohmann::json::object();
    // The YAML's native model options as resolver --set settings (service.max_context=..., tuning.tp_ownership=...).
    std::vector<std::string> sets;
};

inline Preset preset(const std::string& name) {
    struct Entry {
        const char* name;
        Preset (*make)();
    };
    static const Entry table[] = {
#include "test_config_presets.inc"
    };
    for (const Entry& entry : table)
        if (name == entry.name) return entry.make();
    throw std::runtime_error("no test preset named " + name + " (one per engines/radiance/configs/NAME.yaml)");
}

inline bool has_preset(const std::string& name) {
    try {
        (void)preset(name);
        return true;
    } catch (const std::runtime_error&) {
        return false;
    }
}

// What the inspector would hand the resolver for this preset. `configs` is the directory its YAML would live in (the
// bundle's @bundle/heat/... seeds resolve against it).
inline InspectedConfig inspected(const Preset& p, const std::filesystem::path& configs = "/fixture/configs") {
    InspectedConfig c;
    c.name = p.name;
    c.yaml_path = configs / (p.name + ".yaml");
    c.sha256 = std::string(64, '0');
    c.model = p.model;
    c.tp_wire = p.tp_wire;
    c.x.description = p.description;
    c.x.ranks = p.ranks;
    c.x.helpers = p.helpers;
    c.x.service = p.service;
    c.x.memory = p.memory;
    c.x.handoff = p.handoff;
    c.x.helper = p.helper;
    c.x.kernels = p.kernels;
    return c;
}

// Selects the preset for the resolver: options.inspected, its model-option settings ahead of the test's own --set values
// (a test's setting overrides), and layout.name.
inline void use(Options& options, const Preset& p, const std::filesystem::path& configs = "/fixture/configs") {
    options.inspected = inspected(p, configs);
    std::vector<std::string> sets = p.sets;
    sets.push_back("layout.name=" + nlohmann::json(p.name).dump());
    sets.insert(sets.end(), options.sets.begin(), options.sets.end());
    options.sets = std::move(sets);
}
inline void use(Options& options, const std::string& name, const std::filesystem::path& configs = "/fixture/configs") {
    use(options, preset(name), configs);
}

}  // namespace bc::serving::fixture
