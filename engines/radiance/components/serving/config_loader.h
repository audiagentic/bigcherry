#pragma once
/* Serving configs are native Radiance YAML files (engines/radiance/configs/NAME.yaml) with a closed x-bigcherry section. The
 * engine's own parser reads one (the inspector, session.h); this turns what it returned into the layout table the resolver
 * reads (cards, wire, helper groups, settings), so the resolver's checks stay the resolver's. There is no other config source:
 * the RSC12 TOML catalogue and the layout files are gone. */
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "serving.h"

namespace bc::serving {

/* NAME.yaml in the first of the config sources (`given`, else BIGCHERRY_CONFIGS) that holds it; empty when none does. */
std::filesystem::path config_yaml_for(const std::string& name, const std::vector<std::filesystem::path>& given);

/* The layout table the resolver reads for one effective config (its recipe keys and settings tables). The config's model
 * becomes its lowest engine.model setting: an absolute path as given, a relative one under `model_root`; with no root a
 * relative model is left unresolved (`model_unresolved`), and selecting that config is refused. */
nlohmann::json config_as_layout(const nlohmann::json& effective, const std::filesystem::path& model_root = {},
                                const std::filesystem::path& heat_source = {});

/* The same, for a config read from its native YAML (an InspectedConfig): the resolver's only source of configs. */
nlohmann::json inspected_as_layout(const InspectedConfig& config, const std::filesystem::path& model_root = {});

} // namespace bc::serving
