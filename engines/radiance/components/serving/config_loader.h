#pragma once
/* RSC12: readable serving configs. One TOML file per model holds named configs; a variant names its base and states only
 * what it changes. A config is an input to the existing resolver, not a second one: config_as_layout() turns its effective
 * values into the layout table the resolver already reads (cards, wire, helper groups, settings), and the settings keep
 * the resolver's own names (service.*, engine.*, tuning.*, ...), so their checks are the resolver's.
 *
 *   schema = 1
 *   [config.NAME]                 description (required), base (a variant), model, ranks, tp_wire
 *   ranks = [{ arch = "gfx1100", count = 2 }, { arch = "gfx1201", count = 1 }]    engine ranks in order, by architecture
 *   [[config.NAME.helpers]]       helper card groups: arch, count (number or "all"), placement, budget_mib, share
 *   [config.NAME.service] / .engine / .tuning / .memory / .helper / .handoff      resolver settings
 *
 * A variant inherits every value of its base: tables merge key by key, a scalar or a list (ranks, helpers) replaces.
 * Every config is strict: an unknown key, a setting outside a settings table, a missing or cyclic base, or a name defined
 * twice is refused, naming the file and the key. */
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace bc::serving {

/* The config files: `given` (--configs), else BIGCHERRY_CONFIGS (':'-separated); a directory gives its *.toml in name
 * order. Empty when neither names any: configs are optional and the resolver then reads layouts only. */
std::vector<std::filesystem::path> config_sources(const std::vector<std::filesystem::path>& given);

struct ConfigSet {
    nlohmann::json config = nlohmann::json::object();   // name -> effective table (variants applied, no `base`)
    nlohmann::json source = nlohmann::json::object();   // name -> file that defines it
    nlohmann::json origin = nlohmann::json::object();   // name -> {dotted key -> the config whose file set it}
    nlohmann::json chain = nlohmann::json::object();    // name -> [name, base, base's base, ...]
};
ConfigSet load_configs(const std::vector<std::filesystem::path>& files);

/* The layout table the resolver reads for one effective config ([layout.NAME] in a layouts file). */
nlohmann::json config_as_layout(const nlohmann::json& effective);

} // namespace bc::serving
