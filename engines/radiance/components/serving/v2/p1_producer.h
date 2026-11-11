#pragma once
// Inactive S13 P1-b producer: assembles strictly sourced engine/model, owner,
// host and GPU observations. Production v1 serving does not invoke this path.
#include "serving.h"
#include <filesystem>
#include <nlohmann/json.hpp>

namespace bc::serving::v2 {
nlohmann::json produce_p1_inputs(const bc::serving::Resolved& resolved,
                                  const nlohmann::json& facts,
                                  const std::filesystem::path& sys_root,
                                  const nlohmann::json& request);
}
