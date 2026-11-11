#pragma once
// S13 P1 producer: assembles strictly sourced engine/model, owner, host and GPU observations.
#include "serving.h"
#include <filesystem>
#include <nlohmann/json.hpp>

namespace bc::serving::v2 {
nlohmann::json produce_p1_inputs(const bc::serving::Resolved& resolved,
                                  const nlohmann::json& facts,
                                  const std::filesystem::path& sys_root,
                                  const nlohmann::json& request);
// The same from sources in hand: engine_plan and rank_demand as JSON (the plan cache entry), the binaries, the retention
// target, the host-ledger margin and the run id. bc-serve --write-manifest measures with it at every manifest it writes.
nlohmann::json produce_p1(const bc::serving::Resolved& resolved,
                          const nlohmann::json& facts,
                          const std::filesystem::path& sys_root,
                          const nlohmann::json& sources);
}
