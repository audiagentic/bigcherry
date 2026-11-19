#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace bc::serving {

struct EngineRankGpuDemand {
    std::string bdf;
    int64_t required_bytes = 0;
    int64_t fixed_bytes = 0;
    int64_t claimable_bytes = 0;
    std::string evidence;
};

struct EngineRankDemand {
    int64_t resident_other_bytes = 0;
    std::vector<EngineRankGpuDemand> gpu;
};

/* Decode rad0011's exact per-engine-rank demand wire and bind it to selected BDF order. */
EngineRankDemand decode_engine_rank_demand(const nlohmann::json& record,
                                           const std::vector<std::string>& ordered_bdfs,
                                           const std::string& where);

}  // namespace bc::serving
