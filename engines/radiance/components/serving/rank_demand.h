#pragma once

#include <cstdint>
#include <optional>
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
    // Resolved by the actual engine placement planner during --plan-memory.
    // Missing only in older explicit-placement records; auto must not guess.
    std::optional<std::string> ngram_placement;
    std::vector<EngineRankGpuDemand> gpu;
};

/* Decode rad0011's exact per-engine-rank demand wire and bind it to selected BDF order. */
// Seal the actual native engine tier. An auto intent may be resolved only by
// the engine's authenticated --plan-memory rank receipt, never inferred by
// the writer from a default or a BigCherry host-memory heuristic.
std::string admitted_ngram_placement(const EngineRankDemand& measured, bool has_pinned_ngram,
                                     const std::string& requested);

EngineRankDemand decode_engine_rank_demand(const nlohmann::json& record,
                                           const std::vector<std::string>& ordered_bdfs,
                                           const std::string& where);

}  // namespace bc::serving
