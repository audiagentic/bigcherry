#pragma once
#include "ledger.h"

#include <filesystem>
#include <optional>
#include <string>
#include <nlohmann/json.hpp>

namespace bc::serving {

/* The measured-plan cache (RSC03 slice 1b): the engine's own memory plan (radiance --plan-memory, rad0011), measured
 * once per configuration on its engine card and reused by every resolve that has the same key. A miss is never filled
 * with a guess: bc-serve --explain and --dry-run report it, and the serve path measures. */

/* The container's identity without reading its weights: sha256 over its size and its header, directory, plane and
 * encoding tables (the bytes every expert size and weight placement is computed from). */
std::string container_identity(const std::filesystem::path& model);

/* The engine build's identity: sha256 over the engine binary and every regular file under its runtime home (plugins,
 * kernels, helpers), by relative path and content. */
std::string engine_identity(const std::filesystem::path& binary, const std::filesystem::path& radiance_home);

/* What the key covers: the cache version, the model and engine identities, the engine card's architecture and VRAM, and
 * the planning inputs: every engine.* value except the memory outputs the plan itself feeds (host pool, host KV tier,
 * host headroom), plus layout.name. Any change is a miss. */
nlohmann::json plan_key_inputs(const nlohmann::json& values, const nlohmann::json& facts, const std::string& model_id,
                               const std::string& engine_id);
std::string plan_key(const nlohmann::json& key_inputs);

struct CachedPlan {
    EnginePlan plan;
    nlohmann::json plan_json;   // as rad0011 wrote it
    std::string key, measured_at;
};

/* The entry for `key_inputs` under `dir`, or nothing on a miss. An entry whose recorded key or key inputs differ, whose
 * version is unknown or whose plan fails the strict v1 parse is refused (an error), never treated as a miss. */
std::optional<CachedPlan> plan_cache_lookup(const std::filesystem::path& dir, const nlohmann::json& key_inputs);

/* Records a measured plan (rad0011's JSON, checked by the strict parse first) under its key. Written to a temporary
 * file and renamed into place. */
CachedPlan plan_cache_store(const std::filesystem::path& dir, const nlohmann::json& key_inputs,
                            const nlohmann::json& plan_json);

}  // namespace bc::serving
