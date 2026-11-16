#pragma once
#include "ledger.h"
#include "serving_types.h"   // bc::serving::v2::EnginePlan, the generated v1 type

#include <filesystem>
#include <optional>
#include <string>
#include <vector>
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
 * the engine arguments the launcher passes through verbatim (they can change the plan, e.g. --kv-resident), and the
 * planning inputs: every engine.* value except the memory outputs the plan itself feeds (host pool, host KV tier,
 * host headroom), every tuning.* value (e.g. tp_ownership), plus layout.name. Any change is a miss. */
nlohmann::json plan_key_inputs(const nlohmann::json& values, const nlohmann::json& facts, const std::string& model_id,
                               const std::string& engine_id, const std::string& engine_args);
std::string plan_key(const nlohmann::json& key_inputs);
// Calibration-store key: the plan-key inputs without engine_build. What calibration measures (copy and lane times,
// routing noise, drift) belongs to the cards, layout, model and engine settings, not to the pinned engine commit, so a
// record carries across pin bumps; a change of any remaining input (or of the record schema) requires recalibrating.
std::string calibration_key(const nlohmann::json& key_inputs);

/* rad0011's plan (radiance --plan-memory) through the shared v1 decoder, plus what a single plan must satisfy by itself:
 * each host consumer once, and the checkpoint floor within the want. Throws naming `where` and the decoder's path. */
v2::EnginePlan decode_engine_plan(const nlohmann::json& plan, const std::string& where);

/* A plan or cache entry through the shared strict parser (size limit, UTF-8, duplicate members refused, depth bound) before
 * any decoding, as the engine reads it. Throws naming `source`. */
nlohmann::json parse_plan_json(const std::string& text, const std::string& source);

/* The host memory the engine's neighbours hold while it starts, measured in the same measuring start as the plan (never a
 * constant): the helpers' resident anonymous and shared memory with the shared segment counted once, and what the engine's
 * start-up took from the host before plan() ran (the fall of MemAvailable, before the pools, the n-gram table and the
 * checkpoint store are allocated). The host
 * ledger counts both as pageable consumers, so the checkpoint slots follow what is really free at start. MemAvailable is
 * read fresh every time a manifest is written; only these two figures are cached. Both are positive. */
struct HostResident {
    uint64_t helpers_mib = 0, engine_mib = 0;
};

struct CachedPlan {
    v2::EnginePlan plan;           // decoded by the shared v1 codec
    nlohmann::json plan_json;   // as rad0011 wrote it
    nlohmann::json rank_demand; // the per-rank GPU demand rad0011 wrote with it (--plan-memory-ranks)
    std::string key, measured_at;
    HostResident resident;
};

/* What a plan must agree with in the configuration it is stored for (values, as resolve() returns them): the host pool it
 * echoes, and the expected host consumer list (the n-gram table exactly when it is placed in RAM; embedding and host
 * arenas allowed; nothing else). Throws naming the disagreement. */
void check_plan_for_configuration(const nlohmann::json& values, const v2::EnginePlan& plan);

/* The entry for `key_inputs` under `dir`, or nothing on a miss. An entry whose recorded key or key inputs differ, whose
 * version is unknown or whose plan fails the shared v1 decode (serving_codec) is refused (an error), never treated as a miss. */
std::optional<CachedPlan> plan_cache_lookup(const std::filesystem::path& dir, const nlohmann::json& key_inputs);

/* Records a measured plan (rad0011's JSON, decoded by the shared v1 codec first) under its key. Written to a temporary
 * file and renamed into place. */
CachedPlan plan_cache_store(const std::filesystem::path& dir, const nlohmann::json& key_inputs,
                            const nlohmann::json& plan_json, const nlohmann::json& rank_demand,
                            const HostResident& resident);

}  // namespace bc::serving
