#pragma once
#include "ledger.h"
#include "serving_types.h"   // bc::serving::v1::EnginePlan, the generated v1 type

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
 * the engine arguments the launcher passes through verbatim (they can change the plan, e.g. --kv-resident), and the engine arguments the launcher passes through verbatim (they can change the plan, e.g. --kv-resident), and the planning inputs: every engine.* value except the memory outputs the plan itself feeds (host pool, host KV tier,
 * host headroom), plus layout.name. Any change is a miss. */
nlohmann::json plan_key_inputs(const nlohmann::json& values, const nlohmann::json& facts, const std::string& model_id,
                               const std::string& engine_id, const std::string& engine_args);
std::string plan_key(const nlohmann::json& key_inputs);

/* rad0011's plan (radiance --plan-memory) through the shared v1 decoder, plus what a single plan must satisfy by itself:
 * each host consumer once, and the checkpoint floor within the want. Throws naming `where` and the decoder's path. */
v1::EnginePlan decode_engine_plan(const nlohmann::json& plan, const std::string& where);

struct CachedPlan {
    v1::EnginePlan plan;           // decoded by the shared v1 codec
    nlohmann::json plan_json;   // as rad0011 wrote it
    std::string key, measured_at;
};

/* What a plan must agree with in the configuration it is stored for (values, as resolve() returns them): the host pool it
 * echoes, and the expected host consumer list (the n-gram table exactly when it is placed in RAM; embedding and host
 * arenas allowed; nothing else). Throws naming the disagreement. */
void check_plan_for_configuration(const nlohmann::json& values, const v1::EnginePlan& plan);

/* The entry for `key_inputs` under `dir`, or nothing on a miss. An entry whose recorded key or key inputs differ, whose
 * version is unknown or whose plan fails the shared v1 decode (serving_codec) is refused (an error), never treated as a miss. */
std::optional<CachedPlan> plan_cache_lookup(const std::filesystem::path& dir, const nlohmann::json& key_inputs);

/* Records a measured plan (rad0011's JSON, decoded by the shared v1 codec first) under its key. Written to a temporary
 * file and renamed into place. */
CachedPlan plan_cache_store(const std::filesystem::path& dir, const nlohmann::json& key_inputs,
                            const nlohmann::json& plan_json);

/* The calibrated KV backing floor (calibration.kv_backing_floor.fraction_ppm) for a resolved configuration (`values`, as
 * resolve() returns them), from the compiled-in seeds and the optional refit cache; 0 when the objective selects none
 * (spec section 5.2). The one source of the engine's --plan-check-floor-ppm: never a launcher literal. */
int64_t kv_backing_floor_ppm(const nlohmann::json& values, const std::filesystem::path& calibration_cache);

/* The engine's start-up check flags for a cached plan, as separate arguments: writes the plan (v1 JSON, from the cache
 * entry) next to the cache as <key>.expected-plan.json and returns {"--plan-check", FILE, "--plan-check-floor-ppm", N}.
 * Both flags go to the engine, never one (radiance refuses that). */
std::vector<std::string> plan_check_args(const std::filesystem::path& dir, const CachedPlan& entry, int64_t floor_ppm);

/* Measures the plan: runs `binary --manifest MANIFEST --role engine --plan-memory OUT` with only the engine card
 * visible (ROCR_VISIBLE_DEVICES = its HIP ordinal, HIP_VISIBLE_DEVICES = 0, as a launch does), waits for it and returns
 * the JSON it wrote, decoded by the shared v1 codec. The engine runs its real bring-up up to its memory plan on that card,
 * so the caller holds the gpu lease; this never takes one. `wall_seconds` is set to the engine's run time. */
nlohmann::json measure_engine_plan(const std::filesystem::path& binary, const std::filesystem::path& manifest,
                                   uint32_t engine_card_ordinal, const std::filesystem::path& out,
                                   double* wall_seconds);

}  // namespace bc::serving
