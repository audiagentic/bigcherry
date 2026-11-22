#pragma once
#include "ledger.h"
#include "serving_types.h"   // bc::serving::v2::EnginePlan, the generated v1 type
#include "v2/native_config_snapshot.h"

#include <cstdint>
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
/* The digest of the detected GPUs' IDENTITY: BDF, architecture, HIP ordinal, total VRAM and the other attributes that
   do not change while the host runs. Live VRAM use (vram_used_mib, vram_available_mib) and the negotiated PCIe link
   change from one start to the next (the previous engine's pages are still being released), so they are not part of it:
   a digest over the whole detection made a stored S8 replay refuse an unchanged configuration. */
std::string gpu_inventory_digest(const nlohmann::json& gpus);

/* The engine build's identity: sha256 over the engine binary and every regular file under its runtime home (plugins,
 * kernels, helpers), by relative path and content. */
std::string engine_identity(const std::filesystem::path& binary, const std::filesystem::path& radiance_home);

/* The bytes are hashed once per process for as long as no file's size, inode, modification or change time moves; every
 * later call in a start returns the same digest after a stat of each file, and a file that changed is hashed again.
 * Counts the content hashes run, so a test can prove a start reads the runtime once. */
uint64_t engine_identity_hash_runs();

/* Persist the digest across processes: one small file per runtime home under `work_root` (the platform's work root,
 * --work-root), holding the digest and the stamp of every file it covers. A later process whose stamps all still match
 * reads it and hashes nothing; any difference hashes once and stores again. A file that cannot be read, parsed, or
 * that names another runtime home is ignored. Not called: the per-process memo above only. */
void set_engine_identity_store(const std::filesystem::path& work_root);

/* What the key covers: the cache version, the model and engine identities, the engine card's architecture and VRAM, and
 * the engine arguments the launcher passes through verbatim (they can change the plan, e.g. --kv-resident), and the
 * planning inputs: every engine.* value except the memory outputs the plan itself feeds (host pool, host KV tier,
 * host headroom), every tuning.* value (e.g. tp_ownership), plus layout.name. Any change is a miss. */
nlohmann::json plan_key_inputs(const nlohmann::json& values, const nlohmann::json& facts, const std::string& model_id,
                               const std::string& engine_id, const std::string& engine_args,
                               const NativeConfigSnapshot* native_config = nullptr);
std::string plan_key(const nlohmann::json& key_inputs);
// Separate policy-calibration scope taken from a strictly structured measured serving manifest.
// Includes model, role-specific card identity/architectures, layout, objective, histogram period,
// power profile and the helper/transport execution knobs that affect measured lane and copy times.
// Excludes engine binary, measured-plan hash, host pool, helper VRAM budgets and path strings.
nlohmann::json policy_calibration_inputs(const nlohmann::json& spec);
std::string policy_calibration_key(const nlohmann::json& inputs);
// MRC04: mirrors the platform-owned calibration-kinds-v1.json declaration.
nlohmann::json policy_calibration_key_fields();
void require_declared_policy_calibration_inputs(const nlohmann::json& inputs);

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

/* Provenance from the tracked measuring attempt that owns a native version-3 plan. */
struct OwnerMeasurementEvidence {
    std::string attempt_id;
    std::string measuring_manifest_sha256;
};

struct CachedPlan {
    v2::EnginePlan plan;           // decoded by the shared v1 codec
    nlohmann::json plan_json;   // as rad0011 wrote it
    nlohmann::json rank_demand; // the per-rank GPU demand rad0011 wrote with it (--plan-memory-ranks)
    std::string key, measured_at;
    HostResident resident;
    std::optional<OwnerMeasurementEvidence> owner;
    std::string owner_sha256; // validated digest of the owner receipt, empty for an unowned entry
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

/* Owned entries can only be published by the session owner after a tracked measuring attempt. Native version 3
 * requires ownership; version 2 also accepts it while preserving legacy seven-member cache entries. */
CachedPlan plan_cache_store_owned(const std::filesystem::path& dir, const nlohmann::json& key_inputs,
                                  const nlohmann::json& plan_json, const nlohmann::json& rank_demand,
                                  const nlohmann::json& values, const HostResident& resident,
                                  const OwnerMeasurementEvidence& owner);

}  // namespace bc::serving
