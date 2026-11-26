#pragma once
/* What the owner derived for one engine start and the engine cannot read from the native YAML: the ordered selected
 * devices, the measured host pool/KV/reserve/checkpoint sizes, the expected memory plan and KV floor rad0011 checks before
 * weights, and the typed engine/remote settings the plugins read (bc_engine_settings.h). One strict, sealed, inherited
 * document (`--bc-admission-fd FD`), never persisted. It carries no native setting: the YAML is the only place those
 * live, and a derived value that disagrees with an authored one is a refusal (bc_serving_managed_bind).
 *
 * It is built from the manifest the owner already holds until RSC11 removes that manifest; the codec does not change then. */
#include "manifest.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace bc::serving {

struct EngineAdmission {
    bool measuring = false;               // the measuring start (rad0011 --plan-memory): no expected plan exists yet
    std::vector<std::string> bdfs;        // the ordered engine rank devices, as BDFs
    std::optional<std::vector<std::string>> rank_devices;
    int64_t host_pool_mib_per_rank = 0, host_kv_mib = 0, host_reserve_mib = 0;
    std::optional<int64_t> checkpoint_slots;
    int64_t kv_backing_floor_ppm = 0;
    std::optional<nlohmann::json> expected;   // the rad0011 plan a serving start must admit (v2::EnginePlan encoding)
    // The typed engine settings plugins read through bc_serving_engine_settings.
    bool handoff = false, skip_owned = false, shadow = false, combine = false;
    uint32_t mask_only = 0, mailbox_trace_every = 0;
    int64_t max_rows = 64, histogram_period_s = 0;
    std::string shm;
    std::optional<std::string> histogram_file;
    bool moe_native = true, moe_fp16 = true, moe_ahalf = true, gemv = false, dec11 = false, fuse_stage = true,
         attn_native = true, all_reduce_wire4 = false;
    std::map<std::string, std::string> env;   // the engine switches not yet typed (RSC11), names from the contract's env bindings
    // An offline KL run (record a reference or score against one): the run's files, not configuration. Serving and
    // measuring starts have none.
    struct Quality {
        bool record = false;                 // record: `corpus` is set and `reference` is an empty directory to fill
        std::string reference, corpus, output;
        bool operator==(const Quality&) const = default;
    };
    std::optional<Quality> quality;
};

nlohmann::json engine_admission_to_json(const EngineAdmission& admission);
/* Strict: every member present, none unknown, integers exact, env names only those the contract binds. */
v2::Result<EngineAdmission> engine_admission_from_json(const nlohmann::json& document);
/* The derivation the manifest entry makes today, from the strictly read manifest of a serving or measuring start. */
v2::Result<EngineAdmission> engine_admission_from_manifest(const v2::Manifest& manifest, bool measuring,
                                                           std::optional<EngineAdmission::Quality> quality = std::nullopt);

}  // namespace bc::serving
