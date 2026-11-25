#pragma once
/* bc-serve's manifest writer: the resolver's output, the host facts, the measured plan and the planner's owner map
 * assembled into a serving manifest v2 draft, resolved by S10 (power plan, S8 resource ledger and handoff, live-policy
 * decision), digested and checked.
 *
 * The result is accepted only when the document reads back through the shared strict reader (v2::read_manifest: strict
 * JSON, identity, JCS digest, structure, provenance pointers and coverage, semantics) and its owner map equals the
 * planner's (verify_owner_map). A document that does not is an error carrying the reader's own message, never a file. */
#include "plan_cache.h"
#include "serving.h"
#include "v2/native_config_snapshot.h"
#include "v2/s10_inputs.h"

#include <filesystem>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace bc::serving {

/* The engine plan a measured manifest carries and what it was measured for (the interim plan cache's entry and its key
 * inputs). */
struct MeasuredPlan {
    CachedPlan plan;
    nlohmann::json key_inputs;
};

struct ManifestInputs {
    Resolved resolved;
    std::optional<NativeConfigSnapshot> native_config; // Required by the generated v3 edition.
    std::filesystem::path engine_binary; // Selected matching engine whose bytes were keyed for a measured plan.
    std::string plan_engine_args;        // Exact arguments the launcher supplied to Radiance's measuring plan.
    std::string profile;                     // spec.intent.service.profile: "production" or "lab", named by the caller
    std::string instance_id;                 // UUID v4
    std::string producer_sha256;             // SHA-256 of the running bc-serve executable
    std::filesystem::path calibration_cache; // the refit layer the resolver used ("" = seeds only)
    // The measured plan: present writes plan.state measured; absent writes the unmeasured manifest of the measuring start
    // (owner map and settings only: no engine_memory, no checkpoint slots).
    std::optional<MeasuredPlan> measured;
    // spec.handoff.placement.live_policy: an object with every member (the reader refuses anything less); null = no
    // policy. `live_policy_from` names where it came from, for its provenance record.
    nlohmann::json live_policy;
    std::string live_policy_from;
    // Default off: a measured start with no usable calibration record then asks the helper to calibrate in session
    // (spec.handoff.placement.live_calibration). Set only by an explicit setting; see derive_manifest_live_calibration.
    bool auto_calibrate = false;
    // spec.experiment, lab profile only; null = none. `experiment_from` as above.
    nlohmann::json experiment;
    std::string experiment_from;
    // What S10 resolves the document with: the native P1 measurement (measured manifests), the usable calibration
    // records and the named power profile.
    v2::WriterInputs s10;
};

struct BuiltManifest {
    nlohmann::json document;
    std::string text;    // the document as written: two-space indent, trailing newline
    std::string digest;  // "sha256:" + hex, also in document.metadata.content_digest
};

/* Throws std::runtime_error with the failing step's message. */
BuiltManifest build_manifest(const ManifestInputs& in);

/* One manifest-writing step of a native start, as the owner and the --write-manifest CLI both run it: the inputs the writer
 * child used to receive through argv and a sealed descriptor. */
struct NativeWriterRequest {
    Options options;                      // the resolver options (site, profile, --set list, calibration cache, run dir, ...)
    InspectedConfig inspected;            // the config as the inspector parsed it from the same sealed YAML bytes
    std::optional<NativeConfigSnapshot> snapshot;   // required
    NativeEngineSettings settings{};        // the matched parser's engine settings
    bool unmeasured = false;              // the measuring start's draft; otherwise the measured manifest from the plan cache
    std::filesystem::path engine_binary;
    std::string plan_extra;
    std::string expected_plan_key;        // frozen by the owner; empty only for the CLI's cache-less unmeasured draft
    std::string expected_owner_attempt;   // measured only: the plan cache entry must be owned by this attempt
    std::filesystem::path plan_cache;     // measured only
    std::string power_profile;            // x-bigcherry.power.profile of the config
    v2::S10PowerProfile power_rules;      // that profile's rules from the profile file (empty for stock)
    nlohmann::json calibration_records = nlohmann::json::array();   // P2 records for S10
    std::string manifest_name;            // names the manifest in the P1 record's run id
    std::string producer_sha256;          // SHA-256 of the executable that writes it
    bool lab_profile = false;             // an experiment or a calibration run: spec.intent.service.profile "lab"
    nlohmann::json live_policy;           // null = none; `live_policy_from` names its source
    std::string live_policy_from;
    nlohmann::json experiment;            // null = none
    std::string experiment_from;
    bool auto_calibrate = false;
    bool want_s8_replay_evidence = false; // measured only: also return the exact P1/P2 record S8 consumed
};

struct NativeWriterOutcome {
    enum class Kind { built, cache_miss, host_pool_change };
    Kind kind = Kind::built;
    BuiltManifest built;                  // Kind::built
    int64_t derived_pool_mib = 0;         // Kind::host_pool_change: the pool S8 derives from what was measured
    std::string plan_key;                 // Kind::cache_miss: the key that missed
    nlohmann::json s8_replay_source;      // Kind::built with want_s8_replay_evidence, else null
    std::string notes;                    // what the writer reports about the live policy (one line each)
};

/* Throws std::runtime_error with the failing step's message; a cache miss and a host-pool change are results, not errors. */
NativeWriterOutcome write_native_manifest(const NativeWriterRequest& request);

/* spec.processes.helper_common as the manifest emits it for these resolved values (helper_profile.toml, resolved rows taken
 * from the values): what the helpers run with unless an experiment patches a helper. */
nlohmann::json helper_common_settings(const nlohmann::json& values);

/* The host-ledger margin the calibration gives this layout and objective: {key, source, evidence, mib}. */
nlohmann::json host_ledger_margin(const Resolved& resolved, const std::filesystem::path& calibration_cache);

/* Atomically creates `path` read-only with `text`; refuses an existing path (a manifest is immutable and never replaced). */
void write_manifest_file(const std::filesystem::path& path, const std::string& text);

/* A fresh UUID v4 (the kernel's), and the SHA-256 of a file's bytes. */
std::string new_instance_id();
std::string file_sha256(const std::filesystem::path& file);

}  // namespace bc::serving
