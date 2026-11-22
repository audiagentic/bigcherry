#pragma once
// RSC12 step 5: stable config identity for manifest reuse.
// Ephemeral admission facts and per-session identities are not part of this key.
#include "callback_boundary.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace bc::serving::session {

struct StaticResolutionInputs {
    nlohmann::json effective_values;        // RSC12 resolved values; transport and S8 budgets are projected out
    std::vector<std::string> operator_sets; // original CLI overrides, excluding layout selection and transport
    nlohmann::json ordered_devices;         // BDF, architecture, ordinal, total VRAM; no free/claimable room
    std::string model_digest; // model_stat_identity(): path, size and mtime
    std::string power_profile;
    std::map<std::string, std::string> file_digests; // role -> content identity (never a path): config, site, calibration, heat, lab evidence
};

// A model is identified by its canonical path, size and mtime. The bytes are never read, so starting
// costs nothing however large it is; a model replaced in place with the same size and mtime is not noticed
// here (the engine's own admission is the check on what it loads).
std::string model_stat_identity(const std::filesystem::path& model);

// SHA-256 over RFC8785/JCS of these typed static inputs, with a version.
// Independent of the stored manifest's own digest and per-start execution record.
// A bounded-memory, domain-separated SHA-256 fingerprint over every file byte.
// Each ordered 1 MiB chunk contributes its hash, count and exact byte length.
std::string complete_file_identity(const std::filesystem::path& path);
// Same persisted chunked-file-v1 identity over bytes already read and authenticated.
std::string complete_file_identity_bytes(std::string_view bytes);

std::string static_resolution_key(const StaticResolutionInputs& in);

struct StoredManifestIndex {
    std::string key;
    std::string manifest_filename;
    std::string digest;
    std::string model_digest; // binds immutable manifest to original model; never a digest lookup cache
    std::string s8_digest; // optional full hash of the original writer's native P1/P2 replay evidence
    struct NativeMeasurement {
        std::string attempt_id;
        std::string sha256;
    };
    std::optional<NativeMeasurement> native_measurement;
    std::string directory; // the run directory holding the manifest; empty when it sits beside this index
};
// One index per named config in the work root's named-config store: the most recent measured manifest, reached
// from any later run directory. A different static key is a miss. Anything wrong with the store is a miss too:
// the caller measures and replaces it. Original v2 bytes remain immutable, indexed only by digest.
std::filesystem::path named_store_index(const std::filesystem::path& work_root, const std::string& config);
StoredManifestIndex read_stored_index(const std::filesystem::path& index);
void write_stored_index(const std::filesystem::path& index, const StoredManifestIndex& entry);
std::optional<std::filesystem::path> stored_manifest_for(const std::filesystem::path& index,
                                                         const std::string& static_key);

} // namespace bc::serving::session
