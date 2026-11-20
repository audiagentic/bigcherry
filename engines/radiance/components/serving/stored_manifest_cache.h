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

struct NativeStaticIdentity {
    std::string yaml_sha256;
    std::string resolution_root;
    std::string runtime_home;
};

struct StaticResolutionInputs {
    nlohmann::json effective_values;        // RSC12 resolved values; transport and S8 budgets are projected out
    std::vector<std::string> operator_sets; // original CLI overrides, excluding layout selection and transport
    nlohmann::json ordered_devices;         // BDF, architecture, ordinal, total VRAM; no free/claimable room
    std::string producer_digest;
    std::string engine_digest;
    std::string helper_digest;
    std::string model_digest; // full artifact bytes, not only the container header/tables
    std::string power_profile;
    std::map<std::string, std::string> file_digests; // config, site, calibration, heat, lab evidence
    std::optional<NativeStaticIdentity> native_config;
};

// One Linux file identity: every listed metadata change invalidates a reused
// digest. Nanosecond timestamps are persisted as decimal strings (I-JSON safe).
struct FileStamp {
    uint64_t device = 0, inode = 0, size = 0;
    int64_t mtime_ns = 0, ctime_ns = 0;
    bool operator==(const FileStamp& rhs) const {
        return device == rhs.device && inode == rhs.inode && size == rhs.size && mtime_ns == rhs.mtime_ns &&
               ctime_ns == rhs.ctime_ns;
    }
};
struct ModelIdentity {
    std::string digest;
    FileStamp stamp;
};

// Cache only the full artifact digest; no omitted chunk, header-only shortcut
// or trust in a stale stat identity. The optional full reader is injected at
// the OS file boundary so a test can assert zero reads on a hit. The owner can
// also reuse a verified identity from the same measurement attempt.
using FullFileReader = callback::Guarded<std::string(const std::filesystem::path&)>;
// One shared cache per BC_WORK_ROOT; identical model paths reuse the digest
// across independent run directories and named configurations.
std::filesystem::path model_identity_cache_path(const std::filesystem::path& model,
                                                const std::filesystem::path& work_root);
ModelIdentity cached_model_identity(const std::filesystem::path& model, const std::filesystem::path& work_root,
                                    const FullFileReader& full_reader = {},
                                    const ModelIdentity* verified = nullptr);

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
};
// Exactly one index for a named config within this owner's run root; checked
// before reuse. Original v2 bytes remain immutable, indexed only by digest.
std::filesystem::path stored_index_path(const std::filesystem::path& run_dir, const std::string& config);
StoredManifestIndex read_stored_index(const std::filesystem::path& index);
void write_stored_index(const std::filesystem::path& index, const StoredManifestIndex& entry);
// A key mismatch is a cache miss. A matching key but corrupted/removed
// original is a refusal, never a silent remeasure or rewrite.
std::optional<std::filesystem::path> stored_manifest_for(const std::filesystem::path& run_dir,
                                                         const std::string& config,
                                                         const std::string& static_key);

} // namespace bc::serving::session
