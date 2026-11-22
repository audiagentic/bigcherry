#pragma once

#include "release_install.h"

#include <cstdint>

namespace bc::release {

// The release record is supplied by the selected tag's release endpoint.
// These functions never consult a mutable "latest" index or Python runtime.
struct CheckedAsset {
    Json record;
    Json manifest;
};

Json read_json(const fs::path& path);
std::string sha256_file(const fs::path& path);
void validate_release_record(const Json& record, const std::string& tag, const fs::path& archive,
                             const fs::path& sdk);
CheckedAsset extract_verified(const Json& record, const std::string& tag, const fs::path& archive,
                              const fs::path& prefix);
// The members hashed in full so far in this process (the cost a start must not pay on an unchanged install).
std::uint64_t hashed_files();
// Verify an installed prefix. `trust_stamps`: skip hashing the members when their stamps all match the record in
// `stamps_file` (kept beside the releases, never inside a prefix, so an older tool still sees the prefix it knows);
// any difference runs the full check, which on success records the stamps again.
void verify_installed_files(const fs::path& prefix, const Json& record, const Json& manifest,
                            const std::string& tag, const fs::path& stamps_file, bool trust_stamps);
void fsync_dir(const fs::path& dir);
std::string nonce();
void write_atomic(const fs::path& dest, const std::string& bytes);
std::string required_text(const Json& doc, const char* key, ErrorCode code);

} // namespace bc::release
