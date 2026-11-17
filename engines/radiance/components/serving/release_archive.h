#pragma once

#include "release_install.h"

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
void verify_installed_files(const fs::path& prefix, const Json& record, const Json& manifest,
                            const std::string& tag);
std::string required_text(const Json& doc, const char* key, ErrorCode code);

} // namespace bc::release
