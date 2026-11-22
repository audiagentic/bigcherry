#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace bc::release {

namespace fs = std::filesystem;
using Json = nlohmann::json;

// Every refusal maps to one stable CLI error family. Operational failures
// are never silently treated as a reason to start an unverified engine.
enum class ErrorCode {
    invalid_input = 2,
    digest_mismatch = 3,
    wrong_identity = 4,
    unsafe_archive = 5,
    invalid_install = 6,
    conflict = 7,
    io = 8,
    sdk_mismatch = 9,
};

class Error final : public std::runtime_error {
  public:
    Error(ErrorCode code, const std::string& message);
    ErrorCode code() const noexcept {
        return code_;
    }

  private:
    ErrorCode code_;
};

struct InstallRequest {
    std::string tag;
    fs::path record;  // receipt downloaded from this *exact* release tag
    fs::path archive; // the record's permitted asset, downloaded unmodified
    fs::path root;
    fs::path sdk;
    bool allow_downgrade = false; // explicit operator authorization
};

// This module derives the on-disk trust and retention contract directly from
// RSP01: a single tag-bound engine asset, a last-written atomic receipt,
// revalidated current/previous candidates, and no runtime source checkout.
// It does not invoke, translate, or import the Python release installer.
void install(const InstallRequest& request);
Json verify(const fs::path& source, const fs::path& sdk);
// What a start runs: the current tag only (the previous one is not scanned), and its runtime members are hashed only if
// they differ from the stamps recorded when they were last verified in full (install, or `verify`).
Json verify_current(const fs::path& source, const fs::path& sdk);
Json rollback(const fs::path& source, const fs::path& sdk);

} // namespace bc::release
