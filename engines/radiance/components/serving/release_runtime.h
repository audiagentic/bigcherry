#pragma once

#include <filesystem>
#include <optional>

namespace bc::serving {

// The executable layout implies PREFIX, not trust in PREFIX.
// Only the native verifier establishes current-release identity.
std::optional<std::filesystem::path> installed_prefix(const std::filesystem::path& exe);

// A fresh native digest + tag-bound receipt revalidation on every start and
// detached serve, using exactly the implementation behind bc-release verify.
std::filesystem::path verify_installed_prefix(const std::filesystem::path& exe,
                                              const std::filesystem::path& sdk);

} // namespace bc::serving
