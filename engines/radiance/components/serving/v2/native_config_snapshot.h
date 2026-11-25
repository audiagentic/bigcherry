#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace bc::serving {

// The byte identity carried by a managed manifest. Native option validation
// belongs to the matched engine and takes these exact bytes as its input. The
// caller supplies the execution root; it is never inferred from the owner's cwd.
class NativeConfigSnapshot final {
  public:
    static NativeConfigSnapshot read(const std::filesystem::path& authored,
                                     const std::filesystem::path& resolution_root);
    static NativeConfigSnapshot admitted(std::string yaml, std::string sha256, std::string resolution_root);
    static NativeConfigSnapshot from_manifest(const nlohmann::json& field);
    // Read the exact snapshot envelope from a fully sealed, bounded memfd.
    // Used for owner-to-child handoff; callers retain ownership of `fd`.
    static NativeConfigSnapshot from_sealed_fd(int fd);
    // The raw authored YAML bytes of a sealed memfd, under the execution root the caller holds (the process's own
    // working directory). The owner-to-inspector handoff; the JSON-envelope form above stays only for the manifest
    // writer (RSC02/RSC03 delete it with manifest_field).
    static NativeConfigSnapshot from_raw_fd(int fd);
    static void verify(std::string_view yaml, std::string_view sha256, std::string_view resolution_root);
    nlohmann::json manifest_field() const;

    const std::string& yaml_utf8() const { return yaml_; }
    const std::string& sha256() const { return sha256_; }
    const std::string& resolution_root() const { return resolution_root_; }
    void require_execution_root() const;

  private:
    NativeConfigSnapshot(std::string yaml, std::string root, std::optional<std::string_view> sha256);

    std::string yaml_;
    std::string sha256_;
    std::string resolution_root_;
};

}  // namespace bc::serving
