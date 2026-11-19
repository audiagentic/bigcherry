#include "native_fleet_config.h"

#include "sha256.h"
#include "v2/document_bytes.h"

#include <fstream>
#include <stdexcept>
#include <utility>

namespace bc::serving {

NativeFleetConfig NativeFleetConfig::read(const std::filesystem::path& authored) {
    // Size and read through the same open file, including under atomic author-file replacement.
    std::ifstream in(authored, std::ios::binary | std::ios::ate);
    const std::streamoff size = in.tellg();
    if (size < 0)
        throw std::runtime_error("cannot read native fleet YAML: " + authored.string());
    if (static_cast<uintmax_t>(size) > v2::kMaxDocumentBytes)
        throw std::invalid_argument("native fleet YAML exceeds the document byte limit");
    std::string bytes(static_cast<size_t>(size), '\0');
    in.seekg(0);
    if (!in.read(bytes.data(), size) || in.peek() != std::char_traits<char>::eof() || in.bad())
        throw std::runtime_error("cannot read complete native fleet YAML: " + authored.string());
    return NativeFleetConfig(std::move(bytes), authored.string(), std::nullopt);
}

NativeFleetConfig NativeFleetConfig::replay(std::string yaml_utf8, std::string_view expected_sha256) {
    return NativeFleetConfig(std::move(yaml_utf8), "spec.native_config.yaml_utf8", expected_sha256);
}

NativeFleetConfig::NativeFleetConfig(std::string yaml_utf8, const std::string& origin,
                                     std::optional<std::string_view> expected_sha256)
    : yaml_(std::move(yaml_utf8)) {
    if (yaml_.size() > v2::kMaxDocumentBytes)
        throw std::invalid_argument("native fleet YAML exceeds the document byte limit");
    if (v2::bad_utf8(yaml_) != std::string_view::npos || yaml_.find('\0') != std::string::npos)
        throw std::invalid_argument("native fleet YAML must be UTF-8 text without embedded NULs");
    sha256_ = sha256_hex(yaml_);
    if (expected_sha256 && *expected_sha256 != sha256_)
        throw std::invalid_argument("native fleet YAML SHA-256 mismatch");
    if (rad::fleet::parse(yaml_, origin, &fleet_) != RAD_OK)
        throw std::invalid_argument("Radiance refused the native fleet YAML: " + origin);
    if (fleet_.models.size() != 1)
        throw std::invalid_argument("managed native fleet YAML requires exactly one model");
    // Native provenance includes inherited and aliased server options. Checking
    // the option, not its resolved value, also permits environment references.
    if (fleet_.models.front().cfg.given.contains("--api-key"))
        throw std::invalid_argument("managed native fleet YAML requires server.api-key-env for credentials");
}

}  // namespace bc::serving
