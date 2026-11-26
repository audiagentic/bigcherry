#include "native_config_snapshot.h"

#include "../sha256.h"
#include "document_bytes.h"
#include "strict_json.h"

#include <fstream>
#include <stdexcept>
#include <utility>

namespace bc::serving {
namespace {

std::string current_resolution_root() {
    return std::filesystem::canonical(std::filesystem::current_path()).string();
}

std::string canonical_resolution_root(const std::filesystem::path& root) {
    const std::string& raw = root.native();
    if (v2::bad_utf8(raw) != std::string_view::npos || raw.find('\0') != std::string::npos ||
        !root.is_absolute())
        throw std::invalid_argument("native fleet resolution_root must be an absolute UTF-8 path");
    std::error_code ec;
    const auto canonical = std::filesystem::canonical(root, ec);
    if (ec)
        throw std::invalid_argument("native fleet resolution_root must name an existing directory");
    ec.clear();
    if (!std::filesystem::is_directory(canonical, ec) || ec)
        throw std::invalid_argument("native fleet resolution_root must name an existing directory");
    return canonical.string();
}

std::string checked_digest(std::string_view yaml, std::string_view resolution_root) {
    if (yaml.size() > v2::kMaxDocumentBytes)
        throw std::invalid_argument("native fleet YAML exceeds the document byte limit");
    if (v2::bad_utf8(yaml) != std::string_view::npos || yaml.find('\0') != std::string_view::npos)
        throw std::invalid_argument("native fleet YAML must be UTF-8 text without embedded NULs");
    if (v2::bad_utf8(resolution_root) != std::string_view::npos ||
        resolution_root.find('\0') != std::string_view::npos ||
        !std::filesystem::path(resolution_root).is_absolute())
        throw std::invalid_argument("native fleet resolution_root must be an absolute UTF-8 path");
    Sha256 digest;
    digest.update(yaml);
    return digest.hex_digest();
}

}  // namespace

NativeConfigSnapshot NativeConfigSnapshot::read(const std::filesystem::path& authored,
                                                 const std::filesystem::path& resolution_root) {
    // Size and read through one open file, including under atomic author-file replacement.
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
    return NativeConfigSnapshot(std::move(bytes), canonical_resolution_root(resolution_root), std::nullopt);
}

NativeConfigSnapshot NativeConfigSnapshot::admitted(std::string yaml, std::string sha256,
                                                     std::string resolution_root) {
    return NativeConfigSnapshot(std::move(yaml), std::move(resolution_root), sha256);
}

NativeConfigSnapshot NativeConfigSnapshot::from_manifest(const nlohmann::json& field) {
    constexpr const char* shape = "spec.native_config requires exactly the string fields yaml_utf8, sha256 and resolution_root";
    if (!field.is_object() || field.size() != 3)
        throw std::invalid_argument(shape);
    for (const char* key : {"yaml_utf8", "sha256", "resolution_root"}) {
        const auto it = field.find(key);
        if (it == field.end() || !it->is_string()) throw std::invalid_argument(shape);
    }
    return NativeConfigSnapshot(field.at("yaml_utf8").get_ref<const std::string&>(),
                                field.at("resolution_root").get_ref<const std::string&>(),
                                field.at("sha256").get_ref<const std::string&>());
}

NativeConfigSnapshot NativeConfigSnapshot::from_raw_fd(int fd) {
    return NativeConfigSnapshot(v2::read_sealed_document(fd, "native YAML descriptor"), current_resolution_root(),
                                std::nullopt);
}

nlohmann::json NativeConfigSnapshot::manifest_field() const {
    return {{"yaml_utf8", yaml_}, {"sha256", sha256_}, {"resolution_root", resolution_root_}};
}

void NativeConfigSnapshot::verify(std::string_view yaml, std::string_view sha256, std::string_view resolution_root) {
    if (checked_digest(yaml, resolution_root) != sha256)
        throw std::invalid_argument("native fleet YAML SHA-256 mismatch");
}

void NativeConfigSnapshot::require_execution_root() const {
    if (resolution_root_ != current_resolution_root())
        throw std::invalid_argument("native fleet resolution_root differs from the current execution root");
}

NativeConfigSnapshot::NativeConfigSnapshot(std::string yaml, std::string root,
                                           std::optional<std::string_view> expected_sha256)
    : yaml_(std::move(yaml)), sha256_(checked_digest(yaml_, root)), resolution_root_(std::move(root)) {
    if (expected_sha256 && *expected_sha256 != sha256_)
        throw std::invalid_argument("native fleet YAML SHA-256 mismatch");
}

}  // namespace bc::serving
