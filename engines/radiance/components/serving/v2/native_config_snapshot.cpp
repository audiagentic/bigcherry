#include "native_config_snapshot.h"

#include "../sha256.h"
#include "document_bytes.h"
#include "strict_json.h"

#include <cerrno>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

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

NativeConfigSnapshot NativeConfigSnapshot::from_sealed_fd(int fd) {
    if (fd < 3) throw std::invalid_argument("native snapshot descriptor must be greater than 2");
    constexpr int required_seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
    const int seals = ::fcntl(fd, F_GET_SEALS);
    if (seals < 0 || (seals & required_seals) != required_seals)
        throw std::invalid_argument("native snapshot descriptor must be a fully sealed memfd");
    struct stat info {};
    if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size <= 0 ||
        static_cast<uintmax_t>(info.st_size) > v2::kMaxDocumentBytes)
        throw std::invalid_argument("native snapshot descriptor must be a bounded regular file");
    std::string bytes(static_cast<size_t>(info.st_size), '\0');
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = ::pread(fd, bytes.data() + offset, bytes.size() - offset,
                                      static_cast<off_t>(offset));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("could not read the complete native snapshot descriptor");
        offset += static_cast<size_t>(count);
    }
    const auto parsed = v2::parse_strict_json(bytes, "native snapshot descriptor");
    if (!parsed) throw std::invalid_argument(v2::render_human("native snapshot descriptor", parsed.error()));
    return from_manifest(parsed.value());
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
