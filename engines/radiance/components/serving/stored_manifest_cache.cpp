#include "stored_manifest_cache.h"

#include "jcs.h"
#include "sha256.h"
#include "v2/strict_json.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

namespace bc::serving::session {
namespace {
using Json = nlohmann::json;

bool hex_digest(const std::string& value) {
    if (value.size() != 64)
        return false;
    return std::all_of(value.begin(), value.end(),
                       [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
void check_digest(const std::string& digest, const char* where) {
    if (!hex_digest(digest))
        throw std::runtime_error(std::string(where) + ": invalid SHA-256 identity");
}
void check_filename(const std::string& filename) {
    const std::filesystem::path path(filename);
    if (filename.empty() || path.is_absolute() || path.filename() != path ||
        filename.find("..") != std::string::npos || filename.size() > 220 ||
        filename.rfind("manifest-", 0) != 0 || path.extension() != ".json")
        throw std::runtime_error("stored manifest index has unsafe filename");
}
} // namespace

std::string complete_file_identity(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open static input file " + path.string());
    std::string accumulator = "bigcherry/rsc12/chunked-file-v1\n";
    std::string chunk(1024ULL * 1024ULL, '\0');
    uint64_t size = 0;
    for (;;) {
        in.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const std::streamsize count = in.gcount();
        if (count < 0 || in.bad())
            throw std::runtime_error("cannot read static input file " + path.string());
        if (count == 0)
            break;
        if (size > UINT64_MAX - static_cast<uint64_t>(count))
            throw std::runtime_error("static input file size overflows");
        size += static_cast<uint64_t>(count);
        accumulator += std::to_string(static_cast<uint64_t>(count)) + ":" +
                       sha256_hex(chunk.substr(0, static_cast<size_t>(count))) + "\n";
        if (in.eof())
            break;
    }
    if (!in.eof() && in.fail())
        throw std::runtime_error("cannot finish static input file " + path.string());
    return sha256_hex("bytes=" + std::to_string(size) + "\n" + accumulator);
}

std::string static_resolution_key(const StaticResolutionInputs& in) {
    if (!in.effective_values.is_object() || !in.ordered_devices.is_array() || in.ordered_devices.empty() ||
        in.power_profile.empty())
        throw std::runtime_error(
            "static resolution key needs effective config, ordered devices and power profile");
    for (const auto& card : in.ordered_devices) {
        if (!card.is_object() || card.size() != 5 || !card.contains("role_group") ||
            !card.at("role_group").is_string() || !card.contains("bdf") || !card.at("bdf").is_string() ||
            !card.contains("arch") || !card.at("arch").is_string() || !card.contains("hip_ordinal") ||
            !card.at("hip_ordinal").is_number_integer() || !card.contains("vram_total_mib") ||
            !card.at("vram_total_mib").is_number_integer())
            throw std::runtime_error(
                "static key cards must include only role, BDF, arch, ordinal and total VRAM");
    }
    for (const auto* s : {&in.producer_digest, &in.engine_digest, &in.helper_digest, &in.model_digest})
        check_digest(*s, "static resolution key");
    Json inputs = {{"version", 1},
                   {"effective_values", in.effective_values},
                   {"ordered_devices", in.ordered_devices},
                   {"producer", in.producer_digest},
                   {"engine", in.engine_digest},
                   {"helper", in.helper_digest},
                   {"model_artifact", in.model_digest},
                   {"power_profile", in.power_profile},
                   {"inputs", in.file_digests}};
    for (const auto& [key, digest] : in.file_digests) {
        if (key.empty())
            throw std::runtime_error("static resolution key has an unnamed file");
        check_digest(digest, "static input file");
    }
    std::string canonical, error;
    if (!jcs_canonicalize(inputs, &canonical, &error))
        throw std::runtime_error("static resolution key cannot canonicalize: " + error);
    return sha256_hex(canonical);
}

std::filesystem::path stored_index_path(const std::filesystem::path& run_dir, const std::string& config) {
    if (config.empty())
        throw std::runtime_error("named config is required for stored manifest reuse");
    return run_dir / ("stored-" + sha256_hex(config) + ".index.json");
}

StoredManifestIndex read_stored_index(const std::filesystem::path& index) {
    std::ifstream in(index, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open stored manifest index " + index.string());
    // Read at most one byte beyond the protocol bound. An untrusted sparse
    // or corrupt index must never determine our allocation size.
    std::string bytes(8193U, '\0');
    in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    const auto consumed = in.gcount();
    if (in.bad())
        throw std::runtime_error("cannot read stored manifest index " + index.string());
    if (consumed > 8192)
        throw std::runtime_error("stored manifest index exceeds 8 KiB");
    bytes.resize(static_cast<size_t>(consumed));
    const auto parsed = v2::parse_strict_json(bytes, index.string());
    if (!parsed)
        throw std::runtime_error("stored manifest index is invalid: " + parsed.error().message);
    const auto& j = parsed.value();
    if (!j.is_object() || (j.size() != 4 && j.size() != 5) || j.value("version", 0) != 1 ||
        !j.contains("key") || !j.at("key").is_string() || !j.contains("manifest") ||
        !j.at("manifest").is_string() || !j.contains("digest") || !j.at("digest").is_string() ||
        (j.size() == 5 && (!j.contains("model_digest") || !j.at("model_digest").is_string())))
        throw std::runtime_error("stored manifest index has an unknown or missing field");
    StoredManifestIndex result{j.at("key").get<std::string>(), j.at("manifest").get<std::string>(),
                               j.at("digest").get<std::string>(), j.value("model_digest", std::string())};
    check_digest(result.key, "stored key");
    check_digest(result.digest, "stored digest");
    if (!result.model_digest.empty())
        check_digest(result.model_digest, "stored model artifact");
    check_filename(result.manifest_filename);
    return result;
}

void write_stored_index(const std::filesystem::path& index, const StoredManifestIndex& entry) {
    check_digest(entry.key, "stored key");
    check_digest(entry.digest, "stored digest");
    if (!entry.model_digest.empty())
        check_digest(entry.model_digest, "stored model artifact");
    check_filename(entry.manifest_filename);
    Json record = {
        {"version", 1}, {"key", entry.key}, {"manifest", entry.manifest_filename}, {"digest", entry.digest}};
    if (!entry.model_digest.empty())
        record["model_digest"] = entry.model_digest;
    const std::string bytes = record.dump(2) + "\n";
    const std::filesystem::path temp = index.string() + ".tmp." + std::to_string(::getpid());
    const int fd = ::open(temp.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
    if (fd < 0)
        throw std::runtime_error("cannot create stored index temporary file: " +
                                 std::string(std::strerror(errno)));
    bool opened = true;
    bool renamed = false;
    try {
        size_t written = 0;
        while (written != bytes.size()) {
            const ssize_t n = ::write(fd, bytes.data() + written, bytes.size() - written);
            if (n < 0 && errno == EINTR)
                continue;
            if (n <= 0)
                throw std::runtime_error("cannot write stored index");
            written += static_cast<size_t>(n);
        }
        if (::fdatasync(fd) != 0)
            throw std::runtime_error("cannot sync stored index");
        const int closed = ::close(fd);
        opened = false;
        if (closed != 0)
            throw std::runtime_error("cannot close stored index");
        if (::rename(temp.c_str(), index.c_str()) != 0)
            throw std::runtime_error("cannot atomically update stored index: " +
                                     std::string(std::strerror(errno)));
        renamed = true;
        const int dir = ::open(index.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dir < 0)
            throw std::runtime_error("cannot open stored index directory");
        const int synced = ::fsync(dir);
        ::close(dir);
        if (synced != 0)
            throw std::runtime_error("cannot persist stored index directory");
    } catch (...) {
        if (opened)
            ::close(fd);
        if (!renamed)
            ::unlink(temp.c_str());
        throw;
    }
}

std::optional<std::filesystem::path> stored_manifest_for(const std::filesystem::path& run_dir,
                                                         const std::string& config,
                                                         const std::string& static_key) {
    check_digest(static_key, "static key");
    const auto index = stored_index_path(run_dir, config);
    if (!std::filesystem::exists(index))
        return std::nullopt;
    if (!std::filesystem::is_regular_file(index) || std::filesystem::is_symlink(index))
        throw std::runtime_error("stored index is not an ordinary file");
    const auto entry = read_stored_index(index);
    if (entry.key != static_key)
        return std::nullopt;
    auto saved = run_dir / entry.manifest_filename;
    if (!std::filesystem::is_regular_file(saved) || std::filesystem::is_symlink(saved) ||
        complete_file_identity(saved) != entry.digest)
        throw std::runtime_error("stored manifest cache: matching index refers to missing or tampered bytes");
    return saved;
}

} // namespace bc::serving::session
