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
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
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

int64_t nanoseconds(const timespec& timestamp) {
    constexpr int64_t kBillion = 1000000000LL;
    if (timestamp.tv_nsec < 0 || timestamp.tv_nsec >= kBillion)
        throw std::runtime_error("file metadata has an invalid nanosecond timestamp");
    const int64_t sec = static_cast<int64_t>(timestamp.tv_sec);
    if (sec > (INT64_MAX - timestamp.tv_nsec) / kBillion || sec < (INT64_MIN + timestamp.tv_nsec) / kBillion)
        throw std::runtime_error("file metadata timestamp is out of range");
    return sec * kBillion + timestamp.tv_nsec;
}

FileStamp file_stamp(const std::filesystem::path& path) {
    struct stat metadata{};
    if (::stat(path.c_str(), &metadata) != 0 || !S_ISREG(metadata.st_mode) || metadata.st_size < 0)
        throw std::runtime_error("model artifact must be a stat-able ordinary file " + path.string());
    return FileStamp{static_cast<uint64_t>(metadata.st_dev), static_cast<uint64_t>(metadata.st_ino),
                     static_cast<uint64_t>(metadata.st_size), nanoseconds(metadata.st_mtim),
                     nanoseconds(metadata.st_ctim)};
}

uint64_t unsigned_field(const Json& input, const char* key) {
    if (!input.contains(key) || !input.at(key).is_string())
        throw std::runtime_error(std::string("stored model stat lacks ") + key);
    const std::string text = input.at(key).get<std::string>();
    size_t end = 0;
    uint64_t result = 0;
    try {
        result = std::stoull(text, &end);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("stored model stat has invalid ") + key);
    }
    if (end != text.size() || std::to_string(result) != text)
        throw std::runtime_error(std::string("stored model stat has noncanonical ") + key);
    return result;
}

int64_t signed_field(const Json& input, const char* key) {
    if (!input.contains(key) || !input.at(key).is_string())
        throw std::runtime_error(std::string("stored model stat lacks ") + key);
    const std::string text = input.at(key).get<std::string>();
    size_t end = 0;
    int64_t result = 0;
    try {
        result = std::stoll(text, &end);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("stored model stat has invalid ") + key);
    }
    if (end != text.size() || std::to_string(result) != text)
        throw std::runtime_error(std::string("stored model stat has noncanonical ") + key);
    return result;
}

FileStamp parse_stamp(const Json& record) {
    if (!record.is_object() || record.size() != 5)
        throw std::runtime_error("stored model stat must have five identity fields");
    return FileStamp{unsigned_field(record, "device"), unsigned_field(record, "inode"),
                     unsigned_field(record, "size"), signed_field(record, "mtime_ns"),
                     signed_field(record, "ctime_ns")};
}

Json write_stamp(const FileStamp& stamp) {
    return {{"device", std::to_string(stamp.device)},
            {"inode", std::to_string(stamp.inode)},
            {"size", std::to_string(stamp.size)},
            {"mtime_ns", std::to_string(stamp.mtime_ns)},
            {"ctime_ns", std::to_string(stamp.ctime_ns)}};
}

} // namespace

std::string complete_file_identity(const std::filesystem::path& path) {
    // The historical v1 chunked-file digest is kept BYTE FOR BYTE: only the
    // accumulator implementation changes. Hashing its 93-byte records as
    // produced avoids O(model size / MiB) retention of chunk hashes.
    const FileStamp before = file_stamp(path);
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open static input file " + path.string());
    Sha256 accumulator;
    accumulator.update("bytes=" + std::to_string(before.size) + "\n");
    accumulator.update("bigcherry/rsc12/chunked-file-v1\n");
    std::string chunk(1024ULL * 1024ULL, '\0');
    uint64_t read_bytes = 0;
    for (;;) {
        in.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const std::streamsize count = in.gcount();
        if (count < 0 || in.bad())
            throw std::runtime_error("cannot read static input file " + path.string());
        if (count == 0)
            break;
        if (read_bytes > UINT64_MAX - static_cast<uint64_t>(count))
            throw std::runtime_error("static input file size overflows");
        read_bytes += static_cast<uint64_t>(count);
        accumulator.update(std::to_string(static_cast<uint64_t>(count)));
        accumulator.update(":");
        accumulator.update(sha256_hex(std::string(chunk.data(), static_cast<size_t>(count))));
        accumulator.update("\n");
        if (in.eof())
            break;
    }
    if ((!in.eof() && in.fail()) || read_bytes != before.size || !(file_stamp(path) == before))
        throw std::runtime_error("static input file changed during full fingerprint " + path.string());
    return accumulator.hex_digest();
}

ModelIdentity cached_model_identity(const std::filesystem::path& model, const std::filesystem::path& index,
                                    FullFileReader full_reader) {
    // ctime_ns cannot be restored by normal in-place writes: even a same-size
    // write followed by a manual mtime reset changes this identity. An actor
    // capable of forging kernel inode/ctime fields is outside the threat model.
    const FileStamp before = file_stamp(model);
    if (std::filesystem::exists(index)) {
        if (!std::filesystem::is_regular_file(index) || std::filesystem::is_symlink(index))
            throw std::runtime_error("stored model identity index is not an ordinary file");
        const StoredManifestIndex recorded = read_stored_index(index);
        if (recorded.model_stamp && recorded.model_stamp.value() == before && !recorded.model_digest.empty())
            return {recorded.model_digest, before};
    }
    const std::string digest = full_reader ? full_reader(model) : complete_file_identity(model);
    check_digest(digest, "full model identity");
    if (!(file_stamp(model) == before))
        throw std::runtime_error("model artifact changed while calculating full identity");
    return {digest, before};
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
    // One projection for BOTH the new writer and stored replay. The owner
    // supplies original resolved operator/config/variant values; S8's derived
    // host pool/KV and the per-start POSIX transport never enter static
    // identity. Explicit CLI memory overrides remain in operator_sets, and
    // changed site/config bytes remain in file_digests.
    Json static_values = in.effective_values;
    if (static_values.contains("handoff") && static_values["handoff"].is_object())
        static_values["handoff"].erase("shm");
    if (static_values.contains("memory") && static_values["memory"].is_object()) {
        static_values["memory"].erase("host_pool_mib");
        static_values["memory"].erase("host_kv_mib");
    }
    Json inputs = {{"version", 2},
                   {"effective_values", std::move(static_values)},
                   {"operator_sets", in.operator_sets},
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
    if (!j.is_object() || j.value("version", 0) != 1 || !j.contains("key") || !j.at("key").is_string() ||
        !j.contains("manifest") || !j.at("manifest").is_string() || !j.contains("digest") ||
        !j.at("digest").is_string() ||
        j.size() != 4U + size_t(j.contains("model_digest")) + size_t(j.contains("model_stat")) ||
        (j.contains("model_digest") && !j.at("model_digest").is_string()) ||
        (j.contains("model_stat") && !j.contains("model_digest")))
        throw std::runtime_error("stored manifest index has an unknown or missing field");
    StoredManifestIndex result{j.at("key").get<std::string>(), j.at("manifest").get<std::string>(),
                               j.at("digest").get<std::string>(), j.value("model_digest", std::string())};
    if (j.contains("model_stat"))
        result.model_stamp = parse_stamp(j.at("model_stat"));
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
    if (entry.model_stamp) {
        if (entry.model_digest.empty())
            throw std::runtime_error("model stat requires the full model digest");
        record["model_stat"] = write_stamp(*entry.model_stamp);
    }
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
