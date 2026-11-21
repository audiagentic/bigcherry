#include "stored_manifest_cache.h"

#include "jcs.h"
#include "sha256.h"
#include "v2/strict_json.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
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

void check_native_measurement(const StoredManifestIndex& entry) {
    if (!entry.native_measurement)
        return;
    const auto& measurement = *entry.native_measurement;
    const std::string& attempt = measurement.attempt_id;
    if (attempt.size() != 36)
        throw std::runtime_error("stored native measurement has an invalid attempt ID");
    for (size_t i = 0; i < attempt.size(); ++i) {
        const char c = attempt[i];
        const bool hyphen = i == 8 || i == 13 || i == 18 || i == 23;
        if (hyphen ? c != '-' : !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            throw std::runtime_error("stored native measurement has an invalid attempt ID");
    }
    if (entry.manifest_filename != "manifest-native-" + attempt + ".json" ||
        entry.model_digest.empty() || entry.s8_digest.empty())
        throw std::runtime_error("stored native measurement is missing its manifest, model or S8 attestation");
    check_digest(measurement.sha256, "stored native measurement");
}

int64_t nanoseconds(const timespec& timestamp) {
    constexpr int64_t kBillion = 1000000000LL;
    if (timestamp.tv_nsec < 0 || timestamp.tv_nsec >= kBillion)
        throw std::runtime_error("file metadata has an invalid nanosecond timestamp");
    const int64_t sec = static_cast<int64_t>(timestamp.tv_sec);
    if (sec > (INT64_MAX - timestamp.tv_nsec) / kBillion || sec < (INT64_MIN + timestamp.tv_nsec) / kBillion)
        throw std::runtime_error("file metadata timestamp is out of range");
    return (sec * kBillion) + timestamp.tv_nsec;
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

void seed_file_identity(Sha256& accumulator, uint64_t size) {
    accumulator.update("bytes=" + std::to_string(size) + "\n");
    accumulator.update("bigcherry/rsc12/chunked-file-v1\n");
}

void add_file_identity_chunk(Sha256& accumulator, std::string_view chunk) {
    accumulator.update(std::to_string(chunk.size()));
    accumulator.update(":");
    Sha256 chunk_hash;
    chunk_hash.update(chunk);
    accumulator.update(chunk_hash.hex_digest());
    accumulator.update("\n");
}

} // namespace

std::string complete_file_identity_bytes(std::string_view bytes) {
    Sha256 accumulator;
    seed_file_identity(accumulator, bytes.size());
    constexpr size_t chunk_bytes = 1024ULL * 1024ULL;
    for (size_t offset = 0; offset < bytes.size(); offset += chunk_bytes)
        add_file_identity_chunk(accumulator, bytes.substr(offset, std::min(chunk_bytes, bytes.size() - offset)));
    return accumulator.hex_digest();
}

std::string complete_file_identity(const std::filesystem::path& path) {
    // The historical v1 chunked-file digest is kept BYTE FOR BYTE: only the
    // accumulator implementation changes. Hashing its 93-byte records as
    // produced avoids O(model size / MiB) retention of chunk hashes.
    const FileStamp before = file_stamp(path);
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open static input file " + path.string());
    Sha256 accumulator;
    seed_file_identity(accumulator, before.size);
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
        add_file_identity_chunk(accumulator, std::string_view(chunk.data(), static_cast<size_t>(count)));
        if (in.eof())
            break;
    }
    if ((!in.eof() && in.fail()) || read_bytes != before.size || !(file_stamp(path) == before))
        throw std::runtime_error("static input file changed during full fingerprint " + path.string());
    return accumulator.hex_digest();
}

namespace {

// Strict, bounded, ordinary-file cache reader. O_NOFOLLOW + fstat prevents a
// symlink swap between a path check and open from bypassing the contract.
std::optional<ModelIdentity> read_shared_model_identity(const std::filesystem::path& cache,
                                                         const std::string& canonical) {
    struct stat st{};
    if (::lstat(cache.c_str(), &st) != 0) {
        if (errno == ENOENT)
            return std::nullopt;
        throw std::runtime_error("cannot inspect shared model identity cache " + cache.string());
    }
    if (!S_ISREG(st.st_mode))
        throw std::runtime_error("shared model identity cache is not an ordinary file");
    const int fd = ::open(cache.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0)
        throw std::runtime_error("cannot open shared model identity cache " + cache.string());
    bool opened = true;
    try {
        if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0 || st.st_size > 4096)
            throw std::runtime_error("shared model identity cache is not a bounded ordinary file");
        std::string bytes(static_cast<size_t>(st.st_size), '\0');
        size_t read = 0;
        while (read < bytes.size()) {
            const ssize_t count = ::read(fd, bytes.data() + read, bytes.size() - read);
            if (count < 0 && errno == EINTR)
                continue;
            if (count <= 0)
                throw std::runtime_error("cannot read shared model identity cache");
            read += static_cast<size_t>(count);
        }
        // Detect a growth race rather than accepting a valid prefix of a
        // concurrently mutated file (normal writers always rename).
        char trailing = 0;
        const ssize_t extra = ::read(fd, &trailing, 1);
        if (extra != 0)
            throw std::runtime_error("shared model identity cache changed while reading");
        const auto parsed = v2::parse_strict_json(bytes, cache.string());
        if (!parsed)
            throw std::runtime_error("invalid shared model identity cache: " + parsed.error().message);
        const Json& record = parsed.value();
        if (!record.is_object() || record.size() != 4 || !record.contains("version") ||
            !record.at("version").is_number_integer() || record.at("version") != 1 ||
            !record.contains("model") || !record.at("model").is_string() ||
            record.at("model").get<std::string>() != canonical ||
            !record.contains("stamp") || !record.contains("digest") || !record.at("digest").is_string())
            throw std::runtime_error("shared model identity cache has an invalid shape or path");
        ModelIdentity result{record.at("digest").get<std::string>(), parse_stamp(record.at("stamp"))};
        check_digest(result.digest, "shared model identity cache");
        const int closed = ::close(fd);
        opened = false;
        if (closed != 0)
            throw std::runtime_error("cannot close shared model identity cache");
        return result;
    } catch (...) {
        if (opened)
            ::close(fd);
        throw;
    }
}

void ensure_private_directory(const std::filesystem::path& directory) {
    if (::mkdir(directory.c_str(), 0700) != 0 && errno != EEXIST)
        throw std::runtime_error("cannot create shared model identity directory " + directory.string());
    struct stat st{};
    if (::lstat(directory.c_str(), &st) != 0 || !S_ISDIR(st.st_mode))
        throw std::runtime_error("shared model identity directory must be ordinary, not a symlink");
}

std::filesystem::path shared_cache_dir(const std::filesystem::path& work_root) {
    if (work_root.empty() || !work_root.is_absolute())
        throw std::runtime_error("shared model identity requires an absolute BC_WORK_ROOT");
    struct stat st{};
    if (::lstat(work_root.c_str(), &st) != 0 || !S_ISDIR(st.st_mode))
        throw std::runtime_error("BC_WORK_ROOT must be an ordinary directory");
    const auto cache = work_root / "cache";
    ensure_private_directory(cache);
    const auto identities = cache / "model-identities";
    ensure_private_directory(identities);
    return identities;
}

void write_shared_model_identity(const std::filesystem::path& cache, const std::string& canonical,
                                 const ModelIdentity& identity) {
    // The path hash is an index only. A strict file also stores the complete
    // canonical model pathname, so a collision never becomes a trusted hit.
    const std::string bytes =
        Json{{"version", 1}, {"model", canonical}, {"stamp", write_stamp(identity.stamp)},
             {"digest", identity.digest}}.dump(2) + "\n";
    const std::string pattern = cache.string() + ".tmp.XXXXXX";
    std::vector<char> temp(pattern.begin(), pattern.end());
    temp.push_back('\0');
    const int fd = ::mkstemp(temp.data());
    if (fd < 0)
        throw std::runtime_error("cannot create shared model identity temporary file");
    bool opened = true;
    bool renamed = false;
    try {
        if (::fcntl(fd, F_SETFD, FD_CLOEXEC) != 0)
            throw std::runtime_error("cannot mark shared model identity temporary file close-on-exec");
        size_t written = 0;
        while (written < bytes.size()) {
            const ssize_t count = ::write(fd, bytes.data() + written, bytes.size() - written);
            if (count < 0 && errno == EINTR)
                continue;
            if (count <= 0)
                throw std::runtime_error("cannot write shared model identity cache");
            written += static_cast<size_t>(count);
        }
        if (::fdatasync(fd) != 0)
            throw std::runtime_error("cannot sync shared model identity cache");
        const int closed = ::close(fd);
        opened = false;
        if (closed != 0)
            throw std::runtime_error("cannot close shared model identity cache");
        // Reject a malicious symlink, including a dangling one, rather than
        // accepting it as a miss. rename is atomic across racing writers.
        struct stat st{};
        if (::lstat(cache.c_str(), &st) == 0) {
            if (!S_ISREG(st.st_mode))
                throw std::runtime_error("shared model identity destination is not an ordinary file");
        } else if (errno != ENOENT) {
            throw std::runtime_error("cannot inspect shared model identity destination");
        }
        if (::rename(temp.data(), cache.c_str()) != 0)
            throw std::runtime_error("cannot atomically update shared model identity cache");
        renamed = true;
        const int dir_fd = ::open(cache.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
        if (dir_fd < 0)
            throw std::runtime_error("cannot open shared model identity directory for sync");
        const int synced = ::fsync(dir_fd);
        ::close(dir_fd);
        if (synced != 0)
            throw std::runtime_error("cannot sync shared model identity directory");
    } catch (...) {
        if (opened)
            ::close(fd);
        if (!renamed)
            ::unlink(temp.data());
        throw;
    }
}

} // namespace

std::filesystem::path model_identity_cache_path(const std::filesystem::path& model,
                                                const std::filesystem::path& work_root) {
    const auto canonical = std::filesystem::canonical(model);
    return shared_cache_dir(work_root) / (sha256_hex(canonical.string()) + ".json");
}

ModelIdentity cached_model_identity(const std::filesystem::path& model, const std::filesystem::path& work_root,
                                    const FullFileReader& full_reader, const ModelIdentity* verified) {
    // Keep the historic exact stamp: size, device, inode, mtime_ns, ctime_ns.
    // Cache files are independent of any run dir, config or manifest index.
    const auto canonical = std::filesystem::canonical(model);
    const auto cache = shared_cache_dir(work_root) / (sha256_hex(canonical.string()) + ".json");
    const FileStamp before = file_stamp(model);
    const auto recorded = read_shared_model_identity(cache, canonical.string());
    if (recorded && recorded->stamp == before) {
        if (!(file_stamp(model) == before) || std::filesystem::canonical(model) != canonical)
            throw std::runtime_error("model artifact changed while reading cached identity");
        return *recorded;
    }
    if (verified && verified->stamp == before) {
        check_digest(verified->digest, "verified model identity");
        if (!(file_stamp(model) == before) || std::filesystem::canonical(model) != canonical)
            throw std::runtime_error("model artifact changed while reusing verified identity");
        write_shared_model_identity(cache, canonical.string(), *verified);
        return *verified;
    }
    if (!full_reader)
        std::cerr << "plan: fingerprinting " << before.size << " bytes of model content\n";
    const std::string digest = full_reader ? full_reader.require("stored.model.full_fingerprint", model)
                                           : complete_file_identity(model);
    check_digest(digest, "full model identity");
    if (!(file_stamp(model) == before) || std::filesystem::canonical(model) != canonical)
        throw std::runtime_error("model artifact changed while calculating full identity");
    const ModelIdentity result{digest, before};
    write_shared_model_identity(cache, canonical.string(), result);
    return result;
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
    Json inputs = {{"version", in.native_config ? 3 : 2},
                   {"effective_values", std::move(static_values)},
                   {"operator_sets", in.operator_sets},
                   {"ordered_devices", in.ordered_devices},
                   {"producer", in.producer_digest},
                   {"engine", in.engine_digest},
                   {"helper", in.helper_digest},
                   {"model_artifact", in.model_digest},
                   {"power_profile", in.power_profile},
                   {"inputs", in.file_digests}};
    if (in.native_config) {
        const auto& native = *in.native_config;
        check_digest(native.yaml_sha256, "static native YAML");
        if (!std::filesystem::path(native.resolution_root).is_absolute() ||
            !std::filesystem::path(native.runtime_home).is_absolute())
            throw std::runtime_error("static native identity requires absolute execution and runtime directories");
        inputs["native_config"] = {{"sha256", native.yaml_sha256}, {"resolution_root", native.resolution_root},
                                    {"runtime_home", native.runtime_home}};
    }
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
        j.size() != 4U + size_t(j.contains("model_digest")) +
                        size_t(j.contains("s8_digest")) + size_t(j.contains("native_measurement")) ||
        (j.contains("model_digest") && !j.at("model_digest").is_string()) ||
        (j.contains("s8_digest") && !j.at("s8_digest").is_string()))
        throw std::runtime_error("stored manifest index has an unknown or missing field");
    StoredManifestIndex result{j.at("key").get<std::string>(), j.at("manifest").get<std::string>(),
                               j.at("digest").get<std::string>(), j.value("model_digest", std::string())};
    if (const auto native = j.find("native_measurement"); native != j.end()) {
        if (!native->is_object() || native->size() != 2 ||
            !native->contains("attempt_id") || !native->at("attempt_id").is_string() ||
            !native->contains("sha256") || !native->at("sha256").is_string())
            throw std::runtime_error("stored native measurement has an invalid shape");
        result.native_measurement = StoredManifestIndex::NativeMeasurement{
            native->at("attempt_id").get<std::string>(), native->at("sha256").get<std::string>()};
    }
    result.s8_digest = j.value("s8_digest", std::string());
    if (!result.s8_digest.empty())
        check_digest(result.s8_digest, "stored S8 writer evidence");
    check_digest(result.key, "stored key");
    check_digest(result.digest, "stored digest");
    if (!result.model_digest.empty())
        check_digest(result.model_digest, "stored model artifact");
    check_filename(result.manifest_filename);
    check_native_measurement(result);
    return result;
}

void write_stored_index(const std::filesystem::path& index, const StoredManifestIndex& entry) {
    check_digest(entry.key, "stored key");
    check_digest(entry.digest, "stored digest");
    if (!entry.model_digest.empty())
        check_digest(entry.model_digest, "stored model artifact");
    check_filename(entry.manifest_filename);
    check_native_measurement(entry);
    Json record = {
        {"version", 1}, {"key", entry.key}, {"manifest", entry.manifest_filename}, {"digest", entry.digest}};
    if (entry.native_measurement)
        record["native_measurement"] = {{"attempt_id", entry.native_measurement->attempt_id},
                                        {"sha256", entry.native_measurement->sha256}};
    if (!entry.model_digest.empty())
        record["model_digest"] = entry.model_digest;
    if (!entry.s8_digest.empty()) {
        check_digest(entry.s8_digest, "stored S8 writer evidence");
        record["s8_digest"] = entry.s8_digest;
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
