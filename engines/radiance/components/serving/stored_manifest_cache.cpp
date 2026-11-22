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

void check_directory(const std::string& directory) {
    // Empty: the manifest sits beside its own index. Otherwise the run directory that wrote it.
    if (!directory.empty() && (!std::filesystem::path(directory).is_absolute() || directory.size() > 4096 ||
                               directory.find('\0') != std::string::npos))
        throw std::runtime_error("stored manifest index has an unsafe run directory");
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

constexpr size_t kMaxInputDigests = 64; // a bound on what an index may carry, not a tuning value
constexpr size_t kMaxInputName = 300;
constexpr size_t kMaxIndexBytes = 8192; // the read bound below

int64_t nanoseconds(const timespec& timestamp) {
    constexpr int64_t kBillion = 1000000000LL;
    if (timestamp.tv_nsec < 0 || timestamp.tv_nsec >= kBillion)
        throw std::runtime_error("file metadata has an invalid nanosecond timestamp");
    const int64_t sec = static_cast<int64_t>(timestamp.tv_sec);
    if (sec > (INT64_MAX - timestamp.tv_nsec) / kBillion || sec < (INT64_MIN + timestamp.tv_nsec) / kBillion)
        throw std::runtime_error("file metadata timestamp is out of range");
    return (sec * kBillion) + timestamp.tv_nsec;
}

// The stat identity a file must keep while it is being fingerprinted.
struct FileStamp {
    uint64_t device = 0, inode = 0, size = 0;
    int64_t mtime_ns = 0, ctime_ns = 0;
    bool operator==(const FileStamp& rhs) const {
        return device == rhs.device && inode == rhs.inode && size == rhs.size && mtime_ns == rhs.mtime_ns &&
               ctime_ns == rhs.ctime_ns;
    }
};

FileStamp file_stamp(const std::filesystem::path& path) {
    struct stat metadata{};
    if (::stat(path.c_str(), &metadata) != 0 || !S_ISREG(metadata.st_mode) || metadata.st_size < 0)
        throw std::runtime_error("static input must be a stat-able ordinary file " + path.string());
    return FileStamp{static_cast<uint64_t>(metadata.st_dev), static_cast<uint64_t>(metadata.st_ino),
                     static_cast<uint64_t>(metadata.st_size), nanoseconds(metadata.st_mtim),
                     nanoseconds(metadata.st_ctim)};
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

std::string model_stat_identity(const std::filesystem::path& model) {
    // Path, size and mtime: the model is never read. A model replaced in place keeping both size and mtime is
    // not detected here; the engine checks the container it loads, and its admission refuses a plan that does
    // not fit.
    std::error_code error;
    const auto canonical = std::filesystem::canonical(model, error);
    struct stat metadata{};
    if (error || ::stat(canonical.c_str(), &metadata) != 0 || !S_ISREG(metadata.st_mode) || metadata.st_size < 0)
        throw std::runtime_error("model artifact must be a stat-able ordinary file " + model.string());
    return sha256_hex("bigcherry/model-stat-v1\n" + canonical.string() + "\n" +
                      std::to_string(static_cast<uint64_t>(metadata.st_size)) + "\n" +
                      std::to_string(nanoseconds(metadata.st_mtim)) + "\n");
}

namespace {
Json static_key_inputs(const StaticResolutionInputs& in) {

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
    check_digest(in.model_digest, "static resolution key");
    // One projection for BOTH the new writer and stored replay. The owner
    // supplies original resolved operator/config/variant values; S8's derived
    // host pool/KV and the per-start POSIX transport never enter static
    // identity. Explicit CLI memory overrides remain in operator_sets, and
    // changed site/config bytes remain in file_digests.
    Json static_values = in.effective_values;
    // Which installed build runs the plan is provenance, not identity: a different build with the same inputs
    // reuses the plan (the engine's admission still refuses one that does not fit).
    if (static_values.contains("engine") && static_values["engine"].is_object())
        static_values["engine"].erase("radiance_home");
    if (static_values.contains("handoff") && static_values["handoff"].is_object()) {
        auto& handoff = static_values["handoff"];
        handoff.erase("shm");
        // The heat file enters through file_digests by content; its path is the installed prefix's location.
        handoff.erase("heat");
        // Where live heat is written (by default under the start's run directory) is per-start; only whether it
        // is recorded ("off" or not) can decide the plan.
        if (handoff.contains("histogram") && handoff["histogram"].is_string() &&
            handoff["histogram"].get<std::string>() != "off")
            handoff["histogram"] = "on";
    }
    if (static_values.contains("memory") && static_values["memory"].is_object()) {
        static_values["memory"].erase("host_pool_mib");
        static_values["memory"].erase("host_kv_mib");
    }
    Json inputs = {{"version", 4},
                   {"effective_values", std::move(static_values)},
                   {"operator_sets", in.operator_sets},
                   {"ordered_devices", in.ordered_devices},
                   {"model_artifact", in.model_digest},
                   {"power_profile", in.power_profile},
                   {"inputs", in.file_digests}};
    for (const auto& [key, digest] : in.file_digests) {
        if (key.empty())
            throw std::runtime_error("static resolution key has an unnamed file");
        check_digest(digest, "static input file");
    }
    return inputs;
}
} // namespace

std::string static_resolution_key(const StaticResolutionInputs& in) {
    std::string canonical, error;
    if (!jcs_canonicalize(static_key_inputs(in), &canonical, &error))
        throw std::runtime_error("static resolution key cannot canonicalize: " + error);
    return sha256_hex(canonical);
}

std::map<std::string, std::string> static_input_digests(const StaticResolutionInputs& in) {
    const Json inputs = static_key_inputs(in);
    std::map<std::string, std::string> digests;
    auto digest_of = [](const Json& value) {
        std::string canonical, error;
        if (!jcs_canonicalize(value, &canonical, &error))
            throw std::runtime_error("static input digest cannot canonicalize: " + error);
        return sha256_hex(canonical);
    };
    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
        if (it.key() == "version")
            continue;
        if (it.key() == "inputs") {
            // A file is named by its role in the config, never by its path.
            for (auto file = it.value().begin(); file != it.value().end(); ++file)
                digests["file:" + file.key()] = file.value().get<std::string>();
        } else {
            digests[it.key()] = digest_of(it.value());
        }
    }
    return digests;
}

std::filesystem::path named_store_index(const std::filesystem::path& work_root, const std::string& config) {
    if (config.empty())
        throw std::runtime_error("named config is required for stored manifest reuse");
    if (work_root.empty() || !work_root.is_absolute())
        throw std::runtime_error("the named-config store needs the absolute work root");
    return work_root / "cache" / "named-configs" / (sha256_hex(config) + ".json");
}

StoredManifestIndex read_stored_index(const std::filesystem::path& index) {
    std::ifstream in(index, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open stored manifest index " + index.string());
    // Read at most one byte beyond the protocol bound. An untrusted sparse
    // or corrupt index must never determine our allocation size.
    std::string bytes(kMaxIndexBytes + 1U, '\0');
    in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    const auto consumed = in.gcount();
    if (in.bad())
        throw std::runtime_error("cannot read stored manifest index " + index.string());
    if (consumed > static_cast<std::streamsize>(kMaxIndexBytes))
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
                        size_t(j.contains("s8_digest")) + size_t(j.contains("native_measurement")) +
                        size_t(j.contains("directory")) + size_t(j.contains("input_digests")) ||
        (j.contains("model_digest") && !j.at("model_digest").is_string()) ||
        (j.contains("directory") && !j.at("directory").is_string()) ||
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
    if (const auto inputs = j.find("input_digests"); inputs != j.end()) {
        // Diagnostic only: a malformed record must not turn an otherwise valid entry into a miss.
        if (inputs->is_object() && inputs->size() <= kMaxInputDigests)
            for (auto it = inputs->begin(); it != inputs->end(); ++it)
                if (it.value().is_string() && hex_digest(it.value().get<std::string>()) &&
                    it.key().size() <= kMaxInputName)
                    result.input_digests[it.key()] = it.value().get<std::string>();
    }
    result.s8_digest = j.value("s8_digest", std::string());
    result.directory = j.value("directory", std::string());
    check_directory(result.directory);
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
    check_directory(entry.directory);
    check_native_measurement(entry);
    Json record = {
        {"version", 1}, {"key", entry.key}, {"manifest", entry.manifest_filename}, {"digest", entry.digest}};
    if (entry.native_measurement)
        record["native_measurement"] = {{"attempt_id", entry.native_measurement->attempt_id},
                                        {"sha256", entry.native_measurement->sha256}};
    if (!entry.directory.empty())
        record["directory"] = entry.directory;
    if (!entry.model_digest.empty())
        record["model_digest"] = entry.model_digest;
    if (!entry.s8_digest.empty()) {
        check_digest(entry.s8_digest, "stored S8 writer evidence");
        record["s8_digest"] = entry.s8_digest;
    }
    std::string bytes = record.dump(2) + "\n";
    // Per-input digests only explain a later miss: when they would push the index past its read bound, the entry
    // is written without them rather than not at all.
    if (!entry.input_digests.empty()) {
        Json with = record;
        with["input_digests"] = entry.input_digests;
        const std::string longer = with.dump(2) + "\n";
        if (longer.size() <= kMaxIndexBytes)
            bytes = longer;
    }
    std::filesystem::create_directories(index.parent_path());
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

const char* plan_reuse_outcome_name(PlanReuseOutcome outcome) {
    switch (outcome) {
    case PlanReuseOutcome::hit: return "hit";
    case PlanReuseOutcome::absent: return "absent";
    case PlanReuseOutcome::index_unreadable: return "index_unreadable";
    case PlanReuseOutcome::static_key_changed: return "static_key_changed";
    case PlanReuseOutcome::manifest_missing: return "manifest_missing";
    case PlanReuseOutcome::integrity_mismatch: return "integrity_mismatch";
    case PlanReuseOutcome::carried_plan_unusable: return "carried_plan_unusable";
    case PlanReuseOutcome::live_admission_refused: return "live_admission_refused";
    case PlanReuseOutcome::carried: return "carried";
    }
    return "unknown";
}

std::string plan_reuse_line(const PlanReuse& reuse) {
    if (reuse.outcome == PlanReuseOutcome::hit)
        return "bc-serve: stored plan reused (static inputs unchanged; manifest " + reuse.manifest.filename().string() + ")";
    if (reuse.outcome == PlanReuseOutcome::carried)
        return "bc-serve: stored plan carried (same release at another location; measurement reused, sealed again here)";
    std::string why;
    switch (reuse.outcome) {
    case PlanReuseOutcome::absent: why = "no stored entry"; break;
    case PlanReuseOutcome::index_unreadable: why = "stored entry unreadable"; break;
    case PlanReuseOutcome::static_key_changed: {
        why = "static key changed";
        if (!reuse.inputs_recorded) {
            why += ": the entry predates per-input records";
        } else if (reuse.changed_inputs.empty()) {
            why += ": no recorded input differs";
        } else {
            why += ": ";
            for (size_t i = 0; i < reuse.changed_inputs.size() && i < kReuseLineInputs; ++i)
                why += (i ? ", " : "") + reuse.changed_inputs[i];
            if (reuse.changed_inputs.size() > kReuseLineInputs)
                why += " and " + std::to_string(reuse.changed_inputs.size() - kReuseLineInputs) + " more";
        }
        break;
    }
    case PlanReuseOutcome::manifest_missing: why = "stored manifest missing"; break;
    case PlanReuseOutcome::integrity_mismatch: why = "stored manifest bytes changed"; break;
    case PlanReuseOutcome::carried_plan_unusable: why = "stored plan unusable: " + reuse.detail; break;
    case PlanReuseOutcome::live_admission_refused: why = "live admission refused: " + reuse.detail; break;
    case PlanReuseOutcome::hit:
    case PlanReuseOutcome::carried: break;
    }
    return "bc-serve: no stored plan reused (" + why + "); measuring";
}

nlohmann::json plan_reuse_json(const PlanReuse& reuse) {
    Json j = {{"outcome", plan_reuse_outcome_name(reuse.outcome)}};
    if (reuse.outcome == PlanReuseOutcome::hit)
        j["manifest"] = reuse.manifest.filename().string();
    if (reuse.outcome == PlanReuseOutcome::static_key_changed) {
        j["changed_inputs"] = reuse.changed_inputs;
        j["inputs_recorded"] = reuse.inputs_recorded;
    }
    if (!reuse.detail.empty())
        j["detail"] = reuse.detail;
    return j;
}

PlanReuse lookup_stored_manifest(const std::filesystem::path& index, const std::string& static_key,
                                 const std::map<std::string, std::string>& current_inputs) {
    check_digest(static_key, "static key");
    // Whatever is wrong with the store is a classified miss, never an error: the caller measures and stores a
    // new one. Nothing read from the store is echoed beyond input names this start computed itself.
    PlanReuse reuse;
    try {
        struct stat st{};
        if (::lstat(index.c_str(), &st) != 0) {
            reuse.outcome = errno == ENOENT ? PlanReuseOutcome::absent : PlanReuseOutcome::index_unreadable;
            return reuse;
        }
        if (!S_ISREG(st.st_mode)) {
            reuse.outcome = PlanReuseOutcome::index_unreadable;
            return reuse;
        }
        const auto entry = read_stored_index(index);
        if (entry.key != static_key) {
            reuse.outcome = PlanReuseOutcome::static_key_changed;
            reuse.inputs_recorded = !entry.input_digests.empty() && !current_inputs.empty();
            if (reuse.inputs_recorded) {
                // Names come from this start's own inputs, never from the stored record's text.
                for (const auto& [name, digest] : current_inputs) {
                    const auto was = entry.input_digests.find(name);
                    if (was == entry.input_digests.end() || was->second != digest)
                        reuse.changed_inputs.push_back(name + (was == entry.input_digests.end() ? " (new)" : ""));
                }
                for (const auto& [name, digest] : entry.input_digests)
                    if (!current_inputs.count(name))
                        reuse.changed_inputs.push_back("(removed input)");
            }
            return reuse;
        }
        if (entry.directory.empty()) {
            reuse.outcome = PlanReuseOutcome::manifest_missing;
            return reuse;
        }
        const auto saved = std::filesystem::path(entry.directory) / entry.manifest_filename;
        if (::lstat(saved.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
            reuse.outcome = PlanReuseOutcome::manifest_missing;
            return reuse;
        }
        if (complete_file_identity(saved) != entry.digest) {
            reuse.outcome = PlanReuseOutcome::integrity_mismatch;
            return reuse;
        }
        reuse.outcome = PlanReuseOutcome::hit;
        reuse.manifest = saved;
        return reuse;
    } catch (const std::exception&) {
        reuse = PlanReuse{};
        reuse.outcome = PlanReuseOutcome::index_unreadable;
        return reuse;
    }
}

} // namespace bc::serving::session
