#include "release_install.h"

#include "release_archive.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

namespace bc::release {
namespace {
constexpr const char* kIndexSchema = "bc-radiance-install-index-v1";
constexpr const char* kReceiptSchema = "bc-radiance-installed-v1";

[[noreturn]] void fail(ErrorCode code, const std::string& message) {
    throw Error(code, message);
}

void require(bool test, ErrorCode code, const std::string& message) {
    if (!test)
        fail(code, message);
}

void ensure_directory(const fs::path& path) {
    std::error_code ec;
    const auto before = fs::symlink_status(path, ec);
    if (!ec && fs::exists(before)) {
        require(fs::is_directory(before) && !fs::is_symlink(before), ErrorCode::conflict,
                "release storage directory is not a plain directory: " + path.string());
        return;
    }
    fs::create_directories(path);
    const auto after = fs::symlink_status(path);
    require(fs::is_directory(after) && !fs::is_symlink(after), ErrorCode::conflict,
            "unsafe release storage directory: " + path.string());
}

// Only a single process can mutate or verify a root at a time, including a
// rollback. This is a state lock, not a permission-based trust assertion.
class RootLock final {
  public:
    explicit RootLock(const fs::path& root) {
        ensure_directory(root);
        const auto lock = root / ".bc-release.lock";
        fd_ = ::open(lock.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
        require(fd_ >= 0, ErrorCode::io, "cannot open release root lock");
        struct stat st{};
        require(::fstat(fd_, &st) == 0 && S_ISREG(st.st_mode), ErrorCode::conflict,
                "release root lock is not regular");
        require(::flock(fd_, LOCK_EX) == 0, ErrorCode::io, "cannot lock release root");
    }

    ~RootLock() {
        if (fd_ >= 0) {
            ::flock(fd_, LOCK_UN);
            ::close(fd_);
        }
    }
    RootLock(const RootLock&) = delete;
    RootLock& operator=(const RootLock&) = delete;

  private:
    int fd_ = -1;
};

std::string installed_at() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    ::gmtime_r(&time, &utc);
    std::ostringstream stream;
    stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

bool valid_tag(const std::string& tag) {
    static const std::regex pattern(R"(^bc-radiance-[0-9]+\.[0-9]+\.[0-9]+$)");
    return std::regex_match(tag, pattern);
}

// Compare numeric components without overflow. Numeric aliases still need consent.
int compare_versions(const std::string& candidate, const std::string& current) {
    static const std::regex pattern(R"(^bc-radiance-([0-9]+)\.([0-9]+)\.([0-9]+)$)");
    std::smatch a, b;
    require(std::regex_match(candidate, a, pattern) && std::regex_match(current, b, pattern),
            ErrorCode::invalid_input, "invalid version for release order");
    for (int i = 1; i <= 3; ++i) {
        std::string x = a[i].str(), y = b[i].str();
        const auto xi = x.find_first_not_of('0'), yi = y.find_first_not_of('0');
        x.erase(0, xi == std::string::npos ? x.size() - 1 : xi);
        y.erase(0, yi == std::string::npos ? y.size() - 1 : yi);
        if (x.size() != y.size())
            return x.size() < y.size() ? -1 : 1;
        if (x != y)
            return x < y ? -1 : 1;
    }
    return 0;
}

void validate_store_tag(const std::string& tag) {
    require(valid_tag(tag), ErrorCode::invalid_install, "invalid tag in release state");
}

Json empty_index() {
    return {{"schema", kIndexSchema}, {"current", nullptr}, {"previous", nullptr}};
}

Json load_index(const fs::path& root) {
    const auto path = root / "index.json";
    std::error_code ec;
    const auto info = fs::symlink_status(path, ec);
    if (ec == std::errc::no_such_file_or_directory || (!ec && !fs::exists(info)))
        return empty_index();
    require(!ec && fs::is_regular_file(info) && !fs::is_symlink(info), ErrorCode::invalid_install,
            "install index is not a plain file");
    const auto index = read_json(path);
    require(index.is_object() && index.value("schema", "") == kIndexSchema && index.contains("current") &&
                index.contains("previous"),
            ErrorCode::invalid_install, "malformed install index");
    require(index["current"].is_string() && (index["previous"].is_null() || index["previous"].is_string()),
            ErrorCode::invalid_install, "invalid index current/previous entries");
    const auto current = index["current"].get<std::string>();
    validate_store_tag(current);
    if (index["previous"].is_string()) {
        const auto previous = index["previous"].get<std::string>();
        validate_store_tag(previous);
        require(previous != current, ErrorCode::invalid_install,
                "current and previous install tags must differ");
    }
    return index;
}

Json load_receipt(const fs::path& root, const std::string& tag) {
    validate_store_tag(tag);
    const auto prefix = root / "releases" / tag;
    require(fs::is_directory(prefix) && !fs::is_symlink(fs::symlink_status(prefix)), ErrorCode::invalid_install,
            "installed engine prefix missing or unsafe: " + tag);
    const auto receipt = read_json(prefix / "install.json");
    require(receipt.is_object() && receipt.value("schema", "") == kReceiptSchema &&
                receipt.value("tag", "") == tag,
            ErrorCode::invalid_install, "installed release receipt does not match tag: " + tag);
    require(receipt.contains("record") && receipt["record"].is_object(), ErrorCode::invalid_install,
            "installed release receipt lacks tag-bound record");
    const auto& record = receipt.at("record");
    require(required_text(record, "release_tag", ErrorCode::wrong_identity) == tag &&
                required_text(record, "product", ErrorCode::wrong_identity) == "bigcherryrad" &&
                required_text(record, "payload_scope", ErrorCode::wrong_identity) == "engine-runtime",
            ErrorCode::wrong_identity, "stored record has wrong tag/product/scope");
    for (const auto& [receipt_key, record_key] : {
             std::pair{"archive_sha256", "sha256"},
             std::pair{"manifest_sha256", "manifest_sha256"},
             std::pair{"asset_name", "name"},
             std::pair{"engine_tree", "engine_tree"},
             std::pair{"config_tree", "config_tree"},
             std::pair{"rocm_version", "rocm_version"},
         }) {
        require(required_text(receipt, receipt_key, ErrorCode::invalid_install) ==
                    required_text(record, record_key, ErrorCode::wrong_identity),
                ErrorCode::wrong_identity, std::string("stored receipt/record mismatch: ") + receipt_key);
    }
    require(required_text(receipt, "prefix", ErrorCode::invalid_install) ==
                fs::absolute(prefix).lexically_normal().string(),
            ErrorCode::wrong_identity, "stored install prefix changed");
    return receipt;
}

Json verify_one(const fs::path& root, const fs::path& sdk, const std::string& tag, bool trust_stamps = false) {
    const auto prefix = root / "releases" / tag;
    const auto receipt = load_receipt(root, tag);
    const auto& record = receipt.at("record");
    const auto sdk_version = sdk / ".info/version";
    require(fs::is_regular_file(sdk_version), ErrorCode::sdk_mismatch, "host ROCm SDK lacks version file");
    std::ifstream input(sdk_version);
    std::string version;
    std::getline(input, version);
    require(version == required_text(record, "rocm_version", ErrorCode::wrong_identity),
            ErrorCode::sdk_mismatch, "installed engine SDK differs from selected host SDK");
    const auto manifest = read_json(prefix / "bundle.json");
    verify_installed_files(prefix, record, manifest, tag, root / "stamps" / (tag + ".json"), trust_stamps);
    require(required_text(receipt, "engine_sha256", ErrorCode::wrong_identity) ==
                    required_text(manifest["files"]["bin/radiance"], "sha256", ErrorCode::wrong_identity) &&
                required_text(receipt, "bc_serve_sha256", ErrorCode::wrong_identity) ==
                    required_text(manifest["files"]["radiance_home/bin/bc-serve"], "sha256",
                                  ErrorCode::wrong_identity),
            ErrorCode::wrong_identity, "installed matched engine/serving identities changed");
    return receipt;
}

Json checked_index(const fs::path& root, const fs::path& sdk) {
    const auto index = load_index(root);
    if (!index["current"].is_null()) {
        verify_one(root, sdk, index["current"].get<std::string>());
        if (index["previous"].is_string())
            verify_one(root, sdk, index["previous"].get<std::string>());
    }
    return index;
}

void prune_old_previous(const fs::path& root, const Json& old_index, const Json& updated) {
    if (!old_index["previous"].is_string())
        return;
    const auto old = old_index["previous"].get<std::string>();
    if (updated["current"] == old || updated["previous"] == old)
        return;
    const auto obsolete = root / "releases" / old;
    fs::remove(root / "stamps" / (old + ".json"));
    // The old previous was verified before the index update; do not blindly
    // delete unknown directories or anything pointed to by a symlink.
    require(!fs::is_symlink(fs::symlink_status(obsolete)), ErrorCode::conflict,
            "obsolete release prefix unexpectedly became a symlink");
    fs::remove_all(obsolete);
    fsync_dir(root / "releases");
}

} // namespace

Error::Error(ErrorCode code, const std::string& message) : std::runtime_error(message), code_(code) {}

void install(const InstallRequest& request) {
    // The selected exact tag, not "latest", binds the supplied release record
    // and archive. No source checkout, executable shell or Python import.
    const auto record = read_json(request.record);
    validate_release_record(record, request.tag, request.archive, request.sdk);
    const auto root = fs::absolute(request.root).lexically_normal();
    RootLock lock(root);
    // An interrupted installation is never inferred to be complete or
    // silently reused. The operator must inspect/remove its staging tree.
    for (const auto& entry : fs::directory_iterator(root)) {
        const auto name = entry.path().filename().string();
        require(name.rfind(".staging-", 0) != 0, ErrorCode::invalid_install,
                "interrupted release staging tree must be resolved: " + name);
    }
    ensure_directory(root / "releases");
    const auto before = checked_index(root, request.sdk);
    // Refuse unauthorized version switches before extracting a candidate.
    if (before["current"].is_string()) {
        const auto active = before["current"].get<std::string>();
        if (active != request.tag && compare_versions(request.tag, active) <= 0)
            require(request.allow_downgrade, ErrorCode::conflict,
                    "downgrade or same-version identity change requires --allow-downgrade true");
    }
    const auto final = root / "releases" / request.tag;
    if (fs::exists(final) || fs::is_symlink(fs::symlink_status(final))) {
        const auto existing = verify_one(root, request.sdk, request.tag);
        require(existing.at("record") == record, ErrorCode::conflict,
                "existing installed tag belongs to different release record");
    } else {
        const auto stage = root / (".staging-" + nonce());
        ensure_directory(stage);
        try {
            const auto checked = extract_verified(record, request.tag, request.archive, stage);
            const Json receipt = {
                {"schema", kReceiptSchema},
                {"tag", request.tag},
                {"asset_name", required_text(record, "name", ErrorCode::wrong_identity)},
                {"archive_sha256", required_text(record, "sha256", ErrorCode::wrong_identity)},
                {"manifest_sha256", required_text(record, "manifest_sha256", ErrorCode::wrong_identity)},
                {"engine_sha256",
                 required_text(checked.manifest["files"]["bin/radiance"], "sha256", ErrorCode::wrong_identity)},
                {"bc_serve_sha256", required_text(checked.manifest["files"]["radiance_home/bin/bc-serve"],
                                                  "sha256", ErrorCode::wrong_identity)},
                {"engine_tree", required_text(record, "engine_tree", ErrorCode::wrong_identity)},
                {"config_tree", required_text(record, "config_tree", ErrorCode::wrong_identity)},
                {"rocm_version", required_text(record, "rocm_version", ErrorCode::wrong_identity)},
                {"prefix", fs::absolute(final).lexically_normal().string()},
                {"installed_at", installed_at()},
                {"record", record},
            };
            // Last file to be committed to the staged prefix: its verified
            // metadata. The index publication is still the final transaction.
            write_atomic(stage / "install.json", receipt.dump(2) + "\n");
            fs::rename(stage, final);
            fsync_dir(root / "releases");
            verify_one(root, request.sdk, request.tag);
        } catch (...) {
            if (fs::exists(stage))
                fs::remove_all(stage);
            throw;
        }
    }
    const auto& current = before["current"];
    if (current == request.tag)
        return;
    Json updated = before;
    updated["current"] = request.tag;
    updated["previous"] = current;
    write_atomic(root / "index.json", updated.dump(2) + "\n");
    prune_old_previous(root, before, updated);
}

Json verify(const fs::path& source, const fs::path& sdk) {
    const auto root = fs::absolute(source).lexically_normal();
    RootLock lock(root);
    const auto index = checked_index(root, sdk);
    require(index["current"].is_string(), ErrorCode::invalid_install, "no current installed engine");
    return {{"status", "verified"},
            {"current", index["current"]},
            {"previous", index["previous"]},
            {"root", root.string()}};
}

Json verify_current(const fs::path& source, const fs::path& sdk) {
    const auto root = fs::absolute(source).lexically_normal();
    RootLock lock(root);
    const auto index = load_index(root);
    require(index["current"].is_string(), ErrorCode::invalid_install, "no current installed engine");
    verify_one(root, sdk, index["current"].get<std::string>(), true);
    return {{"status", "verified"}, {"current", index["current"]}, {"root", root.string()}};
}

Json rollback(const fs::path& source, const fs::path& sdk) {
    const auto root = fs::absolute(source).lexically_normal();
    RootLock lock(root);
    auto index = checked_index(root, sdk);
    require(index["current"].is_string() && index["previous"].is_string(), ErrorCode::invalid_install,
            "no verified previous release to roll back to");
    const auto old = index["current"];
    index["current"] = index["previous"];
    index["previous"] = old;
    write_atomic(root / "index.json", index.dump(2) + "\n");
    return {{"status", "rolled-back"},
            {"current", index["current"]},
            {"previous", index["previous"]},
            {"root", root.string()}};
}

} // namespace bc::release
