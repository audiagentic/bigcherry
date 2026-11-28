#include "release_archive.h"

#include "file_stamp.h"

#include "sha256.h"
#include "v2/document_bytes.h"

#include <zlib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bc::release {
namespace {
std::atomic<uint64_t> hashed_file_count{0};
constexpr size_t kIoChunk = size_t{1024} * 1024;
constexpr uint64_t kMaxBytes = 16ULL * kIoChunk * 1024ULL;
constexpr size_t kMaxFiles = 10000;
constexpr uint64_t kMaxManifest = 2ULL * kIoChunk;
constexpr uint64_t kMaxPax = 64ULL * 1024;
constexpr size_t kBlock = 512;
// The native profiles a release may carry: configs/NAME.yaml, one per named serving config, exactly the set the packer
// approves (tools/bigcherry/release/runtime_assets.py NATIVE_RELEASE_PROFILES; a test keeps the two lists equal).
constexpr std::array<std::string_view, 20> kNativeProfiles = {
    "flashnext-2xtx-combine",
    "flashnext-2xtx-rdna2-head",
    "flashnext-2xtx-rdna2-tail-combine",
    "flashnext-2xtx-rdna2-tail",
    "flashnext-2xtx",
    "flashnext-4gpu-bf16i8",
    "flashnext-4gpu",
    "flashnext-r9700",
    "flashnext-tp2xtx-r9700",
    "flashnext-tp2xtx",
    "qwen27b-2xtx",
    "qwen27b-tp3-capacity",
    "qwen27b-tp3-iq6r",
    "qwen27b-tp3-long-bf16",
    "qwen27b-tp3-long-bf16i8",
    "qwen27b-tp3-long-iq4r",
    "qwen27b-tp3-long-iq6r-bf16i8",
    "qwen27b-tp3-long-iq6r",
    "qwen27b-tp3-long",
    "qwen27b-tp3",
};
bool is_native_profile(std::string_view name) {
    constexpr std::string_view head = "configs/", tail = ".yaml";
    if (!name.starts_with(head) || !name.ends_with(tail) || name.size() <= head.size() + tail.size())
        return false;
    const std::string_view stem = name.substr(head.size(), name.size() - head.size() - tail.size());
    return std::ranges::find(kNativeProfiles, stem) != kNativeProfiles.end();
}

[[noreturn]] void refuse(ErrorCode code, const std::string& what) {
    throw Error(code, what);
}

void require(bool condition, ErrorCode code, const std::string& what) {
    if (!condition)
        refuse(code, what);
}

bool prefix(const std::string& text, std::string_view start) {
    return text.compare(0, start.size(), start) == 0;
}

bool ends_in_so(const std::string& name) {
    const auto i = name.rfind(".so");
    return i != std::string::npos && (i + 3 == name.size() || name[i + 3] == '.');
}

// Files are individually allowlisted, never an indiscriminate directory
// copy. The file groups are the native equivalent of RUN_MANIFEST.
bool runtime_member(const std::string& name) {
    static const std::set<std::string> exact = {
        "bin/radiance",
        "radiance_home/bin/bc-serve",
        "radiance_home/bin/bc-release",
        "radiance_home/helpers/r3_moe_worker",
        "bigcherry-build-metadata-radiance.json",
        "heat/flashnext-mixed.hot-first",
        "heat/perf51.rank0.hot-first",
    };
    if (exact.contains(name) || is_native_profile(name))
        return true;
    if (prefix(name, "radiance_home/vectors/"))
        return name.size() > std::string("radiance_home/vectors/").size();
    constexpr std::array<std::string_view, 6> directories = {
        "radiance_home/helpers/",    "radiance_home/kernels/", "radiance_home/architectures/",
        "radiance_home/quantizers/", "runtime-prefix/lib/",    "runtime-prefix/lib64/",
    };
    return std::ranges::any_of(directories,
                               [&](std::string_view dir) { return prefix(name, dir) && ends_in_so(name); });
}

bool safe_member(const std::string& name) {
    if (name.empty() || name.front() == '/' || name.back() == '/' || name.find('\\') != std::string::npos)
        return false;
    size_t from = 0;
    while (from < name.size()) {
        const auto next = name.find('/', from);
        const auto part = name.substr(from, next == std::string::npos ? next : next - from);
        if (part.empty() || part == "." || part == "..")
            return false;
        for (unsigned char c : part) {
            if (c < 0x20 || c == 0x7f)
                return false;
        }
        if (next == std::string::npos)
            break;
        from = next + 1;
    }
    return runtime_member(name);
}

bool forbidden_name(const std::string& name) {
    static const std::set<std::string> script_suffix = {".py", ".pyc", ".pyo", ".sh", ".bash", ".zsh"};
    auto path = fs::path(name);
    for (const auto& component : path) {
        if (component == "__pycache__")
            return true;
    }
    std::string leaf = path.filename().string();
    std::transform(leaf.begin(), leaf.end(), leaf.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    if (prefix(leaf, "python") || prefix(leaf, "pypy"))
        return true;
    const auto i = leaf.find_last_of('.');
    return i != std::string::npos && script_suffix.contains(leaf.substr(i));
}

void check_interpreter(const fs::path& path, const std::string& name) {
    require(!forbidden_name(name), ErrorCode::unsafe_archive, "interpreter file: " + name);
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), ErrorCode::io, "cannot inspect extracted file: " + name);
    char shebang[2]{};
    input.read(shebang, 2);
    require(!(input.gcount() == 2 && shebang[0] == '#' && shebang[1] == '!'), ErrorCode::unsafe_archive,
            "script shebang: " + name);
    const auto suffix = fs::path(name).extension().string();
    const bool native_profile = is_native_profile(name);
    if (!native_profile && suffix != ".toml" && suffix != ".json" && suffix != ".txt" &&
        suffix != ".ini" && suffix != ".conf")
        return;
    static const std::regex interpreter(R"(\b(?:python(?:[0-9]+(?:\.[0-9]+)*)?|pypy(?:[0-9]+)?)\b)",
                                        std::regex_constants::icase);
    input.clear();
    input.seekg(0);
    std::string line;
    while (std::getline(input, line)) {
        if (native_profile)
            require(line.find('\0') == std::string::npos &&
                        bc::serving::v2::bad_utf8(line) == std::string_view::npos,
                    ErrorCode::unsafe_archive, "invalid UTF-8 metadata: " + name);
        require(!std::regex_search(line, interpreter), ErrorCode::unsafe_archive,
                "interpreter reference in metadata: " + name);
    }
}

bool is_regular_plain(const fs::path& path) {
    std::error_code ec;
    const auto kind = fs::symlink_status(path, ec);
    return !ec && fs::is_regular_file(kind) && !fs::is_symlink(kind);
}

std::string read_bounded(const fs::path& path, uint64_t limit) {
    require(is_regular_plain(path), ErrorCode::invalid_input, "not a regular file: " + path.string());
    const auto size = fs::file_size(path);
    require(size <= limit, ErrorCode::invalid_input, "input file too large: " + path.string());
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), ErrorCode::io, "cannot read: " + path.string());
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    require(in.good() || in.eof(), ErrorCode::io, "cannot finish reading: " + path.string());
    return bytes;
}

void required_equal(const Json& left, const Json& right, const char* key) {
    require(left.contains(key) && right.contains(key) && left[key] == right[key], ErrorCode::wrong_identity,
            std::string("record/manifest mismatch: ") + key);
}

uint64_t required_size(const Json& document, const char* key, ErrorCode code) {
    require(document.is_object() && document.contains(key) && document[key].is_number_unsigned(), code,
            std::string("missing unsigned field: ") + key);
    const auto value = document[key].get<uint64_t>();
    require(value <= kMaxBytes, code, std::string("oversized field: ") + key);
    return value;
}

std::set<std::string> file_map(const Json& manifest, const Json& record) {
    require(manifest.contains("files") && manifest["files"].is_object(), ErrorCode::invalid_install,
            "missing runtime file map");
    require(record.contains("file_names") && record["file_names"].is_array(), ErrorCode::invalid_install,
            "missing sorted receipt file names");
    const auto& files = manifest["files"];
    std::set<std::string> names;
    std::vector<std::string> sorted;
    for (auto it = files.begin(); it != files.end(); ++it) {
        const auto& name = it.key();
        require(safe_member(name) && !forbidden_name(name), ErrorCode::unsafe_archive,
                "unapproved runtime member: " + name);
        require(it.value().is_object(), ErrorCode::invalid_install, "invalid file record: " + name);
        const auto digest = required_text(it.value(), "sha256", ErrorCode::invalid_install);
        require(std::regex_match(digest, std::regex("[a-f0-9]{64}")), ErrorCode::invalid_install,
                "invalid file digest: " + name);
        const auto mode = required_size(it.value(), "mode", ErrorCode::invalid_install);
        require(mode == 0644 || mode == 0755, ErrorCode::unsafe_archive, "disallowed file mode: " + name);
        const auto size = required_size(it.value(), "size", ErrorCode::invalid_install);
        if (is_native_profile(name))
            require(mode == 0644 && size > 0 && size <= bc::serving::v2::kMaxDocumentBytes,
                    ErrorCode::unsafe_archive, "native profile must be bounded non-executable YAML: " + name);
        names.insert(name);
        sorted.push_back(name);
    }
    const std::set<std::string> must_have = {
        "bin/radiance",
        "radiance_home/bin/bc-serve",
        "radiance_home/bin/bc-release",
        "radiance_home/helpers/r3_moe_worker",
        "bigcherry-build-metadata-radiance.json",
        "configs/flashnext-4gpu.yaml",   // the production config; the other named configs are its siblings
    };
    require(std::includes(names.begin(), names.end(), must_have.begin(), must_have.end()),
            ErrorCode::invalid_install, "required native runtime member missing");
    require(names.size() <= kMaxFiles, ErrorCode::invalid_install, "too many runtime files");
    require(record["file_names"] == sorted, ErrorCode::wrong_identity,
            "receipt file_names differs from manifest");
    require(manifest["files"]["bin/radiance"]["sha256"] == record.at("binary_sha256"),
            ErrorCode::wrong_identity, "engine binary digest differs from release record");
    return names;
}

void validate_manifest(const Json& manifest, const Json& record, const std::string& tag) {
    require(manifest.is_object(), ErrorCode::invalid_install, "bundle manifest is not an object");
    for (const char* key : {"schema", "product", "payload_scope", "rocm_version", "engine_tree", "config_tree",
                            "binary_sha256", "plan_id", "source_slice_id"}) {
        required_equal(manifest, record, key);
    }
    // The packer records the pre-strip lane SHA separately from the SHA of
    // the shipped ELF. Older verified receipts have only binary_sha256.
    if (manifest.contains("source_binary_sha256") || record.contains("source_binary_sha256"))
        required_equal(manifest, record, "source_binary_sha256");
    require(required_text(manifest, "product", ErrorCode::wrong_identity) == "bigcherryrad",
            ErrorCode::wrong_identity, "not a BigCherry Radiance engine");
    require(required_text(manifest, "payload_scope", ErrorCode::wrong_identity) == "engine-runtime",
            ErrorCode::wrong_identity, "not a scoped runtime asset");
    const auto source_tag = required_text(manifest, "release_tag", ErrorCode::wrong_identity);
    const auto source_expected =
        record.contains("reused_from") ? required_text(record, "reused_from", ErrorCode::wrong_identity) : tag;
    require(source_tag == source_expected, ErrorCode::wrong_identity, "bundle belongs to another release tag");
    const auto names = file_map(manifest, record);
    (void)names;
}

// A tar header has fixed offsets. We accept only regular files and the PAX
// "path" extension produced by Python's deterministic, PAX-format platform
// packer. Other extension types and links are refused.
uint64_t tar_number(const std::array<char, kBlock>& header, size_t offset, size_t count) {
    require(offset + count <= header.size(), ErrorCode::unsafe_archive, "bad tar numeric field");
    const auto first = static_cast<unsigned char>(header[offset]);
    if (first & 0x80U) {
        uint64_t value = first & 0x7fU;
        for (size_t i = offset + 1; i < offset + count; ++i) {
            require(value <= kMaxBytes, ErrorCode::unsafe_archive, "oversized tar number");
            value = (value << 8U) | static_cast<unsigned char>(header[i]);
        }
        require(value <= kMaxBytes, ErrorCode::unsafe_archive, "oversized tar number");
        return value;
    }
    uint64_t value = 0;
    bool saw_digit = false;
    for (size_t i = offset; i < offset + count; ++i) {
        const auto c = header[i];
        if (c == '\0' || c == ' ')
            continue;
        require(c >= '0' && c <= '7', ErrorCode::unsafe_archive, "invalid tar octal field");
        saw_digit = true;
        value = (value * 8U) + static_cast<unsigned>(c - '0');
        require(value <= kMaxBytes, ErrorCode::unsafe_archive, "oversized tar number");
    }
    return saw_digit ? value : 0;
}

std::string tar_text(const std::array<char, kBlock>& header, size_t offset, size_t count) {
    const auto* const begin = header.data() + offset;
    const auto* const end = std::find(begin, begin + count, '\0');
    return std::string(begin, end);
}

bool zero_header(const std::array<char, kBlock>& header) {
    return std::all_of(header.begin(), header.end(), [](char c) { return c == '\0'; });
}

void gz_read_exact(gzFile file, char* data, uint64_t bytes) {
    while (bytes > 0) {
        const auto take = static_cast<unsigned>(std::min<uint64_t>(bytes, kIoChunk));
        const int got = gzread(file, data, take);
        require(got > 0, ErrorCode::unsafe_archive, "truncated or invalid gzip stream");
        data += got;
        bytes -= static_cast<unsigned>(got);
    }
}

void tar_padding(gzFile file, uint64_t size) {
    const auto padding = (kBlock - (size % kBlock)) % kBlock;
    if (padding == 0)
        return;
    std::array<char, kBlock> rest{};
    gz_read_exact(file, rest.data(), padding);
    require(std::all_of(rest.begin(), rest.begin() + static_cast<std::ptrdiff_t>(padding),
                        [](char c) { return c == '\0'; }),
            ErrorCode::unsafe_archive, "nonzero tar member padding");
}

std::map<std::string, std::string> pax_fields(const std::string& data) {
    std::map<std::string, std::string> fields;
    size_t position = 0;
    while (position < data.size()) {
        const auto space = data.find(' ', position);
        require(space != std::string::npos && space - position <= 8, ErrorCode::unsafe_archive,
                "malformed PAX record length");
        const auto len_text = data.substr(position, space - position);
        require(!len_text.empty() && std::all_of(len_text.begin(), len_text.end(),
                                                 [](unsigned char c) { return std::isdigit(c) != 0; }),
                ErrorCode::unsafe_archive, "invalid PAX length");
        const auto length = std::stoul(len_text);
        require(length > space - position + 2 && length <= data.size() - position &&
                    data[position + length - 1] == '\n',
                ErrorCode::unsafe_archive, "truncated PAX record");
        const auto assignment = data.substr(space + 1, position + length - space - 2);
        const auto equal = assignment.find('=');
        require(equal != std::string::npos, ErrorCode::unsafe_archive, "invalid PAX attribute");
        const auto key = assignment.substr(0, equal);
        require(!fields.contains(key), ErrorCode::unsafe_archive, "duplicate PAX attribute");
        fields.emplace(key, assignment.substr(equal + 1));
        position += length;
    }
    return fields;
}

void read_tar_content(gzFile file, const fs::path& dest, uint64_t count, const std::string& want,
                      const std::string& name) {
    std::ofstream out(dest, std::ios::binary | std::ios::trunc);
    require(out.is_open(), ErrorCode::io, "cannot create staged runtime file: " + name);
    serving::Sha256 digest;
    std::array<char, kIoChunk> block{};
    while (count > 0) {
        const auto take = static_cast<unsigned>(std::min<uint64_t>(block.size(), count));
        gz_read_exact(file, block.data(), take);
        out.write(block.data(), take);
        require(out.good(), ErrorCode::io, "cannot write staged runtime file: " + name);
        digest.update(std::string_view(block.data(), take));
        count -= take;
    }
    out.close();
    require(out.good(), ErrorCode::io, "cannot flush staged runtime file: " + name);
    require(digest.hex_digest() == want, ErrorCode::digest_mismatch, "runtime file digest mismatch: " + name);
}

void verify_file_set(const fs::path& prefix_path, const Json& manifest, const Json& record,
                     const std::string& tag, bool installed) {
    validate_manifest(manifest, record, tag);
    const auto expected = file_map(manifest, record);
    std::set<std::string> actual;
    for (auto iter = fs::recursive_directory_iterator(prefix_path); iter != fs::recursive_directory_iterator();
         ++iter) {
        const auto rel = fs::relative(iter->path(), prefix_path).generic_string();
        require(!iter->is_symlink(), ErrorCode::unsafe_archive, "runtime tree contains symlink: " + rel);
        if (iter->is_directory())
            continue;
        require(iter->is_regular_file(), ErrorCode::unsafe_archive, "runtime tree has non-file member: " + rel);
        actual.insert(rel);
    }
    std::set<std::string> full = expected;
    full.insert("bundle.json");
    if (installed)
        full.insert("install.json");
    require(actual == full, ErrorCode::invalid_install, "runtime file membership changed");
    for (const auto& name : expected) {
        const auto& checked = manifest["files"].at(name);
        const auto path = prefix_path / fs::path(name);
        require(is_regular_plain(path), ErrorCode::unsafe_archive, "runtime file is not regular: " + name);
        require(fs::file_size(path) == required_size(checked, "size", ErrorCode::invalid_install),
                ErrorCode::digest_mismatch, "runtime file size changed: " + name);
        require(sha256_file(path) == required_text(checked, "sha256", ErrorCode::invalid_install),
                ErrorCode::digest_mismatch, "installed runtime digest changed: " + name);
        const auto mode = static_cast<unsigned>(fs::status(path).permissions()) & 0777U;
        require(mode == required_size(checked, "mode", ErrorCode::invalid_install), ErrorCode::digest_mismatch,
                "runtime file mode changed: " + name);
        check_interpreter(path, name);
    }
}

} // namespace

void fsync_dir(const fs::path& dir) {
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    require(fd >= 0, ErrorCode::io, "cannot sync release directory: " + dir.string());
    const int outcome = ::fsync(fd);
    ::close(fd);
    require(outcome == 0, ErrorCode::io, "release directory flush failed");
}

std::string nonce() {
    const auto nanos = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::to_string(::getpid()) + "-" + std::to_string(nanos);
}

void write_atomic(const fs::path& dest, const std::string& bytes) {
    const auto pending = fs::path(dest.string() + ".pending-" + nonce());
    const int fd = ::open(pending.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0644);
    require(fd >= 0, ErrorCode::io, "cannot stage release state: " + dest.string());
    try {
        size_t offset = 0;
        while (offset < bytes.size()) {
            const ssize_t written = ::write(fd, bytes.data() + offset, bytes.size() - offset);
            require(written > 0, ErrorCode::io, "cannot write staged release state");
            offset += static_cast<size_t>(written);
        }
        require(::fsync(fd) == 0, ErrorCode::io, "cannot sync staged release state");
    } catch (...) {
        ::close(fd);
        fs::remove(pending);
        throw;
    }
    ::close(fd);
    try {
        fs::rename(pending, dest);
    } catch (...) {
        fs::remove(pending);
        throw;
    }
    fsync_dir(dest.parent_path());
}


std::string required_text(const Json& doc, const char* key, ErrorCode code) {
    require(doc.is_object() && doc.contains(key) && doc[key].is_string(), code,
            std::string("missing text field: ") + key);
    return doc[key].get<std::string>();
}

Json read_json(const fs::path& path) {
    const auto bytes = read_bounded(path, kMaxManifest);
    try {
        return Json::parse(bytes);
    } catch (const Json::exception&) {
        refuse(ErrorCode::invalid_input, "invalid JSON input: " + path.string());
    }
}

std::string sha256_file(const fs::path& path) {
    require(is_regular_plain(path), ErrorCode::invalid_input, "not a regular file: " + path.string());
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), ErrorCode::io, "cannot read file: " + path.string());
    ++hashed_file_count;
    serving::Sha256 digest;
    std::array<char, kIoChunk> bytes{};
    while (input.read(bytes.data(), bytes.size()) || input.gcount() > 0)
        digest.update(std::string_view(bytes.data(), static_cast<size_t>(input.gcount())));
    require(input.eof(), ErrorCode::io, "cannot finish hashing: " + path.string());
    return digest.hex_digest();
}

void validate_release_record(const Json& record, const std::string& tag, const fs::path& archive,
                             const fs::path& sdk) {
    static const std::regex tag_pattern(R"(^bc-radiance-[0-9]+\.[0-9]+\.[0-9]+$)");
    require(std::regex_match(tag, tag_pattern), ErrorCode::invalid_input, "invalid exact Radiance release tag");
    require(record.is_object(), ErrorCode::invalid_input, "release record is not an object");
    require(required_text(record, "release_tag", ErrorCode::wrong_identity) == tag, ErrorCode::wrong_identity,
            "release record belongs to another tag");
    require(required_text(record, "product", ErrorCode::wrong_identity) == "bigcherryrad" &&
                required_text(record, "payload_scope", ErrorCode::wrong_identity) == "engine-runtime",
            ErrorCode::wrong_identity, "release record has wrong product or scope");
    require(required_text(record, "schema", ErrorCode::wrong_identity) == "bigcherry-radiance-runtime-v1",
            ErrorCode::wrong_identity, "unsupported engine release schema");
    const auto filename = required_text(record, "name", ErrorCode::wrong_identity);
    static const std::regex asset_pattern(R"(^bigcherryrad-[a-f0-9]{32}\.tar\.gz$)");
    require(std::regex_match(filename, asset_pattern) && archive.filename() == filename,
            ErrorCode::wrong_identity, "archive name does not match selected tag's release record");
    const auto expected = required_text(record, "sha256", ErrorCode::wrong_identity);
    require(std::regex_match(expected, std::regex("[a-f0-9]{64}")), ErrorCode::wrong_identity,
            "malformed record archive digest");
    require(is_regular_plain(archive) && fs::file_size(archive) <= kMaxBytes, ErrorCode::invalid_input,
            "missing or oversized regular engine archive");
    require(sha256_file(archive) == expected, ErrorCode::digest_mismatch,
            "archive SHA-256 differs from tag-bound release digest");
    require(is_regular_plain(sdk / ".info/version"), ErrorCode::sdk_mismatch,
            "selected host ROCm SDK has no version file");
    auto version = read_bounded(sdk / ".info/version", 1024);
    while (!version.empty() && (version.back() == '\n' || version.back() == '\r' || version.back() == ' '))
        version.pop_back();
    require(version == required_text(record, "rocm_version", ErrorCode::wrong_identity),
            ErrorCode::sdk_mismatch, "selected host ROCm SDK version mismatch");
    if (record.contains("runtime_key")) {
        for (const char* key : {"engine_tree", "config_tree", "rocm_version"})
            required_equal(record.at("runtime_key"), record, key);
    }
}

CheckedAsset extract_verified(const Json& record, const std::string& tag, const fs::path& archive,
                              const fs::path& prefix_path) {
    require(fs::is_directory(prefix_path) && fs::is_empty(prefix_path), ErrorCode::conflict,
            "staging prefix must be empty");
    const auto digest_expected = required_text(record, "manifest_sha256", ErrorCode::wrong_identity);
    gzFile file = gzopen(archive.c_str(), "rb");
    require(file != nullptr, ErrorCode::io, "cannot open engine gzip archive");
    Json manifest;
    std::set<std::string> seen;
    uint64_t aggregate = 0;
    size_t entries = 0;
    bool saw_manifest = false;
    std::map<std::string, std::string> pax;
    try {
        for (;;) {
            std::array<char, kBlock> header{};
            gz_read_exact(file, header.data(), header.size());
            if (zero_header(header)) {
                require(pax.empty(), ErrorCode::unsafe_archive, "unterminated PAX header");
                // The Python tar packer pads to a block-aligned record.
                for (;;) {
                    std::array<char, kBlock> tail{};
                    const int got = gzread(file, tail.data(), tail.size());
                    if (got == 0)
                        break;
                    require(got > 0 &&
                                std::all_of(tail.begin(), tail.begin() + got, [](char c) { return c == '\0'; }),
                            ErrorCode::unsafe_archive, "trailing nonzero tar content");
                }
                break;
            }
            uint64_t checksum = 0;
            for (size_t i = 0; i < header.size(); ++i)
                checksum += i >= 148 && i < 156 ? 32U : static_cast<unsigned char>(header[i]);
            require(tar_number(header, 148, 8) == checksum, ErrorCode::unsafe_archive,
                    "invalid tar header checksum");
            const auto size = tar_number(header, 124, 12);
            const auto mode = tar_number(header, 100, 8);
            const auto type = header[156];
            auto name = tar_text(header, 0, 100);
            const auto parent = tar_text(header, 345, 155);
            if (!parent.empty()) {
                name.insert(0, 1, '/');
                name.insert(0, parent);
            }
            require(size <= kMaxBytes && aggregate <= kMaxBytes - size, ErrorCode::unsafe_archive,
                    "archive exceeds expansion bound");
            aggregate += size;
            if (type == 'x') {
                require(pax.empty() && size <= kMaxPax, ErrorCode::unsafe_archive,
                        "invalid PAX header sequence");
                std::string content(size, '\0');
                gz_read_exact(file, content.data(), size);
                pax = pax_fields(content);
                tar_padding(file, size);
                continue;
            }
            require(type == '0' || type == '\0', ErrorCode::unsafe_archive,
                    "nonregular file in release archive");
            if (pax.contains("path"))
                name = pax.at("path");
            require(!pax.contains("size"), ErrorCode::unsafe_archive,
                    "PAX size override is not a supported engine asset");
            pax.clear();
            require(++entries <= kMaxFiles + 1, ErrorCode::unsafe_archive, "too many archive members");
            require(seen.insert(name).second, ErrorCode::unsafe_archive, "duplicate archive member");
            if (!saw_manifest) {
                require(name == "bundle.json" && size <= kMaxManifest && mode == 0644,
                        ErrorCode::unsafe_archive, "bundle.json must be the first archive member");
                std::string bytes(size, '\0');
                gz_read_exact(file, bytes.data(), size);
                tar_padding(file, size);
                require(serving::sha256_hex(bytes) == digest_expected, ErrorCode::digest_mismatch,
                        "bundle manifest SHA-256 differs from tag-bound release record");
                try {
                    manifest = Json::parse(bytes);
                } catch (const Json::exception&) {
                    refuse(ErrorCode::unsafe_archive, "invalid bundle manifest JSON");
                }
                validate_manifest(manifest, record, tag);
                std::ofstream output(prefix_path / "bundle.json", std::ios::binary);
                output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
                require(output.good(), ErrorCode::io, "cannot stage bundle.json");
                saw_manifest = true;
                continue;
            }
            require(safe_member(name) && !forbidden_name(name) && manifest["files"].contains(name),
                    ErrorCode::unsafe_archive, "unexpected release archive member: " + name);
            const auto& info = manifest["files"].at(name);
            require(size == required_size(info, "size", ErrorCode::invalid_install) &&
                        mode == required_size(info, "mode", ErrorCode::invalid_install),
                    ErrorCode::digest_mismatch, "archive entry size/mode mismatch: " + name);
            const auto dest = prefix_path / fs::path(name);
            fs::create_directories(dest.parent_path());
            read_tar_content(file, dest, size, required_text(info, "sha256", ErrorCode::invalid_install), name);
            tar_padding(file, size);
            fs::permissions(dest, static_cast<fs::perms>(mode), fs::perm_options::replace);
            check_interpreter(dest, name);
        }
    } catch (...) {
        gzclose(file);
        throw;
    }
    const int closed = gzclose(file);
    require(closed == Z_OK, ErrorCode::unsafe_archive, "gzip stream checksum mismatch");
    require(saw_manifest, ErrorCode::unsafe_archive, "release archive lacks bundle.json");
    const auto expected = file_map(manifest, record);
    seen.erase("bundle.json");
    require(seen == expected, ErrorCode::invalid_install, "archive membership differs from runtime manifest");
    verify_file_set(prefix_path, manifest, record, tag, false);
    return {record, manifest};
}

namespace {
using Stamps = std::vector<std::pair<std::string, serving::FileStamp>>;

// The stamp of every member of the prefix, by relative path; empty when the prefix has a
// member that is not a plain file (the full check judges that).
std::optional<Stamps> capture_stamps(const fs::path& prefix_path) {
    Stamps stamps;
    for (auto iter = fs::recursive_directory_iterator(prefix_path); iter != fs::recursive_directory_iterator();
         ++iter) {
        if (iter->is_symlink())
            return std::nullopt;
        if (iter->is_directory())
            continue;
        if (!iter->is_regular_file())
            return std::nullopt;
        stamps.emplace_back(fs::relative(iter->path(), prefix_path).generic_string(), serving::stamp_file(iter->path()));
    }
    std::sort(stamps.begin(), stamps.end(), [](const auto& l, const auto& r) { return l.first < r.first; });
    return stamps;
}

Json stamps_json(const Stamps& stamps) {
    Json files = Json::array();
    for (const auto& [name, s] : stamps)
        files.push_back({name, s.size, s.inode, s.mtime_ns, s.ctime_ns});
    return {{"files", files}};
}

// Was every member of this prefix, as it is now, verified in full and recorded? False on any difference, on a record
// that cannot be read or parsed: the full check then runs.
bool stamps_trusted(const fs::path& prefix_path, const fs::path& file) {
    try {
        if (!is_regular_plain(file))
            return false;
        const auto recorded = Json::parse(read_bounded(file, kMaxManifest));
        const auto now = capture_stamps(prefix_path);
        if (!now || recorded != stamps_json(*now))
            return false;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}
} // namespace

std::uint64_t hashed_files() {
    return hashed_file_count.load();
}

void verify_installed_files(const fs::path& prefix_path, const Json& record, const Json& manifest,
                            const std::string& tag, const fs::path& stamps_file, bool trust_stamps) {
    const auto bytes = read_bounded(prefix_path / "bundle.json", kMaxManifest);
    require(serving::sha256_hex(bytes) == required_text(record, "manifest_sha256", ErrorCode::wrong_identity),
            ErrorCode::digest_mismatch, "installed bundle manifest SHA-256 mismatch");
    require(Json::parse(bytes) == manifest, ErrorCode::invalid_install, "installed manifest content mismatch");
    if (trust_stamps && stamps_trusted(prefix_path, stamps_file)) {
        validate_manifest(manifest, record, tag);
        return;
    }
    // Stamped before any byte is read: a member that changes while it is hashed no longer matches its stamp.
    const auto stamps = capture_stamps(prefix_path);
    verify_file_set(prefix_path, manifest, record, tag, true);
    if (stamps) {
        // Stamps that cannot be written (read-only media) costs the next start a full check; it never fails this one.
        try {
            fs::create_directories(stamps_file.parent_path());
            write_atomic(stamps_file, stamps_json(*stamps).dump() + "\n");
        } catch (const std::exception&) {
        }
    }
}

} // namespace bc::release
