#include "session.h"
#include "power_transaction.h"

#include "config_loader.h"
#include "feature_state.h"
#include "helper_vram.h"
#include "live_policy_ingest.h"
#include "calibration_watch.h"
#include "manifest_writer.h"
#include "plan_cache.h"
#include "rank_demand.h"
#include "resources.h"
#include "s8_replay.h"
#include "sha256.h"
#include "stored_manifest.h"
#include "stored_manifest_cache.h"
#include "v2/document_bytes.h"
#include "v2/s8_rules.h"
#include "serving_codec.h"
#include "v2/strict_json.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <linux/memfd.h>
#include <limits>
#include <rad_abi.h>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace bc::serving::session {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

class NativeExecutionRoot final {
  public:
    explicit NativeExecutionRoot(const fs::path& root) {
        std::error_code error;
        path_ = fs::canonical(root, error);
        if (error || !fs::is_directory(path_, error) || error)
            throw PlanFailed("native config admission: resolution root is not an existing directory");
        fd_ = ::open(path_.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (fd_ < 0)
            throw PlanFailed(std::string("native config admission: cannot retain the resolution root: ") +
                             std::strerror(errno));
        if (fd_ < 3) {
            ::close(fd_);
            fd_ = -1;
            throw PlanFailed("native config admission: root descriptor must be greater than 2");
        }
        try {
            require_matches(path_);
        } catch (...) {
            ::close(fd_);
            fd_ = -1;
            throw;
        }
    }
    ~NativeExecutionRoot() { if (fd_ >= 0) ::close(fd_); }
    NativeExecutionRoot(const NativeExecutionRoot&) = delete;
    NativeExecutionRoot& operator=(const NativeExecutionRoot&) = delete;
    int descriptor() const noexcept { return fd_; }

    static void require_descriptor_matches(int fd, const fs::path& expected_root) {
        std::error_code error;
        const fs::path canonical = fs::canonical(expected_root, error);
        struct stat held{}, named{};
        if (fd < 3 || error || ::fstat(fd, &held) != 0 || ::stat(canonical.c_str(), &named) != 0 ||
            !S_ISDIR(held.st_mode) || held.st_dev != named.st_dev || held.st_ino != named.st_ino)
            throw PlanFailed("native config admission: retained resolution root no longer matches its recorded path");
    }

    void require_matches(const fs::path& expected_root) const {
        require_descriptor_matches(fd_, expected_root);
        std::error_code error;
        if (fs::canonical(expected_root, error) != path_ || error)
            throw PlanFailed("native config admission: retained resolution root no longer matches its recorded path");
    }

  private:
    int fd_ = -1;
    fs::path path_;
};

struct NativeEngineLaunch {
    NativeEngineLaunch(const fs::path& resolution_root, Env complete_environment, fs::path system_root)
        : root(resolution_root), environment(std::move(complete_environment)), sys_root(std::move(system_root)) {}

    NativeExecutionRoot root;
    Env environment;
    fs::path sys_root;
};

// An admitted v2 document or native config snapshot is held in a kernel-sealed
// anonymous file. The path is only a view of these exact bytes; it does not
// reopen the operator's replaceable pathname. The owner retains its descriptor
// until the last child has verified its exit.
class PinnedManifest final {
  public:
    explicit PinnedManifest(std::string_view admitted) {
        fd_ = static_cast<int>(::syscall(SYS_memfd_create, "bc-stored-v2", MFD_CLOEXEC | MFD_ALLOW_SEALING));
        if (fd_ < 0)
            throw PlanFailed("replay: cannot allocate an authenticated manifest descriptor");
        try {
            size_t offset = 0;
            while (offset < admitted.size()) {
                const ssize_t n = ::write(fd_, admitted.data() + offset, admitted.size() - offset);
                if (n < 0 && errno == EINTR)
                    continue;
                if (n <= 0)
                    throw PlanFailed("replay: cannot copy admitted bytes into the sealed handoff");
                offset += static_cast<size_t>(n);
            }
            if (::lseek(fd_, 0, SEEK_SET) != 0)
                throw PlanFailed("replay: cannot rewind admitted manifest");
            constexpr unsigned int seals =
                static_cast<unsigned int>(F_SEAL_WRITE) | static_cast<unsigned int>(F_SEAL_SHRINK) |
                static_cast<unsigned int>(F_SEAL_GROW) | static_cast<unsigned int>(F_SEAL_SEAL);
            if (::fcntl(fd_, F_ADD_SEALS, static_cast<int>(seals)) != 0)
                throw PlanFailed("replay: cannot seal admitted manifest bytes");
            const int applied = ::fcntl(fd_, F_GET_SEALS);
            if (applied < 0 || (static_cast<unsigned int>(applied) & seals) != seals)
                throw PlanFailed("replay: cannot verify admitted manifest seals");
        } catch (...) {
            ::close(fd_);
            fd_ = -1;
            throw;
        }
    }
    ~PinnedManifest() {
        if (fd_ >= 0)
            ::close(fd_);
    }
    PinnedManifest(const PinnedManifest&) = delete;
    PinnedManifest& operator=(const PinnedManifest&) = delete;
    int descriptor() const noexcept {
        return fd_;
    }
    fs::path path() const {
        return "/proc/self/fd/" + std::to_string(fd_);
    }

  private:
    int fd_ = -1;
};

std::filesystem::path EngineStart::execution_manifest() const {
    return pinned_manifest ? pinned_manifest->path() : manifest;
}

namespace {

std::string errno_text();

struct PinnedManifestSnapshot {
    std::shared_ptr<PinnedManifest> pinned;
    v2::Manifest parsed;
    std::string sha256;
    std::string file_identity;
};

std::string read_manifest_file(const fs::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0)
        throw PlanFailed("manifest: cannot open " + path.string() + ": " + errno_text());
    struct Close {
        int fd;
        ~Close() { ::close(fd); }
    } close{fd};

    struct stat st{};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0 ||
        static_cast<uint64_t>(st.st_size) > v2::kMaxDocumentBytes)
        throw PlanFailed("manifest: input is not a bounded regular file: " + path.string());

    std::string bytes;
    bytes.reserve(static_cast<size_t>(st.st_size));
    std::array<char, 64 * 1024> chunk{};
    for (;;) {
        const ssize_t n = ::read(fd, chunk.data(), chunk.size());
        if (n < 0 && errno == EINTR)
            continue;
        if (n < 0)
            throw PlanFailed("manifest: cannot read " + path.string() + ": " + errno_text());
        if (n == 0)
            break;
        if (bytes.size() + static_cast<size_t>(n) > v2::kMaxDocumentBytes)
            throw PlanFailed("manifest: input exceeds the strict document size limit: " + path.string());
        bytes.append(chunk.data(), static_cast<size_t>(n));
    }
    return bytes;
}

PinnedManifestSnapshot pin_manifest_file(const fs::path& path, v2::PlanUse use) {
    const std::string bytes = read_manifest_file(path);
    auto parsed = v2::read_manifest(bytes, path.string(), use);
    if (!parsed)
        throw PlanFailed(v2::render_human("bc-serve manifest admission", parsed.error()));
    auto pinned = std::make_shared<PinnedManifest>(bytes);
    const std::string file_identity = complete_file_identity_bytes(bytes);
    return {std::move(pinned), std::move(parsed.value()), sha256_hex(bytes), file_identity};
}

} // namespace

namespace {

volatile std::sig_atomic_t g_signal = 0;
void on_signal(int sig) {
    g_signal = sig;
}

std::string errno_text() {
    return std::strerror(errno);
}

std::string utc_now() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::string stamp() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%dT%H%M%SZ", &tm);
    return std::string(buf) + "-" + std::to_string(::getpid());
}

void write_atomic(const fs::path& path, const std::string& text) {
    const fs::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out)
            throw std::runtime_error("cannot write " + tmp.string());
        out << text;
        out.flush();
        if (!out)
            throw std::runtime_error("cannot write " + tmp.string());
    }
    fs::rename(tmp, path);
}

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

// MRC02's native owner writes only passive start-of-session host evidence.
// Python is not loaded by bc-serve. The platform ingester validates this
// timestamped receipt instead of sampling the host after calibration finishes.
Json start_fact(std::optional<Json> value, const std::string& reason = "unreadable") {
    if (!value)
        return {{"value", nullptr}, {"readable", false}, {"reason", reason}};
    return {{"value", *value}, {"readable", true}};
}

std::optional<std::string> host_line(const fs::path& path) {
    std::ifstream input(path);
    if (!input)
        return std::nullopt;
    std::string value;
    std::getline(input, value);
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return std::nullopt;
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

Json host_text(const fs::path& path, const std::string& missing = "unreadable") {
    const auto value = host_line(path);
    return start_fact(value ? std::optional<Json>(*value) : std::nullopt, missing);
}

Json host_positive_number(const fs::path& path, const std::string& zero = "invalid_zero") {
    const auto raw = host_line(path);
    if (!raw)
        return start_fact(std::nullopt);
    if (!std::regex_match(*raw, std::regex("[0-9]+")))
        return start_fact(std::nullopt, "malformed");
    try {
        const auto n = std::stoull(*raw);
        return n > 0 ? start_fact(Json(n)) : start_fact(std::nullopt, zero);
    } catch (const std::exception&) {
        return start_fact(std::nullopt, "malformed");
    }
}

Json host_available_kib(const fs::path& path) {
    std::ifstream input(path);
    if (!input)
        return start_fact(std::nullopt);
    std::string line;
    static const std::regex pattern(R"(^MemAvailable:\s+([0-9]+)\s+kB$)");
    while (std::getline(input, line)) {
        std::smatch match;
        if (!std::regex_match(line, match, pattern))
            continue;
        try {
            return start_fact(Json(std::stoull(match[1].str())));
        } catch (const std::exception&) {
            return start_fact(std::nullopt, "malformed_mem_available");
        }
    }
    return start_fact(std::nullopt, "mem_available_unobserved");
}

Json host_clock(const fs::path& path) {
    std::ifstream input(path);
    if (!input)
        return start_fact(std::nullopt);
    std::string line;
    const std::regex mhz("([0-9]+(?:\\.[0-9]+)?)\\s*(M|G)Hz", std::regex_constants::icase);
    while (std::getline(input, line)) {
        if (line.find('*') == std::string::npos)
            continue;
        std::smatch found;
        if (std::regex_search(line, found, mhz)) {
            try {
                const double magnitude = std::stod(found[1].str());
                const double scale = found[2].str() == "G" || found[2].str() == "g" ? 1000.0 : 1.0;
                return start_fact(Json(static_cast<unsigned long long>(magnitude * scale)));
            } catch (const std::exception&) {
                return start_fact(std::nullopt, "malformed_clock");
            }
        }
    }
    return start_fact(std::nullopt, "no_active_clock");
}

Json host_card(const fs::path& device, const std::string& bdf) {
    Json card = {
        {"bdf", start_fact(Json(bdf))},
        {"vram_total_bytes", host_positive_number(device / "mem_info_vram_total")},
        {"firmware_version", host_text(device / "firmware_version")},
        {"sclk_current_mhz", host_clock(device / "pp_dpm_sclk")},
        {"mclk_current_mhz", host_clock(device / "pp_dpm_mclk")},
        {"sclk_od_max_mhz", start_fact(std::nullopt, "native_od_max_unobserved")},
        {"mclk_od_max_mhz", start_fact(std::nullopt, "native_od_max_unobserved")},
    };
    const auto arch = host_line(device / "gfx_target_version");
    std::smatch match;
    if (arch && std::regex_match(*arch, match, std::regex("([0-9]+)\\.([0-9])\\.([0-9])")))
        card["arch"] = start_fact(Json("gfx" + match[1].str() + match[2].str() + match[3].str()));
    else if (arch && std::regex_match(*arch, std::regex("gfx[0-9a-z]+")))
        card["arch"] = start_fact(Json(*arch));
    else
        card["arch"] = start_fact(std::nullopt, arch ? "malformed" : "unreadable");
    fs::path hwmon;
    std::error_code ec;
    if (fs::is_directory(device / "hwmon", ec)) {
        for (const auto& item : fs::directory_iterator(device / "hwmon", ec)) {
            if (item.is_directory(ec) && (hwmon.empty() || item.path() < hwmon))
                hwmon = item.path();
        }
    }
    auto power_reading = [&](const char* name, const std::string& zero = "invalid_zero") {
        if (hwmon.empty())
            return start_fact(std::nullopt, "no_hwmon");
        return host_positive_number(hwmon / name, zero);
    };
    card["power"] = {
        {"cap_uw", power_reading("power1_cap", "zero_cap_suspect")},
        {"cap_default_uw", power_reading("power1_cap_default", "zero_cap_suspect")},
        {"cap_min_uw", power_reading("power1_cap_min")},
        {"cap_max_uw", power_reading("power1_cap_max")},
        {"average_uw", power_reading("power1_average")},
        {"perf_level", host_text(device / "power_dpm_force_performance_level")},
    };
    card["pcie"] = {
        {"card", {
            {"current_speed", host_text(device / "current_link_speed")},
            {"current_width", host_positive_number(device / "current_link_width")},
            {"max_speed", host_text(device / "max_link_speed")},
            {"max_width", host_positive_number(device / "max_link_width")},
        }},
        {"upstream", Json::array()},
        {"upstream_readability", start_fact(std::nullopt, "native_upstream_unobserved")},
    };
    return card;
}

Json native_start_host_facts(const fs::path& sys_root = "/sys", const fs::path& proc_root = "/proc") {
    Json facts = {
        {"kernel_release", host_text(proc_root / "sys/kernel/osrelease")},
        {"kernel_build", host_text(proc_root / "version")},
        {"amdgpu_version", host_text(sys_root / "module/amdgpu/version")},
        {"amdgpu_srcversion", host_text(sys_root / "module/amdgpu/srcversion")},
        {"mem_available_kib", host_available_kib(proc_root / "meminfo")},
        {"memory", {
            {"populated_dimms", start_fact(std::nullopt, "native_dmidecode_unobserved")},
            {"configured_speed_mt_s", start_fact(std::nullopt, "native_dmidecode_unobserved")},
            {"profile", start_fact(std::nullopt, "native_dmidecode_unobserved")},
        }},
        {"gpu_lease_holder", start_fact(std::nullopt, "native_lease_api_unavailable")},
    };
    const char* rocm = std::getenv("ROCM_PATH");
    facts["rocm_version"] = rocm && *rocm ? host_text(fs::path(rocm) / ".info/version")
                                          : start_fact(std::nullopt, "sdk_not_selected");
    const fs::path env_file = "/etc/bigcherry/environment.toml";
    const auto environment = host_line(env_file);
    facts["environment_sha256"] = environment ? start_fact(Json(sha256_hex(read_file(env_file))))
                                             : start_fact(std::nullopt, "environment_unreadable");
    Json cards = Json::object();
    std::error_code ec;
    const fs::path drm = sys_root / "class/drm";
    if (fs::is_directory(drm, ec)) {
        for (const auto& entry : fs::directory_iterator(drm, ec)) {
            const auto name = entry.path().filename().string();
            if (!std::regex_match(name, std::regex("card[0-9]+")))
                continue;
            const auto device = fs::weakly_canonical(entry.path() / "device", ec);
            if (ec)
                continue;
            const auto bdf = device.filename().string();
            if (!std::regex_match(bdf, std::regex("[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\\.[0-7]")))
                continue;
            cards[bdf] = host_card(device, bdf);
        }
    }
    facts["cards"] = cards;
    facts["cards_readability"] = cards.empty() ? start_fact(std::nullopt, "no_cards_detected")
                                                : start_fact(Json(cards.size()));
    return {{"schema_version", 1}, {"capture_phase", "owner_start"},
            {"captured_at", utc_now()}, {"facts", facts}};
}

bool starts_with(const std::string& s, const char* prefix) {
    return s.rfind(prefix, 0) == 0;
}

// Variables that carry bigcherry configuration or choose devices: a child never inherits them.
bool stripped(const std::string& name) {
    for (const char* p : {"R3_", "RAD_", "BC_", "BIGCHERRY_"})
        if (starts_with(name, p))
            return true;
    return name == "ROCR_VISIBLE_DEVICES" || name == "HIP_VISIBLE_DEVICES" || name == "CUDA_VISIBLE_DEVICES" ||
           name == "GPU_DEVICE_ORDINAL" || name == "RADIANCE_HOME";
}

// The control socket's address. sun_path holds at most 107 bytes; a longer run directory is reached through a
// descriptor open on it (/proc/self/fd/N/session.sock), which bind and connect resolve like the real path.
struct SocketAddress {
    sockaddr_un addr{};
    std::string path; // the real path, for messages and chmod
    int dir = -1;
    explicit SocketAddress(const fs::path& run_dir) : path((run_dir / "session.sock").string()) {
        addr.sun_family = AF_UNIX;
        std::string use = path;
        if (use.size() >= sizeof addr.sun_path) {
            dir = ::open(run_dir.c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC);
            if (dir < 0)
                throw std::runtime_error("control socket " + path + ": " + errno_text());
            use = "/proc/self/fd/" + std::to_string(dir) + "/session.sock";
        }
        std::memcpy(addr.sun_path, use.c_str(), use.size() + 1);
    }
    ~SocketAddress() {
        if (dir >= 0)
            ::close(dir);
    }
    SocketAddress(const SocketAddress&) = delete;
    SocketAddress& operator=(const SocketAddress&) = delete;
    const sockaddr* get() const {
        return reinterpret_cast<const sockaddr*>(&addr);
    }
};

// Read until EOF, a newline (when `line`), or the deadline.
std::string read_until(int fd, Clock::time_point deadline, bool line) {
    std::string data;
    char buf[4096];
    while (true) {
        const auto left =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
        if (left <= 0)
            break;
        pollfd p{fd, POLLIN, 0};
        const int r = ::poll(&p, 1, int(std::min<long long>(left, 1000)));
        if (r < 0 && errno == EINTR)
            continue;
        if (r <= 0)
            continue;
        const ssize_t n = ::read(fd, buf, sizeof buf);
        if (n < 0 && (errno == EINTR || errno == EAGAIN))
            continue;
        if (n <= 0)
            break;
        data.append(buf, size_t(n));
        if (line && data.find('\n') != std::string::npos)
            break;
    }
    return data;
}

bool write_all(int fd, const std::string& data) {
    size_t off = 0;
    while (off < data.size()) {
        const ssize_t n = ::write(fd, data.data() + off, data.size() - off);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            return false;
        off += size_t(n);
    }
    return true;
}

std::string dechunk(const std::string& body) {
    std::string out;
    size_t at = 0;
    while (at < body.size()) {
        const size_t eol = body.find("\r\n", at);
        if (eol == std::string::npos)
            break;
        const size_t len = std::stoul(body.substr(at, eol - at), nullptr, 16);
        if (len == 0)
            break;
        out += body.substr(eol + 2, len);
        at = eol + 2 + len + 2;
    }
    return out;
}

// In a forked child before exec: every descriptor above stderr closes at exec (an inherited socket, pipe or
// lock of the owner never reaches an engine). Marks rather than closes, so the exec-failure report pipe still
// works until exec.
void cloexec_above_stderr() {
#ifndef CLOSE_RANGE_CLOEXEC
#define CLOSE_RANGE_CLOEXEC (1U << 2)
#endif
    if (::syscall(SYS_close_range, 3U, ~0U, CLOSE_RANGE_CLOEXEC) == 0)
        return;
    struct rlimit lim{};
    const long top =
        ::getrlimit(RLIMIT_NOFILE, &lim) == 0 && lim.rlim_cur != RLIM_INFINITY ? long(lim.rlim_cur) : 65536;
    for (int fd = 3; fd < top; ++fd) {
        const int flags = ::fcntl(fd, F_GETFD);
        if (flags >= 0) {
            const auto enabled = static_cast<unsigned int>(flags) | static_cast<unsigned int>(FD_CLOEXEC);
            ::fcntl(fd, F_SETFD, static_cast<int>(enabled));
        }
    }
}

// /proc/PID/stat: state (field 3), process group (5), start time (22).
struct ProcStat {
    char state = 0;
    pid_t pgid = 0;
    uint64_t start = 0;
};
std::optional<ProcStat> proc_stat(pid_t pid) {
    std::ifstream in("/proc/" + std::to_string(pid) + "/stat");
    std::string stat;
    if (!std::getline(in, stat))
        return std::nullopt;
    const size_t close = stat.rfind(')');
    if (close == std::string::npos)
        return std::nullopt;
    std::istringstream rest(stat.substr(close + 2));
    std::string field;
    ProcStat p;
    int reached = 2; // a stat cut short is no stat: every field up to the start time must be there
    try {
        for (int i = 3; i <= 22 && rest >> field; ++i) {
            if (i == 3)
                p.state = field[0];
            else if (i == 5)
                p.pgid = pid_t(std::stol(field));
            else if (i == 22)
                p.start = std::stoull(field);
            reached = i;
        }
    } catch (...) {
        return std::nullopt;
    }
    if (reached != 22)
        return std::nullopt;
    return p;
}

// A process of the group that still runs (a zombie holds nothing and only waits for its parent to reap it).
// Fails closed: a /proc that cannot be listed, or a process whose stat exists but cannot be read, counts as a
// live member, so an incomplete scan never verifies an exit.
bool group_has_live_member(pid_t pgid) {
    std::error_code ec;
    fs::directory_iterator it("/proc", ec), end;
    if (ec)
        return true;
    for (; it != end; it.increment(ec)) {
        if (ec)
            return true;
        const std::string name = it->path().filename().string();
        if (name.empty() || name.find_first_not_of("0123456789") != std::string::npos)
            continue;
        const pid_t pid = pid_t(std::stol(name));
        const auto p = proc_stat(pid);
        if (!p) {
            // gone since the listing (no such process) is the only reason not to count it
            if (::kill(pid, 0) != 0 && errno == ESRCH)
                continue;
            return true;
        }
        if (p->pgid == pgid && p->state != 'Z')
            return true;
    }
    return !!ec;
}

// The start gate: a child forked behind it waits, before exec, until the parent has recorded it (release) and
// exits without running anything when the parent closes the gate or dies first, so no process ever runs
// unrecorded.
struct Gate {
    int fd[2] = {-1, -1};
    Gate() {
        if (::pipe2(fd, O_CLOEXEC) != 0)
            throw std::runtime_error("pipe: " + errno_text());
    }
    ~Gate() {
        close_write();
        if (fd[0] >= 0)
            ::close(fd[0]);
    }
    Gate(const Gate&) = delete;
    Gate& operator=(const Gate&) = delete;
    // in the child: true once released
    bool wait_in_child() {
        ::close(fd[1]);
        char go = 0;
        ssize_t n;
        do
            n = ::read(fd[0], &go, 1);
        while (n < 0 && errno == EINTR);
        return n == 1;
    }
    // in the parent, after fork
    void parent_side() {
        ::close(fd[0]);
        fd[0] = -1;
    }
    void release() {
        const char go = 1;
        ssize_t n;
        do
            n = ::write(fd[1], &go, 1);
        while (n < 0 && errno == EINTR);
        close_write();
    }
    void close_write() {
        if (fd[1] >= 0)
            ::close(fd[1]);
        fd[1] = -1;
    }
};

// Every injected callback returns a typed result. The plan step throws PlanFailed
// only after the guarded invocation (never at a raw call).
void require_callback(v2::Status status, std::string_view origin) {
    if (status)
        throw PlanFailed(std::string(origin) + ": " + status->message);
}

template <class T> T require_callback(v2::Result<T> result, std::string_view origin) {
    if (!result)
        throw PlanFailed(std::string(origin) + ": callback failed: " + result.error().message);
    return std::move(result.value());
}

void require_callback(v2::Result<v2::Status> result, std::string_view origin) {
    if (!result)
        throw PlanFailed(std::string(origin) + ": callback failed: " + result.error().message);
    require_callback(std::move(result.value()), origin);
}

// Record a just-forked child through `recorded`, then let it run; when recording fails, close the gate (the
// child exits without exec), reap it and rethrow.
void record_then_release(Gate& gate, pid_t pid, const Callback<void(pid_t)>& recorded) {
    gate.parent_side();
    try {
        if (recorded)
            recorded.require("session.spawn.recorded", pid);
    } catch (...) {
        gate.close_write();
        int st = 0;
        while (::waitpid(pid, &st, 0) < 0 && errno == EINTR) {
        }
        throw;
    }
    gate.release();
}

bool group_gone(pid_t pgid) {
    if (::kill(-pgid, 0) != 0 && errno == ESRCH)
        return true;
    return !group_has_live_member(pgid);
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return s;
}

// Only the session owner can mint a fresh launch identity. It is supplied to
// the engine in the explicitly constructed child environment, not inherited.
std::string heat_launch_nonce() {
    std::array<unsigned char, 16> random{};
    size_t filled = 0;
    while (filled < random.size()) {
        const ssize_t n = ::getrandom(random.data() + filled, random.size() - filled, 0);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            throw std::runtime_error("could not create the engine heat launch nonce");
        filled += static_cast<size_t>(n);
    }
    constexpr char digits[] = "0123456789abcdef";
    std::string token;
    token.reserve(random.size() * 2);
    for (const unsigned char byte : random) {
        const auto code = static_cast<unsigned int>(byte);
        token.push_back(digits[code >> 4U]);
        token.push_back(digits[code & 15U]);
    }
    return token;
}

} // namespace

/* --- children
 * -------------------------------------------------------------------------------------------------- */

Env child_environment(char** inherited, const Env& runtime) {
    Env env;
    for (char** e = inherited; e && *e; ++e) {
        const std::string kv(*e);
        const size_t eq = kv.find('=');
        if (eq == std::string::npos || eq == 0)
            continue;
        const std::string name = kv.substr(0, eq);
        if (!stripped(name))
            env[name] = kv.substr(eq + 1);
    }
    for (const auto& [k, v] : runtime)
        env[k] = v;
    return env;
}

Env current_child_environment(const Env& runtime) { return child_environment(environ, runtime); }

pid_t spawn(const std::vector<std::string>& argv, const Env& env, const fs::path& log,
            const Callback<void(pid_t)>& recorded, int inherited_manifest_fd, int working_directory_fd) {
    if (argv.empty() || argv[0].empty())
        throw std::runtime_error("spawn: an empty command");
    if ((inherited_manifest_fd >= 0 && inherited_manifest_fd < 3) ||
        (working_directory_fd >= 0 && working_directory_fd < 3))
        throw std::runtime_error("spawn: inherited descriptors must be greater than 2");
    std::vector<std::string> envs;
    for (const auto& [k, v] : env) {
        std::string setting;
        setting.reserve(k.size() + 1U + v.size());
        setting.append(k).append("=").append(v);
        envs.push_back(std::move(setting));
    }
    std::vector<char*> cargv, cenv;
    for (const auto& a : argv)
        cargv.push_back(const_cast<char*>(a.c_str()));
    cargv.push_back(nullptr);
    for (const auto& e : envs)
        cenv.push_back(const_cast<char*>(e.c_str()));
    cenv.push_back(nullptr);
    const int out = ::open(log.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (out < 0)
        throw std::runtime_error("cannot open " + log.string() + ": " + errno_text());
    int report[2];
    if (::pipe2(report, O_CLOEXEC) != 0) {
        ::close(out);
        throw std::runtime_error("pipe: " + errno_text());
    }
    std::optional<Gate> gate;
    try {
        gate.emplace();
    } catch (...) {
        ::close(out);
        ::close(report[0]);
        ::close(report[1]);
        throw;
    }
    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(out);
        ::close(report[0]);
        ::close(report[1]);
        throw std::runtime_error("fork: " + errno_text());
    }
    if (pid == 0) {
        // the child: its own session and process group, default signal dispositions, nothing inherited but the
        // log
        ::setsid();
        for (int s : {SIGINT, SIGTERM, SIGHUP, SIGPIPE, SIGCHLD})
            ::signal(s, SIG_DFL);
        sigset_t none;
        sigemptyset(&none);
        ::sigprocmask(SIG_SETMASK, &none, nullptr);
        const int in = ::open("/dev/null", O_RDONLY);
        if (in < 0 || ::dup2(in, 0) < 0 || ::dup2(out, 1) < 0 || ::dup2(out, 2) < 0) {
            const int e = errno;
            (void)!::write(report[1], &e, sizeof e);
            ::_exit(127);
        }
        cloexec_above_stderr();
        if (inherited_manifest_fd >= 0) {
            // The owner keeps every other descriptor close-on-exec. Only the
            // sealed, authenticated manifest becomes readable by the strict
            // engine/worker readers at /proc/self/fd/N after execve.
            const int flags = ::fcntl(inherited_manifest_fd, F_GETFD);
            const auto inherited_flags =
                static_cast<int>(static_cast<unsigned int>(flags) & ~static_cast<unsigned int>(FD_CLOEXEC));
            if (flags < 0 || ::fcntl(inherited_manifest_fd, F_SETFD, inherited_flags) != 0) {
                const int e = errno;
                (void)!::write(report[1], &e, sizeof e);
                ::_exit(127);
            }
        }
        if (!gate->wait_in_child())
            ::_exit(126); // never recorded: run nothing
        if (working_directory_fd >= 0 && ::fchdir(working_directory_fd) != 0) {
            const int e = errno;
            (void)!::write(report[1], &e, sizeof e);
            ::_exit(127);
        }
        ::execve(cargv[0], cargv.data(), cenv.data());
        const int e = errno;
        (void)!::write(report[1], &e, sizeof e);
        ::_exit(127);
    }
    ::close(out);
    ::close(report[1]);
    try {
        record_then_release(*gate, pid, recorded);
    } catch (...) {
        ::close(report[0]);
        throw;
    }
    int child_errno = 0;
    ssize_t n;
    do
        n = ::read(report[0], &child_errno, sizeof child_errno);
    while (n < 0 && errno == EINTR);
    ::close(report[0]);
    if (n > 0) {
        int st = 0;
        ::waitpid(pid, &st, 0);
        throw std::runtime_error("cannot start " + argv[0] + ": " + std::strerror(child_errno));
    }
    return pid;
}

bool running(pid_t pid, int* status) {
    int st = 0;
    pid_t r;
    do
        r = ::waitpid(pid, &st, WNOHANG);
    while (r < 0 && errno == EINTR);
    if (r == 0)
        return true;
    if (r == pid && status)
        *status = st;
    return false; // exited (reaped now) or not a child of ours any more (reaped before)
}

// The finite engine and its helpers are already registered with the caller's owner.
template<class CheckHelpers, class WaitOwner>
std::optional<int> wait_for_finite_engine(pid_t pid, Clock::time_point deadline,
                                          const CheckHelpers& check_helpers, const WaitOwner& wait_owner) {
    int status = 0;
    while (running(pid, &status)) {
        check_helpers();
        if (Clock::now() >= deadline || !wait_owner())
            return std::nullopt;
    }
    return status;
}

bool stop_child(pid_t pid, int first, std::chrono::milliseconds grace) {
    // gone means the child reaped AND no process left in its group (a descendant that outlived it included)
    auto gone = [&] { return !running(pid) && group_gone(pid); };
    auto settle = [&](std::chrono::milliseconds limit) {
        const auto deadline = Clock::now() + limit;
        do {
            if (gone())
                return true;
            std::this_thread::sleep_for(20ms);
        } while (Clock::now() < deadline);
        return gone();
    };
    if (gone())
        return true;
    ::kill(-pid, first);
    if (settle(grace))
        return true;
    ::kill(-pid, SIGKILL);
    return settle(grace);
}

std::optional<uint64_t> start_time(pid_t pid) {
    const auto p = proc_stat(pid);
    if (!p || p->state == 'Z')
        return std::nullopt;
    return p->start;
}

std::vector<Json> alive_recorded(const Json& record) {
    std::vector<Json> alive;
    for (const auto& p : record.value("processes", Json::array())) {
        if (!p.contains("pid") || !p.contains("start_time"))
            continue;
        const pid_t pid = p.at("pid").get<pid_t>();
        const auto t = start_time(pid);
        // the process itself, or (its leader gone) a process left in its group: every child runs in a group of
        // its own whose id is its PID, and that id is not reused while any member lives
        if ((t && *t == p.at("start_time").get<uint64_t>()) || group_has_live_member(p.value("pgid", pid)))
            alive.push_back(p);
    }
    return alive;
}

/* --- HTTP
 * ---------------------------------------------------------------------------------------------------------- */

std::optional<HttpReply> http(int port, const std::string& method, const std::string& path,
                              const std::string& body, std::chrono::milliseconds timeout) {
    const auto deadline = Clock::now() + timeout;
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (fd < 0)
        return std::nullopt;
    struct Closer {
        int fd;
        ~Closer() {
            ::close(fd);
        }
    } closer{fd};
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
        if (errno != EINPROGRESS)
            return std::nullopt;
        const auto left =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
        pollfd p{fd, POLLOUT, 0};
        if (left <= 0 || ::poll(&p, 1, int(left)) != 1)
            return std::nullopt;
        int err = 0;
        socklen_t len = sizeof err;
        if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) != 0 || err != 0)
            return std::nullopt;
    }
    std::string req = method + " " + path + " HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(port) +
                      "\r\nConnection: close\r\n";
    if (!body.empty())
        req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
    req += "\r\n" + body;
    size_t off = 0;
    while (off < req.size()) {
        const ssize_t n = ::send(fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
        if (n > 0) {
            off += size_t(n);
            continue;
        }
        if (n < 0 && (errno == EAGAIN || errno == EINTR)) {
            const auto left =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
            pollfd p{fd, POLLOUT, 0};
            if (left <= 0 || ::poll(&p, 1, int(left)) != 1)
                return std::nullopt;
            continue;
        }
        return std::nullopt;
    }
    const std::string raw = read_until(fd, deadline, false);
    const size_t head_end = raw.find("\r\n\r\n");
    if (!starts_with(raw, "HTTP/1.") || head_end == std::string::npos)
        return std::nullopt;
    HttpReply reply;
    try {
        reply.status = std::stoi(raw.substr(raw.find(' ') + 1, 3));
    } catch (...) {
        return std::nullopt;
    }
    const std::string head = lower(raw.substr(0, head_end));
    reply.body = raw.substr(head_end + 4);
    if (head.find("transfer-encoding: chunked") != std::string::npos)
        reply.body = dechunk(reply.body);
    return reply;
}

std::optional<std::string> canary(int port, const std::string& prompt, const std::string& expect,
                                  std::chrono::milliseconds timeout) {
    const Json request = {{"model", "default"}, {"prompt", prompt}, {"max_tokens", 24}, {"temperature", 0}};
    const auto reply = http(port, "POST", "/v1/completions", request.dump(), timeout);
    if (!reply)
        return "the request failed (no HTTP reply)";
    if (reply->status < 200 || reply->status > 299)
        return "the request failed (HTTP " + std::to_string(reply->status) + ": " + reply->body.substr(0, 80) +
               ")";
    const Json parsed = Json::parse(reply->body, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object() || !parsed.contains("choices") ||
        !parsed.at("choices").is_array() || parsed.at("choices").empty() ||
        !parsed.at("choices")[0].is_object() || !parsed.at("choices")[0].contains("text") ||
        !parsed.at("choices")[0].at("text").is_string())
        return "the reply is not a completion: " + reply->body.substr(0, 80);
    const std::string text = parsed.at("choices")[0].at("text");
    if (lower(text).find(lower(expect)) == std::string::npos)
        return "answered '" + text.substr(0, 80) + "' to '" + prompt + "' (expected it to contain '" + expect +
               "')";
    return std::nullopt;
}

PowerHook hooks_for_power_transaction(power::SessionPower& transaction) {
    PowerHook hook;
    hook.apply = [&transaction]() -> v2::Status { return transaction.begin(); };
    hook.restore = [&transaction]() -> v2::Status { return transaction.finish(); };
    return hook;
}

/* --- the owner
 * ------------------------------------------------------------------------------------------------------ */

namespace {

void require_segment(const std::string& name); // measuring/serving readiness check

v2::Result<v2::P1LiveRoom> measure_session_live_room(const v2::Manifest& manifest, const Json& facts,
                                                    const fs::path& sys_root) {
    auto ordered = manifest.spec.processes.engine.device.gpus;
    for (const auto& helper : manifest.spec.processes.helpers)
        ordered.push_back(helper.gpu);
    const auto room = v2::measure_live_room(facts, sys_root, ordered);
    if (!room)
        return room;
    const auto& gpus = facts.at("gpus");
    for (const auto& gpu : manifest.spec.facts.gpus) {
        if (std::find(ordered.begin(), ordered.end(), gpu.bdf) == ordered.end())
            continue;
        const auto current = std::find_if(gpus.begin(), gpus.end(),
            [&](const Json& fact) { return fact.at("bdf") == gpu.bdf; });
        if (current == gpus.end() || current->at("arch") != gpu.arch ||
            current->at("hip_ordinal") != gpu.hip_ordinal || current->at("vram_total_mib") != gpu.vram.total_mib)
            return v2::make_error(v2::ConfigErrorCode::ownership_mismatch, "spec.facts.gpus",
                                  "selected GPU identity changed: " + gpu.bdf);
    }
    return room;
}

enum class HelperReadiness : std::uint8_t { pending, ready };

HelperReadiness read_helper_readiness(const fs::path& log) {
    std::ifstream input(log);
    for (std::string line; std::getline(input, line);) {
        if (line.starts_with("r3_moe_worker: ready ") ||
            line.starts_with("r3_moe_worker: waiting for engine runtime expert maps")) {
            return HelperReadiness::ready;
        }
    }
    return HelperReadiness::pending;
}

class Owner {
  public:
    explicit Owner(const Session& s) : s_(s) {}
    ~Owner() {
        for (int fd : stop_clients_)
            ::close(fd);
        if (listen_ >= 0) {
            ::close(listen_);
            std::error_code ec;
            fs::remove(s_.run_dir / "session.sock", ec);
        }
        if (handoff_lock_ >= 0)
            ::close(handoff_lock_);
        if (lock_ >= 0)
            ::close(lock_);
    }

    int run() {
        fs::create_directories(s_.run_dir);
        lock_ = ::open((s_.run_dir / "session.lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (lock_ < 0)
            throw std::runtime_error("cannot open the session lock: " + errno_text());
        if (::flock(lock_, LOCK_EX | LOCK_NB) != 0) {
            std::string who;
            if (const auto r = read_record(s_.run_dir))
                who = " (owner pid " + r->value("owner_pid", Json()).dump() + ")";
            std::cerr << "bc-serve: a session already runs in " << s_.run_dir.string() << who
                      << "; `bc-serve stop` it first\n";
            return 98;
        }
        // The lock is ours. An earlier owner that died without stopping its processes left them in its record:
        // while any of them is still the same process, this run directory's cards are taken.
        if (const auto r = read_record(s_.run_dir)) {
            const auto alive = alive_recorded(*r);
            if (!alive.empty()) {
                std::cerr << "bc-serve: the last session in " << s_.run_dir.string() << " (owner pid "
                          << r->value("owner_pid", Json()).dump() << ", " << r->value("state", std::string("?"))
                          << ") left processes running:\n";
                for (const auto& p : alive)
                    std::cerr << "  " << p.value("role", std::string("?")) << " pid " << p.at("pid") << ", log "
                              << p.value("log", std::string()) << '\n';
                std::cerr << "  they hold the cards; stop them (by their PIDs above) before a new session\n";
                return 98;
            }
        }
        // An earlier record is history, and a socket file left by a crashed owner is stale.
        std::error_code ec;
        if (fs::exists(s_.run_dir / "session.json"))
            fs::rename(s_.run_dir / "session.json", s_.run_dir / "session.previous.json", ec);
        if (s_.publish_feature_states) {
            const fs::path prior = s_.run_dir / feature_state::kFileName;
            if (fs::exists(prior)) {
                ec.clear();
                fs::rename(prior, s_.run_dir / "feature-states.previous.json", ec);
                if (ec)
                    throw PlanFailed("cannot archive previous feature-state snapshot: " + ec.message());
            }
        }
        fs::remove(s_.run_dir / "session.sock", ec);
        record_ = {{"schema_version", 1},    {"owner_pid", ::getpid()}, {"started_at", utc_now()},
                   {"state", "created"},     {"events", Json::array()}, {"processes", Json::array()},
                   {"errors", Json::array()}};
        event("created");
        if (save_failed_) {
            std::cerr << "bc-serve: no session starts without its record ("
                      << (s_.run_dir / "session.json").string() << ")\n";
            return 1;
        }
        // Owner-start host readings are sealed before planner/power admission
        // and before any measured GPU child. Never sample at ingest time.
        try {
            const std::string start = native_start_host_facts().dump(2) + "\n";
            write_atomic(s_.run_dir / "host-facts.start.v1.json", start);
            record_["host_facts_start_sha256"] = sha256_hex(start);
            if (!save())
                return 1;
        } catch (const std::exception& e) {
            std::cerr << "bc-serve: cannot persist native host-facts start receipt: " << e.what() << '\n';
            return 1;
        }
        listen_socket();

        struct sigaction sa{};
        sa.sa_handler = on_signal;
        sigemptyset(&sa.sa_mask);
        for (int sig : {SIGINT, SIGTERM, SIGHUP})
            ::sigaction(sig, &sa, nullptr);
        ::signal(SIGPIPE, SIG_IGN);
        g_signal = 0;

        int code = 0;
        std::vector<std::string> failures;
        try {
            if (s_.quality)
                validate_quality_paths(*s_.quality);
            if (s_.power_hook.apply && s_.power_hook.begin_manifest)
                throw PlanFailed("power_hook: choose pre-plan or manifest admission, not both");
            if (bool(s_.power_hook.apply || s_.power_hook.begin_manifest) != bool(s_.power_hook.restore))
                throw PlanFailed("power_hook: begin/apply and restore must both be provided");
            if (s_.power_hook.begin_manifest && (!s_.power_hook.verify_serving || !s_.power_hook.verified_exit))
                throw PlanFailed(
                    "power_hook: manifest admission needs final verification and verified-exit proof");
            // `apply` is the pre-plan admission seam the owner tests drive; production sets `begin_manifest`, the
            // manifest-authenticated admission during the planner, before measurement.
            if (s_.power_hook.apply) {
                power_attempted_ = true; // restore even if apply partially writes before throwing
                require_callback(s_.power_hook.apply.invoke("session.power.apply"), "power_hook.apply");
                event("power_applied");
            }
            event("planning");
            Hooks hooks;
            hooks.helper_stop_timeout = s_.timing.stop_grace;
            hooks.wait = [this](std::chrono::milliseconds d) { return service(d); };
            hooks.started = [this](ChildKind kind, const std::string& role, pid_t pid, const fs::path& log) {
                // Register first, even on forbidden starts; cleanup must
                // reap a child that never passed its pre-exec gate.
                track(role, pid, log);
                if (kind != ChildKind::manifest_writer && kind != ChildKind::native_config_inspector &&
                    s_.power_hook.begin_manifest && !power_admitted_)
                    throw PlanFailed("planner started a measuring child before durable power admission");
            };
            hooks.ended = [this](pid_t pid) { untrack(pid); };
            if (s_.power_hook.begin_manifest) {
                hooks.admit_power = [this](const fs::path& manifest, v2::PlanUse use) -> v2::Status {
                    if (power_attempted_)
                        return v2::make_error(v2::ConfigErrorCode::invalid_state, "session.power",
                                              "duplicate manifest power admission");
                    power_attempted_ = true; // restore even after a failed/partial begin
                    const auto result =
                        s_.power_hook.begin_manifest.invoke("session.power.begin_manifest", manifest, use);
                    if (!result)
                        return result.error();
                    if (result.value())
                        return result.value();
                    record_["power_admission"] = {
                        {"manifest", manifest.string()},
                        {"phase", use == v2::PlanUse::measuring ? "measuring" : "serving"}};
                    if (!event("power_applied"))
                        return v2::make_error(v2::ConfigErrorCode::io_failure, "session.record",
                                              "cannot persist manifest-bound power admission before measuring");
                    power_admitted_ = true;
                    return std::nullopt;
                };
            }
            hooks.selected_segment = [this](const std::string& name) {
                if (name.rfind("/r3-serve-", 0) != 0 || name.size() > 220U)
                    throw PlanFailed("refusing an unqualified shared-memory transport name");
                if (handoff_lock_ >= 0)
                    throw PlanFailed("a session cannot select shared-memory transport twice");
                // The lock resides on the same kernel-backed shared-memory mount
                // as POSIX shm, not under a per-run-dir session root. It stays
                // locked until all children are verified dead and the owner has
                // unlinked its segment. No stored manifest bytes are changed.
                const std::string lock_path = "/dev/shm/.bigcherry-transport-" + sha256_hex(name) + ".lock";
                const int descriptor =
                    ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
                if (descriptor < 0)
                    throw PlanFailed("cannot lock shared-memory transport " + name + ": " + errno_text());
                if (::flock(descriptor, LOCK_EX | LOCK_NB) != 0) {
                    ::close(descriptor);
                    throw PlanFailed("shared-memory transport " + name + " already owned by another session");
                }
                // A crashed prior owner can leave the transport allocated even
                // after its flock is released. Do not reuse or unlink its
                // unknown contents until an operator has verified cleanup.
                const int existing = ::shm_open(name.c_str(), O_RDONLY | O_CLOEXEC, 0);
                if (existing >= 0) {
                    ::close(existing);
                    ::close(descriptor);
                    throw PlanFailed("shared-memory transport " + name +
                                     " still exists after an unverified stop");
                }
                if (errno != ENOENT) {
                    const std::string problem = errno_text();
                    ::close(descriptor);
                    throw PlanFailed("cannot inspect shared-memory transport " + name + ": " + problem);
                }
                handoff_lock_ = descriptor;
                handoff_shm_ = name;
                record_["handoff_shm"] = name; // journal name even if the planner fails after helpers start
                event("segment_selected");
            };
            plan_ = require_callback(s_.plan.invoke("session.plan", s_.run_dir, hooks), "session.plan");
            if (s_.power_hook.begin_manifest && !power_admitted_)
                throw PlanFailed("planner returned without DURABLE authenticated pre-measure power admission");
            if (s_.power_hook.verify_serving)
                require_callback(s_.power_hook.verify_serving.invoke(
                                     "session.power.verify_serving",
                                     plan_.execution_manifest()),
                                 "power final manifest");
            record_["manifest"] = plan_.manifest.string();
            if (plan_.pinned_manifest) {
                record_["power_admission"]["origin_manifest"] = plan_.manifest.string();
                record_["power_admission"]["original_sha256"] = plan_.stored_sha256;
            }
            record_["port"] = plan_.port;
            record_["selection"] = plan_.summary;
            // Why the stored plan was or was not reused, and what each start phase cost: written once here and
            // as the later phases finish, never from a hot path.
            record_["plan_reuse"] = plan_.plan_reuse;
            record_["phases_ms"] = plan_.phase_ms.is_object() ? plan_.phase_ms : Json::object();
            if (s_.publish_feature_states) {
                // The pinned descriptor is the exact authenticated replay; never
                // reopen a replaceable stored-manifest pathname for this snapshot.
                const fs::path admitted = plan_.pinned_manifest ? plan_.pinned_manifest->path() : plan_.manifest;
                const Json document = Json::parse(read_file(admitted));
                const auto observed = feature_state::from_manifest(document, utc_now());
                write_atomic(s_.run_dir / feature_state::kFileName, observed.entries.dump(2) + "\n");
                record_["feature_states"] = observed.entries;
                const auto& spec = document.at("spec");
                const auto& helpers = spec.at("processes").at("helpers");
                if (helpers.empty()) {
                    record_["calibration_record"] = Json{
                        {"status", "none"}, {"reason_code", "no_helpers"},
                        {"reason_detail", "engine-only session has no live policy calibration"}};
                } else {
                    const auto& selected = spec.at("handoff").at("placement").at("live_policy_state");
                    if (selected.at("configured").get<bool>()) {
                        const auto key = policy_calibration_key(policy_calibration_inputs(spec));
                        record_["calibration_record"] = Json{
                            {"status", "selected"}, {"kind", "radiance.live_policy"},
                            {"key_sha256", key}};
                    } else {
                        record_["calibration_record"] = Json{
                            {"status", "none"},
                            {"reason_code", selected.at("reason_code")},
                            {"reason_detail", selected.at("reason_detail")}};
                    }
                }
                for (const auto& warning : observed.warnings)
                    std::cerr << "bc-serve: " << warning << '\n';
                if (s_.strict_production_features && !observed.warnings.empty())
                    throw PlanFailed("strict production feature refusal: " + observed.warnings.front());
            }
            if (!event("planned") && s_.publish_feature_states)
                throw PlanFailed("cannot persist the feature-state snapshot in the owner record");
            if (stop_requested())
                throw Stop{};
            // No-process PWR06 floor, before any serving helper OR engine is
            // launched. The measuring children have already been reaped.
            if (s_.power_hook.gfxoff_window)
                require_callback(s_.power_hook.gfxoff_window.invoke("session.gfxoff.floor", false),
                                 "gfxoff floor receipt");
            if (!plan_.stored_sha256.empty() &&
                sha256_hex(read_file(plan_.execution_manifest())) !=
                    plan_.stored_sha256)
                throw PlanFailed("replay: stored manifest bytes changed after authentication");
            if (s_.before_gpu_start)
                require_callback(s_.before_gpu_start.invoke("session.before_gpu_start", plan_),
                                 "pre-exec process boundary");
            recheck_native_host_room();
            {
                const auto helpers_began = std::chrono::steady_clock::now();
                start_helpers();
                record_phase("helper_start", helpers_began);
            }
            if (s_.quality) {
                score_quality(*s_.quality);
            } else {
                try {
                    const auto engine_began = std::chrono::steady_clock::now();
                    start_engine();
                    if (!wait_health())
                        throw Stop{};
                    record_phase("engine_start_to_ready", engine_began);
                } catch (const Stop&) {
                    throw;
                } catch (...) {
                    // The engine did not come up on a plan this start took from the store: the engine's own
                    // admission is the safety net, so the entry goes and the next start measures afresh.
                    if (!plan_.reused_store.empty()) {
                        std::error_code ignored;
                        fs::remove(plan_.reused_store, ignored);
                        std::cerr << "bc-serve: the engine did not start on the stored plan; the stored entry is "
                                     "removed, the next start measures again\n";
                    }
                    throw;
                }
                event("healthy");
                if (s_.power_hook.gfxoff_window)
                    require_callback(s_.power_hook.gfxoff_window.invoke("session.gfxoff.loaded_idle", true),
                                     "gfxoff loaded idle receipt");
                if (const auto why =
                        canary(plan_.port, s_.canary.prompt, s_.canary.expect, s_.timing.canary_timeout)) {
                    failures.push_back("start canary failed: the engine is up but " + *why);
                    code = 96;
                    throw Failed{};
                }
                event("canary_passed");
                if (s_.runtime_guard) {
                    require_callback(s_.runtime_guard.invoke("session.runtime_guard", plan_, engine_log_),
                                     "runtime replay/heat/policy guard");
                    event("runtime_guard_ready");
                }
                event("running");
                start_calibration_watch();
                while (service(s_.timing.poll)) {
                    poll_calibration_watch();
                    if (s_.runtime_watch)
                        require_callback(s_.runtime_watch.invoke("session.runtime_watch"),
                                         "live heat/policy guard");
                    if (const auto gone = exited_child()) {
                        failures.push_back(*gone + " exited while serving");
                        code = 1;
                        throw Failed{};
                    }
                }
                event("stop_requested", {{"by", stop_reason()}});
            }
        } catch (const Stop&) {
            if (s_.quality) code = 1;
            event("stop_requested", {{"by", stop_reason()}});
        } catch (const Failed&) {
            if (code == 0)
                code = 1; // an explicit failure never looks successful
        } catch (const PlanFailed& e) {
            failures.push_back(e.what());
            code = 97;
        } catch (const callback::Failure& failure) {
            bool plan_refusal = false;
            if (failure.cause()) {
                try {
                    std::rethrow_exception(failure.cause());
                } catch (const PlanFailed&) {
                    plan_refusal = true;
                } catch (...) {
                }
            }
            failures.push_back(failure.what());
            code = plan_refusal ? 97 : 1;
        } catch (const std::exception& e) {
            failures.push_back(e.what());
            if (code == 0)
                code = 1;
        } catch (...) {
            failures.push_back("owner: unexpected non-standard exception");
            if (code == 0)
                code = 1;
        }
        // Stop every child, the engine first, each bounded; a survivor is never reported stopped.
        bool exited = true;
        if (!children_.empty()) {
            event("stopping");
            for (auto it = children_.rbegin(); it != children_.rend(); ++it) {
                if (!stop_child(it->pid, SIGINT, s_.timing.stop_grace)) {
                    exited = false;
                    failures.push_back(
                        it->role + " (pid " + std::to_string(it->pid) +
                        ") or a process of its group is still alive after SIGKILL; not reporting stopped "
                        "while it holds the cards");
                } else {
                    for (auto& p : record_["processes"])
                        if (p.value("pid", 0) == it->pid)
                            p["exited"] = true;
                }
            }
        }
        // Helpers use a cryptographically generated, per-run POSIX segment identity. On verified
        // exit (including a killed creator) the owner can retire it; it must never unlink a shared
        // name while any child or descendant may still map it.
        if (exited && !handoff_shm_.empty()) {
            if (::shm_unlink(handoff_shm_.c_str()) != 0 && errno != ENOENT)
                failures.push_back("cannot retire this session's shared segment " + handoff_shm_ + ": " +
                                   errno_text());
            else
                event("segment_retired");
        }
        // Never restore power if a process or descendant of any tracked group may still hold a card.
        if (exited && power_attempted_) {
            if (s_.power_hook.verified_exit) {
                const auto acknowledged = s_.power_hook.verified_exit.invoke("session.power.verified_exit");
                if (acknowledged)
                    failures.push_back("power exit proof failed: " + acknowledged->message);
            }
            const auto restored = s_.power_hook.restore.invoke("session.power.restore");
            if (!restored)
                failures.push_back("power_hook restore callback failed; keep journal: " +
                                   restored.error().message);
            else if (restored.value())
                failures.push_back("power_hook restore callback failed; keep journal: " +
                                   restored.value()->message);
            else
                event("power_restored");
        }
        // A failed intermediate record write is a session failure even if the hardware
        // and all children were safely stopped; the durable history cannot be trusted.
        if (save_failed_)
            failures.push_back("session record persistence failed; refusing to claim a clean stop");
        for (const auto& f : failures)
            record_["errors"].push_back(f);
        std::string final_state;
        if (!exited)
            final_state = "exit_unverified";
        else if (failures.empty())
            final_state = "stopped";
        else
            final_state = "failed";
        const bool terminal_durable = event(final_state);
        if (!terminal_durable) {
            // The on-disk record may still say "running". Do not present the in-memory
            // stopped state as a successful durable terminal record to any client.
            final_state = "failed";
            record_["state"] = final_state;
            record_["errors"].push_back("terminal session state could not be persisted");
            std::cerr << "bc-serve: terminal session state could not be persisted\n";
        }
        if (!exited)
            code = 3;
        else if ((!terminal_durable || !failures.empty()) && code == 0)
            code = 1;
        for (const auto& f : failures)
            std::cerr << "bc-serve: " << f << '\n';
        reply_stop_clients(final_state, terminal_durable);
        return code;
    }

  private:
    struct Stop : std::exception {};
    struct Failed : std::exception {};
    struct Child {
        std::string role;
        pid_t pid;
        fs::path log;
    };

    const Session& s_;
    int lock_ = -1, listen_ = -1, handoff_lock_ = -1;
    bool stop_ = false;
    bool power_attempted_ = false;
    bool power_admitted_ = false;
    std::string handoff_shm_;
    fs::path engine_log_;
    std::vector<fs::path> helper_logs_;
    std::optional<CalibrationWatch> calibration_watch_;
    std::vector<int> stop_clients_;
    std::vector<Child> children_;
    EngineStart plan_;
    Json record_;

    // A record that cannot be written is reported, never a reason to skip stopping a child (but no process is
    // started that the record cannot name: see track).
    bool save() {
        try {
            write_atomic(s_.run_dir / "session.json", record_.dump(2) + "\n");
            return true;
        } catch (const std::exception& e) {
            if (!save_failed_)
                std::cerr << "bc-serve: the session record cannot be written: " << e.what() << '\n';
            save_failed_ = true;
            return false;
        } catch (...) {
            if (!save_failed_)
                std::cerr << "bc-serve: the session record write threw an unknown exception\n";
            save_failed_ = true;
            return false;
        }
    }
    bool save_failed_ = false;

    void track(const std::string& role, pid_t pid, const fs::path& log,
               const std::vector<std::string>* exact_argv = nullptr) {
        children_.push_back({role, pid, log});
        Json p = {{"role", role}, {"pid", pid}, {"pgid", pid}, {"log", log.string()}};
        if (exact_argv) {
            // Native measurement facts: capture the exact argv that fork/exec
            // received, including its ephemeral heat nonce. Only the owner
            // writes this receipt; no Python is loaded into the engine.
            p["argv"] = *exact_argv;
            p["env"] = plan_.runtime; // curated child overrides, never ambient secrets
        }
        // the child waits behind its start gate until this returns, so it cannot have exited yet; its start
        // time is read whatever its state, so the record names it either way
        if (const auto st = proc_stat(pid))
            p["start_time"] = st->start;
        record_["processes"].push_back(p);
        event("started", {{"role", role}, {"pid", pid}});
        // durable before anything else happens: an owner that dies must leave a record naming this process
        if (!p.contains("start_time") || !save())
            throw std::runtime_error("the session record cannot name " + role + " (pid " + std::to_string(pid) +
                                     "); stopping it: no process runs that a later stop could not find");
    }
    void untrack(pid_t pid) {
        children_.erase(
            std::remove_if(children_.begin(), children_.end(), [&](const Child& c) { return c.pid == pid; }),
            children_.end());
        for (auto& p : record_["processes"])
            if (p.value("pid", 0) == pid)
                p["exited"] = true;
        save();
    }

    // Elapsed monotonic milliseconds of one owner phase, kept in the record's phases_ms (saved by the next event).
    void record_phase(const char* phase, std::chrono::steady_clock::time_point began) {
        record_["phases_ms"][phase] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                          std::chrono::steady_clock::now() - began).count();
    }

    // Off unless the session names a store. Any failure to watch or store is reported, never a serving failure.
    void start_calibration_watch() {
        if (s_.calibration_ingest_store.empty() || helper_logs_.empty()) return;
        try {
            const fs::path admitted = plan_.pinned_manifest ? plan_.pinned_manifest->path() : plan_.manifest;
            calibration_watch_.emplace(helper_logs_.front(), s_.calibration_ingest_store,
                                       live_policy_target_from_manifest(admitted));
            event("calibration_watch", {{"status", "watching"}, {"log", helper_logs_.front().string()},
                                        {"key_sha256", calibration_watch_->store_key()}});
        } catch (const std::exception& e) {
            std::cerr << "bc-serve: calibration ingest not watching: " << e.what() << '\n';
            calibration_watch_.reset();
        }
    }
    void poll_calibration_watch() {
        if (!calibration_watch_) return;
        try {
            if (!calibration_watch_->poll()) return;
            const bool stored = calibration_watch_->state() == CalibrationWatch::State::stored;
            if (!stored) std::cerr << "bc-serve: " << calibration_watch_->detail() << '\n';
            event("calibration_ingest", {{"status", stored ? "stored" : "not_stored"},
                                         {"detail", calibration_watch_->detail()}});
        } catch (const std::exception& e) {
            std::cerr << "bc-serve: calibration ingest: " << e.what() << '\n';
        }
        if (calibration_watch_->state() != CalibrationWatch::State::waiting) calibration_watch_.reset();
    }

    bool event(const std::string& kind, const Json& detail = Json::object()) {
        Json e = {{"at", utc_now()}, {"kind", kind}};
        for (auto it = detail.begin(); it != detail.end(); ++it)
            e[it.key()] = it.value();
        record_["events"].push_back(e);
        record_["state"] = kind;
        return save();
    }

    bool stop_requested() const {
        return stop_ || g_signal != 0;
    }
    std::string stop_reason() const {
        if (g_signal != 0)
            return std::string("signal ") + std::to_string(int(g_signal));
        return stop_ ? "control socket" : "unknown";
    }

    void listen_socket() {
        const SocketAddress addr(s_.run_dir);
        listen_ = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
        if (listen_ < 0)
            throw std::runtime_error("control socket: " + errno_text());
        if (::bind(listen_, addr.get(), sizeof addr.addr) != 0 || ::listen(listen_, 8) != 0)
            throw std::runtime_error("control socket " + addr.path + ": " + errno_text());
        ::chmod(addr.path.c_str(), 0600);
    }

    // Wait up to `d` while answering the control socket; false once a stop is requested.
    bool service(std::chrono::milliseconds d) {
        const auto deadline = Clock::now() + d;
        do {
            if (stop_requested())
                return false;
            const auto left =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
            pollfd p{listen_, POLLIN, 0};
            const int r = ::poll(&p, 1, int(std::clamp<long long>(left, 0, 100)));
            if (r > 0)
                accept_one();
        } while (Clock::now() < deadline);
        return !stop_requested();
    }

    void accept_one() {
        const int fd = ::accept4(listen_, nullptr, nullptr, SOCK_CLOEXEC);
        if (fd < 0)
            return;
        std::string cmd = read_until(fd, Clock::now() + 2s, true);
        cmd = cmd.substr(0, cmd.find('\n'));
        if (cmd == "status") {
            Json reply = record_;
            reply["ok"] = true;
            write_all(fd, reply.dump() + "\n");
            ::close(fd);
        } else if (cmd == "stop") {
            stop_ = true;
            stop_clients_.push_back(fd); // answered once every child has exited (or could not be stopped)
        } else {
            write_all(fd, Json({{"ok", false}, {"error", "unknown command (status, stop)"}}).dump() + "\n");
            ::close(fd);
        }
    }

    void reply_stop_clients(const std::string& state, bool terminal_durable) {
        for (int fd : stop_clients_) {
            Json reply = record_;
            reply["ok"] = terminal_durable && !save_failed_ && state == "stopped";
            if (!terminal_durable) {
                reply["state"] = "failed";
                reply["durable"] = false;
                reply["error"] = "terminal session state could not be persisted";
            }
            write_all(fd, reply.dump() + "\n");
            ::close(fd);
        }
        stop_clients_.clear();
    }

    void recheck_native_host_room() {
        if (!plan_.native_launch || !plan_.pinned_manifest)
            throw PlanFailed("native serving pre-exec requires admitted launch inputs and a sealed manifest");
        const auto manifest = v2::load_manifest(plan_.execution_manifest(), v2::PlanUse::serving);
        if (!manifest)
            throw PlanFailed(v2::render_human("native serving pre-exec", manifest.error()));
        const auto& admitted = manifest.value();
        const auto& spec = admitted.spec;
        const auto native = v2::native_config_view(admitted);
        if (!native || !spec.plan.resource_inputs || !spec.plan.engine_memory)
            throw PlanFailed("native serving pre-exec requires a measured native plan");
        plan_.native_launch->root.require_matches(native->identity.resolution_root);
        const auto& sys_root = plan_.native_launch->sys_root;
        const auto live = measure_session_live_room(admitted, detect(sys_root), sys_root);
        if (!live)
            throw PlanFailed(v2::render_human("native serving pre-exec room", live.error()));
        auto inputs = *spec.plan.resource_inputs;
        inputs.mem_available_bytes = live.value().mem_available_bytes;
        inputs.cgroup_headroom_bytes = live.value().cgroup_headroom_bytes;
        inputs.pin_available_bytes = live.value().pin_available_bytes;
        if (const auto refusal = v2::require_measured_host_room(
                spec.resources.host, spec.processes.engine.scheduler.tp,
                spec.plan.engine_memory->expected.checkpoints.slot_bytes, inputs, plan_.native_helper_resident_mib))
            throw PlanFailed(v2::render_human("native serving pre-exec room", *refusal));
    }

    void start_helpers() {
        // Fresh readiness log per launch: measuring-start output must never be mistaken for current
        // readiness.
        for (const auto& h : plan_.helpers) {
            const fs::path log = s_.run_dir / (h.role + "-" + stamp() + ".log");
            {
                std::ofstream fresh(log, std::ios::trunc);
                if (!fresh)
                    throw std::runtime_error(h.role + ": cannot create readiness log " + log.string());
            }
            const pid_t pid = spawn(
                h.argv, child_environment(environ, h.runtime), log, [&](pid_t p) { track(h.role, p, log); },
                plan_.pinned_manifest ? plan_.pinned_manifest->descriptor() : -1);
            const auto deadline = Clock::now() + s_.timing.helper_timeout;
            bool ready = false;
            while (Clock::now() < deadline) {
                if (const auto gone = exited_child())
                    throw std::runtime_error(h.role + ": helper set lost a child before readiness: " + *gone);
                ready = read_helper_readiness(log) == HelperReadiness::ready;
                if (ready)
                    break;
                if (!service(s_.timing.helper_interval))
                    throw Stop{};
            }
            if (!ready)
                throw std::runtime_error(h.role + " not ready within " +
                                         std::to_string(s_.timing.helper_timeout.count() / 1000) + "s; see " +
                                         log.string());
            // A ready line from a process that just exited does not satisfy the gate.
            if (const auto gone = exited_child())
                throw std::runtime_error(h.role + ": child exited after reporting ready: " + *gone);
            helper_logs_.push_back(log);
            event("ready", {{"role", h.role}, {"pid", pid}});
        }
        if (!plan_.helpers.empty())
            require_segment(plan_.handoff_shm);
    }

    static void validate_quality_paths(const QualityScore& score) {
        if (score.record()) {
            std::error_code error;
            if (!score.reference.is_absolute() || !fs::is_directory(score.reference) ||
                !fs::is_empty(score.reference, error) || error || !score.output.empty())
                throw PlanFailed("quality: recording needs an absolute, existing, empty reference directory and no output");
            if (!score.corpus.is_absolute() || !fs::is_regular_file(score.corpus))
                throw PlanFailed("quality: recording needs an absolute corpus file");
            return;
        }
        if (!score.reference.is_absolute() || !fs::is_regular_file(score.reference / "kld.json"))
            throw PlanFailed("quality: the reference needs an absolute, recorded KL directory");
        if (!score.output.is_absolute() || !fs::is_directory(score.output.parent_path()) ||
            fs::symlink_status(score.output).type() != fs::file_type::not_found ||
            fs::symlink_status(score.output.string() + ".rows").type() != fs::file_type::not_found)
            throw PlanFailed("quality: output and row sidecar need unused absolute paths in an existing directory");
    }

    void score_quality(const QualityScore& score) {
        validate_quality_paths(score);
        event(score.record() ? "recording" : "scoring",
              {{"reference", score.reference.string()}, {"output", score.output.string()}});
        const pid_t engine = start_engine(&score);
        const auto check_helpers = [&] {
            for (const auto& child : children_) {
                if (child.pid != engine && !running(child.pid))
                    throw PlanFailed("quality: " + child.role + " died during KL scoring");
            }
        };
        const auto status = wait_for_finite_engine(engine, Clock::now() + s_.timing.quality_timeout,
                                                   check_helpers, [&] { return service(s_.timing.poll); });
        if (!status) {
            if (stop_requested()) throw Stop{};
            throw PlanFailed("quality: KL scoring timed out");
        }
        check_helpers();
        if (!WIFEXITED(*status) || WEXITSTATUS(*status) != 0)
            throw PlanFailed(std::string("quality: KL ") + (score.record() ? "recording" : "scoring") +
                             " engine failed; see " + engine_log_.string());
        if (score.record()) {
            if (!fs::is_regular_file(score.reference / "kld.json"))
                throw PlanFailed("quality: KL recording did not produce the reference");
            event("quality_recorded", {{"reference", score.reference.string()}});
            return;
        }
        const fs::path rows = score.output.string() + ".rows";
        if (!fs::is_regular_file(score.output) || fs::file_size(score.output) > v2::kMaxDocumentBytes ||
            !fs::is_regular_file(rows))
            throw PlanFailed("quality: KL scoring did not produce the report and row sidecar");
        const Json report = Json::parse(read_file(score.output), nullptr, false);
        const auto positions = report.is_object() && report.contains("all") && report["all"].is_object()
            ? report["all"].value("positions", int64_t{0}) : 0;
        if (positions <= 0 || uint64_t(positions) > UINT64_MAX / 16 ||
            fs::file_size(rows) != uint64_t(positions) * 16)
            throw PlanFailed("quality: KL scoring row sidecar differs from reported positions");
        event("quality_completed", {{"positions", positions}, {"output", score.output.string()}});
    }

    pid_t start_engine(const QualityScore* quality = nullptr) {
        const fs::path log = s_.run_dir / ("engine-" + stamp() + ".log");
        engine_log_ = log;
        // The strict reader accepts this ephemeral flag only for a serving
        // engine with manifest-declared heat. Measuring engines, standalone
        // engines and helpers never receive it; no ambient environment read.
        std::vector<std::string> argv = plan_.argv;
        const fs::path execution_manifest = plan_.execution_manifest();
        const auto manifest = v2::load_manifest(execution_manifest, v2::PlanUse::serving);
        const auto& native_launch = plan_.native_launch;
        if (!native_launch || !plan_.pinned_manifest || !manifest)
            throw PlanFailed("native serving: missing admitted launch inputs or sealed manifest");
        const auto native = v2::native_config_view(manifest.value());
        if (!native || argv.empty() || !fs::path(argv.front()).is_absolute() ||
            argv != std::vector<std::string>{argv.front(), "--manifest", execution_manifest.string(),
                                             "--role", "engine"})
            throw PlanFailed("native serving: launch does not match the admitted engine manifest");
        native_launch->root.require_matches(native->identity.resolution_root);
        if (quality) {
            if (quality->record())
                argv.insert(argv.end(), {"--kld-record", quality->reference.string(), "--kld-corpus", quality->corpus.string()});
            else
                argv.insert(argv.end(), {"--kld-ref", quality->reference.string(), "--kld-out", quality->output.string()});
        } else if (manifest && manifest.value().spec.handoff &&
                   manifest.value().spec.handoff->telemetry.histogram_file) {
            plan_.heat_session_nonce = heat_launch_nonce();
            argv.push_back("--bc-heat-launch-nonce");
            argv.push_back(plan_.heat_session_nonce);
        }
        if (s_.step_phase_trace && !quality) {
            const fs::path trace_dir = s_.run_dir / "step-phases";
            std::error_code error;
            if (!fs::create_directory(trace_dir, error) &&
                (error || !fs::is_directory(trace_dir) || fs::is_symlink(fs::symlink_status(trace_dir))))
                throw PlanFailed("step-phase trace: cannot create safe run-directory output");
            argv.insert(argv.end(), {"--bc-step-phase-dir", trace_dir.string()});
        }
        plan_.engine_launched_at = std::chrono::system_clock::now();
        return spawn(
            argv, native_launch->environment, log,
            [&](pid_t pid) { track("engine", pid, log, &argv); },
            plan_.pinned_manifest->descriptor(), native_launch->root.descriptor());
    }

    std::optional<std::string> exited_child() {
        for (const auto& c : children_) {
            int st = 0;
            if (!running(c.pid, &st)) {
                std::string how = WIFEXITED(st)     ? "exit " + std::to_string(WEXITSTATUS(st))
                                  : WIFSIGNALED(st) ? "signal " + std::to_string(WTERMSIG(st))
                                                    : "gone";
                return c.role + " (pid " + std::to_string(c.pid) + ", " + how + ", log " + c.log.string() + ")";
            }
        }
        return std::nullopt;
    }

    bool wait_health() {
        const auto deadline = Clock::now() + s_.timing.health_timeout;
        while (Clock::now() < deadline) {
            if (const auto gone = exited_child())
                throw std::runtime_error("a child exited during health gate: " + *gone);
            if (const auto reply =
                    http(plan_.port, "GET", "/health", "", std::min(s_.timing.health_interval, 5000ms)))
                if (reply->status >= 200 && reply->status <= 299) {
                    if (const auto gone = exited_child())
                        throw std::runtime_error("a child exited during health gate: " + *gone);
                    return true;
                }
            if (!service(s_.timing.health_interval))
                return false;
        }
        throw std::runtime_error("the engine did not answer /health within " +
                                 std::to_string(s_.timing.health_timeout.count() / 1000) + " s; log " +
                                 children_.back().log.string());
    }
};

} // namespace

int serve(const Session& session) {
    Owner owner(session);
    return owner.run();
}

std::optional<Json> read_record(const fs::path& run_dir) {
    const fs::path path = run_dir / "session.json";
    if (!fs::exists(path))
        return std::nullopt;
    const Json j = Json::parse(read_file(path), nullptr, false);
    if (j.is_discarded())
        return std::nullopt;
    return j;
}

std::optional<Json> control(const fs::path& run_dir, const std::string& command,
                            std::chrono::milliseconds timeout) {
    std::optional<SocketAddress> addr;
    try {
        addr.emplace(run_dir);
    } catch (const std::exception&) {
        return std::nullopt;
    }
    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0)
        return std::nullopt;
    struct Closer {
        int fd;
        ~Closer() {
            ::close(fd);
        }
    } closer{fd};
    if (::connect(fd, addr->get(), sizeof addr->addr) != 0)
        return std::nullopt;
    if (!write_all(fd, command + "\n"))
        return std::nullopt;
    const std::string reply = read_until(fd, Clock::now() + timeout, true);
    const Json j = Json::parse(reply.substr(0, reply.find('\n')), nullptr, false);
    if (j.is_discarded())
        return std::nullopt;
    return j;
}

int start_detached(const std::vector<std::string>& self_argv, const fs::path& run_dir,
                   std::chrono::milliseconds timeout) {
    fs::create_directories(run_dir);
    Env env;
    for (char** e = environ; e && *e; ++e) {
        const std::string kv(*e);
        const size_t eq = kv.find('=');
        if (eq != std::string::npos)
            env[kv.substr(0, eq)] = kv.substr(eq + 1);
    }
    const pid_t pid = spawn(self_argv, env, run_dir / "owner.log");
    const auto deadline = Clock::now() + timeout;
    while (Clock::now() < deadline) {
        int st = 0;
        const bool alive = running(pid, &st);
        if (const auto r = read_record(run_dir); r && r->value("owner_pid", 0) == pid) {
            const std::string state = r->value("state", "");
            if (state == "running" && alive && [&] {
                    const auto answer = control(run_dir, "status", 2s);
                    return answer && answer->value("owner_pid", 0) == pid &&
                           answer->value("state", "") == "running";
                }()) {
                std::cout << "running: owner pid " << pid << ", port " << r->value("port", 0) << ", manifest "
                          << r->value("manifest", "") << "\n";
                return 0;
            }
            if (state == "failed" || state == "stopped" || state == "exit_unverified") {
                std::cerr << "bc-serve start: the session " << state << ":\n";
                for (const auto& e : r->value("errors", Json::array()))
                    std::cerr << "  " << e.get<std::string>() << '\n';
                std::cerr << "owner log " << (run_dir / "owner.log").string() << '\n';
                return 1;
            }
        }
        if (!alive) {
            std::cerr << "bc-serve start: the session owner exited ("
                      << (WIFEXITED(st) ? "exit " + std::to_string(WEXITSTATUS(st)) : std::string("signal"))
                      << "); see " << (run_dir / "owner.log").string() << '\n';
            return WIFEXITED(st) && WEXITSTATUS(st) == 98 ? 98 : 1;
        }
        std::this_thread::sleep_for(200ms);
    }
    std::cerr << "bc-serve start: the session is not running after " << timeout.count() / 1000
              << " s; it goes on in the background (owner pid " << pid
              << "): `bc-serve status` or `bc-serve stop`\n";
    return 1;
}

/* --- the engine-only plan step
 * --------------------------------------------------------------------------------------- */

namespace {

ManifestRun run_planning_child(const std::vector<std::string>& argv, const Env& env, const Hooks& hooks,
                               ChildKind kind, const std::string& role, int inherited_input_fd = -1,
                               int working_directory_fd = -1,
                               size_t max_output_bytes = v2::kMaxDocumentBytes) {
    if (argv.empty() || argv.front().empty())
        throw std::runtime_error("planning child: empty command");
    if ((inherited_input_fd >= 0 && inherited_input_fd < 3) ||
        (working_directory_fd >= 0 && working_directory_fd < 3))
        throw std::runtime_error("planning child: inherited descriptors must be greater than 2");
    const Wait& wait = hooks.wait;
    const fs::path dir = fs::temp_directory_path();
    const std::string tag = stamp() + "-" + std::to_string(Clock::now().time_since_epoch().count());
    const fs::path out = dir / ("bc-serve-" + tag + ".out"), err = dir / ("bc-serve-" + tag + ".err");
    // stdout and stderr apart: spawn sends both to one log, so run through a shell-free split with two logs
    const pid_t pid = [&] {
        int o = ::open(out.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        int e = ::open(err.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (o < 0 || e < 0)
            throw std::runtime_error("cannot create bc-serve output files in " + dir.string());
        std::vector<std::string> envs;
        for (const auto& [k, v] : env) {
            std::string value = k;
            value += '=';
            value += v;
            envs.push_back(std::move(value));
        }
        std::vector<char*> cargv, cenv;
        for (const auto& a : argv)
            cargv.push_back(const_cast<char*>(a.c_str()));
        cargv.push_back(nullptr);
        for (const auto& s : envs)
            cenv.push_back(const_cast<char*>(s.c_str()));
        cenv.push_back(nullptr);
        Gate gate;
        const pid_t p = ::fork();
        if (p == 0) {
            ::setpgid(0, 0); // its own group: stop_child signals the group
            for (int s : {SIGINT, SIGTERM, SIGHUP, SIGPIPE})
                if (::signal(s, SIG_DFL) == SIG_ERR)
                    ::_exit(126);
            const int in = ::open("/dev/null", O_RDONLY);
            if (in >= 0 && ::dup2(in, STDIN_FILENO) < 0)
                ::_exit(126);
            if (::dup2(o, STDOUT_FILENO) < 0 || ::dup2(e, STDERR_FILENO) < 0)
                ::_exit(126);
            cloexec_above_stderr();
            if (!gate.wait_in_child())
                ::_exit(126); // never recorded: run nothing
            if (working_directory_fd >= 0 && ::fchdir(working_directory_fd) != 0)
                ::_exit(126);
            if (inherited_input_fd >= 0) {
                const int flags = ::fcntl(inherited_input_fd, F_GETFD);
                if (flags < 0 || ::fcntl(inherited_input_fd, F_SETFD, flags & ~FD_CLOEXEC) != 0)
                    ::_exit(126);
            }
            ::execve(cargv[0], cargv.data(), cenv.data());
            ::_exit(127);
        }
        ::close(o);
        ::close(e);
        if (p < 0)
            throw std::runtime_error("fork: " + errno_text());
        ::setpgid(p, p); // in both, so the group exists before any stop signals it
        // registered with the owner before it runs (waited for through the owner, whose control socket stays
        // answered): a writer whose exit is not verified here stays the owner's to stop, and the session is
        // never reported stopped while it lives
        record_then_release(gate, p, [&](pid_t q) {
            if (hooks.started)
                hooks.started.require("session.hooks.started", kind, role, q, out);
        });
        return p;
    }();
    bool end_notified = false;
    auto notify_ended = [&] {
        if (hooks.ended && !end_notified) {
            end_notified = true; // an exception must not trigger a duplicate notification
            hooks.ended.require("session.hooks.ended", pid);
        }
    };
    auto finish = [&](bool exited) {
        std::error_code ec;
        fs::remove(out, ec);
        fs::remove(err, ec);
        if (!exited)
            throw std::runtime_error(role == "manifest writer"
                                         ? "plan: the manifest writer (pid " + std::to_string(pid) +
                                               ") is still alive after SIGKILL"
                                         : "plan: " + role + " (pid " + std::to_string(pid) +
                                               ") is still alive after SIGKILL");
        notify_ended();
    };
    int st = 0;
    try {
        while (running(pid, &st)) {
            if (wait && !wait.require("session.hooks.wait", 100ms)) {
                finish(stop_child(pid, SIGTERM, 10s));
                throw PlanFailed(role == "manifest writer" ? "plan: stopped while bc-serve wrote a manifest"
                                                             : "plan: stopped while " + role + " was running");
            }
            if (!wait)
                std::this_thread::sleep_for(20ms);
        }
    } catch (...) {
        // Includes the typed failure of an injected wait or exit notification.
        // Reap first; never let notification errors prevent child cleanup.
        const auto original = std::current_exception();
        if (stop_child(pid, SIGTERM, 10s)) {
            try {
                notify_ended();
            } catch (const std::exception& secondary) {
                std::cerr << role << ": secondary exit notification failed after reap: "
                          << secondary.what() << '\n';
            }
        }
        std::rethrow_exception(original);
    }
    if (!stop_child(pid, SIGKILL, 10s))
        finish(false); // all descendants verified gone
    notify_ended();
    ManifestRun r;
    r.exit = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
    std::error_code output_error, stderr_error;
    const uintmax_t output_size = fs::file_size(out, output_error);
    const uintmax_t error_size = fs::file_size(err, stderr_error);
    if (output_error || stderr_error || output_size > max_output_bytes || error_size > max_output_bytes) {
        std::error_code ignored;
        fs::remove(out, ignored);
        fs::remove(err, ignored);
        throw PlanFailed("plan: " + role + " output exceeds the bounded capture limit");
    }
    r.out = read_file(out);
    r.err = read_file(err);
    std::error_code ec;
    fs::remove(out, ec);
    fs::remove(err, ec);
    return r;
}

} // namespace

ManifestRun run_to_completion(const std::vector<std::string>& argv, const Env& env, const Hooks& hooks) {
    return run_planning_child(argv, env, hooks, ChildKind::manifest_writer, "manifest writer");
}

namespace {
std::string tail_text(const std::string& s, size_t n = 2000);
}

static NativeConfigInspection parse_native_config_report(const Json& doc,
                                                          const NativeConfigSnapshot& snapshot);

NativeConfigInspection inspect_native_config(const fs::path& engine, const NativeConfigSnapshot& snapshot,
                                             const Env& complete_environment, const Hooks& hooks,
                                             int resolution_root_fd) {
    const fs::path absolute_engine = fs::absolute(engine).lexically_normal();
    if (::access(absolute_engine.c_str(), X_OK) != 0)
        throw PlanFailed("native config inspector: matched engine is not executable: " + absolute_engine.string());

    const std::string payload = snapshot.manifest_field().dump();
    auto input = std::make_shared<PinnedManifest>(payload);
    std::optional<NativeExecutionRoot> owned_root;
    if (resolution_root_fd < 0) {
        owned_root.emplace(snapshot.resolution_root());
        resolution_root_fd = owned_root->descriptor();
    }
    NativeExecutionRoot::require_descriptor_matches(resolution_root_fd, snapshot.resolution_root());

    const ManifestRun result = run_planning_child(
        {absolute_engine.string(), "--bc-inspect-native-config", std::to_string(input->descriptor())},
        complete_environment, hooks, ChildKind::native_config_inspector,
        "native config inspector",
        input->descriptor(), resolution_root_fd);
    NativeExecutionRoot::require_descriptor_matches(resolution_root_fd, snapshot.resolution_root());
    if (result.exit != 0)
        throw PlanFailed("native config inspector: matched Radiance rejected the admitted YAML snapshot: " +
                         tail_text(result.err));
    if (result.out.empty() || result.out.size() > v2::kMaxDocumentBytes)
        throw PlanFailed("native config inspector: report is empty or exceeds the strict JSON limit");
    const auto report = v2::parse_strict_json(result.out, "native config inspector report");
    if (!report || !report.value().is_object())
        throw PlanFailed("native config inspector: report failed strict protocol or snapshot identity validation");
    return parse_native_config_report(report.value(), snapshot);
}

static NativeConfigInspection parse_native_config_report(const Json& doc,
                                                         const NativeConfigSnapshot& snapshot) {
    const auto exact_keys = [](const Json& object, std::initializer_list<const char*> keys) {
        if (!object.is_object() || object.size() != keys.size())
            return false;
        return std::all_of(keys.begin(), keys.end(), [&](const char* key) { return object.contains(key); });
    };
    const auto string_field = [](const Json& object, const char* key) {
        return object.contains(key) && object.at(key).is_string();
    };
    const auto nullable_string_field = [&](const Json& object, const char* key) {
        return object.contains(key) && (object.at(key).is_null() || object.at(key).is_string());
    };
    const auto integer_field = [](const Json& object, const char* key) {
        return object.contains(key) && object.at(key).is_number_integer();
    };
    if (!exact_keys(doc, {"protocol", "snapshot", "engine"}) || !string_field(doc, "protocol") ||
        doc.at("protocol") != "bigcherry.native-fleet-inspect/v2" ||
        !exact_keys(doc.value("snapshot", Json::object()), {"sha256", "resolution_root"}) ||
        !string_field(doc.at("snapshot"), "sha256") || doc.at("snapshot").at("sha256") != snapshot.sha256() ||
        !string_field(doc.at("snapshot"), "resolution_root") ||
        doc.at("snapshot").at("resolution_root") != snapshot.resolution_root() ||
        !doc.at("engine").is_object())
        throw PlanFailed("native config inspector: report failed strict protocol or snapshot identity validation");
    const Json& engine_report = doc.at("engine");
    const Json& listener = engine_report.value("listener", Json::object());
    if (!exact_keys(engine_report, {"abi_version", "model", "radiance_home", "radiance_home_source", "kernels",
                                    "kernels_source", "listener", "tp", "tp_wire", "cards", "rank_devices",
                                    "max_context", "max_sequences", "tuning", "x_bigcherry", "settings"}) ||
        !integer_field(engine_report, "abi_version") || !string_field(engine_report, "model") ||
        !string_field(engine_report, "radiance_home") ||
        !nullable_string_field(engine_report, "radiance_home_source") ||
        !engine_report.contains("kernels") || !engine_report.at("kernels").is_array() ||
        !nullable_string_field(engine_report, "kernels_source") || !integer_field(engine_report, "tp") ||
        !engine_report.contains("cards") || !engine_report.at("cards").is_array() ||
        !engine_report.contains("rank_devices") || !engine_report.at("rank_devices").is_array() ||
        !integer_field(engine_report, "max_context") || !integer_field(engine_report, "max_sequences") ||
        !exact_keys(listener, {"host", "port"}) || !string_field(listener, "host") ||
        !integer_field(listener, "port"))
        throw PlanFailed("native config inspector: report omitted typed engine admission fields");

    NativeConfigInspection out;
    const int64_t abi = engine_report.at("abi_version").get<int64_t>();
    const int64_t tp = engine_report.at("tp").get<int64_t>();
    const int64_t port = listener.at("port").get<int64_t>();
    const int64_t context = engine_report.at("max_context").get<int64_t>();
    const int64_t sequences = engine_report.at("max_sequences").get<int64_t>();
    if (abi != static_cast<int64_t>(RAD_ABI_VERSION))
        throw PlanFailed("native config inspector: report ABI differs from the matched engine");
    if (tp <= 0 || tp > std::numeric_limits<int>::max() || port <= 0 ||
        port > std::numeric_limits<uint16_t>::max() || context < 0 || context > UINT32_MAX ||
        sequences <= 0 || sequences > UINT32_MAX)
        throw PlanFailed("native config inspector: report contains out-of-range admission values");
    if (engine_report.at("model").get_ref<const std::string&>().empty() ||
        listener.at("host").get_ref<const std::string&>().empty())
        throw PlanFailed("native config inspector: report contains empty model or listener host");
    out.snapshot_sha256 = snapshot.sha256();
    out.resolution_root = snapshot.resolution_root();
    out.abi_version = static_cast<std::uint32_t>(abi);
    out.model = engine_report.at("model").get<std::string>();
    out.radiance_home = engine_report.at("radiance_home").get<std::string>();
    if (engine_report.at("radiance_home_source").is_string())
        out.radiance_home_source = engine_report.at("radiance_home_source").get<std::string>();
    for (const Json& kernel : engine_report.at("kernels"))
        if (!kernel.is_string() || kernel.get_ref<const std::string&>().empty() ||
            kernel.get_ref<const std::string&>().find(',') != std::string::npos)
            throw PlanFailed("native config inspector: kernels must be individual nonempty names");
    out.settings.kernels = engine_report.at("kernels").get<std::vector<std::string>>();
    if (engine_report.at("kernels_source").is_string())
        out.kernels_source = engine_report.at("kernels_source").get<std::string>();
    for (const Json& card : engine_report.at("cards")) {
        if (!card.is_number_integer())
            throw PlanFailed("native config inspector: cards must contain nonnegative integer ordinals");
        const int64_t ordinal = card.get<int64_t>();
        if (ordinal < 0 || ordinal > std::numeric_limits<int>::max())
            throw PlanFailed("native config inspector: card ordinal is out of range");
        out.cards.push_back(static_cast<int>(ordinal));
    }
    for (const Json& device : engine_report.at("rank_devices"))
        if (!device.is_string() || device.get_ref<const std::string&>().empty())
            throw PlanFailed("native config inspector: rank devices must contain nonempty strings");
    out.rank_devices = engine_report.at("rank_devices").get<std::vector<std::string>>();
    out.listener_host = listener.at("host").get<std::string>();
    out.tp = static_cast<int>(tp);
    if (!nullable_string_field(engine_report, "tp_wire"))
        throw PlanFailed("native config inspector: ill-typed tp_wire");
    if (!engine_report.at("tp_wire").is_null()) out.tp_wire = engine_report.at("tp_wire").get<std::string>();
    try {
        out.x_bigcherry = decode_x_bigcherry(engine_report.at("x_bigcherry"));
    } catch (const std::invalid_argument& error) {
        throw PlanFailed(std::string("native config admission: ") + error.what());
    }
    if (out.x_bigcherry.rank_count() != tp)
        throw PlanFailed("native config admission: x-bigcherry.ranks name " +
                         std::to_string(out.x_bigcherry.rank_count()) + " ranks, but the YAML's tp is " +
                         std::to_string(tp));
    auto& settings = out.settings;
    settings.port = static_cast<uint32_t>(port);
    if (context == 0)
        throw PlanFailed("native config admission: model-defined context is not representable; set max-model-len");
    settings.max_context = static_cast<uint32_t>(context);
    settings.max_sequences = static_cast<uint32_t>(sequences);
    {
        const Json& tuning = engine_report.at("tuning");
        if (!tuning.is_object() ||
            !exact_keys(tuning, {"p2p", "pipelined_prefill", "pipelined_prefill_min_rows", "prefill_governor",
                                 "tp_ownership", "tune_pin"}))
            throw PlanFailed("native config inspector: report omitted the tuning settings");
        for (const auto& [name, value] : tuning.items()) {
            if (value.is_null()) continue;
            const bool rows = name == "pipelined_prefill_min_rows";
            if (rows ? !value.is_number_integer() : (!value.is_string() || value.get_ref<const std::string&>().empty()))
                throw PlanFailed("native config inspector: ill-typed tuning " + name);
            out.settings.tuning.emplace(name, value);
        }
    }
    const Json& runtime = engine_report.at("settings");
    if (!exact_keys(runtime, {"max_batched_tokens", "checkpoint_interval", "placement",
                              "ngram_placement", "kv_dtype", "speculative_tokens", "rope_scaling",
                              "prefix_resume", "prefix_cache", "prefill_interleave"}))
        throw PlanFailed("native config inspector: report omitted typed runtime settings");
    const auto bounded = [&](const char* field, uint32_t min, uint32_t max = UINT32_MAX) {
        const Json& value = runtime.at(field);
        if (!value.is_number_integer() || value < Json(min) || value > Json(max))
            throw PlanFailed(std::string("native config inspector: unrepresentable ") + field);
        return value.get<uint32_t>();
    };
    const auto text = [&](const char* field) {
        const auto& value = runtime.at(field);
        if (!value.is_string() || value.get_ref<const std::string&>().empty())
            throw PlanFailed(std::string("native config inspector: unrepresentable ") + field);
        return value.get<std::string>();
    };
    settings.max_batched_tokens = bounded("max_batched_tokens", 1);
    settings.checkpoint_interval = bounded("checkpoint_interval", 0);
    settings.placement = text("placement");
    settings.ngram_placement = text("ngram_placement");
    // Native "auto" is an authored *intent*: the engine's measuring start
    // resolves its real host/cgroup/pin budgets and publishes the chosen tier.
    // A final manifest must seal that measured tier, not this undecided intent.
    settings.kv_dtype = text("kv_dtype");
    if (runtime.at("speculative_tokens") == -1)
        throw PlanFailed("native config admission: automatic speculative depth is not representable; set num-speculative-tokens");
    settings.speculative_tokens = bounded("speculative_tokens", 0, 64);
    if (runtime.at("rope_scaling") == "")
        throw PlanFailed("native config admission: model-defined RoPE is not representable; set rope-scaling");
    settings.rope_scaling = text("rope_scaling");
    settings.prefix_resume = text("prefix_resume");
    if (!runtime.at("prefix_cache").is_boolean())
        throw PlanFailed("native config inspector: invalid prefix cache setting");
    settings.prefix_cache = runtime.at("prefix_cache").get<bool>();
    if (!runtime.at("prefill_interleave").is_null())
        settings.prefill_interleave = bounded("prefill_interleave", 0);
    out.receipt = doc;
    return out;
}

NativeWriterInput read_native_writer_input(int fd) {
    const std::string bytes = v2::read_sealed_document(fd, "native writer descriptor");
    const auto parsed = v2::parse_strict_json(bytes, "native writer descriptor");
    if (!parsed || !parsed.value().is_object() || parsed.value().size() != 3 ||
        !parsed.value().contains("snapshot") || !parsed.value().contains("inspection") ||
        !parsed.value().contains("config"))
        throw PlanFailed("native writer descriptor: expected sealed snapshot, typed inspection receipt and config label");
    const auto& handoff = parsed.value();
    auto snapshot = NativeConfigSnapshot::from_manifest(handoff.at("snapshot"));
    auto inspection = parse_native_config_report(handoff.at("inspection"), snapshot);
    const Json& label = handoff.at("config");
    if (!label.is_object() || !label.contains("name") || !label.at("name").is_string() ||
        !label.contains("yaml_path") || !label.at("yaml_path").is_string())
        throw PlanFailed("native writer descriptor: config label needs a name and yaml_path");
    InspectedConfig inspected;
    inspected.name = label.at("name").get<std::string>();
    inspected.yaml_path = label.at("yaml_path").get<std::string>();
    inspected.sha256 = inspection.snapshot_sha256;
    inspected.model = inspection.model;
    inspected.tp_wire = inspection.tp_wire;
    inspected.x = inspection.x_bigcherry;
    return {std::move(snapshot), std::move(inspection.settings), std::move(inspected)};
}

static void reconcile_native_config(
    const NativeConfigInspection& report, const NativeConfigSnapshot& snapshot,
    const fs::path& selected_radiance_home, const fs::path& selected_model,
    const std::vector<std::string>& selected_engine_bdfs, int selected_tp) {
    fs::path reported_home(report.radiance_home);
    if (reported_home.is_relative())
        reported_home = snapshot.resolution_root() / reported_home;
    std::error_code home_error, selected_home_error;
    const fs::path resolved_home = fs::weakly_canonical(reported_home, home_error);
    const fs::path verified_home = fs::weakly_canonical(selected_radiance_home, selected_home_error);
    if (home_error || selected_home_error || resolved_home != verified_home)
        throw PlanFailed("native config admission: YAML runtime home differs from the verified matched build home");
    fs::path reported_model(report.model);
    if (reported_model.is_relative())
        reported_model = snapshot.resolution_root() / reported_model;
    std::error_code model_error, selected_model_error;
    const fs::path resolved_model = fs::weakly_canonical(reported_model, model_error);
    const fs::path verified_model = fs::weakly_canonical(selected_model, selected_model_error);
    if (model_error || selected_model_error || resolved_model != verified_model)
        throw PlanFailed("native config admission: YAML model differs from the selected local model");
    if (selected_tp <= 0 || report.tp != selected_tp)
        throw PlanFailed("native config admission: tensor parallelism differs from the selected local profile");
    if (!report.rank_devices.empty() && report.rank_devices != selected_engine_bdfs)
        throw PlanFailed("native config admission: explicit native rank-device order differs from the selected local BDF order");
    if (selected_engine_bdfs.size() != static_cast<size_t>(report.tp))
        throw PlanFailed("native config admission: selected TP and local BDF count differ");
    std::vector<int> expected_cards;
    expected_cards.reserve(selected_engine_bdfs.size());
    for (size_t i = 0; i < selected_engine_bdfs.size(); ++i)
        expected_cards.push_back(static_cast<int>(i));
    if (report.cards != expected_cards)
        throw PlanFailed("native config admission: YAML card ordinals differ from the local rank visibility order");
    if (report.listener_host != "127.0.0.1")
        throw PlanFailed("native config admission: only the loopback listener is supported by the managed owner");
}

static std::shared_ptr<const NativeEngineLaunch> prepare_native_launch(
    const fs::path& resolution_root, const fs::path& runtime_home, const Env& selected, Env overrides,
    const fs::path& sys_root);

InspectedConfig inspect_config_file(const fs::path& engine, const fs::path& yaml, const fs::path& radiance_home,
                                    const fs::path& model_root, const fs::path& sys_root, const Hooks& hooks) {
    return inspect_config_snapshot(engine, NativeConfigSnapshot::read(yaml, model_root), yaml.stem().string(), yaml,
                                   radiance_home, sys_root, hooks);
}

InspectedConfig inspect_config_snapshot(const fs::path& engine, const NativeConfigSnapshot& snapshot,
                                        const std::string& name, const fs::path& yaml_path,
                                        const fs::path& radiance_home, const fs::path& sys_root, const Hooks& hooks,
                                        const Env& runtime_overrides) {
    const fs::path home = fs::canonical(radiance_home);
    const auto launch = prepare_native_launch(snapshot.resolution_root(), home,
                                              {{"ROCR_VISIBLE_DEVICES", "-1"}, {"HIP_VISIBLE_DEVICES", "-1"}},
                                              runtime_overrides, sys_root);
    launch->root.require_matches(snapshot.resolution_root());
    const NativeConfigInspection report =
        inspect_native_config(engine, snapshot, launch->environment, hooks, launch->root.descriptor());
    InspectedConfig out;
    out.name = name;
    out.yaml_path = yaml_path;
    out.radiance_home = home;
    out.sha256 = report.snapshot_sha256;
    out.model = report.model;
    out.tp_wire = report.tp_wire;
    out.x = report.x_bigcherry;
    return out;
}

fs::path derive_build_runtime_home(const fs::path& self, const fs::path& engine_binary) {
    std::error_code error;
    const fs::path engine = fs::weakly_canonical(engine_binary, error);
    if (error || engine.filename() != "radiance" || engine.parent_path().filename() != "bin")
        throw PlanFailed("serve: --engine-binary " + engine_binary.string() + " is not <build>/bin/radiance");
    const fs::path home = engine.parent_path().parent_path() / "radiance_home";
    if (!fs::is_directory(home))
        throw PlanFailed("serve: the engine's build has no radiance_home (" + home.string() + ")");
    const fs::path owner = fs::weakly_canonical(self, error);
    if (error || owner != fs::weakly_canonical(home / "bin" / "bc-serve", error))
        throw PlanFailed("serve: this bc-serve is not " + (home / "bin" / "bc-serve").string() +
                         ": run the bc-serve of the build whose engine serves (--engine-binary)");
    return home;
}

void attach_inspected_config(Options& options, const std::string& name, const fs::path& yaml, const fs::path& self,
                             const fs::path& engine_binary, const Hooks& hooks, const fs::path& resolution_root,
                             const Env& runtime_overrides, const NativeConfigSnapshot* sealed) {
    if (options.inspected) return;
    fs::path file = yaml;
    if (file.empty()) file = config_yaml_for(name, options.configs);
    if (file.empty())
        throw PlanFailed("config " + name + ": no " + name + ".yaml in the config sources (--configs DIR, "
                         "BIGCHERRY_CONFIGS) and no --native-config");
    fs::path home;
    if (const auto layered = layered_string(options, "engine", "radiance_home")) home = *layered;
    if (home.empty() && !engine_binary.empty()) home = derive_build_runtime_home(self, engine_binary);
    if (home.empty() && self.parent_path().filename() == "bin" &&
        self.parent_path().parent_path().filename() == "radiance_home")
        home = self.parent_path().parent_path();
    if (home.empty())
        throw PlanFailed("config " + name + ": the runtime home is unknown (engine.radiance_home from --set, a profile or the site file, or this build's own radiance_home)");
    const fs::path engine = engine_binary.empty() ? home.parent_path() / "bin" / "radiance" : engine_binary;
    // The YAML's relative paths resolve under the admission's resolution root, else the model root.
    const fs::path root = resolution_root.empty() ? options.model_root : resolution_root;
    if (root.empty())
        throw PlanFailed("config " + name + ": reading its YAML needs --model-root (its model paths resolve there)");
    // One read of the bytes serves both inspections when the caller sealed them already.
    options.inspected = sealed ? inspect_config_snapshot(engine, *sealed, name, file, home,
                                                         options.sys_root, hooks, runtime_overrides)
                               : inspect_config_snapshot(engine, NativeConfigSnapshot::read(file, root),
                                                         name, file, home, options.sys_root, hooks,
                                                         runtime_overrides);
    // The runtime home is the machine's fact the inspection used: the resolver takes it as its own setting.
    if (!layered_string(options, "engine", "radiance_home"))
        options.sets.push_back("engine.radiance_home=" + Json(options.inspected->radiance_home.string()).dump());
}

NativeConfigInspection inspect_and_reconcile_native_config(
    const fs::path& engine, const NativeConfigSnapshot& snapshot, const fs::path& selected_radiance_home,
    const fs::path& selected_model, const Env& complete_environment,
    const std::vector<std::string>& selected_engine_bdfs, int selected_tp, const Hooks& hooks,
    int resolution_root_fd) {
    auto report = inspect_native_config(engine, snapshot, complete_environment, hooks, resolution_root_fd);
    reconcile_native_config(report, snapshot, selected_radiance_home, selected_model, selected_engine_bdfs, selected_tp);
    return report;
}

namespace {

std::vector<std::string> resolver_args(const Options& o) {
    std::vector<std::string> a;
    if (o.sys_root != "/")
        a.insert(a.end(), {"--sys-root", o.sys_root.string()});
    if (!o.site.empty())
        a.insert(a.end(), {"--site", o.site.string()});
    if (!o.profile.empty())
        a.insert(a.end(), {"--profile", o.profile.string()});
    for (const auto& c : o.configs)
        a.insert(a.end(), {"--configs", c.string()});
    if (!o.model_root.empty())
        a.insert(a.end(), {"--model-root", o.model_root.string()});
    if (!o.run_dir.empty())
        a.insert(a.end(), {"--run-dir", o.run_dir.string()});
    if (!o.calibration_cache.empty())
        a.insert(a.end(), {"--calibration", o.calibration_cache.string()});
    for (const auto& s : o.sets)
        a.insert(a.end(), {"--set", s});
    return a;
}

std::string tail_text(const std::string& s, size_t n) {
    return s.size() > n ? s.substr(s.size() - n) : s;
}

// One BDF-to-role assignment, as established by the existing C++ resolver. No implicit device selection.
// Roles are ordered; the first helper creates the shared segment and later helpers attach.
struct HelperSelection {
    const Json& processes;
    const Json& facts;
};

std::vector<HelperStart> helper_starts(HelperSelection selected, const fs::path& home,
                                       const fs::path& helper_binary, const fs::path& manifest,
                                       bool measuring) {
    const Json& processes = selected.processes;
    const Json& facts = selected.facts;
    std::vector<HelperStart> starts;
    for (size_t i = 0; i < processes.size(); ++i) {
        const Json& worker = processes.at(i);
        const std::string bdf = worker.at("bdf");
        const Json* detected = nullptr;
        for (const Json& gpu : facts.at("gpus"))
            if (gpu.at("bdf") == bdf)
                detected = &gpu;
        if (detected == nullptr || detected->at("hip_ordinal") != worker.at("hip_ordinal") ||
            detected->at("arch") != worker.at("arch"))
            throw PlanFailed("helper" + std::to_string(i) +
                             ": BDF/ordinal/architecture changed since the resolver selected it");
        HelperStart h;
        h.role = "helper" + std::to_string(i);
        h.argv = {helper_binary.string(), "--manifest", manifest.string(), "--role", h.role};
        if (measuring)
            h.argv.push_back("--measuring");
        h.runtime = {{"RADIANCE_HOME", home.string()},
                     {"ROCR_VISIBLE_DEVICES", std::to_string(worker.at("hip_ordinal").get<int>())},
                     {"HIP_VISIBLE_DEVICES", "0"}};
        starts.push_back(std::move(h));
    }
    return starts;
}

// A plan-cache miss must measure with the same helpers holding their memory as in the real start.
// Each process is recorded before exec; an unverified exit remains on the owner's list.
void require_segment(const std::string& name) {
    if (name.empty())
        return; // engine-only/fake session
    const int raw = ::shm_open(name.c_str(), O_RDONLY | O_CLOEXEC, 0);
    if (raw < 0)
        throw PlanFailed("helpers reported ready but the shared segment " + name +
                         " is absent: " + errno_text());
    struct SharedDescriptor {
        int fd;
        ~SharedDescriptor() {
            ::close(fd);
        }
    };
    const SharedDescriptor segment{raw};
    struct stat st{};
    const bool valid = ::fstat(segment.fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0;
    if (!valid)
        throw PlanFailed("helpers reported ready but " + name + " is not a sized shared segment");
}

void launch_measure_helpers(const std::vector<HelperStart>& helpers, const fs::path& run_dir,
                            const Hooks& hooks, std::vector<pid_t>* pids, int inherited_manifest_fd) {
    for (const HelperStart& h : helpers) {
        const fs::path log = run_dir / (h.role + "-measuring-" + stamp() + ".log");
        {
            std::ofstream fresh(log, std::ios::trunc);
            if (!fresh)
                throw PlanFailed("cannot create fresh helper measurement log " + log.string());
        }
        const pid_t pid = spawn(h.argv, child_environment(environ, h.runtime), log, [&](pid_t p) {
            if (hooks.started)
                require_callback(hooks.started.invoke("session.hooks.started", ChildKind::measuring_helper,
                                                      h.role + " measuring", p, log),
                                 "hooks.started");
        }, inherited_manifest_fd);
        pids->push_back(pid);
        const auto deadline = Clock::now() + std::chrono::minutes(15);
        bool ready = false;
        while (Clock::now() < deadline) {
            for (const pid_t p : *pids)
                if (!running(p))
                    throw PlanFailed(h.role + ": measuring helper exited before the whole set was ready; log " +
                                     log.string());
            ready = read_helper_readiness(log) == HelperReadiness::ready;
            if (ready)
                break;
            if (hooks.wait) {
                if (!require_callback(hooks.wait.invoke("session.hooks.wait", 200ms), "hooks.wait"))
                    throw PlanFailed(h.role + ": measurement stopped during helper readiness");
            } else {
                std::this_thread::sleep_for(200ms);
            }
        }
        if (!ready || !running(pid))
            throw PlanFailed(h.role + ": measuring helper not ready/alive; see " + log.string());
    }
}

uint64_t helper_resident_mib(const std::vector<pid_t>& helpers) {
    uint64_t anon = 0, shared = 0;
    for (const pid_t pid : helpers) {
        if (!running(pid))
            throw PlanFailed("measuring helper exited before resident-memory sample");
        std::ifstream status("/proc/" + std::to_string(pid) + "/status");
        uint64_t a = 0, b = 0;
        bool have_a = false, have_b = false;
        for (std::string line; std::getline(status, line);) {
            std::istringstream words(line);
            std::string label;
            uint64_t n;
            if (words >> label >> n) {
                if (label == "RssAnon:") {
                    a = n;
                    have_a = true;
                }
                if (label == "RssShmem:") {
                    b = n;
                    have_b = true;
                }
            }
        }
        if (!have_a || !have_b)
            throw PlanFailed("cannot measure RssAnon/RssShmem of helper pid " + std::to_string(pid));
        anon += a;
        shared = std::max(shared, b);
    }
    return (anon + shared + 1023) / 1024; // kB -> MiB; shared segment counted once
}

void stop_measure_helpers(std::vector<pid_t>* helpers, const Hooks& hooks) {
    bool all_exited = true;
    std::optional<std::string> callback_failure;
    for (auto it = helpers->rbegin(); it != helpers->rend(); ++it) {
        // Continue reaping every group even if a notification hook fails.
        if (stop_child(*it, SIGINT, hooks.helper_stop_timeout)) {
            if (hooks.ended) {
                const auto result = hooks.ended.invoke("session.hooks.ended", *it);
                if (result && !callback_failure)
                    callback_failure = "hooks.ended: " + result->message;
            }
        } else {
            all_exited = false; // the owner still tracks an unverified group
        }
    }
    helpers->clear();
    if (!all_exited)
        throw PlanFailed("a measuring helper or descendant survived SIGKILL; not reporting stopped");
    if (callback_failure)
        throw PlanFailed(*callback_failure);
}

// A measuring start never transfers these process groups to a serving session.
class MeasuringHelpers {
  public:
    explicit MeasuringHelpers(const Hooks& hooks) : hooks_(hooks) {}
    MeasuringHelpers(const MeasuringHelpers&) = delete;
    MeasuringHelpers& operator=(const MeasuringHelpers&) = delete;
    ~MeasuringHelpers() noexcept {
        try {
            stop_measure_helpers(&pids, hooks_);
        } catch (...) { // NOLINT(bugprone-empty-catch)
            // An unverified group stays registered with the session owner.
        }
    }
    void stop() {
        stop_measure_helpers(&pids, hooks_);
    }
    std::vector<pid_t> pids;

  private:
    const Hooks& hooks_;
};

vram::Baseline measure_helper_baseline(const Json& helpers, const EnginePlanInputs& in) {
    std::vector<std::string> bdfs;
    for (const auto& helper : helpers)
        bdfs.push_back(helper.at("bdf").get<std::string>());
    try {
        return vram::snapshot(bdfs, in.options.sys_root, in.helper_vram);
    } catch (const vram::Refused& error) {
        throw PlanFailed(error.what());
    }
}

void release_measuring_helpers(MeasuringHelpers& group, const vram::Baseline& baseline,
                               const EnginePlanInputs& in, const Hooks& hooks) {
    group.stop();
    // A reaped helper may still have pending VRAM accounting in the driver.
    vram::Hardware io = in.helper_vram;
    if (!io.wait && hooks.wait)
        io.wait = [&](std::chrono::milliseconds delay) {
            return require_callback(hooks.wait.invoke("session.hooks.wait", delay), "helper VRAM release wait");
        };
    try {
        vram::await_release(baseline, in.options.sys_root, hooks.helper_stop_timeout, io);
    } catch (const vram::Refused& error) {
        throw PlanFailed(error.what());
    }
}

int wait_measuring_engine(pid_t pid, const MeasuringHelpers& helpers, const Hooks& hooks) {
    std::optional<int> status;
    std::string refusal;
    try {
        const auto check_helpers = [&] {
            for (size_t i = 0; i < helpers.pids.size(); ++i) {
                if (!running(helpers.pids[i]))
                    throw PlanFailed("plan: helper" + std::to_string(i) + " died during engine memory measurement");
            }
        };
        status = wait_for_finite_engine(pid, Clock::now() + 30min, check_helpers, [&] {
            return require_callback(hooks.wait.invoke("session.hooks.wait", 200ms), "measuring wait");
        });
        if (!status)
            refusal = "plan: the measuring start was stopped (a stop request or 30 min)";
    } catch (...) {
        if (stop_child(pid, SIGINT, 60s) && hooks.ended)
            require_callback(hooks.ended.invoke("session.hooks.ended", pid), "hooks.ended");
        throw;
    }
    if (!stop_child(pid, refusal.empty() ? SIGKILL : SIGINT, refusal.empty() ? 10s : 60s))
        throw PlanFailed("plan: a process of the measuring engine's group survived SIGKILL");
    if (hooks.ended)
        require_callback(hooks.ended.invoke("session.hooks.ended", pid), "hooks.ended");
    if (!refusal.empty())
        throw PlanFailed(refusal);
    // A helper can exit between the last live-engine poll and the engine reap.
    for (size_t i = 0; i < helpers.pids.size(); ++i)
        if (!running(helpers.pids[i]))
            throw PlanFailed("plan: helper" + std::to_string(i) + " died during engine memory measurement");
    return *status;
}

void check_native_helper_roles(const v2::Manifest& manifest, const Json& workers,
                               const std::vector<std::string>& engine_bdfs, const std::string& handoff) {
    const auto& helpers = manifest.spec.processes.helpers;
    if (helpers.size() != workers.size() || bool(manifest.spec.handoff) != !handoff.empty() ||
        (manifest.spec.handoff && manifest.spec.handoff->transport.shm != handoff))
        throw PlanFailed("native v3 manifest changed the selected helpers or owner segment");
    std::set<std::string> occupied(engine_bdfs.begin(), engine_bdfs.end());
    for (size_t i = 0; i < helpers.size(); ++i)
        if (helpers[i].role != "helper" + std::to_string(i) ||
            helpers[i].gpu != workers.at(i).at("bdf").get<std::string>() ||
            !occupied.insert(helpers[i].gpu).second)
            throw PlanFailed("native v3 manifest changed helper BDF order or overlaps an engine/helper BDF");
}

int64_t requested_host_pool(const ManifestRun& writer) {
    static const std::regex receipt(R"(^derived_host_pool_mib=([0-9]+)\r?$)", std::regex::multiline);
    auto match = std::sregex_iterator(writer.out.begin(), writer.out.end(), receipt);
    if (match == std::sregex_iterator())
        throw PlanFailed("plan: bc-serve asked for another host pool without naming it: " + tail_text(writer.err));
    const std::string value = (*match)[1];
    int64_t result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (++match != std::sregex_iterator() || error != std::errc{} || end != value.data() + value.size())
        throw PlanFailed("plan: bc-serve reported an invalid or ambiguous derived host pool");
    return result;
}

// A protocol-level count, not a resource budget. The writer's derived host pool depends on the host memory the
// measured helpers and engine hold, which itself depends on the pool: measuring with the configured pool yields the
// pool that measurement needs (change 1), measuring with that pool confirms it or moves it once more for the memory
// the first change itself released or took (change 2). A third change means the pool does not settle. Repeat
// detection (the native path's set of pools already measured) stops a loop but not a pool that keeps moving to new
// values, and every further change is a full measuring run, so this bound stays and the refusal names the last pool.
constexpr int kMaxHostPoolChanges = 2;

// Names every leaf where two plan-key inputs differ, so a refusal says which input moved (a host fact, a derived
// budget, a device field) and not only that "something changed".
void collect_key_differences(const Json& was, const Json& now, const std::string& path, std::vector<std::string>& out) {
    if (was.is_object() && now.is_object()) {
        for (auto it = was.begin(); it != was.end(); ++it) {
            if (!now.contains(it.key())) out.push_back(path + "/" + it.key() + ": " + it.value().dump() + " -> absent");
            else collect_key_differences(it.value(), now.at(it.key()), path + "/" + it.key(), out);
        }
        for (auto it = now.begin(); it != now.end(); ++it)
            if (!was.contains(it.key())) out.push_back(path + "/" + it.key() + ": absent -> " + it.value().dump());
    } else if (was.is_array() && now.is_array() && was.size() == now.size()) {
        for (size_t i = 0; i < was.size(); ++i) collect_key_differences(was[i], now[i], path + "/" + std::to_string(i), out);
    } else if (was != now) {
        out.push_back(path + ": " + was.dump() + " -> " + now.dump());
    }
}

std::string key_differences(const Json& was, const Json& now) {
    std::vector<std::string> out;
    collect_key_differences(was, now, "", out);
    std::string text;
    for (const std::string& line : out) text += (text.empty() ? "" : "; ") + line;
    return text.empty() ? "no leaf differs" : text;
}

} // namespace

// A named RSC12 config may have a prior immutable manifest. Its key covers
// all static input bytes; live room is intentionally excluded and remeasured
// by replay_stored_manifest on EVERY start.
struct NamedCache {
    std::string key;
    fs::path index;          // the work root's named-config store entry; empty without a work root
    std::string model_digest; // model_stat_identity(): path, size, mtime
    std::map<std::string, std::string> inputs; // static_input_digests(): lets a later miss name what moved
};

struct NativeReplayAdmission {
    const NativeConfigSnapshot& snapshot;
    const NativeConfigInspection& inspection;
    std::shared_ptr<const NativeEngineLaunch> launch;
};

static EngineStart replay_stored_manifest(const EnginePlanInputs& given, const fs::path& run_dir,
                                          const Hooks& hooks, const NativeReplayAdmission* expected_native);

static std::shared_ptr<const NativeEngineLaunch> prepare_native_launch(
    const fs::path& resolution_root, const fs::path& runtime_home, const Env& selected, Env overrides,
    const fs::path& sys_root) {
    for (const char* key : {"RADIANCE_HOME", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                            "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"})
        if (overrides.count(key))
            throw PlanFailed(std::string("native config admission: runtime override cannot select ") + key);
    for (const auto& [key, value] : selected)
        overrides[key] = value;
    overrides["RADIANCE_HOME"] = runtime_home.string();
    return std::make_shared<const NativeEngineLaunch>(resolution_root, current_child_environment(overrides),
                                                      fs::absolute(sys_root));
}

static Env engine_visibility(const fs::path& home, const std::vector<std::string>& cards, const Json& facts) {
    std::string rocr, hip;
    for (size_t i = 0; i < cards.size(); ++i) {
        const auto& inventory = facts.at("gpus");
        const auto gpu = std::find_if(inventory.begin(), inventory.end(), [&](const auto& card) {
            return card.at("bdf") == cards[i];
        });
        if (gpu == inventory.end())
            throw PlanFailed("native config admission: engine BDF not found in current inventory: " + cards[i]);
        rocr += (i ? "," : "") + std::to_string(gpu->at("hip_ordinal").get<int64_t>());
        hip += (i ? "," : "") + std::to_string(i);
    }
    return {{"RADIANCE_HOME", home.string()}, {"ROCR_VISIBLE_DEVICES", rocr}, {"HIP_VISIBLE_DEVICES", hip}};
}

static bool native_settings_match(const NativeEngineSettings& native, const v2::Spec& spec) {
    const auto& engine = spec.processes.engine;
    const auto& execution = engine.execution;
    const auto& service = spec.intent.service;
    const auto prefill = engine.tuning ? engine.tuning->prefill_interleave : std::nullopt;
    return native.port == engine.server.port &&
           native.max_context == service.max_context_tokens && native.max_sequences == service.max_sessions &&
           native.kernels == engine.runtime.kernels &&
           native.max_batched_tokens == engine.scheduler.max_num_batched_tokens &&
           native.checkpoint_interval == engine.scheduler.checkpoint_interval &&
           native.placement == v2::kWeightPlacementNames[static_cast<size_t>(execution.placement)] &&
           (native.ngram_placement == "auto" ||
            native.ngram_placement == v2::kNgramPlacementNames[static_cast<size_t>(execution.ngram_placement)]) &&
           native.kv_dtype == v2::kKvCacheDtypeNames[static_cast<size_t>(execution.kv_cache_dtype)] &&
           native.speculative_tokens == execution.num_speculative_tokens &&
           native.rope_scaling == execution.rope_scaling &&
           native.prefix_resume == v2::kPrefixResumeNames[static_cast<size_t>(execution.prefix_resume)] &&
           native.prefix_cache == !execution.no_prefix_cache.has_value() &&
           native.prefill_interleave == prefill;
}

std::optional<NamedCache> named_cache(const EnginePlanInputs& in, const Resolved& resolved, const Json& facts,
                                      const fs::path& engine, const fs::path& helper, const fs::path& run_dir,
                                      const NativeConfigSnapshot* native = nullptr) {
    if (in.named_config.empty())
        return std::nullopt; // preserve the existing --set/layout serve path
    const Json& values = resolved.values;
    const std::string name = values.at("layout").at("name").get<std::string>();
    if (name != in.named_config)
        throw PlanFailed("named CONFIG differs from the effective selected config: " + name);
    StaticResolutionInputs static_inputs;
    static_inputs.effective_values = values;
    // Preserve operator CLI overrides separately from measured-derived values.
    // RSC12's named layout selection already appears in effective_values.
    for (const std::string& set : in.options.sets) {
        const std::string key = set.substr(0, set.find('='));
        // engine.radiance_home is the installed build's own path (main.cpp adds it as a --set on every start); which
        // build runs the plan is provenance, never part of the key.
        if (key != "layout.name" && key != "handoff.shm" && key != "engine.radiance_home")
            static_inputs.operator_sets.push_back(set);
    }
    // What was measured is reused by the next start of this config from any run directory, whichever build
    // runs it: the key names the inputs that decide the plan, never the engine, helper or bc-serve build. The
    // store sits in the work root (--work-root, passed explicitly: no environment is read here); without one
    // nothing is stored and every start measures.
    const fs::path index = in.work_root.empty() ? fs::path{} : named_store_index(in.work_root, name);
    const std::string model_path = values.at("engine").at("model").get<std::string>();
    static_inputs.model_digest = model_stat_identity(model_path);
    static_inputs.power_profile = in.power_profile;

    static_inputs.ordered_devices = Json::array();
    for (const char* group : {"engine_cards", "helper_cards"}) {
        for (const Json& selected : values.at("layout").at(group)) {
            const Json* detected = nullptr;
            for (const Json& device : facts.at("gpus"))
                if (device.at("bdf") == selected)
                    detected = &device;
            if (!detected)
                throw PlanFailed("static key: selected card absent: " + selected.get<std::string>());
            static_inputs.ordered_devices.push_back({{"role_group", group},
                                                     {"bdf", selected},
                                                     {"arch", detected->at("arch")},
                                                     {"hip_ordinal", detected->at("hip_ordinal")},
                                                     {"vram_total_mib", detected->at("vram_total_mib")}});
        }
    }
    // Each input file is named by its role and keyed by its content identity, never by where it sits: the
    // installed prefix and the run directory differ between starts and builds, the bytes are what decide the plan.
    auto file = [&](const std::string& role, const fs::path& path) {
        if (path.empty())
            return;
        const auto absolute = fs::absolute(path).lexically_normal();
        static_inputs.file_digests[role] = complete_file_identity(absolute);
    };
    for (size_t i = 0; i < in.options.configs.size(); ++i)
        file("config/" + std::to_string(i), in.options.configs[i]);
    file("site", in.options.site);
    file("profile", in.options.profile);
    file("calibration_cache", in.options.calibration_cache);
    file("calibration_records", in.calibration_records_file);
    if (values.contains("handoff") && values.at("handoff").contains("heat") &&
        values.at("handoff").at("heat").is_string() &&
        !values.at("handoff").at("heat").get<std::string>().empty())
        file("handoff.heat", values.at("handoff").at("heat").get<std::string>());
    NamedCache cache{static_resolution_key(static_inputs), index, static_inputs.model_digest};
    cache.inputs = static_input_digests(static_inputs);
    return cache;
}

// The measurement an earlier start stored (its plan, per-rank demand and resident host memory), read back so a
// start under another build can seal it again instead of measuring: the build that measured it is provenance.
struct CarriedMeasurement {
    Json plan_json, rank_demand;
    HostResident resident;
    int64_t host_pool_mib = 0;
    std::string attempt, engine_identity, manifest;
    fs::path runtime_home; // the runtime home the stored manifest embeds (where the measuring build lived)
    Json provenance() const {
        return {{"attempt", attempt}, {"engine_identity", engine_identity}, {"manifest", manifest}};
    }
};

// The carried measurement, or why there is none (a fixed reason class; exception text is the caller's to name).
struct CarriedLoad {
    std::optional<CarriedMeasurement> carried;
    std::string reason;
};

static CarriedLoad load_carried_measurement(const fs::path& stored) {
    try {
        const std::string bytes = read_manifest_file(stored);
        const auto verified = v2::read_manifest(bytes, stored.string(), v2::PlanUse::serving);
        if (!verified || !verified.value().spec.plan.engine_memory)
            return {std::nullopt, "the stored manifest carries no engine measurement"};
        const auto& m = verified.value();
        const StoredManifestIndex bound = read_stored_index(stored.string() + ".replay-index.json");
        if (!bound.native_measurement || bound.manifest_filename != stored.filename().string())
            return {std::nullopt, "the stored manifest has no bound native measurement"};
        const fs::path owner_dir =
            stored.parent_path() / "measurement-attempts" / bound.native_measurement->attempt_id / "plan-cache";
        // One entry per measuring attempt: its own recorded key inputs name it.
        std::vector<fs::path> entries;
        for (const auto& entry : fs::directory_iterator(owner_dir))
            if (entry.is_regular_file() && entry.path().extension() == ".json")
                entries.push_back(entry.path());
        if (entries.size() != 1)
            return {std::nullopt, "the measuring attempt's plan cache does not hold exactly one entry"};
        const Json recorded = parse_plan_json(read_file(entries.front()), entries.front().string());
        const auto cached = plan_cache_lookup(owner_dir, recorded.at("key_inputs"));
        if (!cached || !cached->owner || cached->owner->attempt_id != bound.native_measurement->attempt_id ||
            cached->owner_sha256 != bound.native_measurement->sha256)
            return {std::nullopt, "the measuring attempt's plan cache entry is absent or not the bound owner's"};
        return {CarriedMeasurement{cached->plan_json, cached->rank_demand, cached->resident,
                                   m.spec.resources.host.pool_mib, bound.native_measurement->attempt_id,
                                   m.spec.plan.engine_memory->engine_identity, stored.string(),
                                   m.spec.processes.engine.runtime.radiance_home},
                {}};
    } catch (const std::exception&) {
        return {std::nullopt, "the stored measurement cannot be read back"};
    }
}

EngineStart plan_engine_only(const EnginePlanInputs& given, const fs::path& run_dir, const Hooks& hooks) {
    const Wait& wait = hooks.wait;
    // Monotonic phase durations, summed per phase: O(phases) work, none of it on a hot path.
    Json phases = Json::object();
    auto lap_start = std::chrono::steady_clock::now();
    auto lap = [&](const char* phase) {
        const auto now = std::chrono::steady_clock::now();
        phases[phase] = phases.value(phase, int64_t{0}) +
                        std::chrono::duration_cast<std::chrono::milliseconds>(now - lap_start).count();
        lap_start = now;
    };
    EnginePlanInputs in = given;
    in.options.run_dir = run_dir;
    if (!in.native_admission)
        throw PlanFailed("serve: a fresh start needs an admitted native config YAML (NAME.yaml in the bundle or "
                         "--configs, or --native-config)");
    const NativeAdmissionRequest& admission = *in.native_admission;
    // The YAML's bytes, read once for every inspection.
    const NativeConfigSnapshot sealed_snapshot =
        NativeConfigSnapshot::read(admission.authored_yaml, admission.resolution_root);
    if (!in.named_config.empty())   // the config is its native YAML, read before anything is resolved
        attach_inspected_config(in.options, in.named_config, admission.authored_yaml, in.self, in.engine_binary,
                                hooks, admission.resolution_root, admission.runtime_overrides, &sealed_snapshot);
    if (!in.options.inspected)
        throw PlanFailed("serve: no config was read: name a CONFIG (NAME.yaml in the bundle or --configs)");
    in.options.configs.clear();   // the sealed YAML is the one source: no file is read again

    const Json facts = detect(in.options.sys_root);
    const Resolved r = resolve(in.options, facts);
    const Json& v = r.values;
    const std::string name = v.at("layout").at("name");
    const Json layouts = load_layouts(in.options);
    if (layouts.at("kind").value(name, "") != "config")
        throw PlanFailed(
            "serve: " + name + " is a layout (" + layouts.at("source").value(name, "") +
            "); bc-serve serves configs (engines/radiance/configs, --configs): convert it first (RSC12)");
    const Json& processes = r.manifest.at("process_helpers");
    const fs::path home = v.at("engine").at("radiance_home").get<std::string>();
    const fs::path engine =
        in.engine_binary.empty() ? home.parent_path() / "bin" / "radiance" : in.engine_binary;
    if (::access(engine.c_str(), X_OK) != 0)
        throw PlanFailed("serve: the engine binary " + engine.string() +
                         " is not executable (--engine-binary)");
    const fs::path helper_binary =
        in.helper_binary.empty() ? home / "helpers" / "r3_moe_worker" : in.helper_binary;
    if (!processes.empty() && ::access(helper_binary.c_str(), X_OK) != 0)
        throw PlanFailed("serve: helper executable " + helper_binary.string() + " is absent");
    // bc-serve from a build (<build>/radiance_home/bin/bc-serve) writes manifests the engine of the same build
    // reads: the plan key and the manifest reader are compiled into both, so an engine or runtime home from
    // another build is refused.
    const fs::path own_home = in.self.parent_path().parent_path();
    const fs::path own_engine = own_home.parent_path() / "bin" / "radiance";
    if (in.self.parent_path().filename() == "bin" && own_home.filename() == "radiance_home" &&
        fs::exists(own_engine)) {
        if (fs::weakly_canonical(home) != fs::weakly_canonical(own_home))
            throw PlanFailed("serve: engine.radiance_home " + home.string() +
                             " is not this bc-serve's build (" + own_home.string() +
                             "); run the bc-serve of the build that serves");
        if (fs::weakly_canonical(engine) != fs::weakly_canonical(own_engine))
            throw PlanFailed("serve: the engine " + engine.string() + " is not this bc-serve's build (" +
                             own_engine.string() +
                             "); a manifest is read by the engine of the build that wrote it");
        if (!processes.empty() &&
            fs::weakly_canonical(helper_binary) != fs::weakly_canonical(own_home / "helpers" / "r3_moe_worker"))
            throw PlanFailed("serve: helper binary " + helper_binary.string() +
                             " must come from this bc-serve's build");
    }

    // Rank order is the resolver's card order; the process sees its cards as HIP devices 0..n-1.
    std::string rocr, hip;
    const Json& cards = v.at("layout").at("engine_cards");
    for (size_t i = 0; i < cards.size(); ++i) {
        const Json* gpu = nullptr;
        for (const Json& g : facts.at("gpus"))
            if (g.at("bdf") == cards[i])
                gpu = &g;
        if (gpu == nullptr)
            throw PlanFailed("serve: engine card " + cards[i].get<std::string>() + " is not in the inventory");
        rocr += (i != 0 ? "," : "") + std::to_string(gpu->at("hip_ordinal").get<int64_t>());
        hip += (i != 0 ? "," : "") + std::to_string(i);
    }
    EngineStart out;
    out.port = int(v.at("engine").at("port").get<int64_t>());
    out.runtime = {
        {"RADIANCE_HOME", home.string()}, {"ROCR_VISIBLE_DEVICES", rocr}, {"HIP_VISIBLE_DEVICES", hip}};
    out.summary = {{"config", name},
                   {"source", layouts.at("source").value(name, "")},
                   {"model", v.at("engine").at("model")},
                   {"engine_cards", cards},
                   {"engine", engine.string()},
                   {"helper_cards", v.at("layout").at("helper_cards")}};

    const fs::path frozen_runtime_home = fs::canonical(home);
    if (in.options.inspected && in.options.inspected->radiance_home != frozen_runtime_home)
        throw PlanFailed("native config admission: the runtime home changed between the config parse and the "
                         "admission (" + in.options.inspected->radiance_home.string() + " then " +
                         frozen_runtime_home.string() + ")");
    if (in.quality_score) {
        const fs::path owner = frozen_runtime_home / "bin/bc-serve";
        if (in.self != owner || fs::canonical(owner) != owner)
            throw PlanFailed("quality requires the matched runtime home's bc-serve executable");
    }
    out.native_launch = prepare_native_launch(admission.resolution_root, frozen_runtime_home,
                                              out.runtime, admission.runtime_overrides,
                                              in.options.sys_root);
    const auto& launch = *out.native_launch;
    const NativeConfigSnapshot& snapshot = sealed_snapshot;
    launch.root.require_matches(snapshot.resolution_root());
    const auto selected_bdfs = v.at("layout").at("engine_cards").get<std::vector<std::string>>();
    lap("yaml_inspect");
    const std::string matched_engine_identity = engine_identity(engine, frozen_runtime_home);
    lap("runtime_identity");
    const NativeConfigInspection native_inspection = inspect_and_reconcile_native_config(
        engine, snapshot, frozen_runtime_home, v.at("engine").at("model").get<std::string>(),
        launch.environment, selected_bdfs, v.at("engine").at("tp").get<int>(), hooks, launch.root.descriptor());
    lap("yaml_inspect");
    if (in.native_preflight) {
        // Same inspection and reconcile as a start (above); nothing past it runs: no cache reuse, writer, power
        // admission, engine or helper child, so no GPU is touched.
        out.native_preflight_receipt = native_inspection.receipt;
        out.native_preflight_values = resolve(in.options, facts, &native_inspection.settings).values;
        out.port = native_inspection.settings.port;
        return out;
    }
    if (engine_identity(engine, frozen_runtime_home) != matched_engine_identity)
        throw PlanFailed("native v3 engine or runtime changed during YAML inspection");
    lap("runtime_identity");
    const auto& settings = native_inspection.settings;
    const Resolved native_resolved = resolve(in.options, facts, &settings);
    const auto cache = named_cache(in, native_resolved, facts, engine, helper_binary, run_dir, &snapshot);
    std::optional<CarriedMeasurement> carried;
    PlanReuse reuse;
    reuse.outcome = PlanReuseOutcome::absent;
    if (cache && !cache->index.empty()) {
        reuse = lookup_stored_manifest(cache->index, cache->key, cache->inputs);
        lap("store_lookup");
        if (reuse.outcome == PlanReuseOutcome::hit) {
            const fs::path saved = reuse.manifest;
            // A stored manifest embeds the paths of the runtime home that wrote it, and replay runs exactly those
            // bytes. The same release (same engine identity, already computed above for this start) installed
            // elsewhere cannot replay them, and the home they name may be gone: that is decided here, not found
            // by a refusal. Its measurement is path-free and is sealed again under this build.
            auto loaded = load_carried_measurement(saved);
            const bool elsewhere = loaded.carried && loaded.carried->engine_identity == matched_engine_identity &&
                                   fs::weakly_canonical(loaded.carried->runtime_home) != frozen_runtime_home;
            if (elsewhere) {
                carried = std::move(loaded.carried);
                reuse.outcome = PlanReuseOutcome::carried;
                std::cerr << plan_reuse_line(reuse) << '\n';
            } else {
            EnginePlanInputs replay = in;
            replay.stored_manifest = saved;
            const NativeReplayAdmission admission{snapshot, native_inspection, out.native_launch};
            try {
                EngineStart reused = replay_stored_manifest(replay, run_dir, hooks, &admission);
                reused.reused_store = cache->index;
                lap("replay");
                std::cerr << plan_reuse_line(reuse) << '\n';
                reused.plan_reuse = plan_reuse_json(reuse);
                reused.phase_ms = phases;
                return reused;
            } catch (const PlanFailed& refused) {
                // Nothing was admitted or started yet (power admission is replay's last step). The stored
                // manifest is bound to the build that wrote it; its measurement is not: seal that again under
                // this build, or measure when it cannot be carried.
                reuse.outcome = PlanReuseOutcome::live_admission_refused;
                reuse.detail = refused.what();
                carried = std::move(loaded.carried);
                if (carried) {
                    std::cerr << "bc-serve: stored plan not replayable (live admission refused: " << refused.what()
                              << "); reusing its measurement under this build\n";
                } else {
                    reuse.outcome = PlanReuseOutcome::carried_plan_unusable;
                    reuse.detail = loaded.reason + " (replay refused: " + refused.what() + ")";
                    std::cerr << plan_reuse_line(reuse) << '\n';
                }
            }
            }
        } else {
            std::cerr << plan_reuse_line(reuse) << '\n';
        }
    } else {
        lap("store_lookup");
    }
    out.port = settings.port;

    // The authored snapshot is sealed. An internally derived pool changes only the
    // typed manifest and attempt key. The engine still checks explicitly authored pools.
    if (!hooks.admit_power || !wait)
        throw PlanFailed("native v3 measurement requires durable power admission and the owner stop gate");
    const Json initial_key = plan_key_inputs(
        native_resolved.values, facts, container_identity(v.at("engine").at("model").get<std::string>()),
        matched_engine_identity, "budgets: default", &snapshot);
    const PinnedManifest writer_input(
        Json{{"snapshot", snapshot.manifest_field()}, {"inspection", native_inspection.receipt},
             {"config", {{"name", in.named_config}, {"yaml_path", in.options.inspected->yaml_path.string()}}}}.dump());
    const std::string handoff = processes.empty() ? std::string{} : "/r3-serve-" + new_instance_id();
    std::string helper_identity;
    if (!handoff.empty()) {
        if (!hooks.selected_segment)
            throw PlanFailed("native v3 helpers require an owner-held segment");
        const fs::path expected_helper = frozen_runtime_home / "helpers" / "r3_moe_worker";
        if (fs::canonical(helper_binary) != fs::canonical(expected_helper))
            throw PlanFailed("native v3 helpers must use the worker that the measured writer verifies");
        helper_identity = complete_file_identity(helper_binary);
        require_callback(hooks.selected_segment.invoke("session.hooks.selected_segment", handoff),
                         "native v3 measuring segment");
    }
    Options round_options = in.options;
    const auto initial_pool = v.at("engine").at("host_pool_mib").get<int64_t>();
    std::set<int64_t> measured_pools{initial_pool};
    if (carried) {
        // The carried plan was measured with its own host pool; it must still fit this configuration and
        // these ranks, or there is nothing to carry and this start measures.
        try {
            if (carried->host_pool_mib != initial_pool)
                round_options.sets.push_back("memory.host_pool_mib=" + std::to_string(carried->host_pool_mib));
            const Resolved carried_resolved = resolve(round_options, detect(round_options.sys_root), &settings);
            check_plan_for_configuration(carried_resolved.values,
                                         decode_engine_plan(carried->plan_json, "carried plan"));
            (void)decode_engine_rank_demand(carried->rank_demand, selected_bdfs, "carried rank demand");
            measured_pools.insert(carried->host_pool_mib);
        } catch (const std::exception& unusable) {
            reuse.outcome = PlanReuseOutcome::carried_plan_unusable;
            reuse.detail = unusable.what();
            std::cerr << plan_reuse_line(reuse) << '\n';
            carried.reset();
            round_options = in.options;
        }
    }
    // A carried measurement that fails after this point drops the stored entry: the next start measures.
    try {
    for (int round = 0; round <= kMaxHostPoolChanges; ++round) {
        const Json round_facts = detect(round_options.sys_root);
        const Resolved round_resolved = resolve(round_options, round_facts, &settings);
        const auto& round_values = round_resolved.values;
        if (round_resolved.manifest.at("process_helpers") != processes)
            throw PlanFailed("native v3 helper selection changed before measurement");
        const Json frozen_key = plan_key_inputs(
            round_values, round_facts, initial_key.at("model").get<std::string>(),
            engine_identity(engine, frozen_runtime_home), "budgets: default", &snapshot);
        Json static_key = frozen_key;
        static_key["engine"]["host_pool_mib"] = initial_pool;
        if (static_key != initial_key)
            throw PlanFailed("native v3 static plan inputs changed before measurement: " + key_differences(initial_key, static_key));

        const std::string attempt_id = new_instance_id();
        const fs::path draft = run_dir / ("manifest-native-draft-" + attempt_id + ".json");
        std::vector<std::string> argv = {
            in.self.string(), "--write-manifest", draft.string(), "--power-profile", in.power_profile,
            "--unmeasured", "--expected-plan-key", plan_key(frozen_key), "--engine-binary", engine.string(),
            "--plan-extra", "budgets: default", "--bc-native-config-fd",
            std::to_string(writer_input.descriptor())};
        auto resolver = resolver_args(round_options);
        if (!handoff.empty())
            resolver.insert(resolver.end(), {"--set", "handoff.shm=" + handoff});
        argv.insert(argv.end(), resolver.begin(), resolver.end());
        if (!handoff.empty() && !round_values.at("handoff").value("histogram", std::string{}).empty()) {
            const fs::path histogram = fs::absolute(run_dir) / ("histogram-measuring-" + attempt_id);
            argv.insert(argv.end(), {"--set", "handoff.histogram=" + Json(histogram.string()).dump()});
        }
        const ManifestRun writer = run_planning_child(
            argv, child_environment(environ, {}), hooks, ChildKind::manifest_writer,
            "native v3 measuring draft writer", writer_input.descriptor());
        if (writer.exit != 0)
            throw PlanFailed("native v3 draft writer failed: " + tail_text(writer.err));
        const auto verified = pin_manifest_file(draft, v2::PlanUse::measuring);
        const auto native_view = v2::native_config_view(verified.parsed);
        const auto emitted_devices = verified.parsed.spec.processes.engine.device.gpus;
        std::error_code remove_error;
        fs::remove(draft, remove_error);
        if (remove_error || !native_view || native_view->yaml_utf8 != snapshot.yaml_utf8() ||
            native_view->identity.sha256 != snapshot.sha256() ||
            native_view->identity.resolution_root != snapshot.resolution_root() ||
            emitted_devices != selected_bdfs || verified.parsed.spec.plan.engine_memory)
            throw PlanFailed("native v3 draft did not preserve the admitted snapshot, root, rank order and measuring role");
        if (verified.parsed.spec.resources.host.pool_mib !=
            round_values.at("engine").at("host_pool_mib").get<int64_t>())
            throw PlanFailed("native v3 draft host pool differs from the exact measurement attempt");
        if (!native_settings_match(settings, verified.parsed.spec))
            throw PlanFailed("native v3 draft differs from the matched parser's engine settings");

        const fs::path measuring_execution = verified.pinned->path();
        const Json current_facts = detect(in.options.sys_root);
        if (const auto refusal = resources::preflight_host_pool(verified.parsed, current_facts))
            throw PlanFailed("native resources: " + refusal->message);
        if (round == 0)
            require_callback(hooks.admit_power.invoke("session.hooks.admit_power", measuring_execution,
                                                       v2::PlanUse::measuring),
                             "native v3 pre-measure power admission");

        const fs::path attempt_dir = fs::absolute(run_dir / "measurement-attempts" / attempt_id);
        fs::create_directories(attempt_dir);
        const fs::path plan_path = attempt_dir / "plan.json";
        const fs::path ranks_path = attempt_dir / "rank-demand.json";
        const fs::path measurement_log = attempt_dir / "engine-measuring.log";
        const fs::path absolute_engine = fs::absolute(engine);
        Json measured_plan, rank_demand;
        uint64_t measured_helpers_mib = 0;
        if (carried) {
            // The stored measurement is sealed again under this build: no helper or measuring engine runs.
            measured_plan = carried->plan_json;
            rank_demand = carried->rank_demand;
            measured_helpers_mib = carried->resident.helpers_mib;
            write_atomic(attempt_dir / "carried-from.json", carried->provenance().dump(2) + "\n");
        } else {
        const auto measuring_helpers = helper_starts(
            {processes, current_facts}, frozen_runtime_home, helper_binary, measuring_execution, true);
        const auto baseline = measure_helper_baseline(processes, in);
        MeasuringHelpers group(hooks);
        launch_measure_helpers(measuring_helpers, run_dir, hooks, &group.pids, verified.pinned->descriptor());
        measured_helpers_mib = helper_resident_mib(group.pids);
        require_segment(handoff);
        launch.root.require_matches(snapshot.resolution_root());
        const pid_t measuring_pid = spawn(
            {absolute_engine.string(), "--manifest", measuring_execution.string(), "--role", "engine",
             "--plan-memory", plan_path.string(), "--plan-memory-ranks", ranks_path.string()},
            launch.environment, measurement_log,
            [&](pid_t pid) {
                if (hooks.started)
                    require_callback(hooks.started.invoke("session.hooks.started", ChildKind::measuring_engine,
                                                          "native v3 measuring engine", pid, measurement_log),
                                     "native v3 measuring engine registration");
            },
            verified.pinned->descriptor(), launch.root.descriptor());
        const int measuring_status = wait_measuring_engine(measuring_pid, group, hooks);
        if (!WIFEXITED(measuring_status) || WEXITSTATUS(measuring_status) != 0 ||
            !fs::is_regular_file(plan_path) || !fs::is_regular_file(ranks_path) ||
            fs::file_size(plan_path) == 0 || fs::file_size(ranks_path) == 0 ||
            fs::file_size(plan_path) > v2::kMaxDocumentBytes || fs::file_size(ranks_path) > v2::kMaxDocumentBytes)
            throw PlanFailed("native v3 measuring engine failed or produced incomplete plan/rank output; see " +
                             measurement_log.string());
        release_measuring_helpers(group, baseline, in, hooks);
        if (!handoff.empty() && complete_file_identity(helper_binary) != helper_identity)
            throw PlanFailed("native v3 helper worker changed during measurement");

        measured_plan = parse_plan_json(read_file(plan_path), plan_path.string());
        rank_demand = parse_plan_json(read_file(ranks_path), ranks_path.string());
        }
        const auto decoded_plan = decode_engine_plan(measured_plan, plan_path.string());
        const EngineRankDemand decoded_ranks =
            decode_engine_rank_demand(rank_demand, selected_bdfs, ranks_path.string());
        constexpr uint64_t kMiB = uint64_t{1} << 20;
        const uint64_t engine_resident_mib =
            std::max<uint64_t>(1, (static_cast<uint64_t>(decoded_ranks.resident_other_bytes) + kMiB - 1) / kMiB);
        if (!carried) {
            static const std::regex resident_line(
                R"(--plan-memory: wrote .*; engine start-up took ([0-9]+) MiB of MemAvailable;)" );
            std::vector<uint64_t> logged_resident_mib;
            const std::string log_text = read_file(measurement_log);
            for (auto match = std::sregex_iterator(log_text.begin(), log_text.end(), resident_line);
                 match != std::sregex_iterator(); ++match)
                logged_resident_mib.push_back(std::stoull((*match)[1].str()));
            if (logged_resident_mib.size() != 1 || logged_resident_mib.front() != engine_resident_mib)
                throw PlanFailed("native v3 rank sidecar and rad0011 engine resident-memory log disagree; see " +
                                 measurement_log.string());
        }
        check_plan_for_configuration(round_values, decoded_plan);
        if (decoded_plan.vram.cards.size() != selected_bdfs.size() || decoded_ranks.gpu.size() != selected_bdfs.size())
            throw PlanFailed("native v3 measurement rank count differs from the selected ordered BDFs");
        for (size_t i = 0; i < selected_bdfs.size(); ++i)
            if (decoded_plan.vram.cards[i].free != decoded_ranks.gpu[i].claimable_bytes)
                throw PlanFailed("native v3 measured plan and rank receipt disagree on same-rank free VRAM");

        launch.root.require_matches(snapshot.resolution_root());
        const Json after_facts = detect(in.options.sys_root);
        const Resolved after_resolve = resolve(round_options, after_facts, &settings);
        const fs::path after_home = after_resolve.values.at("engine").at("radiance_home").get<std::string>();
        if (fs::canonical(after_home) != frozen_runtime_home)
            throw PlanFailed("native v3 runtime home changed during measurement; refusing to publish its result");
        const Json after_key = plan_key_inputs(
            after_resolve.values, after_facts,
            container_identity(after_resolve.values.at("engine").at("model").get<std::string>()),
            engine_identity(engine, frozen_runtime_home), "budgets: default", &snapshot);
        if (plan_key(after_key) != plan_key(frozen_key))
            throw PlanFailed("native v3 plan inputs, native snapshot, model or engine build changed during "
                             "measurement; refusing to publish or re-key the result: " +
                             key_differences(frozen_key, after_key));

        const fs::path owner_cache = attempt_dir / "plan-cache";
        const CachedPlan owned = plan_cache_store_owned(
            owner_cache, frozen_key, measured_plan, rank_demand, round_values,
            HostResident{measured_helpers_mib, engine_resident_mib},
            OwnerMeasurementEvidence{attempt_id, verified.sha256});
        if (!owned.owner || owned.owner->attempt_id != attempt_id || owned.key != plan_key(frozen_key))
            throw PlanFailed("native v3 owner cache publication did not return the exact attempt receipt");

        const fs::path measured_manifest = fs::absolute(run_dir) / ("manifest-native-" + attempt_id + ".json");
        const fs::path s8_source = measured_manifest.string() + ".s8-evidence.json";
        std::vector<std::string> measured_argv = {
            in.self.string(), "--write-manifest", measured_manifest.string(), "--power-profile", in.power_profile,
            "--plan-cache", owner_cache.string(), "--engine-binary", absolute_engine.string(), "--plan-extra",
            "budgets: default", "--expected-plan-key", plan_key(frozen_key), "--expected-owner-attempt", attempt_id,
            "--bc-native-config-fd", std::to_string(writer_input.descriptor())};
        if (cache)
            measured_argv.insert(measured_argv.end(), {"--s8-replay-evidence", s8_source.string()});
        measured_argv.insert(measured_argv.end(), resolver.begin(), resolver.end());
        const ManifestRun final_writer = run_planning_child(
            measured_argv, child_environment(environ, {}), hooks, ChildKind::manifest_writer,
            "native v3 measured writer", writer_input.descriptor());
        if (final_writer.exit == 11) {
            const int64_t pool = requested_host_pool(final_writer);
            if (round == kMaxHostPoolChanges || !measured_pools.insert(pool).second)
                throw PlanFailed("native v3 host pool did not converge (last " + std::to_string(pool) + " MiB)");
            if (!handoff.empty()) {
                const int fd = ::shm_open(handoff.c_str(), O_RDONLY | O_CLOEXEC, 0);
                if (fd >= 0) {
                    ::close(fd);
                    throw PlanFailed("native v3 helpers did not retire the measuring transport before retry");
                }
                if (errno != ENOENT)
                    throw PlanFailed("native v3 cannot verify measuring transport retirement: " + errno_text());
            }
            round_options.sets = in.options.sets;
            round_options.sets.push_back("memory.host_pool_mib=" + std::to_string(pool));
            std::cerr << "plan: native v3 measured inputs need a " << pool
                      << " MiB host pool; measuring with it\n";
            continue;
        }
        if (final_writer.exit != 0)
            throw PlanFailed("native v3 measured writer refused its owner receipt (exit " +
                             std::to_string(final_writer.exit) + "): " + tail_text(final_writer.err));
        auto final_snapshot = pin_manifest_file(measured_manifest, v2::PlanUse::serving);
        const auto final_native = v2::native_config_view(final_snapshot.parsed);
        const auto& final_spec = final_snapshot.parsed.spec;
        if (!final_native || final_native->yaml_utf8 != snapshot.yaml_utf8() ||
            final_native->identity.sha256 != snapshot.sha256() ||
            final_native->identity.resolution_root != snapshot.resolution_root())
            throw PlanFailed("native v3 measured serving artifact changed the admitted YAML snapshot identity");
        if (final_spec.intent.model != v.at("engine").at("model").get<std::string>() ||
            final_spec.processes.engine.device.gpus != selected_bdfs)
            throw PlanFailed("native v3 measured serving artifact changed the frozen model or ordered engine devices");
        const auto& draft_host = verified.parsed.spec.resources.host;
        const auto& final_host = final_spec.resources.host;
        // The measuring manifest's host KV tier is a capacity ceiling,
        // not the final number of retained-token bytes. S8 derives that
        // smaller, exact tier from the owned engine KV measurement. The
        // reserve and pool are unchanged; an increased tier still refuses.
        if (!in.quality_score) {
            if (const auto refusal = v2::require_measured_host_tiers(draft_host, final_host))
                throw PlanFailed("native v3 measured host tiers changed: " + refusal->path + ": " +
                                 refusal->message);
        }
        check_native_helper_roles(final_snapshot.parsed, processes, selected_bdfs, handoff);
        // The engine's rank receipt is the only authority for an auto
        // n-gram tier. Seal its exact RAM/disk/VRAM result; a final
        // artifact must not carry the auto intent through to execution.
        if (settings.ngram_placement == "auto" &&
            (!decoded_ranks.ngram_placement ||
             *decoded_ranks.ngram_placement != v2::kNgramPlacementNames[
                 static_cast<size_t>(final_spec.processes.engine.execution.ngram_placement)]))
            throw PlanFailed("native v3 final n-gram placement differs from measured engine rank receipt");
        if (!native_settings_match(settings, final_spec))
            throw PlanFailed("native v3 measured serving artifact differs from the matched parser's engine settings");
        const fs::path final_home = final_spec.processes.engine.runtime.radiance_home;
        if (fs::canonical(final_home) != frozen_runtime_home ||
            engine_identity(engine, frozen_runtime_home) != frozen_key.at("engine_build").get<std::string>() ||
            (!handoff.empty() && complete_file_identity(helper_binary) != helper_identity))
            throw PlanFailed("native v3 measured serving artifact changed the frozen runtime home or engine build");
        if (!final_spec.plan.engine_memory)
            throw PlanFailed("native v3 measured serving artifact has no engine memory receipt");
        // The manifest cache_key is the serving-contract key recomputed by read_manifest(). It is
        // intentionally distinct from the operational owner-cache key used for this measurement.
        if (v2::encode(final_spec.plan.engine_memory->expected) != owned.plan_json ||
            final_spec.plan.engine_memory->model_identity != frozen_key.at("model").get<std::string>() ||
            final_spec.plan.engine_memory->engine_identity != frozen_key.at("engine_build").get<std::string>() ||
            !final_spec.plan.resource_inputs ||
            final_spec.plan.resource_inputs->resident_other_bytes != decoded_ranks.resident_other_bytes)
            throw PlanFailed("native v3 measured serving artifact changed the owner plan or frozen model/build identities");
        if (cache) {
            const auto current =
                named_cache(in, resolve(in.options, after_facts, &settings), after_facts, engine, helper_binary,
                                run_dir, &snapshot);
            if (!current || current->key != cache->key)
                throw PlanFailed("native v3 static inputs changed during measurement; refusing named cache publication");
            if (!fs::is_regular_file(s8_source) || fs::is_symlink(s8_source))
                throw PlanFailed("native v3 measured writer did not publish ordinary S8 replay evidence");
            StoredManifestIndex entry{
                cache->key, measured_manifest.filename().string(), final_snapshot.file_identity, cache->model_digest,
                complete_file_identity(s8_source),
                StoredManifestIndex::NativeMeasurement{attempt_id, owned.owner_sha256}};
            entry.input_digests = cache->inputs;
            write_stored_index(measured_manifest.string() + ".replay-index.json", entry);
            if (!cache->index.empty()) {
                entry.directory = fs::absolute(measured_manifest.parent_path()).string();
                write_stored_index(cache->index, entry);
            }
        }
        out.manifest = measured_manifest;
        out.stored_sha256 = std::move(final_snapshot.sha256);
        out.pinned_manifest = std::move(final_snapshot.pinned);
        out.argv = {absolute_engine.string(), "--manifest", out.execution_manifest().string(), "--role", "engine"};
        out.helpers = helper_starts({processes, after_facts}, frozen_runtime_home, helper_binary,
                                    out.execution_manifest(), false);
        out.native_helper_resident_mib = measured_helpers_mib;
        out.handoff_shm = handoff;
        out.summary["plan_measurement_attempt"] = attempt_id;
        if (carried)
            out.summary["plan_carried_from"] = carried->provenance();
        out.summary["manifest_digest"] = final_snapshot.parsed.metadata.content_digest;
        lap("measure");
        out.plan_reuse = plan_reuse_json(reuse);
        out.phase_ms = phases;
        return out;
    }
    throw PlanFailed("native v3 host pool did not converge");
    } catch (...) {
        if (carried && cache && !cache->index.empty()) {
            std::error_code ignored;
            fs::remove(cache->index, ignored);
        }
        throw;
    }
}

static EngineStart replay_stored_manifest(const EnginePlanInputs& given, const fs::path& run_dir,
                                          const Hooks& hooks, const NativeReplayAdmission* expected_native) {
    if (!hooks.admit_power || !hooks.wait)
        throw PlanFailed("replay: missing required durable power admission or owner stop gate");
    if (given.stored_manifest.empty())
        throw PlanFailed("replay: no stored manifest file");
    const fs::path stored = fs::absolute(given.stored_manifest);
    if (!fs::is_regular_file(stored) || fs::file_size(stored) > 16ULL * 1024ULL * 1024ULL)
        throw PlanFailed("replay: manifest must be an existing bounded regular file");
    const std::string bytes = read_manifest_file(stored);
    const Json source = Json::parse(bytes, nullptr, false);
    const auto verified = v2::read_manifest(bytes, stored.string(), v2::PlanUse::serving);
    if (!verified)
        throw PlanFailed(v2::render_human("bc-serve replay", verified.error()));
    const auto& m = verified.value();
    const auto native = v2::native_config_view(m);
    if (!native)
        throw PlanFailed("replay: the stored manifest has no admitted native YAML snapshot");
    const std::optional<NativeConfigSnapshot> native_snapshot = NativeConfigSnapshot::admitted(
        std::string(native->yaml_utf8), std::string(native->identity.sha256),
        std::string(native->identity.resolution_root));
    if (expected_native && expected_native->snapshot.manifest_field() != native_snapshot->manifest_field())
        throw PlanFailed("replay: stored native snapshot differs from the named start");
    if (given.native_admission &&
        fs::canonical(given.native_admission->resolution_root) != native_snapshot->resolution_root())
        throw PlanFailed("replay: requested resolution root differs from the stored native snapshot");
    const fs::path home = m.spec.processes.engine.runtime.radiance_home;
    const fs::path native_runtime_home = fs::canonical(home);
    const fs::path engine =
        given.engine_binary.empty() ? home.parent_path() / "bin" / "radiance" : given.engine_binary;
    const fs::path helper_binary =
        given.helper_binary.empty() ? home / "helpers" / "r3_moe_worker" : given.helper_binary;
    if (::access(engine.c_str(), X_OK) != 0 ||
        (!m.spec.processes.helpers.empty() && ::access(helper_binary.c_str(), X_OK) != 0))
        throw PlanFailed("replay: matching engine/helper binaries must be executable");
    if (!m.spec.processes.helpers.empty() &&
        fs::canonical(helper_binary) != fs::canonical(native_runtime_home / "helpers" / "r3_moe_worker"))
        throw PlanFailed("replay: native v3 helpers must use the worker that the measured writer verifies");
    const fs::path own_home = given.self.parent_path().parent_path();
    const fs::path own_engine = own_home.parent_path() / "bin" / "radiance";
    if (given.self.parent_path().filename() == "bin" && own_home.filename() == "radiance_home" &&
        fs::exists(own_engine) &&
        (fs::weakly_canonical(home) != fs::weakly_canonical(own_home) ||
         fs::weakly_canonical(engine) != fs::weakly_canonical(own_engine) ||
         (!m.spec.processes.helpers.empty() &&
          fs::weakly_canonical(helper_binary) != fs::weakly_canonical(own_home / "helpers" / "r3_moe_worker"))))
        throw PlanFailed("replay: the engine, helper and runtime home must belong to this bc-serve build");

    EnginePlanInputs in = given;
    in.options.run_dir = run_dir;
    if (in.options.model_root.empty())
        in.options.model_root = native_snapshot->resolution_root();
    const std::string name = m.spec.intent.layout;
    if (in.explicit_power_profile && in.power_profile != m.spec.intent.power.profile)
        throw ReplayFailed(ReplayRefusal::execution_projection,
                           "replay: explicitly requested power profile differs from the stored manifest");
    in.power_profile = m.spec.intent.power.profile;
    in.named_config = name;
    if (!in.options.inspected)   // replay reads the admitted bytes, not a file on disk
        in.options.inspected = inspect_config_snapshot(
            engine, *native_snapshot, name, fs::path(name + ".yaml"), native_runtime_home, in.options.sys_root, hooks,
            given.native_admission ? given.native_admission->runtime_overrides : Env{});
    std::string source_name;
    ReplayEvidence evidence;
    std::optional<NamedCache> attestation;
    std::optional<Resolved> current_config;
    std::shared_ptr<const NativeEngineLaunch> native_launch;
    if (!in.options.inspected)
        throw PlanFailed("replay: the admitted YAML was not read");
    in.options.configs.clear();   // the admitted YAML bytes are the one source
    const auto has_set = [&](const std::string& key) {
        return std::any_of(in.options.sets.begin(), in.options.sets.end(),
                           [&](const std::string& setting) { return setting.starts_with(key + "="); });
    };
    if (!has_set("layout.name"))
        in.options.sets.push_back("layout.name=" + Json(name).dump());
    if (m.spec.handoff) {
        const std::string shm = m.spec.handoff->transport.shm;
        if (shm.rfind("/r3-serve-", 0) != 0 || shm.size() > 220U)
            throw PlanFailed("replay: stored helper shared-memory name is not owner-scoped");
    }
    // The original operator/config/variant inputs must be reconstructed
    // identically on first write and replay. A prior run's transport and
    // S8 allocations are authenticated artifacts, never operator --sets.
    const Json fresh = detect(in.options.sys_root);
    Resolved resolved = resolve(in.options, fresh);
    evidence.engine_identity = engine_identity(engine, native_runtime_home);
    if (evidence.engine_identity != m.spec.plan.engine_memory->engine_identity)
        throw ReplayFailed(ReplayRefusal::build,
                           "replay: engine, kernels or helper build changed");
    const auto& e = resolved.values.at("engine");
    if (fs::canonical(e.at("radiance_home").get<std::string>()) != native_runtime_home)
        throw PlanFailed("replay: native runtime home differs from the current configuration");
    const auto bdfs = resolved.values.at("layout").at("engine_cards").get<std::vector<std::string>>();
    const auto selected = engine_visibility(native_runtime_home, bdfs, fresh);
    native_launch = expected_native ? expected_native->launch : prepare_native_launch(
        native_snapshot->resolution_root(), native_runtime_home, selected,
        given.native_admission ? given.native_admission->runtime_overrides : Env{}, in.options.sys_root);
    native_launch->root.require_matches(native_snapshot->resolution_root());
    for (const auto& [key, value] : selected)
        if (native_launch->environment.at(key) != value)
            throw PlanFailed("replay: ordered native rank visibility changed during admission");
    const auto report = expected_native ? expected_native->inspection : inspect_native_config(
        engine, *native_snapshot, native_launch->environment, hooks, native_launch->root.descriptor());
    reconcile_native_config(report, *native_snapshot, native_runtime_home, e.at("model").get<std::string>(),
                            bdfs, e.at("tp").get<int>());
    if (!native_settings_match(report.settings, m.spec))
        throw PlanFailed("replay: native parser settings differ from the stored engine inputs");
    resolved = resolve(in.options, fresh, &report.settings);
    const Json definitions = load_layouts(in.options);
    source_name = definitions.at("source").value(name, "");
    if (definitions.at("kind").value(name, "") != "config")
        throw PlanFailed(
            "replay: only a current RSC12 named config is eligible for stored-manifest replay");

    const std::string expected_profile = "production";
    const std::string stored_profile =
        source.at("spec").at("intent").at("service").at("profile").get<std::string>();
    if (expected_profile != stored_profile)
        throw PlanFailed("replay: only a production manifest can be replayed");
    ManifestInputs draft;
    draft.resolved = resolved;
    draft.native_config = native_snapshot;
    draft.profile = expected_profile;
    draft.instance_id = new_instance_id();
    draft.producer_sha256 = file_sha256(given.self);
    draft.calibration_cache = in.options.calibration_cache;
    draft.s10.power_profile = m.spec.intent.power.profile;
    draft.s10.now_unix_s = static_cast<int64_t>(std::time(nullptr));
    draft.s10.calibration_records = Json::array();
    if (source.at("spec").contains("experiment"))
        throw PlanFailed(
            "replay: lab experiment requires RSC12 static input-key verification, not available here");
    const BuiltManifest draft_manifest = build_manifest(draft);

    evidence.producer_sha256 = draft.producer_sha256;
    evidence.model_identity = container_identity(m.spec.intent.model);
    evidence.fresh_facts = fresh;
    evidence.effective_spec = draft_manifest.document.at("spec");
    attestation = named_cache(in, resolved, fresh, engine, helper_binary, stored.parent_path(),
                              &*native_snapshot);
    current_config = resolved;
    const Json& facts = evidence.fresh_facts;
    const ReplayResult checked = authenticate_replay(bytes, stored.string(), evidence,
                                                     CalibrationVerification::require_fresh_measured_s8);
    if (const auto* error = std::get_if<ReplayError>(&checked))
        throw ReplayFailed(error->reason, v2::render_human("bc-serve replay", error->detail));
    const auto& accepted = std::get<v2::Manifest>(checked);
    std::optional<StoredManifestIndex> bound;
    if (attestation) {
        const fs::path sidecar = stored.string() + ".replay-index.json";
        if (!fs::is_regular_file(sidecar) || fs::is_symlink(sidecar))
            throw ReplayFailed(
                ReplayRefusal::missing_gate,
                "replay: the original manifest has no durable RSC12 static identity attestation");
        try {
            bound = read_stored_index(sidecar);
        } catch (const std::exception& e) {
            throw ReplayFailed(ReplayRefusal::bytes_or_digest,
                               std::string("replay: stored attestation is corrupt: ") + e.what());
        }
        if (bound->manifest_filename != stored.filename().string() ||
            bound->digest != complete_file_identity_bytes(bytes))
            throw ReplayFailed(ReplayRefusal::bytes_or_digest,
                               "replay: stored attestation no longer binds the original manifest bytes");
        if (bound->model_digest.empty() || bound->model_digest != attestation->model_digest)
            throw ReplayFailed(ReplayRefusal::model,
                               "replay: complete model artifact bytes no longer match the original");
        if (bound->key != attestation->key)
            throw ReplayFailed(
                ReplayRefusal::execution_projection,
                "replay: RSC12 static resolution key changed (config, inputs, calibration or power)");
    }
    std::string native_helper_build;
    uint64_t native_helper_resident_mib = 0;
    if (!bound || !bound->native_measurement || !current_config)
        throw ReplayFailed(ReplayRefusal::missing_gate,
                           "replay: native v3 requires the manifest-bound owner measurement");
    const auto& receipt = *bound->native_measurement;
    const auto& current = current_config->values;
    check_native_helper_roles(accepted, current_config->manifest.at("process_helpers"),
                              current.at("layout").at("engine_cards").get<std::vector<std::string>>(),
                              accepted.spec.handoff ? accepted.spec.handoff->transport.shm : std::string{});
    // The attestation binds the original static selection. The owner receipt key
    // instead binds the exact pool whose measured result the manifest embeds.
    Json measured_values = current;
    measured_values["engine"]["host_pool_mib"] = accepted.spec.resources.host.pool_mib;
    const Json key_inputs = plan_key_inputs(measured_values, facts, evidence.model_identity,
                                            evidence.engine_identity, "budgets: default", &*native_snapshot);
    native_helper_build = key_inputs.at("native_config").value("helper_build", std::string{});
    const auto owner_dir = stored.parent_path() / "measurement-attempts" / receipt.attempt_id / "plan-cache";
    const auto cached = plan_cache_lookup(owner_dir, key_inputs);
    if (!cached || !cached->owner || cached->owner->attempt_id != receipt.attempt_id ||
        cached->owner_sha256 != receipt.sha256 ||
        v2::encode(accepted.spec.plan.engine_memory->expected) != cached->plan_json ||
        (cached->resident.helpers_mib > 0) != !accepted.spec.processes.helpers.empty() ||
        !accepted.spec.plan.resource_inputs)
        throw ReplayFailed(ReplayRefusal::bytes_or_digest,
                           "replay: native owner measurement no longer matches the exact final manifest");
    native_helper_resident_mib = cached->resident.helpers_mib;
    const auto ranks = decode_engine_rank_demand(cached->rank_demand, accepted.spec.processes.engine.device.gpus,
                                                 "stored native owner rank demand");
    if (accepted.spec.plan.resource_inputs->resident_other_bytes != ranks.resident_other_bytes ||
        cached->plan.vram.cards.size() != ranks.gpu.size())
        throw ReplayFailed(ReplayRefusal::bytes_or_digest,
                           "replay: native owner resident or rank facts differ from the final manifest");
    for (size_t i = 0; i < ranks.gpu.size(); ++i)
        if (cached->plan.vram.cards[i].free != ranks.gpu[i].claimable_bytes)
            throw ReplayFailed(ReplayRefusal::bytes_or_digest,
                               "replay: native owner rank facts disagree with the measured plan");

    v2::P1LiveRoom room{};
    if (given.replay_live_room) {
        room = given.replay_live_room.require("session.replay.live_room", accepted, facts);
    } else {
        const auto measured = measure_session_live_room(accepted, facts, in.options.sys_root);
        if (!measured)
            throw PlanFailed(v2::render_human("bc-serve replay room", measured.error()));
        room = measured.value();
    }
    if (const auto error = check_replay_live_room(accepted, facts, room))
        throw PlanFailed(v2::render_human("bc-serve replay room", *error));
    // Production never substitutes the unmeasured RSC12 draft for S8.
    // The original native P1 and raw P2 records were published by the
    // strict measured writer. Requalify them against current scope/time
    // BEFORE pinning/admitting any GPU child.
    if (!current_config || !bound || bound->s8_digest.empty())
        throw ReplayFailed(ReplayRefusal::calibration, "replay: current RSC12 scope or S8 evidence is missing");
    const StoredS8Evidence issued = [&] {
        try {
            return read_stored_s8_evidence(stored.string() + ".s8-evidence.json",
                                          accepted.metadata.content_digest, bound->s8_digest);
        } catch (const std::exception& error) {
            throw ReplayFailed(ReplayRefusal::calibration, error.what());
        }
    }();
    const v2::Status renewed = verify_stored_s8(accepted, *current_config, facts, room, issued, engine,
                                                helper_binary, static_cast<int64_t>(std::time(nullptr)));
    if (renewed)
        throw ReplayFailed(ReplayRefusal::calibration, v2::render_human("bc-serve replay S8", *renewed));

    // Preserve the operator's original file as immutable input evidence;
    // hand every child only the authenticated sealed bytes, never that name.
    auto pinned = std::make_shared<PinnedManifest>(bytes);
    const auto pinned_read = v2::load_manifest(pinned->path(), v2::PlanUse::serving);
    if (!pinned_read || pinned_read.value().metadata.content_digest != accepted.metadata.content_digest)
        throw ReplayFailed(ReplayRefusal::bytes_or_digest,
                           "replay: sealed handoff does not match admitted manifest bytes");
    EngineStart out;
    out.manifest = stored;
    out.stored_sha256 = sha256_hex(bytes);
    out.pinned_manifest = std::move(pinned);
    out.port = int(accepted.spec.processes.engine.server.port);
    out.argv = {engine.string(), "--manifest", out.pinned_manifest->path().string(), "--role", "engine"};
    std::string ordinals, hip_ordinals;
    for (size_t i = 0; i < accepted.spec.processes.engine.device.gpus.size(); ++i) {
        const std::string& bdf = accepted.spec.processes.engine.device.gpus[i];
        const Json* gpu = nullptr;
        for (const auto& g : facts.at("gpus"))
            if (g.at("bdf") == bdf)
                gpu = &g;
        if (!gpu)
            throw PlanFailed("replay: engine BDF vanished after live admission: " + bdf);
        ordinals += (i ? "," : "") + std::to_string(gpu->at("hip_ordinal").get<int>());
        hip_ordinals += (i ? "," : "") + std::to_string(i);
    }
    out.runtime = {{"RADIANCE_HOME", home.string()},
                   {"ROCR_VISIBLE_DEVICES", ordinals},
                   {"HIP_VISIBLE_DEVICES", hip_ordinals}};
    out.argv.front() = fs::absolute(engine).string();
    out.native_launch = std::move(native_launch);
    out.native_launch->root.require_matches(native_snapshot->resolution_root());
    out.native_helper_resident_mib = native_helper_resident_mib;
    if (fs::canonical(home) != native_runtime_home ||
        engine_identity(engine, native_runtime_home) != evidence.engine_identity)
        throw PlanFailed("replay: native parser or build differs from the stored engine inputs");
    if (!native_helper_build.empty() && file_sha256(helper_binary) != native_helper_build)
        throw PlanFailed("replay: native helper build changed during admission");
    out.helpers = helper_starts({current_config->manifest.at("process_helpers"), facts}, native_runtime_home,
                                fs::absolute(helper_binary), out.execution_manifest(), false);
    if (accepted.spec.handoff) {
        out.handoff_shm = accepted.spec.handoff->transport.shm;
        if (!hooks.selected_segment)
            throw PlanFailed("replay: native v3 helpers require an owner-held segment");
        require_callback(hooks.selected_segment.invoke("session.hooks.selected_segment", out.handoff_shm),
                         "replay handoff namespace");
    }
    out.summary = {{"config", name},
                   {"source", source_name},
                   {"model", accepted.spec.intent.model},
                   {"engine", engine.string()},
                   {"engine_cards", accepted.spec.processes.engine.device.gpus},
                   {"helper_cards", source.at("spec").at("processes").at("helpers")},
                   {"manifest_digest", accepted.metadata.content_digest},
                   {"replay", true}};
    out.summary["plan_measurement_attempt"] = bound->native_measurement->attempt_id;
    // Replay starts no writer or measuring child. The owner admits power
    // durably before the serving engine can be forked.
    require_callback(hooks.admit_power.invoke("session.hooks.admit_power", out.pinned_manifest->path(),
                                              v2::PlanUse::serving),
                     "stored manifest power admission");
    return out;
}

EngineStart replay_stored_manifest(const EnginePlanInputs& given, const fs::path& run_dir,
                                   const Hooks& hooks) {
    return replay_stored_manifest(given, run_dir, hooks, nullptr);
}

} // namespace bc::serving::session
