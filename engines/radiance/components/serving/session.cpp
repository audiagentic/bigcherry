#include "session.h"
#include "power_transaction.h"

#include "calibration_lab.h"
#include "config_loader.h"
#include "live_policy_ingest.h"
#include "manifest_writer.h"
#include "resources.h"
#include "stored_manifest.h"
#include "sha256.h"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/mman.h>
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

// Every injected callback returns a typed result. The legacy plan-step throwing
// API converts errors only after the guarded invocation (never at a raw call).
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

pid_t spawn(const std::vector<std::string>& argv, const Env& env, const fs::path& log,
            const Callback<void(pid_t)>& recorded) {
    if (argv.empty() || argv[0].empty())
        throw std::runtime_error("spawn: an empty command");
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
        if (!gate->wait_in_child())
            ::_exit(126); // never recorded: run nothing
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
            if (s_.power_hook.apply && s_.power_hook.begin_manifest)
                throw PlanFailed("power_hook: choose pre-plan or manifest admission, not both");
            if (bool(s_.power_hook.apply || s_.power_hook.begin_manifest) != bool(s_.power_hook.restore))
                throw PlanFailed("power_hook: begin/apply and restore must both be provided");
            if (s_.power_hook.begin_manifest && (!s_.power_hook.verify_serving || !s_.power_hook.verified_exit))
                throw PlanFailed(
                    "power_hook: manifest admission needs final verification and verified-exit proof");
            // Legacy owner test hook remains compatible. Production uses the
            // manifest-authenticated admission during the planner, before measurement.
            if (s_.power_hook.apply) {
                power_attempted_ = true; // restore even if apply partially writes before throwing
                require_callback(s_.power_hook.apply.invoke("session.power.apply"), "power_hook.apply");
                event("power_applied");
            }
            event("planning");
            Hooks hooks;
            hooks.wait = [this](std::chrono::milliseconds d) { return service(d); };
            hooks.started = [this](ChildKind kind, const std::string& role, pid_t pid, const fs::path& log) {
                // Register first, even on forbidden starts; cleanup must
                // reap a child that never passed its pre-exec gate.
                track(role, pid, log);
                if (kind != ChildKind::manifest_writer && s_.power_hook.begin_manifest && !power_admitted_)
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
                if (name.rfind("/r3-serve-", 0) != 0)
                    throw PlanFailed("refusing to own a shared segment without a unique session name");
                handoff_shm_ = name;
                record_["handoff_shm"] = name; // journal name even if the planner fails after helpers start
                event("segment_selected");
            };
            plan_ = require_callback(s_.plan.invoke("session.plan", s_.run_dir, hooks), "session.plan");
            if (s_.power_hook.begin_manifest && !power_admitted_)
                throw PlanFailed("planner returned without DURABLE authenticated pre-measure power admission");
            if (s_.power_hook.verify_serving)
                require_callback(
                    s_.power_hook.verify_serving.invoke("session.power.verify_serving", plan_.manifest),
                    "power final manifest");
            record_["manifest"] = plan_.manifest.string();
            record_["port"] = plan_.port;
            record_["selection"] = plan_.summary;
            event("planned");
            if (stop_requested())
                throw Stop{};
            // No-process PWR06 floor, before any serving helper OR engine is
            // launched. The measuring children have already been reaped.
            if (s_.power_hook.gfxoff_window)
                require_callback(s_.power_hook.gfxoff_window.invoke("session.gfxoff.floor", false),
                                 "gfxoff floor receipt");
            if (!plan_.stored_sha256.empty() &&
                sha256_hex(read_file(plan_.manifest)) != plan_.stored_sha256)
                throw PlanFailed("replay: stored manifest bytes changed after authentication");
            start_helpers();
            start_engine();
            if (!wait_health())
                throw Stop{};
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
                event(s_.laboratory ? "runtime_guard_preflight" : "runtime_guard_ready");
            }
            if (s_.laboratory) {
                event("calibrating");
                calibrate();
                event("calibration_completed");
            } else {
                event("running");
                while (service(s_.timing.poll)) {
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
            /* calibration terminates with the same verified stop/restore as a
             * serving owner; it does not remain in a detached running state. */
        } catch (const Stop&) {
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
    int lock_ = -1, listen_ = -1;
    bool stop_ = false;
    bool power_attempted_ = false;
    bool power_admitted_ = false;
    std::string handoff_shm_;
    fs::path engine_log_;
    std::vector<fs::path> helper_logs_;
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

    void track(const std::string& role, pid_t pid, const fs::path& log) {
        children_.push_back({role, pid, log});
        Json p = {{"role", role}, {"pid", pid}, {"pgid", pid}, {"log", log.string()}};
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
            const pid_t pid = spawn(h.argv, child_environment(environ, h.runtime), log,
                                    [&](pid_t p) { track(h.role, p, log); });
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

    // Only a measured, authenticated lab manifest can run this workload.
    // The observation-space envelope is finite even if the helper's retained
    // drift ring coarsens repeatedly. Each request runs with an HTTP deadline.
    void calibrate() {
        if (!s_.laboratory)
            throw PlanFailed("calibration requested without lab inputs");
        const auto verified = v2::load_manifest(plan_.manifest, v2::PlanUse::serving);
        if (!verified)
            throw PlanFailed("calibration manifest: " + verified.error().message);
        const auto& spec = verified.value().spec;
        if (spec.intent.service.profile != v2::Profile::kLab || spec.experiment || !spec.handoff ||
            !spec.handoff->placement.live_policy || helper_logs_.empty())
            throw PlanFailed(
                "calibration requires a lab-only live-policy manifest with helper0 and no experiment");
        const auto prompts = read_calibration_prompts(s_.laboratory->prompts_a, s_.laboratory->prompts_b);
        if (!prompts)
            throw PlanFailed("calibration prompts: " + prompts.error().message);
        const auto& workload = prompts.value();
        const auto& policy = *spec.handoff->placement.live_policy;
        const fs::path helper_log = helper_logs_.front();
        const auto begun = Clock::now();
        constexpr auto request_timeout = std::chrono::seconds(120);
        CalibrationEnvelope envelope;
        bool have_feed = false;
        const auto checked_limit = [&]() {
            const auto drift = read_calibration_drift(helper_log);
            if (!drift)
                throw PlanFailed("calibration drift: " + drift.error().message);
            const auto next = calibration_envelope(policy, workload.a.size(), workload.b.size(),
                                                   request_timeout, drift.value());
            if (!next)
                throw PlanFailed("calibration envelope: " + next.error().message);
            envelope = envelope.limit.count() == 0 ? next.value()
                                                   : monotonic_calibration_envelope(envelope, next.value());
            have_feed = have_feed || envelope.accepted_feed;
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - begun);
            if (!have_feed && elapsed >= envelope.prefeed)
                throw PlanFailed("calibration: no measured heat feed within the policy-only bound");
            if (elapsed >= envelope.limit)
                throw PlanFailed("calibration: finite measured workload/drift deadline exhausted");
            if (stop_requested())
                throw Stop{};
            if (const auto gone = exited_child())
                throw PlanFailed("calibration: child exited before convergence: " + *gone);
            if (s_.runtime_watch)
                require_callback(s_.runtime_watch.invoke("session.calibration.runtime_watch"),
                                 "calibration live policy guard");
        };
        for (;;) {
            // Every complete iteration is exactly A -> B -> B -> A; a marker
            // from a partial traversal is deliberately not accepted.
            for (const auto* series : {&workload.a, &workload.b, &workload.b, &workload.a}) {
                for (const auto& prompt : *series) {
                    checked_limit();
                    const Json request = {
                        {"model", "default"}, {"prompt", prompt}, {"max_tokens", 24}, {"temperature", 0}};
                    const auto reply =
                        http(plan_.port, "POST", "/v1/completions", request.dump(), request_timeout);
                    if (!reply || reply->status < 200 || reply->status > 299)
                        throw PlanFailed("calibration: workload HTTP request failed");
                    const Json output = Json::parse(reply->body, nullptr, false);
                    if (!output.is_object() || !output.contains("choices") ||
                        !output.at("choices").is_array() || output.at("choices").empty() ||
                        !output.at("choices").at(0).is_object() ||
                        !output.at("choices").at(0).value("text", Json()).is_string())
                        throw PlanFailed("calibration: workload reply is not a completion");
                    checked_limit();
                    if (!service(0ms))
                        throw Stop{};
                }
            }
            const auto marker = complete_calibration_marker(helper_log);
            if (!marker)
                throw PlanFailed("calibration marker: " + marker.error().message);
            if (marker.value())
                break;
            checked_limit();
        }
        if (!plan_.summary.contains("calibration_store_key") ||
            !plan_.summary.at("calibration_store_key").is_string())
            throw PlanFailed("calibration: measured plan store key is absent");
        if (s_.lab_evidence_gate)
            require_callback(s_.lab_evidence_gate.invoke("session.calibration.prove_live"),
                             "calibration final live evidence gate");
        bc::serving::ingest_live_policy_log(helper_log, s_.laboratory->calibration_store, plan_.manifest,
                                            plan_.summary.at("calibration_store_key").get<std::string>());
    }

    void start_engine() {
        const fs::path log = s_.run_dir / ("engine-" + stamp() + ".log");
        engine_log_ = log;
        plan_.engine_launched_at = std::chrono::system_clock::now();
        (void)spawn(plan_.argv, child_environment(environ, plan_.runtime), log,
                    [&](pid_t pid) { track("engine", pid, log); });
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

ManifestRun run_to_completion(const std::vector<std::string>& argv, const Env& env, const Hooks& hooks) {
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
                hooks.started.require("session.hooks.started", ChildKind::manifest_writer, "manifest writer", q,
                                      out);
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
            throw std::runtime_error("plan: the manifest writer (pid " + std::to_string(pid) +
                                     ") is still alive after SIGKILL");
        notify_ended();
    };
    int st = 0;
    try {
        while (running(pid, &st)) {
            if (wait && !wait.require("session.hooks.wait", 100ms)) {
                finish(stop_child(pid, SIGTERM, 10s));
                throw PlanFailed("plan: stopped while bc-serve wrote a manifest");
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
                std::cerr << "manifest writer: secondary exit notification failed after reap: "
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
    r.out = read_file(out);
    r.err = read_file(err);
    std::error_code ec;
    fs::remove(out, ec);
    fs::remove(err, ec);
    return r;
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
    for (const auto& l : o.layouts)
        a.insert(a.end(), {"--layouts", l.string()});
    for (const auto& c : o.configs)
        a.insert(a.end(), {"--configs", c.string()});
    if (!o.model_root.empty())
        a.insert(a.end(), {"--model-root", o.model_root.string()});
    if (!o.calibration_cache.empty())
        a.insert(a.end(), {"--calibration", o.calibration_cache.string()});
    for (const auto& s : o.sets)
        a.insert(a.end(), {"--set", s});
    return a;
}

std::string tail_text(const std::string& s, size_t n = 2000) {
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
                            const Hooks& hooks, std::vector<pid_t>* pids) {
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
        });
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
        if (stop_child(*it, SIGINT, 60s)) {
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

} // namespace

EngineStart plan_engine_only(const EnginePlanInputs& given, const fs::path& run_dir, const Hooks& hooks) {
    const Wait& wait = hooks.wait;
    // The sources as files, frozen: the manifest writer runs with BIGCHERRY_* removed from its environment, so
    // configs or layouts that came from the environment are passed as arguments (load_layouts' rule: configs
    // replace the layout files unless --layouts is given).
    EnginePlanInputs in = given;
    {
        const auto configs = config_sources(given.options.configs);
        in.options.configs = configs;
        in.options.layouts = configs.empty() || !given.options.layouts.empty() ? layout_sources(given.options)
                                                                               : std::vector<fs::path>{};
    }
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

    const ManifestRunner runner = in.runner ? in.runner : [&hooks](const std::vector<std::string>& argv) {
        return run_to_completion(argv, child_environment(environ, {}), hooks);
    };
    const fs::path cache = run_dir / "plan-cache";
    fs::create_directories(cache);
    const std::string id = stamp();
    const fs::path manifest = run_dir / ("manifest-" + id + ".json");
    // The bash launcher's engine-only plan-extra: the two share one plan cache.
    const std::vector<std::string> base = resolver_args(in.options);
    std::vector<std::string> extra_sets;
    const std::string handoff = processes.empty() ? std::string() : "/r3-serve-" + new_instance_id();
    if (!handoff.empty()) {
        extra_sets.push_back("handoff.shm=" + handoff);
        if (hooks.selected_segment)
            require_callback(hooks.selected_segment.invoke("session.hooks.selected_segment", handoff),
                             "hooks.selected_segment");
    }
    auto write = [&](const fs::path& out_file, const std::vector<std::string>& more) {
        std::vector<std::string> argv = {in.self.string(), "--write-manifest", out_file.string(),
                                         "--power-profile", in.power_profile};
        if (in.calibration_run) {
            if (in.live_policy_file.empty())
                throw PlanFailed("calibration: a lab live policy is required");
            argv.insert(argv.end(), {"--calibration-run", "--live-policy", in.live_policy_file.string()});
        }
        argv.insert(argv.end(), more.begin(), more.end());
        argv.insert(argv.end(), base.begin(), base.end());
        for (const auto& s : extra_sets)
            argv.insert(argv.end(), {"--set", s});
        return require_callback(runner.invoke("session.manifest.runner", argv),
                                "manifest runner callback threw");
    };
    if (!processes.empty()) {
        const fs::path draft = run_dir / ("manifest-resources-" + id + ".json");
        const ManifestRun unmeasured = write(draft, {"--unmeasured"});
        if (unmeasured.exit != 0)
            throw PlanFailed("resources: cannot write owner map from the selected build: " +
                             tail_text(unmeasured.err));
        const Json doc = Json::parse(read_file(draft), nullptr, false);
        if (!doc.is_object() || !doc.contains("spec"))
            throw PlanFailed("resources: missing unmeasured owner-map manifest");
        const auto ledger_result =
            resources::derive_launch({v, processes, facts, doc}, in.options.calibration_cache);
        if (!ledger_result)
            throw PlanFailed("resources: " + ledger_result.error().message);
        const auto& ledger = ledger_result.value();
        extra_sets.push_back("memory.host_pool_mib=" + std::to_string(ledger.host_pool_mib));
        extra_sets.push_back("memory.host_kv_mib=" + std::to_string(ledger.host_kv_mib));
        out.summary["resource_ledger"] = resources::to_json(ledger);
        out.summary["derived_tail_budget_mib"] = v.at("memory").at("tail_budget_mib");
        std::cerr << "resources: derived host pool " << ledger.host_pool_mib << " MiB, host KV "
                  << ledger.host_kv_mib << " MiB (C++ RAD/BDF ledger)\n";
        std::error_code ec;
        fs::remove(draft, ec);
    }
    const std::vector<std::string> measured_args = {"--plan-cache",  cache.string(), "--engine-binary",
                                                    engine.string(), "--plan-extra", "budgets: default"};
    int pool_changes = 0;
    bool power_admitted = false;
    for (int round = 0; round < 5; ++round) {
        std::error_code ec;
        fs::remove(manifest, ec);
        ManifestRun w = write(manifest, measured_args);
        if (w.exit == 10) {
            // No measured plan for this configuration: measure it with the engine alone (it stops after its
            // memory plan).
            std::cerr << "plan: no measured plan for " << name
                      << " (cache miss); measuring it first (about 1-2 minutes)\n";
            const fs::path measuring =
                run_dir / ("manifest-measuring-" + id + "-" + std::to_string(round) + ".json");
            const ManifestRun u = write(measuring, {"--unmeasured"});
            if (u.exit != 0)
                throw PlanFailed("plan: the unmeasured manifest was not written: " + tail_text(u.err));
            const fs::path plan = run_dir / "plan-measured.json", ranks = run_dir / "plan-ranks.json";
            fs::remove(plan, ec);
            fs::remove(ranks, ec);
            if (hooks.admit_power && !power_admitted) {
                require_callback(
                    hooks.admit_power.invoke("session.hooks.admit_power", measuring, v2::PlanUse::measuring),
                    "pre-measure manifest power admission");
                power_admitted = true;
            }
            const auto helper_measuring =
                helper_starts({processes, facts}, home, helper_binary, measuring, true);
            MeasuringHelpers group(hooks);
            {
                launch_measure_helpers(helper_measuring, run_dir, hooks, &group.pids);
                // Exactly what the bash launcher measures: anonymous RSS total plus shared RSS once.
                const uint64_t measured_helpers_mib = helper_resident_mib(group.pids);
                require_segment(handoff);
                const fs::path log = run_dir / ("engine-measure-" + id + "-" + std::to_string(round) + ".log");
                const pid_t pid =
                    spawn({engine.string(), "--manifest", measuring.string(), "--role", "engine",
                           "--plan-memory", plan.string(), "--plan-memory-ranks", ranks.string()},
                          child_environment(environ, out.runtime), log,
                          // registered with the owner before it runs: should its exit not be verified here,
                          // the owner stops it and the session is never reported stopped while it lives
                          [&](pid_t p) {
                              if (hooks.started)
                                  require_callback(hooks.started.invoke("session.hooks.started",
                                                                        ChildKind::measuring_engine,
                                                                        "measuring engine", p, log),
                                                   "hooks.started");
                          });
                int st = 0;
                const auto deadline = Clock::now() + std::chrono::minutes(30);
                bool stopped_early = false, helper_died = false;
                std::string failed_role;
                try {
                    while (running(pid, &st)) {
                        // The measuring engine can take minutes: refuse immediately if a helper
                        // vanished after readiness, not just if it fails before the engine starts.
                        for (size_t i = 0; i < group.pids.size(); ++i)
                            if (!running(group.pids[i])) {
                                stopped_early = helper_died = true;
                                failed_role = "helper" + std::to_string(i);
                                break;
                            }
                        if (stopped_early)
                            break;
                        if (Clock::now() > deadline ||
                            !require_callback(wait.invoke("session.hooks.wait", 200ms), "hooks.wait")) {
                            stopped_early = true;
                            break;
                        }
                    }
                } catch (...) {
                    if (stop_child(pid, SIGINT, 60s) && hooks.ended)
                        require_callback(hooks.ended.invoke("session.hooks.ended", pid), "hooks.ended");
                    throw;
                }
                if (stopped_early) {
                    if (!stop_child(pid, SIGINT, 60s))
                        throw std::runtime_error("plan: the measuring engine (pid " + std::to_string(pid) +
                                                 ") is still alive after SIGKILL");
                    if (hooks.ended)
                        require_callback(hooks.ended.invoke("session.hooks.ended", pid), "hooks.ended");
                    if (helper_died)
                        throw PlanFailed("plan: " + failed_role +
                                         " died during engine memory measurement; refusing this plan");
                    throw PlanFailed("plan: the measuring start was stopped (a stop request or 30 min)");
                }
                // the engine exited by itself; anything it left in its group goes too
                if (!stop_child(pid, SIGKILL, 10s))
                    throw std::runtime_error(
                        "plan: a process of the measuring engine's group is still alive after SIGKILL");
                if (hooks.ended)
                    require_callback(hooks.ended.invoke("session.hooks.ended", pid), "hooks.ended");
                if (!WIFEXITED(st) || WEXITSTATUS(st) != 0 || !fs::exists(plan) || fs::file_size(plan) == 0 ||
                    !fs::exists(ranks) || fs::file_size(ranks) == 0)
                    throw PlanFailed("plan: the measuring start failed; see " + log.string());
                std::smatch m;
                const std::string text = read_file(log);
                static const std::regex took(
                    R"(--plan-memory: wrote .*; engine start-up took ([0-9]+) MiB of MemAvailable;)");
                std::string engine_mib;
                for (auto it = std::sregex_iterator(text.begin(), text.end(), took);
                     it != std::sregex_iterator(); ++it)
                    engine_mib = (*it)[1];
                if (engine_mib.empty())
                    throw PlanFailed(
                        "plan: the measuring start did not report the engine's resident host memory; see " +
                        log.string());
                group.stop();
                std::vector<std::string> stored = measured_args;
                stored.insert(stored.end(), {"--measured-plan", plan.string(), "--rank-demand", ranks.string(),
                                             "--helpers-resident-mib", std::to_string(measured_helpers_mib),
                                             "--engine-resident-mib", engine_mib});
                fs::remove(manifest, ec);
                w = write(manifest, stored);
            }
        }
        if (w.exit == 0) {
            if (hooks.admit_power && !power_admitted) {
                require_callback(
                    hooks.admit_power.invoke("session.hooks.admit_power", manifest, v2::PlanUse::serving),
                    "measured manifest power admission");
                power_admitted = true;
            }
            if (in.calibration_run)
                out.summary["calibration_store_key"] =
                    bc::serving::calibration_key(bc::serving::plan_key_inputs(
                        v, facts,
                        bc::serving::container_identity(v.at("engine").at("model").get<std::string>()),
                        bc::serving::engine_identity(engine, home), "budgets: default"));
            out.manifest = manifest;
            out.argv = {engine.string(), "--manifest", manifest.string(), "--role", "engine"};
            out.helpers = helper_starts({processes, facts}, home, helper_binary, manifest, false);
            out.handoff_shm = handoff;
            std::smatch d;
            static const std::regex digest(R"(digest: ([0-9a-f]+))");
            if (std::regex_search(w.out, d, digest))
                out.summary["manifest_digest"] = d[1].str();
            return out;
        }
        if (w.exit == 11) {
            std::smatch p;
            static const std::regex pool(R"(derived_host_pool_mib=([0-9]+))");
            if (!std::regex_search(w.out, p, pool))
                throw PlanFailed("plan: bc-serve asked for another host pool without naming it: " +
                                 tail_text(w.err));
            if (++pool_changes > 2)
                throw PlanFailed("plan: the host pool did not converge (last " + p[1].str() + " MiB)");
            std::cerr << "plan: the measured inputs need a " << p[1].str()
                      << " MiB host pool; measuring with it\n";
            extra_sets.push_back("memory.host_pool_mib=" + p[1].str());
            continue;
        }
        throw PlanFailed("plan: bc-serve --write-manifest failed (exit " + std::to_string(w.exit) +
                         "): " + tail_text(w.err));
    }
    throw PlanFailed("plan: no measured manifest after 5 rounds");
}

EngineStart replay_stored_manifest(const EnginePlanInputs& given, const fs::path&,
                                   const Hooks& hooks) {
    if (!hooks.admit_power || !hooks.wait)
        throw PlanFailed("replay: missing required durable power admission or owner stop gate");
    if (given.stored_manifest.empty())
        throw PlanFailed("replay: no stored manifest file");
    const fs::path stored = fs::absolute(given.stored_manifest);
    if (!fs::is_regular_file(stored) || fs::file_size(stored) > 16U * 1024U * 1024U)
        throw PlanFailed("replay: manifest must be an existing bounded regular file");
    const std::string bytes = read_file(stored);
    const auto verified = v2::read_manifest(bytes, stored.string(), v2::PlanUse::serving);
    if (!verified)
        throw PlanFailed(v2::render_human("bc-serve replay", verified.error()));
    const auto& m = verified.value();
    const Json source = Json::parse(bytes);
    const fs::path home = m.spec.processes.engine.runtime.radiance_home;
    const fs::path engine = given.engine_binary.empty() ? home.parent_path() / "bin" / "radiance"
                                                         : given.engine_binary;
    const fs::path helper_binary = given.helper_binary.empty() ? home / "helpers" / "r3_moe_worker"
                                                               : given.helper_binary;
    if (::access(engine.c_str(), X_OK) != 0 ||
        (!m.spec.processes.helpers.empty() && ::access(helper_binary.c_str(), X_OK) != 0))
        throw PlanFailed("replay: matching engine/helper binaries must be executable");
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
    in.options.configs = config_sources(given.options.configs);
    if (in.options.configs.empty())
        throw PlanFailed("replay: static RSC12 configuration cannot be checked without --configs or BIGCHERRY_CONFIGS");
    in.options.layouts = given.options.layouts;
    const std::string name = m.spec.intent.layout;
    const auto has_set = [&](const std::string& key) {
        return std::any_of(in.options.sets.begin(), in.options.sets.end(), [&](const std::string& setting) {
            return setting.starts_with(key + "=");
        });
    };
    if (!has_set("layout.name"))
        in.options.sets.push_back("layout.name=" + Json(name).dump());
    if (m.spec.handoff) {
        const std::string shm = m.spec.handoff->transport.shm;
        if (shm.rfind("/r3-serve-", 0) != 0 || shm.size() > 220U)
            throw PlanFailed("replay: stored helper shared-memory name is not owner-scoped");
        if (!has_set("handoff.shm"))
            in.options.sets.push_back("handoff.shm=" + shm);
    }
    // These are the stored measured allocations, not new operator settings.
    // Re-resolve with them for the static settings comparison; live room is
    // independently sampled below and can refuse their reuse.
    if (!has_set("memory.host_pool_mib"))
        in.options.sets.push_back("memory.host_pool_mib=" +
                                  std::to_string(m.spec.resources.host.pool_mib));
    if (!has_set("memory.host_kv_mib"))
        in.options.sets.push_back("memory.host_kv_mib=" +
                                  std::to_string(m.spec.resources.host.kv_mib));
    const Json facts = detect(in.options.sys_root);
    const Resolved resolved = resolve(in.options, facts);
    const Json definitions = load_layouts(in.options);
    if (definitions.at("kind").value(name, "") != "config")
        throw PlanFailed("replay: only a current RSC12 named config is eligible for stored-manifest replay");

    const std::string expected_profile = in.calibration_run ? "lab" : "production";
    const std::string stored_profile =
        source.at("spec").at("intent").at("service").at("profile").get<std::string>();
    if (expected_profile != stored_profile)
        throw PlanFailed("replay: a lab manifest requires the explicitly gated foreground calibration run");
    ManifestInputs draft;
    draft.resolved = resolved;
    draft.profile = expected_profile;
    draft.instance_id = new_instance_id();
    draft.producer_sha256 = file_sha256(given.self);
    draft.calibration_cache = in.options.calibration_cache;
    draft.s10.power_profile = m.spec.intent.power.profile;
    draft.s10.now_unix_s = static_cast<int64_t>(std::time(nullptr));
    draft.s10.calibration_records = Json::array();
    if (in.calibration_run && !in.live_policy_file.empty()) {
        draft.live_policy = Json::parse(read_file(in.live_policy_file));
        draft.live_policy_from = in.live_policy_file.string();
    }
    if (source.at("spec").contains("experiment"))
        throw PlanFailed("replay: lab experiment requires RSC12 static input-key verification, not available here");
    const BuiltManifest current = build_manifest(draft);

    ReplayEvidence evidence;
    evidence.producer_sha256 = draft.producer_sha256;
    evidence.model_identity = container_identity(m.spec.intent.model);
    evidence.engine_identity = engine_identity(engine, home);
    evidence.fresh_facts = facts;
    evidence.effective_spec = current.document.at("spec");
    const ReplayResult checked = authenticate_replay(bytes, stored.string(), evidence);
    if (const auto* error = std::get_if<ReplayError>(&checked))
        throw PlanFailed(v2::render_human("bc-serve replay", error->detail));
    const auto& accepted = std::get<v2::Manifest>(checked);

    std::vector<std::string> ordered = accepted.spec.processes.engine.device.gpus;
    for (const auto& h : accepted.spec.processes.helpers)
        ordered.push_back(h.gpu);
    const auto room = v2::measure_live_room(facts, in.options.sys_root, ordered);
    if (!room)
        throw PlanFailed(v2::render_human("bc-serve replay room", room.error()));
    if (const auto error = check_replay_live_room(accepted, facts, room.value()))
        throw PlanFailed(v2::render_human("bc-serve replay room", *error));

    EngineStart out;
    out.manifest = stored;
    out.stored_sha256 = sha256_hex(bytes);
    out.port = int(accepted.spec.processes.engine.server.port);
    out.argv = {engine.string(), "--manifest", stored.string(), "--role", "engine"};
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
    out.runtime = {{"RADIANCE_HOME", home.string()}, {"ROCR_VISIBLE_DEVICES", ordinals},
                   {"HIP_VISIBLE_DEVICES", hip_ordinals}};
    for (const auto& h : accepted.spec.processes.helpers) {
        const Json* gpu = nullptr;
        for (const auto& g : facts.at("gpus"))
            if (g.at("bdf") == h.gpu)
                gpu = &g;
        if (!gpu)
            throw PlanFailed("replay: helper BDF vanished after live admission: " + h.gpu);
        HelperStart helper;
        helper.role = h.role;
        helper.argv = {helper_binary.string(), "--manifest", stored.string(), "--role", h.role};
        helper.runtime = {{"RADIANCE_HOME", home.string()},
                          {"ROCR_VISIBLE_DEVICES", std::to_string(gpu->at("hip_ordinal").get<int>())},
                          {"HIP_VISIBLE_DEVICES", "0"}};
        out.helpers.push_back(std::move(helper));
    }
    if (accepted.spec.handoff) {
        out.handoff_shm = accepted.spec.handoff->transport.shm;
        if (hooks.selected_segment)
            require_callback(hooks.selected_segment.invoke("session.hooks.selected_segment", out.handoff_shm),
                             "replay handoff namespace");
    }
    out.summary = {{"config", name}, {"source", definitions.at("source").value(name, "")},
                   {"model", accepted.spec.intent.model}, {"engine", engine.string()},
                   {"engine_cards", accepted.spec.processes.engine.device.gpus},
                   {"helper_cards", source.at("spec").at("processes").at("helpers")},
                   {"manifest_digest", accepted.metadata.content_digest},
                   {"replay", true}};
    // No writer or measuring child was started. Admission still passes through
    // the real owner's durable event before ANY GPU child can be forked.
    require_callback(hooks.admit_power.invoke("session.hooks.admit_power", stored, v2::PlanUse::serving),
                     "stored manifest power admission");
    return out;
}


} // namespace bc::serving::session
