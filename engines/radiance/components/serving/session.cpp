#include "session.h"

#include "config_loader.h"

#include <algorithm>
#include <cerrno>
#include <csignal>
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
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/syscall.h>
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
void on_signal(int sig) { g_signal = sig; }

std::string errno_text() { return std::strerror(errno); }

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
        if (!out) throw std::runtime_error("cannot write " + tmp.string());
        out << text;
        out.flush();
        if (!out) throw std::runtime_error("cannot write " + tmp.string());
    }
    fs::rename(tmp, path);
}

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

bool starts_with(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

// Variables that carry bigcherry configuration or choose devices: a child never inherits them.
bool stripped(const std::string& name) {
    for (const char* p : {"R3_", "RAD_", "BC_", "BIGCHERRY_"})
        if (starts_with(name, p)) return true;
    for (const char* v : {"ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
                          "RADIANCE_HOME"})
        if (name == v) return true;
    return false;
}

sockaddr_un socket_address(const fs::path& run_dir) {
    const std::string path = (run_dir / "session.sock").string();
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof addr.sun_path)
        throw std::runtime_error("the run directory's control socket path is too long (" + path + "): use a shorter --run-dir");
    std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
    return addr;
}

// Read until EOF, a newline (when `line`), or the deadline.
std::string read_until(int fd, Clock::time_point deadline, bool line) {
    std::string data;
    char buf[4096];
    while (true) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
        if (left <= 0) break;
        pollfd p{fd, POLLIN, 0};
        const int r = ::poll(&p, 1, int(std::min<long long>(left, 1000)));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) continue;
        const ssize_t n = ::read(fd, buf, sizeof buf);
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        if (n <= 0) break;
        data.append(buf, size_t(n));
        if (line && data.find('\n') != std::string::npos) break;
    }
    return data;
}

bool write_all(int fd, const std::string& data) {
    size_t off = 0;
    while (off < data.size()) {
        const ssize_t n = ::write(fd, data.data() + off, data.size() - off);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        off += size_t(n);
    }
    return true;
}

std::string dechunk(const std::string& body) {
    std::string out;
    size_t at = 0;
    while (at < body.size()) {
        const size_t eol = body.find("\r\n", at);
        if (eol == std::string::npos) break;
        const size_t len = std::stoul(body.substr(at, eol - at), nullptr, 16);
        if (len == 0) break;
        out += body.substr(eol + 2, len);
        at = eol + 2 + len + 2;
    }
    return out;
}

// In a forked child before exec: every descriptor above stderr closes at exec (an inherited socket, pipe or lock of the
// owner never reaches an engine). Marks rather than closes, so the exec-failure report pipe still works until exec.
void cloexec_above_stderr() {
#ifndef CLOSE_RANGE_CLOEXEC
#define CLOSE_RANGE_CLOEXEC (1U << 2)
#endif
    if (::syscall(SYS_close_range, 3U, ~0U, CLOSE_RANGE_CLOEXEC) == 0) return;
    for (int fd = 3; fd < 1024; ++fd) {
        const int flags = ::fcntl(fd, F_GETFD);
        if (flags >= 0) ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
    }
}

bool group_gone(pid_t pgid) { return ::kill(-pgid, 0) != 0 && errno == ESRCH; }

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return s;
}

} // namespace

/* --- children -------------------------------------------------------------------------------------------------- */

Env child_environment(char** inherited, const Env& runtime) {
    Env env;
    for (char** e = inherited; e && *e; ++e) {
        const std::string kv(*e);
        const size_t eq = kv.find('=');
        if (eq == std::string::npos || eq == 0) continue;
        const std::string name = kv.substr(0, eq);
        if (!stripped(name)) env[name] = kv.substr(eq + 1);
    }
    for (const auto& [k, v] : runtime) env[k] = v;
    return env;
}

pid_t spawn(const std::vector<std::string>& argv, const Env& env, const fs::path& log) {
    if (argv.empty() || argv[0].empty()) throw std::runtime_error("spawn: an empty command");
    std::vector<std::string> envs;
    for (const auto& [k, v] : env) envs.push_back(k + "=" + v);
    std::vector<char*> cargv, cenv;
    for (const auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
    cargv.push_back(nullptr);
    for (const auto& e : envs) cenv.push_back(const_cast<char*>(e.c_str()));
    cenv.push_back(nullptr);
    const int out = ::open(log.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (out < 0) throw std::runtime_error("cannot open " + log.string() + ": " + errno_text());
    int report[2];
    if (::pipe2(report, O_CLOEXEC) != 0) {
        ::close(out);
        throw std::runtime_error("pipe: " + errno_text());
    }
    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(out); ::close(report[0]); ::close(report[1]);
        throw std::runtime_error("fork: " + errno_text());
    }
    if (pid == 0) {
        // the child: its own session and process group, default signal dispositions, nothing inherited but the log
        ::setsid();
        for (int s : {SIGINT, SIGTERM, SIGHUP, SIGPIPE, SIGCHLD}) ::signal(s, SIG_DFL);
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
        ::execve(cargv[0], cargv.data(), cenv.data());
        const int e = errno;
        (void)!::write(report[1], &e, sizeof e);
        ::_exit(127);
    }
    ::close(out);
    ::close(report[1]);
    int child_errno = 0;
    ssize_t n;
    do n = ::read(report[0], &child_errno, sizeof child_errno); while (n < 0 && errno == EINTR);
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
    do r = ::waitpid(pid, &st, WNOHANG); while (r < 0 && errno == EINTR);
    if (r == 0) return true;
    if (r == pid && status) *status = st;
    return false;   // exited (reaped now) or not a child of ours any more (reaped before)
}

bool stop_child(pid_t pid, int first, std::chrono::milliseconds grace) {
    // gone means the child reaped AND no process left in its group (a descendant that outlived it included)
    auto gone = [&] { return !running(pid) && group_gone(pid); };
    auto settle = [&](std::chrono::milliseconds limit) {
        const auto deadline = Clock::now() + limit;
        do {
            if (gone()) return true;
            std::this_thread::sleep_for(20ms);
        } while (Clock::now() < deadline);
        return gone();
    };
    if (gone()) return true;
    ::kill(-pid, first);
    if (settle(grace)) return true;
    ::kill(-pid, SIGKILL);
    return settle(grace);
}

std::optional<uint64_t> start_time(pid_t pid) {
    std::ifstream in("/proc/" + std::to_string(pid) + "/stat");
    std::string stat;
    if (!std::getline(in, stat)) return std::nullopt;
    const size_t close = stat.rfind(')');
    if (close == std::string::npos) return std::nullopt;
    std::istringstream rest(stat.substr(close + 2));
    std::string field;
    for (int i = 3; i <= 22 && rest >> field; ++i)
        if (i == 22) try { return std::stoull(field); } catch (...) { return std::nullopt; }
    return std::nullopt;
}

std::vector<Json> alive_recorded(const Json& record) {
    std::vector<Json> alive;
    for (const auto& p : record.value("processes", Json::array())) {
        if (!p.contains("pid") || !p.contains("start_time")) continue;
        const auto t = start_time(p.at("pid").get<pid_t>());
        if (t && *t == p.at("start_time").get<uint64_t>()) alive.push_back(p);
    }
    return alive;
}

/* --- HTTP ---------------------------------------------------------------------------------------------------------- */

std::optional<HttpReply> http(int port, const std::string& method, const std::string& path, const std::string& body,
                              std::chrono::milliseconds timeout) {
    const auto deadline = Clock::now() + timeout;
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (fd < 0) return std::nullopt;
    struct Closer { int fd; ~Closer() { ::close(fd); } } closer{fd};
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
        if (errno != EINPROGRESS) return std::nullopt;
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
        pollfd p{fd, POLLOUT, 0};
        if (left <= 0 || ::poll(&p, 1, int(left)) != 1) return std::nullopt;
        int err = 0;
        socklen_t len = sizeof err;
        if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) != 0 || err != 0) return std::nullopt;
    }
    std::string req = method + " " + path + " HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(port) +
                      "\r\nConnection: close\r\n";
    if (!body.empty()) req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
    req += "\r\n" + body;
    size_t off = 0;
    while (off < req.size()) {
        const ssize_t n = ::send(fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
        if (n > 0) { off += size_t(n); continue; }
        if (n < 0 && (errno == EAGAIN || errno == EINTR)) {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
            pollfd p{fd, POLLOUT, 0};
            if (left <= 0 || ::poll(&p, 1, int(left)) != 1) return std::nullopt;
            continue;
        }
        return std::nullopt;
    }
    const std::string raw = read_until(fd, deadline, false);
    const size_t head_end = raw.find("\r\n\r\n");
    if (!starts_with(raw, "HTTP/1.") || head_end == std::string::npos) return std::nullopt;
    HttpReply reply;
    try { reply.status = std::stoi(raw.substr(raw.find(' ') + 1, 3)); } catch (...) { return std::nullopt; }
    const std::string head = lower(raw.substr(0, head_end));
    reply.body = raw.substr(head_end + 4);
    if (head.find("transfer-encoding: chunked") != std::string::npos) reply.body = dechunk(reply.body);
    return reply;
}

std::optional<std::string> canary(int port, const std::string& prompt, const std::string& expect,
                                  std::chrono::milliseconds timeout) {
    const Json request = {{"model", "default"}, {"prompt", prompt}, {"max_tokens", 24}, {"temperature", 0}};
    const auto reply = http(port, "POST", "/v1/completions", request.dump(), timeout);
    if (!reply) return "the request failed (no HTTP reply)";
    if (reply->status < 200 || reply->status > 299)
        return "the request failed (HTTP " + std::to_string(reply->status) + ": " + reply->body.substr(0, 80) + ")";
    const Json parsed = Json::parse(reply->body, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object() || !parsed.contains("choices") || !parsed.at("choices").is_array() ||
        parsed.at("choices").empty() || !parsed.at("choices")[0].is_object() ||
        !parsed.at("choices")[0].contains("text") || !parsed.at("choices")[0].at("text").is_string())
        return "the reply is not a completion: " + reply->body.substr(0, 80);
    const std::string text = parsed.at("choices")[0].at("text");
    if (lower(text).find(lower(expect)) == std::string::npos)
        return "answered '" + text.substr(0, 80) + "' to '" + prompt + "' (expected it to contain '" + expect + "')";
    return std::nullopt;
}

/* --- the owner ------------------------------------------------------------------------------------------------------ */

namespace {

class Owner {
public:
    explicit Owner(const Session& s) : s_(s) {}
    ~Owner() {
        for (int fd : stop_clients_) ::close(fd);
        if (listen_ >= 0) {
            ::close(listen_);
            std::error_code ec;
            fs::remove(s_.run_dir / "session.sock", ec);
        }
        if (lock_ >= 0) ::close(lock_);
    }

    int run() {
        fs::create_directories(s_.run_dir);
        lock_ = ::open((s_.run_dir / "session.lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (lock_ < 0) throw std::runtime_error("cannot open the session lock: " + errno_text());
        if (::flock(lock_, LOCK_EX | LOCK_NB) != 0) {
            std::string who;
            if (const auto r = read_record(s_.run_dir)) who = " (owner pid " + r->value("owner_pid", Json()).dump() + ")";
            std::cerr << "bc-serve: a session already runs in " << s_.run_dir.string() << who
                      << "; `bc-serve stop` it first\n";
            return 98;
        }
        // The lock is ours. An earlier owner that died without stopping its processes left them in its record: while any
        // of them is still the same process, this run directory's cards are taken.
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
        record_ = {{"schema_version", 1}, {"owner_pid", ::getpid()}, {"started_at", utc_now()},
                   {"state", "created"}, {"events", Json::array()}, {"processes", Json::array()},
                   {"errors", Json::array()}};
        event("created");
        listen_socket();

        struct sigaction sa{};
        sa.sa_handler = on_signal;
        sigemptyset(&sa.sa_mask);
        for (int sig : {SIGINT, SIGTERM, SIGHUP}) ::sigaction(sig, &sa, nullptr);
        ::signal(SIGPIPE, SIG_IGN);
        g_signal = 0;

        int code = 0;
        std::vector<std::string> failures;
        try {
            event("planning");
            Hooks hooks;
            hooks.wait = [this](std::chrono::milliseconds d) { return service(d); };
            hooks.started = [this](const std::string& role, pid_t pid, const fs::path& log) { track(role, pid, log); };
            hooks.ended = [this](pid_t pid) { untrack(pid); };
            plan_ = s_.plan(s_.run_dir, hooks);
            record_["manifest"] = plan_.manifest.string();
            record_["port"] = plan_.port;
            record_["selection"] = plan_.summary;
            event("planned");
            if (stop_requested()) throw Stop{};
            start_engine();
            if (!wait_health()) throw Stop{};
            event("healthy");
            if (const auto why = canary(plan_.port, s_.canary.prompt, s_.canary.expect, s_.timing.canary_timeout)) {
                failures.push_back("start canary failed: the engine is up but " + *why);
                code = 96;
                throw Failed{};
            }
            event("canary_passed");
            event("running");
            while (service(s_.timing.poll)) {
                if (const auto gone = exited_child()) {
                    failures.push_back(*gone + " exited while serving");
                    code = 1;
                    throw Failed{};
                }
            }
            event("stop_requested", {{"by", stop_reason()}});
        } catch (const Stop&) {
            event("stop_requested", {{"by", stop_reason()}});
        } catch (const Failed&) {
        } catch (const PlanFailed& e) {
            failures.push_back(e.what());
            code = 97;
        } catch (const std::exception& e) {
            failures.push_back(e.what());
            if (code == 0) code = 1;
        }
        // Stop every child, the engine first, each bounded; a survivor is never reported stopped.
        bool exited = true;
        if (!children_.empty()) {
            event("stopping");
            for (auto it = children_.rbegin(); it != children_.rend(); ++it) {
                if (!stop_child(it->pid, SIGINT, s_.timing.stop_grace)) {
                    exited = false;
                    failures.push_back(it->role + " (pid " + std::to_string(it->pid) +
                                       ") or a process of its group is still alive after SIGKILL; not reporting stopped "
                                       "while it holds the cards");
                } else {
                    for (auto& p : record_["processes"]) if (p.value("pid", 0) == it->pid) p["exited"] = true;
                }
            }
        }
        for (const auto& f : failures) record_["errors"].push_back(f);
        const std::string final_state = !exited ? "exit_unverified" : (failures.empty() ? "stopped" : "failed");
        event(final_state);
        if (!exited) code = 3;
        else if (!failures.empty() && code == 0) code = 1;
        for (const auto& f : failures) std::cerr << "bc-serve: " << f << '\n';
        reply_stop_clients(final_state);
        return code;
    }

private:
    struct Stop {};
    struct Failed {};
    struct Child { std::string role; pid_t pid; fs::path log; };

    const Session& s_;
    int lock_ = -1, listen_ = -1;
    bool stop_ = false;
    std::vector<int> stop_clients_;
    std::vector<Child> children_;
    EngineStart plan_;
    Json record_;

    // A record that cannot be written is reported, never a reason to skip stopping a child.
    void save() {
        try {
            write_atomic(s_.run_dir / "session.json", record_.dump(2) + "\n");
        } catch (const std::exception& e) {
            if (!save_failed_) std::cerr << "bc-serve: the session record cannot be written: " << e.what() << '\n';
            save_failed_ = true;
        }
    }
    bool save_failed_ = false;

    void track(const std::string& role, pid_t pid, const fs::path& log) {
        children_.push_back({role, pid, log});
        Json p = {{"role", role}, {"pid", pid}, {"log", log.string()}};
        if (const auto t = start_time(pid)) p["start_time"] = *t;
        record_["processes"].push_back(p);
        event("started", {{"role", role}, {"pid", pid}});
    }
    void untrack(pid_t pid) {
        children_.erase(std::remove_if(children_.begin(), children_.end(), [&](const Child& c) { return c.pid == pid; }),
                        children_.end());
        for (auto& p : record_["processes"]) if (p.value("pid", 0) == pid) p["exited"] = true;
        save();
    }

    void event(const std::string& kind, const Json& detail = Json::object()) {
        Json e = {{"at", utc_now()}, {"kind", kind}};
        for (auto it = detail.begin(); it != detail.end(); ++it) e[it.key()] = it.value();
        record_["events"].push_back(e);
        record_["state"] = kind;
        save();
    }

    bool stop_requested() const { return stop_ || g_signal != 0; }
    std::string stop_reason() const {
        if (g_signal) return std::string("signal ") + std::to_string(int(g_signal));
        return stop_ ? "control socket" : "unknown";
    }

    void listen_socket() {
        const sockaddr_un addr = socket_address(s_.run_dir);
        listen_ = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
        if (listen_ < 0) throw std::runtime_error("control socket: " + errno_text());
        if (::bind(listen_, reinterpret_cast<const sockaddr*>(&addr), sizeof addr) != 0 || ::listen(listen_, 8) != 0)
            throw std::runtime_error("control socket " + std::string(addr.sun_path) + ": " + errno_text());
        ::chmod(addr.sun_path, 0600);
    }

    // Wait up to `d` while answering the control socket; false once a stop is requested.
    bool service(std::chrono::milliseconds d) {
        const auto deadline = Clock::now() + d;
        do {
            if (stop_requested()) return false;
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
            pollfd p{listen_, POLLIN, 0};
            const int r = ::poll(&p, 1, int(std::clamp<long long>(left, 0, 100)));
            if (r > 0) accept_one();
        } while (Clock::now() < deadline);
        return !stop_requested();
    }

    void accept_one() {
        const int fd = ::accept4(listen_, nullptr, nullptr, SOCK_CLOEXEC);
        if (fd < 0) return;
        std::string cmd = read_until(fd, Clock::now() + 2s, true);
        cmd = cmd.substr(0, cmd.find('\n'));
        if (cmd == "status") {
            Json reply = record_;
            reply["ok"] = true;
            write_all(fd, reply.dump() + "\n");
            ::close(fd);
        } else if (cmd == "stop") {
            stop_ = true;
            stop_clients_.push_back(fd);   // answered once every child has exited (or could not be stopped)
        } else {
            write_all(fd, Json({{"ok", false}, {"error", "unknown command (status, stop)"}}).dump() + "\n");
            ::close(fd);
        }
    }

    void reply_stop_clients(const std::string& state) {
        for (int fd : stop_clients_) {
            Json reply = record_;
            reply["ok"] = state == "stopped";
            write_all(fd, reply.dump() + "\n");
            ::close(fd);
        }
        stop_clients_.clear();
    }

    void start_engine() {
        const fs::path log = s_.run_dir / ("engine-" + stamp() + ".log");
        const pid_t pid = spawn(plan_.argv, child_environment(environ, plan_.runtime), log);
        track("engine", pid, log);
    }

    std::optional<std::string> exited_child() {
        for (const auto& c : children_) {
            int st = 0;
            if (!running(c.pid, &st)) {
                std::string how = WIFEXITED(st) ? "exit " + std::to_string(WEXITSTATUS(st))
                                                : WIFSIGNALED(st) ? "signal " + std::to_string(WTERMSIG(st)) : "gone";
                return c.role + " (pid " + std::to_string(c.pid) + ", " + how + ", log " + c.log.string() + ")";
            }
        }
        return std::nullopt;
    }

    bool wait_health() {
        const auto deadline = Clock::now() + s_.timing.health_timeout;
        while (Clock::now() < deadline) {
            if (const auto reply = http(plan_.port, "GET", "/health", "", std::min(s_.timing.health_interval, 5000ms)))
                if (reply->status >= 200 && reply->status <= 299) return true;
            if (const auto gone = exited_child())
                throw std::runtime_error("the engine did not come up: " + *gone);
            if (!service(s_.timing.health_interval)) return false;
        }
        throw std::runtime_error("the engine did not answer /health within " +
                                 std::to_string(s_.timing.health_timeout.count() / 1000) + " s; log " +
                                 children_.front().log.string());
    }
};

} // namespace

int serve(const Session& session) {
    Owner owner(session);
    return owner.run();
}

std::optional<Json> read_record(const fs::path& run_dir) {
    const fs::path path = run_dir / "session.json";
    if (!fs::exists(path)) return std::nullopt;
    const Json j = Json::parse(read_file(path), nullptr, false);
    if (j.is_discarded()) return std::nullopt;
    return j;
}

std::optional<Json> control(const fs::path& run_dir, const std::string& command, std::chrono::milliseconds timeout) {
    const sockaddr_un addr = socket_address(run_dir);
    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return std::nullopt;
    struct Closer { int fd; ~Closer() { ::close(fd); } } closer{fd};
    if (::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof addr) != 0) return std::nullopt;
    if (!write_all(fd, command + "\n")) return std::nullopt;
    const std::string reply = read_until(fd, Clock::now() + timeout, true);
    const Json j = Json::parse(reply.substr(0, reply.find('\n')), nullptr, false);
    if (j.is_discarded()) return std::nullopt;
    return j;
}

int start_detached(const std::vector<std::string>& self_argv, const fs::path& run_dir, std::chrono::milliseconds timeout) {
    fs::create_directories(run_dir);
    Env env;
    for (char** e = environ; e && *e; ++e) {
        const std::string kv(*e);
        const size_t eq = kv.find('=');
        if (eq != std::string::npos) env[kv.substr(0, eq)] = kv.substr(eq + 1);
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
                    return answer && answer->value("owner_pid", 0) == pid && answer->value("state", "") == "running";
                }()) {
                std::cout << "running: owner pid " << pid << ", port " << r->value("port", 0) << ", manifest "
                          << r->value("manifest", "") << "\n";
                return 0;
            }
            if (state == "failed" || state == "stopped" || state == "exit_unverified") {
                std::cerr << "bc-serve start: the session " << state << ":\n";
                for (const auto& e : r->value("errors", Json::array())) std::cerr << "  " << e.get<std::string>() << '\n';
                std::cerr << "owner log " << (run_dir / "owner.log").string() << '\n';
                return 1;
            }
        }
        if (!alive) {
            std::cerr << "bc-serve start: the session owner exited ("
                      << (WIFEXITED(st) ? "exit " + std::to_string(WEXITSTATUS(st)) : std::string("signal")) << "); see "
                      << (run_dir / "owner.log").string() << '\n';
            return WIFEXITED(st) && WEXITSTATUS(st) == 98 ? 98 : 1;
        }
        std::this_thread::sleep_for(200ms);
    }
    std::cerr << "bc-serve start: the session is not running after " << timeout.count() / 1000
              << " s; it goes on in the background (owner pid " << pid << "): `bc-serve status` or `bc-serve stop`\n";
    return 1;
}

/* --- the engine-only plan step --------------------------------------------------------------------------------------- */

ManifestRun run_to_completion(const std::vector<std::string>& argv, const Env& env, const Wait& wait) {
    const fs::path dir = fs::temp_directory_path();
    const std::string tag = stamp() + "-" + std::to_string(Clock::now().time_since_epoch().count());
    const fs::path out = dir / ("bc-serve-" + tag + ".out"), err = dir / ("bc-serve-" + tag + ".err");
    // stdout and stderr apart: spawn sends both to one log, so run through a shell-free split with two logs
    const pid_t pid = [&] {
        int o = ::open(out.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        int e = ::open(err.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (o < 0 || e < 0) throw std::runtime_error("cannot create bc-serve output files in " + dir.string());
        std::vector<std::string> envs;
        for (const auto& [k, v] : env) envs.push_back(k + "=" + v);
        std::vector<char*> cargv, cenv;
        for (const auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
        cargv.push_back(nullptr);
        for (const auto& s : envs) cenv.push_back(const_cast<char*>(s.c_str()));
        cenv.push_back(nullptr);
        const pid_t p = ::fork();
        if (p == 0) {
            ::setpgid(0, 0);   // its own group: stop_child signals the group
            for (int s : {SIGINT, SIGTERM, SIGHUP, SIGPIPE}) ::signal(s, SIG_DFL);
            const int in = ::open("/dev/null", O_RDONLY);
            if (in >= 0) ::dup2(in, 0);
            ::dup2(o, 1);
            ::dup2(e, 2);
            cloexec_above_stderr();
            ::execve(cargv[0], cargv.data(), cenv.data());
            ::_exit(127);
        }
        ::close(o);
        ::close(e);
        if (p < 0) throw std::runtime_error("fork: " + errno_text());
        ::setpgid(p, p);   // in both, so the group exists before any stop signals it
        return p;
    }();
    // waited for through the owner (its control socket stays answered); a stop request ends the child
    int st = 0;
    while (running(pid, &st)) {
        if (wait && !wait(100ms)) {
            stop_child(pid, SIGTERM, 10s);
            std::error_code ec;
            fs::remove(out, ec);
            fs::remove(err, ec);
            throw PlanFailed("plan: stopped while bc-serve wrote a manifest");
        }
        if (!wait) std::this_thread::sleep_for(20ms);
    }
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
    if (o.sys_root != "/") a.insert(a.end(), {"--sys-root", o.sys_root.string()});
    if (!o.site.empty()) a.insert(a.end(), {"--site", o.site.string()});
    if (!o.profile.empty()) a.insert(a.end(), {"--profile", o.profile.string()});
    for (const auto& l : o.layouts) a.insert(a.end(), {"--layouts", l.string()});
    for (const auto& c : o.configs) a.insert(a.end(), {"--configs", c.string()});
    if (!o.model_root.empty()) a.insert(a.end(), {"--model-root", o.model_root.string()});
    if (!o.calibration_cache.empty()) a.insert(a.end(), {"--calibration", o.calibration_cache.string()});
    for (const auto& s : o.sets) a.insert(a.end(), {"--set", s});
    return a;
}

std::string tail_text(const std::string& s, size_t n = 2000) { return s.size() > n ? s.substr(s.size() - n) : s; }

} // namespace

EngineStart plan_engine_only(const EnginePlanInputs& given, const fs::path& run_dir, const Hooks& hooks) {
    const Wait& wait = hooks.wait;
    // The sources as files, frozen: the manifest writer runs with BIGCHERRY_* removed from its environment, so configs
    // or layouts that came from the environment are passed as arguments (load_layouts' rule: configs replace the layout
    // files unless --layouts is given).
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
        throw PlanFailed("serve: " + name + " is a layout (" + layouts.at("source").value(name, "") +
                         "); bc-serve serves configs (engines/radiance/configs, --configs): convert it first (RSC12)");
    if (!r.manifest.at("process_helpers").empty())
        throw PlanFailed("serve: " + name + " has helper groups; helper sessions come with RSC02 step 2 (until then: the "
                         "bash launcher)");
    const fs::path home = v.at("engine").at("radiance_home").get<std::string>();
    const fs::path engine = in.engine_binary.empty() ? home.parent_path() / "bin" / "radiance" : in.engine_binary;
    if (::access(engine.c_str(), X_OK) != 0)
        throw PlanFailed("serve: the engine binary " + engine.string() + " is not executable (--engine-binary)");
    // bc-serve from a build (<build>/radiance_home/bin/bc-serve) writes manifests the engine of the same build reads: the
    // plan key and the manifest reader are compiled into both, so an engine or runtime home from another build is refused.
    const fs::path own_home = in.self.parent_path().parent_path();
    const fs::path own_engine = own_home.parent_path() / "bin" / "radiance";
    if (in.self.parent_path().filename() == "bin" && own_home.filename() == "radiance_home" && fs::exists(own_engine)) {
        if (fs::weakly_canonical(home) != fs::weakly_canonical(own_home))
            throw PlanFailed("serve: engine.radiance_home " + home.string() + " is not this bc-serve's build (" +
                             own_home.string() + "); run the bc-serve of the build that serves");
        if (fs::weakly_canonical(engine) != fs::weakly_canonical(own_engine))
            throw PlanFailed("serve: the engine " + engine.string() + " is not this bc-serve's build (" + own_engine.string() +
                             "); a manifest is read by the engine of the build that wrote it");
    }

    // Rank order is the resolver's card order; the process sees its cards as HIP devices 0..n-1.
    std::string rocr, hip;
    const Json& cards = v.at("layout").at("engine_cards");
    for (size_t i = 0; i < cards.size(); ++i) {
        const Json* gpu = nullptr;
        for (const Json& g : facts.at("gpus"))
            if (g.at("bdf") == cards[i]) gpu = &g;
        if (!gpu) throw PlanFailed("serve: engine card " + cards[i].get<std::string>() + " is not in the inventory");
        rocr += (i ? "," : "") + std::to_string(gpu->at("hip_ordinal").get<int64_t>());
        hip += (i ? "," : "") + std::to_string(i);
    }
    EngineStart out;
    out.port = int(v.at("engine").at("port").get<int64_t>());
    out.runtime = {{"RADIANCE_HOME", home.string()}, {"ROCR_VISIBLE_DEVICES", rocr}, {"HIP_VISIBLE_DEVICES", hip}};
    out.summary = {{"config", name}, {"source", layouts.at("source").value(name, "")},
                   {"model", v.at("engine").at("model")}, {"engine_cards", cards}, {"engine", engine.string()}};

    const ManifestRunner runner = in.runner ? in.runner : [&wait](const std::vector<std::string>& argv) {
        return run_to_completion(argv, child_environment(environ, {}), wait);
    };
    const fs::path cache = run_dir / "plan-cache";
    fs::create_directories(cache);
    const std::string id = stamp();
    const fs::path manifest = run_dir / ("manifest-" + id + ".json");
    // The bash launcher's engine-only plan-extra: the two share one plan cache.
    const std::vector<std::string> base = resolver_args(in.options);
    std::vector<std::string> extra_sets;
    auto write = [&](const fs::path& out_file, const std::vector<std::string>& more) {
        std::vector<std::string> argv = {in.self.string(), "--write-manifest", out_file.string(), "--power-profile", "stock"};
        argv.insert(argv.end(), more.begin(), more.end());
        argv.insert(argv.end(), base.begin(), base.end());
        for (const auto& s : extra_sets) argv.insert(argv.end(), {"--set", s});
        return runner(argv);
    };
    const std::vector<std::string> measured_args = {"--plan-cache", cache.string(), "--engine-binary", engine.string(),
                                                    "--plan-extra", "budgets: default"};
    int pool_changes = 0;
    for (int round = 0; round < 5; ++round) {
        std::error_code ec;
        fs::remove(manifest, ec);
        ManifestRun w = write(manifest, measured_args);
        if (w.exit == 10) {
            // No measured plan for this configuration: measure it with the engine alone (it stops after its memory plan).
            std::cerr << "plan: no measured plan for " << name << " (cache miss); measuring it first (about 1-2 minutes)\n";
            const fs::path measuring = run_dir / ("manifest-measuring-" + id + "-" + std::to_string(round) + ".json");
            const ManifestRun u = write(measuring, {"--unmeasured"});
            if (u.exit != 0) throw PlanFailed("plan: the unmeasured manifest was not written: " + tail_text(u.err));
            const fs::path plan = run_dir / "plan-measured.json", ranks = run_dir / "plan-ranks.json";
            fs::remove(plan, ec);
            fs::remove(ranks, ec);
            const fs::path log = run_dir / ("engine-measure-" + id + "-" + std::to_string(round) + ".log");
            const pid_t pid = spawn({engine.string(), "--manifest", measuring.string(), "--role", "engine",
                                     "--plan-memory", plan.string(), "--plan-memory-ranks", ranks.string()},
                                    child_environment(environ, out.runtime), log);
            // registered with the owner: should its exit not be verified here, the owner stops it and the session is never
            // reported stopped while it lives
            if (hooks.started) hooks.started("measuring engine", pid, log);
            int st = 0;
            const auto deadline = Clock::now() + std::chrono::minutes(30);
            bool stopped_early = false;
            try {
                while (running(pid, &st)) {
                    if (Clock::now() > deadline || !wait(200ms)) {
                        stopped_early = true;
                        break;
                    }
                }
            } catch (...) {
                if (stop_child(pid, SIGINT, 60s) && hooks.ended) hooks.ended(pid);
                throw;
            }
            if (stopped_early) {
                if (!stop_child(pid, SIGINT, 60s))
                    throw std::runtime_error("plan: the measuring engine (pid " + std::to_string(pid) +
                                             ") is still alive after SIGKILL");
                if (hooks.ended) hooks.ended(pid);
                throw PlanFailed("plan: the measuring start was stopped (a stop request or 30 min)");
            }
            // the engine exited by itself; anything it left in its group goes too
            if (!stop_child(pid, SIGKILL, 10s))
                throw std::runtime_error("plan: a process of the measuring engine's group is still alive after SIGKILL");
            if (hooks.ended) hooks.ended(pid);
            if (!WIFEXITED(st) || WEXITSTATUS(st) != 0 || !fs::exists(plan) || fs::file_size(plan) == 0 ||
                !fs::exists(ranks) || fs::file_size(ranks) == 0)
                throw PlanFailed("plan: the measuring start failed; see " + log.string());
            std::smatch m;
            const std::string text = read_file(log);
            static const std::regex took(R"(--plan-memory: wrote .*; engine start-up took ([0-9]+) MiB of MemAvailable;)");
            std::string engine_mib;
            for (auto it = std::sregex_iterator(text.begin(), text.end(), took); it != std::sregex_iterator(); ++it)
                engine_mib = (*it)[1];
            if (engine_mib.empty())
                throw PlanFailed("plan: the measuring start did not report the engine's resident host memory; see " +
                                 log.string());
            std::vector<std::string> stored = measured_args;
            stored.insert(stored.end(), {"--measured-plan", plan.string(), "--rank-demand", ranks.string(),
                                         "--helpers-resident-mib", "0", "--engine-resident-mib", engine_mib});
            fs::remove(manifest, ec);
            w = write(manifest, stored);
        }
        if (w.exit == 0) {
            out.manifest = manifest;
            out.argv = {engine.string(), "--manifest", manifest.string(), "--role", "engine"};
            std::smatch d;
            static const std::regex digest(R"(digest: ([0-9a-f]+))");
            if (std::regex_search(w.out, d, digest)) out.summary["manifest_digest"] = d[1].str();
            return out;
        }
        if (w.exit == 11) {
            std::smatch p;
            static const std::regex pool(R"(derived_host_pool_mib=([0-9]+))");
            if (!std::regex_search(w.out, p, pool))
                throw PlanFailed("plan: bc-serve asked for another host pool without naming it: " + tail_text(w.err));
            if (++pool_changes > 2) throw PlanFailed("plan: the host pool did not converge (last " + p[1].str() + " MiB)");
            std::cerr << "plan: the measured inputs need a " << p[1].str() << " MiB host pool; measuring with it\n";
            extra_sets.push_back("memory.host_pool_mib=" + p[1].str());
            continue;
        }
        throw PlanFailed("plan: bc-serve --write-manifest failed (exit " + std::to_string(w.exit) + "): " + tail_text(w.err));
    }
    throw PlanFailed("plan: no measured manifest after 5 rounds");
}

} // namespace bc::serving::session
