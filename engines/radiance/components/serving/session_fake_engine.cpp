// Test only (RSC02): a stand-in for the radiance engine, card-free. It takes the engine's command line
// (--manifest M --role engine [--plan-memory P --plan-memory-ranks R]) and reads its behaviour from M, a JSON object:
//   port             the HTTP port (/health, /v1/completions)
//   answer           the completion text (default "Paris.")
//   health_delay_ms  how long /health answers 503 first
//   exit_at_start    exit with this code before listening
//   die_after_ms     exit 7 this long after listening
//   ignore_sigint    true: SIGINT is ignored (only SIGKILL stops it)
//   env_dump         write the environment, one NAME=VALUE a line, to this file
//   engine_mib       with --plan-memory: the start-up MiB it reports (default 1234)
//   descendant       fork a process that ignores SIGINT and SIGTERM, outlives this one, and writes its PID here
//   fd_dump          write what each open descriptor points at, one a line, to this file
// A manifest that is not such an object (a real one, from a test's fake bc-serve) gives the defaults with port 0.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <unistd.h>
#include <dirent.h>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

extern char** environ;

namespace {
volatile std::sig_atomic_t g_stop = 0;
void on_stop(int) { g_stop = 1; }
// Test-only POSIX mailbox creator: graceful exits retire their own name;
 // SIGKILL intentionally bypasses this destructor to exercise owner cleanup.
struct CreatedSegment {
    std::string name;
    int fd = -1;
    explicit CreatedSegment(std::string segment_name) : name(std::move(segment_name)) {}
    CreatedSegment(const CreatedSegment&) = delete;
    CreatedSegment& operator=(const CreatedSegment&) = delete;
    ~CreatedSegment() {
        if (fd >= 0) {
            ::close(fd);
            ::shm_unlink(name.c_str());
        }
    }
    bool create() {
        fd = ::shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
        return fd >= 0 && ::ftruncate(fd, 8192) == 0;
    }
};

std::string arg_after(int argc, char** argv, const char* flag) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return {};
}

void reply(int fd, int status, const std::string& body) {
    const std::string head = "HTTP/1.1 " + std::to_string(status) + (status == 200 ? " OK" : " Unavailable") +
                             "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) +
                             "\r\nConnection: close\r\n\r\n";
    const std::string all = head + body;
    size_t off = 0;
    while (off < all.size()) {
        const ssize_t n = ::send(fd, all.data() + off, all.size() - off, MSG_NOSIGNAL);
        if (n <= 0) break;
        off += size_t(n);
    }
}
} // namespace

int main(int argc, char** argv) {
    using nlohmann::json;
    using Clock = std::chrono::steady_clock;
    const std::string manifest = arg_after(argc, argv, "--manifest");
    json m = json::object();
    {
        std::ifstream in(manifest);
        std::stringstream s;
        s << in.rdbuf();
        const json parsed = json::parse(s.str(), nullptr, false);
        if (parsed.is_object() && !parsed.contains("spec")) m = parsed;
    }
    std::printf("fake engine: pid %d manifest %s\n", int(::getpid()), manifest.c_str());
    std::fflush(stdout);
    if (m.contains("env_dump")) {
        std::ofstream out(m.at("env_dump").get<std::string>());
        for (char** e = environ; *e; ++e) out << *e << '\n';
    }
    if (m.contains("fd_dump")) {
        std::ofstream out(m.at("fd_dump").get<std::string>());
        if (DIR* d = ::opendir("/proc/self/fd")) {
            while (const dirent* e = ::readdir(d)) {
                char target[4096] = {};
                const std::string link = std::string("/proc/self/fd/") + e->d_name;
                if (::readlink(link.c_str(), target, sizeof target - 1) > 0) out << e->d_name << ' ' << target << '\n';
            }
            ::closedir(d);
        }
    }
    if (m.contains("descendant")) {
        const pid_t child = ::fork();
        if (child == 0) {
            std::signal(SIGINT, SIG_IGN);
            std::signal(SIGTERM, SIG_IGN);
            for (int i = 0; i < 600; ++i) ::usleep(100000);   // at most a minute, should the test fail to stop it
            std::_Exit(0);
        }
        std::ofstream(m.at("descendant").get<std::string>()) << child << '\n';
    }
    // A fake helper shares the same manifest fixture. It reports a fresh readiness line, holds its
    // process group until stopped, and can fail before/after readiness or ignore SIGINT.
    const std::string role = arg_after(argc, argv, "--role");
    if (role.rfind("helper", 0) == 0) {
        const json h = m.value("helpers", json::object()).value(role, json::object());
        if (h.contains("env_dump")) {
            std::ofstream out(h.at("env_dump").get<std::string>());
            for (char** e = environ; *e; ++e) out << *e << '\n';
        }
        if (h.contains("exit_at_start")) return h.at("exit_at_start").get<int>();
        // Card-free helper0 models the POSIX mailbox's creator. The real worker
        // creates O_EXCL and retires only its own inode; these fake paths are test-only.
        const std::string shm = m.value("shm", std::string());
        CreatedSegment segment(shm);
        if (role == "helper0" && !shm.empty() && !h.value("no_shm", false) && !segment.create())
            return 8;
        std::signal(SIGINT, h.value("ignore_sigint", false) ? SIG_IGN : on_stop);
        std::signal(SIGTERM, on_stop);
        const auto up = Clock::now();
        const auto ready_at = up + std::chrono::milliseconds(h.value("ready_delay_ms", 0));
        const auto die_after = std::chrono::milliseconds(h.value("die_after_ms", -1));
        bool ready = false;
        while (!g_stop) {
            const auto elapsed = Clock::now() - up;
            if (die_after.count() >= 0 && elapsed >= die_after) return 7;
            if (!ready && Clock::now() >= ready_at && !h.value("never_ready", false)) {
                std::printf("r3_moe_worker: %s helper is ready\n",
                            h.value("waiting_for_maps", false) ? "waiting for engine runtime expert maps" : "ready");
                std::fflush(stdout);
                ready = true;
            }
            ::usleep(10000);
        }
        return 0;
    }
    const std::string plan = arg_after(argc, argv, "--plan-memory");
    if (!plan.empty()) {
        std::ofstream(plan) << "{\"plan\": \"measured\"}\n";
        std::ofstream(arg_after(argc, argv, "--plan-memory-ranks")) << "[{\"rank\": 0}]\n";
        std::printf("I --plan-memory: wrote %s; engine start-up took %d MiB of MemAvailable; done\n", plan.c_str(),
                    m.value("engine_mib", 1234));
        return 0;
    }
    if (m.contains("exit_at_start")) return m.at("exit_at_start").get<int>();
    std::signal(SIGINT, m.value("ignore_sigint", false) ? SIG_IGN : on_stop);
    std::signal(SIGTERM, on_stop);

    const int port = m.value("port", 0);
    const int ls = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    ::setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(ls, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 || ::listen(ls, 16) != 0) {
        std::perror("fake engine: listen");
        return 5;
    }
    const auto up = Clock::now();
    const auto healthy_at = up + std::chrono::milliseconds(m.value("health_delay_ms", 0));
    const long die_after = m.value("die_after_ms", -1L);
    const std::string answer = m.value("answer", std::string("Paris."));
    while (!g_stop) {
        if (die_after >= 0 && Clock::now() - up > std::chrono::milliseconds(die_after)) return 7;
        pollfd p{ls, POLLIN, 0};
        if (::poll(&p, 1, 50) <= 0) continue;
        const int fd = ::accept(ls, nullptr, nullptr);
        if (fd < 0) continue;
        std::string req;
        char buf[4096];
        size_t want = std::string::npos;
        while (true) {
            pollfd q{fd, POLLIN, 0};
            if (::poll(&q, 1, 2000) <= 0) break;
            const ssize_t n = ::read(fd, buf, sizeof buf);
            if (n <= 0) break;
            req.append(buf, size_t(n));
            const size_t head = req.find("\r\n\r\n");
            if (head != std::string::npos && want == std::string::npos) {
                const size_t cl = req.find("Content-Length: ");
                want = head + 4 + (cl != std::string::npos && cl < head ? std::stoul(req.substr(cl + 16)) : 0);
            }
            if (want != std::string::npos && req.size() >= want) break;
        }
        if (req.rfind("GET /health", 0) == 0) {
            reply(fd, Clock::now() >= healthy_at ? 200 : 503, "{}");
        } else if (req.rfind("POST /v1/completions", 0) == 0) {
            reply(fd, 200, json({{"choices", json::array({{{"text", answer}}})}}).dump());
        } else {
            reply(fd, 404, "{}");
        }
        ::close(fd);
    }
    std::printf("fake engine: stopping\n");
    return 0;
}
