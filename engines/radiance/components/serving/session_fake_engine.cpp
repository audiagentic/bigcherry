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
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>
#include <rad_abi.h>

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
    if (argc == 3 && std::strcmp(argv[1], "--bc-inspect-native-config") == 0) {
        const int fd = std::atoi(argv[2]);
        constexpr int seals_required = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
        struct stat info {};
        const int seals = fd < 3 ? -1 : ::fcntl(fd, F_GET_SEALS);
        if (fd < 3 || seals < 0 || (seals & seals_required) != seals_required ||
            ::fstat(fd, &info) != 0 || info.st_size <= 0 || info.st_size > 1024 * 1024) {
            std::fprintf(stderr, "fake inspector: invalid sealed input descriptor\n");
            return 10;
        }
        std::string bytes(static_cast<size_t>(info.st_size), '\0');
        if (::pread(fd, bytes.data(), bytes.size(), 0) != static_cast<ssize_t>(bytes.size())) return 11;
        const json input = json::parse(bytes, nullptr, false);
        const char* configured_yaml = std::getenv("TEST_NATIVE_INSPECT_EXPECT_YAML");
        const std::string expected_yaml = configured_yaml
                                              ? configured_yaml
                                              : "# caf\xc3\xa9\r\ncards: [0]\r\nmodels:\r\n  tiny: {model: tiny.rad}\r\n";
        if (!input.is_object() || input.value("yaml_utf8", std::string()) != expected_yaml ||
            input.value("sha256", std::string()).empty() || !input.contains("resolution_root") ||
            !input.at("resolution_root").is_string()) {
            std::fprintf(stderr, "fake inspector: snapshot bytes did not match the sealed fixture\n");
            return 12;
        }
        std::error_code ec;
        const std::string cwd = std::filesystem::canonical(std::filesystem::current_path(), ec).string();
        if (ec || cwd != input.at("resolution_root").get<std::string>()) {
            std::fprintf(stderr, "fake inspector: execution root did not match snapshot\n");
            return 13;
        }
        if (const char* sentinel = std::getenv("TEST_SENTINEL_FD")) {
            errno = 0;
            if (::fcntl(std::atoi(sentinel), F_GETFD) >= 0 || errno != EBADF) {
                std::fprintf(stderr, "fake inspector: unrelated descriptor survived exec\n");
                return 14;
            }
        }
        if (const char* expected_home = std::getenv("TEST_EXPECT_RADIANCE_HOME")) {
            const char* actual = std::getenv("RADIANCE_HOME");
            if (!actual || std::strcmp(actual, expected_home) != 0) {
                std::fprintf(stderr, "fake inspector: explicit native environment was not applied\n");
                return 15;
            }
        }
        if (const char* expected_kernels = std::getenv("TEST_EXPECT_RADIANCE_KERNELS")) {
            const char* actual = std::getenv("RADIANCE_KERNELS");
            if (!actual || std::strcmp(actual, expected_kernels) != 0) {
                std::fprintf(stderr, "fake inspector: native kernel environment differs\n");
                return 16;
            }
        }
        const char* mode = std::getenv("TEST_INSPECTOR_MISMATCH");
        const std::string reported_digest = mode && std::strcmp(mode, "digest") == 0
                                                ? std::string(64, '0')
                                                : input.at("sha256").get<std::string>();
        int tp = 1;
        if (const char* configured_tp = std::getenv("TEST_NATIVE_INSPECT_TP"))
            tp = std::atoi(configured_tp);
        json rank_devices = json::array({"0000:01:00.0"});
        if (const char* configured_devices = std::getenv("TEST_NATIVE_INSPECT_RANK_DEVICES")) {
            const json parsed = json::parse(configured_devices, nullptr, false);
            if (!parsed.is_array()) {
                std::fprintf(stderr, "fake inspector: invalid test rank-device list\n");
                return 17;
            }
            rank_devices = parsed;
        }
        json cards = json::array({0});
        if (const char* configured_cards = std::getenv("TEST_NATIVE_INSPECT_CARDS")) {
            const json parsed = json::parse(configured_cards, nullptr, false);
            if (!parsed.is_array()) {
                std::fprintf(stderr, "fake inspector: invalid test card list\n");
                return 18;
            }
            cards = parsed;
        }
        const char* reported_home = std::getenv("TEST_NATIVE_INSPECT_REPORTED_HOME");
        if (!reported_home) reported_home = std::getenv("RADIANCE_HOME");
        if (!reported_home) reported_home = "runtime";
        const char* reported_model = std::getenv("TEST_NATIVE_INSPECT_MODEL");
        if (!reported_model) reported_model = "tiny.rad";
        auto configured_integer = [](const char* name, int fallback) {
            const char* value = std::getenv(name);
            return value ? std::atoi(value) : fallback;
        };
        json kernels = json::array({"k0"});
        if (const char* configured_kernels = std::getenv("TEST_NATIVE_INSPECT_KERNELS")) {
            const json parsed = json::parse(configured_kernels, nullptr, false);
            if (!parsed.is_array()) {
                std::fprintf(stderr, "fake inspector: invalid test kernel list\n");
                return 19;
            }
            kernels = parsed;
        }
        json report = {{"protocol", "bigcherry.native-fleet-inspect/v1"},
                       {"snapshot", {{"sha256", reported_digest},
                                     {"resolution_root", input.at("resolution_root")}}},
                       {"engine", {{"abi_version", RAD_ABI_VERSION},
                                    {"model", reported_model},
                                    {"radiance_home", reported_home},
                                    {"radiance_home_source", nullptr},
                                    {"kernels", kernels},
                                    {"kernels_source", nullptr},
                                    {"listener", {{"host", "127.0.0.1"},
                                                   {"port", configured_integer("TEST_NATIVE_INSPECT_PORT", 8123)}}},
                                    {"tp", tp},
                                    {"cards", cards},
                                    {"rank_devices", rank_devices},
                                    {"max_context", configured_integer("TEST_NATIVE_INSPECT_MAX_CONTEXT", 4096)},
                                    {"max_sequences", configured_integer("TEST_NATIVE_INSPECT_MAX_SEQUENCES", 1)}}}};
        if (mode && std::strcmp(mode, "cards") == 0) report["engine"]["cards"] = json::array({"0"});
        std::cout << report.dump() << '\n';
        return 0;
    }
    const std::string manifest = arg_after(argc, argv, "--manifest");
    json m = json::object();
    {
        std::ifstream in(manifest);
        std::stringstream s;
        s << in.rdbuf();
        const json parsed = json::parse(s.str(), nullptr, false);
        // Fresh and measuring launches deliberately pass the manifest as an
        // inherited sealed descriptor. Fail loudly if spawn omitted it; a
        // default fake-engine response would otherwise hide the broken handoff.
        if (manifest.rfind("/proc/self/fd/", 0) == 0 &&
            (!parsed.is_object() || !parsed.contains("spec"))) {
            std::fprintf(stderr, "fake engine: inherited manifest descriptor is not readable\n");
            return 9;
        }
        if (parsed.is_object() && !parsed.contains("spec")) {
            m = parsed;
        } else if (parsed.is_object() && parsed.contains("spec")) {
            // The actual owner replay path consumes a strict v2 file. This
            // card-free engine reads ONLY its fixture port and mailbox.
            const auto& spec = parsed.at("spec");
            m["port"] = spec.at("processes").at("engine").at("server").at("port");
            m["engine_cards"] = spec.at("processes").at("engine").at("device").at("gpus");
            m["host_pool_mib"] = spec.at("resources").at("host").at("pool_mib");
            m["ngram_placement"] = spec.at("processes").at("engine").at("execution").at("ngram_placement");
            if (spec.contains("handoff"))
                m["shm"] = spec.at("handoff").at("transport").at("shm");
        }
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
        if (m.contains("measure_error")) {
            std::printf("%s\n", m.at("measure_error").get<std::string>().c_str());
            std::fflush(stdout);
            return 42;
        }
        const json cards = m.value("engine_cards", json::array({"0000:01:00.0"}));
        json plan_doc = {
            {"plan_version", 1}, {"producer", "radiance --plan-memory (rad0011)"},
            {"host", json::array()},
            {"checkpoints", {{"slot_bytes", 1048576}, {"floor_slots", 8}, {"want_slots", 72},
                              {"from", "engine plan (fake measurement)"}}},
            {"kv_one_session", {{"bytes", 4096}, {"from", "engine plan (fake measurement)"}}},
            {"vram", {{"claimable", 10}, {"statics", 1}, {"elastic", 9}, {"experts", 0}, {"kv", 9},
                       {"stage", 0}, {"kv_resolved", 9}, {"expert_floor", 0}, {"kv_ceiling", 9},
                       {"expert_ceiling", 0}, {"from", "engine plan (fake measurement)"},
                       {"cards", json::array()}}},
            {"host_pool_mib", m.value("host_pool_mib", 9216)}};
        for (size_t i = 0; i < cards.size(); ++i)
            plan_doc["vram"]["cards"].push_back(
                {{"capacity", 10}, {"free", 10}, {"engine_expert_plane", 0}, {"expert_unit_max", 1}});
        if (m.value("ngram_placement", std::string()) == "ram")
            plan_doc["host"].push_back({{"name", "ngram_table"}, {"bytes", 1048576},
                                        {"residence", "pinned"}, {"from", "engine plan (fake measurement)"},
                                        {"remedy", ""}});
        json ranks = {{"schema_version", 1}, {"kind", "radiance_p1_rank_demand"},
                      {"gpu", json::array()}, {"resident_other_bytes", 0}};
        for (const auto& bdf : cards)
            ranks["gpu"].push_back({{"bdf", bdf}, {"required_bytes", 10}, {"fixed_bytes", 1},
                                    {"claimable_bytes", 10}, {"evidence", "rad0011 fake measurement"}});
        std::ofstream(plan) << plan_doc.dump(2) << '\n';
        std::ofstream(arg_after(argc, argv, "--plan-memory-ranks")) << ranks.dump(2) << '\n';
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
