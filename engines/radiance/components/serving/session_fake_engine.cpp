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
// A native (v3) serving manifest takes its port from the manifest and the same keys from the JSON object in the
// environment variable TEST_FAKE_BEHAVIOUR (the serving engine only; not a measuring run, a helper or the inspector).
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
#include <algorithm>
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
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>

#include "sha256.h"
#include <rad_abi.h>

extern char** environ;

namespace {
volatile std::sig_atomic_t g_stop = 0;
volatile std::sig_atomic_t g_report = 0;
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
    if ((argc == 3 || (argc == 5 && std::strcmp(argv[3], "--bc-profile-fd") == 0)) &&
        std::strcmp(argv[1], "--bc-inspect-native-config") == 0) {
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
        const char* configured_yaml = std::getenv("TEST_NATIVE_INSPECT_EXPECT_YAML");
        const std::string expected_yaml = configured_yaml
                                              ? configured_yaml
                                              : "# caf\xc3\xa9\r\ncards: [0]\r\nmodels:\r\n  tiny: {model: tiny.rad}\r\n";
        if (bytes != expected_yaml) {
            std::fprintf(stderr, "fake inspector: snapshot bytes did not match the sealed fixture\n");
            return 12;
        }
        std::error_code ec;
        const std::string cwd = std::filesystem::canonical(std::filesystem::current_path(), ec).string();
        if (ec) {
            std::fprintf(stderr, "fake inspector: no execution root\n");
            return 13;
        }
        bc::serving::Sha256 yaml_digest;
        yaml_digest.update(bytes);
        const std::string yaml_sha256 = yaml_digest.hex_digest();
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
                                                : yaml_sha256;
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
        json native_tuning = {{"p2p", nullptr}, {"pipelined_prefill", nullptr}, {"pipelined_prefill_min_rows", nullptr},
                              {"prefill_governor", nullptr}, {"tp_ownership", nullptr}, {"tune_pin", nullptr}};
        if (const char* configured_tuning = std::getenv("TEST_NATIVE_INSPECT_TUNING"))
            native_tuning.update(json::parse(configured_tuning));
        json x_bigcherry = {{"schema", 1}, {"description", "fake inspector config"},
                            {"ranks", json::array({{{"arch", "gfx1100"}, {"count", tp}}})}};
        if (const char* configured = std::getenv("TEST_NATIVE_INSPECT_XBIGCHERRY"))
            x_bigcherry = json::parse(configured);
        json native_settings = {{"max_batched_tokens", 2048}, {"checkpoint_interval", 4096},
                                {"placement", "expert_tiered"}, {"ngram_placement", "ram"}, {"kv_dtype", "fp8"},
                                {"speculative_tokens", 3}, {"rope_scaling", "none"}, {"prefix_resume", "nearest"},
                                {"prefix_cache", true}, {"prefill_interleave", nullptr}};
        if (const char* configured_settings = std::getenv("TEST_NATIVE_INSPECT_SETTINGS"))
            native_settings.update(json::parse(configured_settings));
        // The fake has no YAML parser: the profile document the real engine would convert comes from the test (default:
        // the stock-only file), and the profile descriptor must be a readable sealed memfd as a real call gets.
        json power_profiles = {{"schema", 1}, {"profiles", {{"stock", {{"default_mode", "observe"}}}}}};
        json power_site = nullptr;   // the site file's JSON also comes from the test; its descriptor must still be sealed
        for (int at = 4; at < argc; at += 2) {
            const int descriptor = std::atoi(argv[at]);
            if (descriptor < 3 || ::fcntl(descriptor, F_GET_SEALS) < 0) {
                std::fprintf(stderr, "fake inspector: invalid sealed profile or site descriptor\n");
                return 20;
            }
        }
        if (const char* configured = std::getenv("TEST_NATIVE_INSPECT_POWER_SITE"))
            power_site = json::parse(configured);
        if (const char* configured = std::getenv("TEST_NATIVE_INSPECT_POWER_PROFILES"))
            power_profiles = json::parse(configured);
        json report = {{"protocol", "bigcherry.native-fleet-inspect/v2"},
                       {"snapshot", {{"sha256", reported_digest},
                                     {"resolution_root", cwd}}},
                       {"engine", {{"abi_version", RAD_ABI_VERSION},
                                    {"model", reported_model},
                                    {"radiance_home", reported_home},
                                    {"kernels", kernels},
                                    {"listener", {{"host", std::getenv("TEST_NATIVE_INSPECT_HOST") ? std::getenv("TEST_NATIVE_INSPECT_HOST") : "127.0.0.1"},
                                                   {"port", configured_integer("TEST_NATIVE_INSPECT_PORT", 8123)}}},
                                    {"tp_wire", std::getenv("TEST_NATIVE_INSPECT_TP_WIRE")
                                                    ? json(std::getenv("TEST_NATIVE_INSPECT_TP_WIRE")) : json(nullptr)},
                                    {"tuning", native_tuning},
                                    {"x_bigcherry", x_bigcherry},
                                    {"power_profiles", power_profiles},
                                    {"power_site", power_site},
                                    {"tp", tp},
                                    {"cards", cards},
                                    {"rank_devices", rank_devices},
                                    {"max_context", configured_integer("TEST_NATIVE_INSPECT_MAX_CONTEXT", 4096)},
                                    {"max_sequences", configured_integer("TEST_NATIVE_INSPECT_MAX_SEQUENCES", 1)},
                                    {"settings", native_settings}}}};
        if (mode && std::strcmp(mode, "cards") == 0) report["engine"]["cards"] = json::array({"0"});
        std::cout << report.dump() << '\n';
        return 0;
    }
    const std::string manifest = arg_after(argc, argv, "--manifest");
    const std::string role = arg_after(argc, argv, "--role");
    const std::string settings_fd_text = arg_after(argc, argv, "--settings-fd");
    const std::string config_path = arg_after(argc, argv, "--config");
    const std::string admission_fd_text = arg_after(argc, argv, "--bc-admission-fd");
    json m = json::object();
    if (!config_path.empty()) {
        // The native entry: the raw YAML at --config (an inherited sealed descriptor) and the owner's derived facts in the
        // admission document at --bc-admission-fd. The port, context and n-gram placement are the native file's own
        // values, which this fake cannot parse, so the test supplies them in the environment as it does for the inspector.
        const auto sealed_text = [](const std::string& label, int fd) {
            constexpr int seals_required = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
            struct stat info {};
            const int seals = fd < 3 ? -1 : ::fcntl(fd, F_GET_SEALS);
            if (fd < 3 || seals < 0 || (seals & seals_required) != seals_required || ::fstat(fd, &info) != 0 ||
                info.st_size <= 0) {
                std::fprintf(stderr, "fake engine: %s is not an inherited sealed descriptor\n", label.c_str());
                std::exit(9);
            }
            std::string bytes(static_cast<size_t>(info.st_size), '\0');
            if (::pread(fd, bytes.data(), bytes.size(), 0) != static_cast<ssize_t>(bytes.size())) std::exit(9);
            return bytes;
        };
        int config_fd = -1;
        if (config_path.rfind("/proc/self/fd/", 0) == 0) config_fd = std::atoi(config_path.c_str() + 14);
        const std::string yaml = sealed_text("--config", config_fd);
        const json admission = json::parse(sealed_text("--bc-admission-fd", std::atoi(admission_fd_text.c_str())), nullptr, false);
        const char* expected_yaml = std::getenv("TEST_NATIVE_INSPECT_EXPECT_YAML");
        const char* expected_frozen = std::getenv("TEST_NATIVE_EXPECT_FROZEN");
        const char* frozen = std::getenv("TEST_NATIVE_FROZEN");
        const bool measuring = !arg_after(argc, argv, "--plan-memory").empty();
        if (!admission.is_object() || admission.value("kind", std::string()) != "bigcherry.engine-admission/v1" ||
            admission.value("measuring", false) != measuring || (expected_yaml && yaml != expected_yaml) ||
            (expected_frozen && (!frozen || std::strcmp(frozen, expected_frozen) != 0))) {
            std::fprintf(stderr, "fake engine: native YAML, admission, environment or role mismatch\n");
            return 9;
        }
        const char* expected_root = std::getenv("TEST_NATIVE_EXPECT_ROOT");
        if (expected_root && std::filesystem::current_path() != expected_root) {
            std::fprintf(stderr, "fake engine: the execution root is not the admitted one\n");
            return 9;
        }
        std::printf("fake engine: native v3 %s snapshot, root and environment verified\n", measuring ? "measuring" : "serving");
        m["port"] = std::atoi(std::getenv("TEST_NATIVE_INSPECT_PORT") ? std::getenv("TEST_NATIVE_INSPECT_PORT") : "0");
        m["engine_cards"] = admission.at("bdfs");
        if (admission.at("quality").is_object()) m["quality"] = admission.at("quality");
        m["host_pool_mib"] = admission.at("host").at("pool_mib_per_rank");
        if (const char* context = std::getenv("TEST_NATIVE_INSPECT_MAX_CONTEXT")) m["max_context"] = std::atoll(context);
        if (const char* settings = std::getenv("TEST_NATIVE_INSPECT_SETTINGS")) {
            const json native = json::parse(settings, nullptr, false);
            if (native.is_object() && native.contains("ngram_placement")) m["ngram_placement"] = native.at("ngram_placement");
        }
        if (admission.at("settings").at("handoff") == true) m["shm"] = admission.at("settings").at("shm");
        if (!measuring) {
            if (const char* behaviour = std::getenv("TEST_FAKE_BEHAVIOUR")) {
                const json given = json::parse(behaviour, nullptr, false);
                if (!given.is_object()) {
                    std::fprintf(stderr, "fake engine: TEST_FAKE_BEHAVIOUR is not a JSON object\n");
                    return 9;
                }
                for (const auto& [key, value] : given.items())
                    if (key != "port") m[key] = value;   // the port is the file's
            }
        }
    } else if (!settings_fd_text.empty()) {
        // A helper started by the owner: its typed settings arrive as one sealed descriptor, never a manifest.
        const int fd = std::atoi(settings_fd_text.c_str());
        struct stat info {};
        constexpr int seals_required = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
        const int seals = fd < 3 ? -1 : ::fcntl(fd, F_GET_SEALS);
        if (fd < 3 || seals < 0 || (seals & seals_required) != seals_required || ::fstat(fd, &info) != 0 ||
            info.st_size <= 0) {
            std::fprintf(stderr, "fake helper: invalid sealed settings descriptor\n");
            return 9;
        }
        std::string bytes(static_cast<size_t>(info.st_size), '\0');
        if (::pread(fd, bytes.data(), bytes.size(), 0) != static_cast<ssize_t>(bytes.size())) return 9;
        const json settings = json::parse(bytes, nullptr, false);
        if (!settings.is_object() || settings.value("role", std::string()) != role || !settings.contains("measuring") ||
            !settings.contains("handoff")) {
            std::fprintf(stderr, "fake helper: the settings are not this helper's\n");
            return 9;
        }
        const char* rocr = std::getenv("ROCR_VISIBLE_DEVICES");
        const char* hip = std::getenv("HIP_VISIBLE_DEVICES");
        if (!rocr || !hip || std::strcmp(hip, "0") != 0) {
            std::fprintf(stderr, "fake helper: helper visibility is not one device as HIP 0\n");
            return 9;
        }
        m["shm"] = settings.at("handoff").at("shm");
        if (settings.contains("forced") && settings.at("forced").is_object())
            m["forced_transactions"] = settings.at("forced").at("steps").size() *
                                       settings.at("forced").at("repeat").get<size_t>();
    } else {
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
            // The card-free engine uses the manifest's fixture port and mailbox.
            const auto& spec = parsed.at("spec");
            bool apply_behaviour = false;   // TEST_FAKE_BEHAVIOUR is for the serving engine, not a measuring run
            if (parsed.value("apiVersion", std::string()) == "bigcherry/v3") {
                const auto& native = spec.at("native_config");
                const char* expected_yaml = std::getenv("TEST_NATIVE_INSPECT_EXPECT_YAML");
                const char* expected_frozen = std::getenv("TEST_NATIVE_EXPECT_FROZEN");
                const char* frozen = std::getenv("TEST_NATIVE_FROZEN");
                const bool helper = role.rfind("helper", 0) == 0;
                const bool measuring = !arg_after(argc, argv, "--plan-memory").empty() ||
                    std::find_if(argv + 1, argv + argc, [](const char* arg) {
                        return std::strcmp(arg, "--measuring") == 0;
                    }) != argv + argc;
                if (manifest.rfind("/proc/self/fd/", 0) != 0 ||
                    (!helper && std::filesystem::current_path() != native.at("resolution_root").get<std::string>()) ||
                    (expected_yaml && native.at("yaml_utf8") != expected_yaml) ||
                    (expected_frozen && (!frozen || std::strcmp(frozen, expected_frozen) != 0)) ||
                    spec.at("plan").contains("engine_memory") == measuring) {
                    std::fprintf(stderr, "fake engine: v3 manifest, execution root, environment or role mismatch\n");
                    return 9;
                }
                if (helper) {
                    const auto& helpers = spec.at("processes").at("helpers");
                    const auto worker = std::find_if(helpers.begin(), helpers.end(),
                        [&](const json& h) { return h.at("role") == role; });
                    const auto& gpus = spec.at("facts").at("gpus");
                    const auto gpu = worker == helpers.end() ? gpus.end() : std::find_if(gpus.begin(), gpus.end(),
                        [&](const json& card) { return card.at("bdf") == worker->at("gpu"); });
                    const char* rocr = std::getenv("ROCR_VISIBLE_DEVICES");
                    const char* hip = std::getenv("HIP_VISIBLE_DEVICES");
                    if (gpu == gpus.end() || !rocr || !hip || std::strcmp(hip, "0") != 0 ||
                        std::to_string(gpu->at("hip_ordinal").get<int>()) != rocr) {
                        std::fprintf(stderr, "fake engine: helper visibility differs from its manifest BDF\n");
                        return 9;
                    }
                }
                apply_behaviour = !helper && !measuring;
                std::printf("fake engine: native v3 %s snapshot, root and environment verified\n",
                            measuring ? "measuring" : "serving");
            }
            m["port"] = spec.at("processes").at("engine").at("server").at("port");
            m["engine_cards"] = spec.at("processes").at("engine").at("device").at("gpus");
            m["host_pool_mib"] = spec.at("resources").at("host").at("pool_mib");
            m["max_context"] = spec.at("intent").at("service").at("max_context_tokens");
            m["ngram_placement"] = spec.at("processes").at("engine").at("execution").at("ngram_placement");
            if (spec.contains("handoff"))
                m["shm"] = spec.at("handoff").at("transport").at("shm");
            // A forced placement schedule in the manifest: helper0 reports it, as the real worker does on SIGUSR1.
            if (spec.contains("experiment") && spec.at("experiment").contains("placement") &&
                spec.at("experiment").at("placement").contains("forced")) {
                const auto& forced = spec.at("experiment").at("placement").at("forced");
                m["forced_transactions"] = forced.at("steps").size() * forced.at("repeat").get<size_t>();
            }
            if (const char* behaviour = apply_behaviour ? std::getenv("TEST_FAKE_BEHAVIOUR") : nullptr) {
                const json given = json::parse(behaviour, nullptr, false);
                if (!given.is_object()) {
                    std::fprintf(stderr, "fake engine: TEST_FAKE_BEHAVIOUR is not a JSON object\n");
                    return 9;
                }
                for (const auto& [key, value] : given.items())
                    if (key != "port") m[key] = value;   // the port is the manifest's
            }
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
        std::signal(SIGUSR1, [](int) { g_report = 1; });
        const auto up = Clock::now();
        const auto ready_at = up + std::chrono::milliseconds(h.value("ready_delay_ms", 0));
        const auto die_after = std::chrono::milliseconds(h.value("die_after_ms", -1));
        bool ready = false;
        const bool report_once = std::getenv("TEST_FAKE_REPORT_ONCE") != nullptr;
        bool reported = false;
        while (!g_stop) {
            const auto elapsed = Clock::now() - up;
            if (die_after.count() >= 0 && elapsed >= die_after) return 7;
            if (!ready && Clock::now() >= ready_at && !h.value("never_ready", false)) {
                std::printf("r3_moe_worker: %s helper is ready\n",
                            h.value("waiting_for_maps", false) ? "waiting for engine runtime expert maps" : "ready");
                std::fflush(stdout);
                ready = true;
            }
            if (g_report) {
                g_report = 0;
                if (role == "helper0" && m.contains("forced_transactions") && (!report_once || !reported)) {
                    reported = true;
                    const unsigned long long attempted = m.at("forced_transactions").get<unsigned long long>();
                    const char* committed_override = std::getenv("TEST_FAKE_FORCED_COMMITTED");
                    const unsigned long long committed =
                        committed_override ? std::strtoull(committed_override, nullptr, 10) : attempted;
                    std::printf("forced resolved transaction=1 epoch=2 swap=0:3:0:9 helpers=1:0\n");
                    std::printf("forced placement helper=0 attempted=%llu committed=%llu aborted=%llu inflight=0 "
                                "bytes_planned=0\n",
                                attempted, committed, attempted - committed);
                    std::fflush(stdout);
                }
            }
            ::usleep(10000);
        }
        return 0;
    }
    // A KL run's files arrive in the admission (the native entry), as the flags the manifest entry took.
    const auto kld_arg = [&](const char* flag) -> std::string {
        if (!m.contains("quality") || !m.at("quality").is_object()) return arg_after(argc, argv, flag);
        const json& q = m.at("quality");
        const bool record = q.at("record").get<bool>();
        const std::string f = flag;
        if (f == "--kld-record") return record ? q.at("reference").get<std::string>() : std::string();
        if (f == "--kld-corpus") return record ? q.at("corpus").get<std::string>() : std::string();
        if (f == "--kld-ref") return record ? std::string() : q.at("reference").get<std::string>();
        if (f == "--kld-out") return record ? std::string() : q.at("output").get<std::string>();
        return arg_after(argc, argv, flag);
    };
    const std::string record_dir = kld_arg("--kld-record");
    if (!record_dir.empty()) {
        const std::string corpus = kld_arg("--kld-corpus");
        if (corpus.empty() || !std::filesystem::path(record_dir).is_absolute() ||
            !std::filesystem::path(corpus).is_absolute() || !std::filesystem::is_regular_file(corpus) ||
            !kld_arg("--kld-ref").empty() || !kld_arg("--kld-out").empty() ||
            !arg_after(argc, argv, "--bc-heat-launch-nonce").empty())
            return 9;
        std::ofstream(std::filesystem::path(record_dir) / "kld.json") << json{{"positions", 1}}.dump() << '\n';
        return 0;
    }
    const std::string reference = kld_arg("--kld-ref");
    if (!reference.empty()) {
        const std::string output = kld_arg("--kld-out");
        std::ifstream metadata(std::filesystem::path(reference) / "kld.json");
        const auto fixture = json::parse(metadata, nullptr, false);
        if (output.empty() || !fixture.is_object() ||
            !std::filesystem::path(reference).is_absolute() || !std::filesystem::path(output).is_absolute() ||
            !arg_after(argc, argv, "--bc-heat-launch-nonce").empty())
            return 9;
        std::this_thread::sleep_for(std::chrono::milliseconds(fixture.value("fake_delay_ms", 0)));
        std::ofstream(output) << json{{"all", {{"positions", 1}}}}.dump() << '\n';
        std::ofstream(output + ".rows", std::ios::binary).write("0123456789abcdef", 16);
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
            {"kv_one_session", {{"bytes", std::max<int64_t>(4096, m.value("max_context", int64_t{4096}))},
                                  {"from", "engine plan (fake measurement)"}}},
            {"vram", {{"claimable", 10}, {"statics", 1}, {"elastic", 9}, {"experts", 0}, {"kv", 9},
                       {"stage", 0}, {"kv_resolved", 9}, {"expert_floor", 0}, {"kv_ceiling", 9},
                       {"expert_ceiling", 0}, {"from", "engine plan (fake measurement)"},
                       {"cards", json::array()}}},
            {"host_pool_mib", m.value("host_pool_mib", 9216)}};
        for (size_t i = 0; i < cards.size(); ++i)
            plan_doc["vram"]["cards"].push_back(
                {{"capacity", 10}, {"free", 10}, {"engine_expert_plane", 0}, {"expert_unit_max", 1}});
        // As the engine does for an "auto" tier: it resolves one (RAM here) and its rank receipt names the choice.
        const std::string requested_ngram = m.value("ngram_placement", std::string());
        const std::string ngram = requested_ngram == "auto" ? "ram" : requested_ngram;
        if (ngram == "ram")
            plan_doc["host"].push_back({{"name", "ngram_table"}, {"bytes", 1048576},
                                        {"residence", "pinned"}, {"from", "engine plan (fake measurement)"},
                                        {"remedy", ""}});
        const int64_t engine_resident_mib = m.value("engine_mib", int64_t{1234});
        if (engine_resident_mib <= 0 || engine_resident_mib > ((int64_t{1} << 53) - 1) / (int64_t{1} << 20))
            throw std::runtime_error("fake engine_mib is outside the measured wire range");
        // A non-integral MiB sidecar exercises rad0011's ceiling-to-MiB resident receipt.
        const int64_t resident_other_bytes = engine_resident_mib * (int64_t{1} << 20) - 1024;
        json ranks = {{"schema_version", 1}, {"kind", "radiance_p1_rank_demand"},
                      {"gpu", json::array()},
                      {"resident_other_bytes", resident_other_bytes}};
        if (requested_ngram == "auto")
            ranks["ngram_placement"] = ngram;
        for (const auto& bdf : cards)
            ranks["gpu"].push_back({{"bdf", bdf}, {"required_bytes", 10}, {"fixed_bytes", 1},
                                    {"claimable_bytes", 10}, {"evidence", "rad0011 fake measurement"}});
        std::ofstream(plan) << plan_doc.dump(2) << '\n';
        std::ofstream(arg_after(argc, argv, "--plan-memory-ranks")) << ranks.dump(2) << '\n';
        std::printf("I --plan-memory: wrote %s; engine start-up took %lld MiB of MemAvailable; done\n", plan.c_str(),
                    static_cast<long long>(engine_resident_mib));
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
    const std::string served_model = m.value("served_model", std::string("qwen27b"));
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
        } else if (req.rfind("GET /v1/models", 0) == 0) {
            reply(fd, 200, json({{"object", "list"}, {"data", json::array({{{"id", served_model}}})}}).dump());
        } else if (req.rfind("POST /v1/completions", 0) == 0) {
            // Like the native front: only the model the file names (and its aliases, none here) is served.
            const size_t body_at = req.find("\r\n\r\n");
            const json asked = json::parse(body_at == std::string::npos ? std::string() : req.substr(body_at + 4), nullptr, false);
            if (asked.is_object() && asked.value("model", std::string()) != served_model) {
                reply(fd, 404, json({{"error", {{"message", "no model '" + asked.value("model", std::string()) +
                                                      "' is served here (serving: " + served_model + ")"}}}}).dump());
            } else
            reply(fd, 200, json({{"choices", json::array({{{"text", answer}}})}}).dump());
        } else {
            reply(fd, 404, "{}");
        }
        ::close(fd);
    }
    std::printf("fake engine: stopping\n");
    return 0;
}
