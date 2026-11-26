// RSC02 step 1: the session owner's lifecycle, card-free, against a fake engine (session_fake_engine.cpp) that
// serves /health and /v1/completions and misbehaves on request (TEST_FAKE_BEHAVIOUR). Every start is a native (v3)
// one: the engine's plan is the real plan step (plan_engine_only) over the card-free native host of
// test_native_session_fixture.h, so the owner launches from a sealed manifest, root and environment as in production. Ported from the Python supervisor's
// fake-process tests (tools/tests/serving/test_supervisor.py): ordered start and stop, a failure at every stage
// unwinds what started, a survivor is never reported stopped, one session at a time, stop through the control
// socket only, inherited configuration never reaches a child.
#include "power_transaction.h"
#include "session.h"
#include "test_native_session_fixture.h"

#include <regex>
#include <rad_abi.h>

#include <array>
#include <cassert>
#ifdef NDEBUG
#error "assert-based test built with NDEBUG: its assertions would not run"
#endif
#include <csignal>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef BC_FAKE_ENGINE
#error "BC_FAKE_ENGINE must name the fake engine binary"
#endif

static_assert(!std::is_invocable_v<bc::serving::session::Wait, std::chrono::milliseconds>);
namespace fs = std::filesystem;
namespace ss = bc::serving::session;
using Json = nlohmann::json;
using namespace std::chrono_literals;

namespace {

int free_port() { return bc::test::free_port(); }

fs::path g_root;
fs::path g_host; // the card-free native host every start plans against
int g_case = 0;
// owners still running when an assertion aborts the test: stopped (SIGTERM: they stop their engines) so none is
// orphaned
std::vector<pid_t> g_owners;
void stop_owners(int) {
    for (pid_t p : g_owners)
        ::kill(p, SIGTERM);
    std::signal(SIGABRT, SIG_DFL);
    std::abort();
}

struct Case {
    fs::path run;
    int port = free_port();
    Json behaviour = Json::object();
    int helpers = 0;
    bool capture_owner_log = false;
    ss::Session session;
    ss::EngineStart prepared;
    bool planned = false;

    explicit Case(const std::string& name) : run(g_root / (std::to_string(++g_case) + "-" + name)) {
        fs::create_directories(run);
        session.run_dir = run;
        session.timing.health_timeout = 5s;
        session.timing.health_interval = 50ms;
        session.timing.canary_timeout = 5s;
        session.timing.stop_grace = 1s;
        session.timing.poll = 20ms;
        session.plan = [this](const fs::path&, const ss::Hooks& hooks) {
            ss::EngineStart s = prepared;
            if (behaviour.contains("shm")) {
                s.handoff_shm = behaviour.at("shm").get<std::string>();
                if (hooks.selected_segment) {
                    const auto status =
                        hooks.selected_segment.invoke("session.test.selected_segment", s.handoff_shm);
                    if (status)
                        throw ss::PlanFailed(status->message);
                }
            }
            const fs::path manifest = run / "fake-manifest.json";
            for (int i = 0; i < helpers; ++i) {
                const std::string role = "helper" + std::to_string(i);
                ss::HelperStart h;
                h.role = role;
                h.argv = {BC_FAKE_ENGINE, "--manifest", manifest.string(), "--role", role};
                h.runtime = {{"RADIANCE_HOME", "/runtime"},
                             {"ROCR_VISIBLE_DEVICES", std::to_string(6 + i)},
                             {"HIP_VISIBLE_DEVICES", "0"}};
                s.helpers.push_back(h);
            }
            return s;
        };
    }

    // The engine's plan is made here, in the test process, the way `bc-serve` plans before the owner exists: the
    // real plan step (inspect, measure, seal) over the native host, with the fake engine's behaviour in its
    // environment. The owner then launches that sealed plan. Helpers (card-free fakes) are added to it in the owner.
    void prepare() {
        if (planned)
            return;
        planned = true;
        Json m = behaviour;
        m["port"] = port;
        std::ofstream(run / "fake-manifest.json") << m.dump(); // the fake helpers' behaviour
        const auto native = bc::test::native_session_inputs(g_host, run / "native", port, behaviour);
        fs::create_directories(run / "plan");
        prepared = ss::plan_engine_only(native.inputs, run / "plan", bc::test::standalone_plan_hooks());
    }

    // the owner in its own process, as `bc-serve serve` runs
    pid_t owner() {
        prepare();
        std::cout.flush();
        const pid_t pid = ::fork();
        if (pid == 0) {
            if (capture_owner_log) {
                const int log = ::open((run / "owner.log").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
                assert(log >= 0 && ::dup2(log, STDERR_FILENO) == STDERR_FILENO);
                ::close(log);
            }
            int code = 1;
            try {
                code = ss::serve(session);
            } catch (const std::exception& e) {
                std::cerr << "owner: " << e.what() << '\n';
            }
            std::_Exit(code);
        }
        g_owners.push_back(pid);
        return pid;
    }

    Json record() const {
        return ss::read_record(run).value_or(Json::object());
    }

    std::string wait_state(const std::set<std::string>& states, std::chrono::milliseconds limit = 10s) const {
        const auto deadline = std::chrono::steady_clock::now() + limit;
        while (std::chrono::steady_clock::now() < deadline) {
            const std::string s = record().value("state", "");
            if (states.contains(s))
                return s;
            std::this_thread::sleep_for(20ms);
        }
        std::cerr << "state never reached; record: " << record().dump(2) << '\n';
        assert(false);
        return {};
    }

    int engine_pid() const {
        for (const auto& p : record().value("processes", Json::array()))
            if (p.at("role") == "engine")
                return p.at("pid");
        return 0;
    }
};

int exit_code(pid_t pid, std::chrono::milliseconds limit = 15s) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    int st = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        if (::waitpid(pid, &st, WNOHANG) == pid)
            return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
        std::this_thread::sleep_for(20ms);
    }
    ::kill(pid, SIGKILL);
    ::waitpid(pid, &st, 0);
    assert(false && "the owner did not exit");
    return -1;
}

// gone: no such process, or a zombie only its new parent has still to reap
bool gone(int pid) {
    if (pid <= 0)
        return false;
    if (::kill(pid, 0) != 0 && errno == ESRCH)
        return true;
    std::ifstream in("/proc/" + std::to_string(pid) + "/stat");
    std::string stat;
    std::getline(in, stat);
    const size_t close = stat.rfind(')');
    return close != std::string::npos && close + 2 < stat.size() && stat[close + 2] == 'Z';
}

std::string text(const fs::path& p) {
    std::ifstream in(p);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

bool any_error_contains(const Json& record, const std::string& needle) {
    for (const auto& e : record.value("errors", Json::array()))
        if (e.get<std::string>().find(needle) != std::string::npos)
            return true;
    return false;
}

void feature_state_start_status_and_warning() {
    // A real engine-only native start: the owner records its host facts before any child, publishes the feature
    // states of the sealed manifest (status and file agree), and reports that no calibration applies.
    Case c("feature-state");
    c.capture_owner_log = true;
    c.session.publish_feature_states = true;
    const pid_t owner = c.owner();
    c.wait_state({"running"});
    const Json host_start = Json::parse(text(c.run / "host-facts.start.v1.json"));
    assert(host_start.at("schema_version") == 1);
    assert(host_start.at("capture_phase") == "owner_start");
    assert(host_start.at("captured_at").is_string());
    assert(host_start.at("facts").at("kernel_release").at("readable").is_boolean());
    {
        const auto& build = host_start.at("facts").at("kernel_build");
        if (build.at("readable").get<bool>())
            assert(!std::regex_search(build.at("value").get<std::string>(),
                                      std::regex("\\([^()\\s@]+@[^()\\s]+\\)")));
    }
    assert(host_start.at("facts").at("mem_available_kib").at("readable").is_boolean());
    assert(host_start.at("facts").at("cards").is_object());
    const auto reply = ss::control(c.run, "status", 2s);
    assert(reply && reply->at("state") == "running");
    const Json features = reply->at("feature_states");
    assert(features.is_array() && features.size() == 7);
    assert(Json::parse(text(c.run / "feature-states.v1.json")) == features);
    assert(features.at(0).at("feature_id") == "live_policy");
    assert(features.at(0).at("state") == "inactive");
    assert(features.at(0).at("reason_code") == "no_helpers");
    assert(features.at(1).at("feature_id") == "mtp" && features.at(1).at("state") == "active");
    assert(reply->at("calibration_record").at("status") == "none");
    assert(reply->at("calibration_record").at("reason_code") == "no_helpers");
    assert(text(c.run / "owner.log").find("WARNING production feature") == std::string::npos); // engine-only: none
    const auto stopped = ss::control(c.run, "stop", 10s);
    assert(stopped && stopped->value("ok", false));
    assert(exit_code(owner) == 0);
    assert(c.record().at("feature_states") == features);

    // A helper-backed manifest whose live policy is inactive: the owner warns in its log, reports the calibration
    // reason, and strict production admission refuses before any child (the plan's manifest is replaced by a
    // helper-backed document; nothing past the refusal reads it as a launch manifest).
    const Json helper_manifest = {
        {"spec",
         {{"intent", {{"service", {{"profile", "production"}, {"max_context_tokens", 32768}}}}},
          {"resources", {{"host", {{"pool_mib", 1024}, {"kv_mib", 4096}}}}},
          {"processes",
           {{"helpers", Json::array({{{"role", "helper0"}}})},
            {"engine", {{"device", {{"gpus", Json::array({"0000:03:00.0"})}}},
                        {"execution", {{"num_speculative_tokens", 2}, {"ngram_placement", "disk"}}},
                        {"vision", {{"vision_tower_placement", "gpu"}, {"mm_max_patches", 64}}}}},
            {"helper_common", {{"combine", {{"enabled", false}}}}}}},
          {"handoff", {{"placement", {{"live_policy_state", {{"requested", "off"}, {"configured", false},
                                                              {"reason_code", "calibration_key_absent"},
                                                              {"reason_detail", "no matching calibration"}}}}}}}}},
        {"provenance", {{"/spec/handoff/placement", {{"source", "live policy disabled: orphaned key"}}}}}};
    for (bool strict : {false, true}) {
        Case h(strict ? "feature-state-strict" : "feature-state-helpers");
        h.capture_owner_log = true;
        h.session.publish_feature_states = true;
        h.session.strict_production_features = strict;
        h.prepare();
        h.prepared.pinned_manifest.reset();
        h.prepared.manifest = h.run / "helper-manifest.json";
        std::ofstream(h.prepared.manifest) << helper_manifest.dump();
        const pid_t o = h.owner();
        assert(exit_code(o) != 0); // strict: refused here; otherwise the manifest is no native launch manifest
        const auto failed = h.record();
        assert(failed.at("state") == "failed" && failed.at("processes").empty());
        assert(failed.at("feature_states").is_array() && failed.at("feature_states").at(0).at("reason_code") ==
                                                              "calibration_key_absent");
        assert(failed.at("calibration_record").at("reason_code") == "calibration_key_absent");
        const std::string log = text(h.run / "owner.log");
        assert(log.find("WARNING production feature live_policy inactive") != std::string::npos);
        assert(log.find("calibration_key_absent") != std::string::npos &&
               log.find("orphaned key") != std::string::npos);
        assert((log.find("strict production feature refusal") != std::string::npos) == strict);
    }
}

void feature_state_invalid_manifest_prevents_child() {
    Case c("feature-invalid");
    c.session.publish_feature_states = true;
    c.prepare();
    c.prepared.pinned_manifest.reset();
    c.prepared.manifest = c.run / "not-a-manifest.json";
    std::ofstream(c.prepared.manifest) << "{}";
    const pid_t owner = c.owner();
    assert(exit_code(owner) != 0);
    assert(c.record().at("state") == "failed");
    assert(c.record().at("processes").empty());
    assert(!fs::exists(c.run / "feature-states.v1.json"));
}

void serve_status_stop() {
    Case c("serve");
    c.behaviour["env_dump"] = (c.run / "engine.env").string();
    c.behaviour["health_delay_ms"] = 200;
    // inherited configuration and device choices must not reach the engine
    ::setenv("R3_MOE_FP16", "1", 1);
    ::setenv("BC_MODEL_ROOT", "/elsewhere", 1);
    ::setenv("BIGCHERRY_CONFIGS", "/elsewhere", 1);
    ::setenv("HIP_VISIBLE_DEVICES", "2,3", 1);
    ::setenv("RSC02_TEST_KEEP", "yes", 1);
    const pid_t owner = c.owner();
    c.wait_state({"running"});
    // the record: every stage in order, the engine started from the plan's manifest
    std::vector<std::string> kinds;
    const Json running_record = c.record();
    for (const auto& e : running_record.at("events"))
        kinds.push_back(e.at("kind").get<std::string>());
    const std::vector<std::string> want = {"created", "planning",      "planned", "started",
                                           "healthy", "canary_passed", "running"};
    assert(kinds == want);
    // the start-phase costs are in the record, written with the events (engine_start_to_ready is saved at "healthy")
    assert(running_record.contains("plan_reuse") && running_record.at("phases_ms").is_object() &&
           running_record.at("phases_ms").contains("helper_start") &&
           running_record.at("phases_ms").at("engine_start_to_ready").is_number_integer());
    const auto status = ss::control(c.run, "status", 2s);
    assert(status && status->value("ok", false) && status->at("state") == "running" &&
           status->at("port") == c.port);
    assert(status->at("owner_pid") == owner);
    const std::string env = text(c.run / "engine.env");
    assert(env.find("R3_MOE_FP16") == std::string::npos && env.find("BC_MODEL_ROOT") == std::string::npos &&
           env.find("BIGCHERRY_CONFIGS") == std::string::npos);
    // the admitted plan's own device visibility replaces the inherited choice; the runtime home is the build's
    assert(env.find("HIP_VISIBLE_DEVICES=0,1,2\n") != std::string::npos &&
           env.find("HIP_VISIBLE_DEVICES=2,3") == std::string::npos);
    assert(env.find("ROCR_VISIBLE_DEVICES=0,1,2\n") != std::string::npos &&
           env.find("\nRADIANCE_HOME=/runtime\n") == std::string::npos &&
           env.find("\nRADIANCE_HOME=/") != std::string::npos);
    assert(env.find("RSC02_TEST_KEEP=yes\n") != std::string::npos); // the runtime's own environment stays
    for (const char* v :
         {"R3_MOE_FP16", "BC_MODEL_ROOT", "BIGCHERRY_CONFIGS", "HIP_VISIBLE_DEVICES", "RSC02_TEST_KEEP"})
        ::unsetenv(v);
    // the engine runs in its own process group (stop signals the group, never the owner's)
    const int engine = c.engine_pid();
    assert(engine > 0 && ::getpgid(engine) == engine);
    // a second owner in the same run directory is refused while this one runs, and changes nothing
    {
        Case second("second");
        second.session.run_dir = c.run;
        bool planned = false;
        second.session.plan = [&](const fs::path&, const ss::Hooks&) -> ss::EngineStart {
            planned = true;
            return {};
        };
        assert(exit_code(second.owner()) == 98 && !planned);
        assert(c.record().at("state") == "running" && c.record().at("owner_pid") == owner);
    }
    // stop over the socket: answered once the engine has exited
    const auto stopped = ss::control(c.run, "stop", 10s);
    assert(stopped && stopped->value("ok", false) && stopped->at("state") == "stopped");
    assert(gone(engine));
    assert(exit_code(owner) == 0);
    assert(c.record().at("state") == "stopped" && c.record().at("errors").empty());
    assert(!fs::exists(c.run / "session.sock"));
    // nobody answers now; a new session in the same directory starts, keeping the last record
    assert(!ss::control(c.run, "status", 1s));
    const pid_t again = c.owner();
    c.wait_state({"running"});
    assert(fs::exists(c.run / "session.previous.json"));
    assert(ss::control(c.run, "stop", 10s)->value("ok", false));
    assert(exit_code(again) == 0);
}

// A run directory whose socket path is longer than sun_path (107 bytes) still serves, answers and stops: the
// address goes through a descriptor open on the directory.
void long_run_dir() {
    Case c("long-" + std::string(120, 'd'));
    assert((c.run / "session.sock").string().size() > 108);
    const pid_t owner = c.owner();
    c.wait_state({"running"});
    const auto status = ss::control(c.run, "status", 2s);
    assert(status && status->at("state") == "running" && status->at("owner_pid") == owner);
    assert(fs::exists(c.run / "session.sock"));
    const auto stopped = ss::control(c.run, "stop", 10s);
    assert(stopped && stopped->value("ok", false) && stopped->at("state") == "stopped");
    assert(exit_code(owner) == 0);
    assert(!fs::exists(c.run / "session.sock"));
}

// A run directory with any mode serves, and a leftover socket file of a crashed owner is simply replaced.
void shared_run_dir_and_stale_socket() {
    Case c("shared-dir");
    c.prepare();
    ::chmod(c.run.c_str(), 0755);
    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    assert(fd >= 0);
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    const int dir = ::open(c.run.c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC); // test paths exceed sun_path
    assert(dir >= 0);
    const std::string via = "/proc/self/fd/" + std::to_string(dir) + "/session.sock";
    std::strncpy(a.sun_path, via.c_str(), sizeof a.sun_path - 1);
    assert(::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0);
    ::close(dir);
    ::close(fd); // the file stays, nobody listens
    const pid_t owner = c.owner();
    c.wait_state({"running"});
    assert(ss::control(c.run, "status", 2s));
    assert(ss::control(c.run, "stop", 10s)->value("ok", false));
    assert(exit_code(owner) == 0);
}

void canary_fails() {
    Case c("canary");
    c.behaviour["answer"] = "Lyon, of course.";
    const pid_t owner = c.owner();
    assert(exit_code(owner) == 96);
    const Json r = c.record();
    assert(r.at("state") == "failed" && any_error_contains(r, "start canary failed") &&
           any_error_contains(r, "Lyon"));
    assert(gone(c.engine_pid()));
}

void engine_fails_to_start() {
    Case c("no-start");
    c.behaviour["exit_at_start"] = 3;
    assert(exit_code(c.owner()) == 1);
    const Json r = c.record();
    assert(r.at("state") == "failed" && any_error_contains(r, "health gate") &&
           any_error_contains(r, "exit 3"));
}

// The engine's structured host-room refusal (typed admission-refused exit, before weights) is a named resource refusal:
// plan-class exit code, no restart, the cause in the record; the owner still stops cleanly.
void engine_refuses_host_room() {
    Case c("room-refused");
    c.behaviour["exit_at_start"] = 4;
    assert(exit_code(c.owner()) == 97);
    const Json r = c.record();
    assert(r.at("state") == "failed" && any_error_contains(r, "refused its host memory plan before loading weights") &&
           any_error_contains(r, "exit 4"));
}

// A start that took its plan from the work root's store and whose engine then does not come up drops the entry, so
// the next start measures afresh; a start that measured (or came up) leaves the store alone.
void engine_refuses_the_stored_plan() {
    Case c("stored-refused");
    const fs::path entry = c.run / "stored-entry.json";
    std::ofstream(entry) << "{}\n";
    c.behaviour["exit_at_start"] = 3;
    c.prepare();   // the real plan step runs here; the stored-plan mark is set on what it produced
    c.prepared.reused_store = entry;
    assert(exit_code(c.owner()) == 1);
    assert(!fs::exists(entry));
    Case measured("measured-refused");
    const fs::path kept = measured.run / "stored-entry.json";
    std::ofstream(kept) << "{}\n";
    measured.behaviour["exit_at_start"] = 3;
    assert(exit_code(measured.owner()) == 1);
    assert(fs::exists(kept));
    // A host-room refusal keeps a stored plan that was fine: the host had less memory just now.
    Case room("room-keeps-store");
    const fs::path room_entry = room.run / "stored-entry.json";
    std::ofstream(room_entry) << "{}\n";
    room.behaviour["exit_at_start"] = 4;
    room.prepare();
    room.prepared.reused_store = room_entry;
    assert(exit_code(room.owner()) == 97);
    assert(fs::exists(room_entry));
}

void engine_never_healthy() {
    Case c("unhealthy");
    c.behaviour["health_delay_ms"] = 60000;
    c.session.timing.health_timeout = 500ms;
    assert(exit_code(c.owner()) == 1);
    assert(any_error_contains(c.record(), "did not answer /health"));
    assert(gone(c.engine_pid()));
}

void engine_dies_while_serving() {
    Case c("dies");
    c.behaviour["die_after_ms"] = 600;
    const pid_t owner = c.owner();
    c.wait_state({"running", "failed"});
    assert(exit_code(owner) == 1);
    const Json r = c.record();
    assert(r.at("state") == "failed" && any_error_contains(r, "exited while serving") &&
           any_error_contains(r, "exit 7"));
}

void engine_ignores_sigint() {
    Case c("stubborn");
    c.behaviour["ignore_sigint"] = true;
    const pid_t owner = c.owner();
    c.wait_state({"running"});
    const int engine = c.engine_pid();
    const auto t0 = std::chrono::steady_clock::now();
    const auto reply = ss::control(c.run, "stop", 10s);
    assert(reply && reply->value("ok", false)); // SIGKILL after the grace, and its exit verified
    assert(std::chrono::steady_clock::now() - t0 >= 900ms);
    assert(gone(engine) && exit_code(owner) == 0);
}

void plan_refused() {
    Case c("plan");
    c.session.plan = [](const fs::path&, const ss::Hooks&) -> ss::EngineStart {
        throw ss::PlanFailed("plan: no measured manifest (test)");
    };
    assert(exit_code(c.owner()) == 97);
    const Json r = c.record();
    assert(r.at("state") == "failed" && r.at("processes").empty() &&
           any_error_contains(r, "no measured manifest"));
}

void stop_while_waiting_for_health() {
    Case c("stop-early");
    c.behaviour["health_delay_ms"] = 60000;
    c.session.timing.health_timeout = 60s;
    const pid_t owner = c.owner();
    c.wait_state({"started"});
    const int engine = c.engine_pid();
    const auto reply = ss::control(c.run, "stop", 10s);
    assert(reply && reply->value("ok", false));
    assert(gone(engine) && exit_code(owner) == 0 && c.record().at("state") == "stopped");
}

void stop_during_plan() {
    // a long plan step (a measuring start) waits through the owner, so a stop request ends it
    Case c("stop-plan");
    c.session.plan = [](const fs::path&, const ss::Hooks& h) -> ss::EngineStart {
        while (true) {
            auto result = h.wait.invoke("session.test.wait", 50ms);
            assert(result);
            if (!result.value())
                break;
        }
        throw ss::PlanStopped("plan: stopped while measuring");
    };
    const pid_t owner = c.owner();
    c.wait_state({"planning"});
    assert(ss::control(c.run, "status", 2s)->at("state") == "planning");
    const auto reply = ss::control(c.run, "stop", 10s);
    assert(reply && reply->value("ok", false));
    assert(exit_code(owner) == 0 && c.record().at("state") == "stopped" &&
           c.record().at("errors").empty());
    assert(c.record().at("processes").empty());
}

void signal_stops() {
    Case c("signal");
    const pid_t owner = c.owner();
    c.wait_state({"running"});
    const int engine = c.engine_pid();
    ::kill(owner, SIGTERM);
    assert(exit_code(owner) == 0);
    assert(gone(engine));
    const Json r = c.record();
    assert(r.at("state") == "stopped");
    bool by_signal = false;
    for (const auto& e : r.at("events"))
        if (e.at("kind") == "stop_requested")
            by_signal = e.at("by") == "signal 15";
    assert(by_signal);
}

void unknown_command_and_no_owner() {
    Case c("control");
    assert(!ss::control(c.run, "status", 500ms)); // nothing there: no owner answers
    const pid_t owner = c.owner();
    c.wait_state({"running"});
    const auto r = ss::control(c.run, "restart", 2s);
    assert(r && !r->value("ok", true));
    assert(c.record().at("state") == "running");
    ss::control(c.run, "stop", 10s);
    assert(exit_code(owner) == 0);
}

void spawn_and_environment() {
    // exec failure is the caller's error, not a silent child
    bool threw = false;
    try {
        ss::spawn({"/nonexistent/engine"}, {}, g_root / "spawn.log");
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("/nonexistent/engine") != std::string::npos;
    }
    assert(threw);
    // the start gate: the child runs nothing until it is recorded, and nothing at all when recording fails
    {
        const fs::path ran = g_root / "gate-ran";
        bool ran_before_recorded = true;
        const pid_t pid = ss::spawn({"/bin/sh", "-c", "touch " + ran.string()}, {{"PATH", "/bin:/usr/bin"}},
                                    g_root / "spawn.log", [&](pid_t) {
                                        std::this_thread::sleep_for(200ms);
                                        ran_before_recorded = fs::exists(ran);
                                    });
        int st = 0;
        ::waitpid(pid, &st, 0);
        assert(!ran_before_recorded && fs::exists(ran));
        const fs::path never = g_root / "gate-never";
        bool refused = false;
        try {
            ss::spawn({"/bin/sh", "-c", "touch " + never.string()}, {{"PATH", "/bin:/usr/bin"}},
                      g_root / "spawn.log", [](pid_t) { throw std::runtime_error("record failed"); });
        } catch (const std::runtime_error& e) {
            refused = std::string(e.what()).find("record failed") != std::string::npos;
        }
        std::this_thread::sleep_for(200ms);
        assert(refused && !fs::exists(never));
        for (bool foreign : {false, true}) {
            const fs::path forbidden = g_root / (foreign ? "nonstd-gate" : "std-gate");
            bool typed = false;
            try {
                ss::spawn(
                    {"/bin/sh", "-c", "touch " + forbidden.string()}, {{"PATH", "/bin:/usr/bin"}},
                    g_root / "spawn.log", [foreign](pid_t) {
                        if (foreign)
                            throw 17; // NOLINT(bugprone-std-exception-baseclass): deliberate nonstandard throw
                        throw std::runtime_error("recording callback failure");
                    });
            } catch (const bc::serving::callback::Failure& err) {
                typed = err.error().code == bc::serving::v2::ConfigErrorCode::io_failure;
            }
            assert(typed && !fs::exists(forbidden));
        }
        // A retained root descriptor pins the child cwd even if its pathname is replaced after registration.
        // Output paths remain absolute in the owner's attempt directory, and the root descriptor is CLOEXEC.
        {
            const fs::path root = g_root / "native-execution-root";
            const fs::path moved = g_root / "native-execution-root-moved";
            const fs::path attempt = g_root / "native-attempt";
            fs::create_directories(root);
            fs::create_directories(attempt);
            std::ofstream(root / "identity") << "original";
            const int root_fd = ::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
            assert(root_fd >= 3);
            const fs::path owner_cwd = fs::current_path();
            const fs::path cwd_output = attempt / "child-cwd";
            const fs::path plan_output = attempt / "plan.json";
            const fs::path ranks_output = attempt / "rank-demand.json";
            const std::string command =
                "pwd > \"$CWD_OUTPUT\" && test \"$(cat identity)\" = original && "
                "touch child-ran && printf plan > \"$PLAN_OUTPUT\" && "
                "printf ranks > \"$RANKS_OUTPUT\" && "
                "test ! -e \"/proc/self/fd/$ROOT_FD\"";
            bool moved_before_child = false;
            const pid_t pid = ss::spawn(
                {"/bin/sh", "-c", command},
                {{"PATH", "/bin:/usr/bin"}, {"CWD_OUTPUT", cwd_output.string()},
                 {"PLAN_OUTPUT", plan_output.string()}, {"RANKS_OUTPUT", ranks_output.string()},
                 {"ROOT_FD", std::to_string(root_fd)}},
                g_root / "spawn.log",
                [&](pid_t) {
                    fs::rename(root, moved);
                    fs::create_directories(root);
                    std::ofstream(root / "identity") << "replacement";
                    moved_before_child = true;
                }, -1, root_fd);
            int st = 0;
            assert(::waitpid(pid, &st, 0) == pid && WIFEXITED(st) && WEXITSTATUS(st) == 0);
            assert(moved_before_child && fs::current_path() == owner_cwd);
            std::ifstream cwd_in(cwd_output);
            std::string child_cwd;
            std::getline(cwd_in, child_cwd);
            assert(fs::path(child_cwd) == moved && fs::exists(moved / "child-ran"));
            assert(!fs::exists(root / "child-ran"));
            std::ifstream plan_in(plan_output), ranks_in(ranks_output);
            std::string plan_text, ranks_text;
            plan_in >> plan_text;
            ranks_in >> ranks_text;
            assert(plan_text == "plan" && ranks_text == "ranks");
            ::close(root_fd);
        }
        // a writer that exits at once is registered before it can exit (the gate), so registration never
        // fails
        std::vector<pid_t> seen;
        ss::Hooks quick;
        quick.wait = [](std::chrono::milliseconds) { return true; };
        quick.started = [&](ss::ChildKind, const std::string&, pid_t registered_pid, const fs::path&) {
            std::this_thread::sleep_for(100ms);
            seen.push_back(registered_pid);
            assert(ss::start_time(registered_pid)); // still waiting at the gate: alive, not a zombie
        };
        assert(ss::run_to_completion({"/bin/true"}, {}, quick).exit == 0 && seen.size() == 1);
    }
    // a manifest write is waited for through the owner: output kept, and a stop request ends a slow one at once
    std::vector<pid_t> registered, verified;
    ss::Hooks hooks;
    hooks.wait = [](std::chrono::milliseconds) { return true; };
    hooks.started = [&](ss::ChildKind, const std::string& role, pid_t pid, const fs::path&) {
        assert(role == "native config inspector");
        registered.push_back(pid);
    };
    hooks.ended = [&](pid_t pid) { verified.push_back(pid); };
    const auto echoed =
        ss::run_to_completion({"/bin/sh", "-c", "echo out; echo err >&2; exit 3"}, {{"PATH", "/bin"}}, hooks);
    assert(echoed.exit == 3 && echoed.out == "out\n" && echoed.err == "err\n");
    assert(registered.size() == 1 && registered == verified); // registered with the owner, its exit verified
    for (bool foreign : {false, true}) {
        ss::Hooks bad = hooks;
        bad.wait = [foreign](std::chrono::milliseconds) -> bool {
            if (foreign)
                throw 17; // NOLINT(bugprone-std-exception-baseclass): deliberate nonstandard throw
            throw std::runtime_error("wait exception");
        };
        bool typed = false;
        const auto before = verified.size();
        try {
            ss::run_to_completion({"/bin/sleep", "30"}, {}, bad);
        } catch (const bc::serving::callback::Failure& err) {
            typed = err.error().code == bc::serving::v2::ConfigErrorCode::io_failure;
        }
        assert(typed && verified.size() == before + 1);

        bad = hooks;
        pid_t just_spawned = -1;
        bad.started = [&](ss::ChildKind, const std::string&, pid_t pid, const fs::path&) {
            just_spawned = pid;
            if (foreign)
                throw 17; // NOLINT(bugprone-std-exception-baseclass): deliberate nonstandard throw
            throw std::runtime_error("started exception");
        };
        typed = false;
        try {
            ss::run_to_completion({"/bin/true"}, {}, bad);
        } catch (const bc::serving::callback::Failure& err) {
            typed = err.error().code == bc::serving::v2::ConfigErrorCode::io_failure;
        }
        assert(typed && just_spawned > 0 && !ss::running(just_spawned));

        bad = hooks;
        bad.ended = [foreign](pid_t) {
            if (foreign)
                throw 17; // NOLINT(bugprone-std-exception-baseclass): deliberate nonstandard throw
            throw std::runtime_error("ended exception");
        };
        typed = false;
        try {
            ss::run_to_completion({"/bin/true"}, {}, bad);
        } catch (const bc::serving::callback::Failure& err) {
            typed = err.error().code == bc::serving::v2::ConfigErrorCode::io_failure;
        }
        assert(typed);
    }
    const size_t extra_verified = verified.size() - 1U;
    int waits = 0;
    hooks.wait = [&](std::chrono::milliseconds) { return ++waits < 3; };
    const auto t0 = std::chrono::steady_clock::now();
    bool stopped = false;
    try {
        ss::run_to_completion({"/bin/sleep", "30"}, {}, hooks);
    } catch (const ss::PlanFailed&) {
        stopped = true;
    }
    assert(stopped && waits == 3 && std::chrono::steady_clock::now() - t0 < 5s &&
           verified.size() == 2U + extra_verified);
    // a writer that leaves a descendant ignoring SIGTERM: its whole group goes (SIGKILL after the grace) before
    // it returns
    const fs::path child_pid = g_root / "writer-descendant.pid";
    waits = 0;
    hooks.wait = [&](std::chrono::milliseconds d) {
        std::this_thread::sleep_for(d);
        return ++waits < 5;
    };
    stopped = false;
    try {
        ss::run_to_completion(
            {"/bin/sh", "-c", "trap '' TERM INT; /bin/sleep 60 & echo $! > " + child_pid.string() + "; wait"},
            {{"PATH", "/bin"}}, hooks);
    } catch (const ss::PlanFailed&) {
        stopped = true;
    }
    assert(stopped && verified.size() == 3U + extra_verified && gone(std::stoi(text(child_pid))));
    char a[] = "PATH=/bin", b[] = "RAD_X=1", c[] = "BC_Y=2", d[] = "ROCR_VISIBLE_DEVICES=1", e[] = "HOME=/h";
    char* inherited[] = {a, b, c, d, e, nullptr};
    const auto env = ss::child_environment(inherited, {{"ROCR_VISIBLE_DEVICES", "4"}});
    assert(env == (ss::Env{{"PATH", "/bin"}, {"HOME", "/h"}, {"ROCR_VISIBLE_DEVICES", "4"}}));
}

void start_detached_runs_and_stops() {
    // `start`: the owner detached; returns once running, and `stop` ends it through the socket
    Case c("detached");
    // the detached owner is this test binary in its owner mode (main below), as `bc-serve start` re-runs
    // `serve`
    const std::vector<std::string> self = {fs::read_symlink("/proc/self/exe").string(), "--owner",
                                           c.run.string(), std::to_string(c.port), g_host.string(), "{}"};
    assert(ss::start_detached(self, c.run, 60s) == 0);
    const auto status = ss::control(c.run, "status", 2s);
    assert(status && status->at("state") == "running");
    const int owner = status->at("owner_pid");
    assert(ss::control(c.run, "stop", 10s)->value("ok", false));
    // (the detached owner is this test's child, as it is the `start` process's until that returns: reap it)
    for (int i = 0; i < 200 && ss::running(owner); ++i)
        std::this_thread::sleep_for(20ms);
    assert(!ss::running(owner) && gone(owner));
    // a start whose session fails reports the failure
    Case f("detached-fail");
    assert(ss::start_detached({self[0], "--owner", f.run.string(), std::to_string(f.port), g_host.string(),
                               Json({{"exit_at_start", 3}}).dump()},
                              f.run, 60s) == 1);
}

void descendant_outliving_the_engine() {
    // the engine exits on SIGINT but a process it forked ignores it: stop is not done until the whole group is
    // gone
    Case c("descendant");
    c.behaviour["descendant"] = (c.run / "descendant.pid").string();
    const pid_t owner = c.owner();
    c.wait_state({"running"});
    const int descendant = std::stoi(text(c.run / "descendant.pid"));
    assert(!gone(descendant) && ::getpgid(descendant) == c.engine_pid());
    const auto reply = ss::control(c.run, "stop", 10s);
    assert(reply && reply->value("ok", false));
    for (int i = 0; i < 100 && !gone(descendant); ++i)
        std::this_thread::sleep_for(20ms);
    assert(gone(descendant) && exit_code(owner) == 0);
}

void inherited_descriptor_not_passed() {
    // a descriptor the owner holds without close-on-exec never reaches the engine
    Case c("fds");
    const fs::path sentinel = c.run / "sentinel.txt";
    const int fd = ::open(sentinel.c_str(), O_WRONLY | O_CREAT, 0600); // inheritable on purpose
    assert(fd > 2);
    c.behaviour["fd_dump"] = (c.run / "engine.fds").string();
    const pid_t owner = c.owner();
    ::close(fd);
    c.wait_state({"running"});
    const std::string fds = text(c.run / "engine.fds");
    assert(!fds.empty() && fds.find("sentinel.txt") == std::string::npos &&
           fds.find("session.lock") == std::string::npos && fds.find("session.sock") == std::string::npos);
    ss::control(c.run, "stop", 10s);
    assert(exit_code(owner) == 0);
}

void owner_killed_fails_closed() {
    // an owner killed with SIGKILL leaves its engine: stop says so (fails), status warns, a new owner refuses
    // the cards
    Case c("killed");
    const pid_t owner = c.owner();
    c.wait_state({"running"});
    const int engine = c.engine_pid();
    ::kill(owner, SIGKILL);
    exit_code(owner);
    assert(!gone(engine) && !ss::control(c.run, "status", 1s));
    const auto alive = ss::alive_recorded(c.record());
    assert(alive.size() == 1 && alive[0].at("pid") == engine);
    Case next("killed-next");
    next.session.run_dir = c.run;
    bool planned = false;
    next.session.plan = [&](const fs::path&, const ss::Hooks&) -> ss::EngineStart {
        planned = true;
        return {};
    };
    assert(exit_code(next.owner()) == 98 && !planned);
    // its leader gone, a process left in its group still counts (a descendant the engine forked)
    ::kill(-engine, SIGKILL);
    for (int i = 0; i < 100 && !gone(engine); ++i)
        std::this_thread::sleep_for(20ms);
    {
        Case d("descendant-left");
        d.behaviour["descendant"] = (d.run / "descendant.pid").string();
        const pid_t o = d.owner();
        d.wait_state({"running"});
        const int leader = d.engine_pid();
        const int descendant = std::stoi(text(d.run / "descendant.pid"));
        ::kill(o, SIGKILL);
        exit_code(o);
        ::kill(leader, SIGKILL); // the leader only: the descendant stays in its group
        for (int i = 0; i < 100 && !gone(leader); ++i)
            std::this_thread::sleep_for(20ms);
        assert(gone(leader) && !gone(descendant));
        const auto left = ss::alive_recorded(d.record());
        assert(left.size() == 1 && left[0].at("pid") == leader);
        ::kill(descendant, SIGKILL);
        for (int i = 0; i < 100 && !gone(descendant); ++i)
            std::this_thread::sleep_for(20ms);
        assert(ss::alive_recorded(d.record()).empty());
    }
    for (int i = 0; i < 100 && !gone(engine); ++i)
        std::this_thread::sleep_for(20ms);
    assert(ss::alive_recorded(c.record()).empty());
    Case after("killed-after");
    after.session.run_dir = after.run = c.run;
    after.port = c.port;
    const pid_t again = after.owner();
    for (int i = 0; i < 500 && !(after.record().value("owner_pid", 0) == again &&
                                 after.record().value("state", "") == "running");
         ++i)
        std::this_thread::sleep_for(20ms); // the killed owner's record says running too: wait for this owner's
    assert(after.record().value("owner_pid", 0) == again);
    assert(ss::control(c.run, "stop", 10s)->value("ok", false));
    assert(exit_code(again) == 0);
}

void injected_owner_plan_exception() {
    for (bool foreign : {false, true}) {
        Case c(foreign ? "nonstandard-plan" : "standard-plan");
        c.session.plan = [foreign](const fs::path&, const ss::Hooks&) -> ss::EngineStart {
            if (foreign)
                throw 17; // NOLINT(bugprone-std-exception-baseclass): deliberate nonstandard throw
            throw std::runtime_error("injected plan exception");
        };
        assert(exit_code(c.owner()) != 0);
        const auto record = c.record();
        assert(record.value("state", "") == "failed");
        assert(any_error_contains(record, "session.plan"));
        assert(record.value("processes", Json::array()).empty());
    }
}

void no_record_no_session() {
    // the record cannot be written at all: no session starts (nothing could find its processes later)
    Case c("no-record");
    fs::create_directories(c.run /
                           "session.json.tmp"); // the atomic write's temporary is a directory: writes fail
    assert(exit_code(c.owner()) == 1 && !fs::exists(c.run / "session.json"));
    for (const auto& e : fs::directory_iterator(c.run))
        assert(e.path().filename().string().rfind("engine-", 0) != 0);
}

void unwritable_record_still_stops() {
    // A live engine is still reaped, and power restored only after verified exit, even when
    // the terminal event cannot be made durable. The owner must not acknowledge success.
    Case c("readonly");
    const fs::path marker = c.run.parent_path() / (c.run.filename().string() + "-power.marker");
    const fs::path restored = c.run.parent_path() / (c.run.filename().string() + "-restored.txt");
    c.session.power_hook.apply = [marker]() -> bc::serving::v2::Status {
        std::ofstream(marker) << "applied";
        return std::nullopt;
    };
    c.session.power_hook.restore = [marker, restored, &c]() -> bc::serving::v2::Status {
        const int pid = c.engine_pid();
        std::ofstream(restored) << (gone(pid) ? "verified" : "live") << '\n';
        fs::remove(marker);
        return std::nullopt;
    };
    const pid_t owner = c.owner();
    c.wait_state({"running"});
    const int engine = c.engine_pid();
    assert(engine > 0 && fs::exists(marker));
    // The atomic record writer opens this temporary path before rename. A directory
    // cannot be opened for writing even by root (unlike a chmod-protected run dir).
    const fs::path blocked_tmp = c.run / "session.json.tmp";
    if (!fs::create_directory(blocked_tmp))
        throw std::runtime_error("cannot install atomic record write failure");
    const int probe = ::open(blocked_tmp.c_str(), O_WRONLY | O_TRUNC);
    if (probe >= 0)
        ::close(probe);
    assert(probe == -1 && errno == EISDIR);
    const auto reply = ss::control(c.run, "stop", 10s);
    const int result = exit_code(owner);
    assert(gone(engine) && !fs::exists(marker) && text(restored) == "verified\n");
    assert(result != 0 && reply && !reply->value("ok", true));
    assert(reply->value("state", "") == "failed");
    assert(reply->value("error", "").find("persist") != std::string::npos);
    const Json durable = c.record();
    assert(durable.value("state", "") == "running");
    for (const auto& e : durable.at("events"))
        assert(e.at("kind") != "stopped");
}

// Port of Python supervisor order, helper readiness and failure cases (all card-free).
// The calibration ingest is off by default, and enabled it can never fail serving: the fake manifest here is not a strict
// serving manifest, so the owner cannot name an identity to store under; it warns and keeps serving.
void calibration_ingest_never_fails_serving() {
    for (const bool enabled : {false, true}) {
        Case c(enabled ? "calibration-ingest-on" : "calibration-ingest-off");
        c.helpers = 1;
        c.capture_owner_log = true;
        const fs::path store = c.run / "calibration.toml";
        if (enabled) c.session.calibration_ingest_store = store;
        c.owner();
        c.wait_state({"running"});
        // The helper's marker arrives while serving: nothing is stored and nothing fails.
        const Json r = c.record();
        for (const auto& p : r.at("processes"))
            if (p.at("role") == "helper0")
                std::ofstream(p.at("log").get<std::string>(), std::ios::app) << "RHP16_CALIBRATION_JSON {}\n";
        std::this_thread::sleep_for(600ms);
        const auto reply = ss::control(c.run, "status", 2s);
        assert(reply && reply->at("state") == "running");
        for (const auto& e : c.record().at("events"))
            assert(e.at("kind") != "calibration_ingest" && e.at("kind") != "calibration_watch");
        assert(!fs::exists(store));
        if (enabled)
            assert(text(c.run / "owner.log").find("calibration ingest not watching") != std::string::npos);
        else
            assert(text(c.run / "owner.log").find("calibration ingest") == std::string::npos);
        assert(ss::control(c.run, "stop", 10s)->value("ok", false));
    }
}

void helpers_order_and_killed() {
    Case c("helper-order");
    c.helpers = 2;
    c.behaviour["helpers"] = {
        {"helper0", {{"ready_delay_ms", 250}, {"env_dump", (c.run / "h0.env").string()}}},
        {"helper1", {{"ready_delay_ms", 100}, {"env_dump", (c.run / "h1.env").string()}}},
    };
    const pid_t owner = c.owner();
    c.wait_state({"running"});
    const Json r = c.record();
    std::vector<std::string> events;
    std::vector<std::string> roles;
    for (const auto& e : r.at("events")) {
        const std::string kind = e.at("kind");
        if (kind == "started" || kind == "ready")
            events.push_back(kind + ":" + e.at("role").get<std::string>());
    }
    assert((events == std::vector<std::string>{"started:helper0", "ready:helper0", "started:helper1",
                                               "ready:helper1", "started:engine"}));
    for (const auto& p : r.at("processes"))
        roles.push_back(p.at("role").get<std::string>());
    assert((roles == std::vector<std::string>{"helper0", "helper1", "engine"}));
    const std::string env = text(c.run / "h1.env");
    assert(env.find("ROCR_VISIBLE_DEVICES=7\n") != std::string::npos &&
           env.find("HIP_VISIBLE_DEVICES=0\n") != std::string::npos);
    for (const auto& p : r.at("processes"))
        assert(!gone(p.at("pid")));
    // Kill helper 0: the owner must stop the engine and every other helper, in reverse start order.
    ::kill(r.at("processes").at(0).at("pid").get<int>(), SIGKILL);
    assert(exit_code(owner) == 1);
    const Json stopped = c.record();
    assert(stopped.at("state") == "failed" && any_error_contains(stopped, "helper0"));
    for (const auto& p : stopped.at("processes"))
        assert(gone(p.at("pid")));
}

void helper_failure_unwinds() {
    for (const auto& which : {"helper0", "helper1"}) {
        Case c(std::string("helper-fail-") + which);
        c.helpers = 2;
        c.behaviour["helpers"] = {{which, {{"exit_at_start", 23}}}};
        assert(exit_code(c.owner()) == 1);
        const Json r = c.record();
        assert(r.at("state") == "failed" && any_error_contains(r, which));
        for (const auto& p : r.at("processes"))
            assert(gone(p.at("pid")));
        for (const auto& p : r.at("processes"))
            assert(p.at("role") != "engine");
    }
    Case blocked("helper-timeout");
    blocked.helpers = 1;
    blocked.behaviour["helpers"] = {{"helper0", {{"never_ready", true}}}};
    blocked.session.timing.helper_timeout = 250ms;
    assert(exit_code(blocked.owner()) == 1);
    assert(any_error_contains(blocked.record(), "not ready"));
    const Json blocked_record = blocked.record();
    for (const auto& p : blocked_record.at("processes"))
        assert(gone(p.at("pid")));
    // A waiting-for-maps message is a valid helper readiness gate.
    Case waiting("helper-waiting-for-maps");
    waiting.helpers = 1;
    waiting.behaviour["helpers"] = {{"helper0", {{"waiting_for_maps", true}}}};
    const auto owner = waiting.owner();
    waiting.wait_state({"running"});
    const auto stop_reply = ss::control(waiting.run, "stop", 10s);
    assert(stop_reply.has_value() && stop_reply->value("ok", false));
    assert(exit_code(owner) == 0);
}

std::string fake_segment(int index) {
    return "/r3-serve-test-" + std::to_string(::getpid()) + "-" + std::to_string(index);
}
bool segment_exists(const std::string& name) {
    const int fd = ::shm_open(name.c_str(), O_RDONLY | O_CLOEXEC, 0);
    if (fd < 0)
        return false;
    struct stat st{};
    const bool good = ::fstat(fd, &st) == 0 && st.st_size > 0;
    ::close(fd);
    return good;
}
void helper_shared_segment_gate() {
    Case c("helper-shm");
    c.helpers = 2;
    c.behaviour["shm"] = fake_segment(g_case);
    const std::string name = c.behaviour.at("shm");
    const pid_t owner = c.owner();
    c.wait_state({"running"});
    assert(segment_exists(name));
    const auto stop_reply = ss::control(c.run, "stop", 10s);
    assert(stop_reply.has_value() && stop_reply->value("ok", false));
    assert(exit_code(owner) == 0);
    assert(!segment_exists(name));
    assert(c.record().at("handoff_shm") == name);

    Case missing("helper-no-segment");
    missing.helpers = 1;
    missing.behaviour["shm"] = fake_segment(g_case);
    missing.behaviour["helpers"] = {{"helper0", {{"no_shm", true}}}};
    assert(exit_code(missing.owner()) == 97); // missing segment is a PlanFailed (pre-engine admission)
    assert(any_error_contains(missing.record(), "shared segment"));
    const Json missing_record = missing.record();
    for (const auto& p : missing_record.at("processes")) {
        assert(gone(p.at("pid")));
        assert(p.at("role") != "engine");
    }

    // A worker SIGKILL skips its own unlink. The owner may retire only its
    // UUID-named segment and only after all helper/engine groups exited.
    Case killed("helper-shm-killed");
    killed.helpers = 2;
    killed.behaviour["shm"] = fake_segment(g_case);
    const std::string leftover = killed.behaviour.at("shm");
    const pid_t stopped = killed.owner();
    killed.wait_state({"running"});
    const Json r = killed.record();
    assert(segment_exists(leftover));
    ::kill(r.at("processes")[0].at("pid").get<pid_t>(), SIGKILL);
    assert(exit_code(stopped) == 1);
    assert(!segment_exists(leftover));
    const Json killed_record = killed.record();
    for (const auto& p : killed_record.at("processes"))
        assert(gone(p.at("pid")));
}

void callback_nonstandard_regression() {
    Case c("nonstandard-apply");
    const auto marker = c.run / "control";
    c.session.power_hook.apply = [marker]() -> bc::serving::v2::Status {
        std::ofstream(marker) << "changed";
        throw 17; // NOLINT(bugprone-std-exception-baseclass)
    };
    c.session.power_hook.restore = [marker]() -> bc::serving::v2::Status {
        fs::remove(marker);
        return std::nullopt;
    };
    const pid_t owner = c.owner();
    assert(exit_code(owner) != 0);
    const Json record = c.record();
    assert(record.value("state", "") == "failed" && !fs::exists(marker) &&
           any_error_contains(record, "callback"));
}

// No raw function escapes the typed wrapper; attempting direct invocation does
// not compile, even from a function taking a mutable Session or Hooks.
using ReturnedStatus = decltype([]() -> bc::serving::v2::Status { return std::nullopt; });
static_assert(!std::is_constructible_v<ss::Callback<void()>, ReturnedStatus>);
static_assert(!std::is_assignable_v<ss::Callback<void()>&, ReturnedStatus>);
static_assert(!std::is_constructible_v<ss::Callback<void()>, decltype([]() -> int { return 3; })>);
static_assert(std::is_constructible_v<ss::Callback<bc::serving::v2::Status()>, ReturnedStatus>);
static_assert(!std::is_invocable_v<ss::Callback<void()>&>);
static_assert(!std::is_invocable_v<ss::Wait&, std::chrono::milliseconds>);

struct UnusualCallbackException {};
enum class ThrowKind : uint8_t { standard, custom, integer };
[[noreturn]] void injected_throw(ThrowKind kind) {
    switch (kind) {
    case ThrowKind::standard:
        throw std::runtime_error("standard injected exception");
    case ThrowKind::custom:
        throw UnusualCallbackException{}; // NOLINT(bugprone-std-exception-baseclass)
    case ThrowKind::integer:
        throw 17; // NOLINT(bugprone-std-exception-baseclass)
    }
    std::abort();
}

void callback_exception_matrix() {
    for (const ThrowKind kind : {ThrowKind::standard, ThrowKind::custom, ThrowKind::integer}) {
        // Each callable kind reports the same typed error rather than leaking
        // an exception, including a custom type and an int.
        ss::Wait wait = [kind](std::chrono::milliseconds) -> bool { injected_throw(kind); };
        const auto waited = wait.invoke("session.test.wait", 1ms);
        assert(!waited && waited.error().code == bc::serving::v2::ConfigErrorCode::io_failure);
        ss::Hooks hooks;
        hooks.started = [kind](ss::ChildKind, const std::string&, pid_t, const fs::path&) {
            injected_throw(kind);
        };
        hooks.ended = [kind](pid_t) { injected_throw(kind); };
        hooks.selected_segment = [kind](const std::string&) { injected_throw(kind); };
        assert(hooks.started.invoke("session.test.started", ss::ChildKind::measuring_engine, "measuring", 1,
                                    fs::path("fake")));
        assert(hooks.ended.invoke("session.test.ended", 1));
        assert(hooks.selected_segment.invoke("session.test.selected_segment", "/r3-serve-injected"));

        // All owner-stage exit paths must reach the final journal and restore
        // once, including an apply that changed a control before throwing.
        for (const std::string stage : {"apply", "plan", "restore"}) {
            Case c("callback-" + stage);
            const fs::path marker = c.run / "power.journal";
            const fs::path count = c.run / "restore.count";
            const fs::path rd = c.run;
            c.session.power_hook.apply = [marker, stage, kind]() -> bc::serving::v2::Status {
                std::ofstream(marker) << "unresolved";
                if (stage == "apply")
                    injected_throw(kind);
                return std::nullopt;
            };
            c.session.power_hook.restore = [marker, count, rd, stage, kind]() -> bc::serving::v2::Status {
                const Json record = ss::read_record(rd).value_or(Json::object());
                for (const auto& child : record.value("processes", Json::array()))
                    assert(gone(child.at("pid").get<pid_t>())); // exit verified BEFORE restore
                std::ofstream(count, std::ios::app) << "r";
                if (stage == "restore")
                    injected_throw(kind);
                fs::remove(marker); // simulated successful journal resolution
                return std::nullopt;
            };
            if (stage == "plan") {
                const auto original = c.session.plan;
                c.session.plan = [original, kind](const fs::path& run, const ss::Hooks& h) -> ss::EngineStart {
                    const auto start = original.invoke("session.test.plan", run, h);
                    if (!start)
                        throw ss::PlanFailed(start.error().message);
                    const fs::path log = run / "measuring-injected.log";
                    const pid_t child = ss::spawn({"/bin/sleep", "30"}, {}, log, [&](pid_t pid) {
                        if (h.started) {
                            const auto result = h.started.invoke(
                                "session.test.started", ss::ChildKind::measuring_engine, "measuring", pid, log);
                            if (result)
                                throw ss::PlanFailed(result->message);
                        }
                    });
                    std::ofstream(run / "measuring.pid") << child;
                    injected_throw(kind);
                };
            }
            const pid_t owner = c.owner();
            if (stage == "restore") {
                c.wait_state({"running"});
                const auto stopped = ss::control(c.run, "stop", 10s);
                assert(stopped && !stopped->value("ok", true));
            }
            assert(exit_code(owner) != 0);
            const Json record = c.record();
            assert(record.value("state", "") == "failed" && any_error_contains(record, "callback"));
            std::ifstream restored(count);
            std::string restore_count;
            std::getline(restored, restore_count);
            assert(restore_count == "r");                       // never twice, even when restore itself throws
            assert(fs::exists(marker) == (stage == "restore")); // retain unresolved journal on restore failure
            for (const auto& child : record.value("processes", Json::array())) {
                assert(child.value("exited", false));
                assert(gone(child.at("pid").get<pid_t>()));
            }
            if (stage == "plan") {
                pid_t pid = 0;
                std::ifstream(c.run / "measuring.pid") >> pid;
                assert(pid > 0 && gone(pid));
            }
        }

        // Actual process registration, wait and exit-notification callback sites.
        // A failing registration keeps the pre-exec gate shut; a failing wait
        // stops the live process; an exit notification is after verified reap.
        for (const std::string stage : {"started", "wait", "ended"}) {
            ss::Hooks h;
            pid_t child = 0;
            h.started = [&](ss::ChildKind, const std::string&, pid_t pid, const fs::path&) {
                child = pid;
                if (stage == "started")
                    injected_throw(kind);
            };
            h.wait = [&](std::chrono::milliseconds) -> bool {
                if (stage == "wait")
                    injected_throw(kind);
                return true;
            };
            h.ended = [&](pid_t pid) {
                assert(pid == child && gone(pid));
                if (stage == "ended")
                    injected_throw(kind);
            };
            bool failed = false;
            try {
                const std::vector<std::string> argv = stage == "ended"
                                                          ? std::vector<std::string>{"/bin/true"}
                                                          : std::vector<std::string>{"/bin/sleep", "30"};
                (void)ss::run_to_completion(argv, {}, h);
            } catch (const bc::serving::callback::Failure& error) {
                failed = error.error().code == bc::serving::v2::ConfigErrorCode::io_failure;
            }
            assert(failed && child > 0 && gone(child));
        }
    }
}

void power_hook_returned_status_errors() {
    using bc::serving::v2::ConfigErrorCode;
    using bc::serving::v2::Status;
    // SessionPower::begin()-style returned error (not a thrown exception).
    Case apply("power-begin-return-error");
    const fs::path journal = apply.run / "power.journal";
    const fs::path restored = apply.run / "power.restored";
    apply.session.power_hook.apply = [journal]() -> Status {
        std::ofstream(journal) << "incomplete apply";
        return bc::serving::v2::make_error(ConfigErrorCode::io_failure, "power", "begin returned failure");
    };
    apply.session.power_hook.restore = [journal, restored]() -> Status {
        std::ofstream(restored, std::ios::app) << "r";
        fs::remove(journal); // simulated rollback
        return std::nullopt;
    };
    assert(exit_code(apply.owner()) != 0);
    const Json begin_result = apply.record();
    assert(begin_result.at("state") == "failed" && any_error_contains(begin_result, "begin returned failure"));
    assert(!fs::exists(journal) && text(restored) == "r");
    for (const auto& e : begin_result.at("events")) {
        assert(e.at("kind") != "power_applied" && e.at("kind") != "running");
    }
    assert(begin_result.at("processes").empty());

    // SessionPower::finish()-style returned error after verified exit leaves an
    // unresolved receipt; neither a success event nor a successful stop reply.
    Case finish("power-finish-return-error");
    const fs::path unresolved = finish.run / "power.journal";
    const fs::path count = finish.run / "power.restore.count";
    finish.session.power_hook.apply = [unresolved]() -> Status {
        std::ofstream(unresolved) << "receipt";
        return std::nullopt;
    };
    finish.session.power_hook.restore = [count, &finish]() -> Status {
        const Json durable = finish.record();
        for (const auto& child : durable.at("processes"))
            assert(gone(child.at("pid").get<pid_t>()));
        std::ofstream(count, std::ios::app) << "r";
        return bc::serving::v2::make_error(ConfigErrorCode::io_failure, "power", "finish returned failure");
    };
    const pid_t owner = finish.owner();
    finish.wait_state({"running"});
    const int engine = finish.engine_pid();
    const auto reply = ss::control(finish.run, "stop", 10s);
    assert(reply && !reply->value("ok", true) && reply->at("state") == "failed");
    assert(exit_code(owner) != 0 && gone(engine));
    assert(fs::exists(unresolved) && text(count) == "r");
    const Json finish_result = finish.record();
    assert(finish_result.at("state") == "failed" &&
           any_error_contains(finish_result, "finish returned failure"));
    for (const auto& e : finish_result.at("events"))
        assert(e.at("kind") != "power_restored");
}

void helper_power_hook_boundary() {
    Case c("hook");
    c.helpers = 2;
    const auto marker = c.run / "power.marker";
    c.session.power_hook.apply = [marker]() -> bc::serving::v2::Status {
        std::ofstream(marker) << "applied";
        return std::nullopt;
    };
    c.session.power_hook.restore = [marker]() -> bc::serving::v2::Status {
        assert(fs::exists(marker));
        fs::remove(marker);
        return std::nullopt;
    };
    const auto owner = c.owner();
    c.wait_state({"running"});
    assert(fs::exists(marker));
    const auto r = ss::control(c.run, "stop", 10s);
    assert(r && r->value("ok", false) && r->at("state") == "stopped");
    assert(exit_code(owner) == 0 && !fs::exists(marker));
    Case f("hook-restore-fail");
    f.session.power_hook.apply = []() -> bc::serving::v2::Status { return std::nullopt; };
    f.session.power_hook.restore = []() -> bc::serving::v2::Status {
        throw std::runtime_error("journal remains");
    };
    const auto pid = f.owner();
    f.wait_state({"running"});
    const auto failed_reply = ss::control(f.run, "stop", 10s);
    assert(failed_reply && !failed_reply->value("ok", true));
    assert(exit_code(pid) == 1 && any_error_contains(f.record(), "journal remains"));
}

} // namespace

void real_session_power_returned_errors();

// Owner receives the real run_to_completion manifest writer BEFORE durable
// manifest admission. This child is CPU-only; power still gates every GPU child.
void manifest_writer_before_durable_admission() {
    Case c("writer-before-admission");
    const auto prior = c.session.plan;
    const fs::path trace = c.run / "writer-order.trace";
    c.session.plan = [prior, trace](const fs::path& run, const ss::Hooks& hooks) -> ss::EngineStart {
        const auto written = ss::run_to_completion({"/bin/true"}, {}, hooks);
        assert(written.exit == 0);
        std::ofstream(trace, std::ios::app) << "writer_exited" << '\n';
        const fs::path provisional = run / "provisional-v2.json";
        std::ofstream(provisional) << "{}";
        const auto result =
            hooks.admit_power.invoke("session.test.admit", provisional, bc::serving::v2::Manifest{}, bc::serving::v2::PlanUse::serving);
        if (!result)
            throw ss::PlanFailed(result.error().message);
        if (result.value())
            throw ss::PlanFailed(result.value()->message);
        std::ofstream(trace, std::ios::app) << "admitted" << '\n';
        return prior.require("session.test.plan", run, hooks);
    };
    c.session.power_hook.begin_manifest = [trace](const fs::path&, const bc::serving::v2::Manifest&,
                                                  bc::serving::v2::PlanUse) -> bc::serving::v2::Status {
        std::ofstream(trace, std::ios::app) << "apply" << '\n';
        return std::nullopt;
    };
    c.session.power_hook.restore = []() -> bc::serving::v2::Status { return std::nullopt; };
    c.session.power_hook.verify_serving = [](const fs::path&) -> bc::serving::v2::Status {
        return std::nullopt;
    };
    c.session.power_hook.verified_exit = [] {};
    const pid_t owner = c.owner();
    c.wait_state({"running"});
    const auto reply = ss::control(c.run, "stop", 10s);
    assert(reply && reply->value("ok", false));
    assert(exit_code(owner) == 0);
    const std::string entries = text(trace);
    assert(entries.find("writer_exited\n") < entries.find("apply\n"));
    assert(entries.find("apply\n") < entries.find("admitted\n"));
    bool recorded = false;
    const Json recorded_session = c.record();
    for (const auto& p : recorded_session.at("processes")) {
        if (p.value("role", std::string()) == "native config inspector") {
            recorded = true;
            assert(p.value("exited", false) && gone(p.at("pid").get<pid_t>()));
        }
    }
    assert(recorded);
}

void native_config_inspector_before_admission() {
    Case c("native-config-inspector");
    c.prepare(); // the engine's plan is made before this test sets the inspector's own expectations below
    const fs::path root_path = c.run / "yaml-root";
    fs::create_directories(root_path);
    const fs::path root = fs::canonical(root_path);
    const fs::path authored = root / "fleet.yaml";
    const std::string yaml = "# caf\xc3\xa9\r\ncards: [0]\r\nmodels:\r\n  tiny: {model: tiny.rad}\r\n";
    std::ofstream(authored, std::ios::binary) << yaml;
    const auto snapshot = bc::serving::NativeConfigSnapshot::read(authored, root);
    const int resolution_root_fd = ::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    assert(resolution_root_fd >= 3);
    const int sentinel = ::open("/dev/null", O_RDONLY);
    assert(sentinel >= 3);
    const std::string sentinel_value = std::to_string(sentinel);
    const auto saved_environment = [](const char* key) -> std::optional<std::string> {
        const char* value = std::getenv(key);
        return value == nullptr ? std::nullopt : std::optional<std::string>(value);
    };
    const auto saved_home = saved_environment("RADIANCE_HOME");
    const auto saved_kernels = saved_environment("RADIANCE_KERNELS");
    ::setenv("TEST_SENTINEL_FD", sentinel_value.c_str(), 1);
    ::setenv("RADIANCE_HOME", "/inherited-wrong-home", 1);
    ::setenv("RADIANCE_KERNELS", "inherited-wrong-kernels", 1);
    ::setenv("TEST_EXPECT_RADIANCE_HOME", "/selected-native-home", 1);
    ::setenv("TEST_EXPECT_RADIANCE_KERNELS", "selected-native-kernels", 1);
    const fs::path owner_cwd = fs::current_path();
    const fs::path trace = c.run / "native-inspector.trace";
    const auto prior = c.session.plan;
    c.session.plan = [prior, snapshot, authored, trace, owner_cwd, resolution_root_fd](const fs::path& run, const ss::Hooks& hooks) {
        std::ofstream(authored, std::ios::binary | std::ios::trunc) << "changed after snapshot\n";
        const ss::Env native_environment = {{"RADIANCE_HOME", "/selected-native-home"},
                                            {"RADIANCE_KERNELS", "selected-native-kernels"}};
        const ss::Env complete_environment = ss::current_child_environment(native_environment);
        const auto report = ss::inspect_and_reconcile_native_config(
            BC_FAKE_ENGINE, snapshot, "/selected-native-home", fs::path(snapshot.resolution_root()) / "tiny.rad",
            complete_environment,
            {"0000:01:00.0"}, 1, hooks, resolution_root_fd);
        assert(report.snapshot_sha256 == snapshot.sha256());
        assert(report.model == "tiny.rad");
        assert(report.abi_version == RAD_ABI_VERSION && report.tp == 1 && report.cards == std::vector<int>{0});
        assert(report.rank_devices == std::vector<std::string>{"0000:01:00.0"});
        const ss::Env implicit_rank_devices = ss::current_child_environment(
            {{"RADIANCE_HOME", "/selected-native-home"},
             {"RADIANCE_KERNELS", "selected-native-kernels"},
             {"TEST_NATIVE_INSPECT_RANK_DEVICES", "[]"}});
        const auto implicit_report = ss::inspect_and_reconcile_native_config(
            BC_FAKE_ENGINE, snapshot, "/selected-native-home", fs::path(snapshot.resolution_root()) / "tiny.rad",
            implicit_rank_devices, {"0000:01:00.0"}, 1, hooks, resolution_root_fd);
        assert(implicit_report.rank_devices.empty());
        for (const auto& [bdfs, tp, expected] :
             std::array<std::tuple<std::vector<std::string>, int, const char*>, 2>{{
                 {{{"0000:02:00.0"}}, 1,
                  "native rank-device order differs from the selected local BDF order"},
                 {{{"0000:01:00.0"}}, 2,
                  "tensor parallelism differs from the selected local profile"}}}) {
            bool refused = false;
            try {
                (void)ss::inspect_and_reconcile_native_config(
                    BC_FAKE_ENGINE, snapshot, "/selected-native-home", fs::path(snapshot.resolution_root()) / "tiny.rad",
                    complete_environment, bdfs, tp, hooks, resolution_root_fd);
            } catch (const ss::PlanFailed& error) {
                refused = std::string_view(error.what()).find(expected) != std::string_view::npos;
            }
            assert(refused);
        }
        bool card_order_refused = false;
        try {
            const ss::Env wrong_cards = ss::current_child_environment(
                {{"RADIANCE_HOME", "/selected-native-home"},
                 {"RADIANCE_KERNELS", "selected-native-kernels"},
                 {"TEST_NATIVE_INSPECT_CARDS", "[1,0]"}});
            (void)ss::inspect_and_reconcile_native_config(
                BC_FAKE_ENGINE, snapshot, "/selected-native-home", fs::path(snapshot.resolution_root()) / "tiny.rad",
                wrong_cards, {"0000:01:00.0"}, 1, hooks, resolution_root_fd);
        } catch (const ss::PlanFailed& error) {
            card_order_refused = std::string_view(error.what()).find(
                                     "YAML card ordinals differ from the local rank visibility order") !=
                                 std::string_view::npos;
        }
        assert(card_order_refused);
        bool unverified_home_refused = false;
        try {
            const ss::Env wrong_home = ss::current_child_environment(
                {{"RADIANCE_HOME", "/selected-native-home"},
                 {"RADIANCE_KERNELS", "selected-native-kernels"},
                 {"TEST_NATIVE_INSPECT_REPORTED_HOME", "/unverified-radiance-home"}});
            (void)ss::inspect_and_reconcile_native_config(
                BC_FAKE_ENGINE, snapshot, "/selected-native-home", fs::path(snapshot.resolution_root()) / "tiny.rad",
                wrong_home, {"0000:01:00.0"}, 1, hooks, resolution_root_fd);
        } catch (const ss::PlanFailed& error) {
            unverified_home_refused = std::string_view(error.what()).find(
                                          "YAML runtime home differs from the verified matched build home") !=
                                      std::string_view::npos;
        }
        assert(unverified_home_refused);
        bool model_mismatch_refused = false;
        try {
            const ss::Env wrong_model = ss::current_child_environment(
                {{"RADIANCE_HOME", "/selected-native-home"},
                 {"RADIANCE_KERNELS", "selected-native-kernels"},
                 {"TEST_NATIVE_INSPECT_MODEL", "other.rad"}});
            (void)ss::inspect_and_reconcile_native_config(
                BC_FAKE_ENGINE, snapshot, "/selected-native-home", fs::path(snapshot.resolution_root()) / "tiny.rad",
                wrong_model, {"0000:01:00.0"}, 1, hooks, resolution_root_fd);
        } catch (const ss::PlanFailed& error) {
            model_mismatch_refused = std::string_view(error.what()).find(
                                         "YAML model differs from the selected local model") !=
                                     std::string_view::npos;
        }
        assert(model_mismatch_refused);
        assert(fs::current_path() == owner_cwd);
        std::ofstream(trace, std::ios::app) << "inspected\n";
        const fs::path provisional = run / "native-provisional-v2.json";
        std::ofstream(provisional) << "{}";
        const auto admission = hooks.admit_power.invoke("native-inspector.test.admit", provisional,
                                                        bc::serving::v2::Manifest{}, bc::serving::v2::PlanUse::serving);
        if (!admission)
            throw ss::PlanFailed(admission.error().message);
        if (admission.value())
            throw ss::PlanFailed(admission.value()->message);
        std::ofstream(trace, std::ios::app) << "admitted\n";
        return prior.require("native-inspector.test.base-plan", run, hooks);
    };
    c.session.power_hook.begin_manifest = [trace](const fs::path&, const bc::serving::v2::Manifest&,
                                                  bc::serving::v2::PlanUse) -> bc::serving::v2::Status {
        std::ofstream(trace, std::ios::app) << "power\n";
        return std::nullopt;
    };
    c.session.power_hook.restore = []() -> bc::serving::v2::Status { return std::nullopt; };
    c.session.power_hook.verify_serving = [](const fs::path&) -> bc::serving::v2::Status {
        return std::nullopt;
    };
    c.session.power_hook.verified_exit = [] {};
    const pid_t owner = c.owner();
    c.wait_state({"running"});
    const auto reply = ss::control(c.run, "stop", 10s);
    assert(reply && reply->value("ok", false));
    assert(exit_code(owner) == 0);
    ::close(resolution_root_fd);
    ::unsetenv("TEST_SENTINEL_FD");
    ::unsetenv("TEST_EXPECT_RADIANCE_HOME");
    ::unsetenv("TEST_EXPECT_RADIANCE_KERNELS");
    if (saved_home)
        ::setenv("RADIANCE_HOME", saved_home->c_str(), 1);
    else
        ::unsetenv("RADIANCE_HOME");
    if (saved_kernels)
        ::setenv("RADIANCE_KERNELS", saved_kernels->c_str(), 1);
    else
        ::unsetenv("RADIANCE_KERNELS");
    ::close(sentinel);
    assert(fs::current_path() == owner_cwd);
    const std::string entries = text(trace);
    assert(entries.find("inspected\n") < entries.find("power\n"));
    assert(entries.find("power\n") < entries.find("admitted\n"));
    bool inspector_recorded = false;
    const Json final_record = c.record();
    for (const auto& process : final_record.at("processes")) {
        if (process.value("role", std::string()) == "native config inspector") {
            inspector_recorded = true;
            assert(process.value("exited", false) && gone(process.at("pid").get<pid_t>()));
        }
    }
    assert(inspector_recorded);
}

void native_config_inspector_invalid_report_refused() {
    for (const auto& [mode, expected_error] :
         std::array<std::pair<const char*, const char*>, 2>{{
             {"digest", "report failed strict protocol or snapshot identity validation"},
             {"cards", "cards must contain nonnegative integer ordinals"}}}) {
        Case c(std::string("native-config-inspector-") + mode);
        c.prepare(); // planned before the mismatching inspector is armed: only the plan wrapper below meets it
        const fs::path root = c.run / "yaml-root";
        fs::create_directories(root);
        const fs::path authored = root / "fleet.yaml";
        std::ofstream(authored, std::ios::binary)
            << "# caf\xc3\xa9\r\ncards: [0]\r\nmodels:\r\n  tiny: {model: tiny.rad}\r\n";
        const auto snapshot = bc::serving::NativeConfigSnapshot::read(authored, root);
        ::setenv("TEST_INSPECTOR_MISMATCH", mode, 1);
        c.session.plan = [snapshot](const fs::path&, const ss::Hooks& hooks) {
            (void)ss::inspect_native_config(BC_FAKE_ENGINE, snapshot, ss::current_child_environment({}), hooks);
            throw ss::PlanFailed("inspector refusal gate did not fire");
            return ss::EngineStart{};
        };
        c.session.power_hook.begin_manifest = [](const fs::path&, const bc::serving::v2::Manifest&,
                                                 bc::serving::v2::PlanUse) -> bc::serving::v2::Status {
            assert(false && "power admission ran after an invalid native inspector report");
            return std::nullopt;
        };
        c.session.power_hook.restore = []() -> bc::serving::v2::Status { return std::nullopt; };
        c.session.power_hook.verify_serving = [](const fs::path&) -> bc::serving::v2::Status {
            return std::nullopt;
        };
        c.session.power_hook.verified_exit = [] {};
        const pid_t owner = c.owner();
        c.wait_state({"failed"});
        assert(exit_code(owner) == 97);
        ::unsetenv("TEST_INSPECTOR_MISMATCH");
        const Json record = c.record();
        assert(!record.contains("power_admission"));
        bool expected_refusal = false;
        for (const auto& error : record.at("errors"))
            if (error.is_string() && error.get<std::string>().find(expected_error) != std::string::npos)
                expected_refusal = true;
        assert(expected_refusal);
        bool inspector_reaped = false;
        for (const auto& process : record.at("processes")) {
            assert(process.value("role", std::string()) != "engine");
            if (process.value("role", std::string()) == "native config inspector")
                inspector_reaped = process.value("exited", false) && gone(process.at("pid").get<pid_t>());
        }
        assert(inspector_reaped);
    }
}

// Fresh planning is native-only: a plan step with no admitted native YAML, or an engine from another build, is
// refused before any measurement starts.
std::string plan_refusal(const bc::test::NativeSessionPlan& native, const fs::path& run) {
    fs::create_directories(run);
    int children = 0;
    ss::Hooks hooks = bc::test::standalone_plan_hooks();
    hooks.started = [&](ss::ChildKind kind, const std::string&, pid_t, const fs::path&) {
        if (kind == ss::ChildKind::measuring_engine) ++children;   // an inspector may run; no measurement may
    };
    try {
        (void)ss::plan_engine_only(native.inputs, run, hooks);
    } catch (const ss::PlanFailed& error) {
        assert(children == 0);
        return error.what();
    }
    return {};
}

void native_only_planning_refusals() {
    const fs::path scratch = g_root / "native-only-planning";
    {
        auto native = bc::test::native_session_inputs(g_host, scratch / "no-yaml", bc::test::free_port());
        native.inputs.native_admission.reset();
        native.inputs.named_config.clear();
        native.inputs.options.configs = {fs::path(BC_CONFIGS_DIR)};
        const std::string why = plan_refusal(native, scratch / "no-yaml-run");
        assert(why.find("native config YAML") != std::string::npos);
        native.inputs.named_config = "qwen27b-tp3";
        assert(plan_refusal(native, scratch / "named-no-yaml-run").find("native config YAML") != std::string::npos);
    }
    {
        // bc-serve of one build refuses the engine of another: the manifest is read by the build that wrote it.
        auto native = bc::test::native_session_inputs(g_host, scratch / "build", bc::test::free_port());
        const fs::path other = scratch / "other-build";
        fs::create_directories(other / "radiance_home/bin");
        bc::test::write_file(other / "bin/radiance", "");
        fs::create_symlink(BC_SERVE_EXE, other / "radiance_home/bin/bc-serve");
        native.inputs.self = other / "radiance_home/bin/bc-serve";
        assert(plan_refusal(native, scratch / "build-run").find("is not this bc-serve's build") !=
               std::string::npos);
    }
}

void premeasurement_manifest_hook_integration() {
    // One owner, one run, and one manifest admission *before* measuring.
    // Explicit fake hooks record their sequencing without ever touching sysfs.
    for (const std::string stage : {"normal", "begin_error", "ignored_begin_error", "verify_throw",
                                    "loaded_error", "runtime_throw", "watch_error", "exit_throw"}) {
        Case c("manifest-" + stage);
        const fs::path trace = c.run / "hooks.trace";
        const auto log = [trace](const std::string& value) {
            std::ofstream out(trace, std::ios::app);
            out << value << '\n';
            if (!out)
                throw std::runtime_error("cannot record integration hook");
        };
        const auto previous = c.session.plan;
        c.session.plan = [previous, stage](const fs::path& run, const ss::Hooks& hooks) -> ss::EngineStart {
            const fs::path provisional = run / "provisional-v2.json";
            std::ofstream(provisional) << "{\"premeasured\":true}";
            const auto admission = hooks.admit_power.invoke("integration.admit_power", provisional,
                                                            bc::serving::v2::Manifest{}, bc::serving::v2::PlanUse::measuring);
            if (!admission)
                throw ss::PlanFailed(admission.error().message);
            if (admission.value() && stage != "ignored_begin_error")
                throw ss::PlanFailed(admission.value()->message);
            return previous.require("integration.base_plan", run, hooks);
        };
        c.session.power_hook.begin_manifest =
            [stage, log](const fs::path&, const bc::serving::v2::Manifest&, bc::serving::v2::PlanUse use) -> bc::serving::v2::Status {
            assert(use == bc::serving::v2::PlanUse::measuring);
            log("begin");
            if (stage == "begin_error" || stage == "ignored_begin_error")
                return bc::serving::v2::make_error(bc::serving::v2::ConfigErrorCode::admission_refused,
                                                   "integration", "begin refused");
            return std::nullopt;
        };
        c.session.power_hook.verify_serving = [stage,
                                               log](const fs::path& manifest) -> bc::serving::v2::Status {
            assert(manifest.string().rfind("/proc/self/fd/", 0) == 0); // the sealed descriptor, never a path
            log("verify");
            if (stage == "verify_throw")
                throw 91; // NOLINT(bugprone-std-exception-baseclass): injected nonstandard callback
            return std::nullopt;
        };
        c.session.power_hook.gfxoff_window = [stage, log](bool loaded) -> bc::serving::v2::Status {
            log(loaded ? "loaded_idle" : "floor");
            if (stage == "loaded_error" && loaded)
                return bc::serving::v2::make_error(bc::serving::v2::ConfigErrorCode::io_failure,
                                                   "integration.gfxoff", "receipt persistence failed");
            return std::nullopt;
        };
        c.session.power_hook.verified_exit = [stage, log, &c] {
            assert(c.engine_pid() <= 0 || gone(c.engine_pid()));
            log("verified_exit");
            if (stage == "exit_throw")
                throw 37; // NOLINT(bugprone-std-exception-baseclass): injected nonstandard callback
        };
        c.session.power_hook.restore = [log, &c]() -> bc::serving::v2::Status {
            assert(c.engine_pid() <= 0 || gone(c.engine_pid()));
            log("finish");
            return std::nullopt;
        };
        c.session.runtime_guard = [stage, log](const ss::EngineStart&,
                                               const fs::path&) -> bc::serving::v2::Status {
            log("runtime_start");
            if (stage == "runtime_throw")
                throw 91; // NOLINT(bugprone-std-exception-baseclass): injected nonstandard callback
            return std::nullopt;
        };
        c.session.runtime_watch = [stage, log]() -> bc::serving::v2::Status {
            log("watch");
            if (stage == "watch_error")
                return bc::serving::v2::make_error(bc::serving::v2::ConfigErrorCode::admission_refused,
                                                   "integration.live", "live policy degraded");
            return std::nullopt;
        };
        const pid_t owner = c.owner();
        if (stage == "normal" || stage == "exit_throw") {
            c.wait_state({"running"});
            const auto stopped = ss::control(c.run, "stop", 10s);
            assert(stopped && stopped->value("ok", true) == (stage == "normal"));
            assert((exit_code(owner) == 0) == (stage == "normal"));
        } else {
            assert(exit_code(owner) != 0);
        }
        if (stage != "normal")
            assert(c.record().value("state", "") == "failed");
        const std::string events = text(trace);
        const auto begin = events.find("begin\n");
        const auto verified = events.find("verified_exit\n");
        const auto finish = events.find("finish\n");
        assert(begin == 0 && verified != std::string::npos && finish != std::string::npos && verified < finish);
        if (stage == "begin_error" || stage == "ignored_begin_error" || stage == "verify_throw") {
            assert(c.record().value("processes", Json::array()).empty());
        } else {
            assert(events.find("floor\n") != std::string::npos &&
                   events.find("loaded_idle\n") != std::string::npos);
            for (const auto& child : c.record().value("processes", Json::array()))
                assert(gone(child.at("pid").get<pid_t>()));
        }
        assert(events.find("verify\n") == std::string::npos || events.find("verify\n") > begin);
    }
}

int test_main(int argc, char** argv) {
    if (argc == 6 && std::string(argv[1]) == "--owner") {
        // owner mode for start_detached_runs_and_stops: `bc-serve start` re-runs `serve`, which plans in the owner
        // (the real plan step over the native host, its power admission a no-op) and then serves
        ss::Session s;
        s.run_dir = argv[2];
        s.timing.health_interval = 50ms;
        s.timing.stop_grace = 1s;
        s.timing.poll = 20ms;
        const int port = std::stoi(argv[3]);
        const fs::path host = argv[4];
        const Json behaviour = Json::parse(argv[5]);
        s.power_hook.begin_manifest = [](const fs::path&, const bc::serving::v2::Manifest&, bc::serving::v2::PlanUse) -> bc::serving::v2::Status {
            return std::nullopt;
        };
        s.power_hook.verify_serving = [](const fs::path&) -> bc::serving::v2::Status { return std::nullopt; };
        s.power_hook.verified_exit = [] {};
        s.power_hook.restore = []() -> bc::serving::v2::Status { return std::nullopt; };
        s.plan = [port, host, behaviour](const fs::path& rd, const ss::Hooks& hooks) {
            const auto native = bc::test::native_session_inputs(host, rd / "native", port, behaviour);
            fs::create_directories(rd / "plan");
            return ss::plan_engine_only(native.inputs, rd / "plan", hooks);
        };
        return ss::serve(s);
    }
    std::signal(SIGABRT, stop_owners);
    std::string group;
    if (argc == 3 && std::string(argv[1]) == "--group") group = argv[2];
    g_root = fs::current_path() / ("bc-session-fixture" + (group.empty() ? std::string() : "-" + group));
    fs::remove_all(g_root);
    fs::create_directories(g_root);
    g_host = g_root / "host";
    bc::test::write_fake_host(g_host, /*unlimited_memlock=*/true);
    if (argc == 2 && std::string(argv[1]) == "--host-start-only") {
        // Focused CPU-only MRC02 owner receipt regression (no hardware lease).
        feature_state_start_status_and_warning();
        std::cout << "PASS: native owner host-facts start receipt before child\n";
        return 0;
    }
    // The tests, in four groups of about equal run time (measured), so that CTest can run them side by side: each
    // group is its own ctest test (`--group N`) with its own fixture directory, and running with no argument runs them
    // all. Every test is in exactly one group.
    static const std::vector<std::vector<void (*)()>> groups = {
        {real_session_power_returned_errors, manifest_writer_before_durable_admission, native_config_inspector_before_admission, callback_exception_matrix, power_hook_returned_status_errors, stop_while_waiting_for_health, stop_during_plan, unwritable_record_still_stops},
        {native_only_planning_refusals, premeasurement_manifest_hook_integration, serve_status_stop, helpers_order_and_killed, calibration_ingest_never_fails_serving, helper_power_hook_boundary, canary_fails, engine_ignores_sigint, unknown_command_and_no_owner, no_record_no_session},
        {injected_owner_plan_exception, feature_state_start_status_and_warning, feature_state_invalid_manifest_prevents_child, helper_failure_unwinds, helper_shared_segment_gate, long_run_dir, shared_run_dir_and_stale_socket, engine_refuses_the_stored_plan, engine_never_healthy, plan_refused},
        {native_config_inspector_invalid_report_refused, callback_nonstandard_regression, spawn_and_environment, engine_fails_to_start, engine_refuses_host_room, engine_dies_while_serving, signal_stops, start_detached_runs_and_stops, descendant_outliving_the_engine, inherited_descriptor_not_passed, owner_killed_fails_closed},
    };
    size_t first = 0, last = groups.size();
    if (!group.empty()) {
        first = std::stoul(group) - 1;   // "1" .. the number of groups
        if (first >= groups.size()) {
            std::cerr << "session test: no group " << group << '\n';
            return 2;
        }
        last = first + 1;
    }
    for (size_t i = first; i < last; ++i)
        for (const auto test : groups[i]) test();
    std::cout
        << "PASS: session owner lifecycle (start order, canary, failures at every stage, stop via socket and "
           "signal, SIGKILL fallback, one owner per run directory, child environment, detached start)\n";
    return 0;
}

// Integration contract: the owner receives the *real* step-4 transaction's
// returned Status, not a mock return or a discarded void callback. Hardware
// is simulated in memory; DiskRecords use only this test's temporary directory.
class PowerIntegrationHardware final : public bc::serving::power::Hardware {
  public:
    int64_t shader = 2600;

  private:
    bc::serving::v2::Result<bc::serving::power::CardSnapshot>
    do_inspect(const bc::serving::power::PciAddress& address) override {
        using namespace bc::serving::power;
        CardSnapshot result;
        result.pci = address;
        result.amdgpu = true;
        result.hwmon = "hwmon2";
        result.controls[static_cast<size_t>(ControlId::voltage)] = Capability{-100, -200, 0, true};
        result.controls[static_cast<size_t>(ControlId::shader)] = Capability{shader, 2000, 3200, true};
        result.controls[static_cast<size_t>(ControlId::memory)] = Capability{1300, 900, 1500, true};
        result.controls[static_cast<size_t>(ControlId::board_cap)] =
            Capability{300000000, 200000000, 350000000, true};
        return result;
    }
    bc::serving::v2::Status do_execute(const bc::serving::power::PowerCommand& command) override {
        if (const auto* set = std::get_if<bc::serving::power::ShaderClock>(&command.value))
            shader = set->value.value;
        return std::nullopt;
    }
};

void real_session_power_returned_errors() {
    namespace pw = bc::serving::power;
    for (bool fail_begin : {true, false}) {
        Case c(fail_begin ? "real-power-begin-error" : "real-power-finish-error");
        PowerIntegrationHardware hardware;
        pw::DiskRecords records(c.run / "power-journals", c.run / "power-receipts");
        pw::PowerPlan plan;
        plan.mode = pw::Mode::apply;
        plan.profile = "card-free-integration";
        plan.profile_digest = pw::Digest{std::string(64, 'a')};
        plan.manifest_digest = pw::Digest{std::string(64, 'b')};
        if (!fail_begin) {
            pw::DevicePolicy policy;
            policy.device = pw::SelectedGpu{pw::Role{pw::RoleKind::engine, 0}, pw::PciAddress{"0000:03:00.0"}};
            policy.tuning.shader_ceiling = pw::Megahertz{2300};
            plan.devices.push_back(policy);
        }
        pw::SessionPower transaction(
            std::move(plan), "owner-binding-" + std::to_string(g_case), hardware, records, [] { return true; },
            [fail_begin] { return fail_begin; });
        // The real methods both return v2::Status. The shared Guarded wrapper
        // must preserve a returned error inside Result<Status>.
        c.session.power_hook = ss::hooks_for_power_transaction(transaction);
        const pid_t owner = c.owner();
        if (!fail_begin) {
            c.wait_state({"running"});
            const auto stop = ss::control(c.run, "stop", 10s);
            assert(stop && !stop->value("ok", true));
        }
        assert(exit_code(owner) != 0);
        const Json record = c.record();
        assert(record.value("state", "") == "failed");
        if (fail_begin) {
            assert(any_error_contains(record, "empty serving GPU inventory"));
            assert(record.value("processes", Json::array()).empty());
        } else {
            assert(any_error_contains(record, "serving process groups may still be alive"));
            assert(hardware.shader == 2600); // parent never touches hardware; only forked owner does
            for (const auto& child : record.value("processes", Json::array()))
                assert(gone(child.at("pid").get<pid_t>()));
        }
    }
}

int main(int argc, char** argv) noexcept {
    try {
        return test_main(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "session test: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "session test: non-standard exception" << '\n';
        return 1;
    }
}
