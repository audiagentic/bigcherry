// RSC02 step 1: the session owner's lifecycle, card-free, against a fake engine (session_fake_engine.cpp) that serves
// /health and /v1/completions and misbehaves on request. Ported from the Python supervisor's fake-process tests
// (tools/tests/serving/test_supervisor.py): ordered start and stop, a failure at every stage unwinds what started, a
// survivor is never reported stopped, one session at a time, stop through the control socket only, inherited
// configuration never reaches a child.
#include "session.h"

#include <cassert>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <thread>

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef BC_FAKE_ENGINE
#error "BC_FAKE_ENGINE must name the fake engine binary"
#endif

namespace fs = std::filesystem;
namespace ss = bc::serving::session;
using Json = nlohmann::json;
using namespace std::chrono_literals;

namespace {

int free_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a);
    socklen_t len = sizeof a;
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len);
    ::close(fd);
    return ntohs(a.sin_port);
}

fs::path g_root;
int g_case = 0;
// owners still running when an assertion aborts the test: stopped (SIGTERM: they stop their engines) so none is orphaned
std::vector<pid_t> g_owners;
void stop_owners(int) {
    for (pid_t p : g_owners) ::kill(p, SIGTERM);
    std::signal(SIGABRT, SIG_DFL);
    std::abort();
}

struct Case {
    fs::path run;
    int port = free_port();
    Json behaviour = Json::object();
    ss::Session session;
    bool plan_called = false;

    explicit Case(const std::string& name) : run(g_root / (std::to_string(++g_case) + "-" + name)) {
        fs::create_directories(run);
        session.run_dir = run;
        session.timing.health_timeout = 5s;
        session.timing.health_interval = 50ms;
        session.timing.canary_timeout = 5s;
        session.timing.stop_grace = 1s;
        session.timing.poll = 20ms;
        session.plan = [this](const fs::path& rd, const ss::Wait&) {
            Json m = behaviour;
            m["port"] = port;
            const fs::path manifest = rd / "fake-manifest.json";
            std::ofstream(manifest) << m.dump();
            ss::EngineStart s;
            s.manifest = manifest;
            s.argv = {BC_FAKE_ENGINE, "--manifest", manifest.string(), "--role", "engine"};
            s.runtime = {{"RADIANCE_HOME", "/runtime"}, {"ROCR_VISIBLE_DEVICES", "3"}, {"HIP_VISIBLE_DEVICES", "0"}};
            s.port = port;
            s.summary = {{"config", "fake"}};
            return s;
        };
    }

    // the owner in its own process, as `bc-serve serve` runs
    pid_t owner() {
        std::cout.flush();
        const pid_t pid = ::fork();
        if (pid == 0) {
            int code = 1;
            try { code = ss::serve(session); } catch (const std::exception& e) { std::cerr << "owner: " << e.what() << '\n'; }
            std::_Exit(code);
        }
        g_owners.push_back(pid);
        return pid;
    }

    Json record() const { return ss::read_record(run).value_or(Json::object()); }

    std::string wait_state(const std::set<std::string>& states, std::chrono::milliseconds limit = 10s) const {
        const auto deadline = std::chrono::steady_clock::now() + limit;
        while (std::chrono::steady_clock::now() < deadline) {
            const std::string s = record().value("state", "");
            if (states.count(s)) return s;
            std::this_thread::sleep_for(20ms);
        }
        std::cerr << "state never reached; record: " << record().dump(2) << '\n';
        assert(false);
        return {};
    }

    int engine_pid() const {
        for (const auto& p : record().value("processes", Json::array())) if (p.at("role") == "engine") return p.at("pid");
        return 0;
    }
};

int exit_code(pid_t pid, std::chrono::milliseconds limit = 15s) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    int st = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        if (::waitpid(pid, &st, WNOHANG) == pid) return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
        std::this_thread::sleep_for(20ms);
    }
    ::kill(pid, SIGKILL);
    ::waitpid(pid, &st, 0);
    assert(!"the owner did not exit");
    return -1;
}

bool gone(int pid) { return pid > 0 && ::kill(pid, 0) != 0 && errno == ESRCH; }

std::string text(const fs::path& p) {
    std::ifstream in(p);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

bool any_error_contains(const Json& record, const std::string& needle) {
    for (const auto& e : record.value("errors", Json::array()))
        if (e.get<std::string>().find(needle) != std::string::npos) return true;
    return false;
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
    for (const auto& e : running_record.at("events")) kinds.push_back(e.at("kind").get<std::string>());
    const std::vector<std::string> want = {"created", "planning", "planned", "started", "healthy", "canary_passed", "running"};
    assert(kinds == want);
    const auto status = ss::control(c.run, "status", 2s);
    assert(status && status->value("ok", false) && status->at("state") == "running" && status->at("port") == c.port);
    assert(status->at("owner_pid") == owner);
    const std::string env = text(c.run / "engine.env");
    assert(env.find("R3_MOE_FP16") == std::string::npos && env.find("BC_MODEL_ROOT") == std::string::npos &&
           env.find("BIGCHERRY_CONFIGS") == std::string::npos);
    assert(env.find("HIP_VISIBLE_DEVICES=0\n") != std::string::npos && env.find("HIP_VISIBLE_DEVICES=2,3") == std::string::npos);
    assert(env.find("ROCR_VISIBLE_DEVICES=3\n") != std::string::npos && env.find("RADIANCE_HOME=/runtime\n") != std::string::npos);
    assert(env.find("RSC02_TEST_KEEP=yes\n") != std::string::npos);   // the runtime's own environment stays
    for (const char* v : {"R3_MOE_FP16", "BC_MODEL_ROOT", "BIGCHERRY_CONFIGS", "HIP_VISIBLE_DEVICES", "RSC02_TEST_KEEP"})
        ::unsetenv(v);
    // the engine runs in its own process group (stop signals the group, never the owner's)
    const int engine = c.engine_pid();
    assert(engine > 0 && ::getpgid(engine) == engine);
    // a second owner in the same run directory is refused while this one runs, and changes nothing
    {
        Case second("second");
        second.session.run_dir = c.run;
        bool planned = false;
        second.session.plan = [&](const fs::path&, const ss::Wait&) -> ss::EngineStart { planned = true; return {}; };
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

void canary_fails() {
    Case c("canary");
    c.behaviour["answer"] = "Lyon, of course.";
    const pid_t owner = c.owner();
    assert(exit_code(owner) == 96);
    const Json r = c.record();
    assert(r.at("state") == "failed" && any_error_contains(r, "start canary failed") && any_error_contains(r, "Lyon"));
    assert(gone(c.engine_pid()));
}

void engine_fails_to_start() {
    Case c("no-start");
    c.behaviour["exit_at_start"] = 4;
    assert(exit_code(c.owner()) == 1);
    const Json r = c.record();
    assert(r.at("state") == "failed" && any_error_contains(r, "did not come up") && any_error_contains(r, "exit 4"));
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
    assert(r.at("state") == "failed" && any_error_contains(r, "exited while serving") && any_error_contains(r, "exit 7"));
}

void engine_ignores_sigint() {
    Case c("stubborn");
    c.behaviour["ignore_sigint"] = true;
    const pid_t owner = c.owner();
    c.wait_state({"running"});
    const int engine = c.engine_pid();
    const auto t0 = std::chrono::steady_clock::now();
    const auto reply = ss::control(c.run, "stop", 10s);
    assert(reply && reply->value("ok", false));   // SIGKILL after the grace, and its exit verified
    assert(std::chrono::steady_clock::now() - t0 >= 900ms);
    assert(gone(engine) && exit_code(owner) == 0);
}

void plan_refused() {
    Case c("plan");
    c.session.plan = [](const fs::path&, const ss::Wait&) -> ss::EngineStart {
        throw ss::PlanFailed("plan: no measured manifest (test)");
    };
    assert(exit_code(c.owner()) == 97);
    const Json r = c.record();
    assert(r.at("state") == "failed" && r.at("processes").empty() && any_error_contains(r, "no measured manifest"));
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
    c.session.plan = [](const fs::path&, const ss::Wait& wait) -> ss::EngineStart {
        while (wait(50ms)) {}
        throw ss::PlanFailed("plan: stopped while measuring");
    };
    const pid_t owner = c.owner();
    c.wait_state({"planning"});
    assert(ss::control(c.run, "status", 2s)->at("state") == "planning");
    ss::control(c.run, "stop", 10s);
    assert(exit_code(owner) == 97);
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
    for (const auto& e : r.at("events")) if (e.at("kind") == "stop_requested") by_signal = e.at("by") == "signal 15";
    assert(by_signal);
}

void unknown_command_and_no_owner() {
    Case c("control");
    assert(!ss::control(c.run, "status", 500ms));   // nothing there: no owner answers
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
    try { ss::spawn({"/nonexistent/engine"}, {}, g_root / "spawn.log"); } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("/nonexistent/engine") != std::string::npos;
    }
    assert(threw);
    char a[] = "PATH=/bin", b[] = "RAD_X=1", c[] = "BC_Y=2", d[] = "ROCR_VISIBLE_DEVICES=1", e[] = "HOME=/h";
    char* inherited[] = {a, b, c, d, e, nullptr};
    const auto env = ss::child_environment(inherited, {{"ROCR_VISIBLE_DEVICES", "4"}});
    assert(env == (ss::Env{{"PATH", "/bin"}, {"HOME", "/h"}, {"ROCR_VISIBLE_DEVICES", "4"}}));
}

void start_detached_runs_and_stops() {
    // `start`: the owner detached; returns once running, and `stop` ends it through the socket
    Case c("detached");
    // the detached owner is this test binary in its owner mode (main below), as `bc-serve start` re-runs `serve`
    Json behaviour = {{"port", c.port}};
    std::ofstream(c.run / "fake-manifest.json") << behaviour.dump();
    const std::vector<std::string> self = {fs::read_symlink("/proc/self/exe").string(), "--owner", c.run.string(),
                                           std::to_string(c.port)};
    assert(ss::start_detached(self, c.run, 20s) == 0);
    const auto status = ss::control(c.run, "status", 2s);
    assert(status && status->at("state") == "running");
    const int owner = status->at("owner_pid");
    assert(ss::control(c.run, "stop", 10s)->value("ok", false));
    // (the detached owner is this test's child, as it is the `start` process's until that returns: reap it)
    for (int i = 0; i < 200 && ss::running(owner); ++i) std::this_thread::sleep_for(20ms);
    assert(!ss::running(owner) && gone(owner));
    // a start whose session fails reports the failure
    Case f("detached-fail");
    std::ofstream(f.run / "fake-manifest.json") << Json({{"port", f.port}, {"exit_at_start", 3}}).dump();
    assert(ss::start_detached({self[0], "--owner", f.run.string(), std::to_string(f.port)}, f.run, 20s) == 1);
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--owner") {
        // owner mode for start_detached_runs_and_stops: serve the fake manifest already in the run directory
        ss::Session s;
        s.run_dir = argv[2];
        s.timing.health_interval = 50ms;
        s.timing.stop_grace = 1s;
        s.timing.poll = 20ms;
        const int port = std::stoi(argv[3]);
        s.plan = [port](const fs::path& rd, const ss::Wait&) {
            ss::EngineStart e;
            e.manifest = rd / "fake-manifest.json";
            e.argv = {BC_FAKE_ENGINE, "--manifest", e.manifest.string(), "--role", "engine"};
            e.port = port;
            return e;
        };
        return ss::serve(s);
    }
    std::signal(SIGABRT, stop_owners);
    g_root = fs::current_path() / "bc-session-fixture";
    fs::remove_all(g_root);
    fs::create_directories(g_root);
    spawn_and_environment();
    serve_status_stop();
    canary_fails();
    engine_fails_to_start();
    engine_never_healthy();
    engine_dies_while_serving();
    engine_ignores_sigint();
    plan_refused();
    stop_while_waiting_for_health();
    stop_during_plan();
    signal_stops();
    unknown_command_and_no_owner();
    start_detached_runs_and_stops();
    std::cout << "PASS: session owner lifecycle (start order, canary, failures at every stage, stop via socket and "
                 "signal, SIGKILL fallback, one owner per run directory, child environment, detached start)\n";
    return 0;
}
