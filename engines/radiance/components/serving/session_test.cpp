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
#include <fcntl.h>
#include <sys/stat.h>

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

    explicit Case(const std::string& name) : run(g_root / (std::to_string(++g_case) + "-" + name)) {
        fs::create_directories(run);
        session.run_dir = run;
        session.timing.health_timeout = 5s;
        session.timing.health_interval = 50ms;
        session.timing.canary_timeout = 5s;
        session.timing.stop_grace = 1s;
        session.timing.poll = 20ms;
        session.plan = [this](const fs::path& rd, const ss::Hooks&) {
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

// gone: no such process, or a zombie only its new parent has still to reap
bool gone(int pid) {
    if (pid <= 0) return false;
    if (::kill(pid, 0) != 0 && errno == ESRCH) return true;
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
        second.session.plan = [&](const fs::path&, const ss::Hooks&) -> ss::EngineStart { planned = true; return {}; };
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

// A run directory whose socket path is longer than sun_path (107 bytes) still serves, answers and stops: the address
// goes through a descriptor open on the directory.
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
    c.session.plan = [](const fs::path&, const ss::Hooks&) -> ss::EngineStart {
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
    c.session.plan = [](const fs::path&, const ss::Hooks& h) -> ss::EngineStart {
        while (h.wait(50ms)) {}
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
    // the start gate: the child runs nothing until it is recorded, and nothing at all when recording fails
    {
        const fs::path ran = g_root / "gate-ran";
        bool ran_before_recorded = true;
        const pid_t pid = ss::spawn({"/bin/sh", "-c", "touch " + ran.string()}, {{"PATH", "/bin:/usr/bin"}}, g_root / "spawn.log",
                                    [&](pid_t) {
                                        std::this_thread::sleep_for(200ms);
                                        ran_before_recorded = fs::exists(ran);
                                    });
        int st = 0;
        ::waitpid(pid, &st, 0);
        assert(!ran_before_recorded && fs::exists(ran));
        const fs::path never = g_root / "gate-never";
        bool refused = false;
        try {
            ss::spawn({"/bin/sh", "-c", "touch " + never.string()}, {{"PATH", "/bin:/usr/bin"}}, g_root / "spawn.log",
                      [](pid_t) { throw std::runtime_error("record failed"); });
        } catch (const std::runtime_error& e) { refused = std::string(e.what()) == "record failed"; }
        std::this_thread::sleep_for(200ms);
        assert(refused && !fs::exists(never));
        // a writer that exits at once is registered before it can exit (the gate), so registration never fails
        std::vector<pid_t> seen;
        ss::Hooks quick;
        quick.wait = [](std::chrono::milliseconds) { return true; };
        quick.started = [&](const std::string&, pid_t pid, const fs::path&) {
            std::this_thread::sleep_for(100ms);
            seen.push_back(pid);
            assert(ss::start_time(pid));   // still waiting at the gate: alive, not a zombie
        };
        assert(ss::run_to_completion({"/bin/true"}, {}, quick).exit == 0 && seen.size() == 1);
    }
    // a manifest write is waited for through the owner: output kept, and a stop request ends a slow one at once
    std::vector<pid_t> registered, verified;
    ss::Hooks hooks;
    hooks.wait = [](std::chrono::milliseconds) { return true; };
    hooks.started = [&](const std::string& role, pid_t pid, const fs::path&) { assert(role == "manifest writer"); registered.push_back(pid); };
    hooks.ended = [&](pid_t pid) { verified.push_back(pid); };
    const auto echoed = ss::run_to_completion({"/bin/sh", "-c", "echo out; echo err >&2; exit 3"}, {{"PATH", "/bin"}}, hooks);
    assert(echoed.exit == 3 && echoed.out == "out\n" && echoed.err == "err\n");
    assert(registered.size() == 1 && registered == verified);   // registered with the owner, its exit verified
    int waits = 0;
    hooks.wait = [&](std::chrono::milliseconds) { return ++waits < 3; };
    const auto t0 = std::chrono::steady_clock::now();
    bool stopped = false;
    try {
        ss::run_to_completion({"/bin/sleep", "30"}, {}, hooks);
    } catch (const ss::PlanFailed&) { stopped = true; }
    assert(stopped && waits == 3 && std::chrono::steady_clock::now() - t0 < 5s && verified.size() == 2);
    // a writer that leaves a descendant ignoring SIGTERM: its whole group goes (SIGKILL after the grace) before it returns
    const fs::path child_pid = g_root / "writer-descendant.pid";
    waits = 0;
    hooks.wait = [&](std::chrono::milliseconds d) { std::this_thread::sleep_for(d); return ++waits < 5; };
    stopped = false;
    try {
        ss::run_to_completion({"/bin/sh", "-c", "trap '' TERM INT; /bin/sleep 60 & echo $! > " + child_pid.string() + "; wait"},
                              {{"PATH", "/bin"}}, hooks);
    } catch (const ss::PlanFailed&) { stopped = true; }
    assert(stopped && verified.size() == 3 && gone(std::stoi(text(child_pid))));
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

void descendant_outliving_the_engine() {
    // the engine exits on SIGINT but a process it forked ignores it: stop is not done until the whole group is gone
    Case c("descendant");
    c.behaviour["descendant"] = (c.run / "descendant.pid").string();
    const pid_t owner = c.owner();
    c.wait_state({"running"});
    const int descendant = std::stoi(text(c.run / "descendant.pid"));
    assert(!gone(descendant) && ::getpgid(descendant) == c.engine_pid());
    const auto reply = ss::control(c.run, "stop", 10s);
    assert(reply && reply->value("ok", false));
    for (int i = 0; i < 100 && !gone(descendant); ++i) std::this_thread::sleep_for(20ms);
    assert(gone(descendant) && exit_code(owner) == 0);
}

void inherited_descriptor_not_passed() {
    // a descriptor the owner holds without close-on-exec never reaches the engine
    Case c("fds");
    const fs::path sentinel = c.run / "sentinel.txt";
    const int fd = ::open(sentinel.c_str(), O_WRONLY | O_CREAT, 0600);   // inheritable on purpose
    assert(fd > 2);
    c.behaviour["fd_dump"] = (c.run / "engine.fds").string();
    const pid_t owner = c.owner();
    ::close(fd);
    c.wait_state({"running"});
    const std::string fds = text(c.run / "engine.fds");
    assert(!fds.empty() && fds.find("sentinel.txt") == std::string::npos && fds.find("session.lock") == std::string::npos &&
           fds.find("session.sock") == std::string::npos);
    ss::control(c.run, "stop", 10s);
    assert(exit_code(owner) == 0);
}

void owner_killed_fails_closed() {
    // an owner killed with SIGKILL leaves its engine: stop says so (fails), status warns, a new owner refuses the cards
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
    next.session.plan = [&](const fs::path&, const ss::Hooks&) -> ss::EngineStart { planned = true; return {}; };
    assert(exit_code(next.owner()) == 98 && !planned);
    // its leader gone, a process left in its group still counts (a descendant the engine forked)
    ::kill(-engine, SIGKILL);
    for (int i = 0; i < 100 && !gone(engine); ++i) std::this_thread::sleep_for(20ms);
    {
        Case d("descendant-left");
        d.behaviour["descendant"] = (d.run / "descendant.pid").string();
        const pid_t o = d.owner();
        d.wait_state({"running"});
        const int leader = d.engine_pid();
        const int descendant = std::stoi(text(d.run / "descendant.pid"));
        ::kill(o, SIGKILL);
        exit_code(o);
        ::kill(leader, SIGKILL);   // the leader only: the descendant stays in its group
        for (int i = 0; i < 100 && !gone(leader); ++i) std::this_thread::sleep_for(20ms);
        assert(gone(leader) && !gone(descendant));
        const auto left = ss::alive_recorded(d.record());
        assert(left.size() == 1 && left[0].at("pid") == leader);
        ::kill(descendant, SIGKILL);
        for (int i = 0; i < 100 && !gone(descendant); ++i) std::this_thread::sleep_for(20ms);
        assert(ss::alive_recorded(d.record()).empty());
    }
    for (int i = 0; i < 100 && !gone(engine); ++i) std::this_thread::sleep_for(20ms);
    assert(ss::alive_recorded(c.record()).empty());
    Case after("killed-after");
    after.session.run_dir = after.run = c.run;
    after.port = c.port;
    const pid_t again = after.owner();
    for (int i = 0; i < 500 && !(after.record().value("owner_pid", 0) == again &&
                                 after.record().value("state", "") == "running"); ++i)
        std::this_thread::sleep_for(20ms);   // the killed owner's record says running too: wait for this owner's
    assert(after.record().value("owner_pid", 0) == again);
    assert(ss::control(c.run, "stop", 10s)->value("ok", false));
    assert(exit_code(again) == 0);
}

void no_record_no_session() {
    // the record cannot be written at all: no session starts (nothing could find its processes later)
    Case c("no-record");
    fs::create_directories(c.run / "session.json.tmp");   // the atomic write's temporary is a directory: writes fail
    assert(exit_code(c.owner()) == 1 && !fs::exists(c.run / "session.json"));
    for (const auto& e : fs::directory_iterator(c.run)) assert(e.path().filename().string().rfind("engine-", 0) != 0);
}

void unwritable_record_still_stops() {
    // the record cannot be written any more (a full or read-only run directory): stopping goes ahead regardless
    Case c("readonly");
    const pid_t owner = c.owner();
    c.wait_state({"running"});
    const int engine = c.engine_pid();
    ::chmod(c.run.c_str(), 0500);
    const auto reply = ss::control(c.run, "stop", 10s);
    ::chmod(c.run.c_str(), 0700);
    assert(reply && reply->value("ok", false) && reply->at("state") == "stopped");
    assert(gone(engine) && exit_code(owner) == 0);
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
        s.plan = [port](const fs::path& rd, const ss::Hooks&) {
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
    long_run_dir();
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
    descendant_outliving_the_engine();
    inherited_descriptor_not_passed();
    owner_killed_fails_closed();
    unwritable_record_still_stops();
    no_record_no_session();
    std::cout << "PASS: session owner lifecycle (start order, canary, failures at every stage, stop via socket and "
                 "signal, SIGKILL fallback, one owner per run directory, child environment, detached start)\n";
    return 0;
}
