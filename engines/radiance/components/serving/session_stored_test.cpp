// RSC02 S5 (native v3): real session::serve -> stored-manifest replay -> live admission -> durable owner power
// admission -> helpers and engine -> verified stop. Only the host (a fake sysfs/procfs tree) and the process binaries
// (the fake engine) are injected: no fake planner and no GPUs. One fresh native plan with two helpers is measured and
// stored once; every case replays that stored manifest.
//
// Run with --installed-engine PATH (bin/radiance of an unpacked release): the same replay, with the installed engine,
// its bc-serve and its runtime home in place of the build's (the release installation test's card-free start gate).
#include "jcs.h"
#include "session.h"
#include "sha256.h"
#include "test_native_session_fixture.h"

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace ss = bc::serving::session;
namespace v2 = bc::serving::v2;
using Json = nlohmann::json;
using namespace std::chrono_literals;

namespace {
std::optional<fs::path> g_installed_engine;

void check(bool ok, const std::string& why) {
    if (!ok)
        throw std::runtime_error(why);
}
// Failure teardown is bounded even if an owner cannot stop.
pid_t reap_for(pid_t pid, int* status, std::chrono::milliseconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    do {
        const pid_t r = ::waitpid(pid, status, WNOHANG);
        if (r != 0)
            return r;
        std::this_thread::sleep_for(20ms);
    } while (std::chrono::steady_clock::now() < deadline);
    return 0;
}
struct OwnerGuard {
    pid_t pid;
    fs::path run;
    ~OwnerGuard() {
        if (pid <= 0)
            return;
        try {
            (void)ss::control(run, "stop", 500ms);
        } catch (...) {
            std::cerr << "stored owner test: control stop unavailable; escalating cleanup\n";
        }
        int status = 0;
        if (reap_for(pid, &status, 3s) == pid)
            return;
        (void)::kill(pid, SIGTERM);
        if (reap_for(pid, &status, 2s) == pid)
            return;
        (void)::kill(pid, SIGKILL);
        (void)reap_for(pid, &status, 2s);
    }
    void disarm() { pid = 0; }
};
struct Temp {
    fs::path root;
    Temp() {
        // beside the test binary: a copied engine must stay executable (the system temp directory may be noexec)
        std::string templ = (fs::current_path() / "bc-stored-owner-XXXXXX").string();
        std::vector<char> chars(templ.begin(), templ.end());
        chars.push_back('\0');
        const char* p = ::mkdtemp(chars.data());
        if (!p)
            throw std::runtime_error("mkdtemp failed");
        root = p;
    }
    ~Temp() {
        std::error_code ignore;
        fs::remove_all(root, ignore);
    }
};
std::string read(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// The card-free host and the files of one measured, stored native plan; everything a case needs to replay it.
struct World {
    fs::path host;
    fs::path scratch;   // the plan's own files: site, YAML and its resolution root, work root
    fs::path run;       // where the plan was measured
    int port = 0;
    fs::path stored;    // the durable manifest, with its replay index and producer evidence beside it
    std::string bytes;

    fs::path runtime_home() const {
        return g_installed_engine ? g_installed_engine->parent_path().parent_path() / "radiance_home"
                                  : fs::path();
    }
    bc::test::NativeSessionPlan plan() const {
        auto native = bc::test::native_session_inputs(host, scratch, port, Json::object(),
                                                      bc::test::SessionLayout::helpers, runtime_home());
        // A shipped config leaves the n-gram tier to the engine ("auto"): the measuring run resolves it and the stored
        // manifest records the result, while the draft of every later start still says "auto".
        auto& inspected = native.inputs.native_admission->runtime_overrides["TEST_NATIVE_INSPECT_SETTINGS"];
        Json settings = Json::parse(inspected);
        settings["ngram_placement"] = "auto";
        inspected = settings.dump();
        if (g_installed_engine) {
            native.inputs.engine_binary = *g_installed_engine;
            native.inputs.helper_binary = g_installed_engine->parent_path().parent_path() / "radiance_home" / "helpers" / "r3_moe_worker";
            native.inputs.self = runtime_home() / "bin/bc-serve";
        }
        return native;
    }
};

struct Case {
    fs::path root;   // this case's run directory and trace
    fs::path run = root / "run";
    fs::path trace = root / "trace";
    fs::path stored;
    std::string bytes;
    ss::EnginePlanInputs inputs;
    ss::Session session;

    Case(const World& world, const fs::path& case_root, bool bad_build = false, bool bad_room = false,
         bool refuse_power = false)
        : root(case_root), stored(world.stored), bytes(world.bytes) {
        fs::create_directories(run);
        inputs = world.plan().inputs;
        inputs.stored_manifest = stored;
        if (bad_build) { // the engine on disk is no longer the one that measured the plan
            const fs::path changed = root / "changed-engine";
            fs::copy_file(inputs.engine_binary, changed, fs::copy_options::overwrite_existing);
            std::ofstream(changed, std::ios::binary | std::ios::app) << '\n';
            inputs.engine_binary = changed;
        }
        if (bad_room)
            inputs.replay_live_room = [](const v2::Manifest&, const Json&) {
                return v2::P1LiveRoom{1, 1, 1};
            };
        session.run_dir = run;
        session.timing.helper_timeout = 8s;
        session.timing.helper_interval = 20ms;
        session.timing.health_timeout = 8s;
        session.timing.health_interval = 25ms;
        session.timing.canary_timeout = 5s;
        session.timing.stop_grace = 1s;
        session.timing.poll = 20ms;
        session.plan = [in = inputs](const fs::path& rd, const ss::Hooks& hooks) {
            return ss::replay_stored_manifest(in, rd, hooks);
        };
        // Real owner hooks: read the same strict manifest, make admission durable in the session record, then allow
        // the GPU child gate.
        session.power_hook.begin_manifest = [this, refuse_power](const fs::path& m,
                                                                 const v2::Manifest&, v2::PlanUse use) -> v2::Status {
            std::ofstream(trace, std::ios::app) << "begin\n";
            const auto checked = v2::load_manifest(m, use);
            if (!checked)
                return checked.error();
            if (refuse_power)
                return v2::make_error(v2::ConfigErrorCode::admission_refused, "session.power",
                                      "injected power preflight refusal");
            return std::nullopt;
        };
        session.power_hook.verify_serving = [this](const fs::path& m) -> v2::Status {
            std::ofstream(trace, std::ios::app) << "verify\n";
            const auto checked = v2::load_manifest(m, v2::PlanUse::serving);
            if (!checked)
                return checked.error();
            return std::nullopt;
        };
        session.power_hook.gfxoff_window = [this](bool loaded) -> v2::Status {
            std::ofstream(trace, std::ios::app) << (loaded ? "idle\n" : "floor\n");
            return std::nullopt;
        };
        session.power_hook.verified_exit = [this] { std::ofstream(trace, std::ios::app) << "verified_exit\n"; };
        session.power_hook.restore = [this]() -> v2::Status {
            std::ofstream(trace, std::ios::app) << "restore\n";
            return std::nullopt;
        };
    }

    // The roles of the recorded children, in start order.
    static std::vector<std::string> roles(const Json& record) {
        std::vector<std::string> out;
        for (const auto& process : record.value("processes", Json::array()))
            out.push_back(process.at("role").get<std::string>());
        return out;
    }
    // Nothing that serves started: only planning children (the two config inspections), all of them reaped.
    static void check_no_serving_child(const Json& record, const std::string& why) {
        for (const auto& process : record.value("processes", Json::array())) {
            const std::string role = process.at("role").get<std::string>();
            check(role == "native config inspector" && process.value("exited", false), why + ": " + role);
        }
    }

    void run_detached(const World& world) const {
        const fs::path self = fs::read_symlink("/proc/self/exe");
        // Real start boundary: fork/exec the owner with its stored file, then use its control socket for a verified
        // stop. The same owner and replay planner execute in the new process, not a fake plan callback.
        std::vector<std::string> args = {self.string(),   "--owner-stored", stored.string(),
                                         run.string(),    trace.string(),   world.host.string(),
                                         world.scratch.string(), std::to_string(world.port)};
        if (g_installed_engine)
            args.push_back(g_installed_engine->string());
        check(ss::start_detached(args, run, 60s) == 0, "detached stored replay did not reach running");
        const auto reply = ss::control(run, "stop", 10s);
        check(reply && reply->value("ok", false), "detached stored replay did not stop cleanly");
        check(read(stored) == bytes, "detached start rewrote stored bytes");
        const auto completed = ss::read_record(run);
        check(completed.has_value(), "detached owner must persist its terminal record");
        const Json final = completed.value_or(Json::object());
        int owner_status = 0;
        const pid_t owner_pid = final.at("owner_pid").get<pid_t>();
        check(reap_for(owner_pid, &owner_status, 15s) == owner_pid && WIFEXITED(owner_status) &&
                  WEXITSTATUS(owner_status) == 0,
              "detached stored replay owner must exit cleanly");
        check(final.at("selection").at("replay") == true, "detached owner must select authenticated stored replay");
        const auto& events = final.at("events");
        size_t admitted = events.size(), first_child = events.size();
        for (size_t i = 0; i < events.size(); ++i) {
            const std::string kind = events[i].at("kind").get<std::string>();
            if (kind == "power_applied")
                admitted = i;
            if (kind == "started" && events[i].value("role", std::string()) != "native config inspector")
                first_child = std::min(first_child, i); // the config inspections are CPU-only, before admission
        }
        check(admitted < first_child && first_child < events.size(),
              "detached start must durably admit power before its first GPU child");
        const std::string log = read(trace);
        check(log.find("begin\n") < log.find("verify\n") && log.find("verified_exit\n") < log.find("restore\n"),
              "detached replay must restore power only after verified exit");
    }

    int run_owner(bool succeed) const {
        const pid_t pid = ::fork();
        check(pid >= 0, "fork");
        if (pid == 0) {
            int result = 1;
            try {
                result = ss::serve(session);
            } catch (const std::exception& e) {
                std::cerr << "owner: " << e.what() << '\n';
            }
            std::_Exit(result);
        }
        OwnerGuard owner_guard{pid, run};
        bool reached = false;
        const auto deadline = std::chrono::steady_clock::now() + 30s;
        std::string state;
        while (std::chrono::steady_clock::now() < deadline) {
            const auto record = ss::read_record(run);
            state = record ? record->value("state", "") : "";
            if (state == "running" || state == "failed" || state == "stopped") {
                reached = true;
                break;
            }
            std::this_thread::sleep_for(20ms);
        }
        if (!reached)
            throw std::runtime_error("owner session timed out in state " + state);
        const auto observed = ss::read_record(run);
        check(observed.has_value(), "owner must have a durable record");
        const Json record = observed.value_or(Json::object());
        if (succeed) {
            check(state == "running", "replay failed before running: " + record.dump(2));
            auto reply = ss::control(run, "stop", 10s);
            check(reply && reply->value("ok", false), "control-socket stop failed");
        } else {
            check(state == "failed", "refused replay started the engine: " + record.dump(2));
        }
        int status = 0;
        const pid_t reaped = reap_for(pid, &status, 15s);
        if (reaped == pid)
            owner_guard.disarm();
        check(reaped == pid, "owner did not exit within 15 s");
        check(WIFEXITED(status) && (WEXITSTATUS(status) == 0) == succeed, "owner return code");
        check(read(stored) == bytes, "stored manifest changed after replay attempt: " + stored.string() + " " +
                                         std::to_string(read(stored).size()) + " vs " + std::to_string(bytes.size()));
        const auto final_record = ss::read_record(run);
        check(final_record.has_value(), "owner must persist terminal record");
        const Json final = final_record.value_or(Json::object());
        if (succeed) {
            check(final.at("power_admission").at("origin_manifest") == stored.string(),
                  "power execution must retain the authenticated origin pathname");
            check(final.at("power_admission").at("manifest") != stored.string(),
                  "power must admit the pinned execution descriptor, never the original pathname");
            check(final.at("power_admission").at("original_sha256").is_string(),
                  "power execution must keep the original byte identity");
            check(final.at("selection").at("replay") == true, "run record must distinguish replay");
            const auto& events = final.at("events");
            size_t admitted = events.size(), first_child = events.size();
            for (size_t i = 0; i < events.size(); ++i) {
                const auto kind = events[i].at("kind").get<std::string>();
                if (kind == "power_applied")
                    admitted = i;
                if (kind == "started" && events[i].value("role", std::string()) != "native config inspector")
                    first_child = std::min(first_child, i); // the config inspections are CPU-only, before admission
            }
            check(first_child < events.size() && admitted < first_child,
                  "durable power admission must precede every recorded child");
            for (const auto& process : final.at("processes"))
                check(process.value("exited", false), "replay reaps every child");
            check(roles(final) == std::vector<std::string>{"native config inspector", "native config inspector",
                                                           "helper0", "helper1", "engine"},
                  "actual replay must inspect the config, then launch all helper and engine roles and write no "
                  "manifest: " + Json(roles(final)).dump());
            const auto log = read(trace);
            check(log.find("begin\n") < log.find("verify\n") && log.find("verify\n") < log.find("restore\n"),
                  "power admission, verification and restoration must be ordered");
            check(log.find("verified_exit\n") < log.find("restore\n"), "restore must follow verified child exit");
        } else {
            check(!final.contains("selection") || final.at("selection").is_null(),
                  "pre-start refusal cannot claim a serving selection");
            check_no_serving_child(final, "no helper or engine may start after a static, live or power refusal");
        }
        return WEXITSTATUS(status);
    }

    bool refused_with(const std::string& needle) const {
        const auto record = ss::read_record(run);
        if (!record)
            return false;
        for (const auto& error : record->value("errors", Json::array()))
            if (error.is_string() && error.get<std::string>().find(needle) != std::string::npos)
                return true;
        return false;
    }
};

// One fresh native plan with both helpers, measured (the fake engine) and stored under the world's run directory.
World make_world(const fs::path& root) {
    World world;
    world.host = root / "host";
    bc::test::write_fake_host(world.host, /*unlimited_memlock=*/true);
    world.scratch = root / "plan";
    world.port = bc::test::free_port();
    world.run = root / "measured";
    fs::create_directories(world.run);
    const auto native = world.plan();
    const ss::EngineStart plan = ss::plan_engine_only(native.inputs, world.run, bc::test::standalone_plan_hooks());
    world.stored = plan.manifest;
    world.bytes = read(world.stored);
    check(!plan.helpers.empty() && fs::exists(world.stored.string() + ".replay-index.json"),
          "the fresh native plan must store a manifest with helpers and its replay index");
    return world;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if ((argc == 8 || argc == 9) && std::string(argv[1]) == "--owner-stored") {
            // The detached owner of Case::run_detached: stored manifest, run dir, trace, host, plan files, port.
            if (argc == 9)
                g_installed_engine = argv[8];
            World world;
            world.stored = argv[2];
            world.host = argv[5];
            world.scratch = argv[6];
            world.port = std::stoi(argv[7]);
            world.bytes = read(world.stored);
            Case owner(world, fs::path(argv[3]).parent_path(), false, false, false);
            owner.run = argv[3];
            owner.trace = argv[4];
            owner.session.run_dir = owner.run;
            return ss::serve(owner.session);
        }
        if (argc == 3 && std::string(argv[1]) == "--installed-engine")
            g_installed_engine = argv[2];
        else
            check(argc == 1, "unexpected command line in stored replay acceptance test");

        Temp tmp;
        const World world = make_world(tmp.root);
        int id = 0;
        const auto next_root = [&] { return tmp.root / ("case-" + std::to_string(++id)); };

        Case healthy(world, next_root());
        healthy.run_owner(true);

        Case changed_build(world, next_root(), /*bad_build=*/true);
        changed_build.run_owner(false);
        // An installed prefix refuses a foreign engine earlier still: it must belong to the build that owns it.
        check(changed_build.refused_with(g_installed_engine ? "must belong to this bc-serve build"
                                                            : "engine, kernels or helper build changed"),
              "a changed engine build must be refused by name");

        Case no_room(world, next_root(), false, /*bad_room=*/true);
        no_room.run_owner(false);
        check(no_room.refused_with("live MemAvailable/cgroup room"), "a stored plan must be refused without live room");

        Case power_refused(world, next_root(), false, false, /*refuse_power=*/true);
        power_refused.run_owner(false);
        check(power_refused.refused_with("injected power preflight refusal"),
              "a refused power preflight must end the session before any serving child");

        Case missing_power_hook(world, next_root());
        missing_power_hook.session.power_hook.begin_manifest = {};
        missing_power_hook.session.power_hook.verify_serving = {};
        missing_power_hook.session.power_hook.verified_exit = {};
        missing_power_hook.session.power_hook.restore = {};
        missing_power_hook.run_owner(false);
        check(missing_power_hook.refused_with("missing required durable power admission"),
              "a replay without the owner's durable power admission must be refused");

        // The stored bytes are altered (the manifest is read-only: replaced by rename) and restored: the replay index
        // no longer binds them.
        {
            const auto replace_stored = [&](const std::string& content) {
                const fs::path replacement = world.stored.string() + ".fixture-replacement";
                std::ofstream(replacement, std::ios::binary) << content;
                fs::rename(replacement, world.stored);
            };
            Case altered(world, next_root());
            altered.bytes = world.bytes + "\n";
            replace_stored(altered.bytes);
            try {
                altered.run_owner(false);
            } catch (...) {
                replace_stored(world.bytes);
                throw;
            }
            replace_stored(world.bytes);
            check(altered.refused_with("no longer binds the original manifest bytes"),
                  "altered stored bytes must be refused by the replay index");
        }

        Case detached(world, next_root());
        detached.run_detached(world);

        // A digest-correct manifest B can atomically replace the admitted manifest A pathname between the owner and
        // the strict child reader. The child must still consume A, byte for byte: the swap happens after the owner's
        // final SHA check and before the first exec, and the engine's own --config argument must still read A's YAML.
        Case during_exec(world, next_root());
        const fs::path swapped = during_exec.root / "swapped.json";
        const fs::path restore_a = during_exec.root / "valid-a-backup.json";
        std::ofstream(swapped, std::ios::binary) << "{\"swapped\":true}\n";
        std::ofstream(restore_a, std::ios::binary) << during_exec.bytes;
        during_exec.session.before_gpu_start = [stored = during_exec.stored, a = during_exec.bytes, swapped,
                                                restore_a](const ss::EngineStart& plan) -> v2::Status {
            fs::rename(swapped, stored);
            // The engine's --config argument is the sealed YAML of A, not any public path.
            const std::string child_yaml = read(plan.argv.at(2));
            const auto admitted_a = v2::read_manifest(a, "valid-a", v2::PlanUse::serving);
            const auto native_a = admitted_a ? v2::native_config_view(admitted_a.value()) : std::nullopt;
            if (plan.argv.at(1) != "--config" || !native_a || child_yaml != native_a->yaml_utf8)
                return v2::make_error(v2::ConfigErrorCode::plan_mismatch, "test.before_gpu_start",
                                      "strict child reader followed the swapped public manifest or lost the admitted YAML");
            fs::rename(restore_a, stored);
            return std::nullopt;
        };
        during_exec.run_owner(true);

        // The same immutable helper manifest must never grant transport ownership to two run directories at once.
        Case first_transport(world, next_root());
        const pid_t running_owner = ::fork();
        check(running_owner >= 0, "fork transport owner");
        if (running_owner == 0)
            std::_Exit(ss::serve(first_transport.session));
        OwnerGuard stop_guard{running_owner, first_transport.run};
        bool first_running = false;
        const auto deadline = std::chrono::steady_clock::now() + 30s;
        while (std::chrono::steady_clock::now() < deadline) {
            const auto rec = ss::read_record(first_transport.run);
            if (rec && rec->value("state", std::string()) == "running") {
                first_running = true;
                break;
            }
            if (rec && rec->value("state", std::string()) == "failed")
                break;
            std::this_thread::sleep_for(25ms);
        }
        check(first_running, "first owner must be running before cross-root transport collision");
        Case second_transport(world, next_root());
        second_transport.run_owner(false);
        const auto second_lookup = ss::read_record(second_transport.run);
        check(second_lookup.has_value(), "second owner must leave a durable refusal record");
        Case::check_no_serving_child(second_lookup.value_or(Json::object()),
                                     "cross-root transport collision must refuse before any helper or engine starts");
        check(second_transport.refused_with("shared-memory transport"),
              "cross-root refusal must name the shared-memory transport");
        const auto stop_first = ss::control(first_transport.run, "stop", 10s);
        check(stop_first && stop_first->value("ok", false), "first owner must stop cleanly");
        int first_status = 0;
        const pid_t reaped_first = reap_for(running_owner, &first_status, 15s);
        if (reaped_first == running_owner)
            stop_guard.disarm();
        check(reaped_first == running_owner && WIFEXITED(first_status) && WEXITSTATUS(first_status) == 0,
              "first owner must keep transport while running and exit cleanly");
        std::cout << "production and detached owner replay: success, static/live/power/missing-gate/tamper "
                     "refusals passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "RSC02 S5 owner integration: " << e.what() << '\n';
        return 1;
    }
}
