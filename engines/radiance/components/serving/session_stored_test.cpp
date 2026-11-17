// RSC02 S5: real session::serve -> replay_stored_manifest -> live admission
// -> durable owner power admission -> helper/engine -> verified stop.
// Only host facts and process binaries are injected. No fake planner and no GPUs.
#include "jcs.h"
#include "session.h"

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

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef BC_FAKE_ENGINE
#error "BC_FAKE_ENGINE must be an executable test engine"
#endif
#ifndef BC_CONTRACT_V2_DOCS_DIR
#error "BC_CONTRACT_V2_DOCS_DIR must reference the checked-in manifest"
#endif

namespace fs = std::filesystem;
namespace ss = bc::serving::session;
namespace v2 = bc::serving::v2;
using Json = nlohmann::json;
using namespace std::chrono_literals;

namespace {
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
    void disarm() {
        pid = 0;
    }
};
struct Temp {
    fs::path root;
    Temp() {
        std::string templ = (fs::temp_directory_path() / "bc-stored-owner-XXXXXX").string();
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
int port() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    check(fd >= 0, "socket");
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    check(::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0, "bind");
    socklen_t len = sizeof a;
    check(::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len) == 0, "getsockname");
    ::close(fd);
    return ntohs(a.sin_port);
}
Json fixture(int selected_port, const std::string& shm) {
    const auto path = fs::path(BC_CONTRACT_V2_DOCS_DIR) / "example.flashnext-4gpu.json";
    Json document = Json::parse(read(path));
    document["spec"]["processes"]["engine"]["server"]["port"] = selected_port;
    document["spec"]["handoff"]["transport"]["shm"] = shm;
    std::string digest, why;
    document["metadata"].erase("content_digest");
    check(bc::serving::manifest_content_digest(document, &digest, &why), "digest: " + why);
    document["metadata"]["content_digest"] = digest;
    const auto parsed = v2::read_manifest(document.dump(2), path.string(), v2::PlanUse::serving);
    if (!parsed)
        throw std::runtime_error(v2::render_human("fixture", parsed.error()));
    return document;
}
ss::ReplayEvidence evidence(const Json& doc) {
    ss::ReplayEvidence result;
    result.producer_sha256 = doc["metadata"]["producer"]["build_sha256"];
    result.model_identity = doc["spec"]["plan"]["engine_memory"]["model_identity"];
    result.engine_identity = doc["spec"]["plan"]["engine_memory"]["engine_identity"];
    result.effective_spec = doc["spec"];
    Json gpus = Json::array();
    for (const auto& g : doc["spec"]["facts"]["gpus"])
        gpus.push_back({{"bdf", g["bdf"]},
                        {"arch", g["arch"]},
                        {"hip_ordinal", g["hip_ordinal"]},
                        {"vram_total_mib", g["vram"]["total_mib"]},
                        {"vram_available_mib", g["vram"]["total_mib"]}});
    result.fresh_facts = {{"gpus", gpus}, {"host", {{"memlock", {{"limit_mib", nullptr}}}}}};
    return result;
}

struct Case {
    Temp tmp;
    fs::path stored = tmp.root / "saved.json";
    fs::path run = tmp.root / "run";
    fs::path trace = tmp.root / "trace";
    Json original;
    std::string bytes;
    ss::EnginePlanInputs inputs;
    ss::Session session;

    explicit Case(int case_id, bool bad_build = false, bool bad_room = false, bool refuse_power = false,
                  fs::path existing_manifest = {}, fs::path existing_run = {}, fs::path existing_trace = {}) {
        if (!existing_manifest.empty()) {
            stored = std::move(existing_manifest);
            run = std::move(existing_run);
            trace = std::move(existing_trace);
            bytes = read(stored);
            original = Json::parse(bytes);
        } else {
            original = fixture(port(), "/r3-serve-cardfree-" + std::to_string(::getpid()) + "-" +
                                           std::to_string(case_id));
            bytes = original.dump(2) + "\n";
            std::ofstream(stored, std::ios::binary) << bytes;
        }
        fs::create_directories(run);
        const ss::ReplayEvidence facts = evidence(original);
        inputs.stored_manifest = stored;
        inputs.engine_binary = BC_FAKE_ENGINE;
        inputs.helper_binary = BC_FAKE_ENGINE;
        inputs.self = "/tmp/cardfree-bc-serve";
        inputs.replay_evidence = [facts, bad_build](const v2::Manifest&) {
            auto snapshot = facts;
            if (bad_build)
                snapshot.producer_sha256 = std::string(64, 'd');
            return snapshot;
        };
        inputs.replay_live_room = [bad_room](const v2::Manifest&, const Json&) {
            v2::P1LiveRoom room{};
            room.mem_available_bytes = bad_room ? 1 : (281474976710656LL);
            room.cgroup_headroom_bytes = 281474976710656LL;
            room.pin_available_bytes = 281474976710656LL;
            return room;
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
        // Real owner hooks: read the same strict-v2 manifest, make admission
        // durable in the session record, then allow the GPU child gate.
        session.power_hook.begin_manifest = [this, refuse_power](const fs::path& m,
                                                                 v2::PlanUse use) -> v2::Status {
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

    void run_detached() const {
        const fs::path self = fs::read_symlink("/proc/self/exe");
        // Real start boundary: fork/exec the owner with its stored file,
        // then use its control socket for a verified stop. The same owner and
        // replay planner execute in the new process, not a fake plan callback.
        const std::vector<std::string> args = {self.string(), "--owner-stored", stored.string(), run.string(),
                                               trace.string()};
        check(ss::start_detached(args, run, 20s) == 0, "detached stored replay did not reach running");
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
        check(final.at("selection").at("replay") == true,
              "detached owner must select authenticated stored replay");
        const auto& events = final.at("events");
        size_t admitted = events.size(), first_child = events.size();
        for (size_t i = 0; i < events.size(); ++i) {
            const std::string kind = events[i].at("kind").get<std::string>();
            if (kind == "power_applied")
                admitted = i;
            if (kind == "started")
                first_child = std::min(first_child, i);
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
        const auto deadline = std::chrono::steady_clock::now() + 15s;
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
        check(read(stored) == bytes, "stored manifest changed after replay attempt");
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
                if (kind == "started")
                    first_child = std::min(first_child, i);
            }
            check(first_child < events.size() && admitted < first_child,
                  "durable power admission must precede every recorded child");
            bool has_engine = false, has_helper0 = false, has_helper1 = false;
            for (const auto& process : final.at("processes")) {
                const std::string role = process.at("role").get<std::string>();
                has_engine = has_engine || role == "engine";
                has_helper0 = has_helper0 || role == "helper0";
                has_helper1 = has_helper1 || role == "helper1";
                check(role != "manifest writer" && process.value("exited", false),
                      "replay never writes a manifest and reaps every child");
            }
            check(has_engine && has_helper0 && has_helper1,
                  "actual replay must launch all helper and engine roles");
            const auto log = read(trace);
            check(log.find("begin\n") < log.find("verify\n") && log.find("verify\n") < log.find("restore\n"),
                  "power admission, verification and restoration must be ordered");
            check(log.find("verified_exit\n") < log.find("restore\n"),
                  "restore must follow verified child exit");
        } else {
            check(!final.contains("selection") || final.at("selection").is_null(),
                  "pre-start refusal cannot claim a serving selection");
            if (final.contains("processes"))
                check(final.at("processes").empty(), "no child may start after static, live or power refusal");
        }
        return WEXITSTATUS(status);
    }
};
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 5 && std::string(argv[1]) == "--owner-stored") {
            Case owner(0, false, false, false, argv[2], argv[3], argv[4]);
            return ss::serve(owner.session);
        }
        check(argc == 1, "unexpected command line in stored replay acceptance test");
        Case healthy(1);
        healthy.run_owner(true);
        Case changed_build(2, true);
        changed_build.run_owner(false);
        Case no_room(3, false, true);
        no_room.run_owner(false);
        Case power_refused(4, false, false, true);
        power_refused.run_owner(false);
        Case missing_power_hook(5);
        missing_power_hook.session.power_hook.begin_manifest = {};
        missing_power_hook.session.power_hook.verify_serving = {};
        missing_power_hook.session.power_hook.verified_exit = {};
        missing_power_hook.session.power_hook.restore = {};
        missing_power_hook.run_owner(false);
        Case altered_bytes(6);
        altered_bytes.bytes = Json({{"bad", "tampered replacement"}}).dump() + "\n";
        std::ofstream(altered_bytes.stored, std::ios::trunc) << altered_bytes.bytes;
        altered_bytes.run_owner(false);
        Case detached(7);
        detached.run_detached();
        // RED P1-4: a digest-correct manifest B can atomically replace the
        // admitted manifest A pathname between the owner and the strict child
        // reader. The child must still consume A, byte for byte.
        Case pinned(10);
        ss::Hooks pin_hooks;
        pin_hooks.wait = [](std::chrono::milliseconds) { return true; };
        pin_hooks.admit_power = [](const fs::path&, v2::PlanUse) -> v2::Status { return std::nullopt; };
        const ss::EngineStart admitted = ss::replay_stored_manifest(pinned.inputs, pinned.run, pin_hooks);
        check(!admitted.stored_sha256.empty(), "stored bytes must have an admission digest");
        const std::string alternate_bytes =
            fixture(port(), pinned.original.at("spec").at("handoff").at("transport").at("shm")).dump(2) + "\n";
        check(alternate_bytes != pinned.bytes, "replacement fixture must have different real v2 bytes");
        const fs::path replacement = pinned.tmp.root / "replacement.json";
        std::ofstream(replacement, std::ios::binary) << alternate_bytes;
        fs::rename(replacement, pinned.stored);
        check(read(pinned.stored) == alternate_bytes, "pathname replacement must succeed");
        const auto child_bytes = read(admitted.argv.at(2));
        const auto child_manifest = v2::read_manifest(child_bytes, admitted.argv.at(2), v2::PlanUse::serving);
        check(bool(child_manifest), "the actual v2 child reader must accept its bound manifest");
        check(child_bytes == pinned.bytes, "child reader must consume exactly the bytes the owner admitted");

        // Exact P1-4 TOCTOU test: the OS/process boundary swaps the public
        // pathname AFTER the owner's final SHA check and BEFORE first exec.
        // The real v2 reader, through the child's actual --manifest argument,
        // must still see A while the public pathname names valid B.
        Case during_exec(11);
        const fs::path new_manifest = during_exec.tmp.root / "valid-b.json";
        const fs::path restore_a = during_exec.tmp.root / "valid-a-backup.json";
        std::ofstream(new_manifest, std::ios::binary)
            << fixture(port(), during_exec.original.at("spec").at("handoff").at("transport").at("shm")).dump(2)
            << "\n";
        std::ofstream(restore_a, std::ios::binary) << during_exec.bytes;
        during_exec.session.before_gpu_start = [stored = during_exec.stored, a = during_exec.bytes,
                                                new_manifest,
                                                restore_a](const ss::EngineStart& plan) -> v2::Status {
            fs::rename(new_manifest, stored);
            const std::string actual_child_bytes = read(plan.argv.at(2));
            const auto consumed = v2::read_manifest(actual_child_bytes, plan.argv.at(2), v2::PlanUse::serving);
            if (!consumed || actual_child_bytes != a)
                return v2::make_error(v2::ConfigErrorCode::plan_mismatch, "test.before_gpu_start",
                                      "strict child reader followed swapped public manifest");
            fs::rename(restore_a, stored);
            return std::nullopt;
        };
        during_exec.run_owner(true);

        // RED P1-3: the same immutable helper manifest must never grant
        // transport ownership to two distinct run directories at once.
        Case first_transport(8);
        const pid_t running_owner = ::fork();
        check(running_owner >= 0, "fork transport owner");
        if (running_owner == 0)
            std::_Exit(ss::serve(first_transport.session));
        OwnerGuard stop_guard{running_owner, first_transport.run};
        bool first_running = false;
        const auto deadline = std::chrono::steady_clock::now() + 15s;
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
        Case second_transport(9, false, false, false, first_transport.stored,
                              first_transport.tmp.root / "other-run", first_transport.tmp.root / "other-trace");
        second_transport.run_owner(false);
        const auto second_lookup = ss::read_record(second_transport.run);
        check(second_lookup.has_value(), "second owner must leave a durable refusal record");
        const Json second_record = second_lookup.value_or(Json::object());
        check(second_record.at("processes").empty(),
              "cross-root transport collision must refuse before any child starts");
        bool transport_refusal = false;
        for (const auto& error : second_record.value("errors", Json::array()))
            if (error.is_string() &&
                error.get<std::string>().find("shared-memory transport") != std::string::npos)
                transport_refusal = true;
        check(transport_refusal, "cross-root refusal must name the shared-memory transport");
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
