#pragma once
/* RSC02 step 1: the serving session owner, engine-only.
 *
 * `bc-serve serve` is one foreground process that owns a whole serving session: it takes the session lock in the run
 * directory, makes the measured manifest (measuring first on a plan-cache miss), starts the engine from that manifest in
 * its own process group, waits for /health, sends the start canary, then answers `status` and `stop` on a control socket
 * in the run directory until a stop request, a signal or a child's exit. It then stops every child in order (SIGINT,
 * bounded wait, SIGKILL, bounded wait) and reports success only when every child has verifiably exited.
 * `bc-serve start` is `serve` detached (setsid), returning once the session is running or has failed; `stop` and
 * `status` talk to the owner over the socket only, never by signalling a recorded PID (a PID can be reused).
 *
 * A child gets its configuration from its manifest only: its environment is the owner's, with every bigcherry setting
 * variable (R3_*, RAD_*, BC_*, BIGCHERRY_*) and every device-visibility variable removed, plus the runtime's own
 * (RADIANCE_HOME and the device visibility of its cards). Helpers, power, the lab modes and the guards are later steps
 * (RSC02 steps 2-4); the reference for the behaviour is the Python supervisor (tools/bigcherry/serving/supervisor.py). */
#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/types.h>

#include <nlohmann/json.hpp>

#include "serving.h"

namespace bc::serving::session {

using Env = std::map<std::string, std::string>;

/* --- children -------------------------------------------------------------------------------------------------- */

/* The environment a child starts with: `inherited` (the owner's, NAME=VALUE) without any bigcherry setting or
 * device-visibility variable, then `runtime` on top. */
Env child_environment(char** inherited, const Env& runtime);

/* Start argv[0] (a path, never searched) in a new session and process group with exactly `env`, stdin from /dev/null,
 * stdout and stderr appended to `log`. Throws, naming the program, when it cannot be started (exec failure included). */
pid_t spawn(const std::vector<std::string>& argv, const Env& env, const std::filesystem::path& log);

/* A child of this process: whether it is still running (reaps it when it has exited; its status goes to `status`). */
bool running(pid_t pid, int* status = nullptr);

/* Stop a child's process group: `first` (SIGINT for the engine), up to `grace` for it to exit, then SIGKILL and up to
 * `grace` again. True only when the child has exited and been reaped and no process is left in its group. */
bool stop_child(pid_t pid, int first, std::chrono::milliseconds grace);

/* A process's start time (clock ticks since boot, /proc/PID/stat field 22): with the PID, it names one process even
 * after the PID is reused. Nothing when the process is gone. */
std::optional<uint64_t> start_time(pid_t pid);

/* The processes a session record lists that are still the same processes (PID and start time): an owner that died
 * without stopping them leaves them here. */
std::vector<nlohmann::json> alive_recorded(const nlohmann::json& record);

/* --- the engine's HTTP endpoint ---------------------------------------------------------------------------------- */

struct HttpReply {
    int status = 0;
    std::string body;
};
/* One HTTP/1.1 request to 127.0.0.1:port; nothing on a connection or protocol failure or after `timeout`. */
std::optional<HttpReply> http(int port, const std::string& method, const std::string& path, const std::string& body,
                              std::chrono::milliseconds timeout);

/* The start canary: a deterministic completion (temperature 0) whose text must contain `expect`, case-insensitively.
 * Nothing when it passes, else why it failed (transport, not a completion, or the answer). */
std::optional<std::string> canary(int port, const std::string& prompt, const std::string& expect,
                                  std::chrono::milliseconds timeout);

/* --- the session ------------------------------------------------------------------------------------------------- */

/* What the plan step hands the owner: the engine's manifest and how to start it. */
struct EngineStart {
    std::filesystem::path manifest;
    std::vector<std::string> argv;   // the engine binary, --manifest M, --role engine; nothing else
    Env runtime;                     // RADIANCE_HOME, ROCR_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES
    int port = 0;
    nlohmann::json summary = nlohmann::json::object();   // what `status` reports about the selection
};

struct Timing {
    std::chrono::milliseconds health_timeout{std::chrono::minutes(15)};
    std::chrono::milliseconds health_interval{std::chrono::seconds(2)};
    std::chrono::milliseconds canary_timeout{std::chrono::minutes(2)};
    std::chrono::milliseconds stop_grace{std::chrono::seconds(60)};
    std::chrono::milliseconds poll{std::chrono::milliseconds(200)};
};

struct Canary {
    std::string prompt = "The capital of France is";
    std::string expect = "Paris";
};

/* Wait up to the given time while the owner answers its control socket; false once a stop is requested. */
using Wait = std::function<bool(std::chrono::milliseconds)>;

/* What the owner gives its plan step: `wait`, and where to register a process the step starts (a measuring engine), so
 * the owner records it and, if the step cannot verify its exit, stops it itself and never reports the session stopped
 * while it lives. */
struct Hooks {
    Wait wait;
    std::function<void(const std::string& role, pid_t pid, const std::filesystem::path& log)> started;
    std::function<void(pid_t pid)> ended;   // the step verified this process's exit
};

/* The plan step refused or could not make the manifest (the session ends with exit 97, nothing started for serving). */
struct PlanFailed : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Session {
    std::filesystem::path run_dir;
    Timing timing;
    Canary canary;
    // The plan step: runs with the session lock held, before any serving process starts (it may run measuring starts
    // of its own, registered through the hooks and stopped before it returns). Throws to refuse.
    std::function<EngineStart(const std::filesystem::path& run_dir, const Hooks& hooks)> plan;
};

/* The owner, in the foreground: 0 after a requested stop with every child exited; non-zero after a refusal, a failed
 * start, a child's exit while serving, or a child that could not be verified to exit. The session record
 * (run_dir/session.json) says which, with every event. */
int serve(const Session& session);

/* `serve` detached: re-executes `self_argv` (this bc-serve with `serve` and its options) in a new session, its output
 * in run_dir/owner.log, and waits up to `timeout` for the session to be running (0) or to have failed (1). */
int start_detached(const std::vector<std::string>& self_argv, const std::filesystem::path& run_dir,
                   std::chrono::milliseconds timeout);

/* The control socket client: "status" or "stop". The owner's reply (JSON), or nothing when no owner answers. */
std::optional<nlohmann::json> control(const std::filesystem::path& run_dir, const std::string& command,
                                      std::chrono::milliseconds timeout);

/* The last session record in the run directory, if any. */
std::optional<nlohmann::json> read_record(const std::filesystem::path& run_dir);

/* --- the engine-only plan step ------------------------------------------------------------------------------------ */

struct ManifestRun {
    int exit = 0;
    std::string out, err;
};
/* Runs this bc-serve (`argv[0]` is its path) to completion and returns its exit and output. Injectable for tests. */
using ManifestRunner = std::function<ManifestRun(const std::vector<std::string>& argv)>;
/* Runs a bc-serve to completion, waiting through `wait` (the owner keeps answering its socket; a stop request
 * ends the child and throws PlanFailed). */
ManifestRun run_to_completion(const std::vector<std::string>& argv, const Env& env, const Wait& wait = {});

struct EnginePlanInputs {
    Options options;                       // the resolver's inputs: site, profile, configs, layouts, model root, --set
    std::filesystem::path self;            // this bc-serve (writes the manifests)
    std::filesystem::path engine_binary;   // empty: <radiance_home>/../bin/radiance
    ManifestRunner runner;                 // empty: run_to_completion with the owner's child environment
};

/* The engine-only plan: resolve the selection (refused when it has helpers or is a layout with server-args, which only
 * the bash launcher understands), then ask bc-serve for the measured manifest of exactly that configuration, measuring
 * it first on a cache miss (exit 10: the engine alone, --plan-memory against the unmeasured manifest) and once more with
 * the derived host pool when bc-serve asks for one (exit 11). The same plan-extra as the bash launcher's engine-only
 * start, so the two share the plan cache. */
EngineStart plan_engine_only(const EnginePlanInputs& inputs, const std::filesystem::path& run_dir, const Hooks& hooks);

} // namespace bc::serving::session
