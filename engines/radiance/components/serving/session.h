#pragma once
/* RSC02: the serving session owner, dense and ordered-helper sessions.
 *
 * `bc-serve serve` is one foreground process that owns a whole serving session: it takes the session lock in
 * the run directory, makes the measured manifest (measuring first on a plan-cache miss), starts the engine from
 * that manifest in its own process group, waits for /health, sends the start canary, then answers `status` and
 * `stop` on a control socket in the run directory until a stop request, a signal or a child's exit. It then
 * stops every child in order (SIGINT, bounded wait, SIGKILL, bounded wait) and reports success only when every
 * child has verifiably exited. `bc-serve start` is `serve` detached (setsid), returning once the session is
 * running or has failed; `stop` and `status` talk to the owner over the socket only, never by signalling a
 * recorded PID (a PID can be reused).
 *
 * A child gets its configuration from its manifest only: its environment is the owner's, with every bigcherry
 * setting variable (R3_*, RAD_*, BC_*, BIGCHERRY_*) and every device-visibility variable removed, plus the
 * runtime's own (RADIANCE_HOME and the device visibility of its cards). Power is an interface for RSC02 step 4;
 * the lab modes and guards are step 3. The Python supervisor remains the behavioural reference through RSC11
 * S16. */
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/types.h>

#include <nlohmann/json.hpp>

#include "callback_boundary.h"
#include "helper_vram.h"
#include "serving.h"
#include "stored_manifest.h"
#include "v2/manifest.h"
#include "v2/native_config_snapshot.h"

namespace bc::serving::power {
class SessionPower;
}

namespace bc::serving::session {

class PinnedManifest; // a sealed, owner-held execution handoff, not a rewritten v2 manifest

using Env = std::map<std::string, std::string>;
// The shared C++17 callback boundary owns every injected callable; no second wrapper.
template <class Signature> using Callback = callback::Guarded<Signature>;

/* --- children
 * -------------------------------------------------------------------------------------------------- */

/* The environment a child starts with: `inherited` (the owner's, NAME=VALUE) without any bigcherry setting or
 * device-visibility variable, then `runtime` on top. */
Env child_environment(char** inherited, const Env& runtime);

/* Start argv[0] (a path, never searched) in a new session and process group with exactly `env`, stdin from
 * /dev/null, stdout and stderr appended to `log`. The child waits before exec until `recorded(pid)` has
 * returned (the caller makes it durable there); when `recorded` throws, or this process dies first, the child
 * exits without running anything and the exception is rethrown. Throws, naming the program, when it cannot be
 * started (exec failure included). */
pid_t spawn(const std::vector<std::string>& argv, const Env& env, const std::filesystem::path& log,
            const Callback<void(pid_t)>& recorded = {}, int inherited_manifest_fd = -1);

/* A child of this process: whether it is still running (reaps it when it has exited; its status goes to
 * `status`). */
bool running(pid_t pid, int* status = nullptr);

/* Stop a child's process group: `first` (SIGINT for the engine), up to `grace` for it to exit, then SIGKILL and
 * up to `grace` again. True only when the child has exited and been reaped and no process is left in its group.
 */
bool stop_child(pid_t pid, int first, std::chrono::milliseconds grace);

/* A process's start time (clock ticks since boot, /proc/PID/stat field 22): with the PID, it names one process
 * even after the PID is reused. Nothing when the process is gone. */
std::optional<uint64_t> start_time(pid_t pid);

/* The processes a session record lists that are still the same processes (PID and start time): an owner that
 * died without stopping them leaves them here. */
std::vector<nlohmann::json> alive_recorded(const nlohmann::json& record);

/* --- the engine's HTTP endpoint
 * ---------------------------------------------------------------------------------- */

struct HttpReply {
    int status = 0;
    std::string body;
};
/* One HTTP/1.1 request to 127.0.0.1:port; nothing on a connection or protocol failure or after `timeout`. */
std::optional<HttpReply> http(int port, const std::string& method, const std::string& path,
                              const std::string& body, std::chrono::milliseconds timeout);

/* The start canary: a deterministic completion (temperature 0) whose text must contain `expect`,
 * case-insensitively. Nothing when it passes, else why it failed (transport, not a completion, or the answer).
 */
std::optional<std::string> canary(int port, const std::string& prompt, const std::string& expect,
                                  std::chrono::milliseconds timeout);

/* --- the session
 * ------------------------------------------------------------------------------------------------- */

/* The role order is manifest order: helper0 makes the shared segment, helper1 attaches to it.
 * Each helper starts with --manifest M --role helperN; device visibility is environment-only. */
struct HelperStart {
    std::string role;
    std::vector<std::string> argv;
    Env runtime;
};

/* What the plan step hands the owner: one manifest and all of its children. */
struct EngineStart {
    std::filesystem::path manifest; // durable writer/replay path, retained for receipts and indexing
    std::string stored_sha256; // original authenticated bytes, verified again against the pinned handoff
    std::shared_ptr<PinnedManifest>
        pinned_manifest;              // keeps a sealed file open across children; null on first start
    std::vector<std::string> argv;    // the engine binary, --manifest M, --role engine; nothing else
    std::vector<HelperStart> helpers; // ordered, prior to the engine
    std::string handoff_shm;          // unique to this owner; non-empty only with helpers
    Env runtime;                      // RADIANCE_HOME, ROCR_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES
    int port = 0;
    // Captured by the owner immediately before the engine's pre-exec fork.
    // A previous session's heat file cannot satisfy this launch's guard.
    std::chrono::system_clock::time_point engine_launched_at{};
    nlohmann::json summary = nlohmann::json::object(); // what `status` reports about the selection
    // Generated by the owner at each serving engine spawn; never persisted in a manifest.
    std::string heat_session_nonce;

    // The exact strict document that was admitted and will be consumed by this role.
    // Stored replay and fresh launches use a sealed descriptor; manifest remains the durable public path.
    std::filesystem::path execution_manifest() const;
};

struct Timing {
    std::chrono::milliseconds helper_timeout{std::chrono::minutes(15)};
    std::chrono::milliseconds helper_interval{std::chrono::milliseconds(200)};
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
using Wait = Callback<bool(std::chrono::milliseconds)>;

// A role is a compile-time authority class, distinct from the event/ledger
// label. Only CPU-only planning children are allowed before power admission.
enum class ChildKind : uint8_t { manifest_writer, native_config_inspector, measuring_helper, measuring_engine };

/* The owner registers every planner child (including the CPU-only writer and native inspector)
 * before its pre-exec gate is released. Every child is reaped and journalled;
 * measuring and serving processes require durable power admission. */
struct Hooks {
    Wait wait;
    Callback<void(ChildKind kind, const std::string& role, pid_t pid, const std::filesystem::path& log)>
        started;
    Callback<void(pid_t pid)> ended;                     // the step verified this process's exit
    Callback<void(const std::string&)> selected_segment; // journal a unique helper segment before measuring
    // Exactly one manifest-authenticated power admission, after the writer
    // resolves selected cards but before any measuring/serving child starts.
    Callback<v2::Status(const std::filesystem::path&, v2::PlanUse)> admit_power;
    // One owner-recorded bound for verified measuring-helper teardown AND counter release.
    std::chrono::milliseconds helper_stop_timeout{std::chrono::seconds(60)};
};

/* The plan step refused or could not make the manifest (the session ends with exit 97, nothing started for
 * serving). */
struct PlanFailed : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct ReplayFailed : PlanFailed {
    ReplayRefusal reason;
    ReplayFailed(ReplayRefusal why, const std::string& detail) : PlanFailed(detail), reason(why) {}
};

/* Step 4 power transaction boundary. Step 2 never writes device settings.
 * Both operations are optional as a pair: apply before measuring and restore only AFTER every child exited,
 * including a partially failed apply. Step 4 supplies the journaled implementation. */
struct PowerHook {
    // Compatible directly with step 4 SessionPower::begin()/finish(): both return v2::Status.
    // A returned failure must be observed, not discarded through a void callback.
    Callback<v2::Status()> apply;
    Callback<v2::Status()> restore;
    // An alternative to apply for actual SessionPower: the planner supplies a
    // strictly authenticated manifest before starting its measuring children.
    Callback<v2::Status(const std::filesystem::path&, v2::PlanUse)> begin_manifest;
    Callback<v2::Status(const std::filesystem::path&)> verify_serving;
    Callback<v2::Status(bool)> gfxoff_window; // false: floor; true: loaded idle
    Callback<void()> verified_exit;           // the owner calls only after verified child exit
};

// Bridge the actual step-4 transaction into the session owner's guarded
// Status-returning hooks. SessionPower must outlive serve().
PowerHook hooks_for_power_transaction(power::SessionPower& transaction);

struct Laboratory {
    std::filesystem::path prompts_a, prompts_b, calibration_store;
};
struct Session {
    PowerHook power_hook;
    std::optional<Laboratory> laboratory; // explicit finite ABBA mode, never a detached serving session
    std::filesystem::path run_dir;
    Timing timing;
    Canary canary;
    // Optional post-canary runtime guard (MTP replay, live heat and policy).
    // The real CLI binds this; owner-only fakes can omit it.
    Callback<v2::Status(const EngineStart&, const std::filesystem::path& engine_log)> runtime_guard;
    // Actual live worker and heat checks, not the manifest's startup decision.
    Callback<v2::Status()> runtime_watch;
    Callback<v2::Status()> lab_evidence_gate;
    // Injectable process boundary after the last authenticated byte check
    // but before fork/exec. Production leaves it empty; a test can replace
    // the operator's pathname at exactly the unsafe TOCTOU boundary.
    Callback<v2::Status(const EngineStart&)> before_gpu_start;
    // The plan step: runs with the session lock held, before any serving process starts (it may run measuring
    // starts of its own, registered through the hooks and stopped before it returns). Throws to refuse.
    Callback<EngineStart(const std::filesystem::path& run_dir, const Hooks& hooks)> plan;
};

/* The owner, in the foreground: 0 after a requested stop with every child exited; non-zero after a refusal, a
 * failed start, a child's exit while serving, or a child that could not be verified to exit. The session
 * record (run_dir/session.json) says which, with every event. */
int serve(const Session& session);

/* `serve` detached: re-executes `self_argv` (this bc-serve with `serve` and its options) in a new session, its
 * output in run_dir/owner.log, and waits up to `timeout` for the session to be running (0) or to have failed
 * (1). */
int start_detached(const std::vector<std::string>& self_argv, const std::filesystem::path& run_dir,
                   std::chrono::milliseconds timeout);

/* The control socket client: "status" or "stop". The owner's reply (JSON), or nothing when no owner answers. */
std::optional<nlohmann::json> control(const std::filesystem::path& run_dir, const std::string& command,
                                      std::chrono::milliseconds timeout);

/* The last session record in the run directory, if any. */
std::optional<nlohmann::json> read_record(const std::filesystem::path& run_dir);

/* --- the engine-only plan step
 * ------------------------------------------------------------------------------------ */

struct ManifestRun {
    int exit = 0;
    std::string out, err;
};
/* Runs this bc-serve (`argv[0]` is its path) to completion and returns its exit and output. Injectable for
 * tests. */
using ManifestRunner = Callback<ManifestRun(const std::vector<std::string>& argv)>;
/* Runs a bc-serve to completion, waiting through `hooks.wait` (the owner keeps answering its socket; a stop
 * request ends the child and throws PlanFailed) and registered through `hooks.started`/`ended` (an exit it
 * cannot verify leaves the child the owner's to stop). */
ManifestRun run_to_completion(const std::vector<std::string>& argv, const Env& env, const Hooks& hooks = {});

struct NativeConfigInspection {
    std::string snapshot_sha256;
    std::filesystem::path resolution_root;
    std::uint32_t abi_version = 0;
    std::string model;
    std::string radiance_home;
    std::optional<std::string> radiance_home_source;
    std::vector<std::string> kernels;
    std::optional<std::string> kernels_source;
    std::string listener_host;
    std::uint16_t listener_port = 0;
    int tp = 0;
    std::vector<int> cards;
    std::vector<std::string> rank_devices;
    int64_t max_context = 0;
    int64_t max_sequences = 0;
};

/* Ask the matched Radiance binary to parse and resolve the admitted native YAML snapshot under its recorded
 * root. `native_environment` supplies the same explicit configuration environment selected for the later child;
 * it is never inferred from a mutable authored path. The inspector is owner-tracked and reaped before return. */
NativeConfigInspection inspect_native_config(const std::filesystem::path& engine,
                                             const NativeConfigSnapshot& snapshot,
                                             const Env& native_environment, const Hooks& hooks = {});

struct EnginePlanInputs {
    Options options;            // the resolver's inputs: site, profile, configs, layouts, model root, --set
    std::filesystem::path self; // this bc-serve (writes the manifests)
    std::filesystem::path stored_manifest; // explicit immutable replay input (never a writer output)
    std::string named_config;              // explicitly selected new RSC12 config; enables static-key reuse
    bool explicit_power_profile = false;   // only an explicit override may conflict with a stored profile
    std::filesystem::path calibration_prompts_a, calibration_prompts_b, calibration_store;
    std::filesystem::path calibration_records_file; // real writer P2 source, part of static input key

    // OS/dependency seams: only the production default calls the real detector,
    // compiled-in resolver, model/build identity and live P1 probe. Tests supply
    // card-free facts but still call replay_stored_manifest through Session::plan.
    Callback<ReplayEvidence(const v2::Manifest&)> replay_evidence;
    Callback<v2::P1LiveRoom(const v2::Manifest&, const nlohmann::json&)> replay_live_room;

    std::filesystem::path engine_binary;    // empty: <radiance_home>/../bin/radiance
    std::filesystem::path helper_binary;    // empty: <radiance_home>/helpers/r3_moe_worker
    ManifestRunner runner;                  // empty: run_to_completion with the owner's child environment
    vram::Hardware helper_vram;             // OS counter/readiness seam, never an admission override
    std::string power_profile = "stock";    // explicit v2 profile, observe by default
    std::filesystem::path live_policy_file; // only with explicit lab calibration
    bool calibration_run = false;
    std::filesystem::path experiment_file; // foreground lab-only v2 experiment; never replay/cache
};

/* Resolve a config then publish its measured manifest. On cache miss (exit 10) bring helpers up sequentially
 * against the unmeasured manifest, measure with --plan-memory and record helper RSS. On exit 11 adjust the
 * derived pool (bounded retries). Python and the launcher remain reference only until RSC11 S16. */
EngineStart replay_stored_manifest(const EnginePlanInputs& given, const std::filesystem::path& run_dir,
                                   const Hooks& hooks);

EngineStart plan_engine_only(const EnginePlanInputs& given, const std::filesystem::path& run_dir,
                             const Hooks& hooks);

} // namespace bc::serving::session
