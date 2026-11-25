#include "serving.h"

#include "serving_codec.h"
#include "live_policy_ingest.h"
#include "manifest_power.h"
#include "lab_experiment.h"
#include "manifest_writer.h"
#include "plan_cache.h"
#include "power_qualification.h"
#include "rank_demand.h"
#include "sha256.h"
#include "config_loader.h"
#include "release_runtime.h"
#include "runtime_guards.h"
#include "sdk_check.h"
#include "session.h"
#include "v2/manifest.h"
#include "v2/p1_producer.h"
#include "v2/s8_rules.h"
#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <ctime>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using bc::serving::Json;
using bc::serving::Options;
namespace fs = std::filesystem;

void usage() {
    std::cout
        << "bc-serve (RSC01 read-only)\n"
           "  --detect [--json] [--sys-root DIR]\n"
           "  --schema\n"
           "  --validate <site.toml> [profile.toml] [--set key=value]\n"
           "  --explain [--json] [--site FILE] [--profile FILE] [--set key=value]\n"
           "  explain CONFIG [--json] [--configs DIR] [--site FILE] [--set key=value]\n"
           "  --configs DIR       directories of NAME.yaml configs (repeatable; default BIGCHERRY_CONFIGS, else the\n"
           "                     installed bundle's configs/): a config is one native Radiance YAML with an x-bigcherry\n"
           "                     section; its name selects as layout.name\n"
           "  --model-root DIR    the directory a config's relative model path is under (none by default)\n"
           "  --calibration FILE  calibration cache over seeds, including RHP16 per-plan measurement records\n"
           "  calibration-prune --calibration FILE [--apply]   (drop rows the strict reader refuses)\n"
           "  calibration-ingest --from-log FILE --calibration FILE --manifest LAB_MANIFEST\n"
           "      validates one settled helper record against the strict lab manifest and atomically appends "
           "it\n"

           "  --write-manifest OUT --plan-cache DIR --engine-binary PATH [--plan-extra TEXT] "
           "[site/profile/set]\n"
           "      writes the serving manifest for this configuration to OUT (an existing OUT is refused) "
           "from its\n"
           "      measured plan and the planner's owner map, and reads it back through the shared strict "
           "reader before it\n"
           "      stays; a miss in the plan cache exits 10. Starts no process. The engine admits the plan the "
           "manifest\n"
           "      carries (expected plan and kv_backing_floor): there is no other way to hand it one.\n"
           "      --unmeasured   (with --write-manifest; no --plan-cache or --engine-binary) writes the "
           "unmeasured manifest\n"
           "                     of the measuring start instead: owner map and settings, plan.state "
           "unmeasured, no plan. Only\n"
           "                     a measuring start (radiance --plan-memory, r3_moe_worker --measuring) accepts "
           "it; every\n"
           "                     serving start refuses it.\n"
           "      --live-policy FILE   lab-only JSON override of every spec.handoff.placement.live_policy "
           "member; present means\n"
           "                     the helper runs the live placement policy. Nothing is defaulted.\n"
           "      --auto-calibrate-live-policy  with --calibration FILE, a measured production start with no record for its layout\n"
           "                           calibrates the live policy in session (default off)\n"
           "      --calibration-run    the lab calibration start: lab profile with the explicit --live-policy "
           "and no\n"
           "                     experiment, so the measured plan is the production plan.\n"
           "      --experiment FILE    a JSON object for spec.experiment (lab profile: the manifest is then "
           "written with\n"
           "                     profile lab; a production manifest never carries one).\n"
           "  --plan-extra TEXT   (with --write-manifest) engine arguments passed through "
           "verbatim; part of the key\n"
           "  --calibration-records FILE   (with --write-manifest) JSON array of P2 records; used where they "
           "match\n"
           "No process is started by these commands.\n"
           "\n"
           "Serving sessions (RSC02; engine-only configs for now):\n"
           "  serve CONFIG --run-dir DIR [--configs FILE|DIR] [--site FILE] [--profile FILE] [--model-root "
           "DIR] "
           "[--set k=v]\n"
           "         [--engine-binary PATH] [--allow-power-apply [--qualify-power]]\n"
           "         [--native-config YAML --model-root DIR]  select native YAML for a v3 named start\n"
           "         [--step-phase-trace]  opt-in per-rank HIP event trace in run-dir; changes timing\n"
           "         pre-measure strict-v2 power admission, optional hardware write only with explicit opt-in "
           "and verified\n"
           "         owner GPU lease; own one session in the foreground: measured manifest (a named start "
           "reuses the last measured one of its config from the --work-root store, else measures), engine from it, /health, start canary, then serve until `stop`, a "
           "signal or a\n"
           "         child's exit; every child stopped and verified gone before it returns. Select the config "
           "with\n"
           "         positional CONFIG selects the RSC12 named config. `start --manifest FILE --run-dir DIR` "
           "replays one named file; live checks and per-start receipts still run. Exit 0 stopped, "
           "96 canary, 97 plan, 98 busy\n"
           "         v3 reads NAME.yaml from the verified bundle, else from --configs DIR; --native-config YAML names another file.\n"
           "  --work-root DIR     the platform's work root: holds the named-config store, so a start reuses the\n"
           "                     last measured plan of its config; without it every start measures\n"
           "  explain CONFIG --native-config YAML --model-root DIR [--engine-binary PATH]\n"
           "         v3: also inspect and reconcile that YAML as a start does, offline (no engine or helper child, no GPU)\n"
           "  start  CONFIG (serve's options)   serve detached; returns once running (0) or failed (1)\n"
           "  start  --manifest FILE --run-dir DIR [--configs FILE|DIR]   replay exact strict-v2 bytes "
           "after current static/live/power checks; refuses changed build/model/devices, missing "
           "attestation or gates; never rewrites FILE\n"
           "  quality CONFIG --native-config YAML --model-root DIR --run-dir DIR\n"
           "                 --kld-ref DIR --kld-out FILE [--lab-experiment FILE]   run a finite native KL score;\n"
           "                 the run writes owner-map.json (the helpers' initial experts) to its run dir; with\n"
           "                 --lab-experiment, FILE is a forced placement schedule (a JSON object holding\n"
           "                 `placement`, see docs/reference/engines/radiance/SERVING_MANIFEST.md) applied by helper0 during\n"
           "                 the score on a lab-profile plan measured afresh; the score fails unless every step committed\n"
            "  quality CONFIG --native-config YAML --model-root DIR --run-dir DIR\n"
            "                 --kld-record DIR --kld-corpus FILE   record a KL reference (DIR existing and empty)\n"
           "  stop   --run-dir DIR       ask the owner to stop; 0 once every child has exited\n"
           "  status --run-dir DIR [--json]   the owner's record: state, config, port, manifest, processes\n";
}

// RSC02: serve | start | stop | status
int session_command(const std::string& cmd, int argc, char** argv) {
    namespace ss = bc::serving::session;
    Options opts;
    fs::path run_dir, engine_binary, stored_manifest, native_config, work_root, lab_experiment_file;
    std::string config_name;
    std::optional<ss::QualityScore> quality;
    if (cmd == "quality") quality.emplace();
    bool record_requested = false;
    fs::path calibration_records;
    bool allow_power_apply = false, qualify_power = false;
    bool strict_production_features = false;
    bool step_phase_trace = false;
    bool json = false;
    std::vector<std::string> passed; // the options as given, for `start` to hand to its detached `serve`
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&]() {
            if (i + 1 >= argc)
                throw std::runtime_error(arg + " requires a value");
            passed.push_back(arg);
            passed.push_back(argv[i + 1]);
            return std::string(argv[++i]);
        };
        const bool session_cmd = cmd == "serve" || cmd == "start" || quality.has_value();
        if (arg == "--help" || arg == "-h") {
            usage();
            return 0;
        } else if (arg == "--run-dir") {
            run_dir = need();
        } else if (arg == "--json" && cmd == "status") {
            json = true;
        } else if (!session_cmd) {
            std::string message = cmd;
            message += ": unknown option ";
            message += arg;
            message += " (--run-dir";
            message += cmd == "status" ? ", --json)" : ")";
            throw std::runtime_error(message);
        } else if (arg == "--sys-root") {
            opts.sys_root = need();
        } else if (arg == "--site") {
            opts.site = need();
        } else if (arg == "--profile") {
            opts.profile = need();
        } else if (arg == "--configs") {
            opts.configs.emplace_back(need());
        } else if (arg == "--model-root") {
            opts.model_root = need();
        } else if (arg == "--calibration") {
            opts.calibration_cache = need();
        } else if (arg == "--calibration-records") {
            calibration_records = need();
        } else if (arg == "--set") {
            opts.sets.push_back(need());
        } else if (arg == "--manifest") {
            stored_manifest = need();
        } else if (arg == "--native-config") {
            native_config = need();
        } else if (arg == "--kld-ref" && quality && quality->reference.empty()) {
            quality->reference = need();
        } else if (arg == "--kld-out" && quality && quality->output.empty()) {
            quality->output = need();
        } else if (arg == "--lab-experiment" && quality && lab_experiment_file.empty()) {
            lab_experiment_file = need();
        } else if (arg == "--kld-record" && quality && quality->reference.empty()) {
            quality->reference = need();
            record_requested = true;
        } else if (arg == "--kld-corpus" && quality && quality->corpus.empty()) {
            quality->corpus = need();
        } else if (!arg.empty() && arg.front() != '-' && config_name.empty()) {
            config_name = arg;
            passed.push_back(arg);
        } else if (arg == "--engine-binary") {
            engine_binary = need();
        } else if (arg == "--work-root") {
            work_root = need();   // the platform's work root: where the named-config store lives
        } else if (arg == "--allow-power-apply") {
            allow_power_apply = true;
            passed.push_back(arg);
        } else if (arg == "--qualify-power") {
            qualify_power = true;   // lab only: the apply that produces the receipt later applies are matched against
            passed.push_back(arg);
        } else if (arg == "--strict-production-features") {
            strict_production_features = true;
            passed.push_back(arg);
        } else if (arg == "--step-phase-trace") {
            if (step_phase_trace) throw std::runtime_error("--step-phase-trace given twice");
            step_phase_trace = true;
            passed.push_back(arg);
        } else {
            std::string message = cmd;
            message += ": unknown option ";
            message += arg;
            throw std::runtime_error(message);
        }
    }
    if (run_dir.empty())
        throw std::runtime_error(cmd +
                                 ": --run-dir DIR is required (the session's lock, socket, record, logs)");
    if (qualify_power && !allow_power_apply)
        throw std::runtime_error("--qualify-power is a lab apply: it needs --allow-power-apply (and a held gpu lease)");
    if (!config_name.empty() && !stored_manifest.empty())
        throw std::runtime_error("choose a named CONFIG or --manifest FILE, never both");
    // Read and checked whole before any plan, so a bad schedule never costs a measurement.
    std::optional<bc::serving::LabPlacementExperiment> lab_experiment;
    if (!lab_experiment_file.empty())
        lab_experiment = bc::serving::read_lab_placement_experiment(lab_experiment_file);
    if (!config_name.empty())
        opts.sets.push_back("layout.name=" + Json(config_name).dump());
    if (!stored_manifest.empty() && !fs::is_regular_file(stored_manifest))
        throw std::runtime_error("--manifest must name an existing regular file");
    run_dir = fs::absolute(run_dir);
    // The engine identity is stored under the work root: a start of an unchanged runtime does not hash it again.
    if (!work_root.empty()) bc::serving::set_engine_identity_store(fs::absolute(work_root));
    if (cmd == "stop") {
        const auto reply = ss::control(run_dir, "stop", std::chrono::minutes(10));
        if (!reply) {
            // No owner: nothing is stopped by a recorded PID (it can be reused). A process of the last session
            // that is still the same process (PID and start time) means its owner died without stopping it: not
            // stopped.
            const auto r = ss::read_record(run_dir);
            const auto alive = r ? ss::alive_recorded(*r) : std::vector<Json>{};
            if (!alive.empty()) {
                std::cerr << "stop failed: no session owner answers in " << run_dir.string()
                          << ", and its last session ("
                          << r.value_or(Json::object()).value("state", std::string("?")) << ") left running:\n";
                for (const auto& p : alive)
                    std::cerr << "  " << p.value("role", std::string("?")) << " pid " << p.at("pid") << ", log "
                              << p.value("log", std::string()) << '\n';
                return 1;
            }
            std::cout << "no session owner answers in " << run_dir.string()
                      << (r ? " (last session: " + r->value("state", std::string("?")) +
                                  "; none of its processes is running)"
                            : std::string())
                      << '\n';
            return 0;
        }
        if (reply->value("ok", false)) {
            std::cout << "stopped\n";
            return 0;
        }
        std::cerr << "stop failed (" << reply->value("state", std::string("?")) << "):\n";
        for (const auto& e : reply->value("errors", Json::array()))
            std::cerr << "  " << e.get<std::string>() << '\n';
        return 1;
    }
    if (cmd == "status") {
        const auto reply = ss::control(run_dir, "status", std::chrono::seconds(5));
        if (json) {
            std::cout << (reply ? *reply : Json({{"ok", false}, {"state", "no_owner"}})).dump(2) << '\n';
            return reply ? 0 : 3;
        }
        if (!reply) {
            const auto r = ss::read_record(run_dir);
            std::cout << "no session running in " << run_dir.string()
                      << (r ? " (last session: " + r->value("state", std::string("?")) + ")" : std::string())
                      << '\n';
            if (r)
                for (const auto& p : ss::alive_recorded(*r))
                    std::cout << "  WARNING: its " << p.value("role", std::string("?")) << " pid "
                              << p.at("pid") << " is still running without an owner\n";
            return 3;
        }
        std::cout << "state    " << reply->value("state", std::string()) << '\n'
                  << "owner    pid " << reply->value("owner_pid", 0) << ", since "
                  << reply->value("started_at", std::string()) << '\n';
        if (reply->contains("selection")) {
            const Json& s = reply->at("selection");
            std::cout << "config   " << s.value("config", std::string()) << " ("
                      << s.value("source", std::string()) << ")\n"
                      << "model    " << s.value("model", std::string()) << '\n'
                      << "cards    " << s.value("engine_cards", Json::array()).dump() << '\n';
        }
        if (reply->contains("port"))
            std::cout << "port     " << reply->at("port") << '\n';
        if (reply->contains("manifest"))
            std::cout << "manifest " << reply->at("manifest").get<std::string>() << '\n';
        if (reply->contains("calibration_record")) {
            const auto& selected = reply->at("calibration_record");
            std::cout << "calibration " << selected.at("status").get<std::string>();
            if (selected.value("status", std::string()) == "selected")
                std::cout << " " << selected.value("kind", std::string())
                          << " key=" << selected.value("key_sha256", std::string());
            else
                std::cout << " " << selected.value("reason_code", std::string())
                          << ": " << selected.value("reason_detail", std::string());
            std::cout << '\n';
        }
        for (const auto& p : reply->value("processes", Json::array()))
            if (!p.value("exited", false))
                std::cout << "process  " << p.value("role", std::string()) << " pid " << p.value("pid", 0)
                          << ", log " << p.value("log", std::string()) << '\n';
        return 0;
    }
    // A released executable must reverify its current tag + entire install
    // before any plan or child can run. A detached start's child verifies again.
    const fs::path self = fs::read_symlink("/proc/self/exe");
    if (const auto refusal = bc::serving::sdk_refusal(self, bc::serving::current_sdk_environment()))
        throw std::runtime_error(*refusal);
    const auto prefix = bc::serving::installed_prefix(self);
    if (prefix) {
        const auto sdk = bc::serving::recorded_sdk_root(self);
        if (!sdk)
            throw std::runtime_error("installed release refusal: SDK provenance is absent");
        bc::serving::verify_installed_prefix(self, *sdk);
        const fs::path home = *prefix / "radiance_home";
        const fs::path trusted_engine = *prefix / "bin/radiance";
        if (!engine_binary.empty()) {
            std::error_code error;
            const fs::path selected = fs::canonical(engine_binary, error);
            if (error || selected != fs::canonical(trusted_engine))
                throw std::runtime_error(
                    "installed release refusal: engine binary differs from verified prefix");
        }
        engine_binary = trusted_engine;
        if (std::any_of(opts.sets.begin(), opts.sets.end(), [](const std::string& setting) {
                return setting.starts_with("engine.radiance_home=");
            }))
            throw std::runtime_error("installed release refusal: runtime home is bound to verified prefix");
        opts.sets.push_back("engine.radiance_home=" + Json(home.string()).dump());
    }
    if (native_config.empty() && stored_manifest.empty() &&
        !config_name.empty() && (cmd == "serve" || cmd == "start")) {
        // A config is its YAML: NAME.yaml of the verified bundle, else of --configs / BIGCHERRY_CONFIGS.
        std::vector<fs::path> sources = opts.configs;
        if (prefix) sources.insert(sources.begin(), *prefix / "configs");
        if (const fs::path found = bc::serving::config_yaml_for(config_name, sources); !found.empty())
            native_config = found;
    }
    if (!native_config.empty()) {
        if (config_name.empty() || !stored_manifest.empty() || opts.model_root.empty())
            throw std::runtime_error("--native-config requires a named CONFIG and --model-root DIR");
        if (!fs::is_regular_file(native_config) || !fs::is_directory(opts.model_root))
            throw std::runtime_error("--native-config and --model-root must name an existing file and directory");
    } else if ((cmd == "serve" || cmd == "start") && stored_manifest.empty()) {
        throw std::runtime_error("serving needs " + config_name + ".yaml in the bundle or --configs DIR (or --native-config YAML) for a fresh start");
    }
    if (quality && step_phase_trace)
        throw std::runtime_error("--step-phase-trace is a serving-only diagnostic");
    if (quality) {
        const bool scoring = !quality->output.empty();
        if (native_config.empty() || quality->reference.empty() || scoring == record_requested ||
            (record_requested && quality->corpus.empty()) || (!record_requested && !quality->corpus.empty()))
            throw std::runtime_error("quality requires named native YAML and either --kld-ref DIR --kld-out FILE "
                                     "(score) or --kld-record DIR --kld-corpus FILE (record)");
        quality->reference = fs::canonical(quality->reference);
        if (scoring) quality->output = fs::absolute(quality->output);
        if (record_requested) quality->corpus = fs::canonical(quality->corpus);
    }
    const bool installed_native_replay = prefix && !stored_manifest.empty();
    if ((!config_name.empty() || installed_native_replay) && opts.configs.empty()) {
        const char* env = std::getenv("BIGCHERRY_CONFIGS");
        if (!env || !*env) {
            if (!prefix)
                throw std::runtime_error(
                    "serve: no --configs or BIGCHERRY_CONFIGS and executable is not a verified release prefix");
            opts.configs.push_back(*prefix / "configs");
        }
    }
    if (cmd == "start") {
        std::vector<std::string> child = {self.string(), "serve"};
        child.insert(child.end(), passed.begin(), passed.end());
        // --run-dir as an absolute path: the detached owner does not share this shell's directory assumptions
        for (size_t i = 0; i + 1 < child.size(); ++i)
            if (child[i] == "--run-dir")
                child[i + 1] = run_dir.string();
        return ss::start_detached(child, run_dir, std::chrono::minutes(45));
    }
    ss::EnginePlanInputs inputs;
    inputs.options = opts;
    inputs.stored_manifest = std::move(stored_manifest);
    inputs.named_config = config_name;
    inputs.work_root = work_root;
    inputs.quality_score = quality.has_value();
    inputs.lab_experiment = std::move(lab_experiment);
    inputs.calibration_records_file = std::move(calibration_records);

    inputs.self = self;
    inputs.engine_binary = engine_binary;
    if (!native_config.empty())
        inputs.native_admission = ss::NativeAdmissionRequest{native_config, opts.model_root, {}};
    // Hardware and durable records live through the entire foreground owner.
    // A real write needs BOTH an explicit opt-in and a live ancestor-bound lease.
    bc::serving::power::LinuxHardware hardware(opts.sys_root);
    // Receipts outlive the run: a later start finds the earlier apply receipts that qualify a setting. Without a work
    // root (tests, one-off runs) they stay in the run directory and qualify nothing across runs.
    const fs::path power_dir = work_root.empty() ? run_dir : fs::absolute(work_root) / "power";
    const fs::path receipts_dir = work_root.empty() ? run_dir / "power-receipts" : power_dir / "receipts";
    bc::serving::power::DiskRecords records(work_root.empty() ? run_dir / "power-journals" : power_dir / "journals",
                                            receipts_dir);
    inputs.power.apply = allow_power_apply;
    inputs.power.evidence.qualify = qualify_power;
    inputs.power.evidence.probe = [&hardware](const std::string& bdf) {
        const auto pci = bc::serving::power::PciAddress::parse(bdf);
        if (!pci) return bc::serving::v2::S10DeviceProbe{};
        const auto card = hardware.inspect(pci.value());
        return card ? bc::serving::power::probe_of(card.value()) : bc::serving::v2::S10DeviceProbe{};
    };
    inputs.power.evidence.qualified_by = [&hardware, receipts_dir](const std::string& bdf,
                                                                   const bc::serving::v2::PowerSettings& settings) {
        const auto pci = bc::serving::power::PciAddress::parse(bdf);
        if (!pci) return std::optional<std::string>{};
        const auto card = hardware.inspect(pci.value());
        if (!card) return std::optional<std::string>{};
        return bc::serving::power::qualifying_receipt(receipts_dir, pci.value(), settings, card.value().identity);
    };
    std::string power_site_yaml;   // this host's per-BDF power settings; the engine's YAML parser reads them
    if (!work_root.empty()) {
        std::ifstream site(fs::absolute(work_root) / "power-site.yaml");
        if (site) power_site_yaml.assign(std::istreambuf_iterator<char>(site), std::istreambuf_iterator<char>());
    }
    bc::serving::power::LinuxSensor sensor(opts.sys_root / "sys");
    auto power = std::make_shared<ss::ManifestPower>(
        hardware, records, sensor, [] { return ss::owner_has_gpu_lease(); }, allow_power_apply, config_name);
    ss::Session session;
    session.power_hook.begin_manifest = [power](const fs::path& path, bc::serving::v2::PlanUse use) {
        return power->begin(path, use);
    };
    session.power_hook.verify_serving = [power](const fs::path& path) { return power->verify_serving(path); };
    session.power_hook.gfxoff_window = [power](bool idle) {
        return power->capture(idle ? bc::serving::power::WindowPhase::engine_loaded_idle
                                   : bc::serving::power::WindowPhase::no_process);
    };
    session.power_hook.verified_exit = [power] { power->children_verified(); };
    session.power_hook.restore = [power] { return power->finish(); };
    auto guards = std::make_shared<ss::RuntimeGuards>();
    const ss::Canary canary = session.canary;
    const auto timeout = session.timing.canary_timeout;
    session.runtime_guard = [guards, canary, timeout](const ss::EngineStart& plan,
                                                      const fs::path& engine_log) -> bc::serving::v2::Status {
        ss::RuntimeGuards::Retry retry = [port = plan.port, canary, timeout]() -> bc::serving::v2::Status {
            if (const auto why = ss::canary(port, canary.prompt, canary.expect, timeout))
                return bc::serving::v2::make_error(bc::serving::v2::ConfigErrorCode::admission_refused,
                                                   "session.runtime.mtp.retry", *why);
            return std::nullopt;
        };
        return guards->startup(plan, engine_log, retry);
    };
    session.runtime_watch = [guards] { return guards->poll(); };
    session.run_dir = run_dir;
    session.quality = std::move(quality);
    session.publish_feature_states = true;
    session.strict_production_features = strict_production_features;
    session.step_phase_trace = step_phase_trace;
    session.plan = [inputs, power_site_yaml](const fs::path& rd, const ss::Hooks& given) {
        ss::Hooks hooks = given;
        hooks.power_site_yaml = power_site_yaml;
        return inputs.stored_manifest.empty() ? ss::plan_engine_only(inputs, rd, hooks)
                                              : ss::replay_stored_manifest(inputs, rd, hooks);
    };
    return ss::serve(session);
}

void explain(const bc::serving::Resolved& result) {
    std::cout << "layout: " << result.typed.layout.name << '\n';
    std::cout << "engine cards: " << result.typed.layout.engine_cards.size() << '\n';
    std::cout << "helpers: " << result.typed.layout.helper_cards.size() << '\n';
    std::cout << "manifest sha256: " << result.hash << '\n';
    for (auto it = result.provenance.begin(); it != result.provenance.end(); ++it) {
        auto key = it.key();
        const auto& meta = it.value();
        std::string group = key;
        auto dot = key.find('.');
        Json val = dot == std::string::npos ? result.values.at(key)
                                            : result.values.at(key.substr(0, dot)).at(key.substr(dot + 1));
        std::cout << key << " = " << val.dump() << "  [" << meta.at("origin").get<std::string>() << ", "
                  << meta.at("lifecycle").get<std::string>() << "] " << meta.at("source").get<std::string>()
                  << '\n';
    }
}

// RSC12 step 3: inspect the *same* detected facts, native config loader and
// resolver used by the session owner, without planning, starting children or
// loading Python layouts. A named config is a positional identity, not an
// alternative spelling of layout.name in a second CLI settings language.
int explain_command(int argc, char** argv) {
    Options opts;
    std::string name;
    bool json = false;
    fs::path native_config, engine_binary, work_root;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&]() {
            if (i + 1 >= argc)
                throw std::runtime_error("explain: " + arg + " requires a value");
            return std::string(argv[++i]);
        };
        if (arg == "--help" || arg == "-h") {
            usage();
            return 0;
        } else if (arg == "--json") {
            json = true;
        } else if (arg == "--configs") {
            opts.configs.emplace_back(need());
        } else if (arg == "--site") {
            opts.site = need();
        } else if (arg == "--profile") {
            opts.profile = need();
        } else if (arg == "--model-root") {
            opts.model_root = need();
        } else if (arg == "--native-config") {
            native_config = need();
        } else if (arg == "--engine-binary") {
            engine_binary = need();
        } else if (arg == "--work-root") {
            work_root = need();
        } else if (arg == "--calibration") {
            opts.calibration_cache = need();
        } else if (arg == "--sys-root") {
            opts.sys_root = need();
        } else if (arg == "--set") {
            const std::string setting = need();
            if (setting.starts_with("layout.name="))
                throw std::runtime_error("explain: named CONFIG cannot be replaced by --set layout.name");
            opts.sets.push_back(setting);
        } else if (!arg.empty() && arg.front() != '-') {
            if (!name.empty())
                throw std::runtime_error("explain: exactly one named CONFIG is required");
            name = arg;
        } else {
            throw std::runtime_error("explain: unknown option " + arg);
        }
    }
    if (name.empty())
        throw std::runtime_error("explain: a named CONFIG is required");
    opts.sets.push_back("layout.name=" + Json(name).dump());
    namespace ss = bc::serving::session;
    const fs::path self = fs::read_symlink("/proc/self/exe");
    if (!work_root.empty()) bc::serving::set_engine_identity_store(fs::absolute(work_root));
    if (const auto prefix = bc::serving::installed_prefix(self)) {
        // An installed bundle binds its own runtime home and engine, as a start does.
        if (!engine_binary.empty() && fs::canonical(engine_binary) != fs::canonical(*prefix / "bin/radiance"))
            throw std::runtime_error("installed release refusal: engine binary differs from verified prefix");
        engine_binary = *prefix / "bin/radiance";
        if (std::any_of(opts.sets.begin(), opts.sets.end(),
                        [](const std::string& s) { return s.starts_with("engine.radiance_home="); }))
            throw std::runtime_error("installed release refusal: runtime home is bound to verified prefix");
        opts.sets.push_back("engine.radiance_home=" + Json((*prefix / "radiance_home").string()).dump());
    }
    if (!native_config.empty() &&
        (opts.model_root.empty() || !fs::is_regular_file(native_config) || !fs::is_directory(opts.model_root)))
        throw std::runtime_error("explain: --native-config and --model-root must name an existing file and directory");
    // The config is its native YAML: read it, device-free, through the engine's parser before resolving.
    try {
        ss::attach_inspected_config(opts, name, native_config, self, engine_binary, ss::Hooks{});
    } catch (const ss::PlanFailed& error) {
        throw std::runtime_error(std::string("explain: ") + error.what());
    }
const std::string cache_note = work_root.empty()
        ? "named-config store: off (no --work-root); every start measures"
        : "named-config store: " + work_root.string();
    const auto resolved = bc::serving::resolve(opts, bc::serving::detect(opts.sys_root));
    if (!native_config.empty()) {
        ss::EnginePlanInputs preflight;
        preflight.options = opts;
        preflight.named_config = name;
        preflight.self = self;
        preflight.engine_binary = engine_binary;
        preflight.native_admission = ss::NativeAdmissionRequest{native_config, opts.model_root, {}};
        preflight.native_preflight = true;
        // The planner's scratch directory: nothing persistent is written, no child outlives the inspector.
        char scratch[] = "/tmp/bc-explain-XXXXXX";
        if (!::mkdtemp(scratch))
            throw std::runtime_error("explain: cannot create a scratch directory");
        struct Cleanup {
            fs::path path;
            ~Cleanup() { std::error_code ignored; fs::remove_all(path, ignored); }
        } cleanup{scratch};
        try {
            const ss::EngineStart plan = ss::plan_engine_only(preflight, scratch, ss::Hooks{});
            const Json& report = plan.native_preflight_receipt;
            if (json) {
                std::cout << Json{{"result", "pass"}, {"config", name}, {"values", plan.native_preflight_values},
                                  {"model_identity_cache", cache_note}}.dump(2) << '\n';
                return 0;
            }
            std::cout << "native admission preflight: PASS\n  " << cache_note << "\n  config " << name << "\n  yaml " << native_config.string()
                      << "\n  snapshot " << report.at("snapshot").at("sha256").get<std::string>() << "\n  model "
                      << report.at("engine").at("model").get<std::string>() << "\n  tp "
                      << report.at("engine").at("tp").get<int>() << ", cards " << report.at("engine").at("cards").dump()
                      << ", port " << plan.port << '\n';
            return 0;
        } catch (const ss::PlanFailed& error) {
            std::cout << "native admission preflight: REFUSED\n  config " << name << "\n  " << error.what() << '\n';
            return 1;
        }
    }
    if (json) {
        std::cout << Json{{"config", name},
                          {"values", resolved.values},
                          {"provenance", resolved.provenance},
                          {"hash", resolved.hash},
                          {"manifest", resolved.manifest},
                          {"model_identity_cache", cache_note}}
                         .dump(2)
                  << '\n';
    } else {
        explain(resolved);
        std::cout << cache_note << '\n';
    }
    return 0;
}

} // namespace

/* A JSON object an operator supplies (a live policy, a lab experiment): read once, never defaulted; empty path
 * = none. `from` names the file for the member's provenance record. */
static void load_json_input(const fs::path& path, const char* flag, Json* out, std::string* from) {
    if (path.empty())
        return;
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error(std::string(flag) + ": cannot read " + path.string());
    std::ostringstream text;
    text << in.rdbuf();
    *out = Json::parse(text.str(), nullptr, false);
    if (!out->is_object())
        throw std::runtime_error(std::string(flag) + ": " + path.string() + " is not a JSON object");
    *from = std::string(flag) + " " + path.filename().string();
}

int main(int argc, char** argv) {
    try {
        if (argc > 1) {
            const std::string cmd(argv[1]);
            if (cmd == "explain")
                return explain_command(argc, argv);
            if (cmd == "helper-exec")
                return bc::serving::session::helper_exec_command(argc, argv);
            if (cmd == "serve" || cmd == "start" || cmd == "quality" || cmd == "stop" || cmd == "status")
                return session_command(cmd, argc, argv);
        }
        if (argc > 1 && std::string(argv[1]) == "calibration-prune") {
            // Remove the rows the strict reader refuses (hash-only, no calibration_inputs). Dry run unless --apply.
            fs::path cache;
            bool apply = false;
            for (int i = 2; i < argc; ++i) {
                const std::string arg(argv[i]);
                if (arg == "--apply") apply = true;
                else if (arg == "--calibration" && i + 1 < argc) cache = argv[++i];
                else throw std::runtime_error("calibration-prune: unknown or incomplete option " + arg);
            }
            if (cache.empty()) throw std::runtime_error("calibration-prune: --calibration FILE is required");
            const auto removed = bc::serving::prune_live_policy_store(cache, apply);
            for (const auto& key : removed) std::cout << (apply ? "removed " : "would remove ") << key << "\n";
            std::cout << "calibration-prune: " << removed.size() << (apply ? " removed" : " to remove (dry run; --apply removes)")
                      << " from " << cache.string() << "\n";
            return 0;
        }
        if (argc > 1 && std::string(argv[1]) == "calibration-ingest") {
            // The helper has only the *typed v1 manifest* cache key. It is
            // different from the RSC03 plan-store key used by the derivation
            // reader. Re-resolve the exact launch and validate its cached
            // measured plan before mapping that trusted log identity.
            Options ingest_opts;
            fs::path log, cache, manifest, plan_dir, binary;
            std::string extra;
            for (int i = 2; i < argc; ++i) {
                const std::string arg(argv[i]);
                if (i + 1 >= argc)
                    throw std::runtime_error(arg + " requires a value");
                if (arg == "--from-log")
                    log = argv[++i];
                else if (arg == "--calibration")
                    cache = argv[++i];
                else if (arg == "--manifest")
                    manifest = argv[++i];
                else if (arg == "--plan-cache")
                    plan_dir = argv[++i];
                else if (arg == "--engine-binary")
                    binary = argv[++i];
                else if (arg == "--plan-extra")
                    extra = argv[++i];
                else if (arg == "--site")
                    ingest_opts.site = argv[++i];
                else if (arg == "--profile")
                    ingest_opts.profile = argv[++i];
                else if (arg == "--configs")
                    ingest_opts.configs.emplace_back(argv[++i]);
                else if (arg == "--model-root")
                    ingest_opts.model_root = argv[++i];
                else if (arg == "--sys-root")
                    ingest_opts.sys_root = argv[++i];
                else if (arg == "--set")
                    ingest_opts.sets.push_back(argv[++i]);
                else
                    throw std::runtime_error("calibration-ingest: unknown option " + arg);
            }
            if (log.empty() || cache.empty() || manifest.empty() || plan_dir.empty() || binary.empty() ||
                ingest_opts.site.empty() || extra.empty())
                throw std::runtime_error("calibration-ingest: --from-log --calibration --manifest --plan-cache "
                                         "--engine-binary --site --plan-extra are required");
            // Do not invent a plan key from a filename or trust a supplied
            // raw key. Verify the exact resolved model, engine, and measured
            // plan against the strict lab manifest.
            if (fs::exists(cache))
                ingest_opts.calibration_cache = cache;
            const auto facts = bc::serving::detect(ingest_opts.sys_root);
            const auto resolved = bc::serving::resolve(ingest_opts, facts);
            const Json& values = resolved.values;
            const fs::path home = values.at("engine").at("radiance_home").get<std::string>();
            const Json inputs = bc::serving::plan_key_inputs(
                values, facts,
                bc::serving::container_identity(values.at("engine").at("model").get<std::string>()),
                bc::serving::engine_identity(binary, home), extra);
            const auto plan = bc::serving::plan_cache_lookup(plan_dir, inputs);
            if (!plan)
                throw std::runtime_error("calibration-ingest: exact measured plan is absent from cache");
            const auto verified = bc::serving::v2::load_manifest(manifest, bc::serving::v2::PlanUse::serving);
            if (!verified)
                throw std::runtime_error("calibration-ingest: strict lab manifest rejected: " +
                                         bc::serving::v2::render_human("bc-serve", verified.error()));
            const auto& m = verified.value();
            if (!m.spec.plan.engine_memory ||
                m.spec.intent.layout != values.at("layout").at("name").get<std::string>() ||
                m.spec.intent.model != values.at("engine").at("model").get<std::string>() ||
                m.spec.plan.engine_memory->model_identity != inputs.at("model").get<std::string>() ||
                m.spec.plan.engine_memory->engine_identity != inputs.at("engine_build").get<std::string>())
                throw std::runtime_error(
                    "calibration-ingest: launch/model/engine identity does not match lab manifest");
            std::ifstream in(manifest, std::ios::binary);
            if (!in)
                throw std::runtime_error("calibration-ingest: cannot reread manifest");
            const Json raw = Json::parse(in);
            if (raw.at("spec").at("plan").at("engine_memory").at("expected") != plan->plan_json)
                throw std::runtime_error("calibration-ingest: cached measured plan differs from lab manifest");
            bc::serving::ingest_live_policy_log(
                log, cache, manifest,
                bc::serving::policy_calibration_key(
                    bc::serving::policy_calibration_inputs(raw.at("spec"))));
            std::cout << "calibration: " << cache << '\n';
            return 0;
        }
        Options opts;
        bool detection = false, schema = false, validation = false;
        bool explanation = false, json = false;
        std::string plan_extra;
        std::string expected_plan_key;
        int native_config_fd = -1;
        fs::path plan_cache, engine_binary;
        fs::path write_manifest, live_policy_file, experiment_file, calibration_records_file,
            s8_replay_evidence_file;
        bool auto_calibrate_live_policy = false;   // default off: no in-session calibration unless asked for
        std::string expected_owner_attempt;
        // --calibration-run: the launcher's lab calibration start (--calibrate). A lab-profile manifest with an
        // explicit live policy and NO experiment, so the plan it measures is the production plan.
        bool calibration_run = false;
        bool unmeasured = false;
        std::vector<fs::path> positional;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto need_arg = [&](const char* what) {
                if (i + 1 >= argc)
                    throw std::runtime_error(std::string(what) + " requires a value");
                return std::string(argv[++i]);
            };
            if (arg == "--help" || arg == "-h") {
                usage();
                return 0;
            }
            if (arg == "--detect")
                detection = true;
            else if (arg == "--schema")
                schema = true;
            else if (arg == "--validate")
                validation = true;
            else if (arg == "--explain")
                explanation = true;
            else if (arg == "--json")
                json = true;
            else if (arg == "--sys-root")
                opts.sys_root = need_arg("--sys-root");
            else if (arg == "--site")
                opts.site = need_arg("--site");
            else if (arg == "--profile")
                opts.profile = need_arg("--profile");
            else if (arg == "--configs")
                opts.configs.emplace_back(need_arg("--configs"));
            else if (arg == "--model-root")
                opts.model_root = need_arg("--model-root");
            else if (arg == "--run-dir")
                opts.run_dir = need_arg("--run-dir");
            else if (arg == "--calibration")
                opts.calibration_cache = need_arg("--calibration");
            else if (arg == "--write-manifest")
                write_manifest = need_arg("--write-manifest");
            else if (arg == "--unmeasured")
                unmeasured = true;
            else if (arg == "--auto-calibrate-live-policy")
                auto_calibrate_live_policy = true;
            else if (arg == "--live-policy")
                live_policy_file = need_arg("--live-policy");
            else if (arg == "--calibration-run")
                calibration_run = true;
            else if (arg == "--experiment")
                experiment_file = need_arg("--experiment");
            else if (arg == "--plan-extra")
                plan_extra = need_arg("--plan-extra");
            else if (arg == "--expected-plan-key")
                expected_plan_key = need_arg("--expected-plan-key");
            else if (arg == "--bc-native-config-fd") {
                const std::string value = need_arg("--bc-native-config-fd");
                const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), native_config_fd);
                if (error != std::errc{} || end != value.data() + value.size() || native_config_fd < 3)
                    throw std::runtime_error("--bc-native-config-fd requires a descriptor greater than 2");
            }
            else if (arg == "--expected-owner-attempt")
                expected_owner_attempt = need_arg("--expected-owner-attempt");
            else if (arg == "--calibration-records")
                calibration_records_file = need_arg("--calibration-records");
            else if (arg == "--s8-replay-evidence")
                s8_replay_evidence_file = need_arg("--s8-replay-evidence");
            else if (arg == "--plan-cache")
                plan_cache = need_arg("--plan-cache");
            else if (arg == "--engine-binary")
                engine_binary = need_arg("--engine-binary");
            else if (arg == "--set")
                opts.sets.push_back(need_arg("--set"));
            else if (!arg.empty() && arg.front() == '-')
                throw std::runtime_error("unknown option " + arg);
            else
                positional.push_back(arg);
        }
        // BRVP06: a build runs against the ROCm SDK it recorded; an environment that points at another one is
        // refused here, before any mode runs (--help above is the only thing that answers in such an
        // environment)
        if (const auto refusal =
                bc::serving::sdk_refusal("/proc/self/exe", bc::serving::current_sdk_environment()))
            throw std::runtime_error(*refusal);
        const int modes = int(detection) + int(schema) + int(validation) + int(explanation) +
                          int(!write_manifest.empty());
        const bool plan_mode = !write_manifest.empty();
        if (native_config_fd >= 0 && !plan_mode)
            throw std::runtime_error("--bc-native-config-fd is an internal --write-manifest input");
        if (plan_mode) {
            // The v3 measuring/writing path is owner-bound to the sealed native
            // YAML snapshot. Never fall back to the v2 key or import caller-
            // supplied plan files: that would create an unowned cache entry
            // before the strict v3 writer can reject the missing snapshot.
            if (native_config_fd < 3)
                throw std::runtime_error("v3 --write-manifest requires --bc-native-config-fd");
        }
        if (!plan_mode && !opts.run_dir.empty())
            throw std::runtime_error("--run-dir in resolver mode is internal to --write-manifest");
        if (!plan_mode && (!plan_cache.empty() || !engine_binary.empty()))
            throw std::runtime_error("--plan-cache and --engine-binary go with --write-manifest");
        if (!plan_mode && !plan_extra.empty())
            throw std::runtime_error("--plan-extra goes with --write-manifest");
        if ((!s8_replay_evidence_file.empty() && (write_manifest.empty() || unmeasured)))
            throw std::runtime_error("--s8-replay-evidence requires a measured --write-manifest");
        if (write_manifest.empty() && !calibration_records_file.empty())
            throw std::runtime_error("--calibration-records goes with --write-manifest");
        if (plan_mode && !unmeasured &&
            expected_owner_attempt.empty())
            throw std::runtime_error("v3 measured cache reads require --expected-owner-attempt");
        if (plan_mode && !unmeasured && expected_plan_key.empty())
            throw std::runtime_error("v3 measured cache reads require --expected-plan-key");
        if (!expected_plan_key.empty() && !unmeasured &&
            !(plan_mode && native_config_fd >= 3))
            throw std::runtime_error(
                "--expected-plan-key is for a session-owned unmeasured call or v3 measured cache read");
        if (!live_policy_file.empty() && experiment_file.empty() && !calibration_run)
            throw std::runtime_error("--live-policy is a lab override; supply --experiment for an explicit lab "
                                     "profile, or --calibration-run");
        if (calibration_run && (live_policy_file.empty() || !experiment_file.empty()))
            throw std::runtime_error("--calibration-run takes an explicit --live-policy and no --experiment "
                                     "(the production plan is measured)");
        if (write_manifest.empty() && (unmeasured || !live_policy_file.empty() || !experiment_file.empty()))
            throw std::runtime_error("--unmeasured, --live-policy and --experiment go with --write-manifest");
        if (modes != 1)
            throw std::runtime_error("choose one mode (see --help)");
        if (schema) {
            if (!positional.empty())
                throw std::runtime_error("--schema takes no files");
            std::cout << bc::serving::json_schema().dump(2) << '\n';
            return 0;
        }
        if (detection) {
            if (!positional.empty() || !opts.site.empty() || !opts.profile.empty() || !opts.sets.empty() ||
                !opts.calibration_cache.empty())
                throw std::runtime_error("--detect takes only --sys-root and --json");
            auto facts = bc::serving::detect(opts.sys_root);
            if (json) {
                std::cout << facts.dump(2) << '\n';
            } else {
                auto host = facts.at("host");
                std::cout << "host: " << host.at("mem_total_mib") << " MiB RAM, "
                          << host.at("mem_available_mib") << " MiB available, " << host.at("swap_total_mib")
                          << " MiB swap, " << host.at("shm_available_mib") << " MiB /dev/shm available, "
                          << host.at("cpu_count") << " CPUs\n";
                for (const auto& g : facts.at("gpus"))
                    std::cout << "HIP " << g.at("hip_ordinal") << "  " << g.at("arch").get<std::string>() << " "
                              << g.at("bdf").get<std::string>() << "  " << g.at("vram_total_mib")
                              << " MiB VRAM  " << g.at("current_link_speed").get<std::string>() << " x"
                              << g.at("current_link_width") << '\n';
            }
            return 0;
        }
        if (validation) {
            if (positional.empty() || positional.size() > 2 || !opts.site.empty() || !opts.profile.empty())
                throw std::runtime_error("--validate expects <site.toml> [profile.toml]");
            opts.site = positional[0];
            if (positional.size() == 2)
                opts.profile = positional[1];
        } else if (!positional.empty()) {
            throw std::runtime_error("unexpected file; use --site or --profile");
        }
        if (plan_mode) {
            if (!positional.empty())
                throw std::runtime_error(
                    "--write-manifest takes --site, --profile, --set and --calibration only");
            if (unmeasured && !plan_cache.empty())
                throw std::runtime_error("--unmeasured has no plan: it takes no --plan-cache");
            if (unmeasured && expected_plan_key.empty() && (!engine_binary.empty() || !plan_extra.empty()))
                throw std::runtime_error(
                    "--unmeasured takes --engine-binary and --plan-extra only with --expected-plan-key");
            if (unmeasured && !expected_plan_key.empty() && (engine_binary.empty() || plan_extra.empty()))
                throw std::runtime_error(
                    "session-owned --unmeasured requires --expected-plan-key, --engine-binary and --plan-extra");
            if (!unmeasured && (plan_cache.empty() || engine_binary.empty()))
                throw std::runtime_error("--write-manifest needs --plan-cache DIR and --engine-binary PATH");
            // The same two steps the session owner runs in-process (write_native_manifest); this command is their CLI.
            bc::serving::NativeWriterRequest request;
            request.calibration_records = Json::array();
            if (!calibration_records_file.empty()) {
                std::ifstream rin(calibration_records_file, std::ios::binary);
                if (!rin)
                    throw std::runtime_error("--calibration-records: cannot read " +
                                             calibration_records_file.string());
                std::ostringstream text;
                text << rin.rdbuf();
                request.calibration_records = Json::parse(text.str(), nullptr, false);
                if (!request.calibration_records.is_array())
                    throw std::runtime_error("--calibration-records: " + calibration_records_file.string() +
                                             " is not a JSON array");
            }
            auto native = bc::serving::session::read_native_writer_input(native_config_fd);
            request.options = opts;
            request.inspected = std::move(native.inspected);
            request.snapshot = std::move(native.snapshot);
            request.settings = std::move(native.settings);
            request.unmeasured = unmeasured;
            request.engine_binary = engine_binary;
            request.plan_extra = plan_extra;
            request.expected_plan_key = expected_plan_key;
            request.expected_owner_attempt = expected_owner_attempt;
            request.plan_cache = plan_cache;
            request.power_profile = request.inspected.x.power_profile();
            request.power_rules = bc::serving::session::selected_power_rules(native.power_profiles, request.power_profile);
            request.manifest_name = write_manifest.filename().string();
            request.producer_sha256 = bc::serving::file_sha256("/proc/self/exe");
            request.lab_profile = !(experiment_file.empty() && !calibration_run);
            request.auto_calibrate = auto_calibrate_live_policy;
            request.want_s8_replay_evidence = !s8_replay_evidence_file.empty();
            load_json_input(live_policy_file, "--live-policy", &request.live_policy, &request.live_policy_from);
            load_json_input(experiment_file, "--experiment", &request.experiment, &request.experiment_from);
            const auto outcome = bc::serving::write_native_manifest(request);
            using Kind = bc::serving::NativeWriterOutcome::Kind;
            if (outcome.kind == Kind::cache_miss) {
                std::cerr << "bc-serve: no measured plan for this configuration (key " << outcome.plan_key.substr(0, 16)
                          << "); a start measures it with the helpers up\n";
                return 10;
            }
            if (outcome.kind == Kind::host_pool_change) {
                std::cerr << outcome.notes;
                std::cout << "derived_host_pool_mib=" << outcome.derived_pool_mib << '\n';
                return 11;
            }
            std::cerr << outcome.notes;
            bc::serving::write_manifest_file(write_manifest, outcome.built.text);
            if (!outcome.s8_replay_source.is_null())
                bc::serving::write_manifest_file(s8_replay_evidence_file, outcome.s8_replay_source.dump(2) + "\n");
            std::cout << "manifest: " << write_manifest << (unmeasured ? " (unmeasured: for the measuring start only)" : "")
                      << '\n'
                      << "digest: " << outcome.built.digest << '\n'
                      << "no processes started\n";
            return 0;
        }
        auto result = bc::serving::resolve(opts, bc::serving::detect(opts.sys_root));
        if (validation) {
            std::cout << "valid: " << result.hash << '\n';
        } else {
            if (json)
                std::cout << result.manifest.dump(2) << '\n';
            else
                explain(result);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bc-serve: " << error.what() << '\n';
        return 2;
    }
}
