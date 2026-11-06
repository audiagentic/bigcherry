#include "serving.h"
#include "manifest_writer.h"
#include "live_policy_ingest.h"
#include "v1/manifest.h"
#include "plan_cache.h"
#include "sdk_check.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
    std::cout << "bc-serve (RSC01 read-only)\n"
                 "  --detect [--json] [--sys-root DIR]\n"
                 "  --schema\n"
                 "  --validate <site.toml> [profile.toml] [--set key=value]\n"
                 "  --explain [--json] [--site FILE] [--profile FILE] [--set key=value]\n"
                 "  --calibration FILE  calibration cache over seeds, including RHP16 per-plan measurement records\n"
                 "  calibration-ingest --from-log FILE --calibration FILE --manifest LAB_MANIFEST\n"
                 "      validates one settled helper record against the strict lab manifest and atomically appends it\n"

                 "  --write-manifest OUT --plan-cache DIR --engine-binary PATH [--plan-extra TEXT] [site/profile/set]\n"
                 "      writes the v1 serving manifest for this configuration to OUT (an existing OUT is refused) from its\n"
                 "      measured plan and the planner's owner map, and reads it back through the shared strict reader before it\n"
                 "      stays; a miss in the plan cache exits 10. Starts no process. The engine admits the plan the manifest\n"
                 "      carries (expected plan and kv_backing_floor): there is no other way to hand it one.\n"
                 "      --measured-plan FILE --helpers-resident-mib N --engine-resident-mib M   the plan a measuring start produced\n"
                 "                     (radiance --plan-memory FILE, helpers up) and the host memory the helpers and the engine held\n"
                 "                     there (resident, MiB): the plan is checked against this configuration (host pool, host\n"
                 "                     consumers) and stored in the plan cache with the two figures under this configuration's key,\n"
                 "                     and the manifest is written from it. The host ledger counts both figures against MemAvailable\n"
                 "                     as read now.\n"
                 "      --unmeasured   (with --write-manifest; no --plan-cache or --engine-binary) writes the unmeasured manifest\n"
                 "                     of the measuring start instead: owner map and settings, plan.state unmeasured, no plan. Only\n"
                 "                     a measuring start (radiance --plan-memory, r3_moe_worker --measuring) accepts it; every\n"
                 "                     serving start refuses it.\n"
                 "      --live-policy FILE   lab-only JSON override of every spec.handoff.placement.live_policy member; present means\n"
                 "                     the helper runs the live placement policy. Nothing is defaulted.\n"
                 "      --experiment FILE    a JSON object for spec.experiment (lab profile: the manifest is then written with\n"
                 "                     profile lab; a production manifest never carries one).\n"
                 "  --plan-extra TEXT   (with --write-manifest) engine arguments the launcher passes through verbatim; part of the key\n"
                 "No process is started by these commands.\n";
}

void explain(const bc::serving::Resolved& result) {
    std::cout << "layout: " << result.typed.layout.name << '\n';
    std::cout << "engine card: " << result.typed.layout.engine_card << '\n';
    std::cout << "helpers: " << result.typed.layout.helper_cards.size() << '\n';
    std::cout << "manifest sha256: " << result.hash << '\n';
    for (auto it = result.provenance.begin(); it != result.provenance.end(); ++it) {
        auto key = it.key();
        const auto& meta = it.value();
        std::string group = key;
        auto dot = key.find('.');
        Json val = dot == std::string::npos
            ? result.values.at(key)
            : result.values.at(key.substr(0, dot)).at(key.substr(dot + 1));
        std::cout << key << " = " << val.dump() << "  ["
                  << meta.at("origin").get<std::string>() << ", "
                  << meta.at("lifecycle").get<std::string>() << "] "
                  << meta.at("source").get<std::string>() << '\n';
    }
}
} // namespace

static uint64_t parse_mib(const std::string& text, const char* flag) {
    size_t used = 0;
    unsigned long long n = 0;
    try { n = std::stoull(text, &used); } catch (const std::exception&) { used = 0; }
    if (used != text.size() || text.empty() || text[0] == '-' || n == 0)
        throw std::runtime_error(std::string(flag) + " must be a positive integer (MiB), is \"" + text + "\"");
    return n;
}

/* A JSON object an operator supplies (a live policy, a lab experiment): read once, never defaulted; empty path = none.
 * `from` names the file for the member's provenance record. */
static void load_json_input(const fs::path& path, const char* flag, Json* out, std::string* from) {
    if (path.empty()) return;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string(flag) + ": cannot read " + path.string());
    std::ostringstream text;
    text << in.rdbuf();
    *out = Json::parse(text.str(), nullptr, false);
    if (!out->is_object()) throw std::runtime_error(std::string(flag) + ": " + path.string() + " is not a JSON object");
    *from = std::string(flag) + " " + path.filename().string();
}

int main(int argc, char** argv) {
    try {
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
                if (i + 1 >= argc) throw std::runtime_error(arg + " requires a value");
                if (arg == "--from-log") log = argv[++i];
                else if (arg == "--calibration") cache = argv[++i];
                else if (arg == "--manifest") manifest = argv[++i];
                else if (arg == "--plan-cache") plan_dir = argv[++i];
                else if (arg == "--engine-binary") binary = argv[++i];
                else if (arg == "--plan-extra") extra = argv[++i];
                else if (arg == "--site") ingest_opts.site = argv[++i];
                else if (arg == "--profile") ingest_opts.profile = argv[++i];
                else if (arg == "--sys-root") ingest_opts.sys_root = argv[++i];
                else if (arg == "--set") ingest_opts.sets.push_back(argv[++i]);
                else throw std::runtime_error("calibration-ingest: unknown option " + arg);
            }
            if (log.empty() || cache.empty() || manifest.empty() || plan_dir.empty() ||
                binary.empty() || ingest_opts.site.empty() || extra.empty())
                throw std::runtime_error(
                    "calibration-ingest: --from-log --calibration --manifest --plan-cache "
                    "--engine-binary --site --plan-extra are required");
            // Do not invent a plan key from a filename or trust a supplied
            // raw key. Verify the exact resolved model, engine, and measured
            // plan against the strict lab manifest.
            if (fs::exists(cache)) ingest_opts.calibration_cache = cache;
            const auto facts = bc::serving::detect(ingest_opts.sys_root);
            const auto resolved = bc::serving::resolve(ingest_opts, facts);
            const Json& values = resolved.values;
            const fs::path home = values.at("engine").at("radiance_home").get<std::string>();
            const Json inputs = bc::serving::plan_key_inputs(
                values, facts, bc::serving::container_identity(
                    values.at("engine").at("model").get<std::string>()),
                bc::serving::engine_identity(binary, home), extra);
            const auto plan = bc::serving::plan_cache_lookup(plan_dir, inputs);
            if (!plan)
                throw std::runtime_error("calibration-ingest: exact measured plan is absent from cache");
            const auto verified = bc::serving::v1::load_manifest(
                manifest, bc::serving::v1::PlanUse::serving);
            if (!verified)
                throw std::runtime_error("calibration-ingest: strict lab manifest rejected: " +
                    bc::serving::v1::render_human("bc-serve", verified.error()));
            const auto& m = verified.value();
            if (!m.spec.plan.engine_memory ||
                m.spec.intent.layout != values.at("layout").at("name").get<std::string>() ||
                m.spec.intent.model != values.at("engine").at("model").get<std::string>() ||
                m.spec.plan.engine_memory->model_identity != inputs.at("model").get<std::string>() ||
                m.spec.plan.engine_memory->engine_identity != inputs.at("engine_build").get<std::string>())
                throw std::runtime_error("calibration-ingest: launch/model/engine identity does not match lab manifest");
            std::ifstream in(manifest, std::ios::binary);
            if (!in) throw std::runtime_error("calibration-ingest: cannot reread manifest");
            const Json raw = Json::parse(in);
            if (raw.at("spec").at("plan").at("engine_memory").at("expected") != plan->plan_json)
                throw std::runtime_error("calibration-ingest: cached measured plan differs from lab manifest");
            bc::serving::ingest_live_policy_log(
                log, cache, manifest, bc::serving::plan_key(inputs));
            std::cout << "calibration: " << cache << '\n';
            return 0;
        }
        Options opts;
        bool detection = false, schema = false, validation = false;
        bool explanation = false, json = false;
        fs::path measured_plan_file;
        std::optional<uint64_t> helpers_resident_mib, engine_resident_mib;
        std::string plan_extra;
        fs::path plan_cache, engine_binary;
        fs::path write_manifest, live_policy_file, experiment_file;
        bool unmeasured = false;
        std::vector<fs::path> positional;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto need_arg = [&](const char* what) {
                if (i + 1 >= argc)
                    throw std::runtime_error(std::string(what) + " requires a value");
                return std::string(argv[++i]);
            };
            if (arg == "--help" || arg == "-h") { usage(); return 0; }
            if (arg == "--detect") detection = true;
            else if (arg == "--schema") schema = true;
            else if (arg == "--validate") validation = true;
            else if (arg == "--explain") explanation = true;
            else if (arg == "--json") json = true;
            else if (arg == "--sys-root") opts.sys_root = need_arg("--sys-root");
            else if (arg == "--site") opts.site = need_arg("--site");
            else if (arg == "--profile") opts.profile = need_arg("--profile");
            else if (arg == "--calibration") opts.calibration_cache = need_arg("--calibration");
            else if (arg == "--write-manifest") write_manifest = need_arg("--write-manifest");
            else if (arg == "--unmeasured") unmeasured = true;
            else if (arg == "--live-policy") live_policy_file = need_arg("--live-policy");
            else if (arg == "--experiment") experiment_file = need_arg("--experiment");
            else if (arg == "--plan-extra") plan_extra = need_arg("--plan-extra");
            else if (arg == "--measured-plan") measured_plan_file = need_arg("--measured-plan");
            else if (arg == "--helpers-resident-mib") helpers_resident_mib = parse_mib(need_arg("--helpers-resident-mib"), "--helpers-resident-mib");
            else if (arg == "--engine-resident-mib") engine_resident_mib = parse_mib(need_arg("--engine-resident-mib"), "--engine-resident-mib");
            else if (arg == "--plan-cache") plan_cache = need_arg("--plan-cache");
            else if (arg == "--engine-binary") engine_binary = need_arg("--engine-binary");
            else if (arg == "--set") opts.sets.push_back(need_arg("--set"));
            else if (!arg.empty() && arg.front() == '-')
                throw std::runtime_error("unknown option " + arg);
            else positional.push_back(arg);
        }
        // BRVP06: a build runs against the ROCm SDK it recorded; an environment that points at another one is refused here, before
        // any mode runs (--help above is the only thing that answers in such an environment)
        if (const auto refusal = bc::serving::sdk_refusal("/proc/self/exe", bc::serving::current_sdk_environment()))
            throw std::runtime_error(*refusal);
        const int modes = int(detection) + int(schema) + int(validation) +
                          int(explanation) + int(!write_manifest.empty());
        const bool plan_mode = !write_manifest.empty();
        if (!plan_mode && (!plan_cache.empty() || !engine_binary.empty()))
            throw std::runtime_error("--plan-cache and --engine-binary go with --write-manifest");
        if (!plan_mode && !plan_extra.empty())
            throw std::runtime_error("--plan-extra goes with --write-manifest");
        if (!live_policy_file.empty() && experiment_file.empty())
            throw std::runtime_error("--live-policy is a lab override; supply --experiment for an explicit lab profile");
        if (write_manifest.empty() && (unmeasured || !live_policy_file.empty() || !experiment_file.empty()))
            throw std::runtime_error("--unmeasured, --live-policy and --experiment go with --write-manifest");
        if (write_manifest.empty() && !measured_plan_file.empty())
            throw std::runtime_error("--measured-plan goes with --write-manifest");
        if (unmeasured && !measured_plan_file.empty())
            throw std::runtime_error("--unmeasured has no plan: it takes no --measured-plan");
        if (measured_plan_file.empty() != (!helpers_resident_mib && !engine_resident_mib))
            throw std::runtime_error("--helpers-resident-mib and --engine-resident-mib go with --measured-plan (the host memory the "
                                     "helpers and the engine held in the measuring start that produced the plan), both or neither");
        if (!measured_plan_file.empty() && (!helpers_resident_mib || !engine_resident_mib))
            throw std::runtime_error("--measured-plan needs both --helpers-resident-mib and --engine-resident-mib");
        if (modes != 1) throw std::runtime_error("choose one mode (see --help)");
        if (schema) {
            if (!positional.empty()) throw std::runtime_error("--schema takes no files");
            std::cout << bc::serving::json_schema().dump(2) << '\n';
            return 0;
        }
        if (detection) {
            if (!positional.empty() || !opts.site.empty() || !opts.profile.empty() ||
                !opts.sets.empty() || !opts.calibration_cache.empty())
                throw std::runtime_error("--detect takes only --sys-root and --json");
            auto facts = bc::serving::detect(opts.sys_root);
            if (json) std::cout << facts.dump(2) << '\n';
            else {
                auto host = facts.at("host");
                std::cout << "host: " << host.at("mem_total_mib") << " MiB RAM, "
                          << host.at("mem_available_mib") << " MiB available, "
                          << host.at("swap_total_mib") << " MiB swap, "
                          << host.at("shm_available_mib") << " MiB /dev/shm available, "
                          << host.at("cpu_count") << " CPUs\n";
                for (const auto& g : facts.at("gpus"))
                    std::cout << "HIP " << g.at("hip_ordinal") << "  "
                              << g.at("arch").get<std::string>() << " "
                              << g.at("bdf").get<std::string>() << "  "
                              << g.at("vram_total_mib") << " MiB VRAM  "
                              << g.at("current_link_speed").get<std::string>() << " x"
                              << g.at("current_link_width") << '\n';
            }
            return 0;
        }
        if (validation) {
            if (positional.empty() || positional.size() > 2 ||
                !opts.site.empty() || !opts.profile.empty())
                throw std::runtime_error("--validate expects <site.toml> [profile.toml]");
            opts.site = positional[0];
            if (positional.size() == 2) opts.profile = positional[1];
        } else if (!positional.empty()) {
            throw std::runtime_error("unexpected file; use --site or --profile");
        }
        if (plan_mode) {
            if (!positional.empty())
                throw std::runtime_error("--write-manifest takes --site, --profile, --set and --calibration only");
            if (unmeasured && (!plan_cache.empty() || !engine_binary.empty() || !plan_extra.empty()))
                throw std::runtime_error("--unmeasured has no plan: it takes no --plan-cache, --engine-binary or --plan-extra");
            if (!unmeasured && (plan_cache.empty() || engine_binary.empty()))
                throw std::runtime_error("--write-manifest needs --plan-cache DIR and --engine-binary PATH");
            const auto facts = bc::serving::detect(opts.sys_root);
            const auto r = bc::serving::resolve(opts, facts);
            const auto& v = r.values;
            const fs::path home = v.at("engine").at("radiance_home").get<std::string>();
            if (unmeasured) {
                bc::serving::ManifestInputs in;
                in.resolved = r;
                in.profile = experiment_file.empty() ? "production" : "lab";
                in.instance_id = bc::serving::new_instance_id();
                in.producer_sha256 = bc::serving::file_sha256("/proc/self/exe");
                in.calibration_cache = opts.calibration_cache;
                load_json_input(live_policy_file, "--live-policy", &in.live_policy, &in.live_policy_from);
                load_json_input(experiment_file, "--experiment", &in.experiment, &in.experiment_from);
                const auto built = bc::serving::build_manifest(in);
                std::cerr << "live policy: unmeasured; calibration not available \n";
                bc::serving::write_manifest_file(write_manifest, built.text);
                std::cout << "manifest: " << write_manifest << " (unmeasured: for the measuring start only)\n"
                          << "digest: " << built.digest << '\n' << "no processes started\n";
                return 0;
            }
            const auto key_inputs = bc::serving::plan_key_inputs(
                v, facts, bc::serving::container_identity(v.at("engine").at("model").get<std::string>()),
                bc::serving::engine_identity(engine_binary, home), plan_extra);
            // The v1 manifest for this configuration from its measured plan: given by a measuring start (--measured-plan,
            // checked against the configuration and stored in the cache under this key) or found in the cache. A miss is exit
            // 10: a manifest without a measured plan is not written.
            std::optional<bc::serving::CachedPlan> plan;
            if (!measured_plan_file.empty()) {
                std::ifstream in(measured_plan_file, std::ios::binary);
                if (!in) throw std::runtime_error("cannot read " + measured_plan_file.string());
                std::ostringstream plan_text;
                plan_text << in.rdbuf();
                const Json measured = bc::serving::parse_plan_json(plan_text.str(), measured_plan_file.string());
                bc::serving::check_plan_for_configuration(
                    v, bc::serving::decode_engine_plan(measured, measured_plan_file.string()));
                bc::serving::plan_cache_store(plan_cache, key_inputs, measured,
                                              bc::serving::HostResident{*helpers_resident_mib, *engine_resident_mib});
                std::cerr << "plan: stored " << bc::serving::plan_key(key_inputs) << '\n';
            }
            plan = bc::serving::plan_cache_lookup(plan_cache, key_inputs);
            if (!plan) {
                std::cerr << "bc-serve: no measured plan for this configuration (key "
                          << bc::serving::plan_key(key_inputs).substr(0, 16)
                          << "); the launcher measures it on a start with the helpers up\n";
                return 10;
            }
            bc::serving::ManifestInputs in;
            in.resolved = r;
            in.instance_id = bc::serving::new_instance_id();
            in.producer_sha256 = bc::serving::file_sha256("/proc/self/exe");
            in.calibration_cache = opts.calibration_cache;
            in.profile = experiment_file.empty() ? "production" : "lab";
            in.measured = bc::serving::MeasuredPlan{*plan, key_inputs};
            load_json_input(live_policy_file, "--live-policy", &in.live_policy, &in.live_policy_from);
            load_json_input(experiment_file, "--experiment", &in.experiment, &in.experiment_from);
            const auto built = bc::serving::build_manifest(in);
            const auto& prov = built.document.at("provenance");
            const auto& placement = built.document.at("spec").at("handoff").at("placement");
            if (!placement.contains("live_policy"))
                std::cerr << "live policy: no calibration for plan " << bc::serving::plan_key(key_inputs)
                          << "; policy omitted (v1 cannot represent observe-only)\n";
            else for (auto it = prov.begin(); it != prov.end(); ++it) {
                const std::string pointer = "/spec/handoff/placement/live_policy/";
                if (it.key().rfind(pointer, 0) != 0) continue;
                const std::string field = it.key().substr(pointer.size());
                std::cerr << "live policy: " << field << "=" << placement.at("live_policy").at(field).dump()
                          << " [" << it.value().at("origin") << "] "
                          << it.value().at("source").get<std::string>() << '\n';
            }
            bc::serving::write_manifest_file(write_manifest, built.text);
            std::cout << "manifest: " << write_manifest << '\n' << "digest: " << built.digest << '\n' << "no processes started\n";
            return 0;
        }
        auto result = bc::serving::resolve(opts, bc::serving::detect(opts.sys_root));
        if (validation) {
            std::cout << "valid: " << result.hash << '\n';
        } else {
            if (json) std::cout << result.manifest.dump(2) << '\n';
            else explain(result);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bc-serve: " << error.what() << '\n';
        return 2;
    }
}
