#include "serving.h"
#include "manifest_writer.h"
#include "plan_cache.h"
#include "sdk_check.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
                 "  --calibration FILE  (with --validate, --explain) a calibration cache over the seeds\n"
                 "  --plan-extra TEXT   (with the --plan-* modes) engine arguments the launcher passes through verbatim; part of the key\n"
                 "  --write-manifest OUT --plan-cache DIR --engine-binary PATH [--plan-extra TEXT] [site/profile/set]\n"
                 "      writes the v1 serving manifest for this configuration to OUT (read-only; an existing OUT is refused)\n"
                 "      from its measured plan and the planner's owner map, and reads it back through the shared strict reader\n"
                 "      before it stays; a miss in the plan cache exits 10. Starts no process.\n"
                 "      --unmeasured   (with --write-manifest; no --plan-cache or --engine-binary) writes the unmeasured manifest\n"
                 "                     of the measuring start instead: owner map and settings, plan.state unmeasured, no plan. Only\n"
                 "                     a measuring start (radiance --plan-memory, r3_moe_worker --measuring) accepts it; every\n"
                 "                     serving start refuses it.\n"
                 "      --live-policy FILE   a JSON object with every spec.handoff.placement.live_policy member; present means\n"
                 "                     the helper runs the live placement policy. Nothing is defaulted.\n"
                 "      --experiment FILE    a JSON object for spec.experiment (lab profile: the manifest is then written with\n"
                 "                     profile lab; a production manifest never carries one).\n"
                 "  --plan-key --plan-cache DIR --engine-binary PATH [--site FILE] [--profile FILE] [--set key=value]\n"
                 "      prints this configuration's plan key (what --plan-store must be told to expect)\n"
                 "  --plan-check-args --plan-cache DIR --engine-binary PATH [--site FILE] [--profile FILE] [--set key=value]\n"
                 "      looks the plan up and prints the engine's start-up check flags, one argument per line:\n"
                 "      --plan-check FILE --plan-check-floor-ppm N, N being the selected calibration\n"
                 "      kv_backing_floor.fraction_ppm (0 when the objective selects none). Starts no process and measures\n"
                 "      nothing: a miss exits 10 with the remedy.\n"
                 "  --plan-store --expect-key KEY --plan-file FILE --plan-cache DIR --engine-binary PATH [site/profile/set]\n"
                 "      records a plan the engine measured (radiance --plan-memory FILE, with its helpers up) under this\n"
                 "      configuration's key, after the strict v1 decode. Refused unless the key still equals KEY (the engine\n"
                 "      build, model and configuration are those the measurement was asked for) and the plan agrees with the\n"
                 "      configuration (host pool, host consumer list).\n"
                 "No process is started by these commands, except --plan-memory on a cache miss.\n";
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
        Options opts;
        bool detection = false, schema = false, validation = false;
        bool explanation = false, json = false, plan_check_args = false, plan_store = false, plan_key_only = false;
        fs::path plan_file;
        std::string expect_key, plan_extra;
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
            else if (arg == "--plan-check-args") plan_check_args = true;
            else if (arg == "--plan-store") plan_store = true;
            else if (arg == "--plan-key") plan_key_only = true;
            else if (arg == "--write-manifest") write_manifest = need_arg("--write-manifest");
            else if (arg == "--unmeasured") unmeasured = true;
            else if (arg == "--live-policy") live_policy_file = need_arg("--live-policy");
            else if (arg == "--experiment") experiment_file = need_arg("--experiment");
            else if (arg == "--expect-key") expect_key = need_arg("--expect-key");
            else if (arg == "--plan-extra") plan_extra = need_arg("--plan-extra");
            else if (arg == "--plan-file") plan_file = need_arg("--plan-file");
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
                          int(explanation) + int(plan_check_args) + int(plan_store) + int(plan_key_only) + int(!write_manifest.empty());
        const bool plan_mode = plan_check_args || plan_store || plan_key_only || !write_manifest.empty();
        if (!plan_mode && (!plan_cache.empty() || !engine_binary.empty()))
            throw std::runtime_error("--plan-cache and --engine-binary go with --plan-check-args, --plan-store or --write-manifest");
        if (!plan_mode && !plan_extra.empty())
            throw std::runtime_error("--plan-extra goes with --plan-check-args, --plan-store, --plan-key or --write-manifest");
        if (write_manifest.empty() && (unmeasured || !live_policy_file.empty() || !experiment_file.empty()))
            throw std::runtime_error("--unmeasured, --live-policy and --experiment go with --write-manifest");
        if (!plan_store && (!plan_file.empty() || !expect_key.empty()))
            throw std::runtime_error("--plan-file and --expect-key go with --plan-store");
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
            const char* mode = plan_check_args ? "--plan-check-args"
                               : plan_store ? "--plan-store" : plan_key_only ? "--plan-key" : "--write-manifest";
            if (plan_store && (plan_file.empty() || expect_key.empty()))
                throw std::runtime_error("--plan-store needs --plan-file FILE and --expect-key KEY");
            if (!positional.empty())
                throw std::runtime_error(std::string(mode) + " takes --site, --profile, --set and --calibration only");
            if (unmeasured && (!plan_cache.empty() || !engine_binary.empty() || !plan_extra.empty()))
                throw std::runtime_error("--unmeasured has no plan: it takes no --plan-cache, --engine-binary or --plan-extra");
            if (!unmeasured && (plan_cache.empty() || engine_binary.empty()))
                throw std::runtime_error(std::string(mode) + " needs --plan-cache DIR and --engine-binary PATH");
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
                bc::serving::write_manifest_file(write_manifest, built.text);
                std::cout << "manifest: " << write_manifest << " (unmeasured: for the measuring start only)\n"
                          << "digest: " << built.digest << '\n' << "no processes started\n";
                return 0;
            }
            const auto key_inputs = bc::serving::plan_key_inputs(
                v, facts, bc::serving::container_identity(v.at("engine").at("model").get<std::string>()),
                bc::serving::engine_identity(engine_binary, home), plan_extra);
            if (plan_key_only) {
                std::cout << bc::serving::plan_key(key_inputs) << '\n';
                return 0;
            }
            if (!write_manifest.empty()) {
                // The v1 manifest for this configuration from its measured plan. A miss is exit 10, as for --plan-check-args:
                // a manifest without a measured plan is not written.
                const auto hit = bc::serving::plan_cache_lookup(plan_cache, key_inputs);
                if (!hit) {
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
                in.measured = bc::serving::MeasuredPlan{*hit, key_inputs};
                load_json_input(live_policy_file, "--live-policy", &in.live_policy, &in.live_policy_from);
                load_json_input(experiment_file, "--experiment", &in.experiment, &in.experiment_from);
                const auto built = bc::serving::build_manifest(in);
                bc::serving::write_manifest_file(write_manifest, built.text);
                std::cout << "manifest: " << write_manifest << '\n' << "digest: " << built.digest << '\n' << "no processes started\n";
                return 0;
            }
            if (plan_store) {
                if (bc::serving::plan_key(key_inputs) != expect_key)
                    throw std::runtime_error("the plan was measured for key " + expect_key + " but this configuration, model or "
                                             "engine build is now key " + bc::serving::plan_key(key_inputs) +
                                             "; it is not stored under the wrong key");
                std::ifstream in(plan_file, std::ios::binary);
                if (!in) throw std::runtime_error("cannot read " + plan_file.string());
                std::ostringstream plan_text;
                plan_text << in.rdbuf();
                const Json measured = bc::serving::parse_plan_json(plan_text.str(), plan_file.string());
                bc::serving::check_plan_for_configuration(
                    v, bc::serving::decode_engine_plan(measured, plan_file.string()));
                const auto stored = bc::serving::plan_cache_store(plan_cache, key_inputs, measured);
                std::cout << "plan: stored " << stored.key << '\n';
                return 0;
            }
            if (plan_check_args) {
                // Lookup only: a plan comes from a measuring start with the helpers up (bc-serve alone cannot measure
                // it: without helper-owned experts the engine's plan is a different one). A miss is exit 10, never a
                // silent skip of the check.
                const auto hit = bc::serving::plan_cache_lookup(plan_cache, key_inputs);
                if (!hit) {
                    std::cerr << "bc-serve: no measured plan for this configuration (key "
                              << bc::serving::plan_key(key_inputs).substr(0, 16)
                              << "); the launcher measures it on a start with the helpers up\n";
                    return 10;
                }
                std::cerr << "plan: cached " << hit->key << " (measured " << hit->measured_at << ")\n";
                for (const std::string& a : bc::serving::plan_check_args(
                         plan_cache, *hit, bc::serving::kv_backing_floor_ppm(v, opts.calibration_cache)))
                    std::cout << a << '\n';
                return 0;
            }
            throw std::logic_error("unreachable plan mode");
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
