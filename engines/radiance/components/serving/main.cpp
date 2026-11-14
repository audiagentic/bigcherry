#include "serving.h"

#include <ctime>
#include "manifest_writer.h"
#include "live_policy_ingest.h"
#include "v2/manifest.h"
#include "plan_cache.h"
#include "sdk_check.h"
#include "v2/p1_live_room.h"
#include "v2/p1_producer.h"
#include "v2/s8_rules.h"
#include "generated/v2/serving_codec.h"

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
                 "  --publish-v2-live-room [--site FILE] [--profile FILE] [--sys-root DIR]\n"
                 "  --publish-v2-p1 FILE [--site FILE] [--profile FILE] [--sys-root DIR]\n"
                 "      native P1-b producer: measuring-start paths, ownership, verified pin room; v2 inactive\n"
                 "      publishes live host/cgroup/AMD pin bytes; v2 still inactive\n"
                 "  --schema\n"
                 "  --validate <site.toml> [profile.toml] [--set key=value]\n"
                 "  --explain [--json] [--site FILE] [--profile FILE] [--set key=value]\n"
                 "  --layouts FILE|DIR  layout definitions (repeatable; a directory gives its *.toml); default BIGCHERRY_LAYOUTS,\n"
                 "                     a ':'-separated list. Layouts are run-time data: none is compiled in\n"
                 "  --configs FILE|DIR  RSC12 configs (repeatable; default BIGCHERRY_CONFIGS): one file per model, named configs and\n"
                 "                     variants (base = ...); their names select as layout.name. Given configs, layout files are\n"
                 "                     read only from --layouts\n"
                 "  --model-root DIR    the directory a config's relative model path is under (default BC_MODEL_ROOT)\n"
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
                 "      --calibration-run    the lab calibration start: lab profile with the explicit --live-policy and no\n"
                 "                     experiment, so the measured plan is the production plan (launcher --calibrate).\n"
                 "      --experiment FILE    a JSON object for spec.experiment (lab profile: the manifest is then written with\n"
                 "                     profile lab; a production manifest never carries one).\n"
                 "  --plan-extra TEXT   (with --write-manifest) engine arguments the launcher passes through verbatim; part of the key\n"
                 "  --power-profile NAME   (with --write-manifest) the named power profile; stock observes the cards\n"
                 "  --rank-demand FILE     (with --measured-plan) the engine's --plan-memory-ranks record of the same start\n"
                 "  --calibration-records FILE   (with --write-manifest) JSON array of P2 records; used where they match\n"
                 "No process is started by these commands.\n";
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

static uint64_t parse_mib(const std::string& text, const char* flag, bool zero_allowed) {
    size_t used = 0;
    unsigned long long n = 0;
    try { n = std::stoull(text, &used); } catch (const std::exception&) { used = 0; }
    if (used != text.size() || text.empty() || text[0] == '-' || (n == 0 && !zero_allowed))
        throw std::runtime_error(std::string(flag) + (zero_allowed ? " must be a whole number of MiB, is \"" : " must be a positive integer (MiB), is \"") + text + "\"");
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
                else if (arg == "--layouts") ingest_opts.layouts.emplace_back(argv[++i]);
                else if (arg == "--configs") ingest_opts.configs.emplace_back(argv[++i]);
                else if (arg == "--model-root") ingest_opts.model_root = argv[++i];
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
            const auto verified = bc::serving::v2::load_manifest(
                manifest, bc::serving::v2::PlanUse::serving);
            if (!verified)
                throw std::runtime_error("calibration-ingest: strict lab manifest rejected: " +
                    bc::serving::v2::render_human("bc-serve", verified.error()));
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
                log, cache, manifest, bc::serving::calibration_key(inputs));
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
        fs::path write_manifest, live_policy_file, experiment_file, rank_demand_file, calibration_records_file;
        std::string power_profile;
        // --calibration-run: the launcher's lab calibration start (--calibrate). A lab-profile manifest with an explicit live
        // policy and NO experiment, so the plan it measures is the production plan.
        bool calibration_run = false;
        bool unmeasured = false;
        bool publish_v2_live_room = false;
        std::string publish_v2_p1;
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
            else if (arg == "--publish-v2-live-room") publish_v2_live_room = true;
            else if (arg == "--publish-v2-p1") publish_v2_p1 = need_arg("--publish-v2-p1");
            else if (arg == "--validate") validation = true;
            else if (arg == "--explain") explanation = true;
            else if (arg == "--json") json = true;
            else if (arg == "--sys-root") opts.sys_root = need_arg("--sys-root");
            else if (arg == "--site") opts.site = need_arg("--site");
            else if (arg == "--profile") opts.profile = need_arg("--profile");
            else if (arg == "--layouts") opts.layouts.emplace_back(need_arg("--layouts"));
            else if (arg == "--configs") opts.configs.emplace_back(need_arg("--configs"));
            else if (arg == "--model-root") opts.model_root = need_arg("--model-root");
            else if (arg == "--calibration") opts.calibration_cache = need_arg("--calibration");
            else if (arg == "--write-manifest") write_manifest = need_arg("--write-manifest");
            else if (arg == "--unmeasured") unmeasured = true;
            else if (arg == "--live-policy") live_policy_file = need_arg("--live-policy");
            else if (arg == "--calibration-run") calibration_run = true;
            else if (arg == "--experiment") experiment_file = need_arg("--experiment");
            else if (arg == "--plan-extra") plan_extra = need_arg("--plan-extra");
            else if (arg == "--rank-demand") rank_demand_file = need_arg("--rank-demand");
            else if (arg == "--calibration-records") calibration_records_file = need_arg("--calibration-records");
            else if (arg == "--power-profile") power_profile = need_arg("--power-profile");
            else if (arg == "--measured-plan") measured_plan_file = need_arg("--measured-plan");
            else if (arg == "--helpers-resident-mib") helpers_resident_mib = parse_mib(need_arg("--helpers-resident-mib"), "--helpers-resident-mib", true);
            else if (arg == "--engine-resident-mib") engine_resident_mib = parse_mib(need_arg("--engine-resident-mib"), "--engine-resident-mib", false);
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
                          int(explanation) + int(!write_manifest.empty()) +
                          int(publish_v2_live_room) + int(!publish_v2_p1.empty());
        const bool plan_mode = !write_manifest.empty();
        if (!plan_mode && (!plan_cache.empty() || !engine_binary.empty()))
            throw std::runtime_error("--plan-cache and --engine-binary go with --write-manifest");
        if (!plan_mode && !plan_extra.empty())
            throw std::runtime_error("--plan-extra goes with --write-manifest");
        if (write_manifest.empty() && (!calibration_records_file.empty() || !power_profile.empty()))
            throw std::runtime_error("--calibration-records and --power-profile go with --write-manifest");
        if (measured_plan_file.empty() != rank_demand_file.empty())
            throw std::runtime_error("--measured-plan and --rank-demand go together (the engine's --plan-memory and --plan-memory-ranks)");
        if (!live_policy_file.empty() && experiment_file.empty() && !calibration_run)
            throw std::runtime_error("--live-policy is a lab override; supply --experiment for an explicit lab profile, or --calibration-run");
        if (calibration_run && (live_policy_file.empty() || !experiment_file.empty()))
            throw std::runtime_error("--calibration-run takes an explicit --live-policy and no --experiment (the production plan is measured)");
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
        if (!publish_v2_p1.empty()) {
            if (!positional.empty() || !opts.calibration_cache.empty() || json ||
                unmeasured || !measured_plan_file.empty() || !plan_extra.empty() ||
                !live_policy_file.empty() || !experiment_file.empty() || calibration_run ||
                helpers_resident_mib || engine_resident_mib)
                throw std::runtime_error(
                    "--publish-v2-p1 accepts only --site/--profile/--set/--sys-root");
            std::ifstream in(publish_v2_p1, std::ios::binary);
            if (!in) throw std::runtime_error("P1 cannot open request " + publish_v2_p1);
            const std::string bytes((std::istreambuf_iterator<char>(in)), {});
            const auto request = bc::serving::parse_plan_json(bytes, publish_v2_p1);
            const auto facts = bc::serving::detect(opts.sys_root);
            const auto resolved = bc::serving::resolve(opts, facts);
            const auto record = bc::serving::v2::produce_p1_inputs(
                resolved, facts, opts.sys_root, request);
            std::cout << record.dump(2) << '\n';
            return 0;
        }
        if (publish_v2_live_room) {
            if (!positional.empty() || !opts.calibration_cache.empty() || json ||
                unmeasured || !measured_plan_file.empty() || !plan_extra.empty() ||
                !live_policy_file.empty() || !experiment_file.empty() || calibration_run)
                throw std::runtime_error(
                    "--publish-v2-live-room accepts only --site/--profile/--set/--sys-root");
            // Native live-room publication is NOT a v2 startup manifest or a
            // substitute for fresh engine/owner-map/worker measurements.
            const auto facts = bc::serving::detect(opts.sys_root);
            const auto resolved = bc::serving::resolve(opts, facts);
            std::vector<std::string> selected = resolved.typed.layout.engine_cards;
            selected.insert(selected.end(), resolved.typed.layout.helper_cards.begin(),
                            resolved.typed.layout.helper_cards.end());
            const auto measured = bc::serving::v2::measure_live_room(
                facts, opts.sys_root, selected);
            if (!measured)
                throw std::runtime_error(
                    "P1 live-room refusal: " + measured.error().path + ": " +
                    measured.error().message);
            const auto& room = measured.value();
            Json record = {
                {"schema_version", 1}, {"kind", "bc_serve_v2_live_room"},
                {"layout", resolved.typed.layout.name},
                {"engine_bdfs", resolved.typed.layout.engine_cards},
                {"helper_bdfs", resolved.typed.layout.helper_cards},
                {"resource_inputs", {
                    {"mem_available_bytes", room.mem_available_bytes},
                    {"cgroup_headroom_bytes", room.cgroup_headroom_bytes},
                    {"pin_available_bytes", room.pin_available_bytes}
                }},
                {"provenance", room.evidence}
            };
            std::cout << record.dump(2) << '\n';
            return 0;
        }
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
            if (power_profile.empty())
                throw std::runtime_error("--write-manifest needs --power-profile NAME (stock observes the cards)");
            bc::serving::v2::WriterInputs s10;
            s10.power_profile = power_profile;
            s10.now_unix_s = static_cast<int64_t>(std::time(nullptr));
            s10.calibration_records = Json::array();
            if (!calibration_records_file.empty()) {
                std::ifstream rin(calibration_records_file, std::ios::binary);
                if (!rin) throw std::runtime_error("--calibration-records: cannot read " + calibration_records_file.string());
                std::ostringstream text;
                text << rin.rdbuf();
                s10.calibration_records = Json::parse(text.str(), nullptr, false);
                if (!s10.calibration_records.is_array())
                    throw std::runtime_error("--calibration-records: " + calibration_records_file.string() + " is not a JSON array");
            }
            const auto facts = bc::serving::detect(opts.sys_root);
            const auto r = bc::serving::resolve(opts, facts);
            const auto& v = r.values;
            const fs::path home = v.at("engine").at("radiance_home").get<std::string>();
            if (unmeasured) {
                bc::serving::ManifestInputs in;
                in.resolved = r;
                in.s10 = s10;
                in.profile = (experiment_file.empty() && !calibration_run) ? "production" : "lab";
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
                Json ranks;
                std::string ranks_from;
                load_json_input(rank_demand_file, "--rank-demand", &ranks, &ranks_from);
                bc::serving::plan_cache_store(plan_cache, key_inputs, measured, ranks,
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
            {
                // P1: measure now, at every manifest written: the cached plan and per-rank demand, the owner map, and the
                // host's live room. The retention target is the host KV tier the launcher runs with, over the measured KV
                // density (one full session's KV over its context), at most every session at full context.
                const int64_t ctx = v.at("service").at("max_context").get<int64_t>();
                const int64_t sessions = v.at("service").at("max_sessions").get<int64_t>();
                const int64_t per_token = plan->plan.kv_one_session.bytes / std::max<int64_t>(1, ctx);
                if (per_token <= 0) throw std::runtime_error("the measured plan has no KV per token");
                const int64_t retention = std::min<int64_t>(
                    v.at("memory").at("host_kv_mib").get<int64_t>() * 1048576 / per_token, ctx * sessions);
                const bool helpers = !r.manifest.at("process_helpers").empty();
                const Json sources = {
                    {"engine_plan", plan->plan_json}, {"rank_demand", plan->rank_demand},
                    {"engine_binary", engine_binary.string()},
                    {"helper_binary", helpers ? (home / "helpers" / "r3_moe_worker").string() : std::string()},
                    {"retention_target_tokens", retention},
                    {"margin", bc::serving::host_ledger_margin(r, opts.calibration_cache)},
                    {"run_id", "bc-serve --write-manifest " + write_manifest.filename().string()}};
                s10.p1_record = bc::serving::v2::produce_p1(r, facts, opts.sys_root, sources);
                // The plan echoes the host pool it was measured with; S8 derives the pool from what was measured. When they
                // differ, the plan is for another pool: exit 11 with the derived one, and the launcher measures once more
                // with it (that configuration has its own cache key, so this happens once per configuration).
                bc::serving::v2::MeasuredResourceInputs measured_inputs;
                if (const auto e = bc::serving::v2::decode(s10.p1_record.at("resource_inputs"), "p1.resource_inputs",
                                                           &measured_inputs))
                    throw std::runtime_error(e->message);
                const auto derived = bc::serving::v2::derived_host_pool_mib(measured_inputs);
                if (!derived) throw std::runtime_error("the derived host pool overflows");
                if (*derived != plan->plan.host_pool_mib) {
                    std::cerr << "bc-serve: the plan was measured with a " << plan->plan.host_pool_mib
                              << " MiB host pool; the measured inputs need " << *derived << " MiB\n";
                    std::cout << "derived_host_pool_mib=" << *derived << '\n';
                    return 11;
                }
            }
            bc::serving::ManifestInputs in;
            in.resolved = r;
            in.instance_id = bc::serving::new_instance_id();
            in.producer_sha256 = bc::serving::file_sha256("/proc/self/exe");
            in.calibration_cache = opts.calibration_cache;
            in.profile = (experiment_file.empty() && !calibration_run) ? "production" : "lab";
            in.measured = bc::serving::MeasuredPlan{*plan, key_inputs};
            in.s10 = s10;
            load_json_input(live_policy_file, "--live-policy", &in.live_policy, &in.live_policy_from);
            load_json_input(experiment_file, "--experiment", &in.experiment, &in.experiment_from);
            const auto built = bc::serving::build_manifest(in);
            const auto& prov = built.document.at("provenance");
            const auto& spec = built.document.at("spec");
            const bool has_handoff = spec.contains("handoff");   // an engine-only manifest has no hand-off section
            const auto placement = has_handoff ? spec.at("handoff").at("placement") : bc::serving::Json::object();
            if (has_handoff && !placement.contains("live_policy"))
                std::cerr << "live policy: no calibration for key " << bc::serving::calibration_key(key_inputs)
                          << "; " << placement.at("live_policy_state").at("reason_code").get<std::string>() << ": "
                          << placement.at("live_policy_state").at("reason_detail").get<std::string>() << '\n';
            else if (has_handoff) for (auto it = prov.begin(); it != prov.end(); ++it) {
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
