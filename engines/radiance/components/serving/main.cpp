#include "serving.h"
#include "plan_cache.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
                 "  --dry-run --out manifest.json [--site FILE] [--profile FILE] [--set key=value]\n"
                 "  --calibration FILE  (with --validate, --explain, --dry-run) a calibration cache over the seeds\n"
                 "  --plan-memory --plan-cache DIR --engine-binary PATH [--site FILE] [--profile FILE] [--set key=value]\n"
                 "      measures the engine's memory plan on the engine card (radiance --plan-memory) unless DIR already\n"
                 "      holds it for this configuration; run it under the gpu lease (agent-lease acquire gpu -- ...)\n"
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

int main(int argc, char** argv) {
    try {
        Options opts;
        bool detection = false, schema = false, validation = false;
        bool explanation = false, dry = false, json = false, plan_memory = false;
        fs::path plan_cache, engine_binary;
        fs::path out;
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
            else if (arg == "--dry-run") dry = true;
            else if (arg == "--json") json = true;
            else if (arg == "--sys-root") opts.sys_root = need_arg("--sys-root");
            else if (arg == "--site") opts.site = need_arg("--site");
            else if (arg == "--profile") opts.profile = need_arg("--profile");
            else if (arg == "--calibration") opts.calibration_cache = need_arg("--calibration");
            else if (arg == "--plan-memory") plan_memory = true;
            else if (arg == "--plan-cache") plan_cache = need_arg("--plan-cache");
            else if (arg == "--engine-binary") engine_binary = need_arg("--engine-binary");
            else if (arg == "--set") opts.sets.push_back(need_arg("--set"));
            else if (arg == "--out") out = need_arg("--out");
            else if (!arg.empty() && arg.front() == '-')
                throw std::runtime_error("unknown option " + arg);
            else positional.push_back(arg);
        }
        const int modes = int(detection) + int(schema) + int(validation) +
                          int(explanation) + int(dry) + int(plan_memory);
        if (!plan_memory && (!plan_cache.empty() || !engine_binary.empty()))
            throw std::runtime_error("--plan-cache and --engine-binary go with --plan-memory");
        if (modes != 1) throw std::runtime_error("choose one mode (see --help)");
        if (schema) {
            if (!positional.empty()) throw std::runtime_error("--schema takes no files");
            std::cout << bc::serving::json_schema().dump(2) << '\n';
            return 0;
        }
        if (detection) {
            if (!positional.empty() || !opts.site.empty() || !opts.profile.empty() ||
                !opts.sets.empty() || !out.empty() || !opts.calibration_cache.empty())
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
        if (plan_memory) {
            if (!positional.empty() || !out.empty())
                throw std::runtime_error("--plan-memory takes --site, --profile, --set and --calibration only");
            if (plan_cache.empty() || engine_binary.empty())
                throw std::runtime_error("--plan-memory needs --plan-cache DIR and --engine-binary PATH");
            const auto facts = bc::serving::detect(opts.sys_root);
            const auto r = bc::serving::resolve(opts, facts);
            const auto& v = r.values;
            const fs::path home = v.at("engine").at("radiance_home").get<std::string>();
            const auto key_inputs = bc::serving::plan_key_inputs(
                v, facts, bc::serving::container_identity(v.at("engine").at("model").get<std::string>()),
                bc::serving::engine_identity(engine_binary, home));
            if (const auto hit = bc::serving::plan_cache_lookup(plan_cache, key_inputs)) {
                std::cout << "plan: cached " << hit->key << " (measured " << hit->measured_at << ")\n";
                return 0;
            }
            const std::string card = v.at("layout").at("engine_card").get<std::string>();
            uint32_t ordinal = UINT32_MAX;
            for (const auto& g : facts.at("gpus"))
                if (g.at("bdf") == card) ordinal = g.at("hip_ordinal").get<uint32_t>();
            if (ordinal == UINT32_MAX) throw std::runtime_error("the engine card " + card + " is not detected");
            fs::create_directories(plan_cache);
            const std::string key = bc::serving::plan_key(key_inputs);
            const fs::path manifest = plan_cache / (key + ".manifest.json"), measured = plan_cache / (key + ".plan.json");
            {
                std::ofstream m(manifest, std::ios::binary | std::ios::trunc);
                m << r.manifest.dump(2) << '\n';
                if (!m) throw std::runtime_error("cannot write " + manifest.string());
            }
            double wall = 0;
            const auto plan = bc::serving::measure_engine_plan(engine_binary, manifest, ordinal, measured, &wall);
            const auto stored = bc::serving::plan_cache_store(plan_cache, key_inputs, plan);
            fs::remove(manifest);
            fs::remove(measured);
            std::cout << "plan: measured " << stored.key << " on " << card << " in " << std::fixed
                      << std::setprecision(1) << wall << " s\n";
            return 0;
        }
        if (dry && out.empty()) throw std::runtime_error("--dry-run requires --out");
        if (!dry && !out.empty()) throw std::runtime_error("--out requires --dry-run");
        auto result = bc::serving::resolve(opts, bc::serving::detect(opts.sys_root));
        if (validation) {
            std::cout << "valid: " << result.hash << '\n';
        } else if (explanation) {
            if (json) std::cout << result.manifest.dump(2) << '\n';
            else explain(result);
        } else {
            // Read-only dry-run writes a manifest, but starts no processes.
            if (fs::exists(out)) throw std::runtime_error("refuse to overwrite " + out.string());
            std::ofstream file(out, std::ios::binary | std::ios::out);
            if (!file) throw std::runtime_error("cannot create " + out.string());
            file << result.manifest.dump(2) << '\n';
            file.close();
            if (!file) throw std::runtime_error("cannot write " + out.string());
            std::cout << "manifest: " << out << '\n'
                      << "sha256: " << result.hash << '\n'
                      << "no processes started\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bc-serve: " << error.what() << '\n';
        return 2;
    }
}
