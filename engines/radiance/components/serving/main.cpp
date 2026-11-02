#include "serving.h"

#include <filesystem>
#include <fstream>
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

int main(int argc, char** argv) {
    try {
        Options opts;
        bool detection = false, schema = false, validation = false;
        bool explanation = false, dry = false, json = false;
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
            else if (arg == "--set") opts.sets.push_back(need_arg("--set"));
            else if (arg == "--out") out = need_arg("--out");
            else if (!arg.empty() && arg.front() == '-')
                throw std::runtime_error("unknown option " + arg);
            else positional.push_back(arg);
        }
        const int modes = int(detection) + int(schema) + int(validation) +
                          int(explanation) + int(dry);
        if (modes != 1) throw std::runtime_error("choose one mode (see --help)");
        if (schema) {
            if (!positional.empty()) throw std::runtime_error("--schema takes no files");
            std::cout << bc::serving::json_schema().dump(2) << '\n';
            return 0;
        }
        if (detection) {
            if (!positional.empty() || !opts.site.empty() || !opts.profile.empty() ||
                !opts.sets.empty() || !out.empty())
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
