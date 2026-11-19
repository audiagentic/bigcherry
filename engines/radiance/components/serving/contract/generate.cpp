// bc_contract_gen [--successor] REGISTRY OUT_DIR [RADIANCE_SRC COMPONENT_TOML...]: write every generated contract file (registry.h,
// generate_all) into OUT_DIR. With RADIANCE_SRC the registry's defaults are verified against their sources first
// (core/rad_core.h Config initialisers and the component files' [setting.R3_*] defaults) and a divergence fails.
// Regenerate after editing the registry:
//   bc_contract_gen engines/radiance/components/serving/contract/serving-v1.toml engines/radiance/components/serving/generated \\
//       <radiance source tree> engines/radiance/components/{moe-remote,libr3,ngram-helper}/component.toml
// and copy generated/v2/serving-v2.schema.json to docs/design/serving-manifest-v2/serving-manifest.v2.schema.json.
// bc_contract_check (CI) fails when a committed copy differs from what this writes.
#include "registry.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
    const bool successor = argc > 1 && std::string_view(argv[1]) == "--successor";
    const int registry_arg = successor ? 2 : 1;
    const int out_arg = registry_arg + 1;
    const int source_arg = out_arg + 1;
    if (argc <= out_arg) {
        std::fprintf(stderr, "usage: bc_contract_gen [--successor] REGISTRY.toml OUT_DIR [RADIANCE_SRC COMPONENT_TOML...]\n");
        return 2;
    }
    std::ifstream in(argv[registry_arg], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "bc_contract_gen: cannot read %s\n", argv[registry_arg]);
        return 2;
    }
    std::ostringstream text;
    text << in.rdbuf();
    bc::contract::Registry r;
    std::string why;
    const auto edition = successor ? bc::contract::Edition::successor : bc::contract::Edition::current;
    if (!bc::contract::load_registry(text.str(), argv[registry_arg], &r, &why, edition)) {
        std::fprintf(stderr, "bc_contract_gen: %s\n", why.c_str());
        return 1;
    }
    if (argc > source_arg) {
        std::vector<std::string> components(argv + source_arg + 1, argv + argc);
        if (!bc::contract::verify_defaults(r, argv[source_arg], components, &why)) {
            std::fprintf(stderr, "bc_contract_gen: a default diverges from its source: %s\n", why.c_str());
            return 1;
        }
    }
    const std::filesystem::path out = argv[out_arg];
    std::filesystem::create_directories(out);
    for (const auto& [name, content] : bc::contract::generate_all(r)) {
        std::ofstream f(out / name, std::ios::binary | std::ios::trunc);
        f << content;
        if (!f) {
            std::fprintf(stderr, "bc_contract_gen: cannot write %s\n", (out / name).string().c_str());
            return 1;
        }
    }
    std::printf("bc_contract_gen: wrote %zu files to %s (%s)\n", bc::contract::generate_all(r).size(), argv[out_arg],
                bc::contract::fingerprint(r).c_str());
    return 0;
}
