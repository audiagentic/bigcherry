// bc_contract_gen REGISTRY OUT_DIR [RADIANCE_SRC COMPONENT_TOML...]: write every generated contract file (registry.h,
// generate_all) into OUT_DIR. With RADIANCE_SRC the registry's defaults are verified against their sources first
// (core/rad_core.h Config initialisers and the component files' [setting.R3_*] defaults) and a divergence fails.
// Regenerate after editing the registry:
//   bc_contract_gen engines/radiance/components/serving/contract/serving-v1.toml engines/radiance/components/serving/generated \\
//       <radiance source tree> engines/radiance/components/{moe-remote,libr3,ngram-helper}/component.toml
// and copy generated/serving-v1.schema.json to docs/design/serving-manifest-v1/serving-manifest.v1.schema.json.
// bc_contract_check (CI) fails when a committed copy differs from what this writes.
#include "registry.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: bc_contract_gen REGISTRY.toml OUT_DIR [RADIANCE_SRC COMPONENT_TOML...]\n");
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "bc_contract_gen: cannot read %s\n", argv[1]);
        return 2;
    }
    std::ostringstream text;
    text << in.rdbuf();
    bc::contract::Registry r;
    std::string why;
    if (!bc::contract::load_registry(text.str(), argv[1], &r, &why)) {
        std::fprintf(stderr, "bc_contract_gen: %s\n", why.c_str());
        return 1;
    }
    if (argc > 3) {
        std::vector<std::string> components(argv + 4, argv + argc);
        if (!bc::contract::verify_defaults(r, argv[3], components, &why)) {
            std::fprintf(stderr, "bc_contract_gen: a default diverges from its source: %s\n", why.c_str());
            return 1;
        }
    }
    const std::filesystem::path out = argv[2];
    std::filesystem::create_directories(out);
    for (const auto& [name, content] : bc::contract::generate_all(r)) {
        std::ofstream f(out / name, std::ios::binary | std::ios::trunc);
        f << content;
        if (!f) {
            std::fprintf(stderr, "bc_contract_gen: cannot write %s\n", (out / name).string().c_str());
            return 1;
        }
    }
    std::printf("bc_contract_gen: wrote %zu files to %s (%s)\n", bc::contract::generate_all(r).size(), argv[2],
                bc::contract::fingerprint(r).c_str());
    return 0;
}
