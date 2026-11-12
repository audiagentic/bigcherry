// bc_contract_check: the committed generated files (generated/v2/) and the specification's schema and field-reference copies
// (docs/design/serving-manifest-v2/) are exactly what the registry generates. Run by
// CI as a card-free component test; on a difference it names the file and the first differing line.
#include "registry.h"
#include "../generated/v2/serving_types.h"   // the committed model must compile, and match the registry

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef BC_CONTRACT_SOURCE_DIR
#error "BC_CONTRACT_SOURCE_DIR must name engines/radiance/components/serving"
#endif
#ifndef BC_CONTRACT_RADIANCE_SRC
#error "BC_CONTRACT_RADIANCE_SRC must name the radiance source tree (core/rad_core.h)"
#endif
#ifndef BC_CONTRACT_DOCS_DIR
#error "BC_CONTRACT_DOCS_DIR must name docs/design/serving-manifest-v2"
#endif

namespace {
bool read(const std::string& path, std::string* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream s;
    s << in.rdbuf();
    *out = s.str();
    return true;
}

int first_difference(const std::string& a, const std::string& b) {
    int line = 1;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        if (a[i] != b[i]) return line;
        if (a[i] == '\n') ++line;
    }
    return line;
}
}  // namespace

int main() {
    const std::string src = BC_CONTRACT_SOURCE_DIR;
    const std::string docs = BC_CONTRACT_DOCS_DIR;
    std::string text, why;
    if (!read(src + "/contract/serving-v2.toml", &text)) {
        std::printf("FAIL: cannot read the registry\n");
        return 1;
    }
    bc::contract::Registry r;
    if (!bc::contract::load_registry(text, "contract/serving-v2.toml", &r, &why)) {
        std::printf("FAIL: %s\n", why.c_str());
        return 1;
    }
    int failures = 0;
    // a default is never copied in by hand: each must equal the value of the place it names (spec section 7)
    const std::vector<std::string> components = {src + "/../moe-remote/component.toml", src + "/../libr3/component.toml",
                                                 src + "/../ngram-helper/component.toml"};
    if (!bc::contract::verify_defaults(r, BC_CONTRACT_RADIANCE_SRC, components, &why)) {
        std::printf("FAIL: a default diverges from its source: %s\n", why.c_str());
        ++failures;
    }
    const auto files = bc::contract::generate_all(r);
    if (files != bc::contract::generate_all(r)) {
        std::printf("FAIL: the generator is nondeterministic\n");
        ++failures;
    }
    auto compare = [&](const std::string& committed_path, const std::string& want) {
        std::string have;
        if (!read(committed_path, &have)) {
            std::printf("FAIL: %s is missing; regenerate (contract/generate.cpp)\n", committed_path.c_str());
            ++failures;
        } else if (have != want) {
            std::printf("FAIL: %s differs from the registry's output from line %d; regenerate (contract/generate.cpp)\n",
                        committed_path.c_str(), first_difference(have, want));
            ++failures;
        }
    };
    for (const auto& [name, content] : files) compare(src + "/generated/v2/" + name, content);
    compare(docs + "/serving-manifest.v2.schema.json", files.at("serving-v2.schema.json"));
    compare(docs + "/serving-fields.md", files.at("serving-fields.md"));
    if (bc::serving::v2::kFingerprint != bc::contract::fingerprint(r) ||
        bc::serving::v2::kSemanticFingerprint != bc::contract::semantic_fingerprint(r)) {
        std::printf("FAIL: the compiled generated/v2/serving_types.h has other fingerprints\n");
        ++failures;
    }
    static_assert(bc::serving::v2::kContractVersion == 2);
    if (failures) return 1;
    std::printf("ok %s\n", bc::contract::fingerprint(r).c_str());
    return 0;
}
