// bc_contract_identity_test: the contract has one identity everywhere it is used. The registry (contract/serving.toml), the
// committed fingerprint file, the typed model and codec every consumer links (the one the owner's writer and reader share, and
// the engine builds from the same registry) and the generated JSON Schema all describe the same fields. Nothing here is a
// recorded literal: every expectation is computed from the registry in this process.
//
//   1. both fingerprints computed from the registry equal the ones compiled into the codec (kFingerprint,
//      kSemanticFingerprint) and the committed generated/v3/contract.fingerprint, which the engine and the owner builds
//      each compare against their own generated copy;
//   2. the registry's flattened field list (every field path with its kind, provenance class, required/nullable flags and
//      Radiance binding) is exactly the compiled kFields table, in order;
//   3. every flattened field path resolves in the generated JSON Schema (through $ref, items and map values), and the
//      schema agrees with the registry on whether the field is required.
#include "registry.h"
#include "serving_codec.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef BC_CONTRACT_SOURCE_DIR
#error "BC_CONTRACT_SOURCE_DIR must name engines/radiance/components/serving"
#endif

namespace v3 = bc::serving::v2;
using Json = nlohmann::json;

namespace {
int failures = 0;

void expect(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

bool read(const std::string& path, std::string* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream s;
    s << in.rdbuf();
    *out = s.str();
    return true;
}

std::vector<std::string> split_path(const std::string& path) {
    std::vector<std::string> parts;
    if (path.empty()) return parts;   // the document root
    size_t from = path.front() == '/' ? 1 : 0;
    while (from <= path.size()) {
        const size_t to = path.find('/', from);
        parts.push_back(path.substr(from, to == std::string::npos ? std::string::npos : to - from));
        if (to == std::string::npos) break;
        from = to + 1;
    }
    return parts;
}

// Follows "$ref": "#/$defs/NAME" until the node is a schema body.
const Json* deref(const Json& schema, const Json* node) {
    while (node && node->is_object() && node->contains("$ref")) {
        const std::string ref = node->at("$ref").get<std::string>();
        const std::string prefix = "#/$defs/";
        if (ref.rfind(prefix, 0) != 0 || !schema.at("$defs").contains(ref.substr(prefix.size()))) return nullptr;
        node = &schema.at("$defs").at(ref.substr(prefix.size()));
    }
    return node;
}

// The schema node a field path names, or null; `parent` receives its containing object's schema node and `key` its name
// there, when the last step is an object member.
const Json* resolve(const Json& schema, const std::string& path, const Json** parent, std::string* key) {
    *parent = nullptr;
    key->clear();
    auto parts = split_path(path);
    const Json* node = &schema;
    size_t i = 0;
    if (!parts.empty() && parts[0] == "$defs") {
        if (parts.size() < 2 || !schema.at("$defs").contains(parts[1])) return nullptr;
        node = &schema.at("$defs").at(parts[1]);
        i = 2;
    }
    for (; i < parts.size(); ++i) {
        node = deref(schema, node);
        if (!node || !node->is_object()) return nullptr;
        const std::string& step = parts[i];
        if (step == "[]") {
            if (!node->contains("items")) return nullptr;
            node = &node->at("items");
        } else if (step == "*") {
            if (node->contains("additionalProperties") && node->at("additionalProperties").is_object())
                node = &node->at("additionalProperties");
            else if (node->contains("patternProperties") && !node->at("patternProperties").empty())
                node = &node->at("patternProperties").begin().value();
            else
                return nullptr;
        } else {
            if (!node->contains("properties") || !node->at("properties").contains(step)) return nullptr;
            *parent = node;
            *key = step;
            node = &node->at("properties").at(step);
        }
        if (step == "[]" || step == "*") { *parent = nullptr; key->clear(); }
    }
    return node;
}
}  // namespace

int main() {
    const std::string src = BC_CONTRACT_SOURCE_DIR;
    std::string text, why;
    if (!read(src + "/contract/serving.toml", &text)) {
        std::printf("FAIL: cannot read the registry\n");
        return 1;
    }
    bc::contract::Registry r;
    if (!bc::contract::load_registry(text, "contract/serving.toml", &r, &why)) {
        std::printf("FAIL: %s\n", why.c_str());
        return 1;
    }

    // 1. one identity
    const std::string schema_fp = bc::contract::fingerprint(r);
    const std::string semantic_fp = bc::contract::semantic_fingerprint(r);
    expect(schema_fp == v3::kFingerprint,
           "the codec's schema fingerprint " + std::string(v3::kFingerprint) + " is not the registry's " + schema_fp);
    expect(semantic_fp == v3::kSemanticFingerprint,
           "the codec's semantic fingerprint " + std::string(v3::kSemanticFingerprint) + " is not the registry's " + semantic_fp);
    const std::string want_file = "schema " + schema_fp + "\nsemantics " + semantic_fp + "\n";
    std::string committed;
    expect(read(src + "/generated/v3/contract.fingerprint", &committed) && committed == want_file,
           "generated/v3/contract.fingerprint is not the registry's two fingerprints");
    const auto generated = bc::contract::generate_all(r);
    expect(generated.at("contract.fingerprint") == want_file, "the generator's fingerprint file is not the registry's two fingerprints");

    // 2. the flattened registry is the compiled field table. The generator lists a node when it is a documented, non-patch,
    // non-calibration-key node (contract/registry.cpp, Emitter::fields); the test applies the same listing rule to the
    // registry it loaded and compares every column.
    std::vector<const bc::contract::Node*> flat;
    for (const auto& n : r.nodes)
        if (n.path.rfind("calibration:", 0) != 0 && !n.doc.empty() && !n.patch) flat.push_back(&n);
    expect(flat.size() == v3::kFields.size(), "the registry flattens to " + std::to_string(flat.size()) +
                                                  " fields, the compiled table has " + std::to_string(v3::kFields.size()));
    for (size_t i = 0; i < flat.size() && i < v3::kFields.size(); ++i) {
        const auto& n = *flat[i];
        const auto& f = v3::kFields[i];
        const bool same = f.path == n.path && f.kind == n.kind && f.value_class == n.cls && f.required == !n.optional &&
                          f.nullable == n.nullable && f.radiance_option == n.radiance_option &&
                          f.radiance_config == n.radiance_config && f.radiance_switch == n.radiance_switch &&
                          f.ref_def == n.ref;
        if (!same) {
            expect(false, "field " + std::to_string(i) + " (" + n.path + ") differs between the registry and the compiled table (table: " +
                              std::string(f.path) + ")");
            break;
        }
    }

    // 3. the generated schema places every flattened field, with the registry's required flag
    const Json schema = Json::parse(generated.at("serving-v3.schema.json"));
    size_t checked = 0;
    for (const auto* n : flat) {
        const Json* parent = nullptr;
        std::string key;
        const Json* at = resolve(schema, n->path, &parent, &key);
        if (!at) {
            expect(false, "the schema has no place for registry field " + n->path);
            continue;
        }
        ++checked;
        if (parent) {
            bool required = false;
            if (parent->contains("required"))
                for (const auto& k : parent->at("required")) required = required || k.get<std::string>() == key;
            expect(required == !n->optional, "the schema and the registry disagree on whether " + n->path + " is required");
        }
    }
    expect(checked > 0, "no field was checked against the schema");

    if (failures) return 1;
    std::printf("ok %zu fields, %s %s\n", flat.size(), schema_fp.c_str(), semantic_fp.c_str());
    return 0;
}
