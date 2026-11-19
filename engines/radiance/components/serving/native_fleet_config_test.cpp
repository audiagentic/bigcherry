#include "native_fleet_config.h"
#include "jcs.h"
#include "sha256.h"
#include "v2/document_bytes.h"
#include "v2/manifest.h"
#include "v2/strict_json.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

using bc::serving::NativeFleetConfig;
namespace fs = std::filesystem;
namespace v2 = bc::serving::v2;
using Json = nlohmann::json;

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void refused(const Json& field, const std::string& diagnostic) {
    try {
        (void)NativeFleetConfig::replay(field);
    } catch (const std::invalid_argument& error) {
        check(std::string(error.what()).find(diagnostic) != std::string::npos, error.what());
        return;
    }
    throw std::runtime_error("expected refusal: " + diagnostic);
}

void rehash(Json& doc) {
    std::string digest, why;
    if (!bc::serving::manifest_content_digest(doc, &digest, &why)) throw std::runtime_error(why);
    doc.at("metadata")["content_digest"] = digest;
}

void write(const fs::path& file, const std::string& bytes) {
    std::ofstream out(file, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << bytes;
    out.close();
}

}  // namespace

int main() {
    const std::string original = "# café\r\ncards: [0]\r\nmodels:\r\n  tiny:\r\n"
                                 "    model: tiny.rad\r\n    mixed-step-target-ms: 0\r\n";
    const std::string expected = "b6c5c7937797350f664b9d8a8572156499ae6ce648295786772a61da690af641";
    std::string directory = (fs::temp_directory_path() / "bc-native-fleet-XXXXXX").string();
    check(::mkdtemp(directory.data()) != nullptr, "cannot create test directory");
    const fs::path path = fs::path(directory) / "fleet.yaml";
    write(path, original);
    const auto first = NativeFleetConfig::read(path);
    check(first.sha256() == expected && first.yaml_utf8() == original, "original bytes or digest changed");
    const auto& model = first.fleet().models.at(0);
    check(model.cfg.mixed_step_target_ms == 0 &&
              model.cfg.given.at("--mixed-step-target-ms") == "models.tiny.mixed-step-target-ms",
          "native option or its YAML provenance changed");

    write(fs::path(directory) / "next", "# replaced\n" + original);
    fs::rename(fs::path(directory) / "next", path);
    check(NativeFleetConfig::read(path).sha256() != expected, "formatting edit reused the old identity");
    fs::remove(path);
    const Json field = first.manifest_field();
    const auto execution_root = fs::current_path();
    fs::current_path(directory);
    refused(field, "resolution_root differs");
    fs::current_path(execution_root);
    fs::remove(directory);

    std::ifstream example(fs::path(BC_CONTRACT_DIR) / "example.flashnext-4gpu.json", std::ios::binary);
    const std::string measured_bytes((std::istreambuf_iterator<char>(example)), {});
    check(v2::read_manifest(measured_bytes, "measured example", v2::PlanUse::serving).ok(),
          "qualified measured example failed v2 admission");
    Json measured = Json::parse(measured_bytes);
    measured.at("spec")["native_config"] = field;
    rehash(measured);
    auto legacy = v2::read_manifest(measured.dump(), "current wire", v2::PlanUse::serving);
    check(!legacy && legacy.error().code == v2::ConfigErrorCode::unknown_field,
          "current wire ignored a native_config field");
    // Exercise successor field storage in the measured envelope. Full v3
    // contract admission belongs to the coordinated reader/producer migration.
    measured["apiVersion"] = "bigcherry/v3";
    measured.at("metadata")["contract_version"] = 3;
    rehash(measured);
    const std::string recorded = measured.dump();
    auto successor = v2::read_manifest(recorded, "successor wire", v2::PlanUse::serving);
    check(!successor && successor.error().code == v2::ConfigErrorCode::unsupported_version,
          "current launcher accepted a successor document");
    auto strict = v2::parse_strict_json(recorded, "saved manifest");
    check(strict.ok(), "stored manifest failed strict JSON parsing");
    std::string oversize = recorded;
    oversize.resize(v2::kMaxDocumentBytes + 1, ' ');
    check(!v2::parse_strict_json(oversize, "oversize manifest"), "complete manifest exceeded the byte limit");
    const auto saved = NativeFleetConfig::replay(strict.value().at("spec").at("native_config"));
    check(saved.yaml_utf8() == original && saved.fleet().models.at(0).cfg.given == model.cfg.given,
          "replay lost the recorded native bytes or provenance");
    const Json outer_digest = measured.at("metadata").at("content_digest");
    Json& embedded = measured.at("spec").at("native_config");
    embedded["yaml_utf8"] = original + "\n";
    rehash(measured);
    check(measured.at("metadata").at("content_digest") != outer_digest, "manifest digest missed the YAML edit");
    refused(embedded, "SHA-256 mismatch");
    Json missing = field;
    missing.erase("sha256");
    refused(missing, "requires exactly");
    Json extra = field;
    extra["unadmitted"] = true;
    refused(extra, "requires exactly");
    Json malformed = field;
    malformed["resolution_root"] = 1;
    refused(malformed, "requires exactly");
    malformed = field;
    malformed["sha256"] = "";
    refused(malformed, "SHA-256 mismatch");
    const auto config_field = [&](const std::string& bytes) {
        Json changed = field;
        changed["yaml_utf8"] = bytes;
        changed["sha256"] = bc::serving::sha256_hex(bytes);
        return changed;
    };
    const auto invalid = [&](const std::string& bytes, const char* message) {
        refused(config_field(bytes), message);
    };
    invalid(original + "    port: 8123\n", "Radiance refused");
    invalid(original + "  second: {model: other.rad}\n", "exactly one model");
    invalid("x-server: &auth {api-key: fixture-key}\nserver: {<<: *auth}\n" + original, "api-key-env");
    invalid(original + std::string("#\xff", 2), "UTF-8");
    invalid(original + std::string(1, '\0'), "embedded NUL");
    malformed = field;
    malformed["yaml_utf8"] = std::string(v2::kMaxDocumentBytes + 1, ' ');
    refused(malformed, "byte limit");

    check(::setenv("BC_TEST_FLEET_API_KEY", "local-fixture-value", 1) == 0, "cannot set fixture environment");
    const std::string referenced = "server: {api-key-env: BC_TEST_FLEET_API_KEY}\n" + original;
    const auto env = NativeFleetConfig::replay(config_field(referenced));
    check(env.manifest_field().dump().find("local-fixture-value") == std::string::npos &&
              env.fleet().models.front().cfg.api_key_env == "BC_TEST_FLEET_API_KEY",
          "snapshot lost the native credential reference");
}
