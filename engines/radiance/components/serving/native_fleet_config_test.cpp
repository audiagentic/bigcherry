#include "native_fleet_config.h"
#include "sha256.h"
#include "v2/document_bytes.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

using bc::serving::NativeFleetConfig;
namespace fs = std::filesystem;

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void refused(const std::string& yaml, const std::string& digest, const std::string& diagnostic) {
    try {
        (void)NativeFleetConfig::replay(yaml, digest);
    } catch (const std::invalid_argument& error) {
        check(std::string(error.what()).find(diagnostic) != std::string::npos, error.what());
        return;
    }
    throw std::runtime_error("expected refusal: " + diagnostic);
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
    fs::remove(directory);
    const auto saved = NativeFleetConfig::replay(first.yaml_utf8(), first.sha256());
    check(saved.yaml_utf8() == original && saved.fleet().models.at(0).cfg.given == model.cfg.given,
          "replay lost the recorded native bytes or provenance");

    refused(original + "\n", expected, "SHA-256 mismatch");
    refused(original, "", "SHA-256 mismatch");
    const auto invalid = [](const std::string& bytes, const char* message) {
        refused(bytes, bc::serving::sha256_hex(bytes), message);
    };
    invalid(original + "    port: 8123\n", "Radiance refused");
    invalid(original + "  second: {model: other.rad}\n", "exactly one model");
    invalid("x-server: &auth {api-key: fixture-key}\nserver: {<<: *auth}\n" + original, "api-key-env");
    invalid(original + std::string("#\xff", 2), "UTF-8");
    invalid(original + std::string(1, '\0'), "embedded NUL");
    refused(std::string(bc::serving::v2::kMaxDocumentBytes + 1, ' '), expected, "byte limit");

    check(::setenv("BC_TEST_FLEET_API_KEY", "local-fixture-value", 1) == 0, "cannot set fixture environment");
    const std::string referenced = "server: {api-key-env: BC_TEST_FLEET_API_KEY}\n" + original;
    const auto env = NativeFleetConfig::replay(referenced, bc::serving::sha256_hex(referenced));
    check(env.yaml_utf8().find("local-fixture-value") == std::string::npos &&
              env.fleet().models.front().cfg.api_key_env == "BC_TEST_FLEET_API_KEY",
          "snapshot lost the native credential reference");
}
