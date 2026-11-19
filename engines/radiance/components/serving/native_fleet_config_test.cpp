#define _GNU_SOURCE 1
#include "native_fleet_config.h"
#include "jcs.h"
#include "sha256.h"
#include "v2/document_bytes.h"
#include "v2/manifest.h"
#include "v2/strict_json.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

using bc::serving::NativeConfigSnapshot;
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
        (void)NativeFleetConfig::replay(NativeConfigSnapshot::from_manifest(field));
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

#if defined(__linux__)
int sealed_memfd(const std::string& bytes) {
    const int fd = ::memfd_create("bc-native-config-test", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) throw std::runtime_error("cannot create sealed-config fixture");
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = ::write(fd, bytes.data() + offset, bytes.size() - offset);
        if (count <= 0) {
            ::close(fd);
            throw std::runtime_error("cannot write sealed-config fixture");
        }
        offset += static_cast<size_t>(count);
    }
    if (::fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) != 0) {
        ::close(fd);
        throw std::runtime_error("cannot seal config fixture");
    }
    return fd;
}
#endif

}  // namespace

int main() {
    const std::string original = "# café\r\ncards: [0]\r\nmodels:\r\n  tiny:\r\n"
                                 "    model: tiny.rad\r\n    mixed-step-target-ms: 17\r\n"
                                 "    prefill-interleave: 0\r\n";
    const std::string expected = "70e3e90e6f05596a0f84f078437ec8b0c2e68811bd6d9e77c87bdf4b4255fb77";
    std::string directory = (fs::temp_directory_path() / "bc-native-fleet-XXXXXX").string();
    check(::mkdtemp(directory.data()) != nullptr, "cannot create test directory");
    const fs::path path = fs::path(directory) / "fleet.yaml";
    write(path, original);
    const auto execution_root = fs::canonical(fs::current_path());
    const auto first = NativeFleetConfig::read(path, execution_root);
    check(first.snapshot().sha256() == expected && first.snapshot().yaml_utf8() == original,
          "original bytes or digest changed");
    const auto& model = first.fleet().models.at(0);
    check(model.cfg.mixed_step_target_ms == 17 && model.cfg.prefill_interleave == 0 &&
              model.cfg.prefill_interleave_set &&
              model.cfg.given.at("--mixed-step-target-ms") == "models.tiny.mixed-step-target-ms" &&
              model.cfg.given.at("--prefill-interleave") == "models.tiny.prefill-interleave",
          "native option or its YAML provenance changed");

    write(fs::path(directory) / "next", "# replaced\n" + original);
    fs::rename(fs::path(directory) / "next", path);
    check(NativeConfigSnapshot::read(path, execution_root).sha256() != expected,
          "formatting edit reused the old identity");
    fs::remove(path);
    const Json field = first.snapshot().manifest_field();
    fs::current_path(directory);
    check(NativeConfigSnapshot::from_manifest(field).manifest_field() == field,
          "decoding rewrote the recorded execution root");
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
    const auto saved = NativeFleetConfig::replay(
        NativeConfigSnapshot::from_manifest(strict.value().at("spec").at("native_config")));
    check(saved.snapshot().manifest_field() == field && saved.fleet().models.at(0).cfg.given == model.cfg.given,
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
    malformed["resolution_root"] = "relative-root";
    refused(malformed, "absolute UTF-8 path");
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
    invalid(original + "    park: {vram: statics}\n", "does not support managed fleet settings");
    invalid("x-server: &auth {api-key: fixture-key}\nserver: {<<: *auth}\n" + original, "api-key-env");
    invalid(original + std::string("#\xff", 2), "UTF-8");
    invalid(original + std::string(1, '\0'), "embedded NUL");
    malformed = field;
    malformed["yaml_utf8"] = std::string(v2::kMaxDocumentBytes + 1, ' ');
    refused(malformed, "byte limit");

    check(::setenv("BC_TEST_FLEET_API_KEY", "local-fixture-value", 1) == 0, "cannot set fixture environment");
    const std::string referenced = "server: {api-key-env: BC_TEST_FLEET_API_KEY}\n" + original;
    const auto env = NativeFleetConfig::replay(NativeConfigSnapshot::from_manifest(config_field(referenced)));
    check(env.snapshot().manifest_field().dump().find("local-fixture-value") == std::string::npos &&
              env.fleet().models.front().cfg.api_key_env == "BC_TEST_FLEET_API_KEY",
          "snapshot lost the native credential reference");

#if defined(__linux__)
    const std::string inspected_yaml = "server: {host: 127.0.0.1, port: 8123}\ncards: [2, 0]\nmodels:\n"
                                       "  tiny:\n    model: tiny.rad\n    mixed-step-target-ms: 17\n";
    const Json inspected_field = config_field(inspected_yaml);
    const int sealed_fd = sealed_memfd(inspected_field.dump());
    const std::string fd_text = std::to_string(sealed_fd);
    char command[] = "radiance";
    char inspect[] = "--bc-inspect-native-config";
    char* inspect_argv[] = {command, inspect, const_cast<char*>(fd_text.c_str()), nullptr};
    std::ostringstream output;
    std::streambuf* old_out = std::cout.rdbuf(output.rdbuf());
    const auto result = bc::serving::inspect_native_fleet_config_command(3, inspect_argv);
    std::cout.rdbuf(old_out);
    ::close(sealed_fd);
    check(result && *result == 0, "sealed native config inspection failed");
    const Json report = Json::parse(output.str());
    check(report.at("protocol") == "bigcherry.native-fleet-inspect/v1" &&
              report.at("snapshot").at("sha256") == inspected_field.at("sha256") &&
              report.at("snapshot").at("resolution_root") == execution_root.string() &&
              report.at("engine").at("listener").at("port") == 8123 &&
              report.at("engine").at("cards") == Json::array({2, 0}) &&
              report.at("engine").at("model") == "tiny.rad" &&
              report.at("engine").at("max_context") == 0 &&
              report.dump().find(inspected_yaml) == std::string::npos,
          "native inspect report changed identity, ordering, parsed values or exposed source YAML");

    const int unsealed_fd = ::memfd_create("bc-native-config-unsealed-test", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    check(unsealed_fd >= 0, "cannot create unsealed-config fixture");
    const std::string unsealed_bytes = inspected_field.dump();
    check(::write(unsealed_fd, unsealed_bytes.data(), unsealed_bytes.size()) ==
              static_cast<ssize_t>(unsealed_bytes.size()),
          "cannot write unsealed-config fixture");
    const std::string unsealed_text = std::to_string(unsealed_fd);
    char* unsealed_argv[] = {command, inspect, const_cast<char*>(unsealed_text.c_str()), nullptr};
    std::ostringstream errors;
    std::streambuf* old_err = std::cerr.rdbuf(errors.rdbuf());
    const auto unsealed_result = bc::serving::inspect_native_fleet_config_command(3, unsealed_argv);
    std::cerr.rdbuf(old_err);
    ::close(unsealed_fd);
    check(unsealed_result && *unsealed_result == 2 && errors.str().find("fully sealed memfd") != std::string::npos,
          "native inspect accepted an unsealed descriptor");

    Json wrong_root = inspected_field;
    wrong_root["resolution_root"] = "/tmp";
    const int wrong_root_fd = sealed_memfd(wrong_root.dump());
    const std::string wrong_root_text = std::to_string(wrong_root_fd);
    char* wrong_root_argv[] = {command, inspect, const_cast<char*>(wrong_root_text.c_str()), nullptr};
    errors.str("");
    old_err = std::cerr.rdbuf(errors.rdbuf());
    const auto wrong_root_result = bc::serving::inspect_native_fleet_config_command(3, wrong_root_argv);
    std::cerr.rdbuf(old_err);
    ::close(wrong_root_fd);
    check(wrong_root_result && *wrong_root_result == 2 &&
              errors.str().find("resolution_root differs") != std::string::npos,
          "native inspect accepted a changed execution root");

    Json wrong_digest = inspected_field;
    wrong_digest["yaml_utf8"] = inspected_yaml + "# changed\n";
    const int wrong_digest_fd = sealed_memfd(wrong_digest.dump());
    const std::string wrong_digest_text = std::to_string(wrong_digest_fd);
    char* wrong_digest_argv[] = {command, inspect, const_cast<char*>(wrong_digest_text.c_str()), nullptr};
    errors.str("");
    old_err = std::cerr.rdbuf(errors.rdbuf());
    const auto wrong_digest_result = bc::serving::inspect_native_fleet_config_command(3, wrong_digest_argv);
    std::cerr.rdbuf(old_err);
    ::close(wrong_digest_fd);
    check(wrong_digest_result && *wrong_digest_result == 2 &&
              errors.str().find("SHA-256 mismatch") != std::string::npos,
          "native inspect accepted a changed YAML digest");
#endif
}
