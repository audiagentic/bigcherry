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
#include <iterator>
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

// The x-bigcherry subtree: YAML 1.2 core typing, no coercion, a duplicate key refused with its path.
void test_x_bigcherry_subtree() {
    const std::string head = "cards: [0]\nmodels:\n  tiny: {model: tiny.rad}\n";
    check(bc::serving::x_bigcherry_subtree(head).is_null(), "a file without x-bigcherry must yield null");
    const Json typed = bc::serving::x_bigcherry_subtree(head +
        "x-bigcherry:\n  schema: 1\n  on: true\n  quoted: \"true\"\n  ratio: 0.5\n  none:\n  words: off\n"
        "  list: [1, \"2\", x]\n  nest: {a: 1}\n");
    check(typed.at("schema") == 1 && typed.at("on") == true && typed.at("quoted") == "true" &&
              typed.at("ratio") == 0.5 && typed.at("none").is_null() && typed.at("words") == "off" &&
              typed.at("list") == Json::array({1, "2", "x"}) && typed.at("nest").at("a") == 1,
          "x-bigcherry scalars must follow YAML 1.2 core typing without coercion");
    try {
        (void)bc::serving::x_bigcherry_subtree(head + "x-bigcherry:\n  helper: {combine: true}\n  helper: {combine: false}\n");
    } catch (const std::invalid_argument& error) {
        check(std::string(error.what()).find("x-bigcherry.helper is given twice") != std::string::npos, error.what());
        try {
            (void)bc::serving::x_bigcherry_subtree(head + "x-bigcherry:\n  helper:\n    combine: true\n    combine: false\n");
        } catch (const std::invalid_argument& nested) {
            check(std::string(nested.what()).find("x-bigcherry.helper.combine is given twice") != std::string::npos,
                  nested.what());
            return;
        }
        throw std::runtime_error("expected a nested duplicate refusal");
    }
    throw std::runtime_error("expected a duplicate refusal");
}

int main() {
    test_x_bigcherry_subtree();
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
    // Edition 3 is the only wire: the embedded native configuration is part of it, so the recorded document is the
    // measured example with the field added, not a successor edition.
    const std::string recorded = measured.dump();
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
    invalid("memory: {host-mib: 65536}\n" + original, "fleet-only setting: memory.host-mib");
    invalid("memory: {disk-tier: local-cache}\n" + original, "fleet-only setting: memory.disk-tier");
    invalid("scheduler: {load-timeout: 10s}\n" + original, "fleet-only setting: scheduler.load-timeout");
    invalid("scheduler: {drain-timeout: 10s}\n" + original, "fleet-only setting: scheduler.drain-timeout");
    invalid("scheduler: {keepalive: 5s}\n" + original, "fleet-only setting: scheduler.keepalive");
    invalid("scheduler: {stuck-exit: 0s}\n" + original, "fleet-only setting: scheduler.stuck-exit");
    invalid("scheduler: {warm-up: false}\n" + original, "fleet-only setting: scheduler.warm-up");
    invalid(original + "    aliases: [tiny-alias]\n", "fleet-only setting: aliases");
    invalid(original + "    priority: 1\n", "fleet-only setting: priority");
    invalid(original + "    pin: true\n", "fleet-only setting: pin");
    invalid("defaults: {ttl: 30s}\n" + original, "fleet-only setting: ttl");
    invalid(original + "    flex: {kv: {max-tokens: 8192}}\n", "fleet-only setting: flex");
    invalid(original + "    park: {ram: required}\n", "fleet-only setting: park");
    const auto defaults_replay = NativeFleetConfig::replay(NativeConfigSnapshot::from_manifest(config_field(
        "memory: {gpu-headroom-mib: 512}\nscheduler: {warm-up: true}\n" +
        original + "    park: {ram: allowed}\n    pin: false\n")));
    check(defaults_replay.fleet().models.front().cfg.given.at("--gpu-headroom-mib") == "memory.gpu-headroom-mib",
          "managed admission lost ordinary engine memory settings");
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
    // The JSON-envelope form: the manifest writer's handoff only; it dies with manifest_field (RSC02/RSC03).
    const int envelope_fd = sealed_memfd(inspected_field.dump());
    const auto received = NativeConfigSnapshot::from_sealed_fd(envelope_fd);
    ::close(envelope_fd);
    check(received.manifest_field() == inspected_field,
          "shared sealed-fd receiver changed native snapshot bytes or identity");
    // The inspector's form: the authored YAML bytes themselves, the root being the process's own working directory.
    const int sealed_fd = sealed_memfd(inspected_yaml);
    const auto raw = NativeConfigSnapshot::from_raw_fd(sealed_fd);
    check(raw.yaml_utf8() == inspected_yaml && raw.sha256() == inspected_field.at("sha256") &&
              raw.resolution_root() == execution_root.string(),
          "raw-YAML receiver changed the bytes, their digest or the execution root");
    {   // The sealed descriptor is also readable by path (what a child's `--config /proc/self/fd/N` opens): same bytes.
        std::ifstream by_path("/proc/self/fd/" + std::to_string(sealed_fd), std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(by_path)), std::istreambuf_iterator<char>());
        check(text == inspected_yaml, "the sealed YAML descriptor is not readable by its /proc/self/fd path");
    }
    const std::string fd_text = std::to_string(sealed_fd);
    char command[] = "radiance";
    char inspect[] = "--bc-inspect-native-config";
    char* inspect_argv[] = {command, inspect, const_cast<char*>(fd_text.c_str()), nullptr};
    std::ostringstream output;
    std::streambuf* old_out = std::cout.rdbuf(output.rdbuf());
    // The first step of a start is a device-free parse: it must work with every GPU hidden, as bc-serve runs it.
    check(::setenv("ROCR_VISIBLE_DEVICES", "-1", 1) == 0 && ::setenv("HIP_VISIBLE_DEVICES", "-1", 1) == 0,
          "cannot hide the devices for the inspector");
    const auto result = bc::serving::inspect_native_fleet_config_command(3, inspect_argv);
    std::cout.rdbuf(old_out);
    ::close(sealed_fd);
    check(result && *result == 0, "sealed native config inspection failed");
    const Json report = Json::parse(output.str());
    check(report.at("protocol") == "bigcherry.native-fleet-inspect/v2" &&
              report.at("snapshot").at("sha256") == inspected_field.at("sha256") &&
              report.at("snapshot").at("resolution_root") == execution_root.string() &&
              report.at("engine").at("listener").at("port") == 8123 &&
              report.at("engine").at("cards") == Json::array({2, 0}) &&
              report.at("engine").at("model") == "tiny.rad" &&
              report.at("engine").at("x_bigcherry").is_null() && report.at("engine").at("tp_wire").is_null() &&
              report.at("engine").at("tuning").at("tune_pin").is_null() &&
              report.at("engine").at("max_context") == 0 &&
              report.at("engine").at("settings") == Json({
                  {"max_batched_tokens", 2048}, {"checkpoint_interval", 2048},
                  {"placement", "auto"}, {"ngram_placement", "auto"}, {"kv_dtype", "bf16"},
                  {"speculative_tokens", -1}, {"rope_scaling", ""}, {"prefix_resume", "exact"},
                  {"prefix_cache", true}, {"prefill_interleave", nullptr}}) &&
              report.dump().find(inspected_yaml) == std::string::npos,
          "native inspect report changed identity, ordering, parsed values or exposed source YAML");

    const int unsealed_fd = ::memfd_create("bc-native-config-unsealed-test", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    check(unsealed_fd >= 0, "cannot create unsealed-config fixture");
    const std::string unsealed_bytes = inspected_yaml;
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

    const int bad_text_fd = sealed_memfd(inspected_yaml + std::string("#\xff", 2));
    const std::string bad_text = std::to_string(bad_text_fd);
    char* bad_text_argv[] = {command, inspect, const_cast<char*>(bad_text.c_str()), nullptr};
    errors.str("");
    old_err = std::cerr.rdbuf(errors.rdbuf());
    const auto bad_text_result = bc::serving::inspect_native_fleet_config_command(3, bad_text_argv);
    std::cerr.rdbuf(old_err);
    ::close(bad_text_fd);
    check(bad_text_result && *bad_text_result == 2 && errors.str().find("UTF-8") != std::string::npos,
          "native inspect accepted a YAML that is not UTF-8 text");
#endif
}
