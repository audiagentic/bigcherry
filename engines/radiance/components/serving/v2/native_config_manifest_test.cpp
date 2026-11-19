#include "manifest.h"
#include "native_config_snapshot.h"
#include "s10_resolver.h"
#include "serving_codec.h"

#include "jcs.h"
#include "sha256.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace v3 = bc::serving::v2;
using Json = nlohmann::json;
using Code = v3::ConfigErrorCode;

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read(const char* path) {
    std::ifstream in(path, std::ios::binary);
    check(in.good(), "cannot open test input");
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

void rehash(Json& doc) {
    std::string digest, why;
    if (!bc::serving::manifest_content_digest(doc, &digest, &why))
        throw std::runtime_error(why);
    doc.at("metadata")["content_digest"] = digest;
}

void refused(Json doc, Code code, const char* message) {
    rehash(doc);
    const auto result = v3::read_manifest(doc.dump(), "native manifest fixture", v3::PlanUse::serving);
    if (result || result.error().code != code) {
        throw std::runtime_error(result ? message :
            std::string(message) + ": " + v3::render_human("fixture", result.error()));
    }
}

}  // namespace

int main(int argc, char** argv) {
    check(argc == 3, "expected the measured fixture and successor fingerprints");
    const std::string expected = "schema " + std::string(v3::kFingerprint) +
                                 "\nsemantics " + std::string(v3::kSemanticFingerprint) + "\n";
    check(read(argv[2]) == expected, "generated successor fingerprint differs from the recorded identity");
    const std::string old_bytes = read(argv[1]);
    const auto old = v3::read_manifest(old_bytes, "retired wire fixture", v3::PlanUse::serving);
    check(!old && old.error().code == Code::unsupported_version, "successor reader accepted v2");

    const std::string yaml = "# café\r\ncards: [0]\r\nmodels:\r\n  tiny: {model: tiny.rad}\r\n";
    const std::string root = "/fixture";
    const std::string yaml_digest = bc::serving::sha256_hex(yaml);
    Json doc = Json::parse(old_bytes);
    doc["apiVersion"] = v3::kApiVersion;
    doc.at("metadata")["contract_version"] = v3::kContractVersion;
    doc.at("spec")["native_config"] = {{"yaml_utf8", yaml}, {"sha256", yaml_digest}, {"resolution_root", root}};
    doc.at("provenance")["/spec/native_config"] = {
        {"origin", "detected"}, {"source", "fixture YAML"}, {"class", "fact"},
        {"lifecycle", "startup_only"}, {"confidence", "exact"}, {"inputs", Json::array()}};

    v3::Manifest measured;
    if (const auto error = v3::decode(doc, "", &measured))
        throw std::runtime_error(v3::render_human("fixture", *error));
    measured.spec.plan.engine_memory->cache_key = v3::plan_cache_key(measured);
    const auto written = v3::write_s10(measured);
    if (!written)
        throw std::runtime_error(v3::render_human("writer", written.error()));
    const auto accepted = v3::read_manifest(written.value(), "written manifest", v3::PlanUse::serving);
    if (!accepted)
        throw std::runtime_error(v3::render_human("reader", accepted.error()));
    const auto engine = v3::engine_view(accepted.value());
    check(engine && engine.value().native_config && engine.value().native_config->yaml_utf8 == yaml &&
              engine.value().native_config->identity.sha256 == yaml_digest &&
              engine.value().native_config->identity.resolution_root == root,
          "the engine view lost exact native configuration");
    const auto helper = v3::helper_view(accepted.value(), accepted.value().spec.processes.helpers.at(0).role);
    check(helper && helper.value().native_config && helper.value().native_config->sha256 == yaml_digest &&
              helper.value().native_config->resolution_root == root,
          "the helper view lost the native configuration identity");
    const auto measuring = v3::read_manifest(written.value(), "measuring start", v3::PlanUse::measuring);
    check(!measuring && measuring.error().code == Code::plan_mismatch, "measuring accepted a serving manifest");

    const Json original = Json::parse(written.value());
    Json bad = original;
    bad.at("spec").erase("native_config");
    refused(bad, Code::missing_field, "the successor accepted missing native configuration");
    bad = original;
    bad.at("spec").at("native_config")["surplus"] = true;
    refused(bad, Code::unknown_field, "the successor accepted an extra native field");
    bad = original;
    bad.at("spec").at("native_config")["yaml_utf8"] = yaml + "\n";
    refused(bad, Code::invalid_value, "the successor accepted a stale YAML digest after rehashing the envelope");
    bad.at("spec").at("native_config")["sha256"] = bc::serving::sha256_hex(yaml + "\n");
    refused(bad, Code::plan_mismatch, "the successor accepted a plan measured for different YAML");
    bad = original;
    bad.at("spec").at("native_config")["resolution_root"] = "/different-root";
    refused(bad, Code::plan_mismatch, "the successor accepted a plan measured under a different root");
    bad.at("spec").at("native_config")["resolution_root"] = "relative";
    refused(bad, Code::invalid_value, "the successor accepted a relative execution root");

    measured.spec.native_config.yaml_utf8 += '\n';
    const auto stale_write = v3::write_s10(measured);
    check(!stale_write && stale_write.error().code == Code::invalid_value, "the writer accepted a stale YAML digest");
    measured.api_version = "bigcherry/v2";
    measured.metadata.contract_version = 2;
    const auto old_write = v3::write_s10(measured);
    check(!old_write && old_write.error().code == Code::unsupported_version, "the successor writer accepted v2");
}
