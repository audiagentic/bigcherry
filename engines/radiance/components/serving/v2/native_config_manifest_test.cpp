#include "manifest.h"
#include "native_config_snapshot.h"
#include "s10_resolver.h"
#include "serving_codec.h"

#include "jcs.h"
#include "sha256.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <filesystem>
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
    const std::string old_bytes = read(argv[1]);   // the committed example is an edition 3 manifest
    {
        Json retired = Json::parse(old_bytes);   // the same document under the retired edition is refused
        retired["apiVersion"] = "bigcherry/v2";
        retired.at("metadata")["contract_version"] = 2;
        rehash(retired);
        const auto old = v3::read_manifest(retired.dump(), "retired wire fixture", v3::PlanUse::serving);
        check(!old && old.error().code == Code::unsupported_version, "reader accepted the retired v2 edition");
    }

    const std::string yaml = "# café\r\ncards: [0]\r\nmodels:\r\n  tiny: {model: tiny.rad}\r\n";
    const auto execution_root = std::filesystem::canonical(std::filesystem::current_path());
    const auto fixture_id = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto fixture_root = std::filesystem::temp_directory_path() / ("bc-native-config-root-" + fixture_id);
    check(std::filesystem::create_directory(fixture_root), "cannot create distinct execution-root fixture");
    const auto temporary_yaml = std::filesystem::temp_directory_path() / ("bc-native-config-" + fixture_id + ".yaml");
    {
        std::ofstream out(temporary_yaml, std::ios::binary);
        out << yaml;
    }
    const auto snapshot = bc::serving::NativeConfigSnapshot::read(temporary_yaml, execution_root);
    check(snapshot.yaml_utf8() == yaml &&
              snapshot.manifest_field().at("resolution_root") == execution_root.string(),
          "native snapshot did not preserve the caller-selected execution root");
    snapshot.require_execution_root();
    bool relative_root_refused = false;
    try { (void)bc::serving::NativeConfigSnapshot::read(temporary_yaml, "relative-root"); }
    catch (const std::invalid_argument&) { relative_root_refused = true; }
    check(relative_root_refused, "native snapshot accepted a relative execution root");
    bool file_root_refused = false;
    try { (void)bc::serving::NativeConfigSnapshot::read(temporary_yaml, temporary_yaml); }
    catch (const std::invalid_argument&) { file_root_refused = true; }
    check(file_root_refused, "native snapshot accepted a file as the execution root");
    bool nul_root_refused = false;
    try {
        const std::string bad_root = execution_root.string() + std::string("\0suffix", 7);
        (void)bc::serving::NativeConfigSnapshot::read(temporary_yaml, bad_root);
    } catch (const std::invalid_argument& error) {
        nul_root_refused = std::string(error.what()).find("absolute UTF-8 path") != std::string::npos;
    }
    check(nul_root_refused, "native snapshot accepted an execution root containing NUL");
    bool invalid_utf8_root_refused = false;
    try {
        std::string bad_root = execution_root.string();
        bad_root.push_back(static_cast<char>(0xff));
        (void)bc::serving::NativeConfigSnapshot::read(temporary_yaml, bad_root);
    } catch (const std::invalid_argument& error) {
        invalid_utf8_root_refused = std::string(error.what()).find("absolute UTF-8 path") != std::string::npos;
    }
    check(invalid_utf8_root_refused, "native snapshot accepted an execution root with invalid UTF-8");
    const auto other_root = fixture_root;
    const auto mismatched_root = bc::serving::NativeConfigSnapshot::read(temporary_yaml, other_root);
    bool root_refused = false;
    try { mismatched_root.require_execution_root(); }
    catch (const std::invalid_argument&) { root_refused = true; }
    check(root_refused, "native snapshot accepted execution under a different root");
    std::filesystem::remove(temporary_yaml);
    std::filesystem::remove(fixture_root);
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
    v3::Manifest unmeasured = measured;
    unmeasured.spec.plan.state = v3::PlanState::kUnmeasured;
    v3::S10Inputs measuring_inputs;
    measuring_inputs.profile.name = unmeasured.spec.intent.power.profile;
    const auto resolved_unmeasured = v3::resolve_s10(unmeasured, measuring_inputs);
    check(resolved_unmeasured.ok(), "S10 resolver rejected the selected generated v3 edition");
    {   // mixed-architecture TP: ranks of one role on two card types resolve per architecture (they were refused)
        v3::Manifest mixed = unmeasured;
        mixed.spec.intent.power.profile = "mixed-test";
        auto& gpus = mixed.spec.facts.gpus;
        const std::string first_bdf = mixed.spec.processes.engine.device.gpus.front();
        v3::GpuFact other = *std::find_if(gpus.begin(), gpus.end(), [&](const v3::GpuFact& g) { return g.bdf == first_bdf; });
        other.bdf = "0000:ee:00.0";
        other.arch = "gfx1201";
        other.hip_ordinal = 90;
        for (auto& g : gpus) if (g.bdf == first_bdf) g.arch = "gfx1100";
        gpus.push_back(other);
        mixed.spec.processes.engine.device.gpus.push_back(other.bdf);
        mixed.spec.processes.engine.scheduler.tp = 2;
        v3::S10Inputs inputs;
        inputs.profile.name = "mixed-test";
        v3::S10PowerRule xtx;
        xtx.role = "engine"; xtx.arch = "gfx1100"; xtx.evidence = "fixture";
        xtx.settings.vddgfx_offset_mv = -125; xtx.settings.sclk_max_mhz = 2700;
        v3::S10PowerRule r9700;
        r9700.role = "engine"; r9700.arch = "gfx1201"; r9700.evidence = "fixture";
        r9700.settings.power_cap_w = 200;
        inputs.profile.rules = {xtx, r9700};
        const auto resolved = v3::resolve_s10(mixed, inputs);
        if (!resolved) std::cerr << v3::render_human("mixed tp", resolved.error());
        check(resolved.ok(), "S10 refused mixed-architecture TP ranks with architecture-specific settings");
        // a stock plan keeps the digest it had: its roles are on one architecture or empty
        v3::S10Inputs stock;
        stock.profile.name = unmeasured.spec.intent.power.profile;
        check(v3::resolve_s10(unmeasured, stock).ok(), "stock S10 resolution changed");

        // PWR08 S3: apply needs the capability probe and a qualifying receipt (or the lab-only qualifying apply)
        const auto control = [](int64_t lo, int64_t hi) {
            v3::S10ControlProbe c;
            c.minimum = lo; c.maximum = hi; c.supported = true; c.original_readable = true;
            return c;
        };
        v3::S10Inputs apply = inputs;
        apply.mode = v3::PowerMode::kApply;
        apply.power.probe = [&](const std::string& bdf) {
            v3::S10DeviceProbe probe;
            probe.bdf = bdf; probe.driver = "amdgpu";
            probe.controls["vddgfx_offset_mv"] = control(-500, 0);
            probe.controls["sclk_max_mhz"] = control(500, 5000);
            probe.controls["power_cap_w"] = control(1000000, 400000000);
            return probe;
        };
        const auto refusal_of = [&](const v3::S10Inputs& in) {
            const auto r = v3::resolve_s10(mixed, in);
            return r.ok() ? std::string() : v3::render_human("apply", r.error());
        };
        check(refusal_of(apply).find("receipt") != std::string::npos, "an unqualified apply was not refused for its receipt");
        apply.power.qualified_by = [](const std::string&, const v3::PowerSettings&) { return std::optional<std::string>("run-1"); };
        const auto applied = v3::resolve_s10(mixed, apply);
        check(applied.ok() && applied.value().spec.plan.power.mode == v3::PowerMode::kApply, "a qualified apply was refused");
        v3::S10Inputs qualifying = apply;
        qualifying.power.qualified_by = nullptr;
        qualifying.power.qualify = true;
        check(v3::resolve_s10(mixed, qualifying).ok(), "the qualifying apply was refused without a receipt");
        v3::S10Inputs out_of_range = qualifying;
        out_of_range.power.probe = [&](const std::string& bdf) {
            auto probe = apply.power.probe(bdf);
            probe.controls["power_cap_w"] = control(1000000, 100000000);   // 200 W is above this card's cap range
            return probe;
        };
        check(refusal_of(out_of_range).find("outside") != std::string::npos, "the qualifying apply ignored the probe range");
        v3::S10Inputs unpaired = qualifying;
        unpaired.profile.rules[0].settings.sclk_max_mhz.reset();
        check(refusal_of(unpaired).find("paired SCLK") != std::string::npos, "the qualifying apply ignored the gfx1100 pairing rule");
        // a site override replaces one card's setting, by BDF only; a BDF the host does not have is refused by the writer
        v3::S10Inputs site = apply;
        site.power.site_overrides[first_bdf].sclk_max_mhz = 2600;
        const auto with_site = v3::resolve_s10(mixed, site);
        check(with_site.ok(), "a site override was refused");
        const auto& device_settings = with_site.value().spec.plan.power.devices;
        const auto first = std::find_if(device_settings.begin(), device_settings.end(), [&](const auto& d) { return d.bdf == first_bdf; });
        check(first != device_settings.end() && first->settings.sclk_max_mhz == 2600 && first->settings.vddgfx_offset_mv == -125,
              "the site override did not replace only the named setting of that BDF");
    }
    v3::Manifest wrong_edition = unmeasured;
    wrong_edition.api_version = "bigcherry/v2";
    wrong_edition.metadata.contract_version = 2;
    const auto refused_edition = v3::resolve_s10(wrong_edition, measuring_inputs);
    check(!refused_edition && refused_edition.error().code == Code::unsupported_version,
          "S10 resolver accepted a manifest outside the selected generated edition");
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
