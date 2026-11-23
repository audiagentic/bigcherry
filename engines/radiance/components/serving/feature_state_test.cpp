// MRC03: independent, card-free contract tests for native v1 feature-state publication.
#include "feature_state.h"

#include <cassert>
#include <string>

using Json = nlohmann::json;
namespace feature = bc::serving::feature_state;

namespace {

Json manifest(bool active = false, std::string profile = "production") {
    return {
        {"spec",
         {{"intent", {{"service", {{"profile", profile}}}}},
          {"resources", {{"host", {{"kv_mib", 12288}}}}},
          {"processes",
           {{"helpers", Json::array({{{"role", "helper0"}}})},
            {"engine",
             {{"execution", {{"num_speculative_tokens", 3}, {"no_prefix_cache", "on"}}},
              {"vision", {{"vision_tower_placement", "cpu"}, {"mm_max_patches", 128}}}}},
            {"helper_common", {{"combine", {{"enabled", true}}}}}}},
          {"handoff",
           {{"placement",
             {{"live_policy_state",
               {{"requested", "on"}, {"configured", active}, {"reason_code", active ? "configured" : "key_absent"},
                {"reason_detail", active ? "live policy ready" : "no matching calibration record"}}}}}}}}},
        {"provenance",
         {{"/spec/handoff/placement",
           {{"source", "live policy disabled: no calibration for key abc"}}}}}};
}

Json find(const Json& entries, const std::string& name) {
    for (const auto& e : entries)
        if (e.at("feature_id") == name)
            return e;
    assert(false && "missing feature");
    return {};
}

void policy_reason_and_warning() {
    const auto snapshot = feature::from_manifest(manifest(), "2026-11-19T12:34:56Z");
    assert(snapshot.entries.is_array() && snapshot.entries.size() == 7);
    const Json live = find(snapshot.entries, "live_policy");
    assert(live.at("state") == "inactive");
    assert(live.at("reason_code") == "key_absent");
    assert(live.at("reason_detail").get<std::string>().find("no calibration for key abc") != std::string::npos);
    assert(live.at("reason_detail").get<std::string>().find("requested=on; configured=false") != std::string::npos);
    assert(live.at("engine") == "radiance");
    assert(live.at("observed_at") == "2026-11-19T12:34:56Z");
    assert(snapshot.warnings.size() == 1);
    assert(snapshot.warnings.front().find("WARNING") != std::string::npos);
    assert(snapshot.warnings.front().find("live_policy") != std::string::npos);
    assert(snapshot.warnings.front().find("key_absent") != std::string::npos);
    assert(find(snapshot.entries, "mtp").at("state") == "active");
    assert(find(snapshot.entries, "prefix_cache").at("state") == "inactive");
    assert(find(snapshot.entries, "kv_host_tier").at("state") == "active");
    assert(find(snapshot.entries, "helper_combine").at("state") == "active");
    assert(find(snapshot.entries, "vision_tower_placement").at("reason_detail") == "placement=cpu");
    assert(find(snapshot.entries, "mm_max_patches").at("reason_detail") == "max_patches=128");
}

void active_and_lab_do_not_warn() {
    Json enabled = manifest(true);
    enabled["spec"]["handoff"]["placement"]["live_policy"] = {{"policy_version", 1}};
    const auto good = feature::from_manifest(enabled, "2026-11-19T12:34:56Z");
    assert(find(good.entries, "live_policy").at("state") == "active");
    assert(good.warnings.empty());
    const auto lab = feature::from_manifest(manifest(false, "lab"), "2026-11-19T12:34:56Z");
    assert(lab.warnings.empty());
}

void off_and_no_helpers() {
    Json off = manifest(false);
    off["spec"]["handoff"]["placement"]["live_policy_state"]["requested"] = "off";
    const auto disabled = feature::from_manifest(off, "2026-11-19T12:34:56Z");
    assert(find(disabled.entries, "live_policy").at("state") == "inactive");
    assert(disabled.warnings.size() == 1);

    Json dense = manifest(false);
    dense["spec"]["processes"]["helpers"] = Json::array();
    dense["spec"].erase("handoff");
    dense["spec"]["processes"].erase("helper_common");
    dense["spec"]["processes"]["engine"]["execution"]["num_speculative_tokens"] = 0;
    dense["spec"]["processes"]["engine"]["execution"].erase("no_prefix_cache");
    dense["spec"]["processes"]["engine"].erase("vision");
    dense["spec"]["resources"]["host"]["kv_mib"] = 0;
    const auto solo = feature::from_manifest(dense, "2026-11-19T12:34:56Z");
    assert(find(solo.entries, "live_policy").at("state") == "inactive");
    assert(find(solo.entries, "live_policy").at("reason_code") == "no_helpers");
    assert(find(solo.entries, "mtp").at("state") == "inactive");
    assert(find(solo.entries, "prefix_cache").at("state") == "active");
    assert(find(solo.entries, "kv_host_tier").at("state") == "inactive");
    assert(find(solo.entries, "helper_combine").at("state") == "inactive");
    assert(find(solo.entries, "vision_tower_placement").at("state") == "inactive");
    assert(solo.warnings.empty());
}

void not_applicable_and_newline_warning() {
    Json not_applicable = manifest(false);
    not_applicable["spec"]["handoff"]["placement"]["live_policy_state"]["requested"] = "not_applicable";
    const auto na = feature::from_manifest(not_applicable, "2026-11-19T12:34:56Z");
    assert(find(na.entries, "live_policy").at("state") == "inactive");
    assert(na.warnings.empty());

    // RHP16: calibrating is a started feature, not a production warning; it carries the live_calibration block.
    Json calibrating = manifest(false);
    calibrating["spec"]["handoff"]["placement"]["live_policy_state"]["requested"] = "calibrating";
    calibrating["spec"]["handoff"]["placement"]["live_calibration"] = {
        {"effect_windows", 2}, {"false_revert_rate_ppm", 10000}, {"max_heat_age_ns", 2000000000}};
    calibrating["spec"]["intent"]["service"]["profile"] = "production";
    const auto cal = feature::from_manifest(calibrating, "2026-11-19T12:34:56Z");
    assert(find(cal.entries, "live_policy").at("reason_detail").get<std::string>().find("requested=calibrating") == 0);
    assert(cal.warnings.empty());
    Json block_without_state = manifest(false);
    block_without_state["spec"]["handoff"]["placement"]["live_calibration"] = calibrating["spec"]["handoff"]["placement"]["live_calibration"];
    bool inconsistent = false;
    try { (void)feature::from_manifest(block_without_state, "2026-11-19T12:34:56Z"); } catch (const std::exception&) { inconsistent = true; }
    assert(inconsistent);

    Json unsafe = manifest(false);
    unsafe["spec"]["handoff"]["placement"]["live_policy_state"]["reason_detail"] = "first\nsecond";
    const auto snapshot = feature::from_manifest(unsafe, "2026-11-19T12:34:56Z");
    assert(snapshot.warnings.size() == 1);
    assert(snapshot.warnings.front().find('\n') == std::string::npos);
}

void malformed_manifest_is_refused() {
    Json broken = manifest(false);
    broken["spec"]["handoff"]["placement"].erase("live_policy_state");
    bool refused = false;
    try {
        (void)feature::from_manifest(broken, "2026-11-19T12:34:56Z");
    } catch (const std::exception&) {
        refused = true;
    }
    assert(refused);
}
} // namespace

int main() {
    policy_reason_and_warning();
    active_and_lab_do_not_warn();
    off_and_no_helpers();
    malformed_manifest_is_refused();
    not_applicable_and_newline_warning();
}
