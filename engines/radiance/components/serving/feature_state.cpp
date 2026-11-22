#include "feature_state.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace bc::serving::feature_state {
namespace {

using Json = nlohmann::json;

void append(Snapshot& out, const std::string& name, const std::string& state,
            const std::string& reason, const std::string& detail, const std::string& at) {
    out.entries.push_back({
        {"feature_id", name}, {"state", state}, {"reason_code", reason},
        {"reason_detail", detail}, {"engine", "radiance"}, {"observed_at", at},
    });
}

std::string string_or(const Json& object, const char* key, const std::string& fallback) {
    const auto it = object.find(key);
    return it != object.end() && it->is_string() ? it->get<std::string>() : fallback;
}

const Json& optional_object(const Json& object, const char* key) {
    static const Json empty = Json::object();
    const auto it = object.find(key);
    return it != object.end() && it->is_object() ? *it : empty;
}

} // namespace

Snapshot from_manifest(const nlohmann::json& manifest, const std::string& observed_at) {
    // These required fields are read from the *admitted* v2 document, not from
    // unsealed operator YAML, historical session.json or a profile default.
    const Json& spec = manifest.at("spec");
    const Json& service = spec.at("intent").at("service");
    const std::string profile = service.at("profile").get<std::string>();
    const Json& processes = spec.at("processes");
    const Json& helpers = processes.at("helpers");
    if (!helpers.is_array())
        throw std::runtime_error("feature states: helpers must be an array");
    const bool has_helpers = !helpers.empty();
    const Json& engine = processes.at("engine");
    const Json& execution = engine.at("execution");
    const Json& host = spec.at("resources").at("host");
    Snapshot out;

    if (has_helpers) {
        const Json& placement = spec.at("handoff").at("placement");
        const Json& policy = placement.at("live_policy_state");
        const bool configured = policy.at("configured").get<bool>();
        const std::string requested = policy.at("requested").get<std::string>();
        if (configured != placement.contains("live_policy") ||
            (requested == "calibrating") != placement.contains("live_calibration"))
            throw std::runtime_error("feature states: inconsistent live_policy_state and live_policy presence");
        if (requested != "on" && requested != "off" && requested != "not_applicable" && requested != "calibrating")
            throw std::runtime_error("feature states: unknown live_policy_state.requested");
        const std::string state = configured ? "active" : "inactive";
        const std::string reason = policy.at("reason_code").get<std::string>();
        std::string detail = "requested=" + requested + "; configured=" + (configured ? "true" : "false") +
                             "; " + policy.at("reason_detail").get<std::string>();
        // The derived-policy skip reason is a second distinct source of truth:
        // it explains *why* the calibration was not imported before S10.
        if (!configured && manifest.contains("provenance") && manifest.at("provenance").is_object()) {
            const auto& provenance = manifest.at("provenance");
            const auto it = provenance.find("/spec/handoff/placement");
            if (it != provenance.end() && it->is_object()) {
                const std::string source = string_or(*it, "source", "");
                constexpr const char* marker = "live policy disabled: ";
                if (source.rfind(marker, 0) == 0)
                    detail += "; provenance_skip=" + source.substr(std::char_traits<char>::length(marker));
            }
        }
        append(out, "live_policy", state, reason, detail, observed_at);
        // Live placement is the required production feature for helper-backed
        // layouts. Default-off optional knobs are reported but never spam the log.
        if (profile == "production" && requested != "not_applicable" &&
            requested != "calibrating" && state != "active") {
            std::string message = "WARNING production feature live_policy " + state + ": " + reason + ": " + detail;
            // Owner logs have one warning per feature; never let a source
            // detail inject extra lines into a machine-readable incident log.
            std::replace_if(message.begin(), message.end(),
                            [](char c) { return c == '\n' || c == '\r'; }, ' ');
            out.warnings.push_back(std::move(message));
        }
    } else {
        append(out, "live_policy", "inactive", "no_helpers", "engine-only layout", observed_at);
    }

    const auto depth = execution.at("num_speculative_tokens").get<int64_t>();
    if (depth < 0)
        throw std::runtime_error("feature states: negative MTP depth");
    append(out, "mtp", depth > 0 ? "active" : "inactive",
           depth > 0 ? "depth_configured" : "depth_zero", "depth=" + std::to_string(depth), observed_at);

    const std::string cache_off = string_or(execution, "no_prefix_cache", "");
    if (!cache_off.empty() && cache_off != "on")
        throw std::runtime_error("feature states: unknown no_prefix_cache setting");
    append(out, "prefix_cache", cache_off == "on" ? "inactive" : "active",
           cache_off == "on" ? "disabled_by_config" : "enabled_by_config",
           cache_off == "on" ? "no_prefix_cache=on" : "no_prefix_cache absent", observed_at);

    const auto kv_mib = host.at("kv_mib").get<int64_t>();
    if (kv_mib < 0)
        throw std::runtime_error("feature states: negative host KV budget");
    append(out, "kv_host_tier", kv_mib > 0 ? "active" : "inactive",
           kv_mib > 0 ? "budget_configured" : "budget_zero",
           "kv_mib=" + std::to_string(kv_mib), observed_at);

    bool combined = false;
    if (has_helpers)
        combined = processes.at("helper_common").at("combine").at("enabled").get<bool>();
    append(out, "helper_combine", combined ? "active" : "inactive",
           !has_helpers ? "no_helpers" : combined ? "configured" : "disabled_by_config",
           !has_helpers ? "engine-only layout" : combined ? "combine.enabled=true" : "combine.enabled=false",
           observed_at);

    const Json& vision = optional_object(engine, "vision");
    const std::string placement = string_or(vision, "vision_tower_placement", "");
    append(out, "vision_tower_placement", placement.empty() ? "inactive" : "active",
           placement.empty() ? "not_configured" : "placement_configured",
           placement.empty() ? "vision tower placement absent" : "placement=" + placement,
           observed_at);
    const auto patches = vision.find("mm_max_patches");
    if (patches != vision.end()) {
        const auto value = patches->get<int64_t>();
        if (value < 0)
            throw std::runtime_error("feature states: negative mm_max_patches");
        append(out, "mm_max_patches", value > 0 ? "active" : "inactive",
               value > 0 ? "limit_configured" : "limit_zero",
               "max_patches=" + std::to_string(value), observed_at);
    } else {
        append(out, "mm_max_patches", "inactive", "not_configured", "mm_max_patches absent", observed_at);
    }
    return out;
}

} // namespace bc::serving::feature_state
