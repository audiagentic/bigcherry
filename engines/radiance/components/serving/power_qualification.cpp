#include "power_qualification.h"

#include <fstream>

#include <nlohmann/json.hpp>

namespace bc::serving::power {
namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;

constexpr std::pair<const char*, ControlId> kControls[] = {{"vddgfx_offset_mv", ControlId::voltage},
                                                           {"sclk_max_mhz", ControlId::shader},
                                                           {"mclk_max_mhz", ControlId::memory},
                                                           {"power_cap_w", ControlId::board_cap}};

Json requested(const v2::PowerSettings& s) {
    Json out = Json::object();
    const auto put = [&](const char* key, const std::optional<int64_t>& value) {
        if (value)
            out[key] = *value;
    };
    put("vddgfx_offset_mv", s.vddgfx_offset_mv);
    put("sclk_max_mhz", s.sclk_max_mhz);
    put("mclk_max_mhz", s.mclk_max_mhz);
    put("power_cap_w", s.power_cap_w);
    return out;
}
const Json* card_of(const Json& observation, const std::string& bdf) {
    if (!observation.is_object() || !observation.contains("cards") || !observation.at("cards").is_array())
        return nullptr;
    for (const auto& card : observation.at("cards"))
        if (card.is_object() && card.value("bdf", "") == bdf)
            return &card;
    return nullptr;
}
bool has_event(const Json& events, const char* kind, const std::string& bdf) {
    for (const auto& event : events)
        if (event.value("kind", "") == kind && event.value("bdf", "") == bdf)
            return true;
    return false;
}
// The receipt's readback field for one control, as the number a setting must equal.
bool reads_back(const Json& card, const Json& want) {
    for (const auto& [key, id] : kControls) {
        (void)id;
        const bool is_cap = std::string_view(key) == "power_cap_w";
        const std::string field = is_cap ? "power_cap_uw" : key;
        if (!want.contains(key))
            continue;
        const auto it = card.find(field);
        if (it == card.end() || !it->is_number_integer() || !want.at(key).is_number_integer())
            return false;
        if (it->get<int64_t>() != want.at(key).get<int64_t>() * (is_cap ? 1000000 : 1))
            return false;
    }
    return true;
}
bool same_values(const Json& a, const Json& b) {
    for (const char* key : {"vddgfx_offset_mv", "sclk_max_mhz", "mclk_max_mhz", "power_cap_uw"})
        if (a.value(key, Json()) != b.value(key, Json()))
            return false;
    return true;
}
bool qualifies(const Json& receipt, const std::string& bdf, const Json& want, std::string_view identity) {
    if (!receipt.is_object() || receipt.value("schema_version", 0) != 1 || !receipt.contains("request") ||
        !receipt.contains("transaction") || !receipt.contains("observations"))
        return false;
    const auto& request = receipt.at("request");
    const auto& transaction = receipt.at("transaction");
    if (request.value("mode", "") != "apply" || transaction.value("state", "") != "restored" ||
        !transaction.contains("events") || !transaction.at("events").is_array())
        return false;
    bool requested_here = false;
    for (const auto& device : request.value("devices", Json::array()))
        if (device.value("bdf", "") == bdf && device.value("settings", Json()) == want)
            requested_here = true;
    const auto& events = transaction.at("events");
    if (!requested_here || !has_event(events, "applied", bdf) || !has_event(events, "restored", bdf))
        return false;
    for (const char* bad : {"apply_failed", "restore_failed", "restore_refused", "rollback"})
        if (has_event(events, bad, bdf))
            return false;
    const auto& seen = receipt.at("observations");
    if (!seen.is_object() || !seen.contains("start") || !seen.contains("post_apply") || !seen.contains("stop"))
        return false;
    const Json* start = card_of(seen.at("start"), bdf);
    const Json* applied = card_of(seen.at("post_apply"), bdf);
    const Json* stop = card_of(seen.at("stop"), bdf);
    if (!start || !applied || !stop || identity.empty() || start->value("identity", "") != identity)
        return false;
    return reads_back(*applied, want) && same_values(*start, *stop);
}
} // namespace

v2::S10DeviceProbe probe_of(const CardSnapshot& card) {
    v2::S10DeviceProbe probe;
    probe.bdf = card.pci.value;
    probe.driver = card.amdgpu ? "amdgpu" : "";
    for (const auto& [key, id] : kControls) {
        const Capability& cap = card.at(id);
        v2::S10ControlProbe control;
        control.supported = cap.writable && cap.minimum && cap.maximum;
        control.minimum = cap.minimum.value_or(0);
        control.maximum = cap.maximum.value_or(0);
        control.original_readable = cap.current.has_value();
        probe.controls.emplace(key, control);
    }
    return probe;
}

std::optional<std::string> qualifying_receipt(const fs::path& receipts, const PciAddress& pci,
                                              const v2::PowerSettings& settings, std::string_view identity) {
    const Json want = requested(settings);
    std::error_code ec;
    for (fs::directory_iterator it(receipts, ec), end; !ec && it != end; it.increment(ec)) {
        const auto name = it->path().filename().string();
        if (name.size() < 11 || name.substr(name.size() - 11) != ".power.json")
            continue;
        std::ifstream in(it->path());
        const Json receipt = Json::parse(in, nullptr, false);
        if (qualifies(receipt, pci.value, want, identity))
            return receipt.value("run_id", name);
    }
    return std::nullopt;
}

} // namespace bc::serving::power
