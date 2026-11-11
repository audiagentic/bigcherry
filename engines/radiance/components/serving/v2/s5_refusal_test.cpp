#include "admission.h"

#include <nlohmann/json.hpp>

#include <cassert>
#include <cstdint>
#include <string>

namespace v2 = bc::serving::v2;
using Json = nlohmann::json;

constexpr int64_t MiB = 1024 * 1024;
constexpr const char* kBdf = "0000:09:00.0";

void check_wire(const v2::ConfigError& error, const char* reason, const char* resource,
                int64_t expected_shortfall, bool gpu) {
    assert(error.code == v2::ConfigErrorCode::admission_refused);
    assert(v2::exit_code_of(error) == 4);
    assert(error.room_refusal);
    const v2::RoomRefusal& r = *error.room_refusal;
    Json j = Json::parse(v2::room_refusal_json(r));
    assert(j.size() == 13);
    assert(j.at("schema_version") == 1);
    assert(j.at("kind") == "serving_admission_refusal");
    assert(j.at("phase") == "before_weights");
    assert(j.at("reason_code") == reason);
    assert(j.at("resource") == resource);
    assert(j.at("shortfall_bytes") == expected_shortfall);
    assert(j.at("required_bytes").get<int64_t>() -
               j.at("available_bytes").get<int64_t>() == expected_shortfall);
    assert(j.at("required_bytes").get<int64_t>() ==
               j.at("allocations_still_to_come_bytes").get<int64_t>() +
                   j.at("headroom_bytes").get<int64_t>());
    int64_t terms = 0;
    for (auto it = j.at("component_breakdown").begin(); it != j.at("component_breakdown").end(); ++it)
        terms += it.value().get<int64_t>();
    assert(terms == j.at("allocations_still_to_come_bytes").get<int64_t>());
    if (gpu) {
        assert(j.at("device_bdf") == kBdf);
        assert(j.at("may_change") == Json::array({"gpu_elastic_kv"}));
    } else {
        assert(j.at("device_bdf").is_null());
    }
    // Human text and generic error JSON are rendered from precisely this struct.
    const std::string human = v2::render_human("radiance", error);
    assert(human.find(std::to_string(expected_shortfall)) != std::string::npos);
    assert(human.find(resource) != std::string::npos);
    assert(Json::parse(v2::render_json(error)).at("admission_refusal") == j);
}

int main() {
    v2::EnginePlan expected{};
    expected.vram.statics = 4 * MiB;
    expected.vram.experts = 7 * MiB;
    expected.vram.stage = 2 * MiB;
    expected.vram.expert_floor = 7 * MiB;
    expected.vram.kv_ceiling = 8 * MiB;
    expected.kv_one_session.bytes = 3 * MiB;
    expected.vram.kv_resolved = 3 * MiB;
    expected.vram.claimable = 17 * MiB;
    v2::EnginePlan live = expected;
    const int64_t required = 4 * MiB + 7 * MiB + 2 * MiB + 3 * MiB;
    live.vram.claimable = required - MiB;
    auto refused = v2::check_engine_plan(expected, live, 0, kBdf);
    assert(refused);
    check_wire(*refused, "gpu_room", "vram", MiB, true);
    Json components = Json::parse(v2::room_refusal_json(*refused->room_refusal)).at("component_breakdown");
    assert(components == Json({{"statics", 4 * MiB}, {"experts", 7 * MiB},
                               {"stage", 2 * MiB}, {"kv_required", 3 * MiB}}));

    // A round-up/partition near-miss also produces a true GPU ceiling, not
    // a misleading KV-only byte requirement.
    live.vram.claimable = required;
    live.vram.kv_resolved = 3 * MiB - MiB;
    refused = v2::check_engine_plan(expected, live, 0, kBdf);
    assert(refused);
    check_wire(*refused, "gpu_room", "vram", MiB, true);

    // No authoritative BDF: offline tests can inspect the config error, but
    // must not manufacture a retryable S11 wire result.
    refused = v2::check_engine_plan(expected, live, 0);
    assert(refused && !refused->room_refusal);

    // Host/pinned memory near miss: required = exact measured components +
    // measured headroom, not a guessed reserve or a coarse MiB threshold.
    const int64_t pool = 8 * MiB;
    const int64_t embedding = 2 * MiB;
    refused = v2::check_host_room("host_pool", pool + embedding - MiB, 0,
                                  {{"host_pool", pool}, {"embedding", embedding}});
    assert(refused);
    check_wire(*refused, "pin_room", "host_pool", MiB, false);
    assert(refused->room_refusal->may_change == std::vector<std::string>({"host_kv"}));

    refused = v2::check_host_room("host_ram", 12 * MiB - MiB, 2 * MiB,
                                  {{"host_pool", 7 * MiB}, {"ngram", 3 * MiB}});
    assert(refused);
    check_wire(*refused, "host_room", "host_ram", MiB, false);

    // Fits exactly, invalid figures and absent components are not refusals.
    assert(!v2::check_host_room("host_pool", pool + embedding, 0,
                                {{"host_pool", pool}, {"embedding", embedding}}));
    auto invalid = v2::check_host_room("host_pool", -1, 0, {{"host_pool", pool}});
    assert(invalid && invalid->code == v2::ConfigErrorCode::invalid_value &&
           v2::exit_code_of(*invalid) == 2 && !invalid->room_refusal);
    invalid = v2::check_host_room("host_ram", 0, 1, {});
    assert(invalid && invalid->code == v2::ConfigErrorCode::invalid_value);
    return 0;
}
