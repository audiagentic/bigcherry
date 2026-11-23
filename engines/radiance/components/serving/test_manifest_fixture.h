#pragma once

#include "manifest_writer.h"

namespace bc::test {

using serving::Json;
namespace sv = serving;

/* A native P1 measurement record (bc_serve_v2_measured_inputs) consistent with the manifest `unmeasured`
 * describes: its owner map and helpers, a host pool that sums to `pool_mib` with the given margin, and live
 * room far above the need. Test-only: real records come from the session-owned P1 producer on the cards. */
inline Json p1_record_for(const Json& unmeasured, int64_t pool_mib, int64_t margin_mib, int64_t reserve_mib) {
    constexpr int64_t MiB = 1048576;
    const Json& spec = unmeasured.at("spec");
    const std::string owner = spec.at("plan").at("owner_map").at("owner_map_sha256");
    Json footprints = Json::array();
    for (const Json& h : spec.at("processes").at("helpers"))
        footprints.push_back({{"role", h.at("role")},
                              {"bdf", h.at("gpu")},
                              {"lanes_bytes", MiB},
                              {"pointer_banks_bytes", MiB},
                              {"slot_map_bytes", MiB},
                              {"incoming_slot_bytes", MiB},
                              {"total_bytes", 4 * MiB}});
    const Json scope = {{"model_digest", std::string(64, '1')},
                        {"engine_build_digest", std::string(64, '2')},
                        {"helper_build_digest", std::string(64, '3')},
                        {"layout_digest", std::string(64, '4')},
                        {"inventory_digest", std::string(64, '5')},
                        {"owner_map_digest", owner},
                        {"context_target_tokens", 4096},
                        {"retention_target_tokens", 4096},
                        {"kv_bytes_per_token", 1}};
    const Json inputs = {
        {"host_embedding_bytes", (pool_mib - margin_mib) * MiB},
        {"unowned_expert_bytes", 0},
        {"staging_bytes", 0},
        {"ngram_bytes", MiB},
        {"checkpoint_bytes", 0},
        {"resident_other_bytes", 0},
        {"kv_bytes_per_token", 1},
        {"context_target_tokens", 4096},
        {"retention_target_tokens", 4096},
        {"mem_available_bytes", 92160 * MiB},
        {"cgroup_headroom_bytes", 92160 * MiB},
        {"pin_available_bytes", 92160 * MiB},
        {"owner_map_digest", owner},
        {"host_pool_margin_calibration",
         {{"key", "host_ledger_margin.mib"}, {"source", "cache"}, {"evidence", "test"}, {"mib", margin_mib}}},
        {"helper_footprints", footprints}};
    return {{"schema_version", 1},
            {"kind", "bc_serve_v2_measured_inputs"},
            {"planning_key", std::string(64, '6')},
            {"provenance", "serving_test synthetic record"},
            {"resource_inputs", inputs},
            {"gpu", [&] {
                 Json cards = Json::array();
                 for (const Json& bdf : spec.at("processes").at("engine").at("device").at("gpus"))
                     cards.push_back({{"bdf", bdf}, {"required_bytes", 10}, {"fixed_bytes", 1},
                                      {"claimable_bytes", 10}});
                 return cards;
             }()},
            {"concurrent_sessions", 1},
            {"host_reserve_bytes", reserve_mib * MiB},
            {"scope", scope}};
}

inline void set_measured_manifest_fixture(sv::ManifestInputs& inputs, const sv::BuiltManifest& unmeasured,
                                          const Json& facts) {
    const auto& resolved = inputs.resolved;
    Json plan = Json::parse(R"json({
      "plan_version": 1, "producer": "radiance --plan-memory (rad0011)",
      "host": [{"name": "ngram_table", "bytes": 1048576, "residence": "pinned", "from": "engine plan (test)", "remedy": ""}],
      "checkpoints": {"slot_bytes": 1048576, "floor_slots": 8, "want_slots": 72, "from": "engine plan (test)"},
      "kv_one_session": {"bytes": 4096, "from": "engine plan (test)"},
      "vram": {"claimable": 10, "statics": 1, "elastic": 9, "experts": 0, "kv": 9, "stage": 0, "kv_resolved": 9, "expert_floor": 0,
               "kv_ceiling": 9, "expert_ceiling": 0, "from": "engine plan (test)", "cards": []},
      "host_pool_mib": 9216})json");
    const size_t ranks = resolved.values.at("layout").at("engine_cards").size();
    plan["vram"]["cards"] = Json::array();
    for (size_t i = 0; i < ranks; ++i)
        plan["vram"]["cards"].push_back(
            {{"capacity", 10}, {"free", 10}, {"engine_expert_plane", 0}, {"expert_unit_max", 1}});
    int64_t pool_mib = resolved.values.at("memory").at("host_pool_mib").get<int64_t>();
    if (pool_mib <= 1024)
        pool_mib = 9216;
    plan["host_pool_mib"] = pool_mib;

    sv::MeasuredPlan measured;
    measured.plan.plan = sv::decode_engine_plan(plan, "session fixture plan");
    measured.plan.plan_json = plan;
    measured.plan.resident = {1536, 1024};
    measured.key_inputs = sv::plan_key_inputs(
        resolved.values, facts, sv::container_identity(resolved.values.at("engine").at("model").get<std::string>()),
        std::string(64, 'b'), "");
    inputs.measured = std::move(measured);
    inputs.s10.p1_record =
        p1_record_for(unmeasured.document, pool_mib, 1024,
                      resolved.values.at("memory").at("host_reserve_mib").get<int64_t>());
}

} // namespace bc::test
