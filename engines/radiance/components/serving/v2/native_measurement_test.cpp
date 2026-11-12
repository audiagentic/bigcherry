#include "v2/native_measurement.h"

#include "v2/s8_rules.h"
#include "worker_footprint.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <nlohmann/json.hpp>

using namespace bc::serving::v2;

int main() {
  const std::string sha(64, 'a');

  NativeMeasurement in{};
  in.planning_key = sha;
  in.actual_owner_map_digest = sha;
  in.unowned_expert_bytes = 1024;
  in.unowned_expert_evidence = "owner-map accounting";
  in.resident_other_bytes = 2048;
  in.resident_other_evidence = "measured resident";
  in.concurrent_sessions = 2;
  in.mem_available_bytes = 8192;
  in.cgroup_headroom_bytes = 8192;
  in.pin_available_bytes = 8192;
  in.host_live_evidence = "live host";
  in.host_margin.key = "host_ledger_margin.mib";
  in.host_margin.source = MarginSource::kCache;
  in.host_margin.evidence = "calibration-run.json";
  in.host_margin.mib = 0;
  in.host_reserve_bytes = 4096;
  in.host_reserve_evidence = "host reserve";
  in.dispatch_row_capacity = 0;
  in.transport_shm = "";
  in.live_policy_requested = false;

  in.engine_plan.plan_version = 1;
  in.engine_plan.producer = "radiance --plan-memory (rad0011)";
  in.engine_plan.kv_one_session.bytes = 1280;
  in.engine_plan.kv_one_session.from = "engine plan (rad0011:kv)";
  in.engine_plan.host.push_back({"embedding", 512, bc::serving::v2::Residence::kPinned, "engine plan (rad0011:h)", ""});
  in.engine_plan.host.push_back({"host_arenas", 256, bc::serving::v2::Residence::kPageable, "engine plan (rad0011:h)", ""});
  in.engine_plan.host.push_back({"ngram_table", 64, bc::serving::v2::Residence::kPinned, "engine plan (rad0011:h)", ""});
  in.engine_plan.checkpoints = {32, 1, 3, "engine plan (rad0011:ckpt)"};
  in.engine_plan.vram.cards.push_back({});

  in.scope.model_digest = sha;
  in.scope.engine_build_digest = sha;
  in.scope.helper_build_digest = sha;
  in.scope.layout_digest = sha;
  in.scope.inventory_digest = sha;
  in.scope.owner_map_digest = sha;
  in.scope.context_target_tokens = 128;
  in.scope.retention_target_tokens = 128;
  in.scope.kv_bytes_per_token = 0;

  in.gpu.push_back({"0000:03:00.0", 4000, 1000, 5000, "engine rank"});

  // no helpers: transport must be empty (already)

  auto out = produce_native_measurement(in);
  assert(out);
  assert(out.value().measured.engine_vram_required_bytes == 4000);
  assert(out.value().measured.engine_vram_claimable_bytes == 5000);
  // wire json parseable and kind correct
  auto j = nlohmann::json::parse(out.value().s11_wire_json);
  assert(j.at("kind") == "bc_serve_v2_measured_inputs");
  assert(j.at("resource_inputs").at("kv_bytes_per_token") == 10);
  assert(j.at("resource_inputs").at("unowned_expert_bytes") == 1024);
  assert(j.at("resource_inputs").at("resident_other_bytes") == 2048);
  assert(j.at("scope").at("kv_bytes_per_token") == 10);

  // Helper case
  NativeHelper h;
  h.room.role = "helper0";
  h.room.bdf = "0000:08:00.0";
  h.room.physical_vram_bytes = 1000;
  h.room.live_claimable_bytes = 900;
  h.room.arch_reserve_bytes = 100;
  h.room.reserve_evidence = "reserve";
  h.room.owned_experts = 4;
  h.shape = {2, 4, 64, 128, 2, 8};
  h.lanes.small_rows[0] = 2;
  h.lanes.small_count = 1;
  h.lanes.large_rows = 4;
  h.lanes.open[0] = true;
  h.incoming_stride_bytes = r3::worker_footprint::stride_bound(512);
  h.footprint_evidence = "footprint";
  in.helpers.push_back(h);
  in.dispatch_row_capacity = 8;
  in.transport_shm = "shm";
  in.engine_plan.vram.cards.push_back({}); // second rank? keep one gpu? need gpu count match cards
  // gpu count must match cards count; currently one gpu but two cards -> fail. revert cards to one
  in.engine_plan.vram.cards.clear();
  in.engine_plan.vram.cards.push_back({});
  auto out2 = produce_native_measurement(in);
  assert(out2);
  auto j2 = nlohmann::json::parse(out2.value().s11_wire_json);
  assert(!j2.at("resource_inputs").at("helper_footprints").empty());

  // refusal: non-divisible KV
  in.engine_plan.kv_one_session.bytes = 1281;
  assert(!produce_native_measurement(in));
  in.engine_plan.kv_one_session.bytes = 1280;

  // refusal: pin headroom missing
  in.pin_available_bytes = 0;
  assert(!produce_native_measurement(in));
  in.pin_available_bytes = 8192;

  // refusal: duplicate gpu bdf
  in.gpu.push_back(in.gpu[0]);
  assert(!produce_native_measurement(in));

  return 0;
}
