#include "v2/p1_measurement.h"

#include <cassert>
#include <string>

using namespace bc::serving::v2;

int main() {
  P1Native m{};
  const std::string sha(64, 'a');
  m.planning_key = sha;
  m.model_digest = sha;
  m.engine_build_digest = sha;
  m.helper_build_digest = sha;
  m.layout_digest = sha;
  m.inventory_digest = sha;
  m.provenance = "engine measuring-start /run/test";
  m.owner_map.owner_map_sha256 = sha;
  m.context_target_tokens = 128;
  m.retention_target_tokens = 128;
  m.concurrent_sessions = 2;
  m.unowned_expert_bytes = 1024;
  m.resident_other_bytes = 2048;
  m.host_reserve_bytes = 4096;
  m.mem_available_bytes = 8192;
  m.cgroup_headroom_bytes = 8192;
  m.pin_available_bytes = 8192;
  m.host_margin.key = "host_ledger_margin.mib";
  m.host_margin.source = MarginSource::kCache;
  m.host_margin.evidence = "calibration-run.json";
  m.host_margin.mib = 0;
  m.engine_plan.plan_version = 1;
  m.engine_plan.producer = "radiance --plan-memory (rad0011)";
  m.engine_plan.kv_one_session.bytes = 1280;
  m.engine_plan.host.push_back({"embedding", 512, Residence::kPinned, "engine", ""});
  m.engine_plan.host.push_back({"host_arenas", 256, Residence::kPageable, "engine", ""});
  m.engine_plan.host.push_back({"ngram_table", 64, Residence::kPinned, "engine", ""});
  m.engine_plan.checkpoints = {32, 1, 3, "engine"};
  m.gpu.push_back({"0000:03:00.0", 4000, 1000});

  auto published = publish_p1_measurement(m);
  assert(published);
  const auto& doc = published.value();
  const auto& r = doc.at("resource_inputs");
  assert(r.at("kv_bytes_per_token") == 10);
  assert(r.at("host_embedding_bytes") == 512);
  assert(r.at("staging_bytes") == 256);
  assert(r.at("ngram_bytes") == 64);
  assert(r.at("checkpoint_bytes") == 96);
  assert(r.at("unowned_expert_bytes") == 1024);
  assert(r.at("helper_footprints").empty());
  assert(doc.at("scope").at("kv_bytes_per_token") == 10);

  // A calibration target is NOT a substitute for an exact engine KV size.
  m.engine_plan.kv_one_session.bytes = 1281;
  assert(!publish_p1_measurement(m));
  m.engine_plan.kv_one_session.bytes = 1280;
  m.owner_map.helpers.emplace_back();
  assert(!publish_p1_measurement(m)); // no phantom helper footprint

  P1Helper h;
  h.role = "helper0";
  h.bdf = "0000:08:00.0";
  h.shape = {2, 4, 64, 128, 2, 8};
  h.lanes.small_rows[0] = 2;
  h.lanes.small_count = 1;
  h.lanes.large_rows = 4;
  h.lanes.open[0] = true;
  h.largest_expert_device_bytes = 512;
  m.helpers.push_back(h);
  published = publish_p1_measurement(m);
  assert(published);
  const auto& fp = published.value().at("resource_inputs").at("helper_footprints").at(0);
  assert(fp.at("lanes_bytes").get<uint64_t>() > 0);
  assert(fp.at("total_bytes") ==
         fp.at("lanes_bytes").get<uint64_t>() +
         fp.at("pointer_banks_bytes").get<uint64_t>() +
         fp.at("slot_map_bytes").get<uint64_t>() +
         fp.at("incoming_slot_bytes").get<uint64_t>());

  m.pin_available_bytes = -1;
  assert(!publish_p1_measurement(m));
  m.pin_available_bytes = 8192;
  m.helpers.push_back(h);
  assert(!publish_p1_measurement(m)); // duplicate BDF/role
}
