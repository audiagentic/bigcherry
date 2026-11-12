#include "v2/p1_measurement.h"

#include <algorithm>
#include <limits>
#include <set>

namespace bc::serving::v2 {
namespace {
using Json = nlohmann::json;
constexpr int64_t kWireMax = 9007199254740991LL;
bool count(int64_t n) { return n >= 0 && n <= kWireMax; }
bool positive(int64_t n) { return n > 0 && n <= kWireMax; }
bool digest(const std::string& s) {
  return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}
bool sum(int64_t a, int64_t b, int64_t *out) {
  if (!count(a) || !count(b) || a > kWireMax - b) return false;
  *out = a + b;
  return true;
}
bool product(int64_t a, int64_t b, int64_t *out) {
  if (!count(a) || !count(b) || (a && b > kWireMax / a)) return false;
  *out = a * b;
  return true;
}
ConfigError fail(std::string field, std::string why) {
  return make_error(ConfigErrorCode::invalid_value,
                    "measurement." + field, std::move(why));
}
} // namespace

Result<Json> publish_p1_measurement(const P1Native& m) {
  if (!digest(m.planning_key) || !digest(m.model_digest) ||
      !digest(m.engine_build_digest) || !digest(m.helper_build_digest) ||
      !digest(m.layout_digest) || !digest(m.inventory_digest) ||
      !digest(m.owner_map.owner_map_sha256) || m.provenance.empty())
    return fail("identity", "complete model/build/layout/inventory/owner measurements required");
  int64_t service_capacity = 0;
  if (!positive(m.context_target_tokens) || m.retention_target_tokens < 0 ||   // 0: no host KV tier (amendment 19)
      !positive(m.concurrent_sessions) ||
      !product(m.context_target_tokens, m.concurrent_sessions, &service_capacity) ||
      m.retention_target_tokens > service_capacity)
    return fail("targets", "missing or inconsistent context/session/retention targets");
  const auto& p = m.engine_plan;
  if (p.plan_version != 1 || p.producer != "radiance --plan-memory (rad0011)" ||
      !positive(p.kv_one_session.bytes) ||
      p.kv_one_session.bytes % m.context_target_tokens)
    return fail("engine_plan.kv_one_session", "engine KV bytes must cover exactly one full context in integral bytes/token");
  const int64_t kv_per_token = p.kv_one_session.bytes / m.context_target_tokens;
  if (!positive(kv_per_token) || !count(m.unowned_expert_bytes) ||
      !count(m.resident_other_bytes) || !count(m.host_reserve_bytes) ||
      !positive(m.mem_available_bytes) || !positive(m.cgroup_headroom_bytes) ||
      !positive(m.pin_available_bytes))
    return fail("host", "engine residency and live host/cgroup/pin measurements required");
  if (m.host_margin.key != "host_ledger_margin.mib" ||
      !count(m.host_margin.mib) || m.host_margin.evidence.empty())
    return fail("host_margin", "qualified P2 margin record and evidence required");

  int64_t embedding = 0, ngram = 0, staging = 0, checkpoint = 0;
  std::set<std::string> consumers;
  for (const auto& c : p.host) {
    if (!consumers.insert(c.name).second || !count(c.bytes) || c.from.empty())
      return fail("engine_plan.host", "duplicate or unmeasured host consumer");
    if (c.name == "embedding") embedding = c.bytes;
    else if (c.name == "ngram_table") ngram = c.bytes;
    else if (c.name == "host_arenas") staging = c.bytes;
    else return fail("engine_plan.host", "unclassified engine host consumer: " + c.name);
  }
  if (!product(p.checkpoints.slot_bytes, p.checkpoints.want_slots, &checkpoint) ||
      !count(p.checkpoints.floor_slots) ||
      p.checkpoints.floor_slots > p.checkpoints.want_slots ||
      p.checkpoints.from.empty())
    return fail("engine_plan.checkpoints", "checkpoint allocation lacks measured slot geometry");

  Json footprints = Json::array();
  std::set<std::string> roles, helper_bdfs;
  if (m.helpers.size() != m.owner_map.helpers.size())
    return fail("owner_map.helpers", "actual owner map/helper footprint count differs");
  for (const auto& h : m.helpers) {
    if (h.role.empty() || h.bdf.empty() || !roles.insert(h.role).second ||
        !helper_bdfs.insert(h.bdf).second || !h.largest_expert_device_bytes ||
        !h.shape.act_group)
      return fail("helpers", "selected helper has no unique role/BDF or measured model shape");
    namespace fp = r3::worker_footprint;
    const uint64_t lane = fp::lanes_bytes(h.shape, h.lanes);
    const uint64_t pointers = fp::pointer_bank_bytes(h.shape);
    const uint64_t slots = fp::slot_map_bytes(h.shape);
    const uint64_t incoming = fp::stride_bound(h.largest_expert_device_bytes);
    const uint64_t total = fp::fixed_bytes(h.shape, h.lanes, incoming);
    if (total > uint64_t(kWireMax) || total == 0 ||
        lane > uint64_t(kWireMax) || pointers > uint64_t(kWireMax) ||
        slots > uint64_t(kWireMax) || incoming > uint64_t(kWireMax) ||
        lane + pointers + slots + incoming != total)
      return fail("helpers", "worker_footprint.h sizing exceeds wire range");
    footprints.push_back({{"role", h.role}, {"bdf", h.bdf},
                          {"lanes_bytes", lane}, {"pointer_banks_bytes", pointers},
                          {"slot_map_bytes", slots}, {"incoming_slot_bytes", incoming},
                          {"total_bytes", total}});
  }

  Json gpu = Json::array();
  std::set<std::string> gpu_bdfs;
  for (const auto& g : m.gpu) {
    if (g.bdf.empty() || !gpu_bdfs.insert(g.bdf).second ||
        !positive(g.required_bytes) || !count(g.fixed_bytes) ||
        g.fixed_bytes > g.required_bytes)
      return fail("gpu", "missing or duplicate measured engine rank demand");
    gpu.push_back({{"bdf", g.bdf}, {"required_bytes", g.required_bytes},
                   {"fixed_bytes", g.fixed_bytes}, {"claimable_bytes", g.claimable_bytes}});
  }
  if (gpu.empty())
    return fail("gpu", "engine pre-weight VRAM demand is required");

  const auto& margin = m.host_margin;
  Json inputs = {
    {"host_embedding_bytes", embedding},
    {"unowned_expert_bytes", m.unowned_expert_bytes},
    {"staging_bytes", staging},
    {"ngram_bytes", ngram},
    {"checkpoint_bytes", checkpoint},
    {"resident_other_bytes", m.resident_other_bytes},
    {"kv_bytes_per_token", kv_per_token},
    {"context_target_tokens", m.context_target_tokens},
    {"retention_target_tokens", m.retention_target_tokens},
    {"mem_available_bytes", m.mem_available_bytes},
    {"cgroup_headroom_bytes", m.cgroup_headroom_bytes},
    {"pin_available_bytes", m.pin_available_bytes},
    {"owner_map_digest", m.owner_map.owner_map_sha256},
    {"host_pool_margin_calibration", {
      {"key", margin.key},
      {"source", margin.source == MarginSource::kSeed ? "seed" : "cache"},
      {"evidence", margin.evidence}, {"mib", margin.mib}}},
    {"helper_footprints", std::move(footprints)}
  };
  return Json{
    {"schema_version", 1}, {"kind", "bc_serve_v2_measured_inputs"},
    {"planning_key", m.planning_key}, {"provenance", m.provenance},
    {"resource_inputs", std::move(inputs)}, {"gpu", std::move(gpu)},
    {"concurrent_sessions", m.concurrent_sessions},
    {"host_reserve_bytes", m.host_reserve_bytes},
    {"scope", {{"model_digest", m.model_digest},
               {"engine_build_digest", m.engine_build_digest},
               {"helper_build_digest", m.helper_build_digest},
               {"layout_digest", m.layout_digest},
               {"inventory_digest", m.inventory_digest},
               {"owner_map_digest", m.owner_map.owner_map_sha256},
               {"context_target_tokens", m.context_target_tokens},
               {"retention_target_tokens", m.retention_target_tokens},
               {"kv_bytes_per_token", kv_per_token}}}
  };
}
} // namespace bc::serving::v2
