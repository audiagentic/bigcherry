#include "v2/native_measurement.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <nlohmann/json.hpp>

namespace bc::serving::v2 {
namespace {
using json = nlohmann::json;
constexpr int64_t kMax = 9007199254740991LL;
constexpr int64_t kMiB = 1024 * 1024;

ConfigError invalid(std::string field, std::string detail) {
  return make_error(ConfigErrorCode::invalid_value, std::move(field),
                    std::move(detail), "remeasure with the native engine/worker producer");
}
bool safe(int64_t value) { return value >= 0 && value <= kMax; }
bool positive(int64_t value) { return value > 0 && value <= kMax; }
bool add(int64_t a, int64_t b, int64_t *out) {
  if (!safe(a) || !safe(b) || a > kMax - b) return false;
  *out = a + b; return true;
}
bool multiply(int64_t a, int64_t b, int64_t *out) {
  if (!safe(a) || !safe(b) || (b && a > kMax / b)) return false;
  *out = a * b; return true;
}
bool digest(const std::string &value) {
  if (value.size() != 64) return false;
  for (char ch : value)
    if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')))
      return false;
  return true;
}
json scope_json(const S8Scope &s) {
  return {{"model_digest",s.model_digest},
          {"engine_build_digest",s.engine_build_digest},
          {"helper_build_digest",s.helper_build_digest},
          {"layout_digest",s.layout_digest},
          {"inventory_digest",s.inventory_digest},
          {"owner_map_digest",s.owner_map_digest},
          {"context_target_tokens",s.context_target_tokens},
          {"retention_target_tokens",s.retention_target_tokens},
          {"kv_bytes_per_token",s.kv_bytes_per_token}};
}
} // namespace

Result<NativeMeasurementOutput> produce_native_measurement(const NativeMeasurement &in) {
  const auto fail = [](std::string field, std::string why) -> Result<NativeMeasurementOutput> {
    return invalid(std::move(field), std::move(why));
  };
  const auto &plan = in.engine_plan;
  // Provenance must be the native pre-weights engine planning entrypoint.
  if (plan.plan_version != 1 || plan.producer != "radiance --plan-memory (rad0011)")
    return fail("engine_plan.producer", "not the native rad0011 measured engine plan");
  if (!digest(in.planning_key) || !digest(in.actual_owner_map_digest) ||
      in.actual_owner_map_digest != in.scope.owner_map_digest)
    return fail("scope.owner_map_digest", "missing or mismatched actual owner-map identity");
  for (const auto &d : {in.scope.model_digest, in.scope.engine_build_digest,
                        in.scope.helper_build_digest, in.scope.layout_digest,
                        in.scope.inventory_digest})
    if (!digest(d)) return fail("scope", "all measured identity digests are mandatory");

  if (!positive(in.scope.context_target_tokens) ||
      in.scope.retention_target_tokens < 0 ||   // 0: no host KV tier (amendment 19)
      !positive(in.concurrent_sessions) ||
      in.scope.retention_target_tokens > in.scope.context_target_tokens * in.concurrent_sessions)
    return fail("targets", "invalid measured request context, concurrency or retained tokens");
  if (!positive(plan.kv_one_session.bytes) ||
      plan.kv_one_session.from.find("engine plan (rad0011:") != 0)
    return fail("engine_plan.kv_one_session", "native per-session KV measurement missing");
  // KV bytes per token are the exact native full-session measurement divided by
  // the target context; a non-integral result is not a valid engine plan.
  if (plan.kv_one_session.bytes % in.scope.context_target_tokens != 0)
    return fail("engine_plan.kv_one_session",
                "native KV bytes must divide exactly by the target context");
  const int64_t kv_unit = plan.kv_one_session.bytes / in.scope.context_target_tokens;
  if (!positive(kv_unit) ||
      (in.scope.kv_bytes_per_token != 0 && in.scope.kv_bytes_per_token != kv_unit))
    return fail("scope.kv_bytes_per_token",
                "KV bytes/token differs from the native full-session plan measurement");
  if (!safe(in.unowned_expert_bytes) || in.unowned_expert_evidence.empty() ||
      !safe(in.resident_other_bytes) || in.resident_other_evidence.empty())
    return fail("engine_resident", "missing measured owner-map unowned experts or resident bytes");
  if (!positive(in.mem_available_bytes) || !positive(in.cgroup_headroom_bytes) ||
      !positive(in.pin_available_bytes) || in.host_live_evidence.empty())
    return fail("host.live", "live MemAvailable, effective cgroup and pin headroom needed");
  if (!safe(in.host_reserve_bytes) || in.host_reserve_evidence.empty() ||
      in.host_margin.key != "host_ledger_margin.mib" ||
      in.host_margin.source != MarginSource::kCache ||
      in.host_margin.evidence.empty() || !safe(in.host_margin.mib))
    return fail("host.calibration", "host reserve and measured P2 margin are required");
  if (!safe(plan.checkpoints.slot_bytes) || !safe(plan.checkpoints.floor_slots) ||
      !safe(plan.checkpoints.want_slots) ||
      plan.checkpoints.floor_slots > plan.checkpoints.want_slots ||
      plan.checkpoints.from.find("engine plan (rad0011:") != 0)
    return fail("engine_plan.checkpoints", "missing native checkpoint geometry");
  int64_t checkpoint_bytes = 0;
  if (!multiply(plan.checkpoints.slot_bytes, plan.checkpoints.want_slots, &checkpoint_bytes))
    return fail("engine_plan.checkpoints", "checkpoint allocation overflow");
  int64_t embedding = 0, ngram = 0, staging = 0;
  std::set<std::string> host_names;
  for (const auto &item : plan.host) {
    if (!safe(item.bytes) || item.from.find("engine plan (rad0011:") != 0 ||
        !host_names.insert(item.name).second)
      return fail("engine_plan.host", "duplicate, negative or non-native host consumer");
    if (item.name == "embedding") embedding = item.bytes;
    else if (item.name == "ngram_table") ngram = item.bytes;
    else if (item.name == "host_arenas") staging = item.bytes;
    else return fail("engine_plan.host", "unrecognised host allocation: " + item.name);
  }

  S8Measured m{};
  auto &r = m.resource_inputs;
  m.scope = in.scope;
  m.scope.kv_bytes_per_token = kv_unit;
  m.actual_owner_map_digest = in.actual_owner_map_digest;
  r.owner_map_digest = in.actual_owner_map_digest;
  r.host_embedding_bytes = embedding;
  r.unowned_expert_bytes = in.unowned_expert_bytes;
  r.staging_bytes = staging;
  r.ngram_bytes = ngram;
  r.checkpoint_bytes = checkpoint_bytes;
  r.resident_other_bytes = in.resident_other_bytes;
  r.kv_bytes_per_token = kv_unit;
  r.context_target_tokens = m.scope.context_target_tokens;
  r.retention_target_tokens = m.scope.retention_target_tokens;
  r.mem_available_bytes = in.mem_available_bytes;
  r.cgroup_headroom_bytes = in.cgroup_headroom_bytes;
  r.pin_available_bytes = in.pin_available_bytes;
  r.host_pool_margin_calibration = in.host_margin;
  m.host_reserve_bytes = in.host_reserve_bytes;
  m.host_reserve_evidence = in.host_reserve_evidence;
  m.checkpoint_slots = plan.checkpoints.want_slots;
  m.dispatch_row_capacity = in.dispatch_row_capacity;
  m.transport_shm = in.transport_shm;
  m.histogram_file = in.histogram_file;
  m.live_policy_requested = in.live_policy_requested;

  json helpers = json::array();
  json per_helper = json::object();
  std::set<std::string> seen;
  for (size_t i = 0; i < in.helpers.size(); ++i) {
    const auto &native = in.helpers[i];
    const auto &h = native.room;
    const std::string role = "helper" + std::to_string(i);
    if (h.role != role || h.bdf.empty() || !seen.insert(h.bdf).second ||
        !positive(h.physical_vram_bytes) || !positive(h.live_claimable_bytes) ||
        !safe(h.arch_reserve_bytes) || h.reserve_evidence.empty() ||
        !safe(h.owned_experts) || native.footprint_evidence.empty())
      return fail("helpers", "missing or mismatched live helper role/BDF/room provenance");
    const auto &shape = native.shape;
    if (!shape.layers || !shape.experts || !shape.hidden || !shape.intermediate ||
        !shape.top_k || !shape.act_group || !native.incoming_stride_bytes ||
        native.incoming_stride_bytes > static_cast<uint64_t>(kMax))
      return fail("helpers.shape", "worker_footprint.h needs measured model shape and incoming stride");
    namespace fp = r3::worker_footprint;
    const uint64_t lanes = fp::lanes_bytes(shape, native.lanes);
    const uint64_t banks = fp::pointer_bank_bytes(shape);
    const uint64_t slots = fp::slot_map_bytes(shape);
    const uint64_t stride = native.incoming_stride_bytes;
    const uint64_t total = fp::fixed_bytes(shape, native.lanes, stride);
    if (lanes > static_cast<uint64_t>(kMax) || banks > static_cast<uint64_t>(kMax) ||
        slots > static_cast<uint64_t>(kMax) || total > static_cast<uint64_t>(kMax) ||
        total != lanes + banks + slots + stride || !total)
      return fail("helpers.footprint", "worker fixed bytes overflow or mismatch");
    const int64_t total_i = static_cast<int64_t>(total);
    r.helper_footprints.push_back({role,h.bdf,static_cast<int64_t>(lanes),
        static_cast<int64_t>(banks), static_cast<int64_t>(slots),
        static_cast<int64_t>(stride), total_i});
    m.helpers.push_back(h);
    per_helper[h.bdf] = total_i;
    helpers.push_back({{"role",role},{"bdf",h.bdf},
        {"lanes_bytes",lanes},{"pointer_banks_bytes",banks},
        {"slot_map_bytes",slots},{"incoming_slot_bytes",stride},
        {"total_bytes",total}});
  }

  if (in.gpu.empty() || in.gpu.size() != plan.vram.cards.size())
    return fail("engine.gpu", "one live measured engine GPU entry per native plan rank required");
  json gpu = json::array();
  int64_t minimum_room = std::numeric_limits<int64_t>::max();
  for (const auto &device : in.gpu) {
    if (device.bdf.empty() || !seen.insert(device.bdf).second ||
        !positive(device.required_bytes) ||
        !positive(device.claimable_bytes) ||
        !positive(device.fixed_bytes) ||
        device.fixed_bytes > device.required_bytes || device.evidence.empty())
      return fail("engine.gpu", "missing or duplicated per-rank native engine room");
    gpu.push_back({{"bdf",device.bdf},{"required_bytes",device.required_bytes},
                   {"fixed_bytes",device.fixed_bytes}});
    const int64_t room = device.claimable_bytes - device.required_bytes;
    if (room < minimum_room) {
      minimum_room = room;
      m.engine_vram_required_bytes = device.required_bytes;
      m.engine_vram_claimable_bytes = device.claimable_bytes;
    }
  }
  if (in.helpers.empty() &&
      (!in.transport_shm.empty() || in.histogram_file || in.live_policy_requested))
    return fail("transport", "handoff requested without real remote helpers");
  if (!in.helpers.empty() && (in.transport_shm.empty() || !positive(in.dispatch_row_capacity)))
    return fail("transport", "helper handoff needs measured dispatch and shared memory transport");
  if (!safe(r.host_pool_margin_calibration.mib * kMiB))
    return fail("host.calibration", "measured margin overflow");

  json ledger = {
    {"host_embedding_bytes",r.host_embedding_bytes},
    {"unowned_expert_bytes",r.unowned_expert_bytes},
    {"staging_bytes",r.staging_bytes},
    {"ngram_bytes",r.ngram_bytes},
    {"checkpoint_bytes",r.checkpoint_bytes},
    {"resident_other_bytes",r.resident_other_bytes},
    {"kv_bytes_per_token",r.kv_bytes_per_token},
    {"context_target_tokens",r.context_target_tokens},
    {"retention_target_tokens",r.retention_target_tokens},
    {"mem_available_bytes",r.mem_available_bytes},
    {"cgroup_headroom_bytes",r.cgroup_headroom_bytes},
    {"pin_available_bytes",r.pin_available_bytes},
    {"owner_map_digest",r.owner_map_digest},
    {"host_pool_margin_calibration",{
      {"key",r.host_pool_margin_calibration.key},
      {"mib",r.host_pool_margin_calibration.mib},
      {"evidence",r.host_pool_margin_calibration.evidence},
      {"source","cache"}}},
    {"helper_footprints",helpers}
  };
  json result = {
    {"schema_version",1}, {"kind","bc_serve_v2_measured_inputs"},
    {"planning_key",in.planning_key},
    {"provenance",plan.producer + "; " + in.unowned_expert_evidence +
      "; " + in.host_live_evidence},
    {"resource_inputs",ledger},
    {"gpu",gpu},
    {"concurrent_sessions",in.concurrent_sessions},
    {"host_reserve_bytes",in.host_reserve_bytes},
    {"scope",scope_json(m.scope)}
  };
  return NativeMeasurementOutput{std::move(m), result.dump()};
}
} // namespace bc::serving::v2
