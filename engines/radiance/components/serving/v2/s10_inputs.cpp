#include "v2/s10_inputs.h"

#include "generated/v2/serving_codec.h"

#include <algorithm>

namespace bc::serving::v2 {
namespace {

using json = nlohmann::json;
constexpr int64_t kMiB = 1024 * 1024;

ConfigError fail(std::string path, std::string why) {
  return make_error(ConfigErrorCode::invalid_value, std::move(path), std::move(why),
                    "resolve the manifest again with bc-serve");
}

bool integer(const json &j, const char *key) {
  return j.is_object() && j.contains(key) && j.at(key).is_number_integer();
}

std::string text(const json &j, const char *key) {
  return j.is_object() && j.contains(key) && j.at(key).is_string() ? j.at(key).get<std::string>() : std::string();
}

// The resolver's value of `key` (dotted, as serving.cpp names fields) and the source it records for it.
S8Setting setting(const bc::serving::Resolved &r, const std::string &key) {
  const auto dot = key.find('.');
  const json &v = r.values.at(key.substr(0, dot)).at(key.substr(dot + 1));
  const int64_t value = v.is_boolean() ? (v.get<bool>() ? 1 : 0) : v.get<int64_t>();
  std::string source = key;
  if (r.provenance.contains(key))
    source = text(r.provenance.at(key), "origin") + ": " + text(r.provenance.at(key), "source");
  return {value, source};
}

Result<S8Scope> scope_of(const json &j) {
  if (!j.is_object()) return fail("p1.scope", "the P1 record has no scope");
  S8Scope s;
  s.model_digest = text(j, "model_digest");
  s.engine_build_digest = text(j, "engine_build_digest");
  s.helper_build_digest = text(j, "helper_build_digest");
  s.layout_digest = text(j, "layout_digest");
  s.inventory_digest = text(j, "inventory_digest");
  s.owner_map_digest = text(j, "owner_map_digest");
  for (const auto &[key, out] : {std::pair{"context_target_tokens", &s.context_target_tokens},
                                 std::pair{"retention_target_tokens", &s.retention_target_tokens},
                                 std::pair{"kv_bytes_per_token", &s.kv_bytes_per_token}}) {
    if (!integer(j, key)) return fail(std::string("p1.scope.") + key, "missing integer");
    *out = j.at(key).get<int64_t>();
  }
  return s;
}

} // namespace

Result<S8Calibration> parse_calibration_record(const json &r) {
  S8Calibration c;
  c.key = text(r, "key");
  c.evidence_id = text(r, "evidence_id");
  c.raw_run = text(r, "raw_run");
  c.method = text(r, "method");
  c.recorded_date = text(r, "recorded_date");
  if (c.key.empty() || !integer(r, "value") || !integer(r, "measured_at_unix_s") || !integer(r, "valid_until_unix_s"))
    return fail("calibration_records", "a record needs key, value and its measurement interval");
  c.value = r.at("value").get<int64_t>();
  c.measured_at_unix_s = r.at("measured_at_unix_s").get<int64_t>();
  c.valid_until_unix_s = r.at("valid_until_unix_s").get<int64_t>();
  c.source = text(r, "source") == "seed" ? CalibrationSource::kSeed : CalibrationSource::kCache;
  if (r.contains("measured_candidates") && r.at("measured_candidates").is_array())
    for (const json &v : r.at("measured_candidates"))
      if (v.is_number_integer()) c.measured_candidates.push_back(v.get<int64_t>());
  auto scope = scope_of(r.contains("scope") ? r.at("scope") : json());
  if (!scope) return scope.error();
  c.scope = scope.value();
  return c;
}

Result<S10Inputs> s10_inputs(const Manifest &m, const bc::serving::Resolved &r, const WriterInputs &w) {
  S10Inputs in;
  in.now_unix_s = w.now_unix_s;
  in.profile.name = w.power_profile;
  if (w.power_profile != "stock")
    return fail("spec.intent.power.profile",
                "bc-serve knows the rules of the stock profile only; a named non-stock profile needs its rule file");
  in.mode = PowerMode::kObserve;

  const bool has_helpers = !m.spec.processes.helpers.empty();
  const auto &placement = m.spec.handoff ? std::optional(m.spec.handoff->placement) : std::nullopt;
  const bool live = placement && placement->live_policy.has_value();
  in.preflight.requested = !has_helpers ? LivePolicyRequested::kNotApplicable
                           : live      ? LivePolicyRequested::kOn
                                       : LivePolicyRequested::kOff;
  in.preflight.applicable = has_helpers && m.spec.processes.engine.device.gpus.size() == 1;
  // The writer has already derived the policy from the heat feed (or been given one in the lab); its presence is the
  // fresh-heat decision.
  in.preflight.fresh_heat = live;
  if (live) {
    in.preflight.effect_gate = placement->live_policy->effect_gate;
    in.preflight.max_heat_age_ns = placement->live_policy->max_heat_age_ns;
  }

  if (m.spec.plan.state != PlanState::kMeasured) return in;   // the measuring start: no measurement yet

  const json &p1 = w.p1_record;
  if (!p1.is_object() || text(p1, "kind") != "bc_serve_v2_measured_inputs")
    return fail("p1_record", "a measured manifest needs the native P1 measurement record (bc-serve --publish-v2-p1)");
  S8Measured &s = in.measured;
  if (auto e = decode(p1.at("resource_inputs"), "p1.resource_inputs", &s.resource_inputs)) return *e;
  auto scope = scope_of(p1.contains("scope") ? p1.at("scope") : json());
  if (!scope) return scope.error();
  s.scope = scope.value();
  s.actual_owner_map_digest = m.spec.plan.owner_map.owner_map_sha256;

  for (size_t i = 0; i < m.spec.processes.helpers.size(); ++i) {
    const HelperProcess &h = m.spec.processes.helpers[i];
    const auto gpu = std::find_if(m.spec.facts.gpus.begin(), m.spec.facts.gpus.end(),
                                  [&](const GpuFact &g) { return g.bdf == h.gpu; });
    if (gpu == m.spec.facts.gpus.end()) return fail("spec.processes.helpers", h.gpu + " is not a detected GPU");
    S8HelperRoom room;
    room.role = h.role;
    room.bdf = h.gpu;
    room.physical_vram_bytes = gpu->vram.total_mib * kMiB;
    room.live_claimable_bytes = gpu->vram.available_mib * kMiB;
    room.arch_reserve_bytes = static_cast<int64_t>(bc::serving::arch_reserve_mib(gpu->arch)) * kMiB;
    room.reserve_evidence = "archs.toml reserve-mib for " + gpu->arch;
    room.owned_experts = m.spec.plan.owner_map.helpers.at(i).count;
    s.helpers.push_back(room);
  }

  if (!integer(p1, "host_reserve_bytes")) return fail("p1.host_reserve_bytes", "missing integer");
  s.host_reserve_bytes = p1.at("host_reserve_bytes").get<int64_t>();
  s.host_reserve_evidence = setting(r, "memory.host_reserve_mib").source;
  int64_t required = 0;
  if (p1.contains("gpu") && p1.at("gpu").is_array())
    for (const json &g : p1.at("gpu"))
      if (integer(g, "required_bytes")) required = std::max(required, g.at("required_bytes").get<int64_t>());
  s.engine_vram_required_bytes = required;
  s.engine_vram_claimable_bytes = m.spec.plan.engine_memory->expected.vram.claimable;
  s.checkpoint_slots = m.spec.resources.host.checkpoint_slots.value_or(0);

  if (has_helpers) {
    const auto &handoff = *m.spec.handoff;
    s.transport_shm = handoff.transport.shm;
    if (handoff.telemetry.histogram_file && !handoff.telemetry.histogram_file->empty())
      s.histogram_file = handoff.telemetry.histogram_file;
    s.live_policy_requested = live;
    // The worker's lanes are sized for the resolver's max_rows (worker_footprint.h): that is its dispatch capacity.
    s.dispatch_row_capacity = setting(r, "handoff.max_rows").value;
    for (const char *key : {"handoff.max_rows", "handoff.small_rows", "handoff.gemv_large_rows",
                            "handoff.moe_down_tile", "handoff.histogram_period_s"})
      s.settings[key] = setting(r, key);
    if (live)
      s.settings["handoff.live_cadence_ns"] = {placement->live_policy->cadence_ns, "spec.handoff.placement.live_policy"};
  }

  if (w.calibration_records.is_array())
    for (const json &record : w.calibration_records) {
      auto c = parse_calibration_record(record);
      if (c) in.calibrations.push_back(c.value());   // an unreadable record is evidence we cannot use, not a refusal
    }
  return in;
}

} // namespace bc::serving::v2
