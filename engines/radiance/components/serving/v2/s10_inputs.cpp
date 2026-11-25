#include "v2/s10_inputs.h"

#include "serving_codec.h"

#include <algorithm>
#include <regex>

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

const std::string &embedded_power_profiles_yaml() {
  static const std::string text =
#include "power_profiles_seed.inc"
      ;
  return text;
}

Result<PowerProfiles> decode_power_profiles(const json &doc) {
  const auto bad = [](std::string path, std::string why) {
    return make_error(ConfigErrorCode::invalid_value, std::move(path), std::move(why),
                      "fix engines/radiance/power-profiles.yaml");
  };
  const auto closed = [&](const json &o, const std::string &path, std::initializer_list<const char *> keys) -> Status {
    if (!o.is_object()) return bad(path, "expected a mapping");
    for (auto it = o.begin(); it != o.end(); ++it)
      if (std::find_if(keys.begin(), keys.end(), [&](const char *k) { return it.key() == k; }) == keys.end())
        return bad(path + "." + it.key(), "unknown key");
    return std::nullopt;
  };
  if (auto e = closed(doc, "power-profiles", {"schema", "profiles"})) return *e;
  if (!doc.contains("schema") || doc.at("schema") != 1 || !doc.contains("profiles"))
    return bad("power-profiles", "expected schema: 1 and a profiles mapping");
  static const std::regex name_form("^[a-z][a-z0-9_-]{0,31}$"), arch_form("^gfx[0-9a-f]+$"),
      role_form("^(engine|helper\\*|helper[0-9]+)$");
  PowerProfiles out;
  for (auto p = doc.at("profiles").begin(); p != doc.at("profiles").end(); ++p) {
    const std::string at = "profiles." + p.key();
    if (!std::regex_match(p.key(), name_form)) return bad(at, "invalid profile name");
    if (auto e = closed(p.value(), at, {"default_mode", "rules"})) return *e;
    if (p.value().contains("default_mode") && p.value().at("default_mode") != "observe")
      return bad(at + ".default_mode", "a named profile observes; only an explicit operator apply writes");
    S10PowerProfile profile;
    profile.name = p.key();
    if (p.value().contains("rules")) {
      const json &rules = p.value().at("rules");
      if (!rules.is_array()) return bad(at + ".rules", "expected a list");
      for (std::size_t i = 0; i < rules.size(); ++i) {
        const std::string rat = at + ".rules[" + std::to_string(i) + "]";
        if (auto e = closed(rules[i], rat, {"role", "arch", "evidence", "settings"})) return *e;
        for (const char *key : {"role", "arch", "evidence", "settings"})
          if (!rules[i].contains(key)) return bad(rat, std::string("needs ") + key);
        S10PowerRule rule;
        if (!rules[i].at("role").is_string() || !std::regex_match(rules[i].at("role").get<std::string>(), role_form))
          return bad(rat + ".role", "engine, helperN or helper*");
        if (!rules[i].at("arch").is_string() || !std::regex_match(rules[i].at("arch").get<std::string>(), arch_form))
          return bad(rat + ".arch", "a gfx architecture");
        if (!rules[i].at("evidence").is_string() || rules[i].at("evidence").get<std::string>().empty() ||
            rules[i].at("evidence").get<std::string>().rfind("/", 0) == 0)
          return bad(rat + ".evidence", "missing or host-specific evidence");
        rule.role = rules[i].at("role").get<std::string>();
        rule.arch = rules[i].at("arch").get<std::string>();
        rule.evidence = rules[i].at("evidence").get<std::string>();
        const json &settings = rules[i].at("settings");
        if (auto e = closed(settings, rat + ".settings",
                            {"vddgfx_offset_mv", "sclk_max_mhz", "mclk_max_mhz", "sclk_offset_mhz", "power_cap_w"}))
          return *e;
        if (settings.empty()) return bad(rat + ".settings", "a rule needs at least one setting");
        for (auto s = settings.begin(); s != settings.end(); ++s)
          if (!s.value().is_number_integer()) return bad(rat + ".settings." + s.key(), "expected an integer");
        const auto optional = [&](const char *key) -> std::optional<int64_t> {
          return settings.contains(key) ? std::optional<int64_t>(settings.at(key).get<int64_t>()) : std::nullopt;
        };
        rule.settings.vddgfx_offset_mv = optional("vddgfx_offset_mv");
        rule.settings.sclk_max_mhz = optional("sclk_max_mhz");
        rule.settings.mclk_max_mhz = optional("mclk_max_mhz");
        rule.settings.sclk_offset_mhz = optional("sclk_offset_mhz");
        rule.settings.power_cap_w = optional("power_cap_w");
        profile.rules.push_back(std::move(rule));
      }
    }
    out.emplace(p.key(), std::move(profile));
  }
  if (!out.count("stock") || !out.at("stock").rules.empty())
    return bad("profiles.stock", "stock must exist and have no settings rules");
  return out;
}

Result<PowerSite> decode_power_site(const json &doc) {
  if (doc.is_null()) return PowerSite{};
  const auto bad = [](std::string path, std::string why) {
    return make_error(ConfigErrorCode::invalid_value, std::move(path), std::move(why), "fix this host's power-site.yaml");
  };
  const auto closed = [&](const json &o, const std::string &path, std::initializer_list<const char *> keys) -> Status {
    if (!o.is_object()) return bad(path, "expected a mapping");
    for (auto it = o.begin(); it != o.end(); ++it)
      if (std::find_if(keys.begin(), keys.end(), [&](const char *k) { return it.key() == k; }) == keys.end())
        return bad(path + "." + it.key(), "unknown key");
    return std::nullopt;
  };
  if (auto e = closed(doc, "power-site", {"schema", "devices"})) return *e;
  if (!doc.contains("schema") || doc.at("schema") != 1 || !doc.contains("devices") || !doc.at("devices").is_object())
    return bad("power-site", "expected schema: 1 and a devices mapping");
  static const std::regex bdf_form("^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\\.[0-7]$");
  PowerSite out;
  for (auto d = doc.at("devices").begin(); d != doc.at("devices").end(); ++d) {
    const std::string at = "power-site.devices." + d.key();
    if (!std::regex_match(d.key(), bdf_form)) return bad(at, "expected a lower-case PCI BDF such as 0000:03:00.0");
    if (auto e = closed(d.value(), at, {"settings", "evidence"})) return *e;
    if (!d.value().contains("settings") || !d.value().contains("evidence") || !d.value().at("evidence").is_string() ||
        d.value().at("evidence").get<std::string>().empty())
      return bad(at, "needs settings and a non-empty evidence string");
    const json &settings = d.value().at("settings");
    if (auto e = closed(settings, at + ".settings", {"vddgfx_offset_mv", "sclk_max_mhz", "mclk_max_mhz", "power_cap_w"}))
      return *e;
    if (settings.empty()) return bad(at + ".settings", "a device needs at least one setting");
    for (auto s = settings.begin(); s != settings.end(); ++s)
      if (!s.value().is_number_integer()) return bad(at + ".settings." + s.key(), "expected an integer");
    const auto optional = [&](const char *key) -> std::optional<int64_t> {
      return settings.contains(key) ? std::optional<int64_t>(settings.at(key).get<int64_t>()) : std::nullopt;
    };
    PowerPatch value;
    value.vddgfx_offset_mv = optional("vddgfx_offset_mv");
    value.sclk_max_mhz = optional("sclk_max_mhz");
    value.mclk_max_mhz = optional("mclk_max_mhz");
    value.power_cap_w = optional("power_cap_w");
    out.emplace(d.key(), value);
  }
  return out;
}

Result<S10Inputs> s10_inputs(const Manifest &m, const bc::serving::Resolved &r, const WriterInputs &w) {
  S10Inputs in;
  in.now_unix_s = w.now_unix_s;
  in.profile.name = w.power_profile;
  // stock is built in and writes nothing; any other name must come from the loaded profile file with its rules.
  if (w.power_profile != "stock") {
    if (w.power_rules.name != w.power_profile)
      return fail("spec.intent.power.profile",
                  "power profile " + w.power_profile + " is not in engines/radiance/power-profiles.yaml");
    in.profile = w.power_rules;
  }
  // A named profile observes unless the operator explicitly asked to apply (--allow-power-apply); stock never writes.
  in.mode = w.power_apply && w.power_profile != "stock" ? PowerMode::kApply : PowerMode::kObserve;
  in.power = w.power_evidence;
  for (const auto &[bdf, settings] : in.power.site_overrides)
    if (std::none_of(m.spec.facts.gpus.begin(), m.spec.facts.gpus.end(), [&](const GpuFact &g) { return g.bdf == bdf; }))
      return fail("site.power", bdf + " is not a detected GPU of this host");

  const bool has_helpers = !m.spec.processes.helpers.empty();
  const auto &placement = m.spec.handoff ? std::optional(m.spec.handoff->placement) : std::nullopt;
  const bool live = placement && placement->live_policy.has_value();
  const bool calibrating = placement && placement->live_calibration.has_value();
  in.preflight.requested = !has_helpers ? LivePolicyRequested::kNotApplicable
                           : live        ? LivePolicyRequested::kOn
                           : calibrating ? LivePolicyRequested::kCalibrating
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
    return fail("p1_record", "a measured manifest needs the native P1 measurement record (the session-owned measuring start)");
  S8Measured &s = in.measured;
  s.engine_ranks = m.spec.processes.engine.scheduler.tp;
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
  // Engine VRAM admission is per rank, each against its own live claimable room after the rank split: S8 checks the
  // rank with the least headroom.
  bool any_rank = false;
  if (p1.contains("gpu") && p1.at("gpu").is_array())
    for (const json &g : p1.at("gpu")) {
      if (!integer(g, "required_bytes") || !integer(g, "claimable_bytes"))
        return fail("p1.gpu", "each engine rank needs its required and claimable bytes");
      const int64_t required = g.at("required_bytes").get<int64_t>(), claimable = g.at("claimable_bytes").get<int64_t>();
      if (!any_rank || claimable - required < s.engine_vram_claimable_bytes - s.engine_vram_required_bytes) {
        s.engine_vram_required_bytes = required;
        s.engine_vram_claimable_bytes = claimable;
      }
      any_rank = true;
    }
  if (!any_rank) return fail("p1.gpu", "the P1 record has no engine rank");
  s.checkpoint_slots = m.spec.resources.host.checkpoint_slots.value_or(0);
  s.checkpoint_slot_bytes = m.spec.plan.engine_memory->expected.checkpoints.slot_bytes;

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
