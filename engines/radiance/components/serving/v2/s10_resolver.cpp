#include "v2/s10_resolver.h"

#include "generated/v2/serving_codec.h"
#include "jcs.h"
#include "sha256.h"
#include "v2/strict_json.h"

#include <algorithm>
#include <limits>
#include <set>
#include <utility>

namespace bc::serving::v2 {
namespace {
using json = nlohmann::json;
constexpr int64_t kSafe = 9007199254740991LL;

ConfigError fail(ConfigErrorCode code, std::string path, std::string why) {
  return make_error(code, std::move(path), std::move(why));
}
bool safe(int64_t v) { return v >= -kSafe && v <= kSafe; }
bool populated(const PowerSettings &s) {
  return s.vddgfx_offset_mv || s.sclk_max_mhz || s.sclk_offset_mhz ||
         s.power_cap_w;
}
PowerSettings patch(const PowerSettings &base, const PowerPatch &p) {
  PowerSettings result = base;
  if (p.vddgfx_offset_mv) result.vddgfx_offset_mv = p.vddgfx_offset_mv;
  if (p.sclk_max_mhz) result.sclk_max_mhz = p.sclk_max_mhz;
  if (p.sclk_offset_mhz) result.sclk_offset_mhz = p.sclk_offset_mhz;
  if (p.power_cap_w) result.power_cap_w = p.power_cap_w;
  return result;
}
bool populated(const PowerPatch &p) {
  return p.vddgfx_offset_mv || p.sclk_max_mhz || p.sclk_offset_mhz ||
         p.power_cap_w;
}
Status valid_settings(const PowerSettings &s, const std::string &path) {
  if ((s.vddgfx_offset_mv && !safe(*s.vddgfx_offset_mv)) ||
      (s.sclk_offset_mhz && !safe(*s.sclk_offset_mhz)) ||
      (s.sclk_max_mhz && (*s.sclk_max_mhz <= 0 || !safe(*s.sclk_max_mhz))) ||
      (s.power_cap_w && (*s.power_cap_w <= 0 || !safe(*s.power_cap_w))) ||
      (s.sclk_offset_mhz && s.sclk_max_mhz))
    return fail(ConfigErrorCode::invalid_value, path,
                "invalid typed power values or mutually exclusive SCLK controls");
  return std::nullopt;
}
json settings_json(const PowerSettings &s) {
  json j = json::object();
  if (s.vddgfx_offset_mv) j["vddgfx_offset_mv"] = *s.vddgfx_offset_mv;
  if (s.sclk_max_mhz) j["sclk_max_mhz"] = *s.sclk_max_mhz;
  if (s.sclk_offset_mhz) j["sclk_offset_mhz"] = *s.sclk_offset_mhz;
  if (s.power_cap_w) j["power_cap_w"] = *s.power_cap_w;
  return j;
}
bool same_settings(const PowerSettings &a, const PowerSettings &b) {
  return settings_json(a) == settings_json(b);
}
struct Target {
  std::string role, bdf, arch;
};
Result<std::vector<Target>> targets(const Manifest &m) {
  std::map<std::string, std::string> arch;
  for (const auto &gpu : m.spec.facts.gpus)
    if (gpu.bdf.empty() || gpu.arch.empty() ||
        !arch.emplace(gpu.bdf, gpu.arch).second)
      return fail(ConfigErrorCode::ownership_mismatch, "spec.facts.gpus",
                  "duplicate or incomplete GPU inventory");
  std::vector<Target> out;
  std::set<std::string> bdfs;
  const auto add = [&](const std::string &role, const std::string &bdf)
      -> Status {
    if (!bdfs.insert(bdf).second || !arch.count(bdf))
      return fail(ConfigErrorCode::ownership_mismatch, "spec.plan.power.devices",
                  "unknown or duplicate engine/helper BDF: " + bdf);
    out.push_back({role, bdf, arch.at(bdf)});
    return std::nullopt;
  };
  for (const auto &bdf : m.spec.processes.engine.device.gpus)
    if (auto error = add("engine", bdf)) return *error;
  if (m.spec.processes.engine.device.gpus.empty())
    return fail(ConfigErrorCode::ownership_mismatch,
                "spec.processes.engine.device.gpus", "empty engine TP set");
  for (size_t i = 0; i < m.spec.processes.helpers.size(); ++i) {
    const auto &h = m.spec.processes.helpers[i];
    if (h.role != "helper" + std::to_string(i))
      return fail(ConfigErrorCode::ownership_mismatch,
                  "spec.processes.helpers", "helper roles must match selection order");
    if (auto error = add(h.role, h.gpu)) return *error;
  }
  return out;
}
Result<PowerPlan> power(const Manifest &m, const S10Inputs &in,
                       const std::vector<Target> &selected) {
  const auto &name = m.spec.intent.power.profile;
  if (name != in.profile.name || name.empty())
    return fail(ConfigErrorCode::plan_mismatch, "spec.plan.power.profile",
                "named power selection differs from intent");
  if (name == "stock" && (!in.profile.rules.empty() ||
                          in.mode != PowerMode::kObserve))
    return fail(ConfigErrorCode::invalid_value, "spec.plan.power.mode",
                "stock is an empty observe-only plan");
  std::set<std::string> roles;
  for (const auto &t : selected) roles.insert(t.role);
  for (const auto &[role, patch_value] : m.spec.intent.power.role_overrides) {
    if (!roles.count(role) || !populated(patch_value))
      return fail(ConfigErrorCode::invalid_value, "spec.intent.power.role_overrides",
                  "unknown role or empty override: " + role);
  }
  for (const auto &rule : in.profile.rules) {
    if (rule.evidence.empty() || rule.arch.empty() ||
        !(rule.role == "engine" || rule.role == "helper*" ||
          roles.count(rule.role)) ||
        valid_settings(rule.settings, "power.profile.rules"))
      return fail(ConfigErrorCode::invalid_value, "power.profile.rules",
                  "invalid named rule, empty evidence or nonselected role");
  }
  std::map<std::string, PowerSettings> resolved_roles;
  std::vector<PowerDevice> devices;
  for (const auto &target : selected) {
    PowerSettings setting{};
    // A wildcard helper rule is overlaid by a more-specific role rule.
    for (const auto &rule : in.profile.rules)
      if (rule.role == "helper*" && target.role != "engine" &&
          rule.arch == target.arch) {
        PowerPatch p{rule.settings.vddgfx_offset_mv, rule.settings.sclk_max_mhz,
                     rule.settings.sclk_offset_mhz, rule.settings.power_cap_w};
        setting = patch(setting, p);
      }
    for (const auto &rule : in.profile.rules)
      if (rule.role == target.role && rule.arch == target.arch) {
        PowerPatch p{rule.settings.vddgfx_offset_mv, rule.settings.sclk_max_mhz,
                     rule.settings.sclk_offset_mhz, rule.settings.power_cap_w};
        setting = patch(setting, p);
      }
    const auto [it, inserted] = resolved_roles.emplace(target.role, setting);
    if (!inserted && !same_settings(it->second, setting))
      return fail(ConfigErrorCode::plan_mismatch, "spec.plan.power",
                  "TP engine ranks disagree on architecture-specific role settings");
    devices.push_back({target.role, target.bdf, setting});
  }

  // Named-profile digest excludes CLI overrides, just as the PWR08 Python
  // profile digest does; sorted JSON integer objects are JCS.
  json roles_json = json::object();
  for (const auto &[role, s] : resolved_roles) roles_json[role] = settings_json(s);
  json digest_input = {{"name", name}, {"roles", roles_json}};
  std::string canonical, why;
  if (!bc::serving::jcs_canonicalize(digest_input, &canonical, &why))
    return fail(ConfigErrorCode::invalid_format, "spec.plan.power.profile_digest", why);
  PowerPlan plan{};
  plan.profile = name;
  plan.mode = in.mode;
  plan.profile_digest = "sha256:" + bc::serving::sha256_hex(canonical);
  for (auto &dev : devices) {
    const auto &target = *std::find_if(selected.begin(), selected.end(),
        [&](const Target &t) { return t.bdf == dev.bdf; });
    auto it = m.spec.intent.power.role_overrides.find(dev.role);
    if (it != m.spec.intent.power.role_overrides.end())
      dev.settings = patch(dev.settings, it->second);
    if (auto error = valid_settings(dev.settings, "spec.plan.power.devices"))
      return *error;
    // A voltage-offset-only gfx1100 plan is unsafe even in observe mode.
    if (target.arch == "gfx1100" && dev.settings.vddgfx_offset_mv &&
        !dev.settings.sclk_max_mhz)
      return fail(ConfigErrorCode::invalid_value, "spec.plan.power.devices",
                  "gfx1100 VDDGFX offset requires a paired SCLK maximum");
    if (in.mode == PowerMode::kObserve) {
      dev.settings = {};
    } else if (populated(dev.settings)) {
      // The PowerTransaction backend does not implement SCLK offsets yet.
      if (dev.settings.sclk_offset_mhz)
        return fail(ConfigErrorCode::invalid_value, "spec.plan.power.devices",
                    "SCLK offset has no transaction/restore backend");
      auto q = in.qualifications.find(dev.bdf);
      auto p = in.probes.find(dev.bdf);
      if (q == in.qualifications.end() || p == in.probes.end() ||
          !q->second.qualified || q->second.bdf != dev.bdf ||
          q->second.arch != target.arch || q->second.profile != name ||
          q->second.evidence.empty() || q->second.build_id.empty() ||
          q->second.driver_id.empty() ||
          !same_settings(q->second.settings, dev.settings) ||
          p->second.bdf != dev.bdf || p->second.driver != "amdgpu")
        return fail(ConfigErrorCode::calibration_invalid, "spec.plan.power.devices",
                    "apply requires per-BDF evidence, build/driver identity and fresh probe");
      const auto values = settings_json(dev.settings);
      for (auto it = values.begin(); it != values.end(); ++it) {
        const auto ctl = p->second.controls.find(it.key());
        const int64_t raw = it.value().get<int64_t>();
        if (it.key() == "power_cap_w" && raw > kSafe / 1000000LL)
          return fail(ConfigErrorCode::invalid_value, "spec.plan.power.devices",
                      "power cap overflows probe micro-watt units");
        const int64_t value = raw *
            (it.key() == "power_cap_w" ? 1000000LL : 1LL);
        if (ctl == p->second.controls.end() || !ctl->second.supported ||
            !ctl->second.original_readable ||
            value < ctl->second.minimum || value > ctl->second.maximum)
          return fail(ConfigErrorCode::invalid_value, "spec.plan.power.devices",
                      dev.bdf + ": requested setting outside PWR07 probed range");
      }
    }
    plan.devices.push_back(std::move(dev));
  }
  return plan;
}
LivePolicyStartup startup(const S10PolicyPreflight &p,
                          const S8Resolution &s8) {
  LivePolicyStartup out{};
  out.requested = p.requested;
  out.configured = false;
  out.calibration_key = p.qualified_record ? p.calibration_key : "";
  out.record_identity = p.qualified_record ? p.record_identity : "";
  out.reason_detail = p.reason_detail;
  out.effect_gate = p.effect_gate;
  out.histogram_period_s = s8.telemetry ? s8.telemetry->period_s : 0;
  out.cadence_ns = s8.live_cadence_ns.value_or(0);
  out.max_heat_age_ns = p.max_heat_age_ns;
  if (!p.applicable || p.requested == LivePolicyRequested::kNotApplicable) {
    out.requested = LivePolicyRequested::kNotApplicable;
    out.reason_code = "not_applicable";
  } else if (p.requested == LivePolicyRequested::kOff) {
    out.reason_code = "disabled_by_request";
  } else if (!p.qualified_record) {
    out.reason_code = "no_qualified_record";
  } else if (!p.fresh_heat) {
    out.reason_code = "stale_heat_feed";
  } else {
    out.reason_code = "configured";
    out.configured = true;
  }
  return out;
}
bool allowed(std::string_view csv, std::string_view value) {
  if (csv.empty()) return true;
  const std::string list = "," + std::string(csv) + ",";
  return list.find("," + std::string(value) + ",") != std::string::npos;
}
std::string_view origin_name(Origin origin) {
  switch (origin) {
  case Origin::kDerived: return "derived";
  case Origin::kCalibrated: return "calibrated";
  case Origin::kProfile: return "profile";
  case Origin::kSchemaDefault: return "schema_default";
  case Origin::kSite: return "site";
  case Origin::kCli: return "cli";
  case Origin::kDetected: return "detected";
  case Origin::kExperiment: return "experiment";
  }
  return "";
}
std::string_view confidence_name(Confidence c) {
  switch (c) {
  case Confidence::kExact: return "exact";
  case Confidence::kMeasured: return "measured";
  case Confidence::kEstimate: return "estimate";
  }
  return "";
}
void present_paths(const json &doc, std::string_view field, size_t start,
                   const std::string &prefix, std::vector<std::string> *out) {
  if (start == field.size()) {
    out->push_back(prefix);
    return;
  }
  if (field[start] != '/') return;
  const size_t next = field.find('/', start + 1);
  const size_t end = next == std::string_view::npos ? field.size() : next;
  const auto segment = field.substr(start + 1, end - start - 1);
  const auto *node = &doc;
  if (!prefix.empty()) {
    try { node = &doc.at(json::json_pointer(prefix)); }
    catch (const json::exception &) { return; }
  }
  if (segment == "[]") {
    if (!node->is_array()) return;
    for (size_t i = 0; i < node->size(); ++i)
      present_paths(doc, field, end, prefix + "/" + std::to_string(i), out);
  } else if (node->is_object() && node->contains(std::string(segment))) {
    present_paths(doc, field, end, prefix + "/" + std::string(segment), out);
  }
}
} // namespace

Status validate_s10_provenance(const Manifest &m) {
  if (m.provenance.empty() || !m.provenance.count(""))
    return fail(ConfigErrorCode::provenance_invalid, "provenance",
                "root provenance and covering evidence required");
  const auto doc = encode(m);
  for (const auto &field : kFields) {
    if (field.path.empty() || field.path[0] != '/' ||
        (field.allowed_origins.empty() && field.allowed_confidences.empty()))
      continue;
    std::vector<std::string> paths;
    present_paths(doc, field.path, 0, "", &paths);
    for (const auto &path : paths) {
      // A child's own record wins over its nearest inherited ancestor.
      auto at = path;
      auto found = m.provenance.end();
      while (true) {
        found = m.provenance.find(at);
        if (found != m.provenance.end() || at.empty()) break;
        const auto sep = at.find_last_of('/');
        at = sep == 0 ? "" : at.substr(0, sep);
      }
      if (found == m.provenance.end() || found->second.source.empty() ||
          !allowed(field.allowed_origins, origin_name(found->second.origin)) ||
          !allowed(field.allowed_confidences,
                   confidence_name(found->second.confidence)))
        return fail(ConfigErrorCode::provenance_invalid, path,
                    "effective provenance is not measured/derived or calibrated");
    }
  }
  return std::nullopt;
}

Result<Manifest> resolve_s10(const Manifest &base, const S10Inputs &in) {
  if (base.api_version != "bigcherry/v2" || base.metadata.contract_version != 2)
    return fail(ConfigErrorCode::unsupported_version, "apiVersion",
                "S10 only resolves the inactive v2 typed manifest");
  const auto selected = targets(base);
  if (!selected) return selected.error();
  auto power_plan = power(base, in, selected.value());
  if (!power_plan) return power_plan.error();
  if (base.spec.plan.owner_map.owner_map_sha256 !=
      in.measured.actual_owner_map_digest)
    return fail(ConfigErrorCode::ownership_mismatch, "spec.plan.owner_map",
                "actual engine owner map differs from v2 plan");
  auto result = resolve_s8(in.measured, in.calibrations, in.now_unix_s);
  if (!result) return result.error(); // S8 admission; S5/S11 own structured refusal
  const auto &s8 = result.value();
  Manifest out = base;
  out.spec.plan.power = std::move(power_plan.value());
  out.spec.plan.resource_inputs = s8.resource_inputs;
  out.spec.resources.host = s8.host;
  // Keep unrelated qualified calibration keys; replace only the S8 keys.
  // No second handoff/margin policy or duplicate calibration entry is emitted.
  for (const auto &entry : s8.calibration_entries) {
    const auto same = [&](const CalibrationEntry &existing) {
      return existing.key == entry.key;
    };
    auto it = std::find_if(out.spec.calibration.begin(),
                           out.spec.calibration.end(), same);
    if (it == out.spec.calibration.end())
      out.spec.calibration.push_back(entry);
    else
      *it = entry;
  }
  if (out.spec.plan.state == PlanState::kMeasured) {
    if (!out.spec.plan.engine_memory ||
        out.spec.plan.engine_memory->expected.host_pool_mib != s8.host.pool_mib ||
        out.spec.plan.engine_memory->expected.vram.claimable !=
            in.measured.engine_vram_claimable_bytes)
      return fail(ConfigErrorCode::plan_mismatch, "spec.plan.engine_memory.expected",
                  "measured engine host-pool or live VRAM plan disagrees with S8");
  }
  if (out.spec.processes.helpers.size() != s8.helpers.size())
    return fail(ConfigErrorCode::ownership_mismatch, "spec.processes.helpers",
                "helper count differs from measured worker footprints");
  for (size_t i = 0; i < s8.helpers.size(); ++i) {
    auto &helper = out.spec.processes.helpers[i];
    const auto &budget = s8.helpers[i];
    if (helper.role != budget.role || helper.gpu != budget.bdf)
      return fail(ConfigErrorCode::ownership_mismatch, "spec.processes.helpers",
                  "helper budget role/BDF differs from v2 process");
    helper.vram_budget_mib = budget.expert_budget_mib;
  }
  if (s8.transport) {
    if (!out.spec.handoff)
      return fail(ConfigErrorCode::missing_field, "spec.handoff.placement",
                  "remote helper placement must be supplied by engine preflight");
    out.spec.handoff->transport = *s8.transport;
    out.spec.handoff->kernels = *s8.kernels;
    out.spec.handoff->telemetry = *s8.telemetry;
    const auto &p = in.preflight;
    const bool applicable = s8.helpers.size() > 0 &&
        out.spec.processes.engine.device.gpus.size() == 1;
    if (p.applicable != applicable ||
        (p.requested == LivePolicyRequested::kOn) !=
            in.measured.live_policy_requested)
      return fail(ConfigErrorCode::plan_mismatch,
                  "spec.handoff.placement.live_policy_state",
                  "policy preflight differs from measured layout/request");
    const auto decision = startup(p, s8);
    if (decision.configured &&
        (p.calibration_key.empty() || p.record_identity.empty() ||
         p.max_heat_age_ns <= 0 || !s8.live_cadence_ns ||
         !out.spec.handoff->placement.live_policy ||
         out.spec.handoff->placement.live_policy->cadence_ns != decision.cadence_ns ||
         out.spec.handoff->placement.live_policy->max_heat_age_ns !=
             p.max_heat_age_ns ||
         out.spec.handoff->placement.live_policy->effect_gate != p.effect_gate))
      return fail(ConfigErrorCode::calibration_invalid,
                  "spec.handoff.placement.live_policy_state",
                  "qualified policy/heat preflight disagrees with live policy");
    out.spec.handoff->placement.live_policy_state = decision;
    if (!decision.configured)
      out.spec.handoff->placement.live_policy.reset();
  } else if (out.spec.handoff) {
    return fail(ConfigErrorCode::plan_mismatch, "spec.handoff",
                "engine-only plan cannot have remote handoff");
  }
  for (const auto &[path, record] : s8.provenance_by_path)
    out.provenance[path] = record; // measured evidence overrides inherited defaults
  out.provenance["/spec/plan/power"] = {
      Origin::kDerived, "PWR08 named profile + selected BDF inventory",
      ValueClass::kDerived, Lifecycle::kStartupOnly, Confidence::kExact,
      {"/spec/intent/power", "/spec/facts/gpus"}};
  if (out.spec.handoff)
    out.provenance["/spec/handoff/placement/live_policy_state"] = {
        Origin::kDerived, "RHP18 qualified policy preflight",
        ValueClass::kDerived, Lifecycle::kStartupOnly, Confidence::kExact,
        {"/spec/calibration", "/spec/handoff/telemetry"}};
  if (auto error = validate_s10_provenance(out)) return *error;
  return out;
}

Result<std::string> write_s10(const Manifest &m) {
  if (m.api_version != "bigcherry/v2" || m.metadata.contract_version != 2)
    return fail(ConfigErrorCode::unsupported_version, "apiVersion",
                "writer accepts only v2");
  if (auto error = validate_s10_provenance(m)) return *error;
  auto json_doc = encode(m);
  std::string digest, why;
  if (!bc::serving::manifest_content_digest(json_doc, &digest, &why))
    return fail(ConfigErrorCode::invalid_format, "metadata.content_digest", why);
  json_doc["metadata"]["content_digest"] = digest;
  Manifest decoded;
  if (auto error = decode(json_doc, "", &decoded)) return *error;
  return json_doc.dump();
}
} // namespace bc::serving::v2
