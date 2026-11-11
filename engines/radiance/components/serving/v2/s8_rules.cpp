#include "v2/s8_rules.h"

#include <algorithm>
#include <limits>
#include <set>
#include <sstream>
#include <utility>

namespace bc::serving::v2 {
namespace {
constexpr int64_t kMiB = 1024 * 1024; // unit, not a resource allowance
constexpr int64_t kMaxWireInteger = 9007199254740991LL;

ConfigError bad(ConfigErrorCode code, std::string path, std::string why) {
  return make_error(code, std::move(path), std::move(why));
}
bool count(int64_t n) { return n >= 0 && n <= kMaxWireInteger; }
bool positive(int64_t n) { return n > 0 && n <= kMaxWireInteger; }
bool add(int64_t a, int64_t b, int64_t *out) {
  if (!count(a) || !count(b) || a > kMaxWireInteger - b)
    return false;
  *out = a + b;
  return true;
}
bool mul(int64_t a, int64_t b, int64_t *out) {
  if (!count(a) || !count(b) || (b && a > kMaxWireInteger / b))
    return false;
  *out = a * b;
  return true;
}
int64_t ceil_mib(int64_t value) { return value / kMiB + (value % kMiB != 0); }
bool digest(const std::string &s) {
  if (s.size() != 64)
    return false;
  return std::all_of(s.begin(), s.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}
bool iso_date(const std::string &d) {
  if (d.size() != 10 || d[4] != '-' || d[7] != '-')
    return false;
  for (size_t i = 0; i < d.size(); ++i)
    if (i != 4 && i != 7 && (d[i] < '0' || d[i] > '9'))
      return false;
  return d.substr(5, 2) >= "01" && d.substr(5, 2) <= "12" &&
         d.substr(8, 2) >= "01" && d.substr(8, 2) <= "31";
}
bool valid_scope(const S8Scope &s) {
  return digest(s.model_digest) && digest(s.engine_build_digest) &&
         digest(s.helper_build_digest) && digest(s.layout_digest) &&
         digest(s.inventory_digest) && digest(s.owner_map_digest) &&
         positive(s.context_target_tokens) &&
         positive(s.retention_target_tokens) && positive(s.kv_bytes_per_token);
}

// The registry class of a leaf (contract [[class_default]]), so a record covers exactly its own class.
ValueClass class_of(const std::string &path) {
  for (const auto &f : kFields)
    if (f.path == path)
      for (size_t i = 0; i < kValueClassNames.size(); ++i)
        if (kValueClassNames[i] == f.value_class) return static_cast<ValueClass>(i);
  return ValueClass::kDerived;
}
} // namespace

Status require_measured_vram(int64_t required, int64_t claimable,
                             const std::string &bdf) {
  if (!positive(required) || !positive(claimable))
    return bad(
        ConfigErrorCode::invalid_value, "spec.plan.engine_memory.expected.vram",
        "live VRAM required/claimable must be measured positive bytes for " +
            bdf);
  if (required > claimable) {
    std::ostringstream message;
    message << bdf << ": " << required << " required bytes exceed " << claimable
            << " live claimable bytes by " << (required - claimable)
            << "; recalibrate the live VRAM reserve or change placement";
    return bad(ConfigErrorCode::admission_refused,
               "spec.plan.engine_memory.expected.vram.claimable",
               message.str());
  }
  return std::nullopt;
}

Result<S8Resolution> resolve_s8(const S8Measured &m,
                                const std::vector<S8Calibration> &records,
                                int64_t now) {
  const auto fail = [](ConfigErrorCode c, const std::string &path,
                       const std::string &why) -> Result<S8Resolution> {
    return bad(c, path, why);
  };
  const auto &r = m.resource_inputs;
  const auto scope_path = std::string("spec.plan.resource_inputs");
  if (!valid_scope(m.scope) || m.scope.owner_map_digest != r.owner_map_digest ||
      m.scope.owner_map_digest != m.actual_owner_map_digest ||
      m.scope.context_target_tokens != r.context_target_tokens ||
      m.scope.retention_target_tokens != r.retention_target_tokens ||
      m.scope.kv_bytes_per_token != r.kv_bytes_per_token) {
    return fail(ConfigErrorCode::ownership_mismatch, scope_path,
                "engine/owner-map/model/service measurement scope is missing "
                "or has changed");
  }
  if (now <= 0 || now > kMaxWireInteger)
    return fail(ConfigErrorCode::invalid_value, "calibration.now_unix_s",
                "invalid time");
  for (const auto v : {r.host_embedding_bytes, r.unowned_expert_bytes,
                       r.staging_bytes, r.ngram_bytes, r.checkpoint_bytes,
                       r.resident_other_bytes, r.mem_available_bytes,
                       r.cgroup_headroom_bytes, r.pin_available_bytes}) {
    if (!count(v))
      return fail(
          ConfigErrorCode::invalid_value, scope_path,
          "all resident/live byte counts must be measured I-JSON integers");
  }
  const bool has_helpers = !m.helpers.empty();
  const bool telemetry_on = m.histogram_file && !m.histogram_file->empty();
  if ((!has_helpers && (telemetry_on || m.live_policy_requested)) ||
      (m.live_policy_requested && !telemetry_on))
    return fail(ConfigErrorCode::invalid_value, "spec.handoff",
                "live policy/histogram needs a real remote helper and active "
                "histogram");
  if (!positive(r.mem_available_bytes) || !positive(r.cgroup_headroom_bytes) ||
      !positive(r.pin_available_bytes) || !count(m.host_reserve_bytes) ||
      m.host_reserve_evidence.empty() || !count(m.checkpoint_slots) ||
      (has_helpers &&
       (!positive(m.dispatch_row_capacity) || m.transport_shm.empty()))) {
    return fail(ConfigErrorCode::invalid_value, scope_path,
                "missing host reserve provenance, checkpoint/dispatch "
                "measurement or live room");
  }
  if (auto e = require_measured_vram(m.engine_vram_required_bytes,
                                     m.engine_vram_claimable_bytes, "engine"))
    return *e;

  const auto &cal = r.host_pool_margin_calibration;
  if (!count(cal.mib))
    return fail(ConfigErrorCode::invalid_value,
                scope_path + ".host_pool_margin_calibration.mib",
                "host-ledger margin must be a non-negative integer");

  // The settings S8 places: each has the caller's value and its source (site, layout, calibration seed). A calibration
  // record replaces it when one matches this measurement's scope and is current; a record that does not is skipped and
  // the skip is said in the provenance. Records are evidence, never a precondition: no record still starts.
  std::set<std::string> wanted;
  if (has_helpers)
    wanted.insert({"handoff.max_rows", "handoff.small_rows",
                   "handoff.gemv_large_rows", "handoff.moe_down_tile"});
  if (telemetry_on)
    wanted.insert("handoff.histogram_period_s");
  if (m.live_policy_requested)
    wanted.insert("handoff.live_cadence_ns");
  struct Chosen {
    int64_t value = 0;
    std::string source;
    const S8Calibration *row = nullptr;
  };
  std::map<std::string, Chosen> chosen;
  for (const auto &key : wanted) {
    const auto it = m.settings.find(key);
    if (it == m.settings.end() || it->second.source.empty())
      return fail(ConfigErrorCode::missing_field, "calibration." + key,
                  "no value given for " + key + " (site, layout or calibration record)");
    chosen[key] = {it->second.value, it->second.source, nullptr};
  }
  std::vector<std::string> skipped;
  for (const auto &row : records) {
    const auto it = chosen.find(row.key);
    if (it == chosen.end()) continue;   // a record for a setting this start does not place
    std::string why;
    if (!(row.scope == m.scope)) why = "measured for another scope";
    else if (row.valid_until_unix_s <= now) why = "expired";
    else if (!count(row.value)) why = "not a non-negative integer";
    if (!why.empty()) {
      skipped.push_back(row.key + " record " + row.evidence_id + ": " + why);
      continue;
    }
    if (!it->second.row) it->second = {row.value, row.evidence_id + " (" + row.raw_run + "; " + row.method + ")", &row};
  }

  const auto choice = [&](const char *key) { return chosen.at(key).value; };
  const int64_t max_rows = has_helpers ? choice("handoff.max_rows") : 0;
  const int64_t small_rows = has_helpers ? choice("handoff.small_rows") : 0;
  const int64_t gemv_rows = has_helpers ? choice("handoff.gemv_large_rows") : 0;
  const int64_t tile = has_helpers ? choice("handoff.moe_down_tile") : 0;
  if (has_helpers &&
      (!positive(max_rows) || max_rows > m.dispatch_row_capacity ||
       !positive(small_rows) || !positive(gemv_rows) ||
       small_rows >= gemv_rows || gemv_rows > max_rows || tile > 2)) {
    return fail(ConfigErrorCode::invalid_value, "spec.handoff",
                "handoff rows/tile do not fit the worker's dispatch geometry");
  }

  // This is the S7 resource ledger's byte-exact host-pool/KV arithmetic,
  // now taking only the S9 typed, engine-measured resource inputs.
  int64_t margin_bytes, pool_bytes, retained_bytes, committed, pinned;
  if (!mul(cal.mib, kMiB, &margin_bytes) ||
      !add(r.host_embedding_bytes, r.unowned_expert_bytes, &pool_bytes) ||
      !add(pool_bytes, r.staging_bytes, &pool_bytes) ||
      !add(pool_bytes, margin_bytes, &pool_bytes) ||
      !mul(r.kv_bytes_per_token, r.retention_target_tokens, &retained_bytes)) {
    return fail(ConfigErrorCode::invalid_value, scope_path,
                "derived host budget overflow");
  }
  const int64_t pool_mib = ceil_mib(pool_bytes);
  const int64_t kv_mib = ceil_mib(retained_bytes);
  const int64_t reserve_mib = ceil_mib(m.host_reserve_bytes);
  int64_t reserve_rounded_bytes;
  if (!mul(reserve_mib, kMiB, &reserve_rounded_bytes) ||
      !mul(pool_mib + kv_mib, kMiB, &pinned) ||
      !add(pinned, r.ngram_bytes, &committed) ||
      !add(committed, r.checkpoint_bytes, &committed) ||
      !add(committed, r.resident_other_bytes, &committed) ||
      !add(committed, reserve_rounded_bytes, &committed)) {
    return fail(ConfigErrorCode::invalid_value, scope_path,
                "derived host admission overflow");
  }
  const int64_t host_room =
      std::min(r.mem_available_bytes, r.cgroup_headroom_bytes);
  if (committed > host_room || pinned > r.pin_available_bytes) {
    std::ostringstream msg;
    msg << "host bytes required=" << committed << " available=" << host_room
        << "; pinned bytes required=" << pinned
        << " available=" << r.pin_available_bytes
        << "; adjust retention, ownership or calibrated reserve";
    return fail(ConfigErrorCode::admission_refused, "spec.resources.host",
                msg.str());
  }

  if (r.helper_footprints.size() != m.helpers.size())
    return fail(ConfigErrorCode::ownership_mismatch,
                scope_path + ".helper_footprints",
                "selected helper set changed");
  std::map<std::pair<std::string, std::string>, const S8HelperRoom *>
      device_by_role;
  for (const auto &dev : m.helpers) {
    if (dev.role.empty() || dev.bdf.empty() ||
        !device_by_role.emplace(std::make_pair(dev.role, dev.bdf), &dev).second)
      return fail(ConfigErrorCode::ownership_mismatch, scope_path,
                  "duplicated helper role/BDF");
  }
  S8Resolution out;
  out.resource_inputs = r;
  out.host = {pool_mib, kv_mib, reserve_mib, m.checkpoint_slots};
  bool has_owned_experts = false;
  for (const auto &fp : r.helper_footprints) {
    const auto it = device_by_role.find({fp.role, fp.bdf});
    if (it == device_by_role.end())
      return fail(ConfigErrorCode::ownership_mismatch,
                  scope_path + ".helper_footprints",
                  "helper footprint role/BDF differs from inventory");
    const auto &d = *it->second;
    int64_t total = 0;
    for (auto v : {fp.lanes_bytes, fp.pointer_banks_bytes, fp.slot_map_bytes,
                   fp.incoming_slot_bytes})
      if (!add(total, v, &total))
        return fail(ConfigErrorCode::invalid_value, scope_path,
                    "helper footprint overflow");
    if (total != fp.total_bytes || !positive(d.physical_vram_bytes) ||
        !positive(d.live_claimable_bytes) || !count(d.arch_reserve_bytes) ||
        d.arch_reserve_bytes >= d.physical_vram_bytes ||
        d.reserve_evidence.empty() || !count(d.owned_experts))
      return fail(
          ConfigErrorCode::invalid_value, scope_path + ".helper_footprints",
          "missing/invalid BDF footprint, reserve or ownership evidence");
    const int64_t claimable = std::min(
        d.live_claimable_bytes, d.physical_vram_bytes - d.arch_reserve_bytes);
    if (total >= claimable)
      return fail(ConfigErrorCode::admission_refused,
                  "spec.processes.helpers.vram_budget_mib",
                  d.bdf +
                      ": non-expert footprint exhausts live claimable VRAM");
    const int64_t expert_bytes = claimable - total;
    const int64_t budget = expert_bytes / kMiB;
    if (budget == 0)
      return fail(ConfigErrorCode::admission_refused,
                  "spec.processes.helpers.vram_budget_mib",
                  d.bdf + ": less than one MiB available for expert plane");
    out.helpers.push_back({d.role, d.bdf, budget, expert_bytes % kMiB});
    has_owned_experts |= (d.owned_experts > 0);
  }

  if (has_helpers) {
    out.transport = HandoffTransport{m.transport_shm, max_rows, small_rows,
                                     gemv_rows, has_owned_experts};
    out.kernels = HandoffKernels{tile};
    out.telemetry = HandoffTelemetry{
        m.histogram_file,
        telemetry_on ? choice("handoff.histogram_period_s") : 0};
    if (telemetry_on && !positive(out.telemetry->period_s))
      return fail(ConfigErrorCode::invalid_value,
                  "spec.handoff.telemetry",
                  "histogram period must be positive");
    if (m.live_policy_requested) {
      out.live_cadence_ns = choice("handoff.live_cadence_ns");
      if (!positive(*out.live_cadence_ns))
        return fail(ConfigErrorCode::invalid_value,
                    "spec.handoff.placement.live_policy_state.cadence_ns",
                    "live cadence must be positive");
    }
  }

  out.evidence_by_path["/spec/plan/resource_inputs"] =
      "engine measured plan; owner map " + r.owner_map_digest;
  out.evidence_by_path["/spec/resources/host"] = "S7 measured byte ledger; " +
                                                 m.host_reserve_evidence +
                                                 "; " + cal.evidence;
  out.evidence_by_path["/spec/processes/helpers"] =
      "S7 current per-BDF footprint, reserve and live claimable VRAM";
  if (has_helpers)
    out.evidence_by_path["/spec/handoff/transport/skip_owned"] =
        "derived from actual owned expert sets and matching owner-map digest";
  const auto record = [&](const char *key, const char *path) {
    out.evidence_by_path[path] = chosen.at(key).source;
  };
  if (has_helpers) {
    record("handoff.max_rows", "/spec/handoff/transport/max_rows");
    record("handoff.small_rows", "/spec/handoff/transport/small_rows");
    record("handoff.gemv_large_rows",
           "/spec/handoff/transport/gemv_large_rows");
    record("handoff.moe_down_tile", "/spec/handoff/kernels/moe_down_tile");
    if (telemetry_on)
      record("handoff.histogram_period_s", "/spec/handoff/telemetry/period_s");
    if (m.live_policy_requested)
      record("handoff.live_cadence_ns",
             "/spec/handoff/placement/live_policy_state/cadence_ns");
  }
  // The records used enter spec.calibration; skipped ones are said in the calibration provenance.
  for (const auto &[key, c] : chosen) {
    if (!c.row) continue;
    auto pos = std::find(kCalibrationKeyNames.begin(),
                         kCalibrationKeyNames.end(), key);
    if (pos == kCalibrationKeyNames.end())
      return fail(ConfigErrorCode::invalid_value, "spec.calibration",
                  "S8 record key is not a v2 calibration key: " + key);
    out.calibration_entries.push_back(
        {static_cast<CalibrationKey>(pos - kCalibrationKeyNames.begin()),
         CalibrationValue{c.row->value}, c.row->source, c.row->recorded_date, c.source});
  }
  out.skipped_records = skipped;
  const auto derived = [&](std::string path, std::string source,
                           std::vector<std::string> inputs) {
    out.provenance_by_path.emplace(
        std::move(path),
        ProvenanceRecord{Origin::kDerived, std::move(source),
                         ValueClass::kDerived, Lifecycle::kStartupOnly,
                         Confidence::kExact, std::move(inputs)});
  };
  const auto calibrated = [&](std::string path, std::string key,
                              Lifecycle when) {
    const Chosen &c = chosen.at(key);
    const ValueClass leaf_class = class_of(path);
    out.provenance_by_path.emplace(
        std::move(path),
        ProvenanceRecord{c.row ? Origin::kCalibrated : Origin::kProfile, c.source,
                         leaf_class, when,
                         c.row ? Confidence::kMeasured : Confidence::kEstimate,
                         c.row ? std::vector<std::string>{"/spec/calibration", "/spec/plan/resource_inputs"}
                               : std::vector<std::string>{}});
  };
  derived("/spec/plan/resource_inputs",
          out.evidence_by_path.at("/spec/plan/resource_inputs"),
          {"/spec/plan/owner_map", "/spec/facts/host"});
  derived("/spec/resources/host",
          out.evidence_by_path.at("/spec/resources/host"),
          {"/spec/plan/resource_inputs", "/spec/calibration"});
  derived("/spec/processes/helpers",
          out.evidence_by_path.at("/spec/processes/helpers"),
          {"/spec/plan/resource_inputs/helper_footprints", "/spec/facts/gpus"});
  if (has_helpers) {
    derived("/spec/handoff/transport/skip_owned",
            out.evidence_by_path.at("/spec/handoff/transport/skip_owned"),
            {"/spec/plan/owner_map",
             "/spec/plan/resource_inputs/owner_map_digest"});
    calibrated("/spec/handoff/transport/max_rows", "handoff.max_rows",
               Lifecycle::kStartupOnly);
    calibrated("/spec/handoff/transport/small_rows", "handoff.small_rows",
               Lifecycle::kStartupOnly);
    calibrated("/spec/handoff/transport/gemv_large_rows",
               "handoff.gemv_large_rows", Lifecycle::kStartupOnly);
    calibrated("/spec/handoff/kernels/moe_down_tile", "handoff.moe_down_tile",
               Lifecycle::kStartupOnly);
    if (telemetry_on)
      calibrated("/spec/handoff/telemetry/period_s",
                 "handoff.histogram_period_s", Lifecycle::kLive);
    else
      derived("/spec/handoff/telemetry/period_s",
              "histogram disabled by explicit input",
              {"/spec/handoff/telemetry/histogram_file"});
    if (m.live_policy_requested)
      calibrated("/spec/handoff/placement/live_policy_state/cadence_ns",
                 "handoff.live_cadence_ns", Lifecycle::kStartupOnly);
  }
  return out;
}

} // namespace bc::serving::v2
