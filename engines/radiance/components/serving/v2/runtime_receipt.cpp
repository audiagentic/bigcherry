#include "v2/runtime_receipt.h"

#include <algorithm>
#include <regex>
#include <set>

namespace bc::serving::v2 {
namespace {
using Json = nlohmann::json;
using PC = r3::moe_shm::PlacementControl;
using PJ = r3::moe_shm::PlacementJournal;

ConfigError fail(std::string field, std::string why) {
  return make_error(ConfigErrorCode::invalid_value,
                    "runtime_receipt." + field, std::move(why));
}
bool digest64(const std::string& s) {
  static const std::regex re("^(sha256:)?[0-9a-f]{64}$");
  return std::regex_match(s, re);
}
Json snapshot_json(const RuntimeSnapshot& s) {
  Json j = Json::object();
  j["manifest_digest"] = s.manifest_digest;
  j["epoch"] = s.epoch;
  j["owner_generation_digest"] = s.owner_generation_digest;
  if (!s.owner_generation_digest_hex.empty())
    j["owner_generation_digest_hex"] = s.owner_generation_digest_hex;
  j["state"] = s.state;
  j["fault_code"] = s.fault_code;
  j["fault_actor"] = s.fault_actor;
  j["transaction_id"] = s.transaction_id;
  j["cleanup_fault"] = s.cleanup_fault;
  j["cleanup_actor"] = s.cleanup_actor;
  j["placement_stopped"] = s.placement_stopped;
  j["degraded"] = s.degraded;
  return j;
}
} // namespace

Result<Json> publish_runtime_receipt(
    const LivePolicyStartup& startup,
    const std::string& manifest_content_digest,
    uint64_t epoch,
    uint64_t owner_generation_digest,
    const PC& control,
    const PJ& journal) {

  if (manifest_content_digest.empty() || !digest64(manifest_content_digest))
    return fail("manifest_content_digest",
                "manifest content digest (sha256) is required to bound the receipt");

  // Validate startup requested/configured consistency. Policy is not called
  // active during transition; the startup decision stays separate.
  if (startup.reason_code.empty())
    return fail("startup_decision.reason_code", "startup reason_code is required");
  if (startup.configured) {
    if (startup.requested != LivePolicyRequested::kOn)
      return fail("startup_decision.configured",
                  "configured startup requires requested=on");
    if (startup.reason_code != "configured")
      return fail("startup_decision.reason_code",
                  "configured startup requires reason_code=configured");
  } else {
    if (startup.requested == LivePolicyRequested::kOn &&
        startup.reason_code == "configured")
      return fail("startup_decision.configured",
                  "configured=true required when startup is configured");
  }
  if ((startup.requested == LivePolicyRequested::kOff ||
       startup.requested == LivePolicyRequested::kNotApplicable) &&
      startup.configured)
    return fail("startup_decision.configured",
                "configured must be false when policy is off or not applicable");

  // Genuine worker_epoch telemetry.
  if (control.active_epoch == 0)
    return fail("control.active_epoch", "placement control active_epoch is required");
  if (control.active_digest == 0)
    return fail("control.active_digest", "placement control active_digest is required");
  if (control.transaction_id == 0)
    return fail("control.transaction_id", "placement control transaction_id is required");
  if (control.journal_index > 1)
    return fail("control.journal_index", "placement control journal_index must be 0 or 1");

  // Epoch / owner-generation identity must match control.
  if (epoch == 0)
    return fail("epoch", "epoch is required");
  if (epoch != control.active_epoch)
    return fail("epoch", "epoch does not match placement control active_epoch");
  if (owner_generation_digest == 0)
    return fail("owner_generation_digest", "owner generation digest is required");
  if (owner_generation_digest != control.active_digest)
    return fail("owner_generation_digest",
                "owner generation digest does not match placement control active_digest");

  // Matching current journal (not the stale other slot).
  const uint32_t idx = control.journal_index;
  const auto& current = journal.entry[idx];
  if (current.transaction_id == 0)
    return fail("journal", "current journal record is missing");
  if (current.transaction_id != control.transaction_id)
    return fail("journal", "current journal transaction_id does not match control");

  // Build one bounded snapshot from the current control + current journal.
  RuntimeSnapshot s;
  s.manifest_digest = manifest_content_digest;
  s.epoch = control.active_epoch;
  s.owner_generation_digest = control.active_digest;
  s.state = control.state;
  s.fault_code = control.fault_code;
  s.fault_actor = control.fault_actor;
  s.transaction_id = control.transaction_id;
  s.cleanup_fault = current.cleanup_fault;
  s.cleanup_actor = current.cleanup_actor;
  // placement_stopped is classified ONLY from the current matched journal
  // cleanup fault. stop_placement_after_cleanup does not set control.fault_code.
  s.placement_stopped = (current.cleanup_fault != 0);
  s.degraded = (control.fault_code != 0) ||
               (current.fault != 0) || (current.cleanup_fault != 0);
  // Ordinary in-flight transitions are not degradation.
  const bool transitioning =
      control.state != static_cast<uint32_t>(r3::moe_shm::PlacementState::ACTIVE);

  std::vector<RuntimeSnapshot> snapshots;
  snapshots.push_back(s);

  std::string runtime_status = "missing_evidence";
  bool placement_stopped = s.placement_stopped;
  bool degraded = s.degraded;
  if (placement_stopped) runtime_status = "placement_stopped";
  else if (degraded) runtime_status = "degraded";
  else if (transitioning) runtime_status = "transitioning";
  else runtime_status = "active";

  Json snap_arr = Json::array();
  for (const auto& sn : snapshots) snap_arr.push_back(snapshot_json(sn));

  Json out = Json::object();
  out["schema_version"] = 1;
  out["kind"] = "bc_serve_v2_runtime_receipt";
  out["manifest"] = Json::object();
  out["manifest"]["content_digest"] = manifest_content_digest;
  out["startup_decision"] = Json::object();
  out["startup_decision"]["requested"] = startup.requested == LivePolicyRequested::kOn ? "on" :
      startup.requested == LivePolicyRequested::kOff ? "off" : "not_applicable";
  out["startup_decision"]["configured"] = startup.configured;
  out["startup_decision"]["reason_code"] = startup.reason_code;
  out["startup_decision"]["reason_detail"] = startup.reason_detail;
  out["startup_decision"]["calibration_key"] = startup.calibration_key;
  out["startup_decision"]["record_identity"] = startup.record_identity;
  out["startup_decision"]["effect_gate"] = startup.effect_gate;
  out["startup_decision"]["histogram_period_s"] = startup.histogram_period_s;
  out["startup_decision"]["cadence_ns"] = startup.cadence_ns;
  out["startup_decision"]["max_heat_age_ns"] = startup.max_heat_age_ns;

  out["bounding"] = Json::object();
  out["bounding"]["manifest_digest"] = manifest_content_digest;
  out["bounding"]["epoch"] = control.active_epoch;
  out["bounding"]["owner_generation_digest"] = control.active_digest;
  out["runtime_status"] = runtime_status;
  out["placement_stopped"] = placement_stopped;
  out["degraded"] = degraded;
  out["bounded"] = true;
  out["max_snapshots"] = 2;
  out["snapshot_count"] = snapshots.size();
  out["snapshots"] = snap_arr;
  out["required_producer"] =
      "r3_moe_worker/worker_epoch.h:PlacementControl+PlacementJournal control/journal flags";
  return out;
}

} // namespace bc::serving::v2