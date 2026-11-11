// RSC11 S10 synthetic card-free acceptance; these are NOT hardware evidence.
#include "v2/s10_resolver.h"
#include "generated/v2/serving_codec.h"
#include "v2/strict_json.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

namespace v2 = bc::serving::v2;
constexpr int64_t MiB = 1024 * 1024;

v2::Manifest fixture() {
  std::ifstream in(std::string(BC_CONTRACT_V2_DOCS_DIR) +
                   "/example.flashnext-4gpu.json");
  assert(in);
  std::stringstream buffer;
  buffer << in.rdbuf();
  auto parsed = v2::parse_strict_json(buffer.str(), "s10 fixture");
  assert(parsed);
  v2::Manifest m;
  auto error = v2::decode(parsed.value(), "", &m);
  assert(!error);
  m.spec.intent.power.profile = "xtx-sclk2700-uv125";
  return m;
}
v2::S10Inputs inputs(const v2::Manifest &m) {
  v2::S10Inputs in;
  in.profile.name = "xtx-sclk2700-uv125";
  v2::PowerSettings cap{};
  cap.sclk_max_mhz = 2700;
  cap.mclk_max_mhz = 1400;
  cap.vddgfx_offset_mv = -125;
  in.profile.rules = {
      {"engine", "gfx1100", cap, "docs/reference/power-tuning-evidence-2026-11.md"},
      {"helper*", "gfx1100", cap, "docs/reference/power-tuning-evidence-2026-11.md"},
  };
  in.now_unix_s = 1800000000;
  auto &s = in.measured;
  s.scope = {std::string(64, 'a'), std::string(64, 'b'),
             std::string(64, 'c'), std::string(64, 'd'),
             std::string(64, 'e'), m.spec.plan.owner_map.owner_map_sha256,
             8192, 4096, 128};
  auto &r = s.resource_inputs;
  r.owner_map_digest = s.scope.owner_map_digest;
  r.host_embedding_bytes = 8 * MiB;
  r.unowned_expert_bytes = 1 * MiB;
  r.staging_bytes = 2 * MiB;
  r.ngram_bytes = 3 * MiB;
  r.checkpoint_bytes = 2 * MiB;
  r.resident_other_bytes = 1 * MiB;
  r.kv_bytes_per_token = s.scope.kv_bytes_per_token;
  r.context_target_tokens = s.scope.context_target_tokens;
  r.retention_target_tokens = s.scope.retention_target_tokens;
  r.mem_available_bytes = 100 * MiB;
  r.cgroup_headroom_bytes = 90 * MiB;
  r.pin_available_bytes = 60 * MiB;
  r.host_pool_margin_calibration = {
      "host_ledger_margin.mib", v2::MarginSource::kCache, "s10-margin", 2};
  s.actual_owner_map_digest = r.owner_map_digest;
  for (size_t i = 0; i < m.spec.processes.helpers.size(); ++i) {
    const auto &h = m.spec.processes.helpers[i];
    r.helper_footprints.push_back(
        {h.role, h.gpu, MiB, MiB, MiB, MiB, 4 * MiB});
    s.helpers.push_back(
        {h.role, h.gpu, 24 * 1024 * MiB, 23 * 1024 * MiB,
         1024 * MiB, "synthetic helper reserve", 2});
  }
  s.host_reserve_bytes = 4 * MiB;
  s.host_reserve_evidence = "synthetic host-reserve sample";
  s.engine_vram_claimable_bytes = 100 * MiB;
  s.engine_vram_required_bytes = 80 * MiB;
  s.checkpoint_slots = 5;
  s.dispatch_row_capacity = 1024;
  s.transport_shm = "/rsc11-s10-synthetic";
  s.histogram_file = "/tmp/s10-synthetic-heat";
  s.live_policy_requested = true;
  auto row = [&](std::string key, int64_t value, std::string evidence) {
    return v2::S8Calibration{
        std::move(key), value, std::move(evidence),
        "synthetic/s10", s.scope, 1700000000, 2000000000,
        {value, value + 1}, "synthetic A/B; not qualified hardware",
        v2::CalibrationSource::kCache, "2023-11-14"};
  };
  in.calibrations = {
      row("host_ledger_margin.mib", 2, "s10-margin"),
      row("handoff.max_rows", 512, "s10-maxrows"),
      row("handoff.small_rows", 8, "s10-smallrows"),
      row("handoff.gemv_large_rows", 32, "s10-gemv"),
      row("handoff.moe_down_tile", 1, "s10-tile"),
      row("handoff.histogram_period_s", 30, "s10-heat"),
      row("handoff.live_cadence_ns", 5000000000LL, "s10-cadence"),
  };
  in.preflight.requested = v2::LivePolicyRequested::kOn;
  in.preflight.applicable = true;
  in.preflight.qualified_record = false;
  in.preflight.fresh_heat = false;
  in.preflight.reason_detail = "no qualified RHP18 startup record";
  in.preflight.effect_gate = true;
  in.preflight.max_heat_age_ns = 120000000000LL;
  return in;
}
int main() {
  auto m = fixture();
  auto in = inputs(m);
  auto s8 = v2::resolve_s8(in.measured, in.calibrations, in.now_unix_s);
  assert(s8);
  m.spec.plan.engine_memory->expected.host_pool_mib = s8.value().host.pool_mib;
  m.spec.plan.engine_memory->expected.vram.claimable =
      in.measured.engine_vram_claimable_bytes;
  auto solved = v2::resolve_s10(m, in);
  if (!solved) std::cerr << solved.error().path << ": "
                         << solved.error().message << "\n";
  assert(solved);
  const auto &out = solved.value();
  assert(out.spec.plan.power.profile == in.profile.name);
  assert(out.spec.plan.power.profile_digest.starts_with("sha256:"));
  assert(out.spec.plan.power.devices.size() == 4);
  for (const auto &d : out.spec.plan.power.devices)
    assert(!d.settings.sclk_max_mhz && !d.settings.vddgfx_offset_mv);
  assert(out.spec.resources.host.pool_mib == 13);
  assert(out.spec.resources.host.kv_mib == 1);
  assert(out.spec.processes.helpers[0].vram_budget_mib == 23548);
  assert(out.spec.handoff->placement.live_policy_state.reason_code ==
         "no_qualified_record");
  assert(!out.spec.handoff->placement.live_policy_state.configured);
  assert(!out.spec.handoff->placement.live_policy);
  const auto written = v2::write_s10(out);
  if (!written) std::cerr << written.error().path << ": "
                          << written.error().message << "\n";
  assert(written);
  auto wire = v2::parse_strict_json(written.value(), "written");
  assert(wire);
  assert(wire.value()["metadata"]["content_digest"].get<std::string>().starts_with("sha256:"));

  in.preflight.qualified_record = true;
  in.preflight.calibration_key = "matching-rhp18-record";
  in.preflight.record_identity = "qualified-rhp18";
  solved = v2::resolve_s10(m, in);
  assert(solved && solved.value().spec.handoff->placement.live_policy_state.reason_code ==
                       "stale_heat_feed");
  in.preflight.fresh_heat = true;
  m.spec.handoff->placement.live_policy->cadence_ns = 5000000000LL;
  m.spec.handoff->placement.live_policy->max_heat_age_ns = 120000000000LL;
  m.spec.handoff->placement.live_policy->effect_gate = true;
  solved = v2::resolve_s10(m, in);
  assert(solved && solved.value().spec.handoff->placement.live_policy_state.configured);
  assert(solved.value().spec.handoff->placement.live_policy);

  // TP engine with helpers is not eligible for RHP18 live placement.
  auto tp = m;
  tp.spec.processes.engine.device.gpus.push_back("0000:44:00.0");
  tp.spec.processes.helpers.pop_back();
  tp.spec.plan.owner_map.helpers.pop_back();
  auto tp_input = in;
  tp_input.measured.helpers.pop_back();
  tp_input.measured.resource_inputs.helper_footprints.pop_back();
  tp_input.measured.live_policy_requested = false;
  tp_input.calibrations.pop_back(); // live cadence is not selected
  tp_input.preflight.requested = v2::LivePolicyRequested::kNotApplicable;
  tp_input.preflight.applicable = false;
  tp_input.preflight.qualified_record = false;
  tp_input.preflight.fresh_heat = false;
  auto tp_result = v2::resolve_s10(tp, tp_input);
  assert(tp_result);
  const auto &tp_state = tp_result.value().spec.handoff->placement.live_policy_state;
  assert(tp_state.requested == v2::LivePolicyRequested::kNotApplicable);
  assert(tp_state.reason_code == "not_applicable" && !tp_state.configured);

  // A profile is only a candidate: explicit apply without qualified per-BDF
  // records and independent fresh probed ranges must always fail.
  in.mode = v2::PowerMode::kApply;
  assert(!v2::resolve_s10(m, in));
  for (const auto &h : m.spec.processes.helpers) {
    if (h.gpu == "0000:44:00.0") continue; // gfx1030: untouched
    v2::PowerSettings settings{};
    settings.sclk_max_mhz = 2700;
    settings.mclk_max_mhz = 1400;
    settings.vddgfx_offset_mv = -125;
    in.qualifications[h.gpu] = {h.gpu, "gfx1100", in.profile.name,
        settings, "separate hardware validation", "build-digest",
        "driver-digest", true};
    in.probes[h.gpu] = {h.gpu, "amdgpu",
        {{"sclk_max_mhz", {1000, 3000, true, true}},
         {"mclk_max_mhz", {97, 1500, true, true}},
         {"vddgfx_offset_mv", {-150, 0, true, true}}}};
  }
  solved = v2::resolve_s10(m, in);
  assert(solved);
  assert(solved.value().spec.plan.power.devices[0].settings.sclk_max_mhz == std::nullopt);
  assert(solved.value().spec.plan.power.devices[1].settings.sclk_max_mhz == 2700);
  assert(solved.value().spec.plan.power.devices[1].settings.mclk_max_mhz == 1400);
  assert(solved.value().spec.plan.power.devices[3].settings.sclk_max_mhz == std::nullopt);
  in.probes.at("0000:41:00.0").controls.at("mclk_max_mhz").maximum = 1250;
  assert(!v2::resolve_s10(m, in)); // MCLK range is supplied by the card probe
  in.probes.at("0000:41:00.0").controls.at("mclk_max_mhz").maximum = 1500;
  in.probes.at("0000:41:00.0").controls.at("sclk_max_mhz").maximum = 2600;
  assert(!v2::resolve_s10(m, in));
  in.probes.at("0000:41:00.0").controls.at("sclk_max_mhz").maximum = 3000;
  in.profile.rules[1].settings.sclk_max_mhz.reset();
  assert(!v2::resolve_s10(m, in)); // offset alone increases gfx1100 power
  in.profile.rules[1].settings.sclk_max_mhz = 2700;

  auto duplicate = m;
  duplicate.spec.processes.helpers[0].gpu =
      duplicate.spec.processes.engine.device.gpus[0];
  assert(!v2::resolve_s10(duplicate, in));
  auto bad_role = m;
  bad_role.spec.intent.power.role_overrides["helper9"].power_cap_w = 200;
  assert(!v2::resolve_s10(bad_role, in));
  auto bad_memory = in;
  bad_memory.mode = v2::PowerMode::kObserve;
  bad_memory.measured.resource_inputs.mem_available_bytes = 5 * MiB;
  assert(!v2::resolve_s10(m, bad_memory)); // S8 admission, no guessed fallback
  auto wrong_engine = m;
  wrong_engine.spec.plan.engine_memory->expected.host_pool_mib++;
  assert(!v2::resolve_s10(wrong_engine, bad_memory));
  auto wrong_provenance = out;
  wrong_provenance.provenance["/spec/plan/resource_inputs/host_embedding_bytes"] = {
      v2::Origin::kProfile, "unmeasured literal", v2::ValueClass::kDerived,
      v2::Lifecycle::kStartupOnly, v2::Confidence::kEstimate, {}};
  auto rejected = v2::validate_s10_provenance(wrong_provenance);
  assert(rejected && rejected->code == v2::ConfigErrorCode::provenance_invalid);
  wrong_provenance = out;
  wrong_provenance.provenance.erase("/spec/resources/host");
  wrong_provenance.provenance["/spec/resources"] = {
      v2::Origin::kProfile, "unmeasured inherited literal",
      v2::ValueClass::kDerived, v2::Lifecycle::kStartupOnly,
      v2::Confidence::kEstimate, {}};
  assert(v2::validate_s10_provenance(wrong_provenance));

  // Keep S5/S11's RoomRefusal parser as the only structured admission wire
  // consumer. S10's C++ resource stage returns a typed admission error.
  std::cout << "inactive v2 S10 power/resource/live-policy/provenance tests pass\n";
}
