// GPU-free RSC11 S8 acceptance: v2 typed measurements, scopes, admission,
// calibration.
#include "v2/s8_rules.h"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace s8 = bc::serving::v2;
constexpr int64_t MiB = 1024 * 1024;

s8::S8Measured fixture() {
  s8::S8Measured m;
  m.scope = {std::string(64, 'a'),
             std::string(64, 'b'),
             std::string(64, 'c'),
             std::string(64, 'd'),
             std::string(64, 'f'),
             std::string(64, 'e'),
             8192,
             4096,
             128};
  m.resource_inputs.owner_map_digest = m.scope.owner_map_digest;
  m.resource_inputs.host_embedding_bytes = 8 * MiB;
  m.resource_inputs.unowned_expert_bytes = 1 * MiB;
  m.resource_inputs.staging_bytes = 2 * MiB;
  m.resource_inputs.ngram_bytes = 3 * MiB;
  m.resource_inputs.checkpoint_bytes = 2 * MiB;
  m.resource_inputs.resident_other_bytes = 1 * MiB;
  m.resource_inputs.kv_bytes_per_token = m.scope.kv_bytes_per_token;
  m.resource_inputs.context_target_tokens = m.scope.context_target_tokens;
  m.resource_inputs.retention_target_tokens = m.scope.retention_target_tokens;
  m.resource_inputs.mem_available_bytes = 100 * MiB;
  m.resource_inputs.cgroup_headroom_bytes = 90 * MiB;
  m.resource_inputs.pin_available_bytes = 60 * MiB;
  m.resource_inputs.host_pool_margin_calibration = {
      "host_ledger_margin.mib", s8::MarginSource::kCache, "margin-run", 2};
  m.resource_inputs.helper_footprints = {
      {"tail", "0000:09:00.0", MiB, MiB, MiB, MiB, 4 * MiB}};
  m.actual_owner_map_digest = m.scope.owner_map_digest;
  m.helpers = {{"tail", "0000:09:00.0", 24 * 1024 * MiB, 23 * 1024 * MiB,
                1024 * MiB, "worker-arch-reserve-run", 2}};
  m.host_reserve_bytes = 4 * MiB;
  m.host_reserve_evidence = "S7 host reserve measurement";
  // These are the actual November R9700 claimable/required measurements,
  // set equal for the good fixture and reduced for the shortfall regression.
  m.engine_vram_required_bytes = 33242744312LL;
  m.engine_vram_claimable_bytes = 33242744312LL;
  m.checkpoint_slots = 5;
  m.checkpoint_slot_bytes = MiB / 2;   // 5 slots x 0.5 MiB: the 2 MiB (rounded) the resource inputs carry
  m.dispatch_row_capacity = 1024;
  m.transport_shm = "/rsc11-s8-fixture";
  m.histogram_file = std::string("/measured/histogram.csv");
  m.live_policy_requested = true;
  m.settings = {{"handoff.max_rows", {256, "site.toml:handoff.max_rows"}},
                {"handoff.small_rows", {6, "site.toml:handoff.small_rows"}},
                {"handoff.gemv_large_rows", {16, "site.toml:handoff.gemv_large_rows"}},
                {"handoff.moe_down_tile", {1, "site.toml:handoff.moe_down_tile"}},
                {"handoff.histogram_period_s", {60, "site.toml:handoff.histogram_period_s"}},
                {"handoff.live_cadence_ns", {10000000000LL, "site.toml:handoff.live_cadence_ns"}}};
  return m;
}

std::vector<s8::S8Calibration> records(const s8::S8Scope &scope) {
  const auto row = [&](std::string key, int64_t chosen, std::string id) {
    return s8::S8Calibration{
        std::move(key),
        chosen,
        std::move(id),
        "runs/rsc09-lab-fixture",
        scope,
        1700000000,
        2000000000,
        {chosen, chosen + 1},
        "card-free synthetic A/B fixture; not hardware evidence",
        s8::CalibrationSource::kCache,
        "2023-11-14"};
  };
  return {
      row("host_ledger_margin.mib", 2, "margin-run"),
      row("handoff.max_rows", 512, "maxrows-run"),
      row("handoff.small_rows", 8, "smallrows-run"),
      row("handoff.gemv_large_rows", 32, "gemv-run"),
      row("handoff.moe_down_tile", 1, "tile-run"),
      row("handoff.histogram_period_s", 30, "histogram-run"),
      row("handoff.live_cadence_ns", 5000000000LL, "cadence-run"),
  };
}

int main() {
  const auto fixture_input = fixture();
  const auto calibrated = records(fixture_input.scope);
  constexpr int64_t now = 1800000000;
  auto solved = s8::resolve_s8(fixture_input, calibrated, now);
  assert(solved);
  const auto &result = solved.value();
  assert(result.resource_inputs.owner_map_digest ==
         fixture_input.actual_owner_map_digest);
  assert(result.host.pool_mib == 13); // measured 8+1+2, calibrated margin 2
  assert(result.host.kv_mib == 1); // 4096 retained tokens x 128 measured bytes
  assert(result.host.reserve_mib == 4);
  assert(result.host.checkpoint_slots.value() == 5);
  assert(result.helpers.size() == 1 &&
         result.helpers[0].expert_budget_mib == 23548);
  assert(result.transport->max_rows == 512 && result.transport->skip_owned);
  assert(result.transport->small_rows == 8 &&
         result.transport->gemv_large_rows == 32);
  assert(result.kernels->moe_down_tile == 1 &&
         result.telemetry->period_s == 30);
  assert(result.calibration_entries.size() == calibrated.size() - 1); // the margin arrives in resource_inputs
  assert(result.skipped_records.empty());
  assert(
      result.provenance_by_path.at("/spec/handoff/transport/max_rows").origin ==
      s8::Origin::kCalibrated);
  assert(result.provenance_by_path.at("/spec/handoff/transport/skip_owned")
             .origin == s8::Origin::kDerived);
  assert(result.live_cadence_ns.value() == 5000000000LL);
  assert(result.evidence_by_path.at("/spec/handoff/transport/max_rows")
             .find("maxrows-run") != std::string::npos);

  auto changed = fixture_input;
  changed.resource_inputs.host_embedding_bytes += 3 * MiB;
  solved = s8::resolve_s8(changed, calibrated, now);
  assert(solved && solved.value().host.pool_mib == 16);
  changed = fixture_input;
  changed.resource_inputs.retention_target_tokens = 8192;
  changed.scope.retention_target_tokens = 8192;
  // Records of another scope are skipped, not refused: the given settings apply and say where they came from.
  solved = s8::resolve_s8(changed, calibrated, now);
  assert(solved && solved.value().transport->max_rows == 256 &&
         solved.value().skipped_records.size() == 6 &&
         solved.value().calibration_entries.empty() &&
         solved.value().provenance_by_path.at("/spec/handoff/transport/max_rows").origin == s8::Origin::kProfile &&
         solved.value().evidence_by_path.at("/spec/handoff/transport/max_rows") == "site.toml:handoff.max_rows");
  auto updated = records(changed.scope);
  solved = s8::resolve_s8(changed, updated, now);
  assert(solved && solved.value().host.kv_mib == 1); // 8192x128 exactly 1 MiB
  changed.resource_inputs.retention_target_tokens = 16384;
  changed.scope.retention_target_tokens = 16384;
  updated = records(changed.scope);
  solved = s8::resolve_s8(changed, updated, now);
  assert(solved && solved.value().host.kv_mib == 2);

  changed = fixture_input;
  changed.engine_vram_claimable_bytes = 33241708024LL;
  auto refusal = s8::resolve_s8(changed, calibrated, now);
  assert(!refusal &&
         refusal.error().code == s8::ConfigErrorCode::admission_refused);
  assert(refusal.error().message.find("1036288") != std::string::npos);
  assert(refusal.error().path ==
         "spec.plan.engine_memory.expected.vram.claimable");

  changed = fixture_input;
  changed.resource_inputs.mem_available_bytes = 12 * MiB;
  assert(!s8::resolve_s8(changed, calibrated, now));
  changed = fixture_input;
  changed.resource_inputs.mem_available_bytes = 24 * MiB + 512;
  changed.resource_inputs.cgroup_headroom_bytes = 24 * MiB + 512;
  changed.host_reserve_bytes = 4 * MiB + 1;
  // Unrounded admission would fit; the actual CLI consumes five whole MiB.
  assert(!s8::resolve_s8(changed, calibrated, now));
  changed = fixture_input;
  changed.resource_inputs.pin_available_bytes = 12 * MiB;
  assert(!s8::resolve_s8(changed, calibrated, now));
  changed = fixture_input;
  changed.resource_inputs.owner_map_digest = std::string(64, '0');
  assert(!s8::resolve_s8(changed, calibrated, now));

  // An expired, out-of-scope or unknown record is skipped; the given value applies.
  auto bad_cal = calibrated;
  bad_cal[1].valid_until_unix_s = now;
  solved = s8::resolve_s8(fixture_input, bad_cal, now);
  assert(solved && solved.value().transport->max_rows == 256 && solved.value().skipped_records.size() == 1);
  bad_cal = calibrated;
  bad_cal[1].scope.layout_digest = std::string(64, 'f');
  solved = s8::resolve_s8(fixture_input, bad_cal, now);
  assert(solved && solved.value().transport->max_rows == 256);
  bad_cal = calibrated;
  bad_cal[1].key = "unknown.rows";
  solved = s8::resolve_s8(fixture_input, bad_cal, now);
  assert(solved && solved.value().transport->max_rows == 256 && solved.value().skipped_records.empty());
  bad_cal = calibrated;
  bad_cal.pop_back();
  solved = s8::resolve_s8(fixture_input, bad_cal, now);
  assert(solved && solved.value().live_cadence_ns.value() == 10000000000LL);
  bad_cal = {};
  solved = s8::resolve_s8(fixture_input, bad_cal, now); // no records at all still starts
  assert(solved && solved.value().transport->max_rows == 256 && solved.value().kernels->moe_down_tile == 1 &&
         solved.value().telemetry->period_s == 60 && solved.value().calibration_entries.empty());
  auto unset = fixture_input;
  unset.settings.erase("handoff.max_rows");
  assert(!s8::resolve_s8(unset, {}, now)); // a value must come from somewhere
  bad_cal = calibrated;
  bad_cal[1].value =
      4000; // cannot exceed measured row capacity / tested choices
  bad_cal[1].measured_candidates = {4000, 5000};
  assert(!s8::resolve_s8(fixture_input, bad_cal, now));
  bad_cal = calibrated;
  bad_cal[2].value = 64;
  bad_cal[2].measured_candidates = {64, 65};
  assert(
      !s8::resolve_s8(fixture_input, bad_cal, now)); // small > GEMV threshold

  changed = fixture_input;
  changed.resource_inputs.helper_footprints[0].total_bytes -= 1;
  assert(!s8::resolve_s8(changed, calibrated, now));
  changed = fixture_input;
  changed.helpers[0].owned_experts = 0;
  solved = s8::resolve_s8(changed, calibrated, now);
  assert(solved && !solved.value().transport->skip_owned);

  // Independent tuning sweeps change each selected rule without v1 constants.
  bad_cal = calibrated;
  bad_cal[1].value = 256;
  bad_cal[1].measured_candidates = {256, 512};
  bad_cal[2].value = 6;
  bad_cal[2].measured_candidates = {6, 8};
  bad_cal[3].value = 16;
  bad_cal[3].measured_candidates = {16, 32};
  bad_cal[4].value = 2;
  bad_cal[4].measured_candidates = {1, 2};
  solved = s8::resolve_s8(fixture_input, bad_cal, now);
  assert(solved && solved.value().transport->max_rows == 256 &&
         solved.value().transport->small_rows == 6 &&
         solved.value().transport->gemv_large_rows == 16 &&
         solved.value().kernels->moe_down_tile == 2);
  changed = fixture_input;
  changed.resource_inputs.host_pool_margin_calibration.mib = 4;
  bad_cal = calibrated;
  bad_cal[0].value = 4;
  bad_cal[0].measured_candidates = {2, 4};
  solved = s8::resolve_s8(changed, bad_cal, now);
  assert(solved && solved.value().host.pool_mib == 15);

  changed = fixture_input;
  changed.histogram_file = std::nullopt;
  changed.live_policy_requested = false;
  bad_cal = calibrated;
  bad_cal.pop_back();
  bad_cal.pop_back();
  solved = s8::resolve_s8(changed, bad_cal, now);
  assert(solved && !solved.value().telemetry->histogram_file &&
         solved.value().telemetry->period_s == 0);

  changed = fixture_input;
  changed.live_policy_requested = false; // histogram still on, live policy off
  bad_cal = calibrated;
  bad_cal.pop_back(); // no cadence calibration necessary
  solved = s8::resolve_s8(changed, bad_cal, now);
  assert(solved && solved.value().telemetry->period_s == 30 &&
         !solved.value().live_cadence_ns);

  changed = fixture_input;
  changed.helpers.clear();
  changed.resource_inputs.helper_footprints.clear();
  changed.histogram_file = std::nullopt;
  changed.live_policy_requested = false;
  changed.transport_shm.clear();
  changed.dispatch_row_capacity = 0;
  bad_cal = {calibrated.front()}; // only host margin is needed without helpers
  solved = s8::resolve_s8(changed, bad_cal, now);
  assert(solved && solved.value().helpers.empty() &&
         !solved.value().transport && !solved.value().kernels &&
         !solved.value().telemetry && solved.value().host.pool_mib == 13);
  assert(solved.value().calibration_entries.empty());
  changed.live_policy_requested = true;
  assert(!s8::resolve_s8(changed, bad_cal, now)); // no helper/live histogram

  std::cout
      << "S8 typed v2 rule, scoped calibration and admission fixtures passed\n";
}
