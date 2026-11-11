#pragma once
// RSC11 S10: inactive, card-free v2 resolver. No production v1 path links it.
// PWR07 probe and independent qualification are injected; this code never
// queries sysfs or writes a device. S13 will supply the trusted producers.
#include "v2/s8_rules.h"

#include <map>
#include <string>
#include <vector>

namespace bc::serving::v2 {

struct S10PowerRule {
  std::string role; // engine, helperN, or helper*
  std::string arch; // e.g. gfx1100
  PowerSettings settings;
  std::string evidence;
};
struct S10PowerProfile {
  std::string name;
  std::vector<S10PowerRule> rules; // authoritative named profile, no BDF literals
};
struct S10ControlProbe {
  int64_t minimum = 0;
  int64_t maximum = 0;
  bool supported = false;
  bool original_readable = false; // restoration is mandatory
};
struct S10DeviceProbe {
  std::string bdf;
  std::string driver;
  std::map<std::string, S10ControlProbe> controls;
};
struct S10PowerQualification {
  std::string bdf;
  std::string arch;
  std::string profile;
  PowerSettings settings;
  std::string evidence;
  std::string build_id;
  std::string driver_id;
  bool qualified = false;
};
struct S10PolicyPreflight {
  LivePolicyRequested requested = LivePolicyRequested::kNotApplicable;
  bool applicable = false; // a supported layout with a remote helper and one engine rank
  bool qualified_record = false;
  bool fresh_heat = false;
  std::string calibration_key;
  std::string record_identity;
  std::string reason_detail;
  bool effect_gate = false;
  int64_t max_heat_age_ns = 0; // measured policy limit, not a default
};
struct S10Inputs {
  S8Measured measured;
  std::vector<S8Calibration> calibrations;
  S10PowerProfile profile;
  PowerMode mode = PowerMode::kObserve;
  std::map<std::string, S10DeviceProbe> probes;
  std::map<std::string, S10PowerQualification> qualifications;
  S10PolicyPreflight preflight;
  int64_t now_unix_s = 0;
};

// Produces an immutable v2 manifest from a fully populated typed v2 base.
// S8 owns resource arithmetic/admission. S5/S11 own the *out-of-band*
// structured RoomRefusal; this module does not invent another error envelope.
Result<Manifest> resolve_s10(const Manifest &base, const S10Inputs &input);

// Checks all registry restrictions against effective (nearest ancestor)
// provenance, including every array member; absence is not a measurement.
Status validate_s10_provenance(const Manifest &manifest);

// The plan's power section against the manifest it is part of: the named profile, the role overrides, exactly the
// engine TP and selected helper cards in selection order, and the observe/apply and gfx1100 safety rules. Qualification
// evidence and probes are inputs to resolve_s10 only; a reader cannot re-verify them and does not pretend to.
Status validate_s10_power_plan(const Manifest &manifest);

// Strict v2 writer: checks typed v2 encoding and computes the content digest.
// No v1 translator, no production writer, no file or hardware side effects.
Result<std::string> write_s10(const Manifest &manifest);

} // namespace bc::serving::v2
