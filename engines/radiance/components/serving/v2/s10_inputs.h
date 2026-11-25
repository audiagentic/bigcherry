#pragma once
// The S10 inputs bc-serve builds for one manifest: the native P1 measurement (measured manifests), the usable P2
// calibration records, the resolver's own setting values with their sources, and the named power profile. Records are
// evidence, never a precondition (spec: provenance, not gating).
#include "v2/s10_resolver.h"
#include "serving.h"

#include <nlohmann/json.hpp>
#include <map>
#include <string>

namespace bc::serving::v2 {

struct WriterInputs {
  nlohmann::json p1_record;           // bc_serve_v2_measured_inputs; null for the measuring start's manifest
  nlohmann::json calibration_records; // array of P2 store records (calibration_store.py); may be empty
  std::string power_profile;          // the named profile (x-bigcherry.power.profile); "stock" observes
  S10PowerProfile power_rules;        // that profile's rules from the loaded profile file; empty for stock
  int64_t now_unix_s = 0;
};

// The named power profiles of the one profile file (engines/radiance/power-profiles.yaml), as the engine's YAML parser
// converted them to JSON: exactly {schema: 1, profiles: {NAME: {default_mode, rules}}}, strictly decoded. Every profile
// observes by default; stock exists and has no rules.
using PowerProfiles = std::map<std::string, S10PowerProfile>;
Result<PowerProfiles> decode_power_profiles(const nlohmann::json &document);
// The profile file compiled into bc-serve (the text the engine converts).
const std::string &embedded_power_profiles_yaml();

// `draft` is the decoded manifest with its owner map; `resolved` is what it was resolved from.
Result<S10Inputs> s10_inputs(const Manifest &draft, const bc::serving::Resolved &resolved,
                             const WriterInputs &in);

// The P2 store's record JSON as S8 reads it.
Result<S8Calibration> parse_calibration_record(const nlohmann::json &record);

} // namespace bc::serving::v2
