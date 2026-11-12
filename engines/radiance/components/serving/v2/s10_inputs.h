#pragma once
// The S10 inputs bc-serve builds for one manifest: the native P1 measurement (measured manifests), the usable P2
// calibration records, the resolver's own setting values with their sources, and the named power profile. Records are
// evidence, never a precondition (spec: provenance, not gating).
#include "v2/s10_resolver.h"
#include "serving.h"

#include <nlohmann/json.hpp>
#include <string>

namespace bc::serving::v2 {

struct WriterInputs {
  nlohmann::json p1_record;           // bc_serve_v2_measured_inputs; null for the measuring start's manifest
  nlohmann::json calibration_records; // array of P2 store records (calibration_store.py); may be empty
  std::string power_profile;          // the named profile; "stock" observes
  int64_t now_unix_s = 0;
};

// `draft` is the decoded manifest with its owner map; `resolved` is what it was resolved from.
Result<S10Inputs> s10_inputs(const Manifest &draft, const bc::serving::Resolved &resolved,
                             const WriterInputs &in);

// The P2 store's record JSON as S8 reads it.
Result<S8Calibration> parse_calibration_record(const nlohmann::json &record);

} // namespace bc::serving::v2
