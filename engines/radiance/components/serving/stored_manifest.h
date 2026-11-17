#pragma once
// RSC02 S5: an immutable stored manifest is an input, not a request to write a new manifest.
// Strict v2 authentication and replay compatibility precede all power/child operations.
#include "v2/error.h"
#include "v2/manifest.h"
#include "v2/p1_live_room.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <nlohmann/json.hpp>

namespace bc::serving::session {

enum class ReplayRefusal : uint8_t {
    bytes_or_digest, build, model, devices, calibration, execution_projection,
    live_admission, missing_gate
};

struct ReplayError {
    ReplayRefusal reason;
    v2::ConfigError detail;
};

using ReplayResult = std::variant<v2::Manifest, ReplayError>;

// Captured by the production owner from its own executable, model, runtime,
// strict RSC12 config resolver and a FRESH detector. Values are independently
// injectable for card-free acceptance cases, never read from environment flags.
struct ReplayEvidence {
    std::string producer_sha256;
    std::string model_identity;
    std::string engine_identity;
    nlohmann::json fresh_facts;
    nlohmann::json effective_spec; // a fresh in-memory unmeasured manifest's spec
};

// Compares the static semantic execution projection (not instance id,
// provenance, live facts or an ephemeral handoff name). The calibrated values
// and stored power decision have their own equality checks.
nlohmann::json execution_projection(const nlohmann::json& spec);

// No file writes. The same strict reader used by the engine authenticates
// the original bytes and digest, then the owner verifies the current build,
// model, selected-device identity, calibration, and RSC12 execution projection.
ReplayResult authenticate_replay(std::string_view bytes, std::string_view source,
                                  const ReplayEvidence& evidence);

// A separate live admission on EVERY replay. Called even when the stored
// manifest's original facts and identity still match. It covers host RAM,
// cgroup, pinned and memlock room and all selected engine/helper GPU cards.
v2::Status check_replay_live_room(const v2::Manifest& manifest,
                                  const nlohmann::json& fresh_facts,
                                  const v2::P1LiveRoom& room);

} // namespace bc::serving::session
