#pragma once
// RHP16: the owner persists the helper's settled calibration record. The helper prints one RHP16_CALIBRATION_JSON line in
// its log; this watcher reads the log incrementally while the session serves, and ingests that one line into the
// calibration store with provenance taken from the owner's own running manifest. Off unless the session names a store.
// A store write failure is a warning, never a serving failure.
#include "live_policy_ingest.h"
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace bc::serving {

class CalibrationWatch {
public:
    enum class State { waiting, stored, refused, settled_elsewhere };
    // `target` is the owner's running manifest's identity (live_policy_target_from_manifest).
    CalibrationWatch(std::filesystem::path helper_log, std::filesystem::path store, LivePolicyTarget target);
    // Read what the helper log gained since the last poll. At most one ingest ever happens: the first complete marker
    // line decides, later polls do nothing. Returns true exactly when this poll reached a decision.
    bool poll();
    State state() const { return state_; }
    // The sentence for the owner record or stderr: what was stored, or why not. Empty while waiting.
    const std::string& detail() const { return detail_; }
    const std::string& store_key() const { return target_.store_key; }

private:
    std::filesystem::path log_, store_;
    LivePolicyTarget target_;
    uint64_t offset_ = 0;
    std::string pending_;   // an unterminated trailing line, completed by a later poll
    State state_ = State::waiting;
    std::string detail_;
    bool decide(const std::string& payload);
};

} // namespace bc::serving
