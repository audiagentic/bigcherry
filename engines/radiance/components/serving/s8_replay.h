#pragma once
// RSC02: measured S8 admission on every stored v2 execution. The writer's
// native P1 producer record and P2 raw records are an execution-independent,
// digest-bound evidence file; neither is appended to the immutable manifest.
#include "serving.h"
#include "v2/p1_live_room.h"
#include "v2/s10_inputs.h"

#include <cstdint>
#include <filesystem>
#include <string>

namespace bc::serving::session {

struct StoredS8Evidence {
    nlohmann::json p1_record;
    nlohmann::json calibration_records;
    std::string manifest_digest;
    int64_t produced_at_unix_s = 0;
};

// Bounded strict parsing; no synthesized "manifest-shaped" records.
StoredS8Evidence read_stored_s8_evidence(const std::filesystem::path& path, const std::string& expected_digest,
                                         const std::string& expected_file_identity);

// Uses the SAME real S10-to-S8 typed consumer as the writer. Re-resolves
// qualified P2 records in the current measured model/build/device/owner scope.
// Requires equal semantic calibration and host/handoff decisions, including
// their evidence identity, not just equal numeric fallback values.
v2::Status verify_stored_s8(const v2::Manifest& admitted, const Resolved& current_config,
                            const nlohmann::json& fresh_facts, const v2::P1LiveRoom& live,
                            const StoredS8Evidence& issued, const std::filesystem::path& engine_binary,
                            const std::filesystem::path& helper_binary, int64_t now_unix_s);

} // namespace bc::serving::session
