#pragma once
// RSC02 step 3: runtime evidence comes from the authenticated manifest,
// the fresh engine log, the engine's histogram and the live worker transport.
// It is not inferred from the resolver's configured=true bit alone.
#include "callback_boundary.h"
#include "session.h"
#include "v2/manifest.h"

#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace bc::serving::session {

enum class ReplayVerdict : uint8_t { disabled, profiled, observed, absent };
enum class HistogramVerdict : uint8_t { fresh, missing, unsafe, stale, future };
// Clock injection tests root-safe age, symlink and partial-file refusals.
HistogramVerdict histogram_verdict(const std::filesystem::path& rank_file,
                                   std::chrono::system_clock::time_point now, std::chrono::nanoseconds max_age);

// A purely decoded signal, useful for card-free regression fixtures.
// The typed manifest depth decides whether evidence is required; text cannot
// override manifest authority.
ReplayVerdict replay_verdict(int64_t manifest_depth, bool profiled, std::string_view engine_log);

class RuntimeGuards final {
  public:
    explicit RuntimeGuards(bool lab = false) : lab_pending_(lab) {}
    using Retry = Callback<v2::Status()>;
    // Until the bounded lab produces a settled marker, telemetry is pending:
    // never report it active. A final explicit gate is mandatory for lab mode.
    v2::Status prove_lab();
    v2::Status startup(const EngineStart& plan, const std::filesystem::path& engine_log, const Retry& retry);
    v2::Status poll();
    bool policy_enabled() const noexcept {
        return policy_enabled_;
    }

  private:
    struct SegmentIdentity {
        uint64_t device = 0;
        uint64_t inode = 0;
    };
    v2::Status inspect_heat() const;
    v2::Status inspect_policy();

    std::optional<v2::Manifest> manifest_;
    std::filesystem::path engine_log_;
    std::optional<std::filesystem::path> histogram_;
    std::chrono::nanoseconds heat_age_budget_{std::chrono::seconds(30)};
    std::string segment_;
    std::optional<SegmentIdentity> segment_identity_;
    std::chrono::steady_clock::time_point next_poll_;
    bool policy_enabled_ = false;
    bool lab_pending_ = false;
};

} // namespace bc::serving::session
