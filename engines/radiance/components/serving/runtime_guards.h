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
#include <vector>

namespace bc::serving::session {

enum class ReplayVerdict : uint8_t { disabled, profiled, observed, absent };
// `pending`: the file is authentic (launch nonce, proof, shape, no padded counts) but holds no route counts yet, which
// is what the engine publishes until its first routed pass has been counted. It is not evidence and not a fault.
enum class HistogramVerdict : uint8_t { fresh, missing, unsafe, stale, future, pending };
// A successful sample carries full cumulative counts across atomic file
// replacements, not a pinned inode. The producer stamps each publication with
// the owner-only per-launch nonce in a versioned sidecar.
struct HistogramIdentity {
    uint64_t device = 0;
    uint64_t inode = 0;
    int64_t modified_ns = 0;
    std::vector<uint32_t> counts;
};
struct HistogramExpectation {
    uint64_t layers = 0;
    uint64_t experts = 0;
    std::chrono::system_clock::time_point launched_at{};
    std::optional<HistogramIdentity> identity;
    std::string launch_nonce;
    // The producer renames the raw file and then its proof, so a poll can see the new raw file beside the old proof
    // for as long as the producer takes between the two renames (microseconds, but a loaded disk stretches it). A
    // mismatch is re-read for this long before it is called unsafe; the guards derive it from the declared period.
    std::chrono::nanoseconds proof_grace{std::chrono::milliseconds(100)};
};
// `reason`, when given, names the check that decided a non-fresh verdict (the refusal text carries it).
HistogramVerdict histogram_verdict(const std::filesystem::path& rank_file,
                                   std::chrono::system_clock::time_point now, std::chrono::nanoseconds max_age,
                                   const HistogramExpectation& expected, HistogramIdentity* observed = nullptr,
                                   std::string* reason = nullptr);

// The startup form: re-reads while the verdict is `pending` (authentic, no counts yet) for at most `wait`; any other
// verdict, including every kind of unsafe, is returned at once. A zero `wait` is histogram_verdict itself.
HistogramVerdict histogram_verdict_startup(const std::filesystem::path& rank_file, std::chrono::nanoseconds max_age,
                                           const HistogramExpectation& expected, std::chrono::nanoseconds wait,
                                           HistogramIdentity* observed = nullptr, std::string* reason = nullptr);

// A purely decoded signal, useful for card-free regression fixtures.
// The typed manifest depth decides whether evidence is required; text cannot
// override manifest authority.
ReplayVerdict replay_verdict(int64_t manifest_depth, bool profiled, std::string_view engine_log);
/* Per-op timing restores synchronisations under which a recorded draft pass need not appear: a start is profiled when
 * its admitted native YAML sets profile-ops (EngineStart::native_profile_ops) or its lab experiment sets profile_ops. */
bool start_is_profiled(const v2::Manifest& manifest, bool native_profile_ops);

class RuntimeGuards final {
  public:
    using Retry = Callback<v2::Status()>;
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
    v2::Status inspect_heat();
    v2::Status inspect_policy();

    std::optional<v2::Manifest> manifest_;
    std::filesystem::path engine_log_;
    std::optional<std::filesystem::path> histogram_;
    HistogramExpectation heat_expectation_;
    std::chrono::nanoseconds heat_age_budget_{std::chrono::seconds(30)};
    std::string segment_;
    std::optional<SegmentIdentity> segment_identity_;
    std::chrono::steady_clock::time_point next_poll_;
    // Start-up only: how long a pending (authentic, still all-zero) histogram is waited for before it is refused.
    std::chrono::nanoseconds heat_startup_wait_{0};
    bool policy_enabled_ = false;
};

} // namespace bc::serving::session
