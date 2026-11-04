#pragma once
// Host scheduler boundary. Its policy input comes ONLY from the engine's
// bc_moe_remote_wanted_upto, which is also used to declare mailbox kernels.
#include <atomic>
#include <cstdint>

namespace r3::epoch {
enum class AdmissionBootstrap : uint8_t { LocalOnly, Ready, Uninitialized };
inline AdmissionBootstrap admission_bootstrap(
        bool remote_kernel_possible, const std::atomic<uint32_t>& ready_ranks,
        uint32_t expected_rank_mask) {
    if (!remote_kernel_possible) return AdmissionBootstrap::LocalOnly;
    // Exactly one acquire read per actual remote admission. No digest,
    // mapped Request or GPU device round-trip on the scheduler path.
    return (ready_ranks.load(std::memory_order_acquire) & expected_rank_mask)
               == expected_rank_mask ? AdmissionBootstrap::Ready
                                     : AdmissionBootstrap::Uninitialized;
}
} // namespace r3::epoch
