#pragma once
// The live placement policy's settings, typed from the manifest's spec.handoff.placement.live_policy (serving manifest v1).
// Every member is explicit there (no defaults, scaled integers), so this only converts and re-checks the control invariants
// the policy relies on; a policy that is absent from the manifest never constructs a Settings.
#include "worker_policy_input.h"
#include "worker_share.h"
#include "../serving/generated/v2/serving_types.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace r3::policy {
// Independent helper-side guard: a lab's explicit live_policy bypasses the
// serving-derived policy gate, so the strict v1 reader alone is not sufficient
// to certify that the CPU solver's LAST-helper-is-tail topology holds.
// The only validated live-policy arrangement is one engine rank, at least one
// head and exactly one final tail with zero weight in the head selector.
// TP engines and head-only layouts remain available with static placement.
inline bool live_policy_arrangement_supported(
    uint32_t engine_ranks, const std::string& placement,
    const std::vector<uint64_t>& shares, std::string* why) {
    auto refuse = [&](const char* reason) {
        if (why) *why = reason;
        return false;
    };
    if (engine_ranks != 1)
        return refuse("live_policy: TP/multi-rank engine is not validated for live migration");
    if (placement != "tail")
        return refuse("live_policy: helper planner requires tail strategy; head-only is unsupported");
    if (shares.size() < 2 || shares.size() > worker_share::kMaxHelperShares ||
        shares.back() != 0)
        return refuse("live_policy: requires head helpers followed by one zero-share tail helper");
    bool head = false;
    for (size_t i = 0; i + 1 < shares.size(); ++i) {
        if (shares[i] > 1000000)
            return refuse("live_policy: invalid head share");
        head = head || shares[i] > 0;
    }
    if (!head) return refuse("live_policy: at least one positive head share is required");
    return true;
}
struct Settings {
    bool enabled = false, accept_unwatermarked_heat = false, effect_gate = false;
    uint64_t cadence_ns = 0, abort_cooldown_ns = 0, learned_eligibility_ttl_ns = 0;
    uint64_t effect_window_samples = 0, effect_min_quiet_windows = 0, effect_revert_ttl_ns = 0;
    double effect_noise_multiplier = 0;
    PolicyControls controls{};
    std::vector<uint32_t> head_shares;   // the launch's helper shares from the manifest, one weight a helper
    std::vector<uint32_t> protected_layers;

    /* The enabled policy of a manifest the shared reader accepted (ranges and exclusions were checked there). `shares` are
     * the helpers' shares: the live plan is the static placer's, so it reads the shares the launch placed with. False with
     * *error naming the member when an invariant the policy relies on does not hold. */
    bool from_manifest(const bc::serving::v2::LivePolicy& p, const std::vector<uint32_t>& shares, std::string* error) {
        if (!error) return false;
        auto positive = [&](int64_t v, const char* name) {
            if (v > 0) return true;
            *error = std::string("live_policy.") + name + ": want a positive integer";
            return false;
        };
        if (!positive(p.cadence_ns, "cadence_ns") || !positive(p.max_moves, "max_moves") ||
            !positive(p.max_moved_bytes, "max_moved_bytes") || !positive(p.min_residency_ns, "min_residency_ns") ||
            !positive(p.max_heat_age_ns, "max_heat_age_ns") || !positive(p.min_samples, "min_samples") ||
            !positive(p.abort_cooldown_ns, "abort_cooldown_ns") ||
            !positive(p.learned_eligibility_ttl_ns, "learned_eligibility_ttl_ns") ||
            !positive(p.effect_window_samples, "effect_window_samples") ||
            !positive(p.effect_min_quiet_windows, "effect_min_quiet_windows") ||
            !positive(p.effect_revert_ttl_ns, "effect_revert_ttl_ns") ||
            !positive(p.half_life_observations_milli, "half_life_observations_milli") ||
            !positive(p.effect_noise_multiplier_milli, "effect_noise_multiplier_milli"))
            return false;
        if (p.margin_ppm <= 0 || p.margin_ppm >= 1000000) {
            *error = "live_policy.margin_ppm: want 1..999999";
            return false;
        }
        if (p.max_moves > int64_t(UINT32_MAX) ||
            p.cadence_ns > int64_t(std::chrono::nanoseconds::max().count())) {
            *error = "live_policy.max_moves / cadence_ns: out of range";
            return false;
        }
        enabled = true;
        effect_gate = p.effect_gate;
        accept_unwatermarked_heat = p.accept_unwatermarked_heat;
        cadence_ns = uint64_t(p.cadence_ns);
        abort_cooldown_ns = uint64_t(p.abort_cooldown_ns);
        learned_eligibility_ttl_ns = uint64_t(p.learned_eligibility_ttl_ns);
        effect_window_samples = uint64_t(p.effect_window_samples);
        effect_min_quiet_windows = uint64_t(p.effect_min_quiet_windows);
        effect_revert_ttl_ns = uint64_t(p.effect_revert_ttl_ns);
        effect_noise_multiplier = double(p.effect_noise_multiplier_milli) / 1000.0;
        controls.margin = double(p.margin_ppm) / 1e6;
        controls.half_life_observations = double(p.half_life_observations_milli) / 1000.0;
        controls.max_moves = uint32_t(p.max_moves);
        controls.max_moved_bytes = uint64_t(p.max_moved_bytes);
        controls.min_residency_ns = uint64_t(p.min_residency_ns);
        controls.max_heat_age_ns = uint64_t(p.max_heat_age_ns);
        controls.min_samples = uint64_t(p.min_samples);
        protected_layers.clear();
        for (int64_t layer : p.protected_layers) {
            if (layer < 0 || layer > int64_t(UINT32_MAX) ||
                (!protected_layers.empty() && uint32_t(layer) <= protected_layers.back())) {
                *error = "live_policy.protected_layers: want sorted, distinct layer ids";
                return false;
            }
            protected_layers.push_back(uint32_t(layer));
        }
        uint64_t share_total = 0;
        for (uint32_t w : shares) {
            if (w > 1000000u) {
                *error = "helper share above 1000000";
                return false;
            }
            share_total += w;
        }
        if (shares.empty() || shares.size() > worker_share::kMaxHelperShares || !share_total) {
            *error = "helper shares: one weight for each of the " + std::to_string(shares.size()) +
                     " helpers, not all zero";
            return false;
        }
        head_shares = shares;
        if (!controls_valid(controls)) {
            *error = "live_policy: invalid mandatory controls";
            return false;
        }
        return true;
    }
};
} // namespace r3::policy
