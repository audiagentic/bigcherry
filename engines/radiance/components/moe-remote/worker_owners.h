#pragma once

// Owned-expert selection (heat, budget, balance) and publication. Included only by r3_moe_worker.cpp, at the position the code had in the single file (RCS03, pure move).

#include "worker_bootstrap.h"

namespace {
using bc::experts::Owned;
using r3::epoch::publish_owned;
using r3::epoch::drop_bootstrap_owner;

/* The owned-expert selection is bc::experts::select_owned (components/serving/experts.h), the code bc-serve plans with,
 * fed this helper's settings. On a card the backend contract's expert size must first equal the size the device layout
 * really takes, or no budget arithmetic can be trusted. */
bool select_owned(const Container& f, r3_moe_gpu::Runner* gpu, const uint64_t* budgets, uint32_t helper,
                  uint32_t helpers, Owned* out) {
    const std::function<bool()> layout_check = [&]() {
        if (!gpu) return true;
        const int64_t want = laid_out_bytes(f, gpu, 0, 0), got = gpu->expert_bytes(0, 0);
        if (want != got) {
            std::fprintf(stderr, "r3_moe_worker: expert size disagrees with the device layout (contract %lld, layout %lld)\n",
                         (long long)want, (long long)got);
            return false;
        }
        return true;
    };
    /* This helper's ownership settings, from its manifest section; the selection itself is the shared one. */
    const auto& s = r3_serving::settings();
    bc::experts::OwnerPolicy policy;
    policy.heat = s.handoff.heat;
    policy.placement = s.handoff.placement;
    policy.formats = s.formats;
    for (size_t i = 0; i < s.shares.size(); ++i) policy.shares += (i ? "," : "") + std::to_string(s.shares[i]);
    policy.balance = s.balance;
    policy.balance_speeds = s.balance_speeds;
    return bc::experts::select_owned(f, policy, gpu ? gpu->arch() : std::string("host"), gpu != nullptr,
                                     layout_check, budgets, helper, helpers, out);
}

bool parse_u32(const char* s, uint32_t lo, uint32_t hi, uint32_t* out) {
    if (!s || !*s || !out) return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long v = std::strtoul(s, &end, 10);
    if (errno || !end || *end || v < lo || v > hi) return false;
    *out = (uint32_t)v;
    return true;
}
} // namespace
