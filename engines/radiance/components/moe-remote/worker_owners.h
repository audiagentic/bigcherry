#pragma once

// Owned-expert selection (heat, budget, balance) and publication. Included only by r3_moe_worker.cpp, at the position the code had in the single file (RCS03, pure move).

namespace {
using bc::experts::Owned;

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
    /* This helper's ownership settings, from its environment as before (R3_MOE_WORKER_HEAT, _PLACEMENT, _FORMATS,
     * _SHARES, _BALANCE, _BALANCE_SPEED); the selection itself is the shared one. */
    bc::experts::OwnerPolicy policy;
    if (const char* v = std::getenv("R3_MOE_WORKER_HEAT")) policy.heat = v;
    if (const char* v = std::getenv("R3_MOE_WORKER_PLACEMENT")) policy.placement = v;
    if (const char* v = std::getenv("R3_MOE_WORKER_FORMATS")) policy.formats = v;
    if (const char* v = std::getenv("R3_MOE_WORKER_SHARES")) policy.shares = v;   // parsed by the selection, where the worker parsed it
    const char* bv = std::getenv("R3_MOE_WORKER_BALANCE");
    policy.balance = bv && *bv && std::strcmp(bv, "0") != 0;
    policy.balance_speeds = {1.0, 1.0};
    if (const char* sv = std::getenv("R3_MOE_WORKER_BALANCE_SPEED"); sv && *sv) {
        double a = 0.0, b = 0.0;
        if (std::sscanf(sv, "%lf,%lf", &a, &b) == 2 && a > 0.0 && b > 0.0) policy.balance_speeds = {a, b};
    }
    return bc::experts::select_owned(f, policy, gpu ? gpu->arch() : std::string("host"), gpu != nullptr, layout_check,
                                     budgets, helper, helpers, out);
}

/* Bootstrap only: helpers publish their deterministic, disjoint selections into slot 0.
 * The initial map is mutable until engine bootstrap sealing; there is no live owner update. */
bool publish_owned(void* map, const Owned& owned, uint32_t helper) {
    Header* h = static_cast<Header*>(map);
    uint8_t* owners = generation_map(map, 0);
    if (!owners || helper >= kMaxHelpers) return false;
    for (auto [l, e] : owned.expert) {
        if (l >= h->layer_count || e >= h->expert_count) return false;
        uint8_t* entry = owners + (size_t)l * h->expert_count + e;
        uint8_t expected = kOwnerLocal;
        if (!__atomic_compare_exchange_n(entry, &expected, (uint8_t)helper, false,
                                         __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            std::fprintf(stderr, "r3_moe_worker: duplicate/conflicting ownership (%u,%u): existing=%u, helper=%u\n",
                         l, e, (unsigned)expected, helper);
            return false;
        }
    }
    return true;
}

/* Only initialization-time runtime-map filtering can remove a candidate owner. */
bool drop_bootstrap_owner(void* map, uint32_t l, uint32_t e, uint32_t helper) {
    Header* h = static_cast<Header*>(map);
    if (l >= h->layer_count || e >= h->expert_count || helper >= kMaxHelpers) return false;
    uint8_t* entry = generation_map(map, 0) + (size_t)l * h->expert_count + e;
    uint8_t expected = (uint8_t)helper;
    return __atomic_compare_exchange_n(entry, &expected, kOwnerLocal, false,
                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
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
