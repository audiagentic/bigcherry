#pragma once

// Which helper owns the n-th candidate expert (RAD23). With equal shares this is the round robin the helpers always used (position % helpers);
// R3_MOE_WORKER_SHARES="4,4,1" weights the helpers instead, so a slower card owns a smaller part of the same ranking, and a weight of 0
// gives a helper nothing (the third helper present but idle, to measure the protocol on its own). Every helper evaluates this on its own
// from the same environment, so the result is a pure function of the position and the weights. No GPU, no allocation.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace r3::worker_share {

constexpr uint32_t kMaxHelperShares = 8;

// Parse a comma-separated list of non-negative integers, one a helper. null or "" means every helper has weight 1. False when the count
// differs from `helpers`, an entry is not a number, or every weight is 0.
inline bool parse_shares(const char* text, uint32_t helpers, uint32_t* weights) {
    if (!weights || !helpers || helpers > kMaxHelperShares) return false;
    if (!text || !*text) {
        for (uint32_t i = 0; i < helpers; ++i) weights[i] = 1;
        return true;
    }
    uint64_t total = 0;
    uint32_t n = 0;
    for (const char* p = text;;) {
        char* end = nullptr;
        if (*p < '0' || *p > '9') return false;
        const unsigned long v = std::strtoul(p, &end, 10);
        if (end == p || v > 1000000ul || n >= helpers) return false;
        weights[n++] = (uint32_t)v;
        total += v;
        if (*end == '\0') break;
        if (*end != ',') return false;
        p = end + 1;
    }
    return n == helpers && total > 0;
}

// The helper that owns candidate number `position`: weights repeat as blocks, helper 0's first, so equal weights give position % helpers.
inline uint32_t owner_of(uint64_t position, const uint32_t* weights, uint32_t helpers) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < helpers; ++i) total += weights[i];
    uint64_t at = position % total;
    for (uint32_t i = 0; i < helpers; ++i) {
        if (at < weights[i]) return i;
        at -= weights[i];
    }
    return helpers - 1;
}

// R3_MOE_WORKER_MIBS="22528,22528,14336": the budget of each helper in MiB, one a helper (RAD23: the cards differ). null or "" gives every
// helper `fallback_bytes`. False on a wrong count, a non-number, or a budget that does not fit 64 bits when scaled to bytes.
inline bool parse_budgets(const char* text, uint32_t helpers, uint64_t fallback_bytes, uint64_t* budgets) {
    if (!budgets || !helpers || helpers > kMaxHelperShares) return false;
    if (!text || !*text) {
        for (uint32_t i = 0; i < helpers; ++i) budgets[i] = fallback_bytes;
        return true;
    }
    uint32_t n = 0;
    for (const char* p = text;;) {
        char* end = nullptr;
        if (*p < '0' || *p > '9') return false;
        const unsigned long long v = std::strtoull(p, &end, 10);
        if (end == p || v > (1ull << 40) || n >= helpers) return false;   // 2^40 MiB is far past any card
        budgets[n++] = (uint64_t)v << 20;
        if (*end == '\0') break;
        if (*end != ',') return false;
        p = end + 1;
    }
    return n == helpers;
}

// R3_MOE_WORKER_PLACEMENT=tail: the experts of a ranking (hottest first) go to the head helpers 0..H-2 by their shares, each up to its own
// budget, as the round robin does; the LAST helper takes from what is left - every candidate a head helper had no room for - the COLDEST
// that fit its budget, from the end of the ranking backwards. For one helper behind two XTXs that is the tail of the heat ranking, and a
// smaller card holds the experts that see the least traffic. `bytes[i]` is the size of candidate i (all > 0);
// the result says which helper owns each candidate, or -1 for none. A pure function of its arguments, so every helper process reaches the
// same plan on its own.
inline std::vector<int32_t> plan_tail(const std::vector<int64_t>& bytes, uint32_t helpers, const uint32_t* head_shares, const uint64_t* budgets) {
    std::vector<int32_t> owner(bytes.size(), -1);
    if (helpers < 2 || helpers > kMaxHelperShares) return owner;
    const uint32_t heads = helpers - 1;
    uint64_t held[kMaxHelperShares] = {};
    uint64_t position = 0;
    uint32_t total = 0;
    std::vector<size_t> left;   // candidates no head helper had room for, in ranking order
    for (uint32_t i = 0; i < heads; ++i) total += head_shares[i];
    for (size_t i = 0; i < bytes.size(); ++i) {
        const uint64_t b = (uint64_t)bytes[i];
        const int32_t who = total ? (int32_t)owner_of(position++, head_shares, heads) : -1;
        if (who >= 0 && b <= budgets[who] - held[who]) {
            held[who] += b;
            owner[i] = who;
        } else {
            left.push_back(i);
        }
    }
    for (size_t k = left.size(); k-- > 0;) {
        const uint64_t b = (uint64_t)bytes[left[k]];
        if (b <= budgets[heads] - held[heads]) {
            held[heads] += b;
            owner[left[k]] = (int32_t)heads;
        }
    }
    return owner;
}

}  // namespace r3::worker_share
