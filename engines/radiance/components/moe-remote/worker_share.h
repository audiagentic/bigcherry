#pragma once

// Which helper owns the n-th candidate expert (RAD23). With equal shares this is the round robin the helpers always used (position % helpers);
// R3_MOE_WORKER_SHARES="4,4,1" weights the helpers instead, so a slower card owns a smaller part of the same ranking, and a weight of 0
// gives a helper nothing (the third helper present but idle, to measure the protocol on its own). Every helper evaluates this on its own
// from the same environment, so the result is a pure function of the position and the weights. No GPU, no allocation.

#include <cstdint>
#include <cstdlib>
#include <cstring>

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

}  // namespace r3::worker_share
