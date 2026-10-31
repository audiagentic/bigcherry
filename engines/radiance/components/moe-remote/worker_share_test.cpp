#include "worker_share.h"
#include <cstdio>
#include <cstdlib>

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "r3_worker_share_test: FAILED %s\n", what); std::exit(1); }
}

using namespace r3::worker_share;
int main() {
    uint32_t w[kMaxHelperShares] = {};
    // default: every helper weight 1, and owner_of is the round robin the helpers always used
    for (uint32_t helpers = 1; helpers <= 4; ++helpers) {
        check(parse_shares(nullptr, helpers, w) && parse_shares("", helpers, w), "no list means equal shares");
        bool same = true;
        for (uint64_t pos = 0; pos < 1000; ++pos) same = same && owner_of(pos, w, helpers) == pos % helpers;
        check(same, "equal shares are position % helpers");
    }
    // weighted
    check(parse_shares("2,1", 2, w) && w[0] == 2 && w[1] == 1, "2,1 parses");
    check(owner_of(0, w, 2) == 0 && owner_of(1, w, 2) == 0 && owner_of(2, w, 2) == 1 && owner_of(3, w, 2) == 0, "2,1 repeats as blocks");
    check(parse_shares("4,4,1", 3, w), "4,4,1 parses");
    uint32_t count[3] = {};
    for (uint64_t pos = 0; pos < 9000; ++pos) ++count[owner_of(pos, w, 3)];
    check(count[0] == 4000 && count[1] == 4000 && count[2] == 1000, "shares are proportions of the ranking");
    // zero weight owns nothing
    check(parse_shares("1,1,0", 3, w), "a zero weight is allowed while another is not");
    count[0] = count[1] = count[2] = 0;
    for (uint64_t pos = 0; pos < 1000; ++pos) ++count[owner_of(pos, w, 3)];
    check(count[2] == 0 && count[0] == 500 && count[1] == 500, "a zero weight owns nothing");
    // refusals
    check(!parse_shares("0,0", 2, w), "all zero is refused");
    check(!parse_shares("1,2,3", 2, w) && !parse_shares("1", 2, w), "a count that is not the helper count is refused");
    check(!parse_shares("1,x", 2, w) && !parse_shares("1,,2", 3, w) && !parse_shares("-1,2", 2, w) && !parse_shares("1,2,", 3, w), "malformed lists are refused");
    check(!parse_shares("1,2", 0, w) && !parse_shares("1,2", kMaxHelperShares + 1, w) && !parse_shares("1,2", 2, nullptr), "bad arguments are refused");
    // budgets
    uint64_t b[kMaxHelperShares] = {};
    check(parse_budgets(nullptr, 3, 100, b) && b[0] == 100 && b[1] == 100 && b[2] == 100, "no list: every helper gets the fallback");
    check(parse_budgets("22528,22528,14336", 3, 7, b) && b[0] == (22528ull << 20) && b[2] == (14336ull << 20), "budgets are MiB, one a helper");
    check(!parse_budgets("1,2", 3, 7, b) && !parse_budgets("1,2,3,4", 3, 7, b) && !parse_budgets("1,a,3", 3, 7, b) && !parse_budgets("1,2,", 3, 7, b), "a wrong count or a malformed budget is refused");
    check(!parse_budgets("99999999999999,1", 2, 7, b) && !parse_budgets("1", 0, 7, b) && !parse_budgets("1", 1, 7, nullptr), "an absurd budget and bad arguments are refused");
    // tail placement: heads by shares to their budgets, the last helper takes the rest
    {
        const std::vector<int64_t> bytes(12, 1);
        uint32_t hs[2] = {1, 1};
        const uint64_t bud[3] = {3, 3, 5};
        const auto o = plan_tail(bytes, 3, hs, bud);
        const int32_t want[12] = {0, 1, 0, 1, 0, 1, -1, 2, 2, 2, 2, 2};
        bool same = true;
        for (int i = 0; i < 12; ++i) same = same && o[i] == want[i];
        check(same, "heads fill alternately, then the tail helper takes the coldest of the rest that fit its budget");
    }
    {
        // a head that fills early hands its later candidates to the tail; budgets are never exceeded; owners are disjoint by construction
        std::vector<int64_t> bytes;
        for (int i = 0; i < 400; ++i) bytes.push_back(1 + (i % 5));
        uint32_t hs[2] = {3, 1};
        const uint64_t bud[3] = {60, 40, 90};
        const auto o = plan_tail(bytes, 3, hs, bud);
        uint64_t held[3] = {};
        int owned[3] = {};
        for (size_t i = 0; i < o.size(); ++i) if (o[i] >= 0) { held[o[i]] += (uint64_t)bytes[i]; ++owned[o[i]]; }
        check(held[0] <= 60 && held[1] <= 40 && held[2] <= 90, "no helper goes over its budget");
        check(held[0] > 50 && held[1] > 30 && held[2] > 80, "and each fills close to it");
        check(o[0] == 0 && o[1] == 0 && o[2] == 0 && o[3] == 1, "the 3,1 shares show at the head of the ranking");
        int tail_first = -1, tail_last = -1;
        for (size_t i = 0; i < o.size(); ++i) if (o[i] == 2) { if (tail_first < 0) tail_first = (int)i; tail_last = (int)i; }
        check(tail_first >= 0 && tail_last == 399, "the tail helper owns something, and the very end of the ranking is its");
        // a smaller tail budget holds the coldest of the leftovers, not the hottest
        const uint64_t small[3] = {60, 40, 20};
        const auto o2 = plan_tail(bytes, 3, hs, small);
        int first2 = -1;
        for (size_t i = 0; i < o2.size(); ++i) if (o2[i] == 2) { first2 = (int)i; break; }
        check(first2 > tail_first && o2[399] == 2, "with less room the tail helper starts later in the ranking");
    }
    {
        const std::vector<int64_t> bytes(6, 2);
        uint32_t hs[1] = {1};
        const uint64_t bud[2] = {4, 0};
        const auto o = plan_tail(bytes, 2, hs, bud);
        check(o[0] == 0 && o[1] == 0 && o[2] == -1 && o[5] == -1, "a tail helper with budget 0 owns nothing");
        check(plan_tail(bytes, 1, hs, bud)[0] == -1, "tail placement needs two helpers");
        const uint32_t zero[1] = {0};
        const auto z = plan_tail(bytes, 2, zero, bud);
        check(z[0] == -1, "heads with no share and a tail with no room own nothing");
    }
    std::puts("r3_worker_share_test: ok");
    return 0;
}
