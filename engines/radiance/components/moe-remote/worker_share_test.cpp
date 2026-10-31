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
    std::puts("r3_worker_share_test: ok");
    return 0;
}
