#pragma once
// Cases shared by the golden generator (compiled against the PRE-MOVE worker headers, commit 6a3d11433) and by
// bc_experts_test (compiled against bc::experts): the same container, heat file and environment settings in, the owned
// sets out. See golden/README.md for the command that regenerates golden/owned_golden.txt.
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace golden {

constexpr uint64_t kExpertBytes = 25344;   // a W4NL64 expert of the fixture, hidden 128 width 128, canonical layout

struct Case {
    const char* name;
    std::vector<std::pair<const char*, std::string>> env;   // R3_MOE_WORKER_<key>; HEAT's value "@heat" means the fixture's heat file
    uint32_t helpers;
    std::vector<uint64_t> budget_experts;                  // each helper's budget, in whole experts
};

inline const std::vector<Case>& cases() {
    static const std::vector<Case> c = {
        {"noheat-rr-1", {}, 1, {20}},
        {"noheat-rr-2", {}, 2, {10, 10}},
        {"noheat-rr-3-shares-211", {{"SHARES", "2,1,1"}}, 3, {12, 12, 12}},
        {"heat-rr-2", {{"HEAT", "@heat"}}, 2, {10, 10}},
        {"heat-rr-2-shares-31", {{"HEAT", "@heat"}, {"SHARES", "3,1"}}, 2, {20, 20}},
        {"heat-rr-2-share-zero", {{"HEAT", "@heat"}, {"SHARES", "1,0"}}, 2, {20, 20}},
        {"heat-rr-2-shares-allzero", {{"HEAT", "@heat"}, {"SHARES", "0,0"}}, 2, {20, 20}},
        {"heat-rr-2-shares-bad", {{"HEAT", "@heat"}, {"SHARES", "1,x"}}, 2, {20, 20}},
        {"heat-tail-3", {{"HEAT", "@heat"}, {"PLACEMENT", "tail"}}, 3, {20, 20, 10}},
        {"heat-tail-3-shares-211", {{"HEAT", "@heat"}, {"PLACEMENT", "tail"}, {"SHARES", "2,1,1"}}, 3, {20, 20, 10}},
        {"heat-tail-1", {{"HEAT", "@heat"}, {"PLACEMENT", "tail"}}, 1, {20}},
        {"heat-bogus-placement", {{"HEAT", "@heat"}, {"PLACEMENT", "bogus"}}, 2, {10, 10}},
        {"heat-balance-2", {{"HEAT", "@heat"}, {"BALANCE", "1"}}, 2, {16, 16}},
        {"heat-balance-2-speed-21", {{"HEAT", "@heat"}, {"BALANCE", "1"}, {"BALANCE_SPEED", "2,1"}}, 2, {16, 16}},
        {"heat-balance-2-speed-bad", {{"HEAT", "@heat"}, {"BALANCE", "1"}, {"BALANCE_SPEED", "0,1"}}, 2, {16, 16}},
        {"heat-balance-2-shares-allzero", {{"HEAT", "@heat"}, {"BALANCE", "1"}, {"SHARES", "0,0"}}, 2, {16, 16}},
        {"heat-balance-2-shares-bad", {{"HEAT", "@heat"}, {"BALANCE", "1"}, {"SHARES", "junk"}}, 2, {16, 16}},
        {"heat-balance-tail", {{"HEAT", "@heat"}, {"BALANCE", "1"}, {"PLACEMENT", "tail"}}, 2, {16, 16}},
        {"heat-balance-unequal", {{"HEAT", "@heat"}, {"BALANCE", "1"}}, 2, {16, 12}},
        {"heat-balance-3", {{"HEAT", "@heat"}, {"BALANCE", "1"}}, 3, {16, 16, 16}},
        {"noheat-balance", {{"BALANCE", "1"}}, 2, {16, 16}},
        {"heat-missing-file", {{"HEAT", "/nonexistent/heat"}}, 2, {10, 10}},
        {"heat-fp8-only-formats", {{"HEAT", "@heat"}, {"FORMATS", "fp8_e4m3_block"}}, 2, {10, 10}},
        {"heat-no-budget", {{"HEAT", "@heat"}}, 2, {0, 10}},
        {"heat-tiny-budget", {{"HEAT", "@heat"}}, 2, {1, 1}},
    };
    return c;
}

}  // namespace golden
