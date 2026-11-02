// Card-free test of rad0011's start-up plan check (core/bc_plan_memory_check.cpp): a pure function of two plan
// documents. Built and run by tools/tests/patch/test_radiance_patch_packages.py. Prints "ok" on success.
#include "bc_plan_memory_check.h"

#include <cstdio>
#include <string>

namespace {
int failures = 0;

nlohmann::json base() {
    return nlohmann::json::parse(R"json({
      "plan_version": 1, "producer": "radiance --plan-memory (rad0011)",
      "host": [{"name": "ngram_table", "bytes": 1000, "residence": "pinned", "from": "engine plan (rad0011: weight_footprint)",
                "remedy": "set engine.ngram_placement = \"disk\""}],
      "checkpoints": {"slot_bytes": 100, "floor_slots": 8, "want_slots": 72, "from": "engine plan (rad0011: ckpt)"},
      "kv_one_session": {"bytes": 500, "from": "engine plan (rad0011: kv_one)"},
      "vram": {"claimable": 9000, "statics": 3000, "elastic": 6000, "experts": 0, "kv": 5000, "stage": 200,
               "expert_floor": 0, "kv_ceiling": 7000, "expert_ceiling": 100, "from": "engine plan (rad0011: vram)",
               "cards": [{"capacity": 10000, "free": 9900, "engine_expert_plane": 100, "expert_unit_max": 10}]},
      "host_pool_mib": 9216})json");
}

// The examples' calibration: kv_backing_floor.fraction_ppm = 600000, so kv_required = max(500, ceil(7000 * 0.6)) = 4200
// and the required split is 3000 + 0 + 200 + 4200 = 7400 (the measured split, with kv 5000, is 8200).
constexpr int64_t kFloorPpm = 600000;

void expect(bool want, const nlohmann::json& live, const std::string& needle, int64_t ppm = kFloorPpm,
            const nlohmann::json& expected = base()) {
    std::string why;
    const bool got = bc_plan_memory_check(expected, live, ppm, &why);
    if (got != want || (!want && why.find(needle) == std::string::npos)) {
        std::printf("FAIL: want %d got %d for '%s': %s\n", want, got, needle.c_str(), why.c_str());
        ++failures;
    }
}
}  // namespace

int main() {
    expect(true, base(), "");
    nlohmann::json live = base();
    live["host"][0]["bytes"] = 999;   // smaller is fine
    live["vram"]["claimable"] = 9500; // more room is fine
    expect(true, live, "");
    live = base(); live["host"][0]["bytes"] = 1001;
    expect(false, live, "host consumer ngram_table: live 1001 bytes exceed the planned 1000 bytes (engine plan (rad0011: weight_footprint)); set engine.ngram_placement = \"disk\"");
    live = base(); live["host"].push_back({{"name", "embedding"}, {"bytes", 1}, {"residence", "pinned"},
                                           {"from", "engine plan (rad0011: emb)"}, {"remedy", ""}});
    expect(false, live, "host consumer embedding: live 1 bytes exceed the planned 0 bytes");
    expect(false, live, "bc-serve --plan-memory");
    live = base(); live["checkpoints"]["slot_bytes"] = 101;
    expect(false, live, "checkpoint slot: live 101 bytes exceed the planned 100");
    live = base(); live["checkpoints"]["floor_slots"] = 9;
    expect(false, live, "checkpoint floor: live 900 bytes exceed the planned 800");
    live = base(); live["kv_one_session"]["bytes"] = 501;
    expect(false, live, "one full-length session's KV: live 501");
    live = base(); live["vram"]["statics"] = 3001;
    expect(false, live, "VRAM statics: live 3001");
    live = base(); live["vram"]["stage"] = 201;
    expect(false, live, "VRAM stage: live 201");
    live = base(); live["vram"]["claimable"] = 9000 - 100;    // a live reading that moved a little: the plan still fits
    expect(true, live, "");
    // Admission (spec §5.2, amendment 5): what the plan requires, not the measured share, must fit.
    live = base(); live["vram"]["claimable"] = 8199;           // below the old measured split (8200): admitted
    expect(true, live, "");
    live = base(); live["vram"]["claimable"] = 7400;           // exactly the required split: admitted
    expect(true, live, "");
    live = base(); live["vram"]["claimable"] = 7399;           // one byte short of the required split: refused
    expect(false, live, "claimable VRAM (for the required split: statics + experts + stage + kv_required, kv_required 4200 bytes): live 7399 bytes fall short of the required 7400 bytes");
    live = base(); live["vram"]["claimable"] = 7400; live["vram"]["kv"] = 0;   // the live kv share is not an input
    expect(true, live, "");
    live = base(); live["vram"]["claimable"] = 3700;           // no floor selected: kv_required is one session (500)
    expect(true, live, "", 0);
    live = base(); live["vram"]["claimable"] = 3699;
    expect(false, live, "kv_required 500 bytes): live 3699 bytes fall short of the required 3700 bytes", 0);
    {
        nlohmann::json big = base();                           // one session larger than the floor: it is the requirement
        big["kv_one_session"]["bytes"] = 4500;
        live = big; live["vram"]["claimable"] = 7699;
        expect(false, live, "kv_required 4500 bytes): live 7699 bytes fall short of the required 7700 bytes", kFloorPpm, big);
        nlohmann::json odd = base();                           // the floor rounds up: ceil(7001 * 0.6) = ceil(4200.6) = 4201
        odd["vram"]["kv_ceiling"] = 7001;
        live = odd; live["vram"]["claimable"] = 7400;
        expect(false, live, "kv_required 4201 bytes): live 7400 bytes fall short of the required 7401 bytes", kFloorPpm, odd);
        live["vram"]["claimable"] = 7401;
        expect(true, live, "", kFloorPpm, odd);
    }
    {   // the schema's largest byte count: ceil((2^53 - 1) * 999999 / 10^6) = 9007190247541737, with no int64 overflow
        nlohmann::json huge = base();
        huge["vram"]["kv_ceiling"] = 9007199254740991LL;
        huge["vram"]["claimable"] = 9007190247541737LL + 3200;
        expect(true, huge, "", 999999, huge);
        huge["vram"]["claimable"] = 9007190247541737LL + 3199;
        expect(false, huge, "kv_required 9007190247541737 bytes)", 999999, huge);
        huge["vram"]["kv_ceiling"] = 9007199254740992LL;            // one past the schema's maximum
        expect(false, huge, "byte counts in [0, 2^53 - 1]", 999999, huge);
    }
    expect(false, base(), "kv_backing_floor_ppm must be in [0, 1000000)", -1);
    expect(false, base(), "kv_backing_floor_ppm must be in [0, 1000000)", 1000000);
    live = base(); live["vram"]["cards"][0]["capacity"] = 9999;
    expect(false, live, "card 0 capacity: live 9999 bytes fall short of the planned 10000");
    live = base(); live["vram"]["cards"].push_back(live["vram"]["cards"][0]);
    expect(false, live, "the live plan has 2 card(s), the manifest's 1");
    live = base(); live["plan_version"] = 2;
    expect(false, live, "plan_version 1");
    if (failures) return 1;
    std::printf("ok\n");
    return 0;
}
