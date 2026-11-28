// Test the actual composed bring-up block with only its architecture probe faked.
// Included by upstream config_test.cpp; the parser and Config are real.
namespace {
struct InterleaveResult {
    std::vector<rad::diag::Decision> decisions;
    int probes = 0;
};

InterleaveResult resolve_interleave_for_test(Config& cfg_, bool generator) {
    InterleaveResult result;
    int plugins_ = 0, meta_ = 0;
    auto rad_tools_probe = [&](int, int, RadArchProbe* pb, auto, auto) {
        ++result.probes;
        pb->gen_rows = generator ? 1 : 0;
        return 0;
    };
    auto decide = [&](const std::string& key, const std::string& value,
                      const std::string& why) {
        rad::diag::decide(&result.decisions, key, value, why);
    };
#include "bc_interleave_default.inc"
    return result;
}
}

TEST(bc_interleave_effective_default) {
    Config c;
    REQUIRE_EQ(parse({}, &c), RAD_OK);
    CHECK(!c.prefill_interleave_set);
    auto r = resolve_interleave_for_test(c, false);
    CHECK_EQ(r.probes, 1);
    REQUIRE_EQ(r.decisions.size(), size_t(1));
    CHECK_EQ(r.decisions[0].key, std::string("--prefill-interleave"));
    CHECK_EQ(r.decisions[0].value, std::string("auto:1"));
    CHECK_EQ(c.prefill_interleave, -1);
    CHECK_EQ(c.prefill_interleave_share, 1.0);
}

TEST(bc_interleave_effective_exceptions_and_precedence) {
    for (int exception : {0, 1, 2}) {
        Config c;
        REQUIRE_EQ(parse({}, &c), RAD_OK);
        if (exception == 0) c.deterministic = true;
        if (exception == 1) c.kld_ref = "fixture";
        const auto r = resolve_interleave_for_test(c, exception == 2);
        REQUIRE_EQ(r.decisions.size(), size_t(1));
        CHECK_EQ(r.decisions[0].value, std::string("0"));
        CHECK_EQ(c.prefill_interleave, 0);
    }
    for (const char* flag : {"0", "2", "auto:0.25"}) {
        Config c;
        REQUIRE_EQ(parse({"--prefill-interleave", flag}, &c), RAD_OK);
        const auto steps = c.prefill_interleave;
        const auto share = c.prefill_interleave_share;
        c.deterministic = true;
        auto r = resolve_interleave_for_test(c, true);
        CHECK(c.prefill_interleave_set);
        CHECK_EQ(r.probes, 0);
        CHECK(r.decisions.empty());
        CHECK_EQ(c.prefill_interleave, steps);
        CHECK_EQ(c.prefill_interleave_share, share);
    }
}
