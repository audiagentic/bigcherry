/* bc_rank_devices_test.cpp -- BC_VTP14_N1 (BigCherry managed patch rad0016): per-rank kernel code
 * eligibility and the explicit --rank-devices assignment, card-free.
 *
 * A fake inventory of two gfx1100 cards and one gfx1201 card stands for the XTX/XTX/R9700 world.
 * The registry's code-runs probe stands for the fat binaries a real dlopen registers, so the
 * per-rank selection is tested on the real test plugins (core/plugin/testplugin) without a card. */
#include "rad_test.h"

#include "plugin/bc_rank_devices.h"
#include "plugin/registry.h"

#include <string>
#include <vector>

using namespace rad;

namespace {

std::string kernels_dir() { return home() + "/testkernels"; }
std::string picked(const KernelRow* k) { return k ? k->info->name : std::string("<none>"); }

/* Two XTX and one R9700, ordinals in KFD order, deliberately not in rank order below. */
std::vector<BcDeviceInfo> mixed_inventory() {
    return {
        { 0, "0000:0c:00.0", "gfx1201", "AMD Radeon AI PRO R9700", false },
        { 1, "0000:03:00.0", "gfx1100", "Radeon RX 7900 XTX", false },
        { 2, "0000:07:00.0", "gfx1100", "Radeon RX 7900 XTX", false },
        { 3, "0000:7a:00.0", "gfx1036", "AMD Radeon Graphics", true },
    };
}

Geometry norm_geom() {
    Geometry g;
    g.set_i("M", 1);
    g.set_i("n", 4096);
    g.set_s("dtype", "bf16");
    return g;
}

/* radtest_over's code runs on the XTX cards (ordinals 1, 2) and not on the R9700 (ordinal 0);
 * radtest_base's runs on every card. */
int xtx_only_over(const std::string& plugin, int device, std::string* why) {
    if (plugin == "radtest_over" && device == 0) {
        if (why) *why = "no code object for gfx1201";
        return 0;
    }
    return 1;
}

int over_nowhere(const std::string& plugin, int, std::string* why) {
    if (plugin == "radtest_over") {
        if (why) *why = "no code object for this card";
        return 0;
    }
    return 1;
}

}  /* namespace */

/* ------------------------------------------------------------------ addresses */
TEST(a_card_address_is_the_full_lower_case_bdf) {
    std::string c;
    CHECK(bc_parse_bdf("0000:03:00.0", &c));
    CHECK_EQ(c, "0000:03:00.0");
    CHECK(bc_parse_bdf("0000:0C:00.0", &c));
    CHECK_EQ(c, "0000:0c:00.0");
    CHECK(!bc_parse_bdf("03:00.0", &c));          /* the domain is part of the identity */
    CHECK(!bc_parse_bdf("0000:03:00.8", &c));     /* a function is 0-7 */
    CHECK(!bc_parse_bdf("0000:03:20.0", &c));     /* a device is five bits */
    CHECK(!bc_parse_bdf("0000:03:00.0 ", &c));
    CHECK(!bc_parse_bdf("", &c));

    std::vector<std::string> list;
    std::string why;
    CHECK(bc_parse_bdf_list("0000:03:00.0,0000:07:00.0,0000:0C:00.0", &list, &why));
    CHECK_EQ(list.size(), (size_t)3);
    CHECK_EQ(list[2], "0000:0c:00.0");
    CHECK(!bc_parse_bdf_list("0000:03:00.0,,0000:07:00.0", &list, &why));
    CHECK(has(why, "empty"));
    CHECK(!bc_parse_bdf_list("0000:03:00.0,gpu1", &list, &why));
    CHECK(has(why, "gpu1"));
}

/* ------------------------------------------------------------------ the assignment */
TEST(ranks_take_the_named_cards_in_the_named_order) {
    std::vector<int> ord;
    std::string why;
    CHECK_OK(bc_rank_devices_plan(mixed_inventory(), { "0000:03:00.0", "0000:07:00.0", "0000:0c:00.0" }, 3, &ord, &why));
    REQUIRE_EQ(ord.size(), (size_t)3);
    CHECK_EQ(ord[0], 1);
    CHECK_EQ(ord[1], 2);
    CHECK_EQ(ord[2], 0);
    const std::vector<std::string> archs = bc_rank_archs(mixed_inventory(), ord);
    REQUIRE_EQ(archs.size(), (size_t)2);
    CHECK_EQ(archs[0], "gfx1100");
    CHECK_EQ(archs[1], "gfx1201");
}

TEST(a_wrong_assignment_is_refused_before_anything_loads) {
    std::vector<int> ord;
    std::string why;
    CHECK_EQ(bc_rank_devices_plan(mixed_inventory(), { "0000:03:00.0", "0000:07:00.0" }, 3, &ord, &why), RAD_E_DEVICE);
    CHECK(has(why, "--tp 3"));
    CHECK(ord.empty());

    CHECK_EQ(bc_rank_devices_plan(mixed_inventory(), { "0000:03:00.0", "0000:03:00.0", "0000:0c:00.0" }, 3, &ord, &why),
             RAD_E_DEVICE);
    CHECK(has(why, "both name card 0000:03:00.0"));

    CHECK_EQ(bc_rank_devices_plan(mixed_inventory(), { "0000:03:00.0", "0000:05:00.0", "0000:0c:00.0" }, 3, &ord, &why),
             RAD_E_DEVICE);
    CHECK(has(why, "rank 1: no visible card at 0000:05:00.0"));

    CHECK_EQ(bc_rank_devices_plan(mixed_inventory(), { "0000:03:00.0", "0000:7a:00.0" }, 2, &ord, &why), RAD_E_DEVICE);
    CHECK(has(why, "integrated"));

    std::vector<BcDeviceInfo> host = { { 0, "", "cpu", "host", false } };
    CHECK_EQ(bc_rank_devices_plan(host, { "0000:03:00.0" }, 1, &ord, &why), RAD_E_DEVICE);
    CHECK(has(why, "no PCI addresses"));
}

/* ------------------------------------------------------------------ collectives */
TEST(every_rank_must_issue_the_same_collectives) {
    const BcCollStep a3 = { "all_reduce", "ar_twoshot_3rank_exact_host", "libr3" };
    const BcCollStep a3_r4d = { "all_reduce", "ar_twoshot_3rank_exact_host", "libr4d" };
    const BcCollStep a1 = { "all_reduce", "ar_oneshot_3rank_exact_host", "libr4d" };
    std::string why;
    /* The library may differ between architectures; the kernel is the protocol and may not. */
    CHECK_OK(bc_rank_collectives_agree({ { a3, a3 }, { a3, a3 }, { a3_r4d, a3_r4d } }, &why));
    CHECK_EQ(bc_rank_collectives_agree({ { a3, a3 }, { a3, a3 }, { a3_r4d, a1 } }, &why), RAD_E_STATE);
    CHECK(has(why, "collective 1"));
    CHECK(has(why, "rank 2"));
    CHECK_EQ(bc_rank_collectives_agree({ { a3 }, { a3, a3 } }, &why), RAD_E_STATE);
    CHECK(has(why, "waits forever"));
    CHECK_OK(bc_rank_collectives_agree({ { a3 } }, &why));

    /* The launch geometry (with the tuned axes) must match too; an untuned rank's empty geometry is not compared. */
    const BcCollStep g16 = { "all_reduce", "ar_twoshot_3rank_exact_host", "libr3", "M=64 nblocks=16 nthreads=1024" };
    const BcCollStep g16b = { "all_reduce", "ar_twoshot_3rank_exact_host", "libr4d", "M=64 nblocks=16 nthreads=1024" };
    const BcCollStep g8 = { "all_reduce", "ar_twoshot_3rank_exact_host", "libr4d", "M=64 nblocks=8 nthreads=1024" };
    CHECK_OK(bc_rank_collectives_agree({ { g16 }, { g16 }, { g16b } }, &why));
    CHECK_EQ(bc_rank_collectives_agree({ { g16 }, { g16 }, { g8 } }, &why), RAD_E_STATE);
    CHECK(has(why, "nblocks=8"));
    CHECK(has(why, "rank 2"));
    CHECK(has(why, "spins forever"));
    CHECK_OK(bc_rank_collectives_agree({ { g16 }, { a3 } }, &why));
}

/* ------------------------------------------------------------------ the registry */
TEST(the_rank_list_is_checked_before_any_library_loads) {
    Registry r;
    CHECK_EQ(r.set_code_devices({}), RAD_E_INVAL);
    CHECK_EQ(r.set_code_devices({ 1, -1 }), RAD_E_INVAL);
    CHECK_EQ(r.set_code_devices({ 1, 2, 1 }), RAD_E_INVAL);
    std::vector<int> too_many(33);
    for (size_t i = 0; i < too_many.size(); ++i) too_many[i] = (int)i;
    CHECK_EQ(r.set_code_devices(too_many), RAD_E_INVAL);
    CHECK_OK(r.set_code_devices({ 1, 2, 0 }));
    CHECK_EQ(r.code_devices().size(), (size_t)3);

    Registry loaded;
    CHECK_OK(loaded.load_dir(kernels_dir(), { "radtest_over", "radtest_base" }));
    CHECK_EQ(loaded.set_code_devices({ 1, 2, 0 }), RAD_E_INVAL);   /* rows exist: too late */

    /* The one-device setter returns to the unchanged path. */
    Registry back;
    CHECK_OK(back.set_code_devices({ 1, 2, 0 }));
    back.set_code_device(1);
    CHECK(back.code_devices().empty());
}

TEST(each_rank_selects_a_row_its_own_card_runs) {
    Registry r;
    CHECK_OK(r.set_code_devices({ 1, 2, 0 }));   /* XTX, XTX, R9700 */
    r.set_code_runs_probe(xtx_only_over);
    CHECK_OK(r.load_dir(kernels_dir(), { "radtest_over", "radtest_base" }));

    const Geometry g = norm_geom();
    /* The XTX ranks take the library above, which their cards run. */
    CHECK_EQ(picked(r.select("t_norm", g, RAD_DOMAIN_DEVICE, nullptr, true, nullptr, 0)), "t_over_norm");
    CHECK_EQ(picked(r.select("t_norm", g, RAD_DOMAIN_DEVICE, nullptr, true, nullptr, 1)), "t_over_norm");
    /* The R9700 rank falls through to the next library, with the refusal named. */
    SelectionTrace t;
    CHECK_EQ(picked(r.select("t_norm", g, RAD_DOMAIN_DEVICE, &t, true, nullptr, 2)), "t_norm_specialised");
    bool named = false;
    for (const SelectionTrace::Candidate& c : t.candidates)
        if (c.kernel == "t_over_norm") named = !c.ok && has(c.why, "rank 2");
    CHECK(named);
    /* A question asked for no rank in particular gets only rows every rank's card runs. */
    CHECK_EQ(picked(r.select("t_norm", g, RAD_DOMAIN_DEVICE)), "t_norm_specialised");
    /* A rank outside the list matches no device row. */
    CHECK_EQ(picked(r.select("t_norm", g, RAD_DOMAIN_DEVICE, nullptr, true, nullptr, 3)), "<none>");
    /* Host rows are not a card's code. */
    CHECK_EQ(picked(r.select("t_norm", g, RAD_DOMAIN_HOST, nullptr, true, nullptr, 2)), "t_norm_host");

    /* resolve and the bucket tables carry the rank through. */
    CHECK_EQ(picked(r.resolve("t_norm", g, RAD_DOMAIN_DEVICE, nullptr, nullptr, 0).row), "t_over_norm");
    CHECK_EQ(picked(r.resolve("t_norm", g, RAD_DOMAIN_DEVICE, nullptr, nullptr, 2).row), "t_norm_specialised");
    const std::vector<Band> b0 = r.build_bands("t_norm", "", 0, 0, g, nullptr, 0);
    const std::vector<Band> b2 = r.build_bands("t_norm", "", 0, 0, g, nullptr, 2);
    REQUIRE_EQ(b0.size(), (size_t)1);
    REQUIRE_EQ(b2.size(), (size_t)1);
    CHECK_EQ(picked(b0[0].dom[RAD_DOMAIN_DEVICE].row), "t_over_norm");
    CHECK_EQ(picked(b2[0].dom[RAD_DOMAIN_DEVICE].row), "t_norm_specialised");

    /* The mask is fixed at load: the over row runs on ranks 0 and 1 only. */
    for (const KernelRow& row : r.rows())
        if (row.plugin == "radtest_over" && row.info->domain == RAD_DOMAIN_DEVICE) CHECK_EQ(row.code_ranks, 3u);
}

TEST(a_library_no_rank_can_run_loses_its_device_rows) {
    Registry r;
    CHECK_OK(r.set_code_devices({ 1, 2, 0 }));
    r.set_code_runs_probe(over_nowhere);
    CHECK_OK(r.load_dir(kernels_dir(), { "radtest_over", "radtest_base" }));
    const Plugin* over = r.plugin("radtest_over");
    REQUIRE(over != nullptr);
    CHECK(over->device_code_absent);
    for (const KernelRow& row : r.rows()) CHECK(row.plugin != "radtest_over");
    for (int rank = 0; rank < 3; ++rank)
        CHECK_EQ(picked(r.select("t_norm", norm_geom(), RAD_DOMAIN_DEVICE, nullptr, true, nullptr, rank)),
                 "t_norm_specialised");
}

TEST(the_one_device_path_is_unchanged) {
    /* No rank list: the rank argument is ignored and every row is a candidate, as upstream. */
    Registry r;
    r.set_code_device(0);
    CHECK_OK(r.load_dir(kernels_dir(), { "radtest_over", "radtest_base" }));
    for (int rank : { -1, 0, 2, 7 })
        CHECK_EQ(picked(r.select("t_norm", norm_geom(), RAD_DOMAIN_DEVICE, nullptr, true, nullptr, rank)),
                 "t_over_norm");
    for (const KernelRow& row : r.rows()) CHECK_EQ(row.code_ranks, ~0u);
}

RAD_TEST_MAIN()
