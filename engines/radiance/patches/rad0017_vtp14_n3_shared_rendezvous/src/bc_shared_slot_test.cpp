/* bc_shared_slot_test.cpp -- BC_VTP14_N3 (BigCherry managed patch rad0017): the process-wide named
 * slot two kernel libraries share for their all-reduce rendezvous. Card-free. */
#include "rad_test.h"

#include "plugin/bc_shared_slot.h"
#include "rad_abi.h"

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

using namespace rad;

namespace {
std::atomic<int> g_inits{0};
struct Rv { long a; int b; };
void init_rv(void* p) { ++g_inits; static_cast<Rv*>(p)->b = 7; }
}  /* namespace */

TEST(two_libraries_asking_for_one_key_share_one_slot_constructed_once) {
    g_inits = 0;
    void* a = nullptr;
    void* b = nullptr;
    std::string why;
    REQUIRE_EQ(bc_shared_slot_for("libr3.so", "test.rv/1", sizeof(Rv), alignof(Rv), init_rv, &a, &why), RAD_OK);
    REQUIRE_EQ(bc_shared_slot_for("libr4d.so", "test.rv/1", sizeof(Rv), alignof(Rv), init_rv, &b, &why), RAD_OK);
    CHECK(a != nullptr);
    CHECK(a == b);
    CHECK_EQ(g_inits.load(), 1);
    CHECK_EQ(static_cast<Rv*>(a)->b, 7);
    const std::vector<std::string> users = bc_shared_slot_users("test.rv/1");
    REQUIRE_EQ(users.size(), 2u);
    CHECK_EQ(users[0], std::string("libr3.so"));
    CHECK_EQ(users[1], std::string("libr4d.so"));
    const std::vector<std::string> keys = bc_shared_slot_keys("test.rv/");
    CHECK(std::find(keys.begin(), keys.end(), std::string("test.rv/1")) != keys.end());
    CHECK(bc_shared_slot_keys("nothing.here/").empty());
}

TEST(a_layout_disagreement_is_refused_not_given_a_second_slot) {
    void* a = nullptr;
    void* b = reinterpret_cast<void*>(0x1);
    std::string why;
    REQUIRE_EQ(bc_shared_slot_for("libr3.so", "test.rv/2", 64, 8, nullptr, &a, &why), RAD_OK);
    CHECK_EQ(bc_shared_slot_for("libr4d.so", "test.rv/2", 72, 8, nullptr, &b, &why), RAD_E_STATE);
    CHECK(b == nullptr);
    CHECK(why.find("disagree") != std::string::npos);
    CHECK(why.find("libr4d.so") != std::string::npos);
    CHECK_EQ(bc_shared_slot_for("libr4d.so", "test.rv/2", 64, 16, nullptr, &b, &why), RAD_E_STATE);
    CHECK_EQ(bc_shared_slot_users("test.rv/2").size(), 1u);
}

TEST(bad_requests_are_refused) {
    void* p = nullptr;
    std::string why;
    CHECK_EQ(bc_shared_slot_for("x", "", 8, 8, nullptr, &p, &why), RAD_E_INVAL);
    CHECK_EQ(bc_shared_slot_for("x", nullptr, 8, 8, nullptr, &p, &why), RAD_E_INVAL);
    CHECK_EQ(bc_shared_slot_for("x", "test.bad", 0, 8, nullptr, &p, &why), RAD_E_INVAL);
    CHECK_EQ(bc_shared_slot_for("x", "test.bad", 8, 3, nullptr, &p, &why), RAD_E_INVAL);
    CHECK_EQ(bc_shared_slot_for("x", "test.bad", 8, 8, nullptr, nullptr, &why), RAD_E_INVAL);
}

TEST(the_slot_is_zeroed_and_aligned) {
    void* p = nullptr;
    std::string why;
    REQUIRE_EQ(bc_shared_slot_for("x", "test.align", 256, 128, nullptr, &p, &why), RAD_OK);
    CHECK_EQ(reinterpret_cast<uintptr_t>(p) % 128u, 0u);
    const unsigned char* c = static_cast<const unsigned char*>(p);
    bool zero = true;
    for (int i = 0; i < 256; ++i) zero = zero && c[i] == 0;
    CHECK(zero);
}

TEST(concurrent_ranks_get_the_same_slot) {
    g_inits = 0;
    std::vector<void*> got(8, nullptr);
    std::vector<std::thread> ts;
    for (int r = 0; r < 8; ++r)
        ts.emplace_back([&, r] {
            std::string why;
            (void)bc_shared_slot_for(r % 2 ? "libr4d.so" : "libr3.so", "test.rv/3", sizeof(Rv), alignof(Rv),
                                     init_rv, &got[(size_t)r], &why);
        });
    for (auto& t : ts) t.join();
    for (void* p : got) CHECK(p == got[0] && p != nullptr);
    CHECK_EQ(g_inits.load(), 1);
}

TEST(the_c_entry_point_reports_the_reason) {
    void* p = nullptr;
    char why[256] = { 0 };
    REQUIRE_EQ(rad_bc_shared_slot("test.c/1", 32, 8, nullptr, &p, why, sizeof why), RAD_OK);
    CHECK_EQ(rad_bc_shared_slot("test.c/1", 48, 8, nullptr, &p, why, sizeof why), RAD_E_STATE);
    CHECK(std::strstr(why, "disagree") != nullptr);
}

RAD_TEST_MAIN()
