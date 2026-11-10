// Engine MoE settings cannot be influenced by inherited R3_* variables.
#define BC_R4D_HELPER_CONFIG_IMPLEMENTATION
#include "r3_moe_settings.h"
#include <cstdio>
#include <cstdlib>

static void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "r3_moe_settings_test: FAILED %s\\n", why); std::exit(1); }
}

int main() {
    check(bc_r4d_helper_config() == nullptr, "no helper configuration installed yet");
    const char* names[3] = {"R3_MOE_NATIVE", "R3_MOE_FP16", "R3_MOE_AHALF"};
    for (unsigned mask = 0; mask < 8; ++mask) {
        for (unsigned i = 0; i < 3; ++i) ::setenv(names[i], (mask & (1u << i)) ? "0" : "1", 1);
        const BcR4dHelperConfig s = r3_moe_settings();
        check(s.struct_size == sizeof(BcR4dHelperConfig), "struct size");
        check(s.native == 1 && s.fp16 == 1 && s.ahalf == 1, "inherited variables inert");
        check(s.moe_probe == 0 && s.refusal_trace == 0, "diagnostic defaults");
    }
    BcR4dHelperConfig helper{};
    helper.struct_size = sizeof(BcR4dHelperConfig);
    helper.native = 0; helper.fp16 = 1; helper.ahalf = 0;
    check(bc_r4d_configure_helper(&helper) == 0, "install helper configuration");
    const BcR4dHelperConfig h = r3_moe_settings();
    check(h.native == 0 && h.fp16 == 1 && h.ahalf == 0, "helper configuration wins");
    for (const char* name : names) ::unsetenv(name);
    std::puts("r3_moe_settings_test: ok");
    return 0;
}
