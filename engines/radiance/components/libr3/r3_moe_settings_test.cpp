// The engine process's MoE form switches (r3_moe_settings.h): with no helper configuration installed, R3_MOE_NATIVE,
// R3_MOE_FP16 and R3_MOE_AHALF each read unset as on (their declared default, component.toml), "0" as off and any other
// number as on, independently of one another; an installed helper configuration (the r3_moe_worker processes) wins over
// the environment. The configuration store is compiled into this test as it is into libr3.
#define BC_R4D_HELPER_CONFIG_IMPLEMENTATION
#include "r3_moe_settings.h"

#include <cstdio>
#include <cstdlib>

static void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "r3_moe_settings_test: FAILED %s\n", why); std::exit(1); }
}

static void set(const char* name, const char* value) {
    if (value) ::setenv(name, value, 1);
    else ::unsetenv(name);
}

int main() {
    check(bc_r4d_helper_config() == nullptr, "no helper configuration installed yet");
    const char* names[3] = {"R3_MOE_NATIVE", "R3_MOE_FP16", "R3_MOE_AHALF"};
    const char* values[4] = {nullptr, "0", "1", "2"};
    for (const char* a : values)
        for (const char* b : values)
            for (const char* c : values) {
                set(names[0], a); set(names[1], b); set(names[2], c);
                const BcR4dHelperConfig s = r3_moe_settings();
                auto want = [](const char* v) { return uint32_t(!v || std::atoi(v) != 0); };
                check(s.struct_size == sizeof(BcR4dHelperConfig), "struct size");
                check(s.native == want(a) && s.fp16 == want(b) && s.ahalf == want(c), "unset on, 0 off, nonzero on, each alone");
                check(s.moe_probe == 0 && s.refusal_trace == 0, "the probe and trace are never the environment's");
            }
    set(names[0], nullptr); set(names[1], nullptr); set(names[2], nullptr);
    const BcR4dHelperConfig d = r3_moe_settings();
    check(d.native == 1 && d.fp16 == 1 && d.ahalf == 1, "all unset: the declared defaults, on");
    /* a helper process installs its manifest's values; the environment no longer matters */
    BcR4dHelperConfig helper{};
    helper.struct_size = sizeof(BcR4dHelperConfig);
    helper.native = 0; helper.fp16 = 1; helper.ahalf = 0;
    check(bc_r4d_configure_helper(&helper) == 0, "install the helper configuration");
    set(names[0], "1"); set(names[2], "1");
    const BcR4dHelperConfig h = r3_moe_settings();
    check(h.native == 0 && h.fp16 == 1 && h.ahalf == 0, "the helper configuration wins over the environment");
    std::puts("r3_moe_settings_test: ok");
    return 0;
}
