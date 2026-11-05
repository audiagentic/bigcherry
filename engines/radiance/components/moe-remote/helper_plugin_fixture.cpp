// A card-free DSO using the exact configuration implementation built into each kernel library.
#define BC_R4D_HELPER_CONFIG_IMPLEMENTATION
#include "bc_helper_config.h"
#undef BC_R4D_HELPER_CONFIG_IMPLEMENTATION
#include "r4d_plugin.h"
#include "../libr3/r3_moe_settings.h"
#include "r4d_args.h"
#include <cstring>

static unsigned env_reads;
// Bind locally (-Bsymbolic-functions) to detect any application environment read.
extern "C" char* getenv(const char* name) noexcept {
    ++env_reads;
    static char off[] = "0", two[] = "2", on[] = "1";
    if (std::strcmp(name, "R3_MOE_PROBE") == 0) return two;
    if (std::strcmp(name, "RAD_R4D_TRACE") == 0) return on;
    return off;
}
extern "C" unsigned test_env_reads() { return env_reads; }
extern "C" BcR4dHelperConfig test_moe_settings() { return r3_moe_settings(); }
extern "C" int test_refusal_trace() { return r4d_trace_on(); }
